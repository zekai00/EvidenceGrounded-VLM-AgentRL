#!/usr/bin/env python3
"""Filter v1.0 layout candidates into a higher-quality v1.0.2 cache.

The v1.0 cache intentionally favors recall. This script applies deterministic
quality gates derived from the v1.0.1 VLM audit:

- require a usable caption-like text signal;
- prefer PDF embedded image blocks;
- keep OpenCV visual regions only when geometry/text-overlap/caption are clean;
- write per-candidate keep/reject reasons for auditability.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/root/datasets/evidence_grounded_vlm_agentrl/layout_candidates_v1_0_pdftext_opencv_20260607_2325")
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")

LANDSCAPE_OR_ART_PATTERN = re.compile(
    r"山水|溪山|寒林|江山|山图|山圖|山居|山楼|山樓|秋山|春山|夏山|雪景|雪溪|洞庭|"
    r"罗浮|羅浮|林泉|峰|岩|巖|崖|石|泉|松|皴|米点|米點|青绿|青綠|浅绛|淺絳|"
    r"手卷|长卷|長卷|卷|册|冊|页|頁|轴|軸|图|圖|painting|landscape|mountain|"
    r"river|handscroll|album|hanging\s+scroll",
    flags=re.IGNORECASE,
)

OBVIOUS_NON_LANDSCAPE_PATTERN = re.compile(
    r"坐像|肖像|圣贤图|聖賢圖|帝后|人物肖像|画虎|畫虎|虎圖|书法|書法|墨迹|墨跡|"
    r"calligraphy|portrait",
    flags=re.IGNORECASE,
)

FIGURE_NUMBER_ONLY_PATTERN = re.compile(
    r"^(?:〔?\[?【?(?:图|圖)|Fig\.?|Figure|FIGURE)\s*[一二三四五六七八九十百0-9IVXivx:：.\-—_ ]+】?\.?$",
    flags=re.IGNORECASE,
)

PAGE_HEADER_PATTERN = re.compile(
    r"^(?:\d+\s+)?(?:SOUTHERN\s+SUNG\s+PAINTING|YUAN\s+LITERATI\s+PAINTING|MAINLAND\s+CHINESE\s+PAINTING)\s*\d*[I1]?$",
    flags=re.IGNORECASE,
)

NON_ART_DIAGRAM_PATTERN = re.compile(
    r"统计图|類型統計|类型统计|布置方式|拱券种类|拱券種類|结构图|結構圖|示意图|示意圖|流程图|流程圖|"
    r"关系图|關係圖|柱状图|柱狀圖|折线图|折線圖|表格|自制",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache-dir", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--min-caption-score", type=float, default=2.0)
    parser.add_argument("--max-caption-chars", type=int, default=260)
    parser.add_argument("--opencv-min-area-ratio", type=float, default=0.018)
    parser.add_argument("--opencv-min-width-ratio", type=float, default=0.10)
    parser.add_argument("--opencv-min-height-ratio", type=float, default=0.07)
    parser.add_argument("--opencv-max-text-overlap", type=float, default=0.18)
    parser.add_argument("--opencv-min-aspect", type=float, default=0.22)
    parser.add_argument("--opencv-max-aspect", type=float, default=7.5)
    parser.add_argument("--max-figures-per-page", type=int, default=12)
    parser.add_argument("--keep-nonlandscape-pdf-image-blocks", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_cache_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(Path(args.output_root))
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    page_rows = read_jsonl(input_dir / "page_candidates.jsonl")
    filtered_pages: list[dict[str, Any]] = []
    filtered_figures: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    kept_counter: Counter[str] = Counter()
    rejected_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()

    for row in page_rows:
        kept_figs: list[dict[str, Any]] = []
        for fig in row.get("figure_candidates") or []:
            decision = decide_candidate(row, fig, args)
            decision_rows.append(decision)
            source = str(fig.get("source") or "unknown")
            if decision["keep"]:
                kept_counter[source] += 1
                kept_fig = copy.deepcopy(fig)
                kept_fig["v1_0_2_filter"] = {
                    "keep": True,
                    "quality_score": decision["quality_score"],
                    "caption_quality": decision["caption_quality"],
                    "reasons": decision["reasons"],
                }
                kept_fig["target_score"] = round(float(kept_fig.get("target_score") or 0.0) + decision["quality_score"], 6)
                kept_figs.append(kept_fig)
            else:
                rejected_counter[source] += 1
                for reason in decision["reasons"]:
                    reason_counter[reason] += 1
        if not kept_figs:
            continue
        kept_figs.sort(key=lambda item: float(item.get("target_score") or 0.0), reverse=True)
        kept_figs = kept_figs[: max(1, args.max_figures_per_page)]
        new_row = copy.deepcopy(row)
        new_row["figure_candidates"] = kept_figs
        filtered_pages.append(new_row)
        for cand in kept_figs:
            item = {key: value for key, value in new_row.items() if key != "figure_candidates"}
            item.update(cand)
            filtered_figures.append(item)

    write_jsonl(output_dir / "page_candidates.jsonl", filtered_pages)
    write_jsonl(output_dir / "figure_candidates.jsonl", filtered_figures)
    write_jsonl(output_dir / "filter_decisions.jsonl", decision_rows)
    summary = build_summary(args, input_dir, output_dir, page_rows, filtered_pages, filtered_figures, kept_counter, rejected_counter, reason_counter)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(build_manifest(args, input_dir, output_dir, summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "v1.0.2候选过滤报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def decide_candidate(row: dict[str, Any], fig: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reasons: list[str] = []
    source = str(fig.get("source") or "unknown")
    caption_text = str(fig.get("caption_text") or "")
    caption_score = safe_float(fig.get("caption_score"), -999.0)
    caption_quality = caption_quality_label(caption_text)
    bbox = fig.get("bbox") or [0, 0, 0, 0]
    page_width = max(1, int(row.get("page_width") or 1))
    page_height = max(1, int(row.get("page_height") or 1))
    width = max(0, int(bbox[2]) - int(bbox[0])) if len(bbox) == 4 else 0
    height = max(0, int(bbox[3]) - int(bbox[1])) if len(bbox) == 4 else 0
    area_ratio = safe_float(fig.get("area_ratio"), 0.0)
    text_overlap = safe_float(fig.get("text_overlap_ratio"), 0.0)
    aspect = width / max(1, height)
    normalized_caption = normalize_caption(caption_text)

    keep = True
    if not caption_text.strip():
        keep = False
        reasons.append("missing_caption_text")
    if caption_score < args.min_caption_score:
        keep = False
        reasons.append("low_caption_score")
    if caption_quality == "plain_body_text":
        keep = False
        reasons.append("caption_not_caption_like")
    if figure_number_only(caption_text):
        keep = False
        reasons.append("caption_number_only")
    if page_header_like(caption_text):
        keep = False
        reasons.append("caption_page_header")
    if incomplete_body_fragment(caption_text):
        keep = False
        reasons.append("caption_body_fragment")
    if len(normalized_caption) > args.max_caption_chars and not explicit_caption_like(caption_text):
        keep = False
        reasons.append("caption_too_long_body_like")
    if OBVIOUS_NON_LANDSCAPE_PATTERN.search(caption_text) and not args.keep_nonlandscape_pdf_image_blocks:
        keep = False
        reasons.append("obvious_non_landscape_caption")
    if NON_ART_DIAGRAM_PATTERN.search(caption_text):
        keep = False
        reasons.append("non_art_diagram_caption")
    if source == "opencv_visual_region":
        if area_ratio < args.opencv_min_area_ratio:
            keep = False
            reasons.append("opencv_area_too_small")
        if width / page_width < args.opencv_min_width_ratio:
            keep = False
            reasons.append("opencv_width_too_small")
        if height / page_height < args.opencv_min_height_ratio:
            keep = False
            reasons.append("opencv_height_too_small")
        if not (args.opencv_min_aspect <= aspect <= args.opencv_max_aspect):
            keep = False
            reasons.append("opencv_bad_aspect")
        if text_overlap > args.opencv_max_text_overlap:
            keep = False
            reasons.append("opencv_text_overlap_high")
    elif source != "pdf_image_block":
        keep = False
        reasons.append("unsupported_candidate_source")

    quality_score = candidate_quality_score(source, caption_text, caption_score, area_ratio, text_overlap, caption_quality)
    if keep:
        reasons.append("kept_by_v1_0_2_rules")
    return {
        "candidate_id": fig.get("candidate_id"),
        "page_id": row.get("page_id"),
        "source_file": row.get("source_file"),
        "page": row.get("page"),
        "source": source,
        "bbox": bbox,
        "caption_text": caption_text,
        "caption_score": caption_score,
        "caption_quality": caption_quality,
        "area_ratio": area_ratio,
        "text_overlap_ratio": text_overlap,
        "aspect": round(aspect, 6),
        "quality_score": quality_score,
        "keep": keep,
        "reasons": reasons,
    }


def caption_quality_label(text: str) -> str:
    if figure_number_only(text) or page_header_like(text) or incomplete_body_fragment(text):
        return "plain_body_text"
    if explicit_caption_like(text):
        return "explicit_caption"
    if title_like(text):
        return "title_caption"
    if short_art_caption(text):
        return "short_art_caption"
    return "plain_body_text"


def explicit_caption_like(text: str) -> bool:
    normalized = normalize_caption(text)
    return bool(re.match(r"^(〔?\[?【?图|〔?\[?【?圖|Fig\.?|Figure|Plate|PLATE)", normalized, flags=re.IGNORECASE))


def title_like(text: str) -> bool:
    return bool(re.search(r"《[^》]{1,40}》", text or "")) and len(normalize_caption(text)) <= 180


def short_art_caption(text: str) -> bool:
    normalized = normalize_caption(text)
    if len(normalized) > 120:
        return False
    if OBVIOUS_NON_LANDSCAPE_PATTERN.search(normalized):
        return False
    return bool(LANDSCAPE_OR_ART_PATTERN.search(normalized))


def figure_number_only(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    compact = normalize_caption(normalized)
    # Very short numbered captions such as 图3.9 or Figure 6.2 do not provide
    # enough evidence for claim writing even when the bbox itself is correct.
    return bool(FIGURE_NUMBER_ONLY_PATTERN.match(normalized)) and len(compact) <= 24


def page_header_like(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return False
    return bool(PAGE_HEADER_PATTERN.match(normalized))


def incomplete_body_fragment(text: str) -> bool:
    stripped = str(text or "").strip()
    if not stripped:
        return False
    if re.match(r"^[a-z]{1,4}\s+", stripped):
        return True
    if stripped.startswith(("，", "。", "、", "；", "：", ",", ".", ";", ":", "”", "’", "'", '"')):
        return True
    compact = normalize_caption(stripped)
    if compact.startswith(("图》", "圖》", "》")):
        return True
    return False


def normalize_caption(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip()


def candidate_quality_score(
    source: str,
    caption_text: str,
    caption_score: float,
    area_ratio: float,
    text_overlap: float,
    caption_quality: str,
) -> float:
    source_bonus = 3.0 if source == "pdf_image_block" else 0.6
    quality_bonus = {
        "explicit_caption": 2.0,
        "title_caption": 1.4,
        "short_art_caption": 0.8,
        "plain_body_text": -4.0,
    }.get(caption_quality, 0.0)
    landscape_bonus = 0.8 if LANDSCAPE_OR_ART_PATTERN.search(caption_text or "") else 0.0
    caption_bonus = max(-2.0, min(4.0, caption_score / 2.0))
    size_bonus = min(1.0, area_ratio / 0.16)
    return round(source_bonus + quality_bonus + landscape_bonus + caption_bonus + size_bonus - 2.0 * text_overlap, 6)


def build_summary(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    page_rows: list[dict[str, Any]],
    filtered_pages: list[dict[str, Any]],
    filtered_figures: list[dict[str, Any]],
    kept_counter: Counter[str],
    rejected_counter: Counter[str],
    reason_counter: Counter[str],
) -> dict[str, Any]:
    input_figures = sum(len(row.get("figure_candidates") or []) for row in page_rows)
    output_sources = Counter(fig.get("source") for fig in filtered_figures)
    caption_quality_counts = Counter((fig.get("v1_0_2_filter") or {}).get("caption_quality") for fig in filtered_figures)
    return {
        "created_at": now(),
        "input_cache_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_page_rows": len(page_rows),
        "input_figure_candidates": input_figures,
        "kept_page_rows": len(filtered_pages),
        "kept_figure_candidates": len(filtered_figures),
        "kept_unique_sources": len({row.get("source_file") for row in filtered_pages}),
        "keep_rate": len(filtered_figures) / max(1, input_figures),
        "kept_by_source": dict(kept_counter),
        "rejected_by_source": dict(rejected_counter),
        "output_candidate_sources": dict(output_sources),
        "caption_quality_counts": dict(caption_quality_counts),
        "top_reject_reasons": dict(reason_counter.most_common(30)),
        "args": vars(args),
    }


def build_manifest(args: argparse.Namespace, input_dir: Path, output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": summary["created_at"],
        "dataset_version": "layout_candidates_v1_0_2_rule_filtered",
        "builder": "scripts/filter_layout_candidates_v1_0_2.py",
        "input_cache_dir": str(input_dir),
        "output_dir": str(output_dir),
        "summary": summary,
        "files": {
            "page_candidates": str(output_dir / "page_candidates.jsonl"),
            "figure_candidates": str(output_dir / "figure_candidates.jsonl"),
            "filter_decisions": str(output_dir / "filter_decisions.jsonl"),
            "summary": str(output_dir / "summary.json"),
            "report": str(output_dir / "v1.0.2候选过滤报告.md"),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.2 Layout Candidate 规则过滤报告",
        "",
        f"生成时间：{summary['created_at']} CST",
        "",
        "## 输入与输出",
        "",
        f"- input_cache_dir：`{summary['input_cache_dir']}`",
        f"- output_dir：`{summary['output_dir']}`",
        "",
        "## 规模变化",
        "",
        f"- input_page_rows：{summary['input_page_rows']}",
        f"- input_figure_candidates：{summary['input_figure_candidates']}",
        f"- kept_page_rows：{summary['kept_page_rows']}",
        f"- kept_figure_candidates：{summary['kept_figure_candidates']}",
        f"- kept_unique_sources：{summary['kept_unique_sources']}",
        f"- keep_rate：{summary['keep_rate']:.4f}",
        "",
        "## 来源分布",
        "",
        f"- kept_by_source：`{json.dumps(summary['kept_by_source'], ensure_ascii=False)}`",
        f"- rejected_by_source：`{json.dumps(summary['rejected_by_source'], ensure_ascii=False)}`",
        f"- caption_quality_counts：`{json.dumps(summary['caption_quality_counts'], ensure_ascii=False)}`",
        "",
        "## 主要拒绝原因",
        "",
        f"`{json.dumps(summary['top_reject_reasons'], ensure_ascii=False)}`",
        "",
        "## 说明",
        "",
        "- 这一步是 deterministic pre-filter，规则来自 v1.0.1 的 VLM 抽检错误模式。",
        "- 重点过滤普通正文被当作 caption、OpenCV 文本块误检、极窄/过小/高文本重叠候选。",
        "- 后续还要对构建出的 AgentBench 做 VLM 抽检或 VLM 裁决，不把 oracle replay 当作唯一质量证明。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def default_output_dir(root: Path) -> Path:
    return root / f"layout_candidates_v1_0_2_rule_filtered_{datetime.now().strftime('%Y%m%d_%H%M')}"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
