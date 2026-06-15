#!/usr/bin/env python3
"""Build a v1.2 remote-VLM-first probe from raw PDFs.

v1.2 treats page-level remote VLM readings as the primary annotation source
while preserving raw PDF text as provenance/audit material.
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_gold_eval_v1_0_4 as gold_review  # noqa: E402
import build_v1_1_clean_evidence_fragment_probe as v11  # noqa: E402


RAW_PDF_ROOT = Path("/root/datasets/chinese_landscape_authority_corpus/raw_pdfs")
SOURCE_JSONL = Path("/root/datasets/chinese_landscape_authority_corpus/metadata/sources.jsonl")
OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DOCS_DIR = REPO_ROOT / "docs" / "02_指标与数据"
DOTENV = REPO_ROOT / ".env"

BASE_FIELDS = ["caption_text", "depicted_work_title", "image_scope", "object_type"]
META_FIELDS = [
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
]

LANDSCAPE_DOMAINS = {
    "landscape_painting",
    "landscape_detail",
    "classical_painting_unclear_landscape",
}
GOOD_OBJECT_TYPES = {"painting", "painting_detail"}

PAGE_LEVEL_PROMPT_V12 = """你是 EvidenceGrounded-VLM-AgentRL v1.2 的 Remote-VLM-first 数据构建员。

你会看到一整页 PDF 渲染图。请直接从页面视觉内容中找出适合构建数据集的“中国/东亚古典山水画或山水画局部”目标，并输出目标图像框、对应图注框、完整 corrected caption、Core4 字段和 Metadata5 字段。

核心原则：
- 远端 VLM 的页面视觉读字是主标注来源；
- 如果 PDF/OCR 文本预览与页面视觉读字不一致，以页面视觉读字为准；
- 但不要使用模型内部知识补全页面没有出现的信息；
- 如果某个 metadata 字段页面中没有可见证据，必须 abstain；
- 如果字段来自图注，请在 source 写 caption_visual；
- 如果字段来自同页正文，请在 source 写 same_page_body_visual；
- 如果无法支持，请 source 写 unsupported。

坐标要求：
- 坐标必须是 0-1000 归一化坐标；
- 原点左上，右下约为 [1000,1000]；
- bbox 格式为 [x1,y1,x2,y2]；
- target_bbox_norm1000 只框目标图像，不要包含图注、正文、相邻图；
- caption_bbox_norm1000 只框对应图注，不要包含正文或其他图注；
- 如果图注是多行，caption_bbox_norm1000 应覆盖完整多行图注。

接受对象：
- 中国/东亚古典山水画、山水画局部、以山水/自然景观为主体的古典绘画；
- 图录页、论文插图页、博物馆图版页中的山水画。

排除对象：
- 纯正文、目录、索引、表格、参考文献页；
- 建筑/器物照片、地图、装饰纹样、一般图式；
- 书法题跋、人物故事、佛教叙事、动物/骑乘/乐人等非山水主体；
- 图像与图注无法明确对应的对象。

字段说明：
- caption_text：页面可见的完整 corrected caption；
- depicted_work_title：当前目标图所描绘的作品题名；
- image_scope：full_work | partial_detail | album_leaf_or_section | multi_work_comparison | unclear；
- object_type：painting | painting_detail | diagram | text_page | photo | other | unclear；
- Metadata5：
  - creator_or_attribution：作者/传称作者/归属；
  - creation_period_or_dynasty：朝代/时期/年份/世纪；
  - collection_institution：藏馆/收藏机构/收藏来源；
  - dimensions：尺寸；
  - medium_material：材质/媒介/设色方式。

只输出 JSON，不要输出 Markdown。schema:
{
  "page_summary": "一句话说明页面内容",
  "detections": [
    {
      "target_bbox_norm1000": [0,0,0,0],
      "caption_bbox_norm1000": [0,0,0,0],
      "caption_text": "页面视觉读出的完整 corrected caption",
      "depicted_work_title": "作品题名，无法确定则空串",
      "image_scope": "full_work|partial_detail|album_leaf_or_section|multi_work_comparison|unclear",
      "object_type": "painting|painting_detail|diagram|text_page|photo|other|unclear",
      "object_domain": "landscape_painting|landscape_detail|classical_painting_unclear_landscape|non_landscape_artwork|text_only|other|unclear",
      "caption_target_match": "yes|no|uncertain",
      "metadata_fields": {
        "creator_or_attribution": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "creation_period_or_dynasty": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "collection_institution": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "dimensions": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "medium_material": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""}
      },
      "accept_for_probe": true,
      "needs_human_review": false,
      "confidence": 0.0,
      "reason": "一句话说明图文绑定和是否可接受"
    }
  ]
}

页面元数据：
"""

PAGE_LEVEL_PROMPT_V12_COMPACT = """你是 v1.2 Remote-VLM-first 数据构建员。请看整页 PDF 图，找出中国/东亚古典山水画或山水画局部目标。

要求：
- bbox 用 0-1000 归一化坐标 [x1,y1,x2,y2]。
- target_bbox_norm1000 只框图像，不含图注/正文/相邻图。
- caption_bbox_norm1000 框完整对应图注，多行图注要全框。
- caption_text 以页面视觉读字为准；如果 PDF 文本预览读错，以图像为准。
- 不要用模型内部知识补全页面没有出现的信息。
- Metadata5 没有页面可见证据就 abstain。

只输出 JSON：
{
  "page_summary": "一句话",
  "detections": [
    {
      "target_bbox_norm1000": [0,0,0,0],
      "caption_bbox_norm1000": [0,0,0,0],
      "caption_text": "完整 corrected caption",
      "depicted_work_title": "题名或空串",
      "image_scope": "full_work|partial_detail|album_leaf_or_section|multi_work_comparison|unclear",
      "object_type": "painting|painting_detail|diagram|text_page|photo|other|unclear",
      "object_domain": "landscape_painting|landscape_detail|classical_painting_unclear_landscape|non_landscape_artwork|text_only|other|unclear",
      "caption_target_match": "yes|no|uncertain",
      "metadata_fields": {
        "creator_or_attribution": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "creation_period_or_dynasty": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "collection_institution": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "dimensions": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "medium_material": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""}
      },
      "accept_for_probe": true,
      "needs_human_review": false,
      "confidence": 0.0,
      "reason": "简短说明"
    }
  ]
}

页面元数据：
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1.2 remote-VLM-first probe.")
    parser.add_argument("--raw-pdf-root", default=str(RAW_PDF_ROOT))
    parser.add_argument("--sources-jsonl", default=str(SOURCE_JSONL))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--input-probe-dir", default="", help="Reuse page_records/text/layout from an existing probe dir and only rerun VLM/postprocess.")
    parser.add_argument("--docs-dir", default=str(DOCS_DIR))
    parser.add_argument("--dotenv", default=str(DOTENV))
    parser.add_argument("--probe-pages", type=int, default=10)
    parser.add_argument("--max-pdfs", type=int, default=100)
    parser.add_argument(
        "--max-pages-per-pdf",
        type=int,
        default=0,
        help="0 means scan all pages for page candidates; positive values sample per PDF.",
    )
    parser.add_argument("--max-pages-per-source", type=int, default=2)
    parser.add_argument("--require-image-block", action="store_true", default=True)
    parser.add_argument("--max-page-aspect-ratio", type=float, default=1.8)
    parser.add_argument("--page-dpi", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--vlm-provider", choices=["dashscope", "local", "offline"], default="dashscope")
    parser.add_argument("--dashscope-model", default="qwen3.7-max-2026-06-08")
    parser.add_argument(
        "--dashscope-fallback-models",
        default="qwen3.7-max,qwen3.7-plus-2026-05-26,qwen3.7-plus,qwen3.6-plus,qwen3.6-27b,glm-5.1,kimi-k2.6,deepseek-v4-pro,deepseek-v4-flash",
    )
    parser.add_argument("--local-model", default="/root/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--local-device", default="cuda:0")
    parser.add_argument("--image-max-side", type=int, default=1800)
    parser.add_argument("--max-new-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--metadata-near-window", type=int, default=500)
    parser.add_argument("--review-limit", type=int, default=80)
    parser.add_argument("--compact-prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_review.load_dotenv(Path(args.dotenv))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"v1_2_remote_vlm_first_probe_{stamp}"
    prepare_output_dir(output_dir, args)

    if args.input_probe_dir:
        input_dir = Path(args.input_probe_dir)
        pages, page_records, text_blocks, layout_blocks = load_existing_probe_assets(input_dir)
        shutil.copy2(input_dir / "page_manifest.jsonl", output_dir / "page_manifest.jsonl")
        v11.write_jsonl(output_dir / "page_records.jsonl", page_records)
        v11.write_jsonl(output_dir / "page_text_blocks.jsonl", text_blocks)
        v11.write_jsonl(output_dir / "page_layout_blocks.jsonl", layout_blocks)
    else:
        source_registry = v11.load_source_registry(Path(args.sources_jsonl))
        pages = select_probe_pages_v12(Path(args.raw_pdf_root), source_registry, args)
        v11.write_jsonl(output_dir / "page_manifest.jsonl", [page_spec_to_record(p) for p in pages])

        page_records, text_blocks, layout_blocks = materialize_pages_and_blocks_v12(output_dir, pages, args)
        v11.write_jsonl(output_dir / "page_records.jsonl", page_records)
        v11.write_jsonl(output_dir / "page_text_blocks.jsonl", text_blocks)
        v11.write_jsonl(output_dir / "page_layout_blocks.jsonl", layout_blocks)

    v11.PAGE_LEVEL_PROMPT = PAGE_LEVEL_PROMPT_V12_COMPACT if args.compact_prompt else PAGE_LEVEL_PROMPT_V12
    v11.normalize_vlm_response = normalize_vlm_response_v12
    client = v11.make_vlm_client(args)
    vlm_rows = v11.run_page_level_vlm(output_dir, page_records, client, args)
    v11.write_jsonl(output_dir / "page_level_vlm_annotations.jsonl", vlm_rows)

    targets = build_figure_targets_v12(output_dir, page_records, text_blocks, vlm_rows, args)
    v11.write_jsonl(output_dir / "figure_targets.jsonl", targets)

    fragments = build_evidence_fragments_v12(targets, text_blocks, args)
    v11.write_jsonl(output_dir / "evidence_fragments.jsonl", fragments)

    field_values, support_labels = build_field_values_and_support_labels_v12(targets, fragments)
    v11.write_jsonl(output_dir / "field_values.jsonl", field_values)
    v11.write_jsonl(output_dir / "field_support_labels_vlm_first.jsonl", support_labels)

    review_package = write_review_package_v12(output_dir, page_records, targets, fragments, field_values, support_labels, args)
    summary = build_summary_v12(
        output_dir,
        args,
        pages,
        page_records,
        text_blocks,
        layout_blocks,
        vlm_rows,
        targets,
        fragments,
        field_values,
        support_labels,
        review_package,
    )
    v11.write_json(output_dir / "manifest.json", summary)
    report = write_report_v12(output_dir / "构建报告.md", summary)
    docs_path = Path(args.docs_dir) / f"{stamp}_v1.2RemoteVLMFirstProbe构建报告.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    summary["artifacts"]["docs_report"] = str(docs_path)
    v11.write_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def prepare_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists; use --resume or --overwrite")
    for child in ["pages", "page_overlays", "overlays", "crops", "review", "cache"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def load_existing_probe_assets(input_dir: Path) -> tuple[list[v11.PageSpec], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    page_records = v11.read_jsonl(input_dir / "page_records.jsonl")
    text_blocks = v11.read_jsonl(input_dir / "page_text_blocks.jsonl")
    layout_blocks = v11.read_jsonl(input_dir / "page_layout_blocks.jsonl")
    manifest_rows = v11.read_jsonl(input_dir / "page_manifest.jsonl")
    pages: list[v11.PageSpec] = []
    raw_root = Path("/")
    for idx, row in enumerate(manifest_rows):
        source_path = Path(row.get("source_path") or page_records[idx].get("source_path") or "")
        if not source_path.is_absolute():
            source_path = raw_root / source_path
        pages.append(
            v11.PageSpec(
                doc_id=str(row.get("doc_id") or page_records[idx].get("doc_id") or ""),
                source_file=str(row.get("source_file") or page_records[idx].get("source_file") or ""),
                source_path=source_path,
                rel_path=str(row.get("rel_path") or page_records[idx].get("rel_path") or ""),
                page_num=int(row.get("page_num") or page_records[idx].get("page_num") or 0),
                page_count=int(row.get("page_count") or page_records[idx].get("page_count") or 0),
                category=str(row.get("category") or page_records[idx].get("category") or ""),
                score=float(row.get("score") or page_records[idx].get("page_score") or 0.0),
                image_count=int(row.get("image_count") or page_records[idx].get("selection_stats", {}).get("image_count") or 0),
                figure_anchor_count=int(row.get("figure_anchor_count") or page_records[idx].get("selection_stats", {}).get("figure_anchor_count") or 0),
                landscape_term_count=int(row.get("landscape_term_count") or page_records[idx].get("selection_stats", {}).get("landscape_term_count") or 0),
                text_preview=str(row.get("text_preview") or ""),
            )
        )
    return pages, page_records, text_blocks, layout_blocks


def select_probe_pages_v12(raw_root: Path, registry: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[v11.PageSpec]:
    del registry
    rng = random.Random(args.seed)
    pdfs = sorted(raw_root.rglob("*.pdf"))
    if args.max_pdfs > 0 and len(pdfs) > args.max_pdfs:
        grouped: dict[str, list[Path]] = defaultdict(list)
        for pdf in pdfs:
            rel = pdf.relative_to(raw_root)
            grouped[rel.parts[0] if len(rel.parts) > 1 else "root"].append(pdf)
        selected: list[Path] = []
        per_cat = max(1, args.max_pdfs // max(1, len(grouped)))
        for _, group in sorted(grouped.items()):
            rng.shuffle(group)
            selected.extend(group[:per_cat])
        if len(selected) < args.max_pdfs:
            rest = [p for p in pdfs if p not in selected]
            rng.shuffle(rest)
            selected.extend(rest[: args.max_pdfs - len(selected)])
        pdfs = selected[: args.max_pdfs]

    candidates: list[v11.PageSpec] = []
    for pdf in pdfs:
        try:
            rel = str(pdf.relative_to(raw_root))
            category = Path(rel).parts[0] if len(Path(rel).parts) > 1 else "root"
            with fitz.open(pdf) as doc:
                page_count = len(doc)
                page_indices = range(page_count) if args.max_pages_per_pdf <= 0 else v11.candidate_page_indices(page_count, args.max_pages_per_pdf)
                for page_index in page_indices:
                    page = doc[page_index]
                    aspect = max(float(page.rect.width) / max(float(page.rect.height), 1.0), float(page.rect.height) / max(float(page.rect.width), 1.0))
                    if args.max_page_aspect_ratio > 0 and aspect > args.max_page_aspect_ratio:
                        continue
                    image_count = len(page.get_images(full=False))
                    if args.require_image_block and image_count == 0:
                        continue
                    text = v11.normalize_space(page.get_text("text") or "")
                    if re.search(r"photograph credits|photo credits|index|contents|bibliography|references", text[:1200], re.I):
                        continue
                    figure_count = len(v11.FIGURE_ANCHOR_RE.findall(text))
                    landscape_count = len(v11.LANDSCAPE_RE.findall(text[:4000]))
                    metadata_hits = count_metadata_hints(text[:4000])
                    noise_penalty = 8 if v11.NON_TASK_PAGE_RE.search(text[:1000]) else 0
                    figure_score = min(figure_count, 10) * 4.0
                    if figure_count > 20 and landscape_count == 0:
                        figure_score -= 24.0
                    image_score = min(image_count, 8) * 8.0
                    if image_count > 12 and figure_count == 0:
                        image_score -= 20.0
                    score = image_score + figure_score + min(landscape_count, 12) * 0.6 + min(metadata_hits, 8) * 1.4 - noise_penalty
                    if score <= 0:
                        continue
                    candidates.append(
                        v11.PageSpec(
                            doc_id=v11.doc_id_for_path(pdf),
                            source_file=pdf.name,
                            source_path=pdf,
                            rel_path=rel,
                            page_num=page_index + 1,
                            page_count=page_count,
                            category=category,
                            score=round(score + rng.random() * 0.01, 4),
                            image_count=image_count,
                            figure_anchor_count=figure_count,
                            landscape_term_count=landscape_count,
                            text_preview=text[:220],
                        )
                    )
        except Exception as exc:
            print(json.dumps({"scan_error": str(pdf), "error": repr(exc)}, ensure_ascii=False), flush=True)

    candidates.sort(key=lambda p: p.score, reverse=True)
    selected: list[v11.PageSpec] = []
    per_source_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    for item in candidates:
        if per_source_counter[item.source_file] >= args.max_pages_per_source:
            continue
        if category_counter[item.category] >= max(2, args.probe_pages // 2) and len(category_counter) > 1:
            continue
        selected.append(item)
        per_source_counter[item.source_file] += 1
        category_counter[item.category] += 1
        if len(selected) >= args.probe_pages:
            break
    if len(selected) < args.probe_pages:
        for item in candidates:
            if item in selected:
                continue
            if per_source_counter[item.source_file] >= args.max_pages_per_source:
                continue
            selected.append(item)
            per_source_counter[item.source_file] += 1
            if len(selected) >= args.probe_pages:
                break
    selected.sort(key=lambda p: (p.category, p.source_file, p.page_num))
    return selected[: args.probe_pages]


def count_metadata_hints(text: str) -> int:
    hits = 0
    hits += len(v11.DIMENSION_RE.findall(text))
    hits += len(v11.DYNASTY_RE.findall(text))
    hits += sum(1 for regex in v11.INSTITUTION_RES if regex.search(text))
    hits += sum(1 for regex, _ in v11.MEDIUM_RES if regex.search(text))
    hits += len(v11.TITLE_RE.findall(text))
    return hits


def materialize_pages_and_blocks_v12(
    output_dir: Path,
    pages: list[v11.PageSpec],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    page_records: list[dict[str, Any]] = []
    text_blocks: list[dict[str, Any]] = []
    layout_blocks: list[dict[str, Any]] = []
    for idx, spec in enumerate(pages, start=1):
        page_id = f"v12p_{idx:04d}"
        page_path = output_dir / "pages" / f"{v11.safe_stem(spec.source_path.stem)}_p{spec.page_num:04d}.png"
        width = height = 0
        pdf_text = ""
        try:
            with fitz.open(spec.source_path) as doc:
                page = doc[spec.page_num - 1]
                pix = page.get_pixmap(dpi=args.page_dpi, colorspace=fitz.csRGB)
                pix.save(page_path)
                width, height = pix.width, pix.height
                sx = width / page.rect.width
                sy = height / page.rect.height
                pdf_text = v11.normalize_space(page.get_text("text") or "")
                for bidx, block in enumerate(page.get_text("blocks") or []):
                    if len(block) < 5:
                        continue
                    x0, y0, x1, y1, text = block[:5]
                    text = v11.normalize_space(text)
                    if not text:
                        continue
                    bbox = v11.scale_bbox([x0, y0, x1, y1], sx, sy)
                    text_blocks.append(
                        {
                            "block_id": f"{page_id}_tb{bidx:04d}",
                            "page_id": page_id,
                            "doc_id": spec.doc_id,
                            "source_file": spec.source_file,
                            "page_num": spec.page_num,
                            "block_type": "text",
                            "bbox": bbox,
                            "bbox_norm1000": v11.norm_bbox(bbox, width, height),
                            "text": text,
                            "raw_text": text,
                            "source": "pymupdf_text_block",
                        }
                    )
                for iidx, img in enumerate(page.get_image_info(xrefs=True) or []):
                    bbox_raw = img.get("bbox")
                    if not bbox_raw:
                        continue
                    bbox = v11.scale_bbox(list(bbox_raw), sx, sy)
                    layout_blocks.append(
                        {
                            "block_id": f"{page_id}_ib{iidx:04d}",
                            "page_id": page_id,
                            "doc_id": spec.doc_id,
                            "source_file": spec.source_file,
                            "page_num": spec.page_num,
                            "block_type": "image",
                            "bbox": bbox,
                            "bbox_norm1000": v11.norm_bbox(bbox, width, height),
                            "text": "",
                            "source": "pymupdf_image_block",
                            "xref": img.get("xref"),
                            "width": img.get("width"),
                            "height": img.get("height"),
                        }
                    )
        except Exception as exc:
            print(json.dumps({"materialize_error": spec.rel_path, "page": spec.page_num, "error": repr(exc)}, ensure_ascii=False), flush=True)
        page_records.append(
            {
                "page_id": page_id,
                "doc_id": spec.doc_id,
                "source_file": spec.source_file,
                "source_path": str(spec.source_path),
                "rel_path": spec.rel_path,
                "category": spec.category,
                "page_num": spec.page_num,
                "page_count": spec.page_count,
                "page_image": str(page_path),
                "width": width,
                "height": height,
                "raw_pdf_text": pdf_text,
                "pdf_text": pdf_text,
                "page_score": spec.score,
                "selection_stats": {
                    "image_count": spec.image_count,
                    "figure_anchor_count": spec.figure_anchor_count,
                    "landscape_term_count": spec.landscape_term_count,
                    "metadata_hint_count": count_metadata_hints(spec.text_preview),
                },
            }
        )
    return page_records, text_blocks, layout_blocks


def normalize_vlm_response_v12(page: dict[str, Any], parsed: dict[str, Any], raw: str, model: str, input_mode: str) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    raw_detections = parsed.get("detections") if isinstance(parsed.get("detections"), list) else []
    for idx, item in enumerate(raw_detections):
        if not isinstance(item, dict):
            continue
        target_norm = v11.normalize_bbox_value(item.get("target_bbox_norm1000"))
        caption_norm = v11.normalize_bbox_value(item.get("caption_bbox_norm1000"))
        target_bbox = v11.scale_norm1000_bbox(target_norm, page["width"], page["height"]) if target_norm else None
        caption_bbox = v11.scale_norm1000_bbox(caption_norm, page["width"], page["height"]) if caption_norm else None
        caption = v11.normalize_space(item.get("caption_text") or "")[:1200]
        metadata = normalize_metadata_fields(item.get("metadata_fields") or item.get("metadata") or {}, caption)
        detections.append(
            {
                "detection_index": idx,
                "target_bbox_norm1000": target_norm,
                "caption_bbox_norm1000": caption_norm,
                "target_bbox": target_bbox,
                "caption_bbox": caption_bbox,
                "caption_text": caption,
                "corrected_caption_text": caption,
                "depicted_work_title": v11.normalize_space(item.get("depicted_work_title") or v11.extract_title(caption))[:160],
                "image_scope": normalize_image_scope(item.get("image_scope")),
                "object_type": normalize_object_type_v12(item.get("object_type"), item.get("object_domain")),
                "object_domain": v11.normalize_space(item.get("object_domain") or "unclear"),
                "caption_target_match": v11.normalize_space(item.get("caption_target_match") or "uncertain"),
                "metadata_fields": metadata,
                "accept_for_probe": bool(item.get("accept_for_probe")),
                "needs_human_review": bool(item.get("needs_human_review")),
                "confidence": max(0.0, min(1.0, v11.safe_float(item.get("confidence"), 0.0))),
                "reason": str(item.get("reason") or "")[:800],
            }
        )
    return {
        **v11.page_base(page),
        "ok": True,
        "review_model": model,
        "input_mode": input_mode,
        "page_summary": str(parsed.get("page_summary") or "")[:800],
        "detections": detections,
        "raw_response": raw,
    }


def normalize_metadata_fields(raw: Any, caption: str) -> dict[str, dict[str, Any]]:
    out = {field: empty_field_entry() for field in META_FIELDS}
    if isinstance(raw, list):
        raw = {str(item.get("field")): item for item in raw if isinstance(item, dict) and item.get("field")}
    if isinstance(raw, dict):
        for field in META_FIELDS:
            item = raw.get(field)
            if isinstance(item, dict):
                value = v11.normalize_space(item.get("value") or "")
                abstain = bool(item.get("abstain")) or not value
                out[field] = {
                    "value": value,
                    "normalized_value": v11.normalize_field_value(field, value) if value else "",
                    "abstain": abstain,
                    "source": v11.normalize_space(item.get("source") or ("unsupported" if abstain else "caption_visual")),
                    "confidence": max(0.0, min(1.0, v11.safe_float(item.get("confidence"), 0.0))),
                    "reason": str(item.get("reason") or "")[:500],
                    "extraction_source": "remote_vlm_page_read",
                }
            elif isinstance(item, str) and item.strip():
                value = v11.normalize_space(item)
                out[field] = {
                    "value": value,
                    "normalized_value": v11.normalize_field_value(field, value),
                    "abstain": False,
                    "source": "caption_visual",
                    "confidence": 0.7,
                    "reason": "VLM returned a string value for this field.",
                    "extraction_source": "remote_vlm_page_read",
                }
    # Low-risk regex fallback only on the VLM corrected caption.
    extracted = v11.extract_metadata(caption)
    for field, value in extracted.items():
        if field in out and out[field]["abstain"]:
            out[field] = {
                "value": value,
                "normalized_value": v11.normalize_field_value(field, value),
                "abstain": False,
                "source": "caption_visual_corrected_regex",
                "confidence": 0.72,
                "reason": "规则从 VLM corrected caption 中抽到明确格式字段。",
                "extraction_source": "regex_on_corrected_caption",
            }
    return out


def empty_field_entry() -> dict[str, Any]:
    return {
        "value": "",
        "normalized_value": "",
        "abstain": True,
        "source": "unsupported",
        "confidence": 0.0,
        "reason": "",
        "extraction_source": "remote_vlm_page_read",
    }


def normalize_image_scope(value: Any) -> str:
    value = v11.normalize_space(value)
    allowed = {"full_work", "partial_detail", "album_leaf_or_section", "multi_work_comparison", "unclear"}
    if value in allowed:
        return value
    if value in {"detail", "局部", "细部"}:
        return "partial_detail"
    return "unclear"


def normalize_object_type_v12(value: Any, domain: Any) -> str:
    value = v11.normalize_space(value)
    domain = v11.normalize_space(domain)
    allowed = {"painting", "painting_detail", "diagram", "text_page", "photo", "other", "unclear"}
    if value in allowed:
        return value
    if domain == "landscape_detail":
        return "painting_detail"
    if domain in {"landscape_painting", "classical_painting_unclear_landscape"}:
        return "painting"
    if domain == "text_only":
        return "text_page"
    return "unclear"


def build_figure_targets_v12(
    output_dir: Path,
    page_records: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    page_by_id = {p["page_id"]: p for p in page_records}
    blocks_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in text_blocks:
        blocks_by_page[block["page_id"]].append(block)
    targets: list[dict[str, Any]] = []
    for row in vlm_rows:
        page = page_by_id.get(row.get("page_id"))
        if not page:
            continue
        for det in row.get("detections") or []:
            conf = v11.safe_float(det.get("confidence"), 0.0)
            if conf < args.min_detection_confidence:
                continue
            target_bbox = det.get("target_bbox")
            caption_bbox = det.get("caption_bbox")
            if not v11.valid_bbox(target_bbox) or not v11.valid_bbox(caption_bbox):
                continue
            corrected_caption = v11.normalize_space(det.get("corrected_caption_text") or det.get("caption_text") or "")
            title = v11.normalize_space(det.get("depicted_work_title") or v11.extract_title(corrected_caption))
            raw_caption_text, raw_caption_blocks = find_raw_caption_text(blocks_by_page.get(page["page_id"], []), caption_bbox, corrected_caption, title)
            status, status_reason = acceptance_status(det, corrected_caption)
            target_id = f"v12_{page['page_id']}_t{len(targets):04d}"
            crop_path = output_dir / "crops" / f"{target_id}.jpg"
            overlay_path = output_dir / "overlays" / f"{target_id}.jpg"
            v11.crop_image(Path(page["page_image"]), target_bbox, crop_path)
            v11.draw_target_overlay(Path(page["page_image"]), target_bbox, caption_bbox, overlay_path)
            targets.append(
                {
                    "target_id": target_id,
                    "page_id": page["page_id"],
                    "doc_id": page["doc_id"],
                    "source_file": page["source_file"],
                    "source_path": page["source_path"],
                    "page_num": page["page_num"],
                    "page_image": page["page_image"],
                    "overlay_image": str(overlay_path),
                    "target_crop": str(crop_path),
                    "image_bbox": target_bbox,
                    "caption_bbox": caption_bbox,
                    "image_bbox_norm1000": det.get("target_bbox_norm1000") or v11.norm_bbox(target_bbox, page["width"], page["height"]),
                    "caption_bbox_norm1000": det.get("caption_bbox_norm1000") or v11.norm_bbox(caption_bbox, page["width"], page["height"]),
                    "caption_text": corrected_caption,
                    "corrected_caption_text": corrected_caption,
                    "raw_caption_text": raw_caption_text,
                    "raw_caption_source_blocks": raw_caption_blocks,
                    "depicted_work_title": title,
                    "image_scope": normalize_image_scope(det.get("image_scope")),
                    "object_type": normalize_object_type_v12(det.get("object_type"), det.get("object_domain")),
                    "object_domain": det.get("object_domain") or "unclear",
                    "caption_target_match": det.get("caption_target_match") or "uncertain",
                    "metadata_fields": det.get("metadata_fields") or normalize_metadata_fields({}, corrected_caption),
                    "acceptance_status": status,
                    "acceptance_reason": status_reason,
                    "confidence": conf,
                    "needs_human_review": bool(det.get("needs_human_review")) or status != "accepted",
                    "reason": det.get("reason") or "",
                    "review_model": row.get("review_model"),
                    "source_stage": "v1.2_remote_vlm_first_probe",
                    "is_model_internal_knowledge": False,
                }
            )
    write_page_overlays(output_dir, page_records, targets)
    return targets


def find_raw_caption_text(blocks: list[dict[str, Any]], caption_bbox: list[int], corrected_caption: str, title: str) -> tuple[str, list[str]]:
    fig = v11.first_figure_anchor(corrected_caption)
    candidates: list[tuple[float, dict[str, Any]]] = []
    for block in blocks:
        text = v11.normalize_space(block.get("text") or "")
        if not text:
            continue
        overlap = bbox_overlap_ratio(caption_bbox, block.get("bbox") or [0, 0, 0, 0])
        anchor_bonus = 0.0
        if fig and fig in text:
            anchor_bonus += 2.0
        if title and title in text:
            anchor_bonus += 1.5
        if corrected_caption and corrected_caption[:24] in text:
            anchor_bonus += 1.0
        score = overlap + anchor_bonus
        if score > 0:
            candidates.append((score, block))
    candidates.sort(key=lambda x: x[0], reverse=True)
    selected = [block for score, block in candidates[:2] if score > 0.05]
    text = " ".join(v11.normalize_space(b.get("text") or "") for b in selected)
    return text, [str(b.get("block_id")) for b in selected]


def bbox_overlap_ratio(a: list[int], b: list[int]) -> float:
    if not v11.valid_bbox(a) or not v11.valid_bbox(b):
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / min(a_area, b_area)


def acceptance_status(det: dict[str, Any], corrected_caption: str) -> tuple[str, str]:
    conf = v11.safe_float(det.get("confidence"), 0.0)
    match = v11.normalize_space(det.get("caption_target_match") or "uncertain")
    domain = v11.normalize_space(det.get("object_domain") or "unclear")
    obj_type = normalize_object_type_v12(det.get("object_type"), domain)
    if match == "no" or obj_type in {"diagram", "text_page", "photo", "other"} or domain in {"non_landscape_artwork", "text_only", "other"}:
        return "rejected", "VLM 判断对象或图文绑定不符合山水画目标。"
    if conf >= 0.85 and match == "yes" and domain in LANDSCAPE_DOMAINS and obj_type in GOOD_OBJECT_TYPES and not caption_is_too_weak(corrected_caption):
        return "accepted", "高置信远端 VLM 页面读字与图文绑定自洽，可进入 v1.2 high-confidence silver。"
    return "needs_review", "置信度、图文绑定、caption 完整性或对象类型仍需人工/二审确认。"


def caption_is_too_weak(text: str) -> bool:
    text = v11.normalize_space(text)
    if len(text) < 12:
        return True
    stripped = re.sub(r"^(图|圖|Fig\.?|Figure|Plate|Pl\.)\s*[\w.\-．:：]+", "", text, flags=re.I).strip()
    return len(stripped) < 5


def build_evidence_fragments_v12(targets: list[dict[str, Any]], text_blocks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    blocks_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in text_blocks:
        blocks_by_page[block["page_id"]].append(block)
    fragments: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        corrected = v11.normalize_space(target.get("corrected_caption_text") or target.get("caption_text") or "")
        raw = v11.normalize_space(target.get("raw_caption_text") or "")
        if corrected:
            fragments.append(make_fragment_v12(target, "local_caption_visual_corrected", raw, corrected, corrected, target.get("raw_caption_source_blocks") or [], [target["caption_bbox"]], "remote_vlm_page_read"))
        if raw and raw != corrected:
            fragments.append(make_fragment_v12(target, "local_caption_pdf_raw", raw, raw, raw, target.get("raw_caption_source_blocks") or [], [target["caption_bbox"]], "pymupdf_text_block_audit"))
        title = v11.normalize_space(target.get("depicted_work_title") or v11.extract_title(corrected))
        fig_anchor = v11.first_figure_anchor(corrected)
        anchors = [a for a in [title, fig_anchor] if a]
        for block in blocks_by_page.get(target["page_id"], []):
            text = v11.normalize_space(block.get("text") or "")
            if not text or not v11.should_keep_text_fragment(text, anchors):
                continue
            focused = v11.focus_text(text, anchors, args.metadata_near_window)
            key = (target["target_id"], block["block_id"], focused)
            if key in seen:
                continue
            seen.add(key)
            fragments.append(
                make_fragment_v12(
                    target,
                    "same_page_body_pdf_text",
                    focused,
                    focused,
                    focused,
                    [block["block_id"]],
                    [block["bbox"]],
                    "pymupdf_same_page_anchor",
                )
            )
    return fragments


def make_fragment_v12(
    target: dict[str, Any],
    fragment_type: str,
    raw_text: str,
    corrected_text: str,
    display_text: str,
    source_blocks: list[str],
    bboxes: list[list[int]],
    source_quality: str,
) -> dict[str, Any]:
    text_for_hash = display_text or corrected_text or raw_text
    fid = f"frag_{v11.sha1_text(target['target_id'] + '|' + fragment_type + '|' + text_for_hash)[:16]}"
    return {
        "fragment_id": fid,
        "target_id": target["target_id"],
        "doc_id": target["doc_id"],
        "source_file": target["source_file"],
        "page_start": target["page_num"],
        "page_end": target["page_num"],
        "fragment_type": fragment_type,
        "raw_text": raw_text,
        "corrected_text": corrected_text,
        "display_text": display_text,
        "text": display_text,
        "source_blocks": source_blocks,
        "bboxes_by_page": {str(target["page_num"]): bboxes},
        "source_image": target["page_image"],
        "source_bbox_norm1000": target.get("caption_bbox_norm1000") if "caption" in fragment_type else None,
        "source_model": target.get("review_model") if "visual_corrected" in fragment_type else "",
        "source_quality": source_quality,
        "confidence": target.get("confidence") if "visual_corrected" in fragment_type else None,
        "is_generated_summary": False,
        "is_model_internal_knowledge": False,
    }


def build_field_values_and_support_labels_v12(
    targets: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fragments_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frag in fragments:
        fragments_by_target[frag["target_id"]].append(frag)
    field_values: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for target in targets:
        caption_frag = first_fragment(fragments_by_target[target["target_id"]], "local_caption_visual_corrected")
        base_payload = {
            "target_id": target["target_id"],
            "acceptance_status": target["acceptance_status"],
            "is_gold_candidate": target["acceptance_status"] == "accepted",
        }
        for field in BASE_FIELDS:
            value = base_field_value(target, field)
            field_values.append(
                {
                    **base_payload,
                    "field": field,
                    "value": value,
                    "normalized_value": value,
                    "abstain": not bool(value),
                    "evidence_ids": [caption_frag["fragment_id"]] if caption_frag and value else [],
                    "source": "caption_visual_corrected" if field != "object_type" else "target_crop_visual",
                    "confidence": target.get("confidence", 0.0),
                    "reason": "BaseLocate4 由远端 VLM 页面读图/读字得到。",
                }
            )
            if caption_frag and value:
                labels.append(make_label(target, field, value, caption_frag, "support", "qwen3.7_page_read", target.get("confidence", 0.0), "页面视觉和 corrected caption 支持该基础定位字段。"))
        metadata = target.get("metadata_fields") or {}
        for field in META_FIELDS:
            entry = metadata.get(field) or empty_field_entry()
            value = v11.normalize_space(entry.get("value") or "")
            abstain = bool(entry.get("abstain")) or not value
            frag = choose_fragment_for_field(fragments_by_target[target["target_id"]], value, entry.get("source"))
            field_values.append(
                {
                    **base_payload,
                    "field": field,
                    "value": value if not abstain else "",
                    "normalized_value": entry.get("normalized_value") or (v11.normalize_field_value(field, value) if value else ""),
                    "abstain": abstain,
                    "evidence_ids": [frag["fragment_id"]] if frag and not abstain else [],
                    "source": entry.get("source") or "unsupported",
                    "confidence": entry.get("confidence", 0.0),
                    "reason": entry.get("reason") or ("页面没有可见证据支持该 metadata 字段。" if abstain else "远端 VLM 页面读字给出该字段。"),
                    "extraction_source": entry.get("extraction_source"),
                }
            )
            if frag and not abstain:
                labels.append(
                    make_label(
                        target,
                        field,
                        value,
                        frag,
                        "support",
                        entry.get("extraction_source") or "remote_vlm_page_read",
                        entry.get("confidence", target.get("confidence", 0.0)),
                        entry.get("reason") or "远端 VLM 判断该 fragment 支持字段值。",
                    )
                )
    return field_values, labels


def base_field_value(target: dict[str, Any], field: str) -> str:
    if field == "caption_text":
        return v11.normalize_space(target.get("corrected_caption_text") or target.get("caption_text") or "")
    return v11.normalize_space(target.get(field) or "")


def first_fragment(fragments: list[dict[str, Any]], fragment_type: str) -> dict[str, Any] | None:
    for frag in fragments:
        if frag.get("fragment_type") == fragment_type:
            return frag
    return None


def choose_fragment_for_field(fragments: list[dict[str, Any]], value: str, source: Any) -> dict[str, Any] | None:
    source_text = v11.normalize_space(source)
    if "caption" in source_text:
        frag = first_fragment(fragments, "local_caption_visual_corrected")
        if frag:
            return frag
    if value:
        for frag in fragments:
            if value in v11.normalize_space(frag.get("display_text") or frag.get("text") or ""):
                return frag
    return first_fragment(fragments, "local_caption_visual_corrected")


def make_label(
    target: dict[str, Any],
    field: str,
    value: str,
    fragment: dict[str, Any],
    label: str,
    judge_source: str,
    confidence: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "target_id": target["target_id"],
        "field": field,
        "candidate_value": value,
        "normalized_value": v11.normalize_field_value(field, value),
        "fragment_id": fragment["fragment_id"],
        "label": label,
        "judge_source": judge_source,
        "judge_confidence": max(0.0, min(1.0, v11.safe_float(confidence, 0.0))),
        "reason": reason,
        "acceptance_status": target.get("acceptance_status"),
        "is_gold_candidate": target.get("acceptance_status") == "accepted",
    }


def write_page_overlays(output_dir: Path, page_records: list[dict[str, Any]], targets: list[dict[str, Any]]) -> None:
    targets_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        targets_by_page[target["page_id"]].append(target)
    for page in page_records:
        src = Path(page["page_image"])
        if not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        for idx, target in enumerate(targets_by_page.get(page["page_id"], []), start=1):
            tb = target["image_bbox"]
            cb = target["caption_bbox"]
            draw.rectangle(tb, outline=(220, 0, 0), width=4)
            draw.text((tb[0] + 4, max(0, tb[1] - 16)), f"T{idx}", fill=(220, 0, 0), font=font)
            draw.rectangle(cb, outline=(0, 180, 200), width=4)
            draw.text((cb[0] + 4, max(0, cb[1] - 16)), f"C{idx}", fill=(0, 130, 150), font=font)
        out = output_dir / "page_overlays" / f"{page['page_id']}.jpg"
        img.save(out, quality=92)
        page["page_overlay_image"] = str(out)


def write_review_package_v12(
    output_dir: Path,
    page_records: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    field_values: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    del labels
    package_dir = output_dir / "review" / f"v1_2_probe_review_{datetime.now().strftime('%Y%m%d_%H%M')}"
    package_dir.mkdir(parents=True, exist_ok=True)
    targets_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        targets_by_page[target["page_id"]].append(target)
    fields_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fv in field_values:
        fields_by_target[fv["target_id"]].append(fv)
    fragments_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frag in fragments:
        fragments_by_target[frag["target_id"]].append(frag)
    selected_targets = targets[: args.review_limit]
    selected_target_ids = {t["target_id"] for t in selected_targets}
    lines = [
        "# v1.2 Remote-VLM-First Probe 人工抽检包",
        "",
        f"- pages：{len(page_records)}",
        f"- targets：{len(targets)}",
        f"- 展示 targets：{len(selected_targets)}",
        "",
        "说明：红框为目标图，青框为对应图注。训练/评测默认使用 `corrected_caption_text`；`raw_caption_text` 只用于审计对照。",
        "",
    ]
    for page in page_records:
        page_targets = [t for t in targets_by_page.get(page["page_id"], []) if t["target_id"] in selected_target_ids]
        overlay_src = Path(page.get("page_overlay_image") or "")
        if overlay_src.exists():
            overlay_dst = package_dir / overlay_src.name
            shutil.copy2(overlay_src, overlay_dst)
            overlay_name = overlay_dst.name
        else:
            overlay_name = ""
        lines.extend(
            [
                f"## 页面 {page['page_id']} `{html.escape(page.get('source_file', ''))}` p{page.get('page_num')}",
                "",
                f"- category：`{page.get('category')}`",
                f"- page_score：`{page.get('page_score')}`",
                f"- selected target count：{len(page_targets)}",
                "",
            ]
        )
        if overlay_name:
            lines.extend([f"![page overlay]({overlay_name})", ""])
        if not page_targets:
            lines.extend(["该页未产生可展示 target。", ""])
            continue
        for idx, target in enumerate(page_targets, start=1):
            overlay_src = Path(target["overlay_image"])
            crop_src = Path(target["target_crop"])
            overlay_dst = package_dir / overlay_src.name
            crop_dst = package_dir / crop_src.name
            if overlay_src.exists():
                shutil.copy2(overlay_src, overlay_dst)
            if crop_src.exists():
                shutil.copy2(crop_src, crop_dst)
            lines.extend(
                [
                    f"### T{idx} {target['target_id']} {html.escape(target.get('depicted_work_title') or '(无题名)')}",
                    "",
                    f"- acceptance_status：`{target.get('acceptance_status')}`",
                    f"- acceptance_reason：{html.escape(target.get('acceptance_reason') or '')}",
                    f"- confidence：`{target.get('confidence')}`",
                    f"- object_domain：`{target.get('object_domain')}`；object_type：`{target.get('object_type')}`；image_scope：`{target.get('image_scope')}`",
                    f"- VLM reason：{html.escape(target.get('reason') or '')}",
                    "",
                    f"![target overlay]({overlay_dst.name})",
                    "",
                    f"![target crop]({crop_dst.name})",
                    "",
                    "#### caption 对照",
                    "",
                    f"- corrected_caption_text：{html.escape(target.get('corrected_caption_text') or '')}",
                    f"- raw_caption_text：{html.escape(target.get('raw_caption_text') or '')}",
                    "",
                    "#### 字段",
                    "",
                    "| field | value | abstain | source | confidence | evidence_ids | reason |",
                    "|---|---|---:|---|---:|---|---|",
                ]
            )
            for fv in fields_by_target.get(target["target_id"], []):
                ev = ",".join(fv.get("evidence_ids") or [])
                lines.append(
                    f"| `{fv['field']}` | {html.escape(str(fv.get('value') or ''))} | `{fv.get('abstain')}` | `{html.escape(str(fv.get('source') or ''))}` | `{fv.get('confidence')}` | `{html.escape(ev)}` | {html.escape(str(fv.get('reason') or ''))} |"
                )
            lines.extend(["", "#### fragments", "", "| fragment_type | display_text | raw_text |", "|---|---|---|"])
            for frag in fragments_by_target.get(target["target_id"], [])[:8]:
                lines.append(
                    f"| `{frag.get('fragment_type')}` | {html.escape(v11.truncate(frag.get('display_text') or '', 220))} | {html.escape(v11.truncate(frag.get('raw_text') or '', 220))} |"
                )
            lines.append("")
    path = package_dir / f"v1.2RemoteVLMFirstProbe人工抽检包_{len(selected_targets)}targets.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_summary_v12(
    output_dir: Path,
    args: argparse.Namespace,
    pages: list[v11.PageSpec],
    page_records: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    layout_blocks: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    field_values: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    review_package: Path,
) -> dict[str, Any]:
    non_abstain = [fv for fv in field_values if not fv.get("abstain")]
    return {
        "dataset_version": "v1.2-probe",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir),
        "args": vars(args),
        "pipeline": "Remote VLM reads full page; corrected_text/display_text is primary; raw PDF text is audit provenance.",
        "paddleocr_vl_status": {
            "installed": True,
            "batch_used": False,
            "reason": "v1.2 主链路采用远端 page-level VLM；PaddleOCR-VL 只作为扫描/低清后备。",
        },
        "counts": {
            "selected_pages": len(pages),
            "page_records": len(page_records),
            "text_blocks": len(text_blocks),
            "image_layout_blocks": len(layout_blocks),
            "vlm_reviewed_pages": len(vlm_rows),
            "raw_detections": sum(len(r.get("detections") or []) for r in vlm_rows),
            "figure_targets": len(targets),
            "accepted_targets": sum(1 for t in targets if t.get("acceptance_status") == "accepted"),
            "needs_review_targets": sum(1 for t in targets if t.get("acceptance_status") == "needs_review"),
            "rejected_targets": sum(1 for t in targets if t.get("acceptance_status") == "rejected"),
            "evidence_fragments": len(fragments),
            "field_values": len(field_values),
            "non_abstain_field_values": len(non_abstain),
            "field_support_labels": len(labels),
        },
        "distributions": {
            "page_categories": dict(Counter(p.category for p in pages)),
            "vlm_models": dict(Counter(r.get("review_model") for r in vlm_rows)),
            "acceptance_status": dict(Counter(t.get("acceptance_status") for t in targets)),
            "object_domain": dict(Counter(t.get("object_domain") for t in targets)),
            "image_scope": dict(Counter(t.get("image_scope") for t in targets)),
            "fragment_type": dict(Counter(f.get("fragment_type") for f in fragments)),
            "field_non_abstain": dict(Counter(fv.get("field") for fv in non_abstain)),
            "support_label": dict(Counter(l.get("label") for l in labels)),
        },
        "artifacts": {
            "page_manifest": str(output_dir / "page_manifest.jsonl"),
            "page_records": str(output_dir / "page_records.jsonl"),
            "page_text_blocks": str(output_dir / "page_text_blocks.jsonl"),
            "page_layout_blocks": str(output_dir / "page_layout_blocks.jsonl"),
            "page_level_vlm_annotations": str(output_dir / "page_level_vlm_annotations.jsonl"),
            "figure_targets": str(output_dir / "figure_targets.jsonl"),
            "evidence_fragments": str(output_dir / "evidence_fragments.jsonl"),
            "field_values": str(output_dir / "field_values.jsonl"),
            "field_support_labels": str(output_dir / "field_support_labels_vlm_first.jsonl"),
            "review_package": str(review_package),
        },
    }


def write_report_v12(path: Path, summary: dict[str, Any]) -> Path:
    counts = summary["counts"]
    lines = [
        "# v1.2 Remote-VLM-First Probe 构建报告",
        "",
        f"生成时间：{summary['created_at']}",
        "",
        "## 1. 本次构建范围",
        "",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- selected pages：{counts['selected_pages']}",
        f"- VLM reviewed pages：{counts['vlm_reviewed_pages']}",
        f"- raw detections：{counts['raw_detections']}",
        f"- figure targets：{counts['figure_targets']}",
        f"- accepted / needs_review / rejected：{counts['accepted_targets']} / {counts['needs_review_targets']} / {counts['rejected_targets']}",
        f"- evidence fragments：{counts['evidence_fragments']}",
        f"- field values：{counts['field_values']}，其中非 abstain：{counts['non_abstain_field_values']}",
        "",
        "## 2. v1.2 主链路",
        "",
        "- 远端 VLM 直接看整页，输出 target bbox、caption bbox、corrected caption、BaseLocate4 和 Metadata5。",
        "- `corrected_text/display_text` 是训练和评测默认文本。",
        "- `raw_text` 只保留审计，不把 PDF text layer 的错误强行传给训练集。",
        "- PaddleOCR-VL 本次未参与主链路，只作为后备方案。",
        "",
        "## 3. 分布",
        "",
        f"- page categories：`{json.dumps(summary['distributions']['page_categories'], ensure_ascii=False)}`",
        f"- VLM models：`{json.dumps(summary['distributions']['vlm_models'], ensure_ascii=False)}`",
        f"- acceptance_status：`{json.dumps(summary['distributions']['acceptance_status'], ensure_ascii=False)}`",
        f"- object_domain：`{json.dumps(summary['distributions']['object_domain'], ensure_ascii=False)}`",
        f"- image_scope：`{json.dumps(summary['distributions']['image_scope'], ensure_ascii=False)}`",
        f"- fragment_type：`{json.dumps(summary['distributions']['fragment_type'], ensure_ascii=False)}`",
        f"- field_non_abstain：`{json.dumps(summary['distributions']['field_non_abstain'], ensure_ascii=False)}`",
        "",
        "## 4. 关键产物",
        "",
    ]
    for key, value in summary["artifacts"].items():
        lines.append(f"- `{key}`：`{value}`")
    lines.extend(
        [
            "",
            "## 5. 人工抽检重点",
            "",
            "1. 红框是否只框目标图像。",
            "2. 青框是否覆盖完整对应图注。",
            "3. corrected caption 是否比 raw PDF text 更正确，且没有凭空补全。",
            "4. Metadata5 是否来自页面可见证据，是否串到相邻图。",
            "5. acceptance_status 为 accepted 的样本是否真的可以自动接受。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def page_spec_to_record(spec: v11.PageSpec) -> dict[str, Any]:
    row = v11.page_spec_to_record(spec)
    row["dataset_version"] = "v1.2-probe"
    return row


if __name__ == "__main__":
    raise SystemExit(main())
