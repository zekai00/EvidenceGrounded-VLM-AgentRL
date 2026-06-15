#!/usr/bin/env python3
"""Build a v1.1 clean evidence fragment probe from raw PDFs.

This is intentionally a probe builder: it favors traceable intermediate
artifacts and human review packages over aggressive automatic acceptance.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_gold_eval_v1_0_4 as gold_review  # noqa: E402
import build_v1_0_5_core4_visual_audited_sft as v105  # noqa: E402


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

FIGURE_ANCHOR_RE = re.compile(
    r"(图|圖)\s*[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：][一二三四五六七八九十百〇零0-9]+)*[a-zA-Z]?|"
    r"(Fig\.?|Figure|Plate)\s*[A-Za-z]?[0-9IVXivx]+(?:[.\-．:：][0-9IVXivx]+)*[a-zA-Z]?",
    re.I,
)
TITLE_RE = re.compile(r"《([^》]{2,40})》")
LANDSCAPE_RE = re.compile(
    r"山水|山|水|溪|泉|林|松|峰|壑|谷|江|河|岸|岩|石|云|雲|秋|春|溪山|林泉|"
    r"landscape|mountain|river|stream|valley|woods|grove|pine|retreat|hermitage",
    re.I,
)
NON_TASK_PAGE_RE = re.compile(r"目录|目錄|参考文献|參考文獻|致谢|摘要|关键词|contents|bibliography|references", re.I)
DIMENSION_RE = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:×|x|X|\*)\s*\d+(?:\.\d+)?(?:\s*(?:×|x|X|\*)\s*\d+(?:\.\d+)?)?\s*(?:厘米|公分|cm|CM))"
)
DIMENSION_LH_RE = re.compile(
    r"(长|縱|纵|高)\s*(\d+(?:\.\d+)?)\s*(?:厘米|公分|cm|CM)[，,、;； ]{0,8}(横|宽|寬)\s*(\d+(?:\.\d+)?)\s*(?:厘米|公分|cm|CM)"
)
DYNASTY_RE = re.compile(
    r"(北宋|南宋|宋代|宋|元代|元|明代|明|清代|清|唐代|唐|五代|辽|遼|金代|民国|民國|"
    r"Northern Song|Southern Song|Song dynasty|Yuan dynasty|Ming dynasty|Qing dynasty|Tang dynasty|"
    r"[0-9]{1,2}(?:st|nd|rd|th)?[-–—]?[0-9]{0,2}(?:st|nd|rd|th)? century|ca\.\s*[0-9]{3,4}(?:[-–—][0-9]{2,4})?)",
    re.I,
)
INSTITUTION_RES = [
    re.compile(r"((?:北京|台北|臺北|南京|上海|辽宁|遼寧)?故宫博物院)"),
    re.compile(r"([\u4e00-\u9fff]{2,20}(?:博物院|博物馆|美术馆|藝術館|艺术馆))"),
    re.compile(r"(The Metropolitan Museum of Art|Metropolitan Museum of Art)", re.I),
    re.compile(r"(National Palace Museum(?:,?\s*Taipei)?)", re.I),
    re.compile(r"(Palace Museum(?:,?\s*Beijing)?)", re.I),
    re.compile(r"(Museum of Fine Arts,?\s*Boston|Freer Gallery of Art|Cleveland Museum of Art)", re.I),
]
MEDIUM_RES = [
    (re.compile(r"纸本|紙本"), "纸本"),
    (re.compile(r"绢本|絹本"), "绢本"),
    (re.compile(r"设色|設色"), "设色"),
    (re.compile(r"水墨"), "水墨"),
    (re.compile(r"ink and color on silk", re.I), "ink and color on silk"),
    (re.compile(r"ink and color on paper", re.I), "ink and color on paper"),
    (re.compile(r"ink on silk", re.I), "ink on silk"),
    (re.compile(r"ink on paper", re.I), "ink on paper"),
]
CREATOR_EN_RE = re.compile(r"(?:Attributed to|After|By)\s+([A-Z][A-Za-z'üÜ\-.]+(?:\s+[A-Z][A-Za-z'üÜ\-.]+){0,4})")
CREATOR_CN_RE = re.compile(r"(?:明|清|元|宋|北宋|南宋|唐|五代)?[·．.\s]*([\u4e00-\u9fff]{2,4})《")
BAD_CREATOR_VALUES = {"代表作", "现藏于", "所藏赵", "藏赵孟", "此图现", "后期所"}

PAGE_LEVEL_PROMPT = """你是 EvidenceGrounded-VLM-AgentRL v1.1 的 page-level 数据构建员。

你会看到一整页 PDF 渲染图。请直接从页面中找出适合构建数据集的“中国/东亚古典山水画或山水画局部”目标，并输出目标图像框、对应图注框、完整可见图注文本和 Core4 字段。

坐标要求：
- 坐标必须是 0-1000 归一化坐标；
- 原点左上，右下约为 [1000,1000]；
- bbox 格式为 [x1,y1,x2,y2]；
- target_bbox_norm1000 只框目标图像，不要包含图注、正文、相邻图；
- caption_bbox_norm1000 只框对应图注，不要包含正文或其他图注。

接受对象：
- 中国/东亚古典山水画、山水画局部、以山水/自然景观为主体的古典绘画。

排除对象：
- 纯正文、目录、表格、图式、建筑/器物照片；
- 书法题跋、人物故事、佛教叙事、动物/骑乘/乐人等非山水主体；
- 图像与图注无法明确对应的对象。

只输出 JSON，不要输出 Markdown。schema:
{
  "page_summary": "一句话说明",
  "detections": [
    {
      "target_bbox_norm1000": [0,0,0,0],
      "caption_bbox_norm1000": [0,0,0,0],
      "caption_text": "页面可见完整图注",
      "depicted_work_title": "作品题名，无法确定则空串",
      "image_scope": "full_work|partial_detail|multi_work_comparison|unclear",
      "object_type": "painting|painting_detail|diagram|text_page|other|unclear",
      "object_domain": "landscape_painting|landscape_detail|classical_painting_unclear_landscape|non_landscape_artwork|text_only|other|unclear",
      "caption_target_match": "yes|no|uncertain",
      "accept_for_probe": true,
      "needs_human_review": false,
      "confidence": 0.0,
      "reason": "一句话说明"
    }
  ]
}

页面元数据：
"""


@dataclass
class PageSpec:
    doc_id: str
    source_file: str
    source_path: Path
    rel_path: str
    page_num: int
    page_count: int
    category: str
    score: float
    image_count: int
    figure_anchor_count: int
    landscape_term_count: int
    text_preview: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1.1 clean evidence fragment probe.")
    parser.add_argument("--raw-pdf-root", default=str(RAW_PDF_ROOT))
    parser.add_argument("--sources-jsonl", default=str(SOURCE_JSONL))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--docs-dir", default=str(DOCS_DIR))
    parser.add_argument("--dotenv", default=str(DOTENV))
    parser.add_argument("--probe-pages", type=int, default=60)
    parser.add_argument("--max-pdfs", type=int, default=80)
    parser.add_argument("--max-pages-per-pdf", type=int, default=8)
    parser.add_argument("--page-dpi", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--vlm-provider", choices=["dashscope", "local", "offline"], default="dashscope")
    parser.add_argument("--dashscope-model", default="qwen3.7-max")
    parser.add_argument(
        "--dashscope-fallback-models",
        default="qwen3.7-max-2026-06-08,qwen3.7-plus-2026-05-26,qwen3.7-plus,qwen3.6-plus,qwen3.6-27b,glm-5.1,kimi-k2.6,deepseek-v4-pro,deepseek-v4-flash",
    )
    parser.add_argument("--local-model", default="/root/models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--local-device", default="cuda:0")
    parser.add_argument("--image-max-side", type=int, default=1600)
    parser.add_argument("--max-new-tokens", type=int, default=1400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--min-detection-confidence", type=float, default=0.45)
    parser.add_argument("--metadata-near-window", type=int, default=420)
    parser.add_argument("--review-limit", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_review.load_dotenv(Path(args.dotenv))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"v1_1_clean_evidence_fragment_probe_{stamp}"
    prepare_output_dir(output_dir, args)

    source_registry = load_source_registry(Path(args.sources_jsonl))
    pages = select_probe_pages(Path(args.raw_pdf_root), source_registry, args)
    write_jsonl(output_dir / "page_manifest.jsonl", [page_spec_to_record(p) for p in pages])

    page_records, text_blocks, layout_blocks = materialize_pages_and_blocks(output_dir, pages, args)
    write_jsonl(output_dir / "page_records.jsonl", page_records)
    write_jsonl(output_dir / "page_text_blocks.jsonl", text_blocks)
    write_jsonl(output_dir / "page_layout_blocks.jsonl", layout_blocks)

    client = make_vlm_client(args)
    vlm_rows = run_page_level_vlm(output_dir, page_records, client, args)
    write_jsonl(output_dir / "page_level_vlm_targets_raw.jsonl", vlm_rows)

    targets = build_figure_targets(output_dir, page_records, vlm_rows, args)
    write_jsonl(output_dir / "figure_targets.jsonl", targets)

    fragments = build_evidence_fragments(targets, text_blocks, args)
    write_jsonl(output_dir / "evidence_fragments.jsonl", fragments)

    candidates, support_labels = build_metadata_candidates(targets, fragments)
    write_jsonl(output_dir / "metadata_candidate_fragments.jsonl", candidates)
    write_jsonl(output_dir / "field_support_labels_rule.jsonl", support_labels)

    review_package = write_review_package(output_dir, page_records, targets, fragments, support_labels, args)
    summary = build_summary(output_dir, args, pages, page_records, text_blocks, layout_blocks, vlm_rows, targets, fragments, candidates, support_labels, review_package)
    write_json(output_dir / "manifest.json", summary)
    report = write_report(output_dir / "构建报告.md", summary)
    docs_path = Path(args.docs_dir) / f"{stamp}_v1.1CleanEvidenceFragmentProbe构建报告.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")
    summary["artifacts"]["docs_report"] = str(docs_path)
    write_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def prepare_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists; use --resume or --overwrite")
    for child in ["pages", "overlays", "crops", "review", "cache"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def load_source_registry(path: Path) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return registry
    for row in read_jsonl(path):
        keys = [
            row.get("source_file"),
            row.get("filename"),
            row.get("file_name"),
            row.get("pdf_name"),
            row.get("path"),
        ]
        for key in keys:
            if key:
                registry[str(key)] = row
                registry[Path(str(key)).name] = row
    return registry


def select_probe_pages(raw_root: Path, registry: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[PageSpec]:
    rng = random.Random(args.seed)
    pdfs = sorted(raw_root.rglob("*.pdf"))
    if args.max_pdfs > 0:
        # Prefer coverage across folder categories, then deterministic randomization.
        grouped: dict[str, list[Path]] = defaultdict(list)
        for pdf in pdfs:
            grouped[pdf.relative_to(raw_root).parts[0] if len(pdf.relative_to(raw_root).parts) > 1 else "root"].append(pdf)
        selected: list[Path] = []
        for _, group in sorted(grouped.items()):
            rng.shuffle(group)
            selected.extend(group[: max(1, args.max_pdfs // max(1, len(grouped)))])
        if len(selected) < args.max_pdfs:
            remaining = [p for p in pdfs if p not in selected]
            rng.shuffle(remaining)
            selected.extend(remaining[: args.max_pdfs - len(selected)])
        pdfs = selected[: args.max_pdfs]

    candidates: list[PageSpec] = []
    for pdf in pdfs:
        try:
            rel = str(pdf.relative_to(raw_root))
            category = Path(rel).parts[0] if len(Path(rel).parts) > 1 else "root"
            with fitz.open(pdf) as doc:
                page_count = len(doc)
                page_indices = candidate_page_indices(page_count, args.max_pages_per_pdf)
                for page_index in page_indices:
                    page = doc[page_index]
                    text = normalize_space(page.get_text("text") or "")
                    if not text and page_count > 80:
                        # Large scanned catalog pages can be slow; still score image pages below.
                        text = ""
                    image_count = len(page.get_images(full=False))
                    figure_count = len(FIGURE_ANCHOR_RE.findall(text))
                    landscape_count = len(LANDSCAPE_RE.findall(text[:3000]))
                    noise_penalty = 6 if NON_TASK_PAGE_RE.search(text[:800]) else 0
                    score = image_count * 4.0 + min(figure_count, 8) * 2.5 + min(landscape_count, 10) * 0.6 - noise_penalty
                    if image_count == 0 and figure_count == 0:
                        continue
                    if score <= 0:
                        continue
                    candidates.append(
                        PageSpec(
                            doc_id=doc_id_for_path(pdf),
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

    # Balanced by category first, then fill by score.
    by_cat: dict[str, list[PageSpec]] = defaultdict(list)
    for item in candidates:
        by_cat[item.category].append(item)
    for group in by_cat.values():
        group.sort(key=lambda p: p.score, reverse=True)
    selected: list[PageSpec] = []
    cat_names = sorted(by_cat)
    per_cat = max(1, args.probe_pages // max(1, len(cat_names)))
    for cat in cat_names:
        for item in by_cat[cat][:per_cat]:
            selected.append(item)
            if len(selected) >= args.probe_pages:
                break
        if len(selected) >= args.probe_pages:
            break
    if len(selected) < args.probe_pages:
        remaining = sorted([p for p in candidates if p not in selected], key=lambda p: p.score, reverse=True)
        selected.extend(remaining[: args.probe_pages - len(selected)])
    selected = selected[: args.probe_pages]
    selected.sort(key=lambda p: (p.category, p.source_file, p.page_num))
    return selected


def candidate_page_indices(page_count: int, max_pages: int) -> list[int]:
    if page_count <= max_pages:
        return list(range(page_count))
    keep: set[int] = set()
    # Cover front/middle/back without scanning the full PDF.
    anchors = [0, 1, 2, page_count // 4, page_count // 2, (page_count * 3) // 4, page_count - 3, page_count - 2, page_count - 1]
    for idx in anchors:
        if 0 <= idx < page_count:
            keep.add(idx)
    idx = 3
    while len(keep) < max_pages and idx < page_count:
        keep.add(idx)
        idx += max(1, page_count // max_pages)
    return sorted(keep)[:max_pages]


def materialize_pages_and_blocks(output_dir: Path, pages: list[PageSpec], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    page_records: list[dict[str, Any]] = []
    text_blocks: list[dict[str, Any]] = []
    layout_blocks: list[dict[str, Any]] = []
    for idx, spec in enumerate(pages, start=1):
        page_id = f"v11p_{idx:04d}"
        page_path = output_dir / "pages" / f"{safe_stem(spec.source_path.stem)}_p{spec.page_num:04d}.png"
        try:
            with fitz.open(spec.source_path) as doc:
                page = doc[spec.page_num - 1]
                pix = page.get_pixmap(dpi=args.page_dpi, colorspace=fitz.csRGB)
                pix.save(page_path)
                width, height = pix.width, pix.height
                sx = width / page.rect.width
                sy = height / page.rect.height
                pdf_text = normalize_space(page.get_text("text") or "")
                blocks = page.get_text("blocks") or []
                for bidx, block in enumerate(blocks):
                    if len(block) < 5:
                        continue
                    x0, y0, x1, y1, text = block[:5]
                    text = normalize_space(text)
                    if not text:
                        continue
                    text_blocks.append(
                        {
                            "block_id": f"{page_id}_tb{bidx:04d}",
                            "page_id": page_id,
                            "doc_id": spec.doc_id,
                            "source_file": spec.source_file,
                            "page_num": spec.page_num,
                            "block_type": "text",
                            "bbox": scale_bbox([x0, y0, x1, y1], sx, sy),
                            "bbox_norm1000": norm_bbox(scale_bbox([x0, y0, x1, y1], sx, sy), width, height),
                            "text": text,
                            "source": "pymupdf_text_block",
                        }
                    )
                for iidx, img in enumerate(page.get_image_info(xrefs=True) or []):
                    bbox = img.get("bbox")
                    if not bbox:
                        continue
                    bbox_px = scale_bbox(list(bbox), sx, sy)
                    layout_blocks.append(
                        {
                            "block_id": f"{page_id}_ib{iidx:04d}",
                            "page_id": page_id,
                            "doc_id": spec.doc_id,
                            "source_file": spec.source_file,
                            "page_num": spec.page_num,
                            "block_type": "image",
                            "bbox": bbox_px,
                            "bbox_norm1000": norm_bbox(bbox_px, width, height),
                            "text": "",
                            "source": "pymupdf_image_block",
                            "xref": img.get("xref"),
                            "width": img.get("width"),
                            "height": img.get("height"),
                        }
                    )
        except Exception as exc:
            width = height = 0
            pdf_text = ""
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
                "pdf_text": pdf_text,
                "page_score": spec.score,
                "selection_stats": {
                    "image_count": spec.image_count,
                    "figure_anchor_count": spec.figure_anchor_count,
                    "landscape_term_count": spec.landscape_term_count,
                },
            }
        )
    return page_records, text_blocks, layout_blocks


def make_vlm_client(args: argparse.Namespace):
    if args.vlm_provider == "offline":
        return OfflineVLMClient(args)
    if args.vlm_provider == "local":
        return LocalQwenPageClient(args)
    return DashScopePageClient(args)


class OfflineVLMClient:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def review_page(self, page: dict[str, Any]) -> dict[str, Any]:
        return {**page_base(page), "ok": True, "review_model": "offline", "page_summary": "offline mode", "detections": [], "raw_response": ""}


class DashScopePageClient:
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.models = [args.dashscope_model] + [m.strip() for m in args.dashscope_fallback_models.split(",") if m.strip()]
        self.models = dedupe(self.models)
        self.args = args

    def review_page(self, page: dict[str, Any]) -> dict[str, Any]:
        prompt = build_page_prompt(page)
        image_url = image_data_url(Path(page["page_image"]), self.args.image_max_side)
        last_error: Exception | None = None
        for model in self.models:
            image_modes = ["image_url", "image"] if "qwen3.7" in model.lower() else v105.image_modes_for_model(model)
            for image_mode in image_modes:
                try:
                    if image_mode == "image":
                        content: Any = [{"type": "image", "image": image_url}, {"type": "text", "text": prompt}]
                    elif image_mode == "text_only":
                        content = prompt + "\n注意：当前未输入图片，只能依据页面文本预览判断。"
                    else:
                        content = [{"type": "image_url", "image_url": {"url": image_url}}, {"type": "text", "text": prompt}]
                    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if "qwen3.7" in model.lower() else None
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_new_tokens,
                        response_format={"type": "json_object"},
                        extra_body=extra_body,
                    )
                    raw = response.choices[0].message.content or ""
                    parsed = parse_json_object(raw)
                    return normalize_vlm_response(page, parsed, raw, model, image_mode)
                except Exception as exc:
                    last_error = exc
                    continue
        return {**page_base(page), "ok": False, "review_model": "dashscope_failed", "error": repr(last_error), "page_summary": "", "detections": [], "raw_response": ""}


class LocalQwenPageClient:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.local_model, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.local_model,
            dtype="auto",
            device_map=args.local_device,
            trust_remote_code=True,
        )
        self.model.eval()

    def review_page(self, page: dict[str, Any]) -> dict[str, Any]:
        prompt = build_page_prompt(page)
        messages = [{"role": "user", "content": [{"type": "image", "image": page["page_image"]}, {"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        kwargs = {"max_new_tokens": self.args.max_new_tokens}
        if self.args.temperature > 0:
            kwargs.update({"do_sample": True, "temperature": self.args.temperature})
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, **kwargs)
        trimmed = [o[len(i) :] for i, o in zip(inputs.input_ids, out)]
        raw = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            parsed = parse_json_object(raw)
            return normalize_vlm_response(page, parsed, raw, self.args.local_model, "local_image")
        except Exception as exc:
            return {**page_base(page), "ok": False, "review_model": self.args.local_model, "error": repr(exc), "page_summary": "", "detections": [], "raw_response": raw}


def run_page_level_vlm(output_dir: Path, page_records: list[dict[str, Any]], client: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    stream = output_dir / "cache" / "page_level_vlm_stream.jsonl"
    existing = {row.get("page_id"): row for row in read_jsonl(stream)} if stream.exists() else {}
    rows: list[dict[str, Any]] = []
    for idx, page in enumerate(page_records, start=1):
        row = existing.get(page["page_id"])
        if row is None or not args.resume:
            row = client.review_page(page)
            append_jsonl(stream, [row])
            if args.sleep:
                time.sleep(args.sleep)
        rows.append(row)
        print(json.dumps({"stage": "page_vlm", "progress": f"{idx}/{len(page_records)}", "page_id": page["page_id"], "detections": len(row.get("detections") or []), "model": row.get("review_model")}, ensure_ascii=False), flush=True)
    return rows


def build_figure_targets(output_dir: Path, page_records: list[dict[str, Any]], vlm_rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    page_by_id = {p["page_id"]: p for p in page_records}
    targets: list[dict[str, Any]] = []
    for row in vlm_rows:
        page = page_by_id.get(row.get("page_id"))
        if not page:
            continue
        for det in row.get("detections") or []:
            conf = safe_float(det.get("confidence"), 0.0)
            if conf < args.min_detection_confidence:
                continue
            target_bbox = det.get("target_bbox")
            caption_bbox = det.get("caption_bbox")
            if not valid_bbox(target_bbox) or not valid_bbox(caption_bbox):
                continue
            caption = normalize_space(det.get("caption_text") or "")
            title = normalize_space(det.get("depicted_work_title") or extract_title(caption))
            target_id = f"v11_{page['page_id']}_t{len(targets):04d}"
            crop_path = output_dir / "crops" / f"{target_id}.jpg"
            overlay_path = output_dir / "overlays" / f"{target_id}.jpg"
            crop_image(Path(page["page_image"]), target_bbox, crop_path)
            draw_target_overlay(Path(page["page_image"]), target_bbox, caption_bbox, overlay_path)
            target = {
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
                "image_bbox_norm1000": norm_bbox(target_bbox, page["width"], page["height"]),
                "caption_bbox_norm1000": norm_bbox(caption_bbox, page["width"], page["height"]),
                "caption_text": caption,
                "depicted_work_title": title,
                "image_scope": normalize_enum(det.get("image_scope"), {"full_work", "partial_detail", "multi_work_comparison", "unclear"}, "unclear"),
                "object_type": normalize_object_type(det.get("object_type"), det.get("object_domain")),
                "object_domain": det.get("object_domain") or "unclear",
                "caption_target_match": det.get("caption_target_match") or "uncertain",
                "confidence": conf,
                "needs_human_review": bool(det.get("needs_human_review")),
                "reason": det.get("reason") or "",
                "source_stage": "v1.1_page_level_vlm_probe",
            }
            targets.append(target)
    return targets


def build_evidence_fragments(targets: list[dict[str, Any]], text_blocks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    blocks_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in text_blocks:
        blocks_by_page[block["page_id"]].append(block)
    fragments: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for target in targets:
        caption = normalize_space(target.get("caption_text") or "")
        if caption:
            fragments.append(
                make_fragment(
                    target=target,
                    fragment_type="local_caption",
                    text=caption,
                    source_blocks=[],
                    bboxes_by_page={str(target["page_num"]): [target["caption_bbox"]]},
                    source_quality="page_level_vlm_caption",
                )
            )
        title = normalize_space(target.get("depicted_work_title") or extract_title(caption))
        fig_anchor = first_figure_anchor(caption)
        anchors = [a for a in [title, fig_anchor] if a]
        for block in blocks_by_page.get(target["page_id"], []):
            text = normalize_space(block.get("text") or "")
            if not text:
                continue
            if not should_keep_text_fragment(text, anchors):
                continue
            focused = focus_text(text, anchors, args.metadata_near_window)
            key = (target["target_id"], block["block_id"], focused)
            if key in seen:
                continue
            seen.add(key)
            frag_type = "body_anchor_window" if any(anchor and anchor in text for anchor in anchors) else "metadata_sentence_window"
            fragments.append(
                make_fragment(
                    target=target,
                    fragment_type=frag_type,
                    text=focused,
                    source_blocks=[block["block_id"]],
                    bboxes_by_page={str(block["page_num"]): [block["bbox"]]},
                    source_quality="pymupdf_text_block",
                )
            )
    return fragments


def build_metadata_candidates(targets: list[dict[str, Any]], fragments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    target_by_id = {t["target_id"]: t for t in targets}
    candidates: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for frag in fragments:
        target = target_by_id.get(frag["target_id"], {})
        extracted = extract_metadata(frag.get("text") or "")
        for field, value in extracted.items():
            if not value:
                continue
            candidate = {
                "candidate_id": f"cand_{len(candidates):06d}",
                "target_id": frag["target_id"],
                "field": field,
                "candidate_value": value,
                "fragment_id": frag["fragment_id"],
                "fragment_type": frag["fragment_type"],
                "retrieval_method": ["local_caption" if frag["fragment_type"] == "local_caption" else "same_page_anchor"],
                "source_quality": frag["source_quality"],
            }
            candidates.append(candidate)
            label, reason = rule_support_label(target, frag, field, value)
            labels.append(
                {
                    "target_id": frag["target_id"],
                    "field": field,
                    "candidate_value": value,
                    "normalized_value": normalize_field_value(field, value),
                    "fragment_id": frag["fragment_id"],
                    "label": label,
                    "judge_source": "v1.1_rule_prior",
                    "judge_confidence": 0.72 if label == "support" else 0.62,
                    "reason": reason,
                }
            )
    return candidates, labels


def write_review_package(
    output_dir: Path,
    page_records: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    package_dir = output_dir / "review" / f"v1_1_probe_review_{datetime.now().strftime('%Y%m%d_%H%M')}"
    package_dir.mkdir(parents=True, exist_ok=True)
    page_by_id = {p["page_id"]: p for p in page_records}
    labels_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        labels_by_target[label["target_id"]].append(label)
    frag_by_id = {f["fragment_id"]: f for f in fragments}
    selected = targets[: args.review_limit]
    lines = [
        "# v1.1 Clean Evidence Fragment Probe 人工抽检包",
        "",
        f"- targets：{len(targets)}",
        f"- 展示：{len(selected)}",
        "",
    ]
    for idx, target in enumerate(selected, start=1):
        overlay_src = Path(target["overlay_image"])
        crop_src = Path(target["target_crop"])
        overlay_dst = package_dir / overlay_src.name
        crop_dst = package_dir / crop_src.name
        if overlay_src.exists():
            shutil.copy2(overlay_src, overlay_dst)
        if crop_src.exists():
            shutil.copy2(crop_src, crop_dst)
        title = target.get("depicted_work_title") or "(无题名)"
        lines.extend(
            [
                f"## V11P{idx:03d} {target['target_id']} {html.escape(str(title))}",
                "",
                f"- source：`{target.get('source_file')}` p{target.get('page_num')}",
                f"- confidence：`{target.get('confidence')}`，object_domain：`{target.get('object_domain')}`，image_scope：`{target.get('image_scope')}`",
                f"- caption_text：{html.escape(str(target.get('caption_text') or ''))}",
                f"- reason：{html.escape(str(target.get('reason') or ''))}",
                "",
                f"![overlay]({overlay_dst.name})",
                "",
                f"![crop]({crop_dst.name})",
                "",
                "| field | value | label | fragment | reason |",
                "|---|---|---|---|---|",
            ]
        )
        for label in labels_by_target.get(target["target_id"], [])[:12]:
            frag = frag_by_id.get(label["fragment_id"], {})
            frag_text = truncate(frag.get("text") or "", 180)
            lines.append(
                f"| `{label['field']}` | {html.escape(str(label.get('candidate_value') or ''))} | `{label.get('label')}` | {html.escape(frag_text)} | {html.escape(str(label.get('reason') or ''))} |"
            )
        lines.append("")
    path = package_dir / f"v1.1CleanEvidenceFragmentProbe人工抽检包_{len(selected)}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_summary(
    output_dir: Path,
    args: argparse.Namespace,
    pages: list[PageSpec],
    page_records: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    layout_blocks: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    review_package: Path,
) -> dict[str, Any]:
    return {
        "dataset_version": "v1.1-probe",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(output_dir),
        "args": vars(args),
        "gpu_policy": "local VLM uses single specified device only; PaddleOCR-VL batch disabled because current Paddle is CPU-only",
        "paddleocr_vl_status": {
            "model_dir": "/root/models/PaddleOCR-VL-1.6",
            "layout_model_dir": "/root/models/PP-DocLayoutV3",
            "installed": True,
            "batch_used": False,
            "reason": "PaddlePaddle in agent_env is CPU-only; single-page smoke was too slow for probe batch.",
        },
        "counts": {
            "selected_pages": len(pages),
            "page_records": len(page_records),
            "text_blocks": len(text_blocks),
            "image_layout_blocks": len(layout_blocks),
            "vlm_reviewed_pages": len(vlm_rows),
            "raw_detections": sum(len(r.get("detections") or []) for r in vlm_rows),
            "figure_targets": len(targets),
            "evidence_fragments": len(fragments),
            "metadata_candidates": len(candidates),
            "field_support_labels_rule": len(labels),
        },
        "distributions": {
            "page_categories": dict(Counter(p.category for p in pages)),
            "vlm_models": dict(Counter(r.get("review_model") for r in vlm_rows)),
            "object_domain": dict(Counter(t.get("object_domain") for t in targets)),
            "image_scope": dict(Counter(t.get("image_scope") for t in targets)),
            "fragment_type": dict(Counter(f.get("fragment_type") for f in fragments)),
            "support_label": dict(Counter(l.get("label") for l in labels)),
            "field": dict(Counter(l.get("field") for l in labels)),
        },
        "artifacts": {
            "page_manifest": str(output_dir / "page_manifest.jsonl"),
            "page_records": str(output_dir / "page_records.jsonl"),
            "page_text_blocks": str(output_dir / "page_text_blocks.jsonl"),
            "page_layout_blocks": str(output_dir / "page_layout_blocks.jsonl"),
            "page_level_vlm_raw": str(output_dir / "page_level_vlm_targets_raw.jsonl"),
            "figure_targets": str(output_dir / "figure_targets.jsonl"),
            "evidence_fragments": str(output_dir / "evidence_fragments.jsonl"),
            "metadata_candidate_fragments": str(output_dir / "metadata_candidate_fragments.jsonl"),
            "field_support_labels_rule": str(output_dir / "field_support_labels_rule.jsonl"),
            "review_package": str(review_package),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> Path:
    counts = summary["counts"]
    lines = [
        "# v1.1 Clean Evidence Fragment Probe 构建报告",
        "",
        f"生成时间：{summary['created_at']}",
        "",
        "## 1. 本次构建范围",
        "",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- selected pages：{counts['selected_pages']}",
        f"- VLM reviewed pages：{counts['vlm_reviewed_pages']}",
        f"- figure targets：{counts['figure_targets']}",
        f"- evidence fragments：{counts['evidence_fragments']}",
        f"- metadata candidates：{counts['metadata_candidates']}",
        f"- rule support labels：{counts['field_support_labels_rule']}",
        "",
        "## 2. 模型与规则使用",
        "",
        f"- page-level VLM provider：`{summary['args'].get('vlm_provider')}`",
        f"- page-level VLM 模型分布：`{json.dumps(summary['distributions']['vlm_models'], ensure_ascii=False)}`",
        "- PDF 页面渲染与 text/image block：PyMuPDF。",
        "- BaseLocate4：由 page-level VLM 输出，并经过 bbox、confidence、字段枚举规则门控。",
        "- Evidence fragment：由 local caption 与同页 PDF text anchor window 构成。",
        "- Metadata5 support：本次先输出 `v1.1_rule_prior` 标签，用作 verifier 后续二审输入；没有把规则标签直接当最终 gold。",
        "",
        "## 3. PaddleOCR-VL 状态",
        "",
        f"- PaddleOCR-VL 模型：`{summary['paddleocr_vl_status']['model_dir']}`",
        f"- PP-DocLayoutV3：`{summary['paddleocr_vl_status']['layout_model_dir']}`",
        f"- 是否批量使用：`{summary['paddleocr_vl_status']['batch_used']}`",
        f"- 原因：{summary['paddleocr_vl_status']['reason']}",
        "",
        "## 4. 分布",
        "",
        f"- page categories：`{json.dumps(summary['distributions']['page_categories'], ensure_ascii=False)}`",
        f"- object_domain：`{json.dumps(summary['distributions']['object_domain'], ensure_ascii=False)}`",
        f"- image_scope：`{json.dumps(summary['distributions']['image_scope'], ensure_ascii=False)}`",
        f"- fragment_type：`{json.dumps(summary['distributions']['fragment_type'], ensure_ascii=False)}`",
        f"- support_label：`{json.dumps(summary['distributions']['support_label'], ensure_ascii=False)}`",
        f"- field：`{json.dumps(summary['distributions']['field'], ensure_ascii=False)}`",
        "",
        "## 5. 关键产物",
        "",
    ]
    for key, value in summary["artifacts"].items():
        lines.append(f"- `{key}`：`{value}`")
    lines.extend(
        [
            "",
            "## 6. 当前结论",
            "",
            "这版是 v1.1 的第一轮 probe，不是最终训练集。它的价值在于验证从 raw PDF 到 page target、fragment、field-level support 候选的链路是否通顺。",
            "",
            "下一步应人工查看 review package，重点判断：",
            "",
            "1. VLM 的 target bbox 是否只框图像。",
            "2. caption bbox 和 caption_text 是否完整。",
            "3. 多图页是否发生图文错配。",
            "4. rule support label 是否把相邻作品 metadata 错当 support。",
            "",
            "如果 page-level target 质量可接受，再把 field support labels 接入 Qwen3-4B/云端 LLM 二审，形成真正可用于 val/test 的 verifier cache。",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_page_prompt(page: dict[str, Any]) -> str:
    meta = {
        "source_file": page.get("source_file"),
        "page_num": page.get("page_num"),
        "page_size_px": [page.get("width"), page.get("height")],
        "pdf_text_preview": truncate(page.get("pdf_text") or "", 900),
        "selection_stats": page.get("selection_stats"),
    }
    return PAGE_LEVEL_PROMPT + json.dumps(meta, ensure_ascii=False, indent=2)


def normalize_vlm_response(page: dict[str, Any], parsed: dict[str, Any], raw: str, model: str, input_mode: str) -> dict[str, Any]:
    detections: list[dict[str, Any]] = []
    for idx, item in enumerate(parsed.get("detections") if isinstance(parsed.get("detections"), list) else []):
        if not isinstance(item, dict):
            continue
        target_norm = normalize_bbox_value(item.get("target_bbox_norm1000"))
        caption_norm = normalize_bbox_value(item.get("caption_bbox_norm1000"))
        target_bbox = scale_norm1000_bbox(target_norm, page["width"], page["height"]) if target_norm else None
        caption_bbox = scale_norm1000_bbox(caption_norm, page["width"], page["height"]) if caption_norm else None
        detections.append(
            {
                "detection_index": idx,
                "target_bbox_norm1000": target_norm,
                "caption_bbox_norm1000": caption_norm,
                "target_bbox": target_bbox,
                "caption_bbox": caption_bbox,
                "caption_text": normalize_space(item.get("caption_text") or "")[:800],
                "depicted_work_title": normalize_space(item.get("depicted_work_title") or "")[:120],
                "image_scope": normalize_space(item.get("image_scope") or "unclear"),
                "object_type": normalize_space(item.get("object_type") or "unclear"),
                "object_domain": normalize_space(item.get("object_domain") or "unclear"),
                "caption_target_match": normalize_space(item.get("caption_target_match") or "uncertain"),
                "accept_for_probe": bool(item.get("accept_for_probe")),
                "needs_human_review": bool(item.get("needs_human_review")),
                "confidence": max(0.0, min(1.0, safe_float(item.get("confidence"), 0.0))),
                "reason": str(item.get("reason") or "")[:500],
            }
        )
    return {**page_base(page), "ok": True, "review_model": model, "input_mode": input_mode, "page_summary": str(parsed.get("page_summary") or "")[:500], "detections": detections, "raw_response": raw}


def page_base(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_id": page.get("page_id"),
        "doc_id": page.get("doc_id"),
        "source_file": page.get("source_file"),
        "page_num": page.get("page_num"),
        "page_image": page.get("page_image"),
        "width": page.get("width"),
        "height": page.get("height"),
    }


def page_spec_to_record(spec: PageSpec) -> dict[str, Any]:
    return {
        "doc_id": spec.doc_id,
        "source_file": spec.source_file,
        "source_path": str(spec.source_path),
        "rel_path": spec.rel_path,
        "page_num": spec.page_num,
        "page_count": spec.page_count,
        "category": spec.category,
        "score": spec.score,
        "image_count": spec.image_count,
        "figure_anchor_count": spec.figure_anchor_count,
        "landscape_term_count": spec.landscape_term_count,
        "text_preview": spec.text_preview,
    }


def make_fragment(target: dict[str, Any], fragment_type: str, text: str, source_blocks: list[str], bboxes_by_page: dict[str, list[list[int]]], source_quality: str) -> dict[str, Any]:
    text = normalize_space(text)
    fid = f"frag_{sha1_text(target['target_id'] + '|' + fragment_type + '|' + text)[:16]}"
    return {
        "fragment_id": fid,
        "target_id": target["target_id"],
        "doc_id": target["doc_id"],
        "source_file": target["source_file"],
        "page_start": target["page_num"],
        "page_end": target["page_num"],
        "fragment_type": fragment_type,
        "text": text,
        "source_blocks": source_blocks,
        "bboxes_by_page": bboxes_by_page,
        "source_quality": source_quality,
        "is_generated_summary": False,
    }


def should_keep_text_fragment(text: str, anchors: list[str]) -> bool:
    if len(text) < 8:
        return False
    if any(anchor and anchor in text for anchor in anchors):
        return True
    if FIGURE_ANCHOR_RE.search(text) and (DIMENSION_RE.search(text) or any(r.search(text) for r in INSTITUTION_RES) or any(r.search(text) for r, _ in MEDIUM_RES)):
        return True
    return False


def focus_text(text: str, anchors: list[str], window: int) -> str:
    text = normalize_space(text)
    positions = [text.find(anchor) for anchor in anchors if anchor and text.find(anchor) >= 0]
    if not positions:
        return text[: min(len(text), window * 2)]
    pos = min(positions)
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return text[start:end]


def extract_metadata(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    creator = extract_creator(text)
    if creator:
        out["creator_or_attribution"] = creator
    period = first_match(DYNASTY_RE, text)
    if period and not bad_period(period, text):
        out["creation_period_or_dynasty"] = period
    inst = ""
    for regex in INSTITUTION_RES:
        inst = first_match(regex, text)
        if inst:
            break
    if inst:
        out["collection_institution"] = inst
    dim = first_match(DIMENSION_RE, text)
    if not dim:
        m = DIMENSION_LH_RE.search(text)
        if m:
            dim = f"{m.group(2)}×{m.group(4)} 厘米"
    if dim:
        out["dimensions"] = dim
    medium = ""
    for regex, fixed in MEDIUM_RES:
        m = regex.search(text)
        if m:
            medium = fixed or m.group(0)
            break
    if medium:
        out["medium_material"] = medium
    return out


def extract_creator(text: str) -> str:
    m = CREATOR_EN_RE.search(text)
    if m:
        return normalize_space(m.group(1))
    m = CREATOR_CN_RE.search(text)
    if m:
        value = normalize_space(m.group(1))
        if value not in BAD_CREATOR_VALUES:
            return value
    return ""


def rule_support_label(target: dict[str, Any], fragment: dict[str, Any], field: str, value: str) -> tuple[str, str]:
    text = fragment.get("text") or ""
    title = normalize_space(target.get("depicted_work_title") or "")
    caption = normalize_space(target.get("caption_text") or "")
    if fragment.get("fragment_type") == "local_caption":
        return "support", "local caption fragment contains extracted field value"
    if title and title in text:
        return "support", "fragment contains target title and field-like value"
    fig = first_figure_anchor(caption)
    if fig and fig in text:
        return "support", "fragment contains target figure anchor and field-like value"
    other_titles = TITLE_RE.findall(text)
    if other_titles and (not title or title not in other_titles):
        return "wrong_target", "fragment has field-like value but title appears to bind another work"
    return "ambiguous", "fragment contains field-like value but target binding is not explicit"


def normalize_field_value(field: str, value: str) -> str:
    value = normalize_space(value)
    if field == "dimensions":
        return value.replace("*", "×").replace(" x ", "×").replace(" X ", "×")
    if field == "creation_period_or_dynasty":
        mapping = {"明": "明代", "清": "清代", "元": "元代", "宋": "宋代", "唐": "唐代"}
        return mapping.get(value, value)
    return value


def normalize_object_type(value: Any, domain: Any) -> str:
    value = normalize_space(value)
    domain = normalize_space(domain)
    if value in {"painting", "painting_detail", "diagram", "text_page", "other", "unclear"}:
        return value
    if domain == "landscape_detail":
        return "painting_detail"
    if domain in {"landscape_painting", "classical_painting_unclear_landscape"}:
        return "painting"
    if domain == "text_only":
        return "text_page"
    return "unclear"


def first_figure_anchor(text: str) -> str:
    m = FIGURE_ANCHOR_RE.search(text or "")
    return normalize_space(m.group(0)) if m else ""


def extract_title(text: str) -> str:
    m = TITLE_RE.search(text or "")
    return normalize_space(m.group(1)) if m else ""


def bad_period(value: str, text: str) -> bool:
    return value == "金" and re.search(r"五行|五星|金、木、水、火|金木水火土|水火土|金碧|泥金|金陵", text)


def first_match(regex: re.Pattern[str], text: str) -> str:
    m = regex.search(text or "")
    return normalize_space(m.group(1) if m and m.lastindex else m.group(0) if m else "")


def draw_target_overlay(page_image: Path, target_bbox: list[int], caption_bbox: list[int], out: Path) -> None:
    image = Image.open(page_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(target_bbox, outline="red", width=4)
    draw.text((target_bbox[0] + 4, max(0, target_bbox[1] - 18)), "target", fill="red", font=ImageFont.load_default())
    draw.rectangle(caption_bbox, outline="cyan", width=3)
    draw.text((caption_bbox[0] + 4, max(0, caption_bbox[1] - 18)), "caption", fill="cyan", font=ImageFont.load_default())
    image.save(out, quality=92)


def crop_image(page_image: Path, bbox: list[int], out: Path) -> None:
    image = Image.open(page_image).convert("RGB")
    w, h = image.size
    box = clamp_bbox(bbox, w, h)
    image.crop(tuple(box)).save(out, quality=92)


def image_data_url(path: Path, max_side: int) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        tmp = Path("/tmp") / f"v11_img_{sha1_text(str(path)+str(max_side))[:12]}.jpg"
        image.save(tmp, format="JPEG", quality=88)
    data = base64.b64encode(tmp.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text or "", flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def scale_bbox(bbox: list[float], sx: float, sy: float) -> list[int]:
    return [int(round(bbox[0] * sx)), int(round(bbox[1] * sy)), int(round(bbox[2] * sx)), int(round(bbox[3] * sy))]


def scale_norm1000_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    return clamp_bbox([round(bbox[0] * width / 1000), round(bbox[1] * height / 1000), round(bbox[2] * width / 1000), round(bbox[3] * height / 1000)], width, height)


def norm_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    return [round(bbox[0] * 1000 / width), round(bbox[1] * 1000 / height), round(bbox[2] * 1000 / width), round(bbox[3] * 1000 / height)]


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    return [x1, y1, x2, y2]


def valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def normalize_bbox_value(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        out = [int(round(float(v))) for v in value]
    except Exception:
        return None
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return [max(0, min(1000, v)) for v in out]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    value = normalize_space(value)
    return value if value in allowed else default


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def truncate(value: Any, limit: int) -> str:
    text = normalize_space(value)
    return text[:limit] + ("..." if len(text) > limit else "")


def safe_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\u4e00-\u9fff]+", "_", value)[:120]


def doc_id_for_path(path: Path) -> str:
    return sha1_text(str(path))[:12] + "_" + safe_stem(path.stem)


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
