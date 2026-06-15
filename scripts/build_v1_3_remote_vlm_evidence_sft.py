#!/usr/bin/env python3
"""Build v1.3 Remote-VLM Evidence SFT from raw PDFs.

v1.3 is intentionally built from raw PDFs rather than reusing older labels.
It creates page-level remote VLM annotations, FigureTargets,
EvidenceFragments, FieldSupportLabels, and multi-step SFT trajectories.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_gold_eval_v1_0_4 as gold_review  # noqa: E402
import build_v1_1_clean_evidence_fragment_probe as v11  # noqa: E402


RAW_PDF_ROOT = Path("/root/datasets/chinese_landscape_authority_corpus/raw_pdfs")
SOURCE_JSONL = Path("/root/datasets/chinese_landscape_authority_corpus/metadata/sources.jsonl")
OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DOTENV = REPO_ROOT / ".env"
DOCS_ROOT = REPO_ROOT / "docs"
MODEL = "qwen3.7-max-2026-06-08"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

BASE_FIELDS = ["caption_text", "depicted_work_title", "image_scope", "object_type"]
META_FIELDS = [
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
]
ALL_FIELDS = BASE_FIELDS + META_FIELDS
LANDSCAPE_DOMAINS = {"landscape_painting", "landscape_detail", "classical_painting_unclear_landscape"}
GOOD_OBJECT_TYPES = {"painting", "painting_detail", "unclear"}

FIG_RE = re.compile(
    r"(图|圖)\s*[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：][一二三四五六七八九十百〇零0-9]+)*[a-zA-Z]?|"
    r"(Fig\.?|Figure|Plate|PLATE)\s*[A-Za-z]?[0-9IVXivx]+(?:[.\-．:：][0-9IVXivx]+)*[a-zA-Z]?",
    re.I,
)
LANDSCAPE_RE = re.compile(
    r"山水|溪山|林泉|云山|雲山|峰|壑|谷|泉|溪|涧|澗|江|河|松|林|岩|石|秋山|春山|"
    r"landscape|mountain|river|stream|valley|pine|woods|retreat|hermitage",
    re.I,
)
METADATA_HINT_RE = re.compile(
    r"藏|博物馆|博物院|美术馆|故宫|台北|北京|厘米|公分|cm|CM|×|x|纸本|紙本|绢本|絹本|设色|設色|"
    r"ink on|color on|Museum|Collection|Palace|scroll|silk|paper",
    re.I,
)
NOISE_PAGE_RE = re.compile(r"目录|目錄|参考文献|參考文獻|致谢|摘要|关键词|contents|bibliography|references", re.I)
TITLE_RE = re.compile(r"《([^》]{2,80})》")
BAD_COLLECTION_RE = re.compile(
    r"\b(Gift|Bequest|Purchase|Accession|Anonymous Loan|Lent by)\b|捐赠|購藏|购藏|入藏|编号|館藏號|藏品編號",
    re.I,
)

PROMPT = """你是 EvidenceGrounded-VLM-AgentRL v1.3 的 page-level 标注员。请看一整页 PDF 图，找出中国/东亚古典山水画或山水画局部目标，并输出 JSON。

核心原则：
1. 只标页面可见事实，不要用内部知识补全。
2. bbox 统一使用 0-1000 归一化坐标 [x1,y1,x2,y2]，不是像素坐标。
3. target_bbox_norm1000 只框目标图像，不含图注、正文、页码、相邻图。
4. caption_bbox_norm1000 只框当前目标对应的完整图注；多图页一图一条 detection。
5. 如果目标图像没有可见图注，caption_text 写 null，caption_bbox_norm1000 写 null；不要把正文硬框成图注。
6. BaseLocate4 通常来自页面视觉、target crop 和 local caption。
7. Metadata5 如果图注可见支持，就从图注写；如果页面正文支持，可用 same_page_body_visual；没有可见证据必须 abstain。
8. collection_institution 只能填明确藏馆/收藏机构名；Gift/Bequest/Purchase/编号/捐赠/购藏不是机构名。

只输出 JSON：
{
  "page_summary": "一句话",
  "detections": [
    {
      "target_bbox_norm1000": [0,0,0,0],
      "caption_bbox_norm1000": [0,0,0,0] 或 null,
      "caption_text": "完整 corrected caption 或 null",
      "depicted_work_title": "题名或空串",
      "image_scope": "full_work|partial_detail|album_leaf_or_section|multi_work_comparison|unclear",
      "object_type": "painting|painting_detail|diagram|text_page|photo|other|unclear",
      "object_domain": "landscape_painting|landscape_detail|classical_painting_unclear_landscape|non_landscape_artwork|text_only|other|unclear",
      "caption_target_match": "yes|no|uncertain",
      "metadata_fields": {
        "creator_or_attribution": {"value": "", "abstain": true, "source": "unsupported|caption_visual|same_page_body_visual", "confidence": 0.0, "reason": ""},
        "creation_period_or_dynasty": {"value": "", "abstain": true, "source": "unsupported|caption_visual|same_page_body_visual", "confidence": 0.0, "reason": ""},
        "collection_institution": {"value": "", "abstain": true, "source": "unsupported|caption_visual|same_page_body_visual", "confidence": 0.0, "reason": ""},
        "dimensions": {"value": "", "abstain": true, "source": "unsupported|caption_visual|same_page_body_visual", "confidence": 0.0, "reason": ""},
        "medium_material": {"value": "", "abstain": true, "source": "unsupported|caption_visual|same_page_body_visual", "confidence": 0.0, "reason": ""}
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
    parser = argparse.ArgumentParser(description="Build v1.3 remote-VLM evidence SFT.")
    parser.add_argument("--raw-pdf-root", default=str(RAW_PDF_ROOT))
    parser.add_argument("--sources-jsonl", default=str(SOURCE_JSONL))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dotenv", default=str(DOTENV))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--page-candidates", type=int, default=760)
    parser.add_argument("--max-pdfs", type=int, default=0)
    parser.add_argument("--max-pages-per-pdf", type=int, default=8)
    parser.add_argument("--candidate-scan-mode", choices=["layout", "images"], default="layout")
    parser.add_argument("--scan-max-pages-per-pdf", type=int, default=120)
    parser.add_argument("--scan-stride-after-limit", type=int, default=10)
    parser.add_argument("--page-dpi", type=int, default=144)
    parser.add_argument("--seed", type=int, default=20260614)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--image-max-side", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=5500)
    parser.add_argument("--timeout", type=float, default=220.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-confidence", type=float, default=0.45)
    parser.add_argument("--train-caption-only", type=int, default=300)
    parser.add_argument("--train-retrieve-needed", type=int, default=300)
    parser.add_argument("--train-abstain-needed", type=int, default=120)
    parser.add_argument("--train-wrong-target", type=int, default=80)
    parser.add_argument("--val-caption-only", type=int, default=38)
    parser.add_argument("--val-retrieve-needed", type=int, default=37)
    parser.add_argument("--val-abstain-needed", type=int, default=15)
    parser.add_argument("--val-wrong-target", type=int, default=10)
    parser.add_argument("--test-caption-only", type=int, default=56)
    parser.add_argument("--test-retrieve-needed", type=int, default=56)
    parser.add_argument("--test-abstain-needed", type=int, default=23)
    parser.add_argument("--test-wrong-target", type=int, default=15)
    parser.add_argument(
        "--allow-cross-type-fill",
        action="store_true",
        help="Allow other trajectory types to fill split-size shortages. Default is strict quota: shortages remain explicit.",
    )
    parser.add_argument("--split-strategy", choices=["quota", "ratio"], default="quota")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--review-limit", type=int, default=80)
    parser.add_argument(
        "--append-pages",
        type=int,
        default=0,
        help="When resuming an existing output dir, append this many new candidate pages not already in page_records.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--skip-vlm",
        action="store_true",
        help="Use existing cache/page_level_vlm_stream.jsonl and continue postprocess only.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_review.load_dotenv(Path(args.dotenv))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"agentbench_v1_3_remote_vlm_evidence_sft_{stamp}"
    prepare_output_dir(out_dir, args)

    registry = v11.load_source_registry(Path(args.sources_jsonl))
    if not (out_dir / "page_records.jsonl").exists():
        page_specs = select_page_candidates(Path(args.raw_pdf_root), registry, args)
        write_jsonl(out_dir / "page_manifest.jsonl", page_specs)
        page_records, text_blocks, layout_blocks = materialize_pages(out_dir, page_specs, args)
        write_jsonl(out_dir / "page_records.jsonl", page_records)
        write_jsonl(out_dir / "page_text_blocks.jsonl", text_blocks)
        write_jsonl(out_dir / "page_layout_blocks.jsonl", layout_blocks)
    else:
        page_records = read_jsonl(out_dir / "page_records.jsonl")
        text_blocks = read_jsonl(out_dir / "page_text_blocks.jsonl")
        layout_blocks = read_jsonl(out_dir / "page_layout_blocks.jsonl")
        if args.append_pages > 0:
            existing_keys = {(str(row.get("source_path")), int(row.get("page_num") or 0)) for row in page_records}
            all_specs = select_page_candidates(Path(args.raw_pdf_root), registry, args)
            new_specs = [
                spec for spec in all_specs
                if (str(spec.get("source_path")), int(spec.get("page_num") or 0)) not in existing_keys
            ][: args.append_pages]
            if new_specs:
                page_manifest = read_jsonl(out_dir / "page_manifest.jsonl")
                page_manifest.extend(new_specs)
                write_jsonl(out_dir / "page_manifest.jsonl", page_manifest)
                new_records, new_text_blocks, new_layout_blocks = materialize_pages(
                    out_dir,
                    new_specs,
                    args,
                    start_index=len(page_records) + 1,
                )
                page_records.extend(new_records)
                text_blocks.extend(new_text_blocks)
                layout_blocks.extend(new_layout_blocks)
                write_jsonl(out_dir / "page_records.jsonl", page_records)
                write_jsonl(out_dir / "page_text_blocks.jsonl", text_blocks)
                write_jsonl(out_dir / "page_layout_blocks.jsonl", layout_blocks)

    if not args.skip_vlm:
        run_remote_vlm(out_dir, page_records, args)
    annotations = read_jsonl(out_dir / "cache" / "page_level_vlm_stream.jsonl")
    write_jsonl(out_dir / "page_level_vlm_annotations.jsonl", annotations)

    targets, fragments, labels = build_targets_fragments_labels(out_dir, page_records, text_blocks, annotations, args)
    write_jsonl(out_dir / "figure_targets.jsonl", targets)
    write_jsonl(out_dir / "evidence_fragments.jsonl", fragments)
    write_jsonl(out_dir / "field_support_labels.jsonl", labels)

    splits, split_summary = split_targets(targets, args)
    tasks, sft_rows = build_sft_dataset(out_dir, splits, fragments, labels, args)
    write_split_files(out_dir, splits, tasks, sft_rows)

    review_path = write_review(out_dir, page_records, splits, fragments, labels, args)
    manifest = build_manifest(out_dir, args, page_records, annotations, targets, fragments, labels, splits, sft_rows, review_path, split_summary)
    write_json(out_dir / "manifest.json", manifest)
    report_path = write_report(out_dir, manifest)
    docs_report = DOCS_ROOT / "02_指标与数据" / f"{stamp}_v1.3RemoteVLMEvidenceSFT构建报告.md"
    docs_report.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    manifest["artifacts"]["docs_report"] = str(docs_report)
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def prepare_output_dir(out_dir: Path, args: argparse.Namespace) -> None:
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"{out_dir} exists; use --resume or --overwrite")
    for sub in ["pages", "overlays", "crops", "captions", "cache", "sft", "tasks", "review"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)


def select_page_candidates(raw_root: Path, registry: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    pdfs = sorted(p for p in raw_root.rglob("*.pdf") if "failed_partial_downloads" not in str(p))
    if args.max_pdfs > 0:
        pdfs = pdfs[: args.max_pdfs]
    rows: list[dict[str, Any]] = []
    for pdf_index, pdf_path in enumerate(pdfs, start=1):
        try:
            doc = fitz.open(pdf_path)
        except Exception as exc:
            print(json.dumps({"pdf_error": str(pdf_path), "error": repr(exc)}, ensure_ascii=False), flush=True)
            continue
        candidates = []
        rel = str(pdf_path.relative_to(raw_root)) if pdf_path.is_relative_to(raw_root) else pdf_path.name
        category = rel.split("/", 1)[0] if "/" in rel else ""
        source_row = registry.get(pdf_path.name) or registry.get(str(pdf_path)) or {}
        page_indices = scan_page_indices(len(doc), args.scan_max_pages_per_pdf, args.scan_stride_after_limit)
        for page_index in page_indices:
            page = doc[page_index]
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            if args.candidate_scan_mode == "images":
                try:
                    image_blocks = len(page.get_images(full=False))
                except Exception:
                    image_blocks = 0
            else:
                try:
                    blocks = page.get_text("dict").get("blocks", [])
                except Exception:
                    blocks = []
                image_blocks = sum(1 for b in blocks if b.get("type") == 1)
            if image_blocks <= 0:
                continue
            fig_refs = len(FIG_RE.findall(text))
            landscape_terms = len(LANDSCAPE_RE.findall(text))
            metadata_hints = len(METADATA_HINT_RE.findall(text))
            noise_penalty = 40 if NOISE_PAGE_RE.search(text[:800]) else 0
            score = image_blocks * 25 + fig_refs * 8 + landscape_terms * 2 + metadata_hints * 3 - noise_penalty
            if score <= 0:
                continue
            candidates.append(
                {
                    "page_index": page_index,
                    "page_num": page_index + 1,
                    "score": round(score, 4),
                    "image_blocks": image_blocks,
                    "figure_refs": fig_refs,
                    "landscape_terms": landscape_terms,
                    "metadata_hints": metadata_hints,
                    "text_preview": v11.truncate(text, 1200),
                }
            )
        doc.close()
        if pdf_index % 10 == 0:
            print(
                json.dumps(
                    {"scan_pdf": f"{pdf_index}/{len(pdfs)}", "candidate_pages_so_far": len(rows), "last_pdf": pdf_path.name},
                    ensure_ascii=False,
                ),
                flush=True,
            )
        candidates.sort(key=lambda r: (-r["score"], r["page_num"]))
        for item in candidates[: max(1, args.max_pages_per_pdf)]:
            stem_key = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:12] + "_" + pdf_path.stem
            rows.append(
                {
                    "doc_id": stem_key,
                    "source_file": pdf_path.name,
                    "source_path": str(pdf_path),
                    "rel_path": rel,
                    "category": category,
                    "page_num": item["page_num"],
                    "page_count": None,
                    "page_score": item["score"],
                    "selection_stats": {
                        "image_blocks": item["image_blocks"],
                        "figure_refs": item["figure_refs"],
                        "landscape_terms": item["landscape_terms"],
                        "metadata_hints": item["metadata_hints"],
                        "source_registry": source_row,
                    },
                    "text_preview": item["text_preview"],
                }
            )
    random.Random(args.seed).shuffle(rows)
    rows.sort(key=lambda r: (-float(r["page_score"]), r["source_file"], r["page_num"]))
    return rows[: args.page_candidates]


def scan_page_indices(page_count: int, max_pages: int, stride_after_limit: int) -> list[int]:
    if max_pages <= 0 or page_count <= max_pages:
        return list(range(page_count))
    head = list(range(max_pages))
    stride = max(1, stride_after_limit)
    tail = list(range(max_pages, page_count, stride))
    return head + tail


def materialize_pages(
    out_dir: Path,
    specs: list[dict[str, Any]],
    args: argparse.Namespace,
    start_index: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    page_records: list[dict[str, Any]] = []
    text_blocks: list[dict[str, Any]] = []
    layout_blocks: list[dict[str, Any]] = []
    font = ImageFont.load_default()
    for idx, spec in enumerate(specs, start=start_index):
        pdf_path = Path(spec["source_path"])
        try:
            doc = fitz.open(pdf_path)
            page = doc[int(spec["page_num"]) - 1]
            spec["page_count"] = len(doc)
            pix = page.get_pixmap(matrix=fitz.Matrix(args.page_dpi / 72, args.page_dpi / 72), alpha=False)
            page_id = f"v13p_{idx:06d}"
            page_key = f"{page_id}_{v11.safe_stem(spec['source_file'])[:50]}_p{int(spec['page_num']):04d}"
            page_path = out_dir / "pages" / f"{page_key}.png"
            pix.save(str(page_path))
            width, height = pix.width, pix.height
            sx = width / page.rect.width
            sy = height / page.rect.height
            raw_text = page.get_text("text") or ""
            blocks = page.get_text("dict").get("blocks", [])
            image = Image.open(page_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            for bidx, block in enumerate(blocks):
                bbox = scale_pdf_bbox(block.get("bbox") or [], sx, sy, width, height)
                if not bbox:
                    continue
                if block.get("type") == 1:
                    row = {
                        "block_id": f"{page_id}_img{bidx:04d}",
                        "page_id": page_id,
                        "source_file": spec["source_file"],
                        "page_num": spec["page_num"],
                        "block_type": "image",
                        "bbox": bbox,
                        "bbox_norm1000": norm_bbox(bbox, width, height),
                        "source": "pymupdf_image_block",
                    }
                    layout_blocks.append(row)
                    draw.rectangle(bbox, outline=(180, 180, 180), width=2)
                elif block.get("type") == 0:
                    text = "".join(
                        span.get("text", "")
                        for line in block.get("lines", [])
                        for span in line.get("spans", [])
                    )
                    text = v11.normalize_space(text)
                    if text:
                        text_blocks.append(
                            {
                                "block_id": f"{page_id}_txt{bidx:04d}",
                                "page_id": page_id,
                                "doc_id": spec["doc_id"],
                                "source_file": spec["source_file"],
                                "source_path": spec["source_path"],
                                "page_num": spec["page_num"],
                                "block_type": "text",
                                "bbox": bbox,
                                "bbox_norm1000": norm_bbox(bbox, width, height),
                                "text": text,
                                "source": "pymupdf_text_block",
                            }
                        )
            draw.text((8, 8), page_id, fill=(180, 0, 0), font=font)
            image.save(out_dir / "cache" / f"{page_key}_layout.jpg", quality=90)
            page_records.append(
                {
                    **spec,
                    "page_id": page_id,
                    "page_key": page_key,
                    "page_image": str(page_path),
                    "width": width,
                    "height": height,
                    "raw_pdf_text": raw_text,
                    "pdf_text": raw_text,
                }
            )
            doc.close()
        except Exception as exc:
            print(json.dumps({"page_materialize_error": spec, "error": repr(exc)}, ensure_ascii=False), flush=True)
    return page_records, text_blocks, layout_blocks


def run_remote_vlm(out_dir: Path, page_records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    stream_path = out_dir / "cache" / "page_level_vlm_stream.jsonl"
    existing = {row.get("page_id"): row for row in read_jsonl(stream_path) if row.get("page_id")} if stream_path.exists() else {}
    pending = [row for row in page_records if row.get("page_id") not in existing]
    if not pending:
        return
    max_workers = max(1, min(args.concurrency, len(pending)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(review_page, row, args, out_dir) for row in pending]
        for future in as_completed(futures):
            row = future.result()
            append_jsonl(stream_path, row)
            print(
                json.dumps(
                    {
                        "page_id": row.get("page_id"),
                        "ok": row.get("ok"),
                        "detections": len(row.get("detections") or []),
                        "elapsed_sec": row.get("elapsed_sec"),
                        "error": row.get("error", "")[:120],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


def review_page(record: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    page_image = Path(record["page_image"])
    with Image.open(page_image) as image:
        width, height = image.size
    image_url, sent_size = image_data_url(page_image, out_dir, args.image_max_side)
    meta = {
        "page_id": record.get("page_id"),
        "source_file": record.get("source_file"),
        "page_num": record.get("page_num"),
        "page_size_px": [width, height],
        "sent_image_size_px": list(sent_size),
        "pdf_text_preview": v11.truncate(record.get("raw_pdf_text") or "", 1400),
    }
    client = OpenAI(
        api_key=args.api_key or os.environ.get(args.api_key_env) or "EMPTY",
        base_url=args.base_url,
        timeout=args.timeout,
    )
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.disable_thinking else None
    start = time.time()
    raw = ""
    parsed: dict[str, Any] = {}
    ok = False
    error = ""
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_url}},
                            {"type": "text", "text": PROMPT + json.dumps(meta, ensure_ascii=False, indent=2)},
                        ],
                    }
                ],
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                response_format={"type": "json_object"},
                extra_body=extra_body,
            )
            raw = response.choices[0].message.content or "{}"
            parsed = parse_json(raw)
            ok = True
            error = ""
            break
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            time.sleep(1.5 * (attempt + 1))
    detections = normalize_detections(parsed.get("detections"), width, height)
    row = {
        **{k: record.get(k) for k in ["page_id", "doc_id", "source_file", "source_path", "rel_path", "category", "page_num", "page_count", "page_image", "width", "height", "raw_pdf_text"]},
        "ok": ok,
        "error": error,
        "review_model": args.model if ok else "dashscope_failed",
        "elapsed_sec": round(time.time() - start, 2),
        "page_summary": str(parsed.get("page_summary") or ""),
        "detections": detections,
        "raw_response": raw,
        "sent_image_size_px": list(sent_size),
    }
    return row


def normalize_detections(value: Any, width: int, height: int) -> list[dict[str, Any]]:
    items = value if isinstance(value, list) else []
    out = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        target_norm = normalize_bbox(item.get("target_bbox_norm1000"))
        caption_norm = normalize_bbox(item.get("caption_bbox_norm1000"))
        metadata = item.get("metadata_fields") if isinstance(item.get("metadata_fields"), dict) else {}
        metadata = {field: normalize_metadata(metadata.get(field)) for field in META_FIELDS}
        metadata["collection_institution"] = strict_collection(metadata.get("collection_institution"))
        caption_raw = item.get("caption_text")
        caption_text = None if caption_raw is None else v11.normalize_space(caption_raw or "")
        out.append(
            {
                "detection_index": idx,
                "target_bbox_norm1000": target_norm,
                "caption_bbox_norm1000": caption_norm,
                "target_bbox_px": denorm_bbox(target_norm, width, height) if target_norm else None,
                "caption_bbox_px": denorm_bbox(caption_norm, width, height) if caption_norm else None,
                "caption_text": caption_text,
                "corrected_caption_text": caption_text,
                "depicted_work_title": v11.normalize_space(item.get("depicted_work_title") or ""),
                "image_scope": v11.normalize_space(item.get("image_scope") or "unclear"),
                "object_type": v11.normalize_space(item.get("object_type") or "unclear"),
                "object_domain": v11.normalize_space(item.get("object_domain") or "unclear"),
                "caption_target_match": v11.normalize_space(item.get("caption_target_match") or "uncertain"),
                "metadata_fields": metadata,
                "accept_for_probe": bool(item.get("accept_for_probe", True)),
                "needs_human_review": bool(item.get("needs_human_review")),
                "confidence": v11.safe_float(item.get("confidence"), 0.0),
                "reason": str(item.get("reason") or ""),
            }
        )
    return out


def build_targets_fragments_labels(
    out_dir: Path,
    page_records: list[dict[str, Any]],
    text_blocks: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    page_by_id = {row["page_id"]: row for row in page_records}
    text_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for block in text_blocks:
        text_by_page[str(block.get("page_id"))].append(block)
    targets: list[dict[str, Any]] = []
    fragments: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    target_serial = 0
    for ann in annotations:
        page = page_by_id.get(str(ann.get("page_id"))) or ann
        detections = ann.get("detections") or []
        for det in detections:
            if not accepted_detection(det, args):
                continue
            target_serial += 1
            target_id = f"v13_t_{target_serial:06d}"
            page_id = str(ann.get("page_id"))
            target = build_target(target_id, page, det)
            crop_paths = write_target_assets(out_dir, target, page)
            target.update(crop_paths)
            local_frags = build_fragments_for_target(target, page, det, text_by_page.get(page_id, []), detections)
            field_labels = build_labels_for_target(target, local_frags, det)
            target["trajectory_type"] = classify_trajectory(target, local_frags, field_labels)
            targets.append(target)
            fragments.extend(local_frags)
            labels.extend(field_labels)
    return targets, fragments, labels


def accepted_detection(det: dict[str, Any], args: argparse.Namespace) -> bool:
    if not det.get("target_bbox_px") or not det.get("target_bbox_norm1000"):
        return False
    if v11.safe_float(det.get("confidence"), 0.0) < args.min_confidence:
        return False
    if det.get("object_domain") not in LANDSCAPE_DOMAINS:
        return False
    if det.get("object_type") not in GOOD_OBJECT_TYPES:
        return False
    if det.get("caption_target_match") == "no":
        return False
    return True


def build_target(target_id: str, page: dict[str, Any], det: dict[str, Any]) -> dict[str, Any]:
    base = {
        "caption_text": det.get("caption_text"),
        "depicted_work_title": det.get("depicted_work_title") or infer_title(det.get("caption_text") or ""),
        "image_scope": det.get("image_scope") or "unclear",
        "object_type": det.get("object_type") or "unclear",
    }
    return {
        "target_id": target_id,
        "page_id": page.get("page_id"),
        "doc_id": page.get("doc_id"),
        "source_file": page.get("source_file"),
        "source_path": page.get("source_path"),
        "rel_path": page.get("rel_path"),
        "category": page.get("category"),
        "page_num": page.get("page_num"),
        "page_image": page.get("page_image"),
        "width": page.get("width"),
        "height": page.get("height"),
        "target_bbox_norm1000": det.get("target_bbox_norm1000"),
        "target_bbox_px": det.get("target_bbox_px"),
        "caption_bbox_norm1000": det.get("caption_bbox_norm1000"),
        "caption_bbox_px": det.get("caption_bbox_px"),
        "base_fields": base,
        "metadata_fields": det.get("metadata_fields") or {},
        "confidence": det.get("confidence"),
        "reason": det.get("reason"),
        "target_quality": "accepted",
    }


def write_target_assets(out_dir: Path, target: dict[str, Any], page: dict[str, Any]) -> dict[str, str]:
    src = Path(str(page["page_image"]))
    key = f"{target['target_id']}_{v11.safe_stem(str(page.get('source_file')))[:36]}_p{int(page.get('page_num')):04d}"
    image = Image.open(src).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    target_bbox = target.get("target_bbox_px")
    caption_bbox = target.get("caption_bbox_px")
    crop_path = out_dir / "crops" / f"{key}_target.jpg"
    overlay_path = out_dir / "overlays" / f"{key}_overlay.jpg"
    cap_path = out_dir / "captions" / f"{key}_caption.jpg"
    if v11.valid_bbox(target_bbox):
        draw.rectangle(target_bbox, outline=(220, 0, 0), width=5)
        image.crop(tuple(v11.clamp_bbox(target_bbox, image.width, image.height))).save(crop_path, quality=92)
    if v11.valid_bbox(caption_bbox):
        draw.rectangle(caption_bbox, outline=(0, 185, 210), width=5)
        image.crop(tuple(v11.clamp_bbox(caption_bbox, image.width, image.height))).save(cap_path, quality=92)
    overlay.save(overlay_path, quality=92)
    return {
        "artwork_image": str(crop_path) if crop_path.exists() else "",
        "caption_image": str(cap_path) if cap_path.exists() else "",
        "overlay_image": str(overlay_path),
    }


def build_fragments_for_target(
    target: dict[str, Any],
    page: dict[str, Any],
    det: dict[str, Any],
    page_text_blocks: list[dict[str, Any]],
    sibling_detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tid = target["target_id"]
    frags: list[dict[str, Any]] = [
        {
            "fragment_id": f"{tid}_visual",
            "target_id": tid,
            "fragment_type": "local_visual",
            "raw_text": "",
            "corrected_text": "目标裁剪图像视觉证据",
            "display_text": "目标裁剪图像视觉证据",
            "source_page_id": page.get("page_id"),
            "source_file": page.get("source_file"),
            "page_num": page.get("page_num"),
            "source_bbox_norm1000": target.get("target_bbox_norm1000"),
            "source_bbox_px": target.get("target_bbox_px"),
            "image_path": target.get("artwork_image"),
            "is_model_internal_knowledge": False,
        }
    ]
    caption = target.get("base_fields", {}).get("caption_text")
    if caption:
        frags.append(
            {
                "fragment_id": f"{tid}_caption",
                "target_id": tid,
                "fragment_type": "local_caption_visual",
                "raw_text": caption,
                "corrected_text": caption,
                "display_text": caption,
                "source_page_id": page.get("page_id"),
                "source_file": page.get("source_file"),
                "page_num": page.get("page_num"),
                "source_bbox_norm1000": target.get("caption_bbox_norm1000"),
                "source_bbox_px": target.get("caption_bbox_px"),
                "image_path": target.get("caption_image"),
                "is_model_internal_knowledge": False,
            }
        )
    body_text = build_body_fragment_text(target, page_text_blocks)
    if body_text:
        frags.append(
            {
                "fragment_id": f"{tid}_body",
                "target_id": tid,
                "fragment_type": "same_page_body",
                "raw_text": body_text,
                "corrected_text": body_text,
                "display_text": body_text,
                "source_page_id": page.get("page_id"),
                "source_file": page.get("source_file"),
                "page_num": page.get("page_num"),
                "source_bbox_norm1000": None,
                "source_bbox_px": None,
                "is_model_internal_knowledge": False,
            }
        )
    wrong = choose_wrong_target_fragment(target, sibling_detections)
    if wrong:
        frags.append(
            {
                "fragment_id": f"{tid}_wrong",
                "target_id": tid,
                "fragment_type": "wrong_target_caption",
                "raw_text": wrong,
                "corrected_text": wrong,
                "display_text": wrong,
                "source_page_id": page.get("page_id"),
                "source_file": page.get("source_file"),
                "page_num": page.get("page_num"),
                "source_bbox_norm1000": None,
                "source_bbox_px": None,
                "is_model_internal_knowledge": False,
            }
        )
    return frags


def build_labels_for_target(target: dict[str, Any], frags: list[dict[str, Any]], det: dict[str, Any]) -> list[dict[str, Any]]:
    tid = target["target_id"]
    frag_by_type = {f["fragment_type"]: f for f in frags}
    labels: list[dict[str, Any]] = []
    caption_frag = frag_by_type.get("local_caption_visual")
    visual_frag = frag_by_type.get("local_visual")
    body_frag = frag_by_type.get("same_page_body")
    wrong_frag = frag_by_type.get("wrong_target_caption")
    for field, value in (target.get("base_fields") or {}).items():
        if field == "caption_text" and value is None:
            labels.append(label(tid, field, None, None, "no_support", "rule-no-visible-caption", 0.9, "目标没有可见图注。"))
        elif field == "object_type" and value:
            labels.append(label(tid, field, value, visual_frag["fragment_id"] if visual_frag else None, "support", "vlm-page-read", 0.9, "目标 crop 视觉支持对象类型。"))
        elif value and caption_frag:
            labels.append(label(tid, field, value, caption_frag["fragment_id"], "support", "vlm-page-read", 0.9, "local caption 支持 BaseLocate4 字段。"))
        elif value and visual_frag:
            labels.append(label(tid, field, value, visual_frag["fragment_id"], "support", "vlm-page-read", 0.7, "视觉证据支持字段。"))
        else:
            labels.append(label(tid, field, None, None, "no_support", "rule-missing-base-field", 0.7, "缺少可靠可见证据。"))
    for field in META_FIELDS:
        entry = (target.get("metadata_fields") or {}).get(field) or {}
        value = entry.get("value")
        if not entry.get("abstain") and value:
            source = str(entry.get("source") or "")
            if "caption" in source and caption_frag:
                labels.append(label(tid, field, value, caption_frag["fragment_id"], "support", "vlm-caption", entry.get("confidence", 0.85), entry.get("reason") or "caption 支持字段。"))
            elif body_frag:
                labels.append(label(tid, field, value, body_frag["fragment_id"], "support", "vlm-body-or-pdf-fragment", entry.get("confidence", 0.75), entry.get("reason") or "正文/fragment 支持字段。"))
            elif caption_frag:
                labels.append(label(tid, field, value, caption_frag["fragment_id"], "support", "vlm-page-read", entry.get("confidence", 0.65), entry.get("reason") or "页面可见证据支持字段。"))
            else:
                labels.append(label(tid, field, value, None, "ambiguous", "missing-fragment", 0.4, "字段有值但缺少可引用 fragment。"))
        else:
            labels.append(label(tid, field, None, None, "no_support", "vlm-abstain", entry.get("confidence", 0.7), entry.get("reason") or "页面没有可见证据。"))
        if wrong_frag:
            wrong_value = extract_wrong_value(wrong_frag["display_text"], field)
            if wrong_value:
                labels.append(label(tid, field, wrong_value, wrong_frag["fragment_id"], "wrong_target", "same-page-negative", 0.9, "相邻图/相似标题 fragment，不支持当前 target。"))
    return labels


def classify_trajectory(target: dict[str, Any], frags: list[dict[str, Any]], labels: list[dict[str, Any]]) -> str:
    support_by_field = {l["field"]: l for l in labels if l.get("label") == "support"}
    caption_support_meta = [
        f for f in META_FIELDS
        if f in support_by_field and str(support_by_field[f].get("fragment_id", "")).endswith("_caption")
    ]
    body_support_meta = [
        f for f in META_FIELDS
        if f in support_by_field and not str(support_by_field[f].get("fragment_id", "")).endswith("_caption")
    ]
    no_support_meta = [l for l in labels if l.get("field") in META_FIELDS and l.get("label") == "no_support"]
    wrong = [l for l in labels if l.get("label") == "wrong_target"]
    if wrong:
        return "wrong_target_negative"
    if body_support_meta:
        return "retrieve_needed"
    if len(caption_support_meta) >= 3:
        return "caption_only"
    if no_support_meta:
        return "abstain_needed"
    return "retrieve_needed"


def split_targets(targets: list[dict[str, Any]], args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if args.split_strategy == "ratio":
        return split_targets_by_type_ratio(targets, args)
    quotas = {
        "train": {
            "caption_only": args.train_caption_only,
            "retrieve_needed": args.train_retrieve_needed,
            "abstain_needed": args.train_abstain_needed,
            "wrong_target_negative": args.train_wrong_target,
        },
        "val": {
            "caption_only": args.val_caption_only,
            "retrieve_needed": args.val_retrieve_needed,
            "abstain_needed": args.val_abstain_needed,
            "wrong_target_negative": args.val_wrong_target,
        },
        "test": {
            "caption_only": args.test_caption_only,
            "retrieve_needed": args.test_retrieve_needed,
            "abstain_needed": args.test_abstain_needed,
            "wrong_target_negative": args.test_wrong_target,
        },
    }
    rng = random.Random(args.seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        buckets[target.get("trajectory_type", "retrieve_needed")].append(target)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    used_docs: dict[str, str] = {}
    splits = {"train": [], "val": [], "test": []}
    shortage: dict[str, dict[str, int]] = defaultdict(dict)
    for split in ["train", "val", "test"]:
        for ttype, quota in quotas[split].items():
            selected = []
            remaining = []
            for target in buckets.get(ttype, []):
                doc = str(target.get("doc_id") or target.get("source_file"))
                if doc in used_docs and used_docs[doc] != split:
                    remaining.append(target)
                    continue
                if len(selected) < quota:
                    target = dict(target)
                    target["split"] = split
                    selected.append(target)
                    used_docs[doc] = split
                else:
                    remaining.append(target)
            buckets[ttype] = remaining
            splits[split].extend(selected)
            if len(selected) < quota:
                shortage[split][ttype] = quota - len(selected)
    if args.allow_cross_type_fill:
        # Optional scale-fill mode for probes only. Formal SFT builds should keep
        # shortages explicit instead of replacing one trajectory type with another.
        for split in ["train", "val", "test"]:
            need = sum(shortage.get(split, {}).values())
            if need <= 0:
                continue
            for ttype in list(buckets):
                if need <= 0:
                    break
                remaining = []
                for target in buckets[ttype]:
                    if need <= 0:
                        remaining.append(target)
                        continue
                    doc = str(target.get("doc_id") or target.get("source_file"))
                    if doc in used_docs and used_docs[doc] != split:
                        remaining.append(target)
                        continue
                    selected_target = dict(target)
                    selected_target["split"] = split
                    splits[split].append(selected_target)
                    used_docs[doc] = split
                    need -= 1
                buckets[ttype] = remaining
    # If document isolation prevents a quota type from reaching the requested
    # count, relax that constraint for the same type only. Never duplicate a
    # target across splits.
    target_sizes = {split: sum(quotas[split].values()) for split in ["train", "val", "test"]}
    selected_ids = {str(t.get("target_id")) for rows in splits.values() for t in rows}
    remaining_all = [t for t in targets if str(t.get("target_id")) not in selected_ids]
    rng.shuffle(remaining_all)
    for split in ["train", "val", "test"]:
        by_type = Counter(t.get("trajectory_type") for t in splits[split])
        for ttype, quota in quotas[split].items():
            while by_type.get(ttype, 0) < quota:
                idx = next((i for i, t in enumerate(remaining_all) if t.get("trajectory_type") == ttype), None)
                if idx is None:
                    break
                target = dict(remaining_all.pop(idx))
                target["split"] = split
                splits[split].append(target)
                by_type[ttype] += 1
        if args.allow_cross_type_fill:
            while len(splits[split]) < target_sizes[split] and remaining_all:
                target = dict(remaining_all.pop())
                target["split"] = split
                splits[split].append(target)
    requested_shortage = {
        split: {
            ttype: max(0, quota - Counter(t.get("trajectory_type") for t in splits[split]).get(ttype, 0))
            for ttype, quota in quotas[split].items()
            if max(0, quota - Counter(t.get("trajectory_type") for t in splits[split]).get(ttype, 0)) > 0
        }
        for split in ["train", "val", "test"]
    }
    requested_shortage = {split: rows for split, rows in requested_shortage.items() if rows}
    summary = {
        "requested_quotas": quotas,
        "shortage": requested_shortage,
        "available_by_type": dict(Counter(t.get("trajectory_type") for t in targets)),
        "allow_cross_type_fill": bool(args.allow_cross_type_fill),
        "selected_by_split_type": {
            split: dict(Counter(t.get("trajectory_type") for t in rows)) for split, rows in splits.items()
        },
        "selected_by_split": {split: len(rows) for split, rows in splits.items()},
    }
    return splits, summary


def split_targets_by_type_ratio(
    targets: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    ratios = {
        "train": float(args.train_ratio),
        "val": float(args.val_ratio),
        "test": float(args.test_ratio),
    }
    ratio_sum = sum(ratios.values())
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value")
    ratios = {key: value / ratio_sum for key, value in ratios.items()}
    rng = random.Random(args.seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in targets:
        buckets[target.get("trajectory_type", "retrieve_needed")].append(target)
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    per_type_counts: dict[str, dict[str, int]] = {}
    for ttype, rows in sorted(buckets.items()):
        rows = list(rows)
        rng.shuffle(rows)
        n = len(rows)
        val_n = int(round(n * ratios["val"]))
        test_n = int(round(n * ratios["test"]))
        if n >= 10:
            val_n = max(1, val_n)
            test_n = max(1, test_n)
        if val_n + test_n > n:
            overflow = val_n + test_n - n
            test_n = max(0, test_n - overflow)
        train_n = n - val_n - test_n
        boundaries = {
            "train": train_n,
            "val": train_n + val_n,
            "test": n,
        }
        for split, split_rows in [
            ("train", rows[: boundaries["train"]]),
            ("val", rows[boundaries["train"] : boundaries["val"]]),
            ("test", rows[boundaries["val"] : boundaries["test"]]),
        ]:
            for target in split_rows:
                selected = dict(target)
                selected["split"] = split
                splits[split].append(selected)
        per_type_counts[ttype] = {
            "train": train_n,
            "val": val_n,
            "test": test_n,
            "total": n,
        }
    for split_rows in splits.values():
        rng.shuffle(split_rows)
    summary = {
        "split_strategy": "ratio",
        "ratios": ratios,
        "allow_cross_type_fill": False,
        "available_by_type": dict(Counter(t.get("trajectory_type") for t in targets)),
        "selected_by_split_type": {
            split: dict(Counter(t.get("trajectory_type") for t in rows)) for split, rows in splits.items()
        },
        "selected_by_split": {split: len(rows) for split, rows in splits.items()},
        "per_type_counts": per_type_counts,
        "shortage": {},
    }
    return splits, summary


def build_sft_dataset(
    out_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    fragments: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frags_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frag in fragments:
        frags_by_target[frag["target_id"]].append(frag)
    for lab in labels:
        labels_by_target[lab["target_id"]].append(lab)
    tasks: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for split, targets in splits.items():
        for target in targets:
            task = build_task(target, frags_by_target[target["target_id"]], labels_by_target[target["target_id"]], split)
            tasks.append(task)
            actions = build_oracle_actions_v13(task)
            rows.extend(build_sft_rows_v13(task, actions))
    return tasks, rows


def build_task(target: dict[str, Any], fragments: list[dict[str, Any]], labels: list[dict[str, Any]], split: str) -> dict[str, Any]:
    return {
        "task_id": target["target_id"],
        "split": split,
        "dataset_version": "v1.3_remote_vlm_evidence_sft",
        "tool_schema_version": "v1.3_remote_vlm_evidence",
        "task_type": "remote_vlm_page_evidence_field_claim",
        "trajectory_type": target.get("trajectory_type"),
        "source_file": target.get("source_file"),
        "source_path": target.get("source_path"),
        "page": target.get("page_num"),
        "page_id": target.get("page_id"),
        "page_image": target.get("page_image"),
        "artwork_image": target.get("artwork_image"),
        "caption_image": target.get("caption_image"),
        "overlay_image": target.get("overlay_image"),
        "target_bbox_norm1000": target.get("target_bbox_norm1000"),
        "target_bbox_px": target.get("target_bbox_px"),
        "caption_bbox_norm1000": target.get("caption_bbox_norm1000"),
        "caption_bbox_px": target.get("caption_bbox_px"),
        "target_fields": list(ALL_FIELDS),
        "fragments": fragments,
        "field_support_labels": labels,
        "gold_fields": build_gold_fields(labels),
    }


def build_oracle_actions_v13(task: dict[str, Any]) -> list[dict[str, Any]]:
    labels = task.get("field_support_labels") or []
    support = {lab["field"]: lab for lab in labels if lab.get("label") == "support"}
    no_support = {lab["field"]: lab for lab in labels if lab.get("label") == "no_support"}
    wrong = [lab for lab in labels if lab.get("label") == "wrong_target"]
    frags = {frag["fragment_id"]: frag for frag in task.get("fragments") or []}
    caption_id = next((fid for fid in frags if fid.endswith("_caption")), "")
    visual_id = next((fid for fid in frags if fid.endswith("_visual")), "")
    body_ids = [fid for fid, frag in frags.items() if frag.get("fragment_type") in {"same_page_body", "nearby_page_body", "same_doc_kb"}]
    wrong_ids = [lab.get("fragment_id") for lab in wrong if lab.get("fragment_id")]
    actions = [
        {"action": "inspect_page", "top_k": 12},
        {
            "action": "localize_target",
            "target_bbox_norm1000": task.get("target_bbox_norm1000"),
            "caption_bbox_norm1000": task.get("caption_bbox_norm1000"),
        },
        {"action": "crop_target", "bbox": task.get("target_bbox_px")},
    ]
    if caption_id:
        actions.append({"action": "open_fragment", "fragment_id": caption_id})
    if visual_id:
        actions.append({"action": "open_fragment", "fragment_id": visual_id})
    for field in BASE_FIELDS:
        if field in support:
            actions.append(write_field_action(support[field]))
        elif field in no_support:
            actions.append({"action": "abstain_field", "field": field, "reason": no_support[field].get("reason") or "证据不足"})
    caption_supported_meta = [
        f for f in META_FIELDS
        if f in support and str(support[f].get("fragment_id", "")).endswith("_caption")
    ]
    retrieve_fields = [
        f for f in META_FIELDS
        if f in support and not str(support[f].get("fragment_id", "")).endswith("_caption")
    ]
    abstain_fields = [f for f in META_FIELDS if f not in support]
    for field in caption_supported_meta:
        actions.append(write_field_action(support[field]))
    if retrieve_fields or wrong_ids or any(f in no_support for f in abstain_fields):
        actions.append(
            {
                "action": "retrieve_fragments",
                "query": build_retrieve_query(task, retrieve_fields or abstain_fields),
                "scope": "same_page_or_same_document",
                "top_k": 5,
            }
        )
        for fid in dedupe([*body_ids[:2], *wrong_ids[:2]]):
            actions.append({"action": "open_fragment", "fragment_id": fid})
    for field in retrieve_fields:
        actions.append(write_field_action(support[field]))
    for field in abstain_fields:
        if field in support:
            continue
        lab = no_support.get(field)
        actions.append({"action": "abstain_field", "field": field, "reason": (lab or {}).get("reason") or "没有可靠支持证据"})
    actions.append({"action": "finish", "status": "done"})
    return actions


def build_sft_rows_v13(task: dict[str, Any], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    history: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    field_state = {field: "remaining" for field in ALL_FIELDS}
    images = [task.get("page_image")]
    for step, action in enumerate(actions):
        row = {
            "task_id": task["task_id"],
            "split": task["split"],
            "trajectory_type": task.get("trajectory_type"),
            "step": step,
            "tool_schema_version": "v1.3_remote_vlm_evidence",
            "action": action,
            "history": json.loads(json.dumps(history, ensure_ascii=False)),
            "tool_results": json.loads(json.dumps(tool_results, ensure_ascii=False)),
            "field_state": dict(field_state),
            "available_actions": [action["action"]],
            "phase_name": "v1_3_" + action["action"],
            "phase_hint": phase_hint_v13(action),
            "images": [item for item in images if item],
        }
        row["prompt_text"] = prompt_text_v13(task, row)
        row["messages"] = build_messages(row)
        rows.append(row)
        result = result_for_action_v13(task, action)
        history.append(action)
        tool_results.append(result)
        if action["action"] == "crop_target" and task.get("artwork_image"):
            images = [task.get("page_image"), task.get("artwork_image")]
        if action["action"] == "write_field_value":
            field_state[str(action.get("field"))] = "written"
        if action["action"] == "abstain_field":
            field_state[str(action.get("field"))] = "abstained"
    return rows


def prompt_text_v13(task: dict[str, Any], row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "你是 EvidenceGrounded-VLM-AgentRL v1.3 的 evidence-grounded VLM tool-call agent。",
            "目标：从整页 PDF 图像中定位目标图和图注，读取 local caption，必要时检索/打开正文或 KB fragment，写出 BaseLocate4 + Metadata5 字段。",
            f"task_id：{task.get('task_id')}；step：{row.get('step')}；trajectory_type：{task.get('trajectory_type')}",
            f"source_file：{task.get('source_file')}；page：{task.get('page')}",
            "字段：caption_text, depicted_work_title, image_scope, object_type, creator_or_attribution, creation_period_or_dynasty, collection_institution, dimensions, medium_material。",
            "策略：BaseLocate4 通常由页面视觉、target crop、local caption 支持。Metadata5 如果 local caption 已支持，不要 retrieve；caption 不支持才 retrieve/open fragment。wrong_target fragment 不能引用为当前 target 证据。",
            '工具：{"action":"inspect_page","top_k":12}; {"action":"localize_target","target_bbox_norm1000":[...],"caption_bbox_norm1000":[...]或null}; {"action":"crop_target","bbox":[...]}; {"action":"open_fragment","fragment_id":"..."}; {"action":"retrieve_fragments","query":"...","scope":"same_page_or_same_document","top_k":5}; {"action":"write_field_value","field":"...","value":...,"evidence_ids":["..."],"confidence":0.0}; {"action":"abstain_field","field":"...","reason":"..."}; {"action":"finish","status":"done"}',
            "约束：只输出一个 JSON 对象；不要 markdown；不要编造页面不可见信息；所有非 abstain 字段必须引用支持 fragment_id；remaining 字段未完成时禁止 finish。",
            "历史动作：",
            json.dumps(row.get("history") or [], ensure_ascii=False, separators=(",", ":")),
            "工具返回摘要：",
            json.dumps(simplify_results_v13(row.get("tool_results") or []), ensure_ascii=False, separators=(",", ":")),
            "当前字段状态：",
            json.dumps(row.get("field_state") or {}, ensure_ascii=False, separators=(",", ":")),
            "当前阶段允许 action：",
            json.dumps(row.get("available_actions") or [], ensure_ascii=False, separators=(",", ":")),
            f"阶段提示：{row.get('phase_hint')}",
            "请输出下一步工具调用 JSON。",
        ]
    )


def result_for_action_v13(task: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    name = action.get("action")
    fragments = {frag["fragment_id"]: frag for frag in task.get("fragments") or []}
    if name == "inspect_page":
        return {
            "tool": "inspect_page",
            "page_id": task.get("page_id"),
            "target_bbox_norm1000": task.get("target_bbox_norm1000"),
            "caption_bbox_norm1000": task.get("caption_bbox_norm1000"),
            "fragment_ids": [frag["fragment_id"] for frag in task.get("fragments") or [] if frag.get("fragment_type") in {"local_caption_visual", "local_visual", "wrong_target_caption"}],
        }
    if name == "localize_target":
        return {"tool": "localize_target", "target_bbox_norm1000": action.get("target_bbox_norm1000"), "caption_bbox_norm1000": action.get("caption_bbox_norm1000")}
    if name == "crop_target":
        return {"tool": "crop_target", "bbox": action.get("bbox"), "crop_path": task.get("artwork_image")}
    if name == "open_fragment":
        frag = fragments.get(str(action.get("fragment_id"))) or {}
        return {
            "tool": "open_fragment",
            "fragment_id": action.get("fragment_id"),
            "fragment_type": frag.get("fragment_type"),
            "display_text": frag.get("display_text"),
            "source_file": frag.get("source_file"),
            "page_num": frag.get("page_num"),
        }
    if name == "retrieve_fragments":
        return {
            "tool": "retrieve_fragments",
            "query": action.get("query"),
            "scope": action.get("scope"),
            "results": [
                {
                    "fragment_id": frag["fragment_id"],
                    "fragment_type": frag.get("fragment_type"),
                    "display_text": v11.truncate(frag.get("display_text") or "", 220),
                }
                for frag in task.get("fragments") or []
                if frag.get("fragment_type") in {"same_page_body", "nearby_page_body", "same_doc_kb", "wrong_target_caption"}
            ][:5],
        }
    if name in {"write_field_value", "abstain_field", "finish"}:
        return {"tool": name, "status": "ok", "action": action}
    return {"tool": name}


def write_split_files(out_dir: Path, splits: dict[str, list[dict[str, Any]]], tasks: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    for split in ["train", "val", "test"]:
        write_jsonl(out_dir / "tasks" / f"{split}_tasks.jsonl", [t for t in tasks if t.get("split") == split])
        write_jsonl(out_dir / "sft" / f"{split}.jsonl", [r for r in rows if r.get("split") == split])
    write_jsonl(out_dir / "tasks" / "all_tasks.jsonl", tasks)
    write_jsonl(out_dir / "sft" / "all.jsonl", rows)


def write_review(
    out_dir: Path,
    page_records: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    fragments: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    args: argparse.Namespace,
) -> Path:
    frag_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    labels_by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frag in fragments:
        frag_by_target[frag["target_id"]].append(frag)
    for lab in labels:
        labels_by_target[lab["target_id"]].append(lab)
    all_targets = [t for split in ["train", "val", "test"] for t in splits.get(split, [])]
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for target in all_targets:
        by_type[target.get("trajectory_type")].append(target)
    rng = random.Random(args.seed + 99)
    sample = []
    per_type = max(5, args.review_limit // 4)
    for ttype, rows in by_type.items():
        rng.shuffle(rows)
        sample.extend(rows[:per_type])
    sample = sample[: args.review_limit]
    path = out_dir / "review" / "v1.3人工抽检.md"
    lines = [
        "# v1.3 Remote VLM Evidence SFT 人工抽检",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 样本数：{len(sample)}",
        "",
    ]
    for target in sample:
        overlay = target.get("overlay_image", "")
        crop = target.get("artwork_image", "")
        cap = target.get("caption_image", "")
        lines.extend(
            [
                f"## `{target['target_id']}` {target.get('trajectory_type')} `{target.get('source_file')}` p{target.get('page_num')}",
                "",
                f"- split：`{target.get('split')}`",
                f"- caption：{html.escape(str((target.get('base_fields') or {}).get('caption_text')))}",
                f"- target_bbox_norm1000：`{target.get('target_bbox_norm1000')}`",
                f"- caption_bbox_norm1000：`{target.get('caption_bbox_norm1000')}`",
                "",
            ]
        )
        if overlay:
            lines.append(f"![overlay]({overlay})")
        if crop:
            lines.append(f"![target]({crop})")
        if cap:
            lines.append(f"![caption]({cap})")
        lines.extend(["", "### Fragments", "", "| id | type | text |", "|---|---|---|"])
        for frag in frag_by_target[target["target_id"]]:
            lines.append(f"| `{frag['fragment_id']}` | `{frag.get('fragment_type')}` | {html.escape(v11.truncate(frag.get('display_text') or '', 180)).replace('|','/')} |")
        lines.extend(["", "### Labels", "", "| field | value | label | fragment | reason |", "|---|---|---|---|---|"])
        for lab in labels_by_target[target["target_id"]]:
            if lab.get("label") == "wrong_target" or lab.get("field") in ALL_FIELDS:
                lines.append(f"| `{lab.get('field')}` | {html.escape(str(lab.get('candidate_value'))).replace('|','/')} | `{lab.get('label')}` | `{lab.get('fragment_id')}` | {html.escape(str(lab.get('reason','')).replace('|','/'))} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    docs_path = DOCS_ROOT / "02_指标与数据" / f"{datetime.now().strftime('%Y%m%d_%H%M')}_v1.3RemoteVLMEvidenceSFT人工抽检.md"
    docs_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def build_manifest(
    out_dir: Path,
    args: argparse.Namespace,
    page_records: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    fragments: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    rows: list[dict[str, Any]],
    review_path: Path,
    split_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "dataset_version": "v1.3_remote_vlm_evidence_sft",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(out_dir),
        "args": vars(args),
        "counts": {
            "page_records": len(page_records),
            "vlm_reviewed_pages": len(annotations),
            "vlm_ok_pages": sum(bool(a.get("ok")) for a in annotations),
            "raw_detections": sum(len(a.get("detections") or []) for a in annotations),
            "accepted_targets": len(targets),
            "evidence_fragments": len(fragments),
            "field_support_labels": len(labels),
            "sft_rows": len(rows),
        },
        "splits": {split: len(items) for split, items in splits.items()},
        "split_summary": split_summary,
        "distributions": {
            "trajectory_type_all": dict(Counter(t.get("trajectory_type") for t in targets)),
            "trajectory_type_selected": {split: dict(Counter(t.get("trajectory_type") for t in items)) for split, items in splits.items()},
            "label": dict(Counter(l.get("label") for l in labels)),
            "field_support": dict(Counter(l.get("field") for l in labels if l.get("label") == "support")),
            "action": dict(Counter((r.get("action") or {}).get("action") for r in rows)),
        },
        "artifacts": {
            "page_records": str(out_dir / "page_records.jsonl"),
            "page_level_vlm_annotations": str(out_dir / "page_level_vlm_annotations.jsonl"),
            "figure_targets": str(out_dir / "figure_targets.jsonl"),
            "evidence_fragments": str(out_dir / "evidence_fragments.jsonl"),
            "field_support_labels": str(out_dir / "field_support_labels.jsonl"),
            "train_sft": str(out_dir / "sft" / "train.jsonl"),
            "val_sft": str(out_dir / "sft" / "val.jsonl"),
            "test_sft": str(out_dir / "sft" / "test.jsonl"),
            "review": str(review_path),
        },
    }


def write_report(out_dir: Path, manifest: dict[str, Any]) -> Path:
    path = out_dir / "v1.3构建报告.md"
    lines = [
        "# v1.3 Remote VLM Evidence SFT 构建报告",
        "",
        f"- 输出目录：`{out_dir}`",
        f"- 生成时间：{manifest.get('created_at')}",
        f"- 模型：`{manifest.get('args', {}).get('model')}`",
        "",
        "## 规模",
        "",
    ]
    for key, value in (manifest.get("counts") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Split", ""])
    for key, value in (manifest.get("splits") or {}).items():
        lines.append(f"- {key}: `{value}` targets")
    lines.extend(["", "## 轨迹分布", "", "```json", json.dumps(manifest.get("distributions", {}).get("trajectory_type_selected"), ensure_ascii=False, indent=2), "```", ""])
    lines.extend(["## 文件", ""])
    for key, value in (manifest.get("artifacts") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## 说明", "", "- v1.3 从 raw PDFs 全新构建；旧数据只作为设计参考，不混入本次 train/val/test。", "- 坐标统一 norm1000；无可见图注使用 null。", "- caption 支持 Metadata5 时不训练 retrieve；caption 不支持才使用 retrieve/open fragment。", "- wrong-target negative 保留为强负例。", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# Small helpers


def image_data_url(path: Path, out_dir: Path, max_side: int) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_side > 0:
            image.thumbnail((max_side, max_side))
        sent_size = image.size
        tmp = out_dir / "cache" / f"send_{hashlib.sha1((str(path)+str(max_side)).encode()).hexdigest()[:12]}.jpg"
        image.save(tmp, quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(tmp.read_bytes()).decode("ascii"), sent_size


def parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        raise ValueError("no JSON object found")
    text = re.sub(r",\s*([}\]])", r"\1", match.group(0))
    return json.loads(text)


def normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        out = [max(0, min(1000, int(round(float(v))))) for v in value]
    except Exception:
        return None
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def denorm_bbox(value: list[int], width: int, height: int) -> list[int]:
    return v11.clamp_bbox(
        [
            round(value[0] * width / 1000),
            round(value[1] * height / 1000),
            round(value[2] * width / 1000),
            round(value[3] * height / 1000),
        ],
        width,
        height,
    )


def norm_bbox(value: list[int], width: int, height: int) -> list[int]:
    return [
        round(value[0] * 1000 / width),
        round(value[1] * 1000 / height),
        round(value[2] * 1000 / width),
        round(value[3] * 1000 / height),
    ]


def scale_pdf_bbox(raw_bbox: Any, sx: float, sy: float, width: int, height: int) -> list[int] | None:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    return v11.clamp_bbox(
        [round(float(raw_bbox[0]) * sx), round(float(raw_bbox[1]) * sy), round(float(raw_bbox[2]) * sx), round(float(raw_bbox[3]) * sy)],
        width,
        height,
    )


def normalize_metadata(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return empty_field()
    value = v11.normalize_space(entry.get("value") or "")
    return {
        "value": value,
        "abstain": bool(entry.get("abstain", not bool(value))),
        "source": v11.normalize_space(entry.get("source") or ("unsupported" if not value else "caption_visual")),
        "confidence": v11.safe_float(entry.get("confidence"), 0.0),
        "reason": str(entry.get("reason") or ""),
    }


def strict_collection(entry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return empty_field()
    value = v11.normalize_space(entry.get("value") or "")
    if not value:
        return {**entry, "value": "", "abstain": True, "source": "unsupported"}
    if BAD_COLLECTION_RE.search(value):
        return {**empty_field(), "reason": "只有入藏方式/编号，没有明确收藏机构名。"}
    return entry


def empty_field() -> dict[str, Any]:
    return {"value": "", "abstain": True, "source": "unsupported", "confidence": 0.0, "reason": ""}


def infer_title(text: str) -> str:
    m = TITLE_RE.search(text or "")
    return v11.normalize_space(m.group(1)) if m else ""


def build_body_fragment_text(target: dict[str, Any], blocks: list[dict[str, Any]]) -> str:
    title = (target.get("base_fields") or {}).get("depicted_work_title") or ""
    caption = (target.get("base_fields") or {}).get("caption_text") or ""
    metadata = target.get("metadata_fields") or {}
    terms = [title, caption]
    for match in FIG_RE.findall(caption or "")[:2]:
        if isinstance(match, tuple):
            terms.extend(part for part in match if part)
        else:
            terms.append(match)
    for entry in metadata.values():
        if not isinstance(entry, dict) or entry.get("abstain"):
            continue
        value = str(entry.get("value") or "")
        if value:
            terms.append(value)
        reason = str(entry.get("reason") or "")
        terms.extend(re.findall(r"[《“'‘]([^》”'’]{2,80})[》”'’]", reason))

    def compact(text: str) -> str:
        return re.sub(r"\s+", "", str(text or ""))

    compact_terms = [compact(t) for t in terms if compact(t)]
    ordered = sorted(
        blocks,
        key=lambda b: (
            (b.get("bbox_norm1000") or [0, 0, 0, 0])[1],
            (b.get("bbox_norm1000") or [0, 0, 0, 0])[0],
        ),
    )
    texts = [b.get("text") or "" for b in ordered if b.get("text")]
    candidates: list[str] = []
    for i, text in enumerate(texts):
        candidates.append(text)
        if i + 1 < len(texts):
            candidates.append(f"{text} {texts[i + 1]}")
        if i + 2 < len(texts):
            candidates.append(f"{text} {texts[i + 1]} {texts[i + 2]}")

    scored = []
    for text in candidates:
        ctext = compact(text)
        score = 0
        for term in compact_terms:
            if term and term in ctext:
                score += 10 if term == compact(title) else 4
        score += len(METADATA_HINT_RE.findall(text)) * 2
        score += len(LANDSCAPE_RE.findall(text))
        if score > 0:
            scored.append((score, text))
    scored.sort(reverse=True, key=lambda x: x[0])
    return v11.truncate(" ".join(text for _, text in scored[:4]), 900)


def choose_wrong_target_fragment(target: dict[str, Any], siblings: list[dict[str, Any]]) -> str:
    own = (target.get("base_fields") or {}).get("caption_text") or ""
    for det in siblings:
        text = det.get("caption_text") or ""
        if text and text != own and (det.get("depicted_work_title") or "") != (target.get("base_fields") or {}).get("depicted_work_title"):
            return text
    return ""


def extract_wrong_value(text: str, field: str) -> str:
    if field == "depicted_work_title":
        return infer_title(text)
    if field == "caption_text":
        return text
    if field == "dimensions":
        m = re.search(r"\d+(?:\.\d+)?\s*(?:×|x|X|\*)\s*\d+(?:\.\d+)?\s*(?:cm|厘米|公分)?", text or "")
        return m.group(0) if m else ""
    if field in {"creator_or_attribution", "creation_period_or_dynasty", "collection_institution", "medium_material"}:
        return ""
    return ""


def label(tid: str, field: str, value: Any, frag_id: str | None, lab: str, judge: str, conf: Any, reason: str) -> dict[str, Any]:
    return {
        "target_id": tid,
        "field": field,
        "candidate_value": value,
        "normalized_value": value,
        "fragment_id": frag_id,
        "label": lab,
        "judge_source": judge,
        "judge_confidence": v11.safe_float(conf, 0.0),
        "reason": reason,
    }


def build_gold_fields(labels: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for lab in labels:
        if lab.get("label") == "support" and lab.get("field") not in out:
            out[lab["field"]] = {"value": lab.get("candidate_value"), "evidence_ids": [lab.get("fragment_id")]}
    for field in ALL_FIELDS:
        out.setdefault(field, {"abstain": True})
    return out


def write_field_action(lab: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "write_field_value",
        "field": lab.get("field"),
        "value": lab.get("candidate_value"),
        "evidence_ids": [lab.get("fragment_id")] if lab.get("fragment_id") else [],
        "confidence": lab.get("judge_confidence", 0.75),
    }


def build_retrieve_query(task: dict[str, Any], fields: list[str]) -> str:
    gold = task.get("gold_fields") or {}
    title = (gold.get("depicted_work_title") or {}).get("value") or ""
    caption = (gold.get("caption_text") or {}).get("value") or ""
    return v11.truncate(" ".join([str(title), str(caption), " ".join(fields), str(task.get("source_file"))]), 400)


def phase_hint_v13(action: dict[str, Any]) -> str:
    return {
        "inspect_page": "先查看页面和可见 fragment，不要直接写字段。",
        "localize_target": "确认目标图和图注 bbox，坐标使用 norm1000。",
        "crop_target": "裁剪目标图，用于 object_type/image_scope。",
        "open_fragment": "打开 local caption/local visual/body/wrong-target fragment。",
        "retrieve_fragments": "只有 caption 不支持 Metadata5 或需要排除 wrong-target 时才 retrieve。",
        "write_field_value": "写一个字段，必须引用支持 fragment_id。",
        "abstain_field": "证据不足或只有 wrong-target 证据时 abstain。",
        "finish": "所有字段写入或 abstain 后结束。",
    }.get(action.get("action"), "")


def simplify_results_v13(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for result in results[-8:]:
        item = {k: result.get(k) for k in ["tool", "fragment_id", "fragment_type", "page_id", "status"] if k in result}
        if result.get("display_text"):
            item["display_text"] = v11.truncate(result.get("display_text"), 220)
        if result.get("results"):
            item["results"] = result.get("results")[:4]
        if result.get("target_bbox_norm1000"):
            item["target_bbox_norm1000"] = result.get("target_bbox_norm1000")
        if result.get("caption_bbox_norm1000") is not None:
            item["caption_bbox_norm1000"] = result.get("caption_bbox_norm1000")
        out.append(item)
    return out


def build_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in row.get("images") or []:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": row.get("prompt_text") or ""})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": json.dumps(row.get("action") or {}, ensure_ascii=False, separators=(",", ":"))},
    ]


def dedupe(values: list[Any]) -> list[Any]:
    out = []
    seen = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
