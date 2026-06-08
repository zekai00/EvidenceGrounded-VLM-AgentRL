#!/usr/bin/env python3
"""Build v0.8 page-capped document-split AgentBench data.

Each PDF page contributes at most one task. Splits are assigned by source PDF,
so no PDF appears in more than one of train/val/test.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_agentbench_v0_4_1_claim_schema import make_sft_row  # noqa: E402
from evidence_agent_env.actions import bbox_iou  # noqa: E402
from evidence_agent_env.data import EvidenceIndex, read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig  # noqa: E402
from evidence_agent_env.tools.claim_tools import apply_claim_write, claim_state, claim_write_result  # noqa: E402
from evidence_agent_env.tools.crop import crop_image, image_size  # noqa: E402


DEFAULT_RAW_PDFS = Path("/root/datasets/chinese_landscape_authority_corpus")
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
CORE5_FIELDS = ["caption_text", "image_scope", "depicted_work_title", "displayed_region", "object_type"]
DYNASTY_WORDS = [
    "先秦",
    "秦",
    "汉",
    "魏晋",
    "东晋",
    "南北朝",
    "隋",
    "唐",
    "五代",
    "北宋",
    "南宋",
    "宋",
    "元",
    "明",
    "清",
    "近现代",
    "现代",
    "当代",
]


@dataclass(frozen=True)
class PageCandidate:
    source_file: str
    source_stem: str
    source_path: Path
    page: int
    page_count: int
    bbox_pt: tuple[float, float, float, float]
    image_bbox: list[int]
    area_ratio: float
    caption_bbox: list[int] | None
    caption_text: str
    caption_score: float
    page_width: int
    page_height: int
    source_meta: dict[str, Any]
    image_blocks: tuple[dict[str, Any], ...]
    text_blocks: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pdfs-dir", default=str(DEFAULT_RAW_PDFS))
    parser.add_argument("--evidence-index-dir", default=str(DEFAULT_EVIDENCE_INDEX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-target", type=int, default=2000)
    parser.add_argument("--train-min", type=int, default=1500)
    parser.add_argument("--train-max", type=int, default=2500)
    parser.add_argument("--val-target", type=int, default=200)
    parser.add_argument("--test-target", type=int, default=200)
    parser.add_argument("--page-dpi", type=int, default=150)
    parser.add_argument("--crop-dpi", type=int, default=200)
    parser.add_argument("--min-area-ratio", type=float, default=0.004)
    parser.add_argument("--max-area-ratio", type=float, default=0.82)
    parser.add_argument("--min-width-ratio", type=float, default=0.05)
    parser.add_argument("--min-height-ratio", type=float, default=0.04)
    parser.add_argument("--top-k-regions", type=int, default=10)
    parser.add_argument("--max-doc-pages-train", type=int, default=80)
    parser.add_argument("--max-doc-pages-eval", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--smoke-max-pdfs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(Path(args.output_root))
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    for child in ["pages", "crops", "overlays", "sft", "episodes", "review"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    source_meta = load_source_meta(Path(args.evidence_index_dir))
    candidates, scan_summary = collect_page_candidates(Path(args.raw_pdfs_dir), source_meta, args)
    split_docs = choose_doc_splits(candidates, args, rng)
    selected = select_split_candidates(candidates, split_docs, args, rng)
    (output_dir / "_split_map.json").write_text(json.dumps(split_docs, ensure_ascii=False, indent=2), encoding="utf-8")

    index = EvidenceIndex(args.evidence_index_dir)
    page_cache: dict[tuple[str, int], Path] = {}
    tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, candidate in enumerate(selected):
        try:
            task = build_task(idx, candidate, output_dir, args, index, page_cache)
            episode = build_episode(task)
            rows = build_sft_rows(task, episode["actions"], output_dir, args)
            tasks.append(task)
            episodes.append(episode)
            sft_rows.extend(rows)
            quality_rows.append(task_quality(task))
        except Exception as exc:
            errors.append(
                {
                    "source_file": candidate.source_file,
                    "page": candidate.page,
                    "image_bbox": candidate.image_bbox,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_outputs(output_dir, tasks, episodes, sft_rows)
    quality = summarize(args, output_dir, candidates, selected, tasks, sft_rows, quality_rows, scan_summary, errors)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.8_page_capped_docsplit_inspect_crop_core5",
        "builder": "scripts/build_agentbench_v0_8_page_capped_docsplit.py",
        "raw_pdfs_dir": str(args.raw_pdfs_dir),
        "evidence_index_dir": str(args.evidence_index_dir),
        "output_dir": str(output_dir),
        "page_cap_policy": "strict: at most one selected task per (source_file,page) in every split",
        "split_policy": "document-level split: each source_file appears in exactly one split",
        "target_claim_fields": CORE5_FIELDS,
        "args": vars(args),
        "quality": quality,
        "files": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
            "oracle_episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
            "sft_train": str(output_dir / "sft" / "train.jsonl"),
            "sft_val": str(output_dir / "sft" / "val.jsonl"),
            "sft_test": str(output_dir / "sft" / "test.jsonl"),
            "review_html": str(output_dir / "review" / "review.html"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    write_review_html(output_dir / "review" / "review.html", tasks[:120])
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def load_source_meta(evidence_index_dir: Path) -> dict[str, dict[str, Any]]:
    path = evidence_index_dir / "authority_sources.jsonl"
    meta: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return meta
    for row in read_jsonl(path):
        filename = str(row.get("filename") or Path(str(row.get("local_path") or "")).name)
        if filename:
            meta[filename] = row
    return meta


def collect_page_candidates(
    raw_pdfs_dir: Path,
    source_meta: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[PageCandidate], dict[str, Any]]:
    candidates: list[PageCandidate] = []
    pdfs = sorted(raw_pdfs_dir.rglob("*.pdf"))
    if args.smoke_max_pdfs:
        pdfs = pdfs[: args.smoke_max_pdfs]
    errors: list[dict[str, str]] = []
    docs_seen = 0
    pages_seen = 0
    pages_with_image_blocks = 0
    for pdf_path in pdfs:
        try:
            with fitz.open(pdf_path) as doc:
                docs_seen += 1
                if docs_seen % 10 == 0:
                    print(
                        f"[scan] pdf {docs_seen}/{len(pdfs)} pages_seen={pages_seen} candidates={len(candidates)} current={pdf_path.name}",
                        file=sys.stderr,
                        flush=True,
                    )
                page_count = len(doc)
                for page_index, page in enumerate(doc):
                    pages_seen += 1
                    page_candidate = best_candidate_for_page(
                        pdf_path,
                        page,
                        page_index + 1,
                        page_count,
                        source_meta.get(pdf_path.name, {}),
                        args,
                    )
                    if page_candidate is None:
                        continue
                    pages_with_image_blocks += 1
                    candidates.append(page_candidate)
        except Exception as exc:
            errors.append({"source_path": str(pdf_path), "error": f"{type(exc).__name__}: {exc}"})
    return candidates, {
        "pdfs_seen": docs_seen,
        "pages_seen": pages_seen,
        "candidate_pages": len(candidates),
        "pages_with_selected_image_block": pages_with_image_blocks,
        "scan_error_count": len(errors),
        "scan_errors": errors[:30],
    }


def best_candidate_for_page(
    pdf_path: Path,
    page: fitz.Page,
    page_num: int,
    page_count: int,
    meta: dict[str, Any],
    args: argparse.Namespace,
) -> PageCandidate | None:
    rect = page.rect
    if rect.width <= 0 or rect.height <= 0:
        return None
    text_blocks = collect_text_blocks(page, rect)
    image_blocks: list[dict[str, Any]] = []
    seen_boxes: set[tuple[int, int, int, int]] = set()
    for block_index, bbox_pt in enumerate(iter_image_rects(page)):
        x0, y0, x1, y1 = bbox_pt
        width = max(0.0, x1 - x0)
        height = max(0.0, y1 - y0)
        area_ratio = (width * height) / max(1.0, rect.width * rect.height)
        if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
            continue
        if width / rect.width < args.min_width_ratio or height / rect.height < args.min_height_ratio:
            continue
        bbox_px = scale_bbox_to_rendered(bbox_pt, rect, args.page_dpi)
        if area_px(bbox_px) <= 0:
            continue
        coarse = tuple(round(value / 4) * 4 for value in bbox_px)
        if coarse in seen_boxes:
            continue
        seen_boxes.add(coarse)
        caption = best_caption_for_image(bbox_px, text_blocks)
        image_blocks.append(
            {
                "bbox_pt": bbox_pt,
                "bbox": bbox_px,
                "area_ratio": round(float(area_ratio), 6),
                "caption_bbox": caption.get("bbox"),
                "caption_text": caption.get("text", ""),
                "caption_score": float(caption.get("score", 0.0)),
                "block_index": block_index,
            }
        )
    if not image_blocks:
        return None
    image_blocks.sort(key=lambda item: page_target_score(item), reverse=True)
    best = image_blocks[0]
    page_width = int(round(rect.width * args.page_dpi / 72.0))
    page_height = int(round(rect.height * args.page_dpi / 72.0))
    return PageCandidate(
        source_file=pdf_path.name,
        source_stem=pdf_path.stem,
        source_path=pdf_path,
        page=page_num,
        page_count=page_count,
        bbox_pt=tuple(float(v) for v in best["bbox_pt"]),
        image_bbox=list(best["bbox"]),
        area_ratio=float(best["area_ratio"]),
        caption_bbox=best.get("caption_bbox"),
        caption_text=str(best.get("caption_text") or ""),
        caption_score=float(best.get("caption_score") or 0.0),
        page_width=page_width,
        page_height=page_height,
        source_meta=meta,
        image_blocks=tuple(image_blocks[:12]),
        text_blocks=tuple(text_blocks[:20]),
    )


def iter_image_rects(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    seen_xrefs: set[int] = set()
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    for image in images:
        try:
            xref = int(image[0])
        except Exception:
            continue
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            placements = page.get_image_rects(xref)
        except Exception:
            placements = []
        for item in placements:
            rects.append((float(item.x0), float(item.y0), float(item.x1), float(item.y1)))
    if rects:
        return rects
    # Fallback for unusual inline-image pages.
    try:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") == 1:
                rects.append(tuple(float(v) for v in block.get("bbox", [0, 0, 0, 0])))
    except Exception:
        pass
    return rects


def collect_text_blocks(page: fitz.Page, rect: fitz.Rect) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    scale = 150.0 / 72.0
    for block_index, block in enumerate(page.get_text("blocks")):
        if len(block) < 5:
            continue
        text = clean_text(str(block[4]))
        if not text:
            continue
        bbox_pt = [float(block[0]), float(block[1]), float(block[2]), float(block[3])]
        blocks.append(
            {
                "block_index": block_index,
                "bbox": [
                    int(round(bbox_pt[0] * scale)),
                    int(round(bbox_pt[1] * scale)),
                    int(round(bbox_pt[2] * scale)),
                    int(round(bbox_pt[3] * scale)),
                ],
                "text": text,
            }
        )
    return blocks


def best_caption_for_image(image_bbox: list[int], text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for block in text_blocks:
        score = caption_score(image_bbox, block["bbox"], block["text"])
        if best is None or score > best["score"]:
            best = {"bbox": block["bbox"], "text": block["text"], "score": score}
    if best is None or best["score"] < -8.0:
        return {"bbox": None, "text": "", "score": -999.0}
    return best


def caption_score(image_bbox: list[int], text_bbox: list[int], text: str) -> float:
    ix1, iy1, ix2, iy2 = image_bbox
    tx1, ty1, tx2, ty2 = text_bbox
    fig_width = max(1, ix2 - ix1)
    horizontal_overlap = max(0, min(ix2, tx2) - max(ix1, tx1)) / max(1, min(fig_width, tx2 - tx1))
    gap_below = ty1 - iy2
    gap_above = iy1 - ty2
    near_gap = min(abs(gap_below), abs(gap_above))
    score = horizontal_overlap * 2.0 - near_gap / 120.0
    if 0 <= gap_below <= 180:
        score += 3.5
    if 0 <= gap_above <= 120:
        score += 1.2
    normalized = normalize_spaces(text)
    if caption_like(normalized[:80]):
        score += 4.0
    if "《" in normalized and "》" in normalized:
        score += 1.2
    if "山水" in normalized or "画" in normalized:
        score += 0.8
    if len(normalized) > 260:
        score -= 2.0
    if horizontal_overlap < 0.08:
        score -= 2.0
    return score


def page_target_score(item: dict[str, Any]) -> float:
    caption_bonus = max(-2.0, min(8.0, float(item.get("caption_score") or 0.0)))
    area = float(item.get("area_ratio") or 0.0)
    size_bonus = 2.0 * min(1.0, area / 0.18)
    full_page_penalty = 2.5 if area > 0.65 else 0.0
    return caption_bonus + size_bonus - full_page_penalty


def choose_doc_splits(
    candidates: list[PageCandidate],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, str]:
    by_doc: dict[str, list[PageCandidate]] = defaultdict(list)
    for item in candidates:
        by_doc[item.source_file].append(item)
    split_docs: dict[str, str] = {}
    available = set(by_doc)
    for split, target in [("val", args.val_target), ("test", args.test_target)]:
        picked = pick_eval_docs(by_doc, available, target, args.max_doc_pages_eval, rng)
        for doc in picked:
            split_docs[doc] = split
        available.difference_update(picked)
    for doc in sorted(available):
        if doc not in split_docs:
            split_docs[doc] = "train"
    return split_docs


def pick_eval_docs(
    by_doc: dict[str, list[PageCandidate]],
    available: set[str],
    target: int,
    per_doc_cap: int,
    rng: random.Random,
) -> list[str]:
    """Pick eval docs while minimizing wasted pages from eval-only documents."""

    docs = list(available)
    rng.shuffle(docs)
    # Avoid assigning huge documents to eval when only a small capped sample from
    # them can be used; otherwise most pages from that document become unusable
    # for train because the split is document-level.
    docs.sort(
        key=lambda doc: (
            max(0, len(by_doc[doc]) - per_doc_cap),
            min(len(by_doc[doc]), per_doc_cap),
            doc,
        )
    )
    picked: list[str] = []
    current = 0
    for doc in docs:
        picked.append(doc)
        current += min(len(by_doc[doc]), per_doc_cap)
        if current >= target:
            break
    return picked


def select_split_candidates(
    candidates: list[PageCandidate],
    split_docs: dict[str, str],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[PageCandidate]:
    by_split_doc: dict[str, dict[str, list[PageCandidate]]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        by_split_doc[split_docs[item.source_file]][item.source_file].append(item)

    selected: list[PageCandidate] = []
    selected.extend(
        balanced_sample_by_doc(by_split_doc["val"], args.val_target, args.max_doc_pages_eval, rng)
    )
    selected.extend(
        balanced_sample_by_doc(by_split_doc["test"], args.test_target, args.max_doc_pages_eval, rng)
    )
    train_target = max(args.train_min, min(args.train_target, args.train_max))
    train_rows = balanced_sample_by_doc(by_split_doc["train"], train_target, args.max_doc_pages_train, rng)
    if len(train_rows) < args.train_min:
        train_rows = balanced_sample_by_doc(by_split_doc["train"], args.train_min, 10**9, rng)
    selected.extend(train_rows[: args.train_max])
    selected.sort(key=lambda item: ({"train": 0, "val": 1, "test": 2}[split_docs[item.source_file]], item.source_file, item.page))
    return selected


def balanced_sample_by_doc(
    by_doc: dict[str, list[PageCandidate]],
    target: int,
    per_doc_cap: int,
    rng: random.Random,
) -> list[PageCandidate]:
    buckets: dict[str, list[PageCandidate]] = {}
    for doc, rows in by_doc.items():
        ordered = sorted(rows, key=lambda item: (-page_target_score_from_candidate(item), item.page))
        buckets[doc] = ordered[: min(len(ordered), per_doc_cap)]
    docs = sorted(buckets)
    rng.shuffle(docs)
    selected: list[PageCandidate] = []
    cursor = 0
    while len(selected) < target and docs:
        doc = docs[cursor % len(docs)]
        bucket = buckets[doc]
        if bucket:
            selected.append(bucket.pop(0))
        docs = [item for item in docs if buckets[item]]
        cursor += 1
    return selected


def page_target_score_from_candidate(item: PageCandidate) -> float:
    return page_target_score({"caption_score": item.caption_score, "area_ratio": item.area_ratio})


def build_task(
    index: int,
    candidate: PageCandidate,
    output_dir: Path,
    args: argparse.Namespace,
    index_backend: EvidenceIndex,
    page_cache: dict[tuple[str, int], Path],
) -> dict[str, Any]:
    split = candidate_split(candidate, output_dir)
    task_id = f"egva_v0_8_pagecap_{index:06d}"
    page_image = render_page(candidate, output_dir / "pages", args.page_dpi, page_cache)
    crop_result = render_gold_crop(candidate, output_dir / "crops", args.crop_dpi)
    caption_text = normalize_spaces(candidate.caption_text)
    caption_evidence = build_local_caption_evidence(task_id, candidate, caption_text)
    query = evidence_query(candidate, caption_text)
    retrieved = index_backend.search(query, "same_document", {"source_file": candidate.source_file, "page": candidate.page}, top_k=8)
    claims = build_core5_claims(candidate, caption_evidence, retrieved)
    region_candidates = build_region_candidates(candidate, task_id, args.top_k_regions)
    overlay = draw_overlay(page_image, candidate.image_bbox, candidate.caption_bbox, output_dir / "overlays" / f"{task_id}.jpg")
    evidence_ids = sorted({eid for claim in claims for eid in claim.get("evidence_ids", [])})
    candidate_evidence_ids = dedupe(
        [caption_evidence["evidence_id"]]
        + [str(item.get("evidence_id")) for item in retrieved if item.get("evidence_id")]
    )
    task = {
        "task_id": task_id,
        "source_task_id": None,
        "split": split,
        "dataset_version": "v0.8_page_capped_docsplit_inspect_crop_core5",
        "tool_schema_version": "v0.8_inspect_crop_core5",
        "task_type": "evidence_grounded_pdf_figure_claim",
        "runtime_mode": "page_capped_no_highlight_inspect_crop",
        "source_type": "pdf_page",
        "source_file": candidate.source_file,
        "source_stem": candidate.source_stem,
        "source_path": str(candidate.source_path),
        "page": candidate.page,
        "page_count": candidate.page_count,
        "page_image": str(page_image),
        "artwork_image": str(crop_result["crop_path"]),
        "overlay_image": str(overlay),
        "goal": "Inspect the PDF page, crop the target Chinese landscape figure, retrieve/open evidence, and write Core5 evidence-grounded claims.",
        "available_tools": [
            "inspect_page",
            "crop_target",
            "select_evidence",
            "retrieve_evidence",
            "open_evidence",
            "write_claim",
            "abstain_claim",
            "write_claims_chunk",
            "finish",
        ],
        "region_candidates": region_candidates,
        "local_evidence": [caption_evidence],
        "gold": {
            "image_bbox": candidate.image_bbox,
            "target_region_id": target_region_id(region_candidates),
            "target_region_bbox": candidate.image_bbox,
            "target_region_iou": 1.0,
            "caption_bbox": candidate.caption_bbox,
            "caption_text": caption_text,
            "claims": claims,
            "claim_schema_fields": CORE5_FIELDS,
            "target_claim_fields": CORE5_FIELDS,
            "evidence_ids": evidence_ids,
            "candidate_evidence_ids": candidate_evidence_ids,
            "evidence_query": query,
            "_retrieval_results": retrieved,
            "auto_label": True,
            "needs_review": True,
            "label_source": "v0_8_page_capped_pdf_image_block_caption_heuristic",
        },
        "candidate_meta": {
            "source": "pdf_image_block",
            "area_ratio": candidate.area_ratio,
            "caption_score": candidate.caption_score,
            "page_cap": "one_task_per_source_page",
            "source_authority_level": candidate.source_meta.get("authority_level"),
            "source_type": candidate.source_meta.get("source_type"),
            "category": candidate.source_meta.get("category"),
        },
        "candidate_augmentation": {"variant": 0},
    }
    return task


def candidate_split(candidate: PageCandidate, output_dir: Path) -> str:
    # Filled after selection by reading the temporary sidecar written below.
    split_map_path = output_dir / "_split_map.json"
    if split_map_path.exists():
        split_map = json.loads(split_map_path.read_text(encoding="utf-8"))
        return str(split_map[candidate.source_file])
    raise RuntimeError("_split_map.json missing")


def build_local_caption_evidence(task_id: str, candidate: PageCandidate, caption_text: str) -> dict[str, Any]:
    text = caption_text or f"{candidate.source_file} 第 {candidate.page} 页目标图像"
    return {
        "evidence_id": f"local_caption_{task_id}",
        "source_file": candidate.source_file,
        "page_start": candidate.page,
        "page_end": candidate.page,
        "authority_level": candidate.source_meta.get("authority_level", "B"),
        "citation_level": "page_caption_region" if caption_text else "page_image_region",
        "source_quality": "pdf_text_block_caption_heuristic" if caption_text else "pdf_image_block_without_caption",
        "display_snippet": text[:500],
        "bbox": candidate.caption_bbox,
    }


def build_core5_claims(
    candidate: PageCandidate,
    local_caption: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    caption = normalize_spaces(candidate.caption_text)
    local_id = local_caption["evidence_id"]
    retrieved_ids = [str(item.get("evidence_id")) for item in retrieved if item.get("evidence_id")]
    candidate_ids = dedupe([local_id] + retrieved_ids[:5])
    claims: list[dict[str, Any]] = []
    if caption:
        claims.append(supported_claim("caption_text", caption, [local_id], candidate_ids, 0.88))
    else:
        claims.append(abstain_claim("caption_text", "页面文本层未给出可靠图注"))

    scope = infer_image_scope(caption)
    if scope:
        claims.append(supported_claim("image_scope", scope, [local_id], candidate_ids, 0.72))
    else:
        claims.append(abstain_claim("image_scope", "图注未明确说明全幅、局部或卷册页片段"))

    title = extract_title(caption)
    if title:
        claims.append(supported_claim("depicted_work_title", title, [local_id], candidate_ids, 0.82))
    else:
        claims.append(abstain_claim("depicted_work_title", "图注未明确给出《作品名》"))

    displayed_region = infer_displayed_region(caption, scope)
    if displayed_region:
        claims.append(supported_claim("displayed_region", displayed_region, [local_id], candidate_ids, 0.68))
    else:
        claims.append(abstain_claim("displayed_region", "图注未明确说明当前展示的是作品哪个区域"))

    object_type = infer_object_type(caption)
    if object_type:
        claims.append(supported_claim("object_type", object_type, [local_id], candidate_ids, 0.76, candidate.image_bbox))
    else:
        claims.append(abstain_claim("object_type", "图注未明确对象类型"))
    return claims


def supported_claim(
    field: str,
    value: Any,
    evidence_ids: list[str],
    candidate_ids: list[str],
    confidence: float,
    visual_bbox: list[int] | None = None,
) -> dict[str, Any]:
    item = {
        "claim_id": field,
        "field": field,
        "value": value,
        "abstain": False,
        "evidence_ids": evidence_ids,
        "candidate_evidence_ids": candidate_ids,
        "support_type": "page_caption_text",
        "confidence": confidence,
    }
    if visual_bbox is not None:
        item["visual_bbox"] = visual_bbox
    return item


def abstain_claim(field: str, reason: str) -> dict[str, Any]:
    return {
        "claim_id": field,
        "field": field,
        "value": None,
        "abstain": True,
        "reason": reason,
        "evidence_ids": [],
        "candidate_evidence_ids": [],
        "support_type": "text",
    }


def build_region_candidates(candidate: PageCandidate, task_id: str, top_k: int) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    caption_id = f"local_caption_{task_id}"
    target = {
        "bbox": candidate.image_bbox,
        "source": "pdf_image_block",
        "type": "figure_candidate",
        "score": round(page_target_score_from_candidate(candidate), 4),
        "hint": "PDF 原生 image block；v0.8 每页唯一目标候选",
        "caption_evidence_id": caption_id,
        "caption_hint": candidate.caption_text[:180] if candidate.caption_text else "",
        "linked_caption_text": candidate.caption_text[:240] if candidate.caption_text else "",
        "linked_caption_region_id": "r_caption_0" if candidate.caption_bbox else None,
        "caption_link_score": round(float(candidate.caption_score), 3),
        "target_caption_match_score": 3.0 if candidate.caption_text else 0.5,
        "target_caption_match_reason": "page_selected_best_caption_candidate",
        "target_region_rank": 1,
        "target_region_sort_score": round(30.0 + max(0.0, candidate.caption_score), 3),
        "target_iou": 1.0,
        "is_target": True,
    }
    regions.append(target)
    for image in candidate.image_blocks:
        bbox = list(image.get("bbox") or [])
        if bbox == candidate.image_bbox:
            continue
        regions.append(
            {
                "bbox": bbox,
                "source": "pdf_image_block",
                "type": "figure_candidate",
                "score": round(page_target_score(image), 4),
                "hint": "同页其他 PDF image block 候选",
                "caption_hint": str(image.get("caption_text") or "")[:160],
                "target_iou": bbox_iou(candidate.image_bbox, bbox),
                "is_target": False,
            }
        )
    if candidate.caption_bbox:
        regions.append(
            {
                "bbox": candidate.caption_bbox,
                "source": "pdf_text_block",
                "type": "text_or_caption_candidate",
                "score": round(float(candidate.caption_score), 4),
                "nearby_text": candidate.caption_text[:220],
                "caption_evidence_id": caption_id,
                "caption_hint": candidate.caption_text[:220],
                "hint": "与目标图像最邻近的图注候选",
                "target_iou": 0.0,
                "is_target": False,
            }
        )
    for block in candidate.text_blocks[:4]:
        bbox = list(block.get("bbox") or [])
        if candidate.caption_bbox and bbox == candidate.caption_bbox:
            continue
        regions.append(
            {
                "bbox": bbox,
                "source": "pdf_text_block",
                "type": "text_or_caption_candidate",
                "score": 0.1,
                "nearby_text": str(block.get("text") or "")[:160],
                "hint": "同页文本/图注干扰候选",
                "target_iou": 0.0,
                "is_target": False,
            }
        )
    regions.extend(grid_distractors(candidate.page_width, candidate.page_height, candidate.image_bbox))
    selected = regions[: max(1, top_k)]
    for idx, item in enumerate(selected):
        item["region_id"] = f"r{idx}"
        item["gold_iou"] = bbox_iou(candidate.image_bbox, item.get("bbox"))
    return selected


def target_region_id(regions: list[dict[str, Any]]) -> str:
    for item in regions:
        if item.get("is_target"):
            return str(item.get("region_id"))
    return str((regions[0] if regions else {}).get("region_id", "r0"))


def build_episode(task: dict[str, Any]) -> dict[str, Any]:
    gold = task["gold"]
    local_id = (task.get("local_evidence") or [{}])[0].get("evidence_id")
    retrieved_id = first_retrieved_id(gold)
    actions: list[dict[str, Any]] = [
        {"action": "inspect_page", "top_k": len(task.get("region_candidates") or [])},
        {"action": "crop_target", "region_id": gold["target_region_id"]},
    ]
    if local_id:
        actions.append({"action": "select_evidence", "evidence_ids": [local_id]})
        actions.append({"action": "open_evidence", "evidence_id": local_id})
    actions.append({"action": "retrieve_evidence", "query": gold.get("evidence_query", ""), "scope": "same_document", "top_k": 5})
    if retrieved_id:
        actions.append({"action": "open_evidence", "evidence_id": retrieved_id})
    current_fields = set()
    for claim in gold.get("claims") or []:
        field = claim.get("field")
        if field in current_fields:
            continue
        current_fields.add(field)
        if claim.get("abstain"):
            actions.append({"action": "write_claims_chunk", "claims": [], "abstains": [{"field": field, "reason": claim.get("reason", "")}]})
        else:
            actions.append(
                {
                    "action": "write_claims_chunk",
                    "claims": [
                        {
                            "field": field,
                            "value": claim.get("value"),
                            "evidence_ids": claim.get("evidence_ids") or [],
                            "visual_bbox": claim.get("visual_bbox"),
                            "confidence": claim.get("confidence", 0.75),
                        }
                    ],
                    "abstains": [],
                }
            )
    actions.append({"action": "finish", "status": "done"})
    return {
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "split": task.get("split"),
        "variant": 0,
        "actions": actions,
    }


def first_retrieved_id(gold: dict[str, Any]) -> str | None:
    for evidence_id in gold.get("candidate_evidence_ids") or []:
        evidence_id = str(evidence_id)
        if not evidence_id.startswith("local_caption_"):
            return evidence_id
    return None


def build_sft_rows(
    task: dict[str, Any],
    actions: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    prompt_config = PromptConfig(
        tool_schema="inspect_crop",
        coordinate_info=True,
        max_history_actions=6,
        max_tool_results=5,
        max_evidence_per_result=2,
        snippet_chars=140,
        max_text_chars=14000,
        head_text_chars=4000,
        compact_claim_state=True,
        region_selection_hint=True,
        strict_claim_phase_hint=True,
    )
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    draft_claims: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    images = [task["page_image"]]
    for step, action in enumerate(actions):
        state = {
            "task_id": task["task_id"],
            "split": task.get("split"),
            "step": step,
            "history": copy.deepcopy(history),
            "tool_results": copy.deepcopy(tool_results),
            "draft_claims": copy.deepcopy(draft_claims),
            "selected_evidence_ids": copy.deepcopy(selected_ids),
            "claim_state": claim_state(draft_claims, target_fields=CORE5_FIELDS),
            "images": copy.deepcopy(images),
        }
        row = make_sft_row(task, state, action, prompt_config, step)
        row["tool_schema_version"] = "v0.8_inspect_crop_core5"
        row["label_source"] = "v0_8_page_capped_docsplit_core5_sft"
        row["claim_state"] = claim_state(draft_claims, target_fields=CORE5_FIELDS)
        rows.append(row)

        result = result_for_action(task, action, output_dir, step, selected_ids, draft_claims)
        history.append(copy.deepcopy(action))
        tool_results.append(result)
        if action.get("action") == "crop_target" and result.get("crop_path"):
            images = [task["page_image"], result["crop_path"]]
        if action.get("action") == "select_evidence":
            for eid in action.get("evidence_ids") or []:
                if eid not in selected_ids:
                    selected_ids.append(str(eid))
        if action.get("action") == "write_claims_chunk":
            draft_claims = apply_claim_write(
                draft_claims,
                claims=action.get("claims") or [],
                abstains=action.get("abstains") or [],
            )
    return rows


def result_for_action(
    task: dict[str, Any],
    action: dict[str, Any],
    output_dir: Path,
    step: int,
    selected_ids: list[str],
    draft_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    name = action.get("action")
    if name == "inspect_page":
        return {
            "tool": "inspect_page",
            "page_image": task.get("page_image"),
            "page_size": image_size(task["page_image"]),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "regions": public_region_candidates(task),
            "layout_regions": public_region_candidates(task),
        }
    if name == "crop_target":
        region = next((item for item in task.get("region_candidates") or [] if item.get("region_id") == action.get("region_id")), None)
        bbox = (region or {}).get("bbox") or task["gold"]["image_bbox"]
        out = output_dir / "crops" / f"{task['task_id']}_sft_step{step:02d}.jpg"
        result = crop_image(task["page_image"], bbox, out)
        return {"tool": "crop_target", "region_id": action.get("region_id"), "crop_mode": "region_id", **result, "bbox_iou": bbox_iou(task["gold"]["image_bbox"], result["bbox"])}
    if name == "select_evidence":
        ids = [str(eid) for eid in action.get("evidence_ids") or []]
        return {"tool": "select_evidence", "selected_evidence_ids": ids, "selected_evidence": [open_task_evidence(task, eid) for eid in ids]}
    if name == "open_evidence":
        return open_task_evidence(task, str(action.get("evidence_id")))
    if name == "retrieve_evidence":
        results = task.get("gold", {}).get("_retrieval_results") or []
        hit_ids = sorted({str(item.get("evidence_id")) for item in results} & set(task.get("gold", {}).get("candidate_evidence_ids") or []))
        return {
            "tool": "retrieve_evidence",
            "query": action.get("query"),
            "scope": action.get("scope"),
            "results": results[: int(action.get("top_k") or 5)],
            "hit_evidence_ids": hit_ids,
        }
    if name == "write_claims_chunk":
        next_claims = apply_claim_write(
            draft_claims,
            claims=action.get("claims") or [],
            abstains=action.get("abstains") or [],
        )
        return claim_write_result(
            "write_claims_chunk",
            next_claims,
            claims=action.get("claims") or [],
            abstains=action.get("abstains") or [],
            target_fields=CORE5_FIELDS,
        )
    if name == "finish":
        return {"tool": "finish", "status": action.get("status", "done"), "draft_claims": draft_claims}
    return {"tool": name}


def public_region_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    hidden = {"is_target", "target_iou", "gold_iou", "source_task_id", "source_gold_bbox", "debug_reason"}
    return [{key: value for key, value in item.items() if key not in hidden} for item in task.get("region_candidates") or []]


def open_task_evidence(task: dict[str, Any], evidence_id: str) -> dict[str, Any]:
    for item in task.get("local_evidence") or []:
        if str(item.get("evidence_id")) == evidence_id:
            return {
                "tool": "open_evidence",
                "evidence_id": evidence_id,
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "authority_level": item.get("authority_level"),
                "citation_level": item.get("citation_level"),
                "display_snippet": item.get("display_snippet"),
            }
    for item in task.get("gold", {}).get("_retrieval_results") or []:
        if str(item.get("evidence_id")) == evidence_id:
            return {
                "tool": "open_evidence",
                "evidence_id": evidence_id,
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "authority_level": item.get("authority_level"),
                "citation_level": item.get("citation_level"),
                "display_snippet": item.get("display_snippet"),
            }
    return {"tool": "open_evidence", "evidence_id": evidence_id, "error": "evidence not found in builder cache"}


def render_page(
    candidate: PageCandidate,
    pages_dir: Path,
    dpi: int,
    cache: dict[tuple[str, int], Path],
) -> Path:
    key = (candidate.source_file, candidate.page)
    if key in cache:
        return cache[key]
    out = pages_dir / f"{safe_name(candidate.source_stem)}_p{candidate.page:04d}.png"
    if not out.exists():
        with fitz.open(candidate.source_path) as doc:
            pix = doc[candidate.page - 1].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            pix.save(out)
    cache[key] = out
    return out


def render_gold_crop(candidate: PageCandidate, crops_dir: Path, dpi: int) -> dict[str, Any]:
    out = crops_dir / f"{safe_name(candidate.source_stem)}_p{candidate.page:04d}_gold.jpg"
    if not out.exists():
        with fitz.open(candidate.source_path) as doc:
            pix = doc[candidate.page - 1].get_pixmap(clip=fitz.Rect(*candidate.bbox_pt), dpi=dpi, colorspace=fitz.csRGB)
            pix.save(out)
    return {"crop_path": str(out)}


def draw_overlay(page_image: Path, image_bbox: list[int], caption_bbox: list[int] | None, out: Path) -> Path:
    image = Image.open(page_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle(image_bbox, outline="red", width=4)
    draw.text((image_bbox[0], max(0, image_bbox[1] - 18)), "target_image", fill="red")
    if caption_bbox:
        draw.rectangle(caption_bbox, outline="cyan", width=3)
        draw.text((caption_bbox[0], max(0, caption_bbox[1] - 18)), "caption_candidate", fill="cyan")
    image.save(out, quality=92)
    return out


def write_outputs(output_dir: Path, tasks: list[dict[str, Any]], episodes: list[dict[str, Any]], sft_rows: list[dict[str, Any]]) -> None:
    for task in tasks:
        task.get("gold", {}).pop("_retrieval_results", None)
    write_jsonl(output_dir / "tasks_all.jsonl", tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)
    write_jsonl(output_dir / "sft" / "all.jsonl", sft_rows)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", [task for task in tasks if task.get("split") == split])
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", [ep for ep in episodes if ep.get("split") == split])
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", [row for row in sft_rows if row.get("split") == split])


def task_quality(task: dict[str, Any]) -> dict[str, Any]:
    claims = task.get("gold", {}).get("claims") or []
    non_abstain = [item for item in claims if not item.get("abstain")]
    return {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "has_caption_text": bool(task.get("gold", {}).get("caption_text")),
        "has_caption_bbox": bool(task.get("gold", {}).get("caption_bbox")),
        "non_abstain_core5": len(non_abstain),
        "region_count": len(task.get("region_candidates") or []),
        "target_region_iou": task.get("gold", {}).get("target_region_iou"),
        "local_evidence_count": len(task.get("local_evidence") or []),
        "candidate_evidence_count": len(task.get("gold", {}).get("candidate_evidence_ids") or []),
    }


def summarize(
    args: argparse.Namespace,
    output_dir: Path,
    candidates: list[PageCandidate],
    selected: list[PageCandidate],
    tasks: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    scan_summary: dict[str, Any],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counts = Counter(task.get("split") for task in tasks)
    source_counts = Counter(task.get("source_file") for task in tasks)
    page_keys = [(task.get("source_file"), task.get("page")) for task in tasks]
    sft_split_counts = Counter(row.get("split") for row in sft_rows)
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in sft_rows)
    doc_by_split: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        doc_by_split[str(task.get("split"))].add(str(task.get("source_file")))
    caption_rows = [row for row in quality_rows if row["has_caption_text"]]
    return {
        "created_at": now(),
        "output_dir": str(output_dir),
        "scan_summary": scan_summary,
        "all_candidate_pages": len(candidates),
        "selected_tasks": len(tasks),
        "requested": {
            "train_target": args.train_target,
            "train_min": args.train_min,
            "train_max": args.train_max,
            "val_target": args.val_target,
            "test_target": args.test_target,
        },
        "split_counts": dict(split_counts),
        "doc_counts_by_split": {split: len(docs) for split, docs in doc_by_split.items()},
        "unique_sources": len(source_counts),
        "unique_pages": len(set(page_keys)),
        "page_cap_violations": len(page_keys) - len(set(page_keys)),
        "doc_split_violations": doc_split_violations(tasks),
        "top_sources": dict(source_counts.most_common(20)),
        "caption_text_rate": len(caption_rows) / max(1, len(quality_rows)),
        "caption_bbox_rate": sum(row["has_caption_bbox"] for row in quality_rows) / max(1, len(quality_rows)),
        "mean_non_abstain_core5": sum(row["non_abstain_core5"] for row in quality_rows) / max(1, len(quality_rows)),
        "mean_region_count": sum(row["region_count"] for row in quality_rows) / max(1, len(quality_rows)),
        "mean_candidate_evidence_count": sum(row["candidate_evidence_count"] for row in quality_rows) / max(1, len(quality_rows)),
        "sft_rows_total": len(sft_rows),
        "sft_split_counts": dict(sft_split_counts),
        "sft_action_counts": dict(action_counts),
        "builder_error_count": len(errors),
        "builder_errors": errors[:30],
        "notes": [
            "Gold labels are silver labels from PDF image blocks, nearby caption heuristics, and local caption evidence.",
            "Every selected task has one source page, and duplicate (source_file,page) pairs are treated as hard quality failures.",
            "Validation/test are document-level splits, so they evaluate unseen PDF documents rather than unseen tasks from seen documents.",
        ],
    }


def doc_split_violations(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_doc: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        by_doc[str(task.get("source_file"))].add(str(task.get("split")))
    return {doc: sorted(splits) for doc, splits in by_doc.items() if len(splits) > 1}


def write_review_html(path: Path, tasks: list[dict[str, Any]]) -> None:
    rows = [
        "<html><head><meta charset='utf-8'><title>v0.8 page-capped review</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px}.task{border:1px solid #bbb;margin:14px 0;padding:12px}img{max-width:720px;border:1px solid #ddd}code{white-space:pre-wrap;display:block;background:#f6f6f6;padding:8px}</style>",
        "</head><body><h1>AgentBench v0.8 Page-Capped Review</h1>",
    ]
    for task in tasks:
        rows.append("<div class='task'>")
        rows.append(f"<h2>{task['task_id']} [{task['split']}]</h2>")
        rows.append(f"<p>{task['source_file']} page {task['page']}</p>")
        rows.append(f"<img src='file://{task['overlay_image']}' />")
        rows.append("<code>" + html_escape(json.dumps(task.get("gold", {}), ensure_ascii=False, indent=2)[:3500]) + "</code>")
        rows.append("</div>")
    rows.append("</body></html>")
    path.write_text("\n".join(rows), encoding="utf-8")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    q = manifest["quality"]
    lines = [
        "# AgentBench v0.8 Page-Capped Doc-Split 构建报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 目标",
        "",
        "构建严格每页最多 1 条任务的数据集，并按 PDF 文档切分 train/val/test，避免同一篇文章排版风格泄漏到评测集。",
        "",
        "## 输出位置",
        "",
        "```text",
        manifest["output_dir"],
        "```",
        "",
        "## 规模",
        "",
        f"- scan PDFs：{q['scan_summary']['pdfs_seen']}",
        f"- scan pages：{q['scan_summary']['pages_seen']}",
        f"- all_candidate_pages：{q['all_candidate_pages']}",
        f"- selected_tasks：{q['selected_tasks']}",
        f"- split_counts：`{json.dumps(q['split_counts'], ensure_ascii=False)}`",
        f"- doc_counts_by_split：`{json.dumps(q['doc_counts_by_split'], ensure_ascii=False)}`",
        f"- unique_pages：{q['unique_pages']}",
        f"- unique_sources：{q['unique_sources']}",
        "",
        "## 硬质量检查",
        "",
        f"- page_cap_violations：{q['page_cap_violations']}",
        f"- doc_split_violations：`{json.dumps(q['doc_split_violations'], ensure_ascii=False)}`",
        f"- builder_error_count：{q['builder_error_count']}",
        "",
        "## 标注质量信号",
        "",
        f"- caption_text_rate：{q['caption_text_rate']:.4f}",
        f"- caption_bbox_rate：{q['caption_bbox_rate']:.4f}",
        f"- mean_non_abstain_core5：{q['mean_non_abstain_core5']:.2f}",
        f"- mean_region_count：{q['mean_region_count']:.2f}",
        f"- mean_candidate_evidence_count：{q['mean_candidate_evidence_count']:.2f}",
        "",
        "## SFT 轨迹",
        "",
        f"- sft_rows_total：{q['sft_rows_total']}",
        f"- sft_split_counts：`{json.dumps(q['sft_split_counts'], ensure_ascii=False)}`",
        f"- sft_action_counts：`{json.dumps(q['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 说明",
        "",
        "- 这版是 silver 数据：目标框来自 PDF image block，图注来自邻近文本块规则，claim 主要由图注证据支撑。",
        "- val/test 是文档级 unseen PDF，不再是同一篇文章里的多页或同页多任务。",
        "- 如果后续要作为最终论文/简历主指标，建议再对 val/test 抽样做人审或强 VLM 审核。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def scale_bbox_to_rendered(bbox_pt: tuple[float, float, float, float], rect: fitz.Rect, dpi: int) -> list[int]:
    scale = dpi / 72.0
    width = int(round(rect.width * scale))
    height = int(round(rect.height * scale))
    x0, y0, x1, y1 = bbox_pt
    return [
        max(0, min(width, int(round(x0 * scale)))),
        max(0, min(height, int(round(y0 * scale)))),
        max(0, min(width, int(round(x1 * scale)))),
        max(0, min(height, int(round(y1 * scale)))),
    ]


def grid_distractors(width: int, height: int, gold_bbox: list[int]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    boxes = [
        [0, 0, int(width * 0.3), int(height * 0.18)],
        [int(width * 0.7), 0, width, int(height * 0.18)],
        [0, int(height * 0.82), int(width * 0.35), height],
        [int(width * 0.65), int(height * 0.82), width, height],
    ]
    for index, box in enumerate(boxes):
        if bbox_iou(box, gold_bbox) < 0.05:
            regions.append(
                {
                    "bbox": box,
                    "source": "layout_distractor",
                    "type": "non_target_page_region",
                    "score": 0.05,
                    "hint": f"页眉页脚/边角布局干扰区域 {index}",
                    "target_iou": 0.0,
                    "is_target": False,
                }
            )
    return regions


def evidence_query(candidate: PageCandidate, caption: str) -> str:
    pieces = [
        candidate.source_stem,
        caption,
        extract_title(caption),
        infer_object_type(caption),
        " ".join(candidate.source_meta.get("topics") or []),
    ]
    return " ".join(item for item in dedupe([normalize_spaces(str(piece)) for piece in pieces]) if item)[:500]


def infer_image_scope(caption: str) -> str:
    text = normalize_spaces(caption)
    if not text:
        return ""
    if re.search(r"(局部|部分|detail|Detail|之一|局部图)", text):
        return "partial_detail"
    if re.search(r"(全图|全幅|整幅|全卷|whole|complete)", text, flags=re.IGNORECASE):
        return "full_figure"
    if caption_like(text):
        return "figure_or_plate"
    return ""


def extract_title(text: str) -> str:
    match = re.search(r"《([^》]{1,60})》", text or "")
    return normalize_spaces(match.group(1)) if match else ""


def infer_displayed_region(caption: str, scope: str) -> str:
    text = normalize_spaces(caption)
    if "局部" in text:
        return "局部图"
    if "部分" in text:
        return "部分画面"
    if scope == "full_figure":
        return "全幅图像"
    return ""


def infer_object_type(caption: str) -> str:
    text = normalize_spaces(caption)
    if re.search(r"(山水画|山水圖|山水图|landscape)", text, flags=re.IGNORECASE):
        return "山水画图像"
    if re.search(r"(画|圖|图|Figure|Fig)", text):
        return "图像"
    return ""


def caption_like(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return bool(re.match(r"^(【?图|圖|Fig\\.?|Figure)", normalized, flags=re.IGNORECASE))


def area_px(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def normalize_spaces(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def dedupe(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value)
    return value[:120] or "untitled"


def html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def default_output_dir(output_root: Path) -> Path:
    return output_root / f"agentbench_v0_8_page_capped_docsplit_sft_{datetime.now().strftime('%Y%m%d_%H%M')}"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
