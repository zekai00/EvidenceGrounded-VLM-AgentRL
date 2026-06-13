#!/usr/bin/env python3
"""Run a page-level VLM probe for v1.0.5 candidate construction.

The probe sends whole PDF pages to a VLM and asks it to directly return
landscape-painting target boxes, caption boxes, and visible caption text.  It
does not write SFT rows; the output is a review package for deciding whether a
page-level builder is worth scaling.
"""

from __future__ import annotations

import argparse
import copy
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
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import fitz
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_agentbench_v0_9_fixedsplit_train_multitarget as v09  # noqa: E402
import build_gold_eval_v1_0_4 as gold_review  # noqa: E402
import build_v1_0_4_core4_dedup_expanded_sft as dedup  # noqa: E402
import build_v1_0_5_core4_visual_audited_sft as v105  # noqa: E402


DEFAULT_DATASET_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_5_visual_audited_expanded_sft_20260613_0010"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DEFAULT_MODEL = "qwen3.7-max"
DEFAULT_FALLBACK_MODELS = (
    "qwen3.7-max-2026-06-08,"
    "qwen3.7-max-preview,"
    "qwen3.7-plus-2026-05-26,"
    "qwen3.7-plus,"
    "qwen3.6-plus,"
    "qwen3.5-plus-2026-04-20,"
    "qwen3.6-27b,"
    "glm-5.1,"
    "kimi-k2.6,"
    "deepseek-v4-pro,"
    "deepseek-v4-flash"
)
DEFAULT_DOTENV = Path("/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/.env")


PAGE_LEVEL_PROMPT = """你是 EvidenceGrounded-VLM-AgentRL 的 page-level 数据构建员。

你会看到一整页 PDF 渲染图。请直接从整页中找出适合 Core4 SFT 的“中国/东亚古典山水画或山水画局部”目标，并为每个目标输出：
- 目标图像框 target_bbox_norm1000；
- 对应图注框 caption_bbox_norm1000；
- 页面可见的完整图注文本 caption_text。

坐标要求：
- 坐标必须是 0-1000 归一化坐标，不要输出页面像素坐标；
- 原点在左上角，整页右下角约为 [1000,1000]；
- bbox 格式必须是 [x1, y1, x2, y2]；
- target_bbox_norm1000 只框目标图像，不要包含图注、正文、相邻图；
- caption_bbox_norm1000 只框对应图注，不要包含正文或其他图注。

只保留这些对象：
- 山水画、山水画局部、以山水/自然景观为主体的古典绘画；
- caption 与目标图像在页面上能明确对应。

排除这些对象：
- 纯正文、目录、表格、示意图、建筑/器物照片；
- 书法题跋、人物故事、佛教叙事、动物/骑乘/乐人等非山水主体；
- target 或 caption 无法对应的相邻图；
- 需要凭常识补全但页面图注不可见的文本。

如果页面中没有合格目标，detections 输出空数组。

只输出 JSON 对象，不要输出 Markdown。JSON schema:
{
  "page_summary": "一句话说明页面情况",
    "detections": [
    {
      "target_bbox_norm1000": [0, 0, 0, 0],
      "caption_bbox_norm1000": [0, 0, 0, 0],
      "caption_text": "页面可见完整图注",
      "object_domain": "landscape_painting|landscape_detail|classical_painting_unclear_landscape|non_landscape_artwork|diagram_or_chart|caption_or_table|text_only|calligraphy_or_inscription|architecture_or_object_photo|other|unclear",
      "image_scope": "full_work|partial_detail|multi_work_comparison|unclear",
      "caption_target_match": "yes|no|uncertain",
      "accept_for_sft": true,
      "needs_human_review": false,
      "confidence": 0.0,
      "reason": "一句话说明"
    }
  ]
}

当前页面元数据：
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run page-level VLM construction probe.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--candidate-cache-dir", default=str(dedup.DEFAULT_CANDIDATE_CACHE))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pilot-size", type=int, default=50)
    parser.add_argument("--page-offset", type=int, default=0)
    parser.add_argument("--page-dpi", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--provider", choices=["dashscope", "offline"], default="dashscope")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--dotenv", default=str(DEFAULT_DOTENV))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--image-max-side", type=int, default=1800)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_review.load_dotenv(Path(args.dotenv))
    dataset_dir = Path(args.dataset_dir)
    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args)

    selected_rows = read_jsonl(dataset_dir / "selected_candidates.jsonl")
    if not selected_rows:
        raise FileNotFoundError(f"No selected candidates found under {dataset_dir}")
    candidates = load_candidate_lookup(args)
    selected_with_candidates = attach_candidates(selected_rows, candidates)
    pages = choose_pages(selected_with_candidates, dataset_dir, args)
    write_jsonl(output_dir / "selected_pages.jsonl", [page_to_json(page) for page in pages])

    client = make_client(args)
    stream_path = output_dir / "review" / "page_level_vlm_stream.jsonl"
    existing = load_existing(stream_path, args)
    reviewed: list[dict[str, Any]] = []
    page_cache: dict[tuple[str, int], Path] = {}
    for pos, page in enumerate(pages, start=1):
        page_key = page["page_key"]
        try:
            rendered = v09.render_page(page["candidate"], output_dir / "pages", args.page_dpi, page_cache)
            page = with_render_geometry(page, rendered)
            row = existing.get(page_key)
            if row is None:
                row = client.review(page, rendered)
                append_jsonl(stream_path, [row])
                if args.sleep:
                    time.sleep(args.sleep)
            row = normalize_review_row(row, page, rendered)
            reviewed.append(row)
            print(
                json.dumps(
                    {
                        "progress": f"{pos}/{len(pages)}",
                        "page_key": page_key,
                        "detections": len(row.get("detections") or []),
                        "model": row.get("review_model"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            row = {
                **page_to_json(page),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "detections": [],
                "page_summary": "page-level probe failed",
            }
            append_jsonl(stream_path, [row])
            reviewed.append(row)
            print(json.dumps({"page_key": page_key, "error": row["error"]}, ensure_ascii=False), flush=True)

    write_jsonl(output_dir / "review" / "page_level_vlm_reviewed.jsonl", reviewed)
    accepted = materialize_detection_assets(output_dir, reviewed)
    write_jsonl(output_dir / "page_level_detections.jsonl", accepted)
    summary = build_summary(output_dir, args, reviewed, accepted)
    write_text_json(output_dir / "manifest.json", summary)
    write_report(output_dir / "构建报告.md", summary)
    package = write_markdown_package(output_dir, reviewed)
    summary["artifacts"]["review_package"] = str(package)
    write_text_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(args.output_root) / f"agentbench_v1_0_5_page_level_vlm_probe_{stamp}"


def prepare_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists; use --resume or --overwrite")
    for child in ["pages", "overlays", "crops", "review"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def load_candidate_lookup(args: argparse.Namespace) -> dict[tuple[Any, ...], v09.PageCandidate]:
    builder_args = argparse.Namespace(**vars(args))
    builder_args.candidate_filter_profile = "visual_audited_expanded"
    builder_args.raw_pdf_root = dedup.DEFAULT_RAW_PDF_ROOT
    builder_args.train_target = 100000
    builder_args.val_target = 150
    builder_args.test_target = 150
    builder_args.train_caption_cap = 2
    builder_args.eval_caption_cap = 1
    builder_args.max_doc_pages_train = 120
    builder_args.max_doc_pages_eval = 40
    builder_args.reserve_largest_docs_for_train = 5
    builder_args.min_caption_score = -999.0
    builder_args.max_caption_chars = 260
    builder_args.opencv_min_area_ratio = 0.018
    builder_args.opencv_max_area_ratio = 0.65
    builder_args.opencv_min_width_ratio = 0.10
    builder_args.opencv_min_height_ratio = 0.07
    builder_args.opencv_max_text_overlap = 0.18
    builder_args.opencv_min_aspect = 0.22
    builder_args.opencv_max_aspect = 7.5
    builder_args.keep_nonlandscape_pdf_image_blocks = False
    candidates, _, _ = dedup.collect_dedup_candidates(Path(args.candidate_cache_dir), builder_args)
    out: dict[tuple[Any, ...], v09.PageCandidate] = {}
    for candidate in candidates:
        out[candidate_key_from_candidate(candidate)] = candidate
    return out


def attach_candidates(rows: list[dict[str, Any]], lookup: dict[tuple[Any, ...], v09.PageCandidate]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        candidate = lookup.get(candidate_key_from_row(row))
        if not candidate:
            continue
        out.append({**row, "_candidate": candidate})
    return out


def choose_pages(rows: list[dict[str, Any]], dataset_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    by_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page[(str(row["source_file"]), int(row["page"]))].append(row)

    forced_keys: list[tuple[str, int]] = []
    stream = dataset_dir / "review" / "visual_audit_stream.jsonl"
    if stream.exists():
        selected_by_task = {str(row["task_id"]): row for row in rows}
        for review in read_jsonl(stream):
            selected = selected_by_task.get(str(review.get("task_id") or ""))
            if selected:
                key = (str(selected["source_file"]), int(selected["page"]))
                if key not in forced_keys:
                    forced_keys.append(key)

    rng = random.Random(args.seed)
    scored: list[tuple[float, tuple[str, int]]] = []
    for key, group in by_page.items():
        if key in forced_keys:
            continue
        max_risk = max(float(item.get("risk_score") or 0.0) for item in group)
        count_bonus = min(2.0, 0.25 * len(group))
        split_bonus = {"train": 0.0, "val": 0.15, "test": 0.2}.get(str(group[0].get("split")), 0.0)
        scored.append((max_risk + count_bonus + split_bonus + rng.random() * 0.01, key))
    scored.sort(reverse=True)

    ordered_keys = list(forced_keys)
    for _, key in scored:
        if key not in ordered_keys:
            ordered_keys.append(key)

    offset = max(0, int(args.page_offset or 0))
    chosen_keys = ordered_keys[offset : offset + args.pilot_size]

    pages = []
    for page_index, key in enumerate(chosen_keys):
        group = by_page[key]
        group = sorted(group, key=lambda item: float(item.get("risk_score") or 0.0), reverse=True)
        candidate = group[0]["_candidate"]
        pages.append(
            {
                "page_index": page_index,
                "page_key": f"{candidate.source_stem}_p{candidate.page:04d}",
                "source_file": candidate.source_file,
                "source_stem": candidate.source_stem,
                "page": candidate.page,
                "split": group[0].get("split"),
                "candidate_count_on_page": len(group),
                "candidate_task_ids": [item.get("task_id") for item in group],
                "candidate_captions": [item.get("caption_text") for item in group[:8]],
                "candidate_anchors": [
                    {
                        "task_id": item.get("task_id"),
                        "image_bbox": item.get("image_bbox"),
                        "caption_bbox": item.get("caption_bbox"),
                        "caption_text": item.get("caption_text"),
                        "risk_score": item.get("risk_score"),
                    }
                    for item in group
                ],
                "page_width": candidate.page_width,
                "page_height": candidate.page_height,
                "candidate": candidate,
            }
        )
    return pages


def make_client(args: argparse.Namespace) -> "PageLevelClient":
    if args.provider == "offline":
        return OfflinePageLevelClient(args)
    return DashScopePageLevelClient(args)


class PageLevelClient:
    def review(self, page: dict[str, Any], page_image: Path) -> dict[str, Any]:
        raise NotImplementedError


class OfflinePageLevelClient(PageLevelClient):
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def review(self, page: dict[str, Any], page_image: Path) -> dict[str, Any]:
        return {
            **page_to_json(page),
            "ok": True,
            "review_model": "offline_rules",
            "input_mode": "offline",
            "page_image": str(page_image),
            "page_summary": "offline smoke only",
            "detections": [],
            "raw_response": json.dumps({"page_summary": "offline smoke only", "detections": []}, ensure_ascii=False),
        }


class DashScopePageLevelClient(PageLevelClient):
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.models = v105.dedupe_keep_order([args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()])
        self.args = args

    def review(self, page: dict[str, Any], page_image: Path) -> dict[str, Any]:
        last_error: Exception | None = None
        for model in self.models:
            for image_mode in v105.image_modes_for_model(model):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=build_messages(page, page_image, self.args, image_mode),
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or ""
                    parsed = gold_review.parse_json_object(content)
                    return {
                        **page_to_json(page),
                        "ok": True,
                        "review_model": model,
                        "input_mode": image_mode,
                        "page_image": str(page_image),
                        "raw_response": content,
                        **normalize_page_response(parsed, page),
                    }
                except Exception as exc:
                    last_error = exc
                    continue
        raise RuntimeError(f"all VLM models failed: {last_error!r}")


def build_messages(page: dict[str, Any], page_image: Path, args: argparse.Namespace, image_mode: str) -> list[dict[str, Any]]:
    metadata = {
        "source_file": page.get("source_file"),
        "page": page.get("page"),
        "page_size_px": [page.get("page_width"), page.get("page_height")],
        "candidate_count_on_page": page.get("candidate_count_on_page"),
        "candidate_caption_examples": page.get("candidate_captions"),
    }
    prompt = PAGE_LEVEL_PROMPT + json.dumps(metadata, ensure_ascii=False, indent=2)
    if image_mode == "image":
        content: Any = [
            {"type": "image", "image": gold_review.image_data_url(page_image, args.image_max_side)},
            {"type": "text", "text": prompt},
        ]
    else:
        content = [
            {"type": "image_url", "image_url": {"url": gold_review.image_data_url(page_image, args.image_max_side)}},
            {"type": "text", "text": prompt},
        ]
    return [{"role": "user", "content": content}]


def normalize_page_response(parsed: dict[str, Any], page: dict[str, Any]) -> dict[str, Any]:
    detections = []
    raw_detections = parsed.get("detections")
    if not isinstance(raw_detections, list):
        raw_detections = []
    for idx, item in enumerate(raw_detections):
        if not isinstance(item, dict):
            continue
        target_norm = v105.normalize_bbox_value(
            item.get("raw_target_bbox_norm1000") or item.get("target_bbox_norm1000") or item.get("target_bbox_0_1000")
        )
        caption_norm = v105.normalize_bbox_value(
            item.get("raw_caption_bbox_norm1000") or item.get("caption_bbox_norm1000") or item.get("caption_bbox_0_1000")
        )
        raw_target = v105.normalize_bbox_value(item.get("raw_target_bbox_page_px"))
        raw_caption = v105.normalize_bbox_value(item.get("raw_caption_bbox_page_px"))
        if not target_norm:
            raw_target = raw_target or v105.normalize_bbox_value(item.get("target_bbox_page_px") or item.get("target_bbox"))
        if not caption_norm:
            raw_caption = raw_caption or v105.normalize_bbox_value(item.get("caption_bbox_page_px") or item.get("caption_bbox"))
        chosen = choose_bbox_interpretation(raw_target, raw_caption, page, target_norm=target_norm, caption_norm=caption_norm)
        target = chosen.get("target_bbox")
        caption = chosen.get("caption_bbox")
        coord_system = str(chosen.get("coord_system") or "missing")
        caption_text = v105.normalize_space(item.get("caption_text"))[:600]
        caption_fix = refine_caption_bbox_from_text(caption_text, target, caption, page)
        if caption_fix.get("accepted"):
            caption = caption_fix.get("caption_bbox")
            coord_system = f"{coord_system}+caption_text_refined"
        object_domain = normalize_enum(
            item.get("object_domain"),
            {
                "landscape_painting",
                "landscape_detail",
                "classical_painting_unclear_landscape",
                "non_landscape_artwork",
                "diagram_or_chart",
                "caption_or_table",
                "text_only",
                "calligraphy_or_inscription",
                "architecture_or_object_photo",
                "other",
                "unclear",
            },
            "unclear",
        )
        domain_gate = object_domain_gate(object_domain, caption_text, str(item.get("reason") or ""))
        detections.append(
            {
                "detection_index": idx,
                "raw_target_bbox_norm1000": target_norm,
                "raw_caption_bbox_norm1000": caption_norm,
                "raw_target_bbox_page_px": raw_target,
                "raw_caption_bbox_page_px": raw_caption,
                "bbox_coord_system": coord_system,
                "bbox_coord_candidates": chosen.get("candidates", []),
                "caption_bbox_refine": caption_fix,
                "target_bbox_page_px": target,
                "caption_bbox_page_px": caption,
                "caption_text": caption_text,
                "object_domain": object_domain,
                "object_domain_gate": domain_gate,
                "image_scope": normalize_enum(
                    item.get("image_scope"),
                    {"full_work", "partial_detail", "multi_work_comparison", "unclear"},
                    "unclear",
                ),
                "caption_target_match": normalize_enum(item.get("caption_target_match"), {"yes", "no", "uncertain"}, "uncertain"),
                "accept_for_sft": v105.coerce_bool(item.get("accept_for_sft")),
                "needs_human_review": v105.coerce_bool(item.get("needs_human_review")),
                "confidence": max(0.0, min(1.0, v105.safe_float(item.get("confidence"), 0.0))),
                "reason": str(item.get("reason") or "")[:500],
                "valid_geometry": valid_detection_geometry(target, caption, page),
            }
        )
    return {"page_summary": str(parsed.get("page_summary") or "")[:500], "detections": detections}


def choose_bbox_interpretation(
    raw_target: list[int] | None,
    raw_caption: list[int] | None,
    page: dict[str, Any],
    target_norm: list[int] | None = None,
    caption_norm: list[int] | None = None,
) -> dict[str, Any]:
    width = int(page.get("page_width") or 0)
    height = int(page.get("page_height") or 0)
    if not width or not height:
        return {
            "coord_system": "missing",
            "target_bbox": None,
            "caption_bbox": None,
            "candidates": [],
        }

    variants: list[dict[str, Any]] = []
    if target_norm and caption_norm:
        variants.append(
            {
                "coord_system": "norm1000_direct",
                "target_bbox": scale_norm1000_bbox(target_norm, width, height),
                "caption_bbox": scale_norm1000_bbox(caption_norm, width, height),
            }
        )
    if raw_target and raw_caption:
        variants.append(
            {
                "coord_system": "page_px_anchor_selected",
                "target_bbox": clamp_bbox(raw_target, width, height),
                "caption_bbox": clamp_bbox(raw_caption, width, height),
            }
        )
    if raw_target and raw_caption and max(raw_target + raw_caption) <= 1000 and (width > 1024 or height > 1024):
        variants.append(
            {
                "coord_system": "norm1000_scaled_anchor_selected",
                "target_bbox": scale_norm1000_bbox(raw_target, width, height),
                "caption_bbox": scale_norm1000_bbox(raw_caption, width, height),
            }
        )
    if not variants:
        return {
            "coord_system": "missing",
            "target_bbox": None,
            "caption_bbox": None,
            "candidates": [],
        }

    scored = []
    for variant in variants:
        score, details = score_bbox_variant(variant["target_bbox"], variant["caption_bbox"], page)
        scored.append({**variant, "score": round(score, 4), "score_details": details})
    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    best = scored[0]
    return {
        "coord_system": best["coord_system"],
        "target_bbox": best["target_bbox"],
        "caption_bbox": best["caption_bbox"],
        "candidates": [
            {
                "coord_system": item["coord_system"],
                "target_bbox": item["target_bbox"],
                "caption_bbox": item["caption_bbox"],
                "score": item["score"],
                "score_details": item["score_details"],
            }
            for item in scored
        ],
    }


def scale_norm1000_bbox(box: list[int], width: int, height: int) -> list[int]:
    scaled = [
        int(round(box[0] * width / 1000.0)),
        int(round(box[1] * height / 1000.0)),
        int(round(box[2] * width / 1000.0)),
        int(round(box[3] * height / 1000.0)),
    ]
    return clamp_bbox(scaled, width, height)


def score_bbox_variant(target: list[int], caption: list[int], page: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    width = max(1, int(page.get("page_width") or 1))
    height = max(1, int(page.get("page_height") or 1))
    anchors = page.get("candidate_anchors") or []
    geometry = score_target_caption_geometry(target, caption, width, height)
    area_ratio = bbox_area(target) / float(width * height)
    area_score = max(0.0, 1.0 - abs(area_ratio - 0.18) / 0.35)
    best_anchor_score = 0.0
    best_anchor: dict[str, Any] = {}
    for anchor in anchors:
        image_bbox = normalize_anchor_bbox(anchor.get("image_bbox"), width, height)
        caption_bbox = normalize_anchor_bbox(anchor.get("caption_bbox"), width, height)
        target_score = anchor_bbox_score(target, image_bbox, width, height)
        caption_score = anchor_bbox_score(caption, caption_bbox, width, height, caption_mode=True)
        score = target_score * 0.45 + caption_score * 0.45 + geometry * 0.10
        if score > best_anchor_score:
            best_anchor_score = score
            best_anchor = {
                "task_id": anchor.get("task_id"),
                "target_anchor_score": round(target_score, 4),
                "caption_anchor_score": round(caption_score, 4),
            }
    final = best_anchor_score * 0.80 + geometry * 0.15 + area_score * 0.05
    if not valid_page_bbox_px(target, width, height, min_width=24, min_height=24):
        final -= 2.0
    if not valid_page_bbox_px(caption, width, height, min_width=12, min_height=8):
        final -= 2.0
    if v105.bbox_overlap_ratio(target, caption, denominator="caption") > 0.05:
        final -= 1.0
    return final, {
        "best_anchor": best_anchor,
        "best_anchor_score": round(best_anchor_score, 4),
        "geometry_score": round(geometry, 4),
        "area_ratio": round(area_ratio, 4),
        "area_score": round(area_score, 4),
    }


def normalize_anchor_bbox(value: Any, width: int, height: int) -> list[int] | None:
    box = v105.normalize_bbox_value(value)
    if not box:
        return None
    return clamp_bbox(box, width, height)


def refine_caption_bbox_from_text(caption_text: str, target_bbox: Any, current_caption: Any, page: dict[str, Any]) -> dict[str, Any]:
    if not caption_text or not isinstance(target_bbox, list):
        return {"accepted": False, "reason": "missing_caption_or_target"}
    width = int(page.get("page_width") or 0)
    height = int(page.get("page_height") or 0)
    blocks = collect_pdf_text_blocks_for_page(page)
    if not blocks:
        return {"accepted": False, "reason": "no_pdf_text_blocks"}
    target_marker = extract_figure_marker(caption_text)
    best: dict[str, Any] | None = None
    for block in blocks:
        text = v105.normalize_space(block.get("text"))
        if not text:
            continue
        marker = extract_figure_marker(text)
        if target_marker and marker and marker != target_marker:
            continue
        if target_marker and not marker:
            continue
        if target_marker and count_figure_markers(text) > 1:
            continue
        similarity = caption_text_similarity(caption_text, text)
        if target_marker and marker == target_marker:
            similarity = max(similarity, 0.72)
        if similarity < 0.42:
            continue
        bbox = block.get("bbox")
        geom = score_target_caption_geometry(target_bbox, bbox, max(1, width), max(1, height))
        overlap = v105.bbox_overlap_ratio(target_bbox, bbox, denominator="caption")
        if overlap > 0.03:
            continue
        distance_bonus = caption_distance_bonus(target_bbox, bbox, max(1, height))
        score = similarity * 0.62 + geom * 0.28 + distance_bonus * 0.10
        row = {
            "bbox": bbox,
            "text": text[:260],
            "similarity": round(similarity, 4),
            "geometry_score": round(geom, 4),
            "distance_bonus": round(distance_bonus, 4),
            "score": round(score, 4),
            "figure_marker": marker,
        }
        if best is None or score > float(best.get("score") or 0.0):
            best = row
    if not best:
        return {"accepted": False, "reason": "no_high_conf_text_match"}
    old_iou = bbox_iou(current_caption, best["bbox"]) if isinstance(current_caption, list) else 0.0
    accepted = bool(best["score"] >= 0.54 and best["similarity"] >= 0.50)
    if accepted and target_marker and best.get("figure_marker") and best.get("figure_marker") != target_marker:
        accepted = False
    return {
        "accepted": accepted,
        "reason": "pdf_text_match" if accepted else "text_match_below_threshold",
        "caption_bbox": best["bbox"],
        "matched_text": best["text"],
        "score": best["score"],
        "similarity": best["similarity"],
        "geometry_score": best["geometry_score"],
        "old_iou": round(old_iou, 4),
    }


def collect_pdf_text_blocks_for_page(page: dict[str, Any]) -> list[dict[str, Any]]:
    candidate = page.get("candidate")
    if not candidate:
        return []
    width = int(page.get("page_width") or 0)
    height = int(page.get("page_height") or 0)
    if not width or not height:
        return []
    out: list[dict[str, Any]] = []
    try:
        with fitz.open(candidate.source_path) as doc:
            pdf_page = doc[int(candidate.page) - 1]
            rect = pdf_page.rect
            sx = width / float(rect.width)
            sy = height / float(rect.height)
            for block_index, block in enumerate(pdf_page.get_text("blocks")):
                if len(block) < 5:
                    continue
                text = v09.clean_text(str(block[4]))
                if not text:
                    continue
                bbox = [
                    int(round(float(block[0]) * sx)),
                    int(round(float(block[1]) * sy)),
                    int(round(float(block[2]) * sx)),
                    int(round(float(block[3]) * sy)),
                ]
                if not valid_page_bbox_px(bbox, width, height, min_width=8, min_height=5):
                    continue
                out.append({"block_index": block_index, "bbox": bbox, "text": text})
    except Exception:
        return []
    return out


def caption_text_similarity(a: str, b: str) -> float:
    ca = compact_caption_text(a)
    cb = compact_caption_text(b)
    if not ca or not cb:
        return 0.0
    if ca in cb or cb in ca:
        return min(1.0, min(len(ca), len(cb)) / max(1, max(len(ca), len(cb))) + 0.25)
    return SequenceMatcher(None, ca, cb).ratio()


def compact_caption_text(text: str) -> str:
    text = v105.normalize_space(text).lower()
    return re.sub(r"[\s,.;:：，。；、（）()【】\\[\\]《》\"'“”‘’·．.\\-—_]+", "", text)


def extract_figure_marker(text: str) -> str:
    normalized = v105.normalize_space(text)
    patterns = [
        r"(图\s*[一二三四五六七八九十百\d]+(?:[：:\-.．]\s*[一二三四五六七八九十百\d]+){0,3})",
        r"(Figure\s*\d+(?:\.\d+){0,3})",
        r"(Fig\.\s*\d+(?:\.\d+){0,3})",
        r"(〔图[一二三四五六七八九十百\d]+〕)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized, flags=re.IGNORECASE)
        if match:
            return compact_caption_text(match.group(1))
    return ""


def count_figure_markers(text: str) -> int:
    normalized = v105.normalize_space(text)
    patterns = [
        r"图\s*[一二三四五六七八九十百\d]+(?:[：:\-.．]\s*[一二三四五六七八九十百\d]+){0,3}",
        r"Figure\s*\d+(?:\.\d+){0,3}",
        r"Fig\.\s*\d+(?:\.\d+){0,3}",
        r"〔图[一二三四五六七八九十百\d]+〕",
    ]
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            seen.add(compact_caption_text(match.group(0)))
    return len(seen)


def caption_distance_bonus(target: list[int], caption: list[int], height: int) -> float:
    gap = min(abs(caption[1] - target[3]), abs(target[1] - caption[3]))
    return max(0.0, min(1.0, 1.0 - gap / max(1.0, height * 0.20)))


def object_domain_gate(object_domain: str, caption_text: str, reason: str) -> dict[str, Any]:
    text = f"{caption_text} {reason}"
    compact = compact_caption_text(text)
    if object_domain in {"non_landscape_artwork", "diagram_or_chart", "caption_or_table", "text_only", "calligraphy_or_inscription", "architecture_or_object_photo", "other", "unclear"}:
        return {"accepted": False, "reason": f"object_domain={object_domain}"}
    hard_negative_terms = [
        "照片",
        "博物馆网",
        "建筑照片",
        "器物",
        "表格",
        "示意图",
        "佛塔",
        "宝带桥佛塔",
        "桥梁照片",
    ]
    if any(term in text for term in hard_negative_terms):
        return {"accepted": False, "reason": "caption_or_reason_contains_non_painting_object"}
    if ("桥" in text or "塔" in text or "建筑" in text) and not any(term in text for term in ["《", "图册", "山水", "画", "卷", "轴"]):
        return {"accepted": False, "reason": "architecture_or_bridge_without_painting_context"}
    return {"accepted": True, "reason": "pass"}


def with_render_geometry(page: dict[str, Any], page_image: Path) -> dict[str, Any]:
    out = copy.deepcopy(page)
    try:
        with Image.open(page_image) as image:
            rendered_width, rendered_height = image.size
    except Exception:
        return out
    old_width = max(1, int(out.get("page_width") or rendered_width))
    old_height = max(1, int(out.get("page_height") or rendered_height))
    if rendered_width == old_width and rendered_height == old_height:
        return out
    sx = rendered_width / float(old_width)
    sy = rendered_height / float(old_height)
    out["page_width"] = rendered_width
    out["page_height"] = rendered_height
    out["anchor_scale_from_original_page_px"] = [round(sx, 6), round(sy, 6)]
    scaled_anchors = []
    for anchor in out.get("candidate_anchors") or []:
        item = dict(anchor)
        item["image_bbox"] = scale_bbox_xy(anchor.get("image_bbox"), sx, sy)
        item["caption_bbox"] = scale_bbox_xy(anchor.get("caption_bbox"), sx, sy)
        scaled_anchors.append(item)
    out["candidate_anchors"] = scaled_anchors
    return out


def scale_bbox_xy(value: Any, sx: float, sy: float) -> list[int] | None:
    box = v105.normalize_bbox_value(value)
    if not box:
        return None
    return [
        int(round(box[0] * sx)),
        int(round(box[1] * sy)),
        int(round(box[2] * sx)),
        int(round(box[3] * sy)),
    ]


def anchor_bbox_score(box: list[int], anchor: list[int] | None, width: int, height: int, caption_mode: bool = False) -> float:
    if not anchor:
        return 0.0
    iou = bbox_iou(box, anchor)
    overlap = min(1.0, bbox_intersection_area(box, anchor) / max(1.0, min(bbox_area(box), bbox_area(anchor))))
    dist = bbox_center_distance(box, anchor) / (height if caption_mode else (width * width + height * height) ** 0.5)
    proximity = max(0.0, 1.0 - dist * (18.0 if caption_mode else 2.5))
    if caption_mode:
        return max(iou * 3.0, overlap * 1.7, proximity)
    return max(iou * 2.0, overlap * 1.2, proximity)


def score_target_caption_geometry(target: list[int], caption: list[int], width: int, height: int) -> float:
    tx1, ty1, tx2, ty2 = target
    cx1, cy1, cx2, cy2 = caption
    horizontal_overlap = max(0, min(tx2, cx2) - max(tx1, cx1)) / max(1, min(tx2 - tx1, cx2 - cx1))
    vertical_gap = min(abs(cy1 - ty2), abs(ty1 - cy2)) / max(1, height)
    center_gap = abs(((tx1 + tx2) / 2.0) - ((cx1 + cx2) / 2.0)) / max(1, width)
    below_or_above = cy1 >= ty1 or cy2 <= ty2
    score = 0.45 * min(1.0, horizontal_overlap) + 0.35 * max(0.0, 1.0 - vertical_gap * 8.0) + 0.20 * max(0.0, 1.0 - center_gap * 3.0)
    if not below_or_above:
        score *= 0.7
    return max(0.0, min(1.0, score))


def bbox_area(box: list[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def bbox_intersection_area(a: list[int], b: list[int]) -> int:
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def bbox_iou(a: list[int], b: list[int]) -> float:
    inter = bbox_intersection_area(a, b)
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def bbox_center_distance(a: list[int], b: list[int]) -> float:
    ax = (a[0] + a[2]) / 2.0
    ay = (a[1] + a[3]) / 2.0
    bx = (b[0] + b[2]) / 2.0
    by = (b[1] + b[3]) / 2.0
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def valid_page_bbox_px(box: Any, width: int, height: int, min_width: int, min_height: int) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    x1, y1, x2, y2 = [int(v) for v in box]
    if x2 - x1 < min_width or y2 - y1 < min_height:
        return False
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return False
    return True


def clamp_bbox(box: list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    return [x1, y1, x2, y2]


def valid_detection_geometry(target: Any, caption: Any, page: dict[str, Any]) -> bool:
    width = int(page.get("page_width") or 0)
    height = int(page.get("page_height") or 0)
    if not valid_page_bbox_px(target, width, height, min_width=24, min_height=24):
        return False
    if not valid_page_bbox_px(caption, width, height, min_width=12, min_height=8):
        return False
    if v105.bbox_overlap_ratio(target, caption, denominator="caption") > 0.05:
        return False
    return True


def normalize_review_row(row: dict[str, Any], page: dict[str, Any], page_image: Path) -> dict[str, Any]:
    out = {**page_to_json(page), **copy.deepcopy(row)}
    out["page_image"] = str(page_image)
    normalized = normalize_page_response({"page_summary": out.get("page_summary"), "detections": out.get("detections")}, page)
    out["page_summary"] = normalized["page_summary"]
    out["detections"] = normalized["detections"]
    return out


def materialize_detection_assets(output_dir: Path, reviewed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in reviewed:
        page_image = Path(str(row.get("page_image") or ""))
        if not page_image.exists():
            continue
        overlay = output_dir / "overlays" / f"{safe_name(row['page_key'])}_pagelevel.jpg"
        draw_page_overlay(page_image, row.get("detections") or [], overlay)
        row["page_level_overlay"] = str(overlay)
        for det in row.get("detections") or []:
            det["page_key"] = row.get("page_key")
            det["source_file"] = row.get("source_file")
            det["page"] = row.get("page")
            det["split"] = row.get("split")
            det["page_image"] = str(page_image)
            det["page_level_overlay"] = str(overlay)
            crop = output_dir / "crops" / f"{safe_name(row['page_key'])}_d{int(det['detection_index']):02d}.jpg"
            if det.get("target_bbox_page_px"):
                crop_image(page_image, det["target_bbox_page_px"], crop)
                if crop.exists():
                    det["crop_image"] = str(crop)
            det["auto_usable_probe"] = bool(
                det.get("accept_for_sft") is True
                and det.get("caption_target_match") == "yes"
                and det.get("object_domain") in {"landscape_painting", "landscape_detail"}
                and (det.get("object_domain_gate") or {}).get("accepted") is True
                and det.get("caption_text")
                and det.get("valid_geometry")
            )
            rows.append(dict(det))
    return rows


def draw_page_overlay(page_image: Path, detections: list[dict[str, Any]], out: Path) -> None:
    image = Image.open(page_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    for det in detections:
        idx = int(det.get("detection_index") or 0) + 1
        target = det.get("target_bbox_page_px")
        caption = det.get("caption_bbox_page_px")
        if target:
            draw.rectangle(target, outline="red", width=5)
            draw.text((target[0], max(0, target[1] - 24)), f"T{idx}", fill="red")
        if caption:
            draw.rectangle(caption, outline="cyan", width=4)
            draw.text((caption[0], max(0, caption[1] - 22)), f"C{idx}", fill="cyan")
    image.save(out, quality=92)


def crop_image(page_image: Path, bbox: list[int], out: Path) -> None:
    image = Image.open(page_image).convert("RGB")
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(image.width, x2)
    y2 = min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return
    image.crop((x1, y1, x2, y2)).save(out, quality=92)


def write_markdown_package(output_dir: Path, reviewed: list[dict[str, Any]]) -> Path:
    package_dir = output_dir / "review" / f"page_level_review_package_{datetime.now().strftime('%Y%m%d_%H%M')}"
    assets = package_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    reviewed = sorted(reviewed, key=lambda item: int(item.get("page_index") or 0))
    md = [
        "# v1.0.5 Page-level VLM Probe 人工查看包",
        "",
        f"- 数据目录：`{output_dir}`",
        f"- 页面数：{len(reviewed)}",
        "- 整页 VLM 直接输出：红框=T target，青框=C caption。",
        "",
    ]
    for row in reviewed:
        overlay_rel = copy_asset(row.get("page_level_overlay"), assets, f"{row.get('page_index'):03d}_{row.get('page_key')}_overlay.jpg")
        page_rel = copy_asset(row.get("page_image"), assets, f"{row.get('page_index'):03d}_{row.get('page_key')}_page.jpg")
        md.extend(
            [
                f"## P{int(row.get('page_index') or 0) + 1:03d} {row.get('page_key')}",
                "",
                f"- split/source/page：`{row.get('split')}` / `{row.get('source_file')}` / `{row.get('page')}`",
                f"- candidate_count_on_page：`{row.get('candidate_count_on_page')}`",
                f"- VLM model：`{row.get('review_model')}`，ok=`{row.get('ok')}`",
                f"- page_summary：{row.get('page_summary')}",
                f"- detections：{len(row.get('detections') or [])}",
                "",
                f"![page-level overlay]({overlay_rel})",
                "",
                f"![page]({page_rel})",
                "",
            ]
        )
        for det in row.get("detections") or []:
            crop_rel = copy_asset(det.get("crop_image"), assets, f"{row.get('page_index'):03d}_{row.get('page_key')}_d{det.get('detection_index')}_crop.jpg")
            md.extend(
                [
                    f"### D{int(det.get('detection_index') or 0) + 1}",
                    "",
                    f"- auto_usable_probe：`{det.get('auto_usable_probe')}`，valid_geometry：`{det.get('valid_geometry')}`",
                    f"- bbox_coord_system：`{det.get('bbox_coord_system')}`",
                    f"- raw_target_bbox_page_px：`{det.get('raw_target_bbox_page_px')}`",
                    f"- raw_caption_bbox_page_px：`{det.get('raw_caption_bbox_page_px')}`",
                    f"- target_bbox_page_px：`{det.get('target_bbox_page_px')}`",
                    f"- caption_bbox_page_px：`{det.get('caption_bbox_page_px')}`",
                    f"- caption_bbox_refine：`{json.dumps(det.get('caption_bbox_refine'), ensure_ascii=False)}`",
                    f"- caption_text：{det.get('caption_text')}",
                    f"- object_domain：`{det.get('object_domain')}`，object_domain_gate：`{json.dumps(det.get('object_domain_gate'), ensure_ascii=False)}`，image_scope：`{det.get('image_scope')}`",
                    f"- caption_target_match：`{det.get('caption_target_match')}`，accept_for_sft：`{det.get('accept_for_sft')}`，needs_human_review：`{det.get('needs_human_review')}`，confidence：`{det.get('confidence')}`",
                    f"- reason：{det.get('reason')}",
                    "",
                ]
            )
            if crop_rel:
                md.extend([f"![crop]({crop_rel})", ""])
    path = package_dir / f"v1.0.5PageLevelVLMProbe_{len(reviewed)}页人工查看包.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


def build_summary(output_dir: Path, args: argparse.Namespace, reviewed: list[dict[str, Any]], detections: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [item for item in detections if item.get("auto_usable_probe")]
    return {
        "version": "v1.0.5_page_level_vlm_probe",
        "output_dir": str(output_dir),
        "pilot_size": args.pilot_size,
        "reviewed_pages": len(reviewed),
        "detections_total": len(detections),
        "auto_usable_probe_total": len(usable),
        "detections_per_page": {
            "zero_detection_pages": sum(1 for row in reviewed if not row.get("detections")),
            "avg": round(len(detections) / max(1, len(reviewed)), 3),
        },
        "model_counts": dict(Counter(row.get("review_model") for row in reviewed)),
        "object_domain_counts": dict(Counter(item.get("object_domain") for item in detections)),
        "accept_for_sft_counts": dict(Counter(str(item.get("accept_for_sft")) for item in detections)),
        "valid_geometry_counts": dict(Counter(str(item.get("valid_geometry")) for item in detections)),
        "auto_usable_by_split": dict(Counter(item.get("split") for item in usable)),
        "artifacts": {
            "stream": str(output_dir / "review" / "page_level_vlm_stream.jsonl"),
            "reviewed": str(output_dir / "review" / "page_level_vlm_reviewed.jsonl"),
            "detections": str(output_dir / "page_level_detections.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.5 Page-level VLM Probe 构建报告",
        "",
        "## 摘要",
        "",
        f"- reviewed_pages：{summary['reviewed_pages']}",
        f"- detections_total：{summary['detections_total']}",
        f"- auto_usable_probe_total：{summary['auto_usable_probe_total']}",
        f"- detections_per_page：`{json.dumps(summary['detections_per_page'], ensure_ascii=False)}`",
        "",
        "## 分布",
        "",
        f"- model_counts：`{json.dumps(summary['model_counts'], ensure_ascii=False)}`",
        f"- object_domain_counts：`{json.dumps(summary['object_domain_counts'], ensure_ascii=False)}`",
        f"- accept_for_sft_counts：`{json.dumps(summary['accept_for_sft_counts'], ensure_ascii=False)}`",
        f"- valid_geometry_counts：`{json.dumps(summary['valid_geometry_counts'], ensure_ascii=False)}`",
        f"- auto_usable_by_split：`{json.dumps(summary['auto_usable_by_split'], ensure_ascii=False)}`",
        "",
        "## 产物",
        "",
    ]
    for key, value in summary["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def page_to_json(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_index": page.get("page_index"),
        "page_key": page.get("page_key"),
        "source_file": page.get("source_file"),
        "source_stem": page.get("source_stem"),
        "page": page.get("page"),
        "split": page.get("split"),
        "candidate_count_on_page": page.get("candidate_count_on_page"),
        "candidate_task_ids": page.get("candidate_task_ids"),
        "candidate_captions": page.get("candidate_captions"),
        "page_width": page.get("page_width"),
        "page_height": page.get("page_height"),
    }


def candidate_key_from_candidate(candidate: v09.PageCandidate) -> tuple[Any, ...]:
    return (
        candidate.source_file,
        candidate.page,
        tuple(candidate.image_bbox),
        tuple(candidate.caption_bbox or []),
        v105.normalize_space(candidate.caption_text),
    )


def candidate_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("source_file"),
        int(row.get("page") or 0),
        tuple(row.get("image_bbox") or []),
        tuple(row.get("caption_bbox") or []),
        v105.normalize_space(row.get("caption_text")),
    )


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_existing(path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not args.resume or not path.exists():
        return {}
    out = {}
    for row in read_jsonl(path):
        if row.get("page_key"):
            out[str(row["page_key"])] = row
    return out


def write_text_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_asset(path_value: Any, assets_dir: Path, name: str) -> str:
    if not path_value:
        return ""
    src = Path(str(path_value))
    if not src.exists() or src.is_dir():
        return ""
    dst = assets_dir / safe_name(name)
    shutil.copy2(src, dst)
    return "assets/" + dst.name


def safe_name(value: Any) -> str:
    text = str(value)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    suffix = ""
    stem = text
    match = re.match(r"^(.*?)(\.[A-Za-z0-9]{1,8})$", text)
    if match:
        stem, suffix = match.group(1), match.group(2)
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    clean = clean[:120] or "asset"
    return f"{clean}_{digest}{suffix}"


if __name__ == "__main__":
    raise SystemExit(main())
