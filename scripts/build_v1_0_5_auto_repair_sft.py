#!/usr/bin/env python3
"""Repair v1.0.5 visual-audited Core4 SFT candidates.

This script starts from the v1.0.5 visual-audit output.  It keeps the already
accepted rows, then attempts two conservative automatic repairs:

1. caption_line_repair: target bbox is clean, but the caption bbox captured
   only the first line.  We extend the caption with adjacent PDF text blocks.
2. bbox_trim_caption: target bbox includes the caption.  We trim the target
   bbox away from the caption and extend the caption if needed.

Every repaired candidate is re-rendered and re-audited by the VLM before it is
allowed into the resulting SFT dataset.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_agentbench_v0_9_fixedsplit_train_multitarget as v09  # noqa: E402
import build_gold_eval_v1_0_4 as gold_review  # noqa: E402
import build_v1_0_4_core4_clean_sft as core4  # noqa: E402
import build_v1_0_4_core4_dedup_expanded_sft as dedup  # noqa: E402
import build_v1_0_5_core4_visual_audited_sft as v105  # noqa: E402
from evidence_agent_env.data import EvidenceIndex  # noqa: E402


DEFAULT_BASE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_5_core4_visual_audited_sft_20260612_1810"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DEFAULT_DOTENV = Path("/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/.env")
ALLOWED_DOMAINS = {"landscape_painting", "landscape_detail"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1.0.5 Core4 auto-repaired SFT.")
    parser.add_argument("--mode", choices=["probe", "full"], default="probe")
    parser.add_argument("--base-dir", default=str(DEFAULT_BASE_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dotenv", default=str(DEFAULT_DOTENV))
    parser.add_argument("--provider", choices=["dashscope", "offline"], default="dashscope")
    parser.add_argument("--model", default="qwen3.7-max-2026-06-08")
    parser.add_argument(
        "--fallback-models",
        default="qwen3.7-max-preview,qwen3.7-plus-2026-05-26,qwen3.7-plus,qwen3.6-plus,qwen3.5-plus-2026-04-20,glm-5.1,kimi-k2.6,deepseek-v4-pro,deepseek-v4-flash",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--image-max-side", type=int, default=1400)
    parser.add_argument("--crop-max-side", type=int, default=900)
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--probe-caption-limit", type=int, default=70)
    parser.add_argument("--probe-bbox-limit", type=int, default=50)
    parser.add_argument("--review-package-rows", type=int, default=200)
    parser.add_argument("--min-confidence", type=float, default=0.78)
    parser.add_argument("--max-caption-target-overlap", type=float, default=0.05)
    parser.add_argument("--train-caption-cap", type=int, default=2)
    parser.add_argument("--eval-caption-cap", type=int, default=1)
    parser.add_argument("--enable-corrected-caption-second-pass", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--second-pass-min-confidence", type=float, default=0.88)
    parser.add_argument("--corrected-caption-min-similarity", type=float, default=0.50)
    parser.add_argument("--second-pass-max-candidates", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_review.load_dotenv(Path(args.dotenv))
    base_dir = Path(args.base_dir)
    base_manifest = read_json(base_dir / "manifest.json")
    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args)

    build_args = base_build_args(base_manifest, args)
    rng = random.Random(int(build_args.seed))
    selected, scan_summary, filter_rows, split_docs = load_selected_candidates(build_args, rng)
    write_json(output_dir / "_split_map.json", split_docs)
    write_jsonl(output_dir / "filter_decisions.jsonl", filter_rows)
    write_jsonl(output_dir / "selected_candidates.jsonl", v105.candidate_rows(selected, split_docs))

    base_reviews = v105.read_jsonl(base_dir / "review" / "visual_audit_reviewed.jsonl")
    base_reviews_by_id = {str(row.get("task_id")): row for row in base_reviews}
    accepted_base_reviews = [row for row in base_reviews if row.get("visual_audit_status") == "accepted_sft"]
    repair_specs_all = build_repair_specs(base_reviews, selected)
    repair_specs = limit_repair_specs(repair_specs_all, args)
    write_jsonl(output_dir / "repair_candidates_all.jsonl", [spec.to_json() for spec in repair_specs_all])
    write_jsonl(output_dir / "repair_candidates_selected.jsonl", [spec.to_json() for spec in repair_specs])

    index = EvidenceIndex(str(build_args.evidence_index_dir))
    page_cache: dict[tuple[str, int], Path] = {}
    task_cache: dict[int, dict[str, Any]] = {}
    repaired_selected = list(selected)
    repair_stream_path = output_dir / "review" / "repair_audit_stream.jsonl"
    existing = load_existing_repair_reviews(repair_stream_path, args)
    repair_reviewed: list[dict[str, Any]] = []
    repair_errors: list[dict[str, Any]] = []
    skipped_repairs: list[dict[str, Any]] = []
    client = v105.make_client(args)

    for pos, spec in enumerate(repair_specs, start=1):
        original = selected[spec.selected_index]
        try:
            repaired, repair_meta = apply_repair_spec(original, spec)
            if repaired is None:
                skipped_repairs.append({**spec.to_json(), "skip_reason": repair_meta.get("skip_reason")})
                continue
            repaired_selected[spec.selected_index] = repaired
            row = existing.get(spec.task_id)
            core5_task = v105.materialize_core5_task(
                spec.selected_index,
                repaired,
                output_dir,
                build_args,
                index,
                page_cache,
                task_cache,
            )
            preview_task, _ = core4.transform_task(core5_task, caption_overrides={})
            if row is None:
                row = v105.review_one(spec.selected_index, preview_task, repaired, client, build_args)
                row["repair_type"] = spec.repair_type
                row["repair_meta"] = repair_meta
                v105.append_jsonl(repair_stream_path, [row])
                if args.sleep:
                    time.sleep(args.sleep)
            row = normalize_repair_review_row(row, spec, preview_task, repaired, build_args)
            repair_reviewed.append(row)
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "progress": f"{pos}/{len(repair_specs)}",
                        "task_id": spec.task_id,
                        "repair_type": spec.repair_type,
                        "status": row.get("repair_audit_status"),
                        "model": row.get("review_model"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            repair_errors.append({**spec.to_json(), "error": f"{type(exc).__name__}: {exc}"})

    second_pass_reviewed: list[dict[str, Any]] = []
    second_pass_errors: list[dict[str, Any]] = []
    second_pass_skipped: list[dict[str, Any]] = []
    if args.enable_corrected_caption_second_pass:
        second_pass_reviewed, second_pass_skipped, second_pass_errors = run_corrected_caption_second_pass(
            repair_reviewed,
            repaired_selected,
            output_dir,
            build_args,
            index,
            page_cache,
            task_cache,
            client,
            args,
        )

    accepted_repair_reviews = [row for row in repair_reviewed if row.get("repair_audit_status") == "accepted_sft"]
    accepted_second_pass_reviews = [row for row in second_pass_reviewed if row.get("repair_audit_status") == "accepted_sft"]
    accepted_repair_reviews.extend(accepted_second_pass_reviews)
    combined_reviews, cap_skipped = apply_caption_caps(accepted_base_reviews, accepted_repair_reviews, repaired_selected, args)
    output_reviews = normalize_base_rows_for_output(accepted_base_reviews, base_reviews_by_id) + repair_reviewed + second_pass_reviewed
    write_jsonl(output_dir / "review" / "repair_audit_reviewed.jsonl", repair_reviewed)
    write_jsonl(output_dir / "review" / "corrected_caption_second_pass_reviewed.jsonl", second_pass_reviewed)
    write_jsonl(output_dir / "review" / "accepted_repair_review.jsonl", accepted_repair_reviews)
    write_jsonl(output_dir / "review" / "repair_skipped.jsonl", skipped_repairs)
    write_jsonl(output_dir / "review" / "corrected_caption_second_pass_skipped.jsonl", second_pass_skipped)
    write_jsonl(output_dir / "review" / "repair_cap_skipped.jsonl", cap_skipped)

    tasks_by_split, episodes_by_split, sft_by_split, accepted_reviews, review_queue, rejected, sft_errors = v105.build_outputs_from_reviews(
        repaired_selected,
        combined_reviews,
        output_dir,
        build_args,
        index,
        page_cache,
        task_cache,
    )
    for row in accepted_reviews:
        row["label_source"] = "v1_0_5_auto_repaired_sft"
    builder_errors = repair_errors + second_pass_errors + sft_errors

    all_tasks = [task for split in ["train", "val", "test"] for task in tasks_by_split.get(split, [])]
    for task in all_tasks:
        task["dataset_version"] = "v1.0.5_auto_repaired_sft"
        task["runtime_mode"] = "v1_0_5_auto_repaired_core4_docsplit_caption_cap"
        task["tool_schema_version"] = "v1.0.5_no_select_core4_auto_repaired"
        task.setdefault("gold", {})["label_source"] = "v1_0_5_auto_repaired_core4_sft"
        if task.get("task_id") in {row.get("task_id") for row in accepted_repair_reviews}:
            task["auto_repair"] = repair_lookup(accepted_repair_reviews).get(task.get("task_id"))
    all_episodes = [ep for split in ["train", "val", "test"] for ep in episodes_by_split.get(split, [])]
    all_sft = [row for split in ["train", "val", "test"] for row in sft_by_split.get(split, [])]
    for row in all_sft:
        row["label_source"] = "v1_0_5_auto_repaired_sft"
        row["tool_schema_version"] = "v1.0.5_no_select_core4_auto_repaired"

    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", [task for task in all_tasks if task.get("split") == split])
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", [ep for ep in all_episodes if ep.get("split") == split])
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", [row for row in all_sft if row.get("split") == split])
    write_jsonl(output_dir / "tasks_all.jsonl", all_tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", all_episodes)
    write_jsonl(output_dir / "sft" / "all.jsonl", all_sft)
    write_jsonl(output_dir / "review" / "visual_audit_reviewed.jsonl", output_reviews)
    write_jsonl(output_dir / "review" / "accepted_review.jsonl", accepted_reviews)
    write_jsonl(output_dir / "review" / "review_queue.jsonl", review_queue)
    write_jsonl(output_dir / "review" / "rejected.jsonl", rejected)
    write_jsonl(output_dir / "builder_errors.jsonl", builder_errors)

    gold_eval_summary = core4.build_gold_eval_core4(
        Path(build_args.gold_eval_dir), output_dir / "gold_eval", core4.load_caption_overrides(Path(build_args.gold_eval_dir))
    )
    summary = build_summary(
        args,
        build_args,
        base_manifest,
        scan_summary,
        filter_rows,
        repair_specs_all,
        repair_specs,
        repair_reviewed,
        accepted_repair_reviews,
        second_pass_reviewed,
        accepted_second_pass_reviews,
        cap_skipped,
        skipped_repairs + second_pass_skipped,
        all_tasks,
        all_sft,
        builder_errors,
        gold_eval_summary,
        output_dir,
    )
    write_json(output_dir / "manifest.json", summary)
    write_report(output_dir / "构建报告.md", summary)
    package_path = v105.write_review_package(output_dir, repair_reviewed + second_pass_reviewed, args.review_package_rows)
    summary["artifacts"]["repair_review_package"] = str(package_path)
    write_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


class RepairSpec:
    def __init__(self, selected_index: int, task_id: str, repair_type: str, source_status: str, source_flags: list[str], reason: str):
        self.selected_index = selected_index
        self.task_id = task_id
        self.repair_type = repair_type
        self.source_status = source_status
        self.source_flags = source_flags
        self.reason = reason

    def to_json(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "task_id": self.task_id,
            "repair_type": self.repair_type,
            "source_status": self.source_status,
            "source_flags": self.source_flags,
            "reason": self.reason,
        }


def build_repair_specs(base_reviews: list[dict[str, Any]], selected: list[v09.PageCandidate]) -> list[RepairSpec]:
    specs: list[RepairSpec] = []
    for row in sorted(base_reviews, key=lambda item: int(item.get("selected_index") or 0)):
        idx = int(row.get("selected_index") or 0)
        if idx < 0 or idx >= len(selected):
            continue
        decision = row.get("decision") or {}
        status = str(row.get("visual_audit_status") or "")
        domain = str(decision.get("object_domain") or "")
        target_error = str(decision.get("target_box_error") or "")
        boundary = str(decision.get("caption_boundary") or "")
        if (
            status == "needs_human_review"
            and decision.get("target_box_ok") == "yes"
            and target_error == "none"
            and decision.get("caption_target_match") == "yes"
            and domain in ALLOWED_DOMAINS
            and boundary in {"truncated", "minor_truncation_correctable"}
        ):
            specs.append(
                RepairSpec(
                    idx,
                    str(row.get("task_id")),
                    "caption_line_repair",
                    status,
                    list(row.get("strict_gate_flags") or []),
                    str(decision.get("reason") or ""),
                )
            )
        elif (
            status == "rejected"
            and target_error in {"includes_caption", "multi_figures"}
            and decision.get("caption_target_match") == "yes"
            and domain in ALLOWED_DOMAINS
        ):
            specs.append(
                RepairSpec(
                    idx,
                    str(row.get("task_id")),
                    "bbox_trim_caption",
                    status,
                    list(row.get("strict_gate_flags") or []),
                    str(decision.get("reason") or ""),
                )
            )
    return specs


def limit_repair_specs(specs: list[RepairSpec], args: argparse.Namespace) -> list[RepairSpec]:
    if args.mode == "full":
        return specs
    by_type: dict[str, list[RepairSpec]] = defaultdict(list)
    for spec in specs:
        by_type[spec.repair_type].append(spec)
    return by_type["caption_line_repair"][: args.probe_caption_limit] + by_type["bbox_trim_caption"][: args.probe_bbox_limit]


def apply_repair_spec(candidate: v09.PageCandidate, spec: RepairSpec) -> tuple[v09.PageCandidate | None, dict[str, Any]]:
    caption_text, caption_bbox, caption_meta = repair_caption_from_text_blocks(candidate)
    if not caption_text or not caption_bbox:
        return None, {"skip_reason": "caption_text_blocks_not_repairable", **caption_meta, "repair_type": spec.repair_type}
    image_bbox = list(candidate.image_bbox)
    bbox_meta: dict[str, Any] = {}
    if spec.repair_type == "bbox_trim_caption":
        fixed_bbox, bbox_meta = repair_image_bbox_away_from_caption(candidate.image_bbox, caption_bbox, candidate)
        if fixed_bbox is None:
            return None, {"skip_reason": "image_bbox_not_repairable", **caption_meta, **bbox_meta, "repair_type": spec.repair_type}
        image_bbox = fixed_bbox
    area_ratio = bbox_area(image_bbox) / max(1.0, float(candidate.page_width * candidate.page_height))
    repaired = replace(
        candidate,
        image_bbox=[int(v) for v in image_bbox],
        caption_bbox=[int(v) for v in caption_bbox],
        caption_text=caption_text,
        area_ratio=round(area_ratio, 6),
        caption_score=max(float(candidate.caption_score or 0.0), 8.0),
        target_variant=700000 + int(spec.selected_index),
        target_source=str(candidate.target_source or "") + "_auto_repaired",
    )
    meta = {
        "repair_type": spec.repair_type,
        "original_caption_text": candidate.caption_text,
        "repaired_caption_text": caption_text,
        "original_caption_bbox": candidate.caption_bbox,
        "repaired_caption_bbox": caption_bbox,
        "original_image_bbox": candidate.image_bbox,
        "repaired_image_bbox": image_bbox,
        "caption_repair": caption_meta,
        "bbox_repair": bbox_meta,
    }
    return repaired, meta


def repair_caption_from_text_blocks(candidate: v09.PageCandidate) -> tuple[str, list[int] | None, dict[str, Any]]:
    caption_bbox = candidate.caption_bbox
    if not caption_bbox:
        return "", None, {"method": "same_column_text_blocks", "lines": []}
    cx1, cy1, cx2, cy2 = [int(v) for v in caption_bbox]
    cap_width = max(1, cx2 - cx1)
    blocks = sorted((dict(block) for block in candidate.text_blocks), key=lambda item: (bbox(item)[1], bbox(item)[0]))
    candidates: list[dict[str, Any]] = []
    for block in blocks:
        bb = bbox(block)
        text = normalize_space(block.get("text"))
        if not text:
            continue
        if bb[1] < cy1 - 8:
            continue
        if bb[1] > cy1 + 260:
            continue
        if same_caption_column(caption_bbox, bb, cap_width):
            candidates.append({"bbox": bb, "text": text})
    if not candidates:
        return "", None, {"method": "same_column_text_blocks", "lines": []}
    start = min(range(len(candidates)), key=lambda i: abs(candidates[i]["bbox"][1] - cy1) + abs(candidates[i]["bbox"][0] - cx1))
    lines: list[dict[str, Any]] = []
    prev_bottom = None
    for item in candidates[start:]:
        text = item["text"]
        bb = item["bbox"]
        if lines and is_caption_start(text):
            break
        if prev_bottom is not None and bb[1] - prev_bottom > 38:
            break
        lines.append(item)
        prev_bottom = bb[3]
        if len(lines) >= 8:
            break
    if not lines:
        return "", None, {"method": "same_column_text_blocks", "lines": []}
    original = normalize_space(candidate.caption_text)
    joined = normalize_space(" ".join(line["text"] for line in lines))
    if original and original.casefold() not in joined.casefold():
        joined_alt = normalize_space(original + " " + " ".join(line["text"] for line in lines[1:]))
        if len(joined_alt) > len(joined):
            joined = joined_alt
    if len(joined) < max(12, len(original)):
        return "", None, {"method": "same_column_text_blocks", "lines": lines, "skip_reason": "joined_caption_not_longer"}
    union = union_bbox([line["bbox"] for line in lines])
    return joined[:500], union, {"method": "same_column_text_blocks", "lines": lines, "line_count": len(lines)}


def same_caption_column(caption_bbox: list[int], block_bbox: list[int], cap_width: int) -> bool:
    cx1, _, cx2, _ = caption_bbox
    bx1, _, bx2, _ = block_bbox
    overlap = max(0, min(cx2, bx2) - max(cx1, bx1))
    overlap_ratio = overlap / max(1, min(cx2 - cx1, bx2 - bx1))
    left_close = abs(bx1 - cx1) <= max(70, int(cap_width * 0.35))
    center_close = abs(((bx1 + bx2) / 2) - ((cx1 + cx2) / 2)) <= max(90, int(cap_width * 0.45))
    return overlap_ratio >= 0.35 or (left_close and center_close)


def repair_image_bbox_away_from_caption(
    image_bbox: list[int], caption_bbox: list[int], candidate: v09.PageCandidate
) -> tuple[list[int] | None, dict[str, Any]]:
    ix1, iy1, ix2, iy2 = [int(v) for v in image_bbox]
    cx1, cy1, cx2, cy2 = [int(v) for v in caption_bbox]
    pad = 8
    new_bbox = [ix1, iy1, ix2, iy2]
    if bbox_overlap_ratio(image_bbox, caption_bbox, denominator="caption") > 0.01:
        if cy1 >= iy1 + int((iy2 - iy1) * 0.45):
            new_bbox[3] = max(iy1 + 40, cy1 - pad)
        elif cy2 <= iy1 + int((iy2 - iy1) * 0.55):
            new_bbox[1] = min(iy2 - 40, cy2 + pad)
        elif cx1 >= ix1 + int((ix2 - ix1) * 0.45):
            new_bbox[2] = max(ix1 + 40, cx1 - pad)
        elif cx2 <= ix1 + int((ix2 - ix1) * 0.55):
            new_bbox[0] = min(ix2 - 40, cx2 + pad)
    else:
        return image_bbox, {"method": "no_overlap_keep_original"}
    if not valid_image_bbox(new_bbox, candidate):
        alternative = best_alternative_image_block(candidate, caption_bbox)
        if alternative:
            return alternative, {"method": "alternative_image_block", "trim_candidate": new_bbox}
        return None, {"method": "trim_away_caption", "invalid_bbox": new_bbox}
    return new_bbox, {"method": "trim_away_caption"}


def best_alternative_image_block(candidate: v09.PageCandidate, caption_bbox: list[int]) -> list[int] | None:
    best: tuple[float, list[int]] | None = None
    original = candidate.image_bbox
    for block in candidate.image_blocks:
        bb = list(block.get("bbox") or [])
        if not valid_image_bbox(bb, candidate):
            continue
        if bbox_overlap_ratio(bb, caption_bbox, denominator="caption") > 0.05:
            continue
        iou = bbox_iou(original, bb)
        if iou <= 0:
            continue
        score = iou + 0.1 * float(block.get("caption_score") or 0.0)
        if best is None or score > best[0]:
            best = (score, [int(v) for v in bb])
    return best[1] if best else None


def run_corrected_caption_second_pass(
    repair_reviewed: list[dict[str, Any]],
    repaired_selected: list[v09.PageCandidate],
    output_dir: Path,
    build_args: argparse.Namespace,
    index: EvidenceIndex,
    page_cache: dict[tuple[str, int], Path],
    task_cache: dict[int, dict[str, Any]],
    client: v105.VisualAuditClient,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = corrected_caption_second_pass_candidates(repair_reviewed, args)
    if args.second_pass_max_candidates > 0:
        candidates = candidates[: args.second_pass_max_candidates]
    stream_path = output_dir / "review" / "corrected_caption_second_pass_stream.jsonl"
    existing = load_existing_repair_reviews(stream_path, args)
    reviewed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for pos, first_pass_row in enumerate(candidates, start=1):
        try:
            idx = int(first_pass_row.get("selected_index") or 0)
            corrected_candidate, meta = apply_corrected_caption_second_pass(
                repaired_selected[idx],
                first_pass_row,
                args,
            )
            spec = RepairSpec(
                idx,
                str(first_pass_row.get("task_id")),
                "corrected_caption_second_pass",
                str(first_pass_row.get("repair_audit_status") or first_pass_row.get("visual_audit_status") or ""),
                list(first_pass_row.get("repair_gate_flags") or first_pass_row.get("strict_gate_flags") or []),
                str((first_pass_row.get("decision") or {}).get("reason") or ""),
            )
            if corrected_candidate is None:
                skipped.append({**spec.to_json(), **meta})
                continue
            row = existing.get(spec.task_id)
            task_cache.pop(idx, None)
            core5_task = v105.materialize_core5_task(
                idx,
                corrected_candidate,
                output_dir,
                build_args,
                index,
                page_cache,
                task_cache,
            )
            preview_task, _ = core4.transform_task(core5_task, caption_overrides={})
            if row is None:
                row = v105.review_one(idx, preview_task, corrected_candidate, client, build_args)
                row["repair_type"] = spec.repair_type
                row["repair_meta"] = meta
                v105.append_jsonl(stream_path, [row])
                if args.sleep:
                    time.sleep(args.sleep)
            row = normalize_repair_review_row(row, spec, preview_task, corrected_candidate, build_args)
            row["first_pass_repair_type"] = first_pass_row.get("repair_type")
            row["first_pass_decision"] = first_pass_row.get("decision")
            if row.get("repair_audit_status") == "accepted_sft":
                repaired_selected[idx] = corrected_candidate
            else:
                task_cache.pop(idx, None)
            reviewed.append(row)
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "second_pass_progress": f"{pos}/{len(candidates)}",
                        "task_id": spec.task_id,
                        "status": row.get("repair_audit_status"),
                        "model": row.get("review_model"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            errors.append(
                {
                    "task_id": first_pass_row.get("task_id"),
                    "selected_index": first_pass_row.get("selected_index"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return reviewed, skipped, errors


def corrected_caption_second_pass_candidates(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        decision = row.get("decision") or {}
        if row.get("repair_audit_status") == "accepted_sft":
            continue
        if row.get("repair_audit_status") not in {"needs_human_review", "rejected"}:
            continue
        if decision.get("target_box_ok") != "yes" or decision.get("target_box_error") != "none":
            continue
        if decision.get("caption_target_match") != "yes":
            continue
        if decision.get("object_domain") not in ALLOWED_DOMAINS:
            continue
        if decision.get("caption_quality") not in {"clean", "minor_ocr_noise", "ocr_noise"}:
            continue
        if v105.safe_float(decision.get("confidence"), 0.0) < args.second_pass_min_confidence:
            continue
        corrected = normalize_space(decision.get("corrected_caption_text"))
        original = normalize_space(row.get("caption_text"))
        if not corrected or len(corrected) < 20:
            continue
        if corrected == original:
            continue
        out.append(row)
    return sorted(out, key=lambda item: int(item.get("selected_index") or 0))


def apply_corrected_caption_second_pass(
    candidate: v09.PageCandidate,
    first_pass_row: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[v09.PageCandidate | None, dict[str, Any]]:
    decision = first_pass_row.get("decision") or {}
    corrected = normalize_space(decision.get("corrected_caption_text"))
    if not corrected:
        return None, {"skip_reason": "missing_corrected_caption_text"}
    located_text, located_bbox, locate_meta = locate_corrected_caption_bbox(candidate, corrected, args)
    if not located_bbox:
        return None, {"skip_reason": "corrected_caption_not_located_in_page_text", **locate_meta}
    image_bbox = list(candidate.image_bbox)
    fixed_bbox, bbox_meta = repair_image_bbox_away_from_caption(image_bbox, located_bbox, candidate)
    if fixed_bbox is None:
        return None, {"skip_reason": "corrected_caption_image_bbox_not_repairable", **locate_meta, **bbox_meta}
    area_ratio = bbox_area(fixed_bbox) / max(1.0, float(candidate.page_width * candidate.page_height))
    repaired = replace(
        candidate,
        image_bbox=[int(v) for v in fixed_bbox],
        caption_bbox=[int(v) for v in located_bbox],
        caption_text=corrected,
        area_ratio=round(area_ratio, 6),
        caption_score=max(float(candidate.caption_score or 0.0), 9.0),
        target_variant=800000 + int(first_pass_row.get("selected_index") or 0),
        target_source=str(candidate.target_source or "") + "_corrected_caption_second_pass",
    )
    meta = {
        "repair_type": "corrected_caption_second_pass",
        "source_repair_type": first_pass_row.get("repair_type"),
        "original_caption_text": candidate.caption_text,
        "first_pass_caption_text": first_pass_row.get("caption_text"),
        "corrected_caption_text": corrected,
        "page_text_aligned_caption": located_text,
        "original_caption_bbox": candidate.caption_bbox,
        "corrected_caption_bbox": located_bbox,
        "original_image_bbox": candidate.image_bbox,
        "corrected_image_bbox": fixed_bbox,
        "caption_locate": locate_meta,
        "bbox_repair": bbox_meta,
    }
    return repaired, meta


def locate_corrected_caption_bbox(
    candidate: v09.PageCandidate,
    corrected_caption: str,
    args: argparse.Namespace,
) -> tuple[str, list[int] | None, dict[str, Any]]:
    if not candidate.caption_bbox:
        return "", None, {"method": "corrected_caption_fuzzy_prefix", "lines": []}
    cx1, cy1, cx2, _cy2 = [int(v) for v in candidate.caption_bbox]
    cap_width = max(1, cx2 - cx1)
    blocks = sorted((dict(block) for block in candidate.text_blocks), key=lambda item: (bbox(item)[1], bbox(item)[0]))
    lines: list[dict[str, Any]] = []
    for block in blocks:
        bb = bbox(block)
        text = normalize_space(block.get("text"))
        if not text:
            continue
        if bb[1] < cy1 - 20 or bb[1] > cy1 + 360:
            continue
        if same_caption_column(candidate.caption_bbox, bb, cap_width):
            lines.append({"bbox": bb, "text": text})
    if not lines:
        return "", None, {"method": "corrected_caption_fuzzy_prefix", "lines": []}
    start = min(range(len(lines)), key=lambda i: abs(lines[i]["bbox"][1] - cy1) + abs(lines[i]["bbox"][0] - cx1))
    prefix_lines: list[dict[str, Any]] = []
    prev_bottom = None
    for item in lines[start:]:
        if prefix_lines and is_caption_start(item["text"]):
            break
        if prev_bottom is not None and item["bbox"][1] - prev_bottom > 58:
            break
        prefix_lines.append(item)
        prev_bottom = item["bbox"][3]
        if len(prefix_lines) >= 8:
            break
    if not prefix_lines:
        return "", None, {"method": "corrected_caption_fuzzy_prefix", "lines": lines[:8], "skip_reason": "no_prefix_lines"}
    best_score = -1.0
    best_count = 0
    best_text = ""
    corrected_norm = fuzzy_caption_norm(corrected_caption)
    for count in range(1, len(prefix_lines) + 1):
        text = normalize_space(" ".join(line["text"] for line in prefix_lines[:count]))
        score = caption_similarity(text, corrected_caption)
        norm_text = fuzzy_caption_norm(text)
        if corrected_norm and (corrected_norm in norm_text or norm_text in corrected_norm):
            score += 0.12
        if score > best_score:
            best_score = score
            best_count = count
            best_text = text
    if best_score < args.corrected_caption_min_similarity:
        return "", None, {
            "method": "corrected_caption_fuzzy_prefix",
            "lines": prefix_lines,
            "best_score": round(best_score, 4),
            "best_text": best_text,
            "skip_reason": "similarity_below_threshold",
        }
    chosen = prefix_lines[:best_count]
    return best_text, union_bbox([line["bbox"] for line in chosen]), {
        "method": "corrected_caption_fuzzy_prefix",
        "lines": chosen,
        "line_count": len(chosen),
        "best_score": round(best_score, 4),
    }


def caption_similarity(left: str, right: str) -> float:
    a = fuzzy_caption_norm(left)
    b = fuzzy_caption_norm(right)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_caption_norm(text: str) -> str:
    text = normalize_space(text).casefold()
    text = text.replace("圖", "图")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("．", ".").replace("·", ".")
    text = text.replace("em", "cm")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def normalize_repair_review_row(
    row: dict[str, Any],
    spec: RepairSpec,
    task: dict[str, Any],
    candidate: v09.PageCandidate,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = v105.normalize_review_row(row, spec.selected_index, task, candidate, args)
    out["repair_type"] = spec.repair_type
    out["source_visual_audit_status"] = spec.source_status
    out["source_strict_gate_flags"] = spec.source_flags
    out["source_reason"] = spec.reason
    status, flags = classify_repair_decision(out, task, args)
    out["repair_audit_status"] = status
    out["repair_gate_flags"] = flags
    if status == "accepted_sft":
        out["visual_audit_status"] = "accepted_sft"
        out["strict_gate_flags"] = []
    return out


def classify_repair_decision(row: dict[str, Any], task: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    flags: list[str] = []
    decision = row.get("decision") or {}
    if not row.get("ok"):
        return "needs_human_review", ["vlm_audit_failed"]
    if v105.safe_float(decision.get("confidence"), 0.0) < args.min_confidence:
        flags.append("confidence_below_threshold")
    if bbox_overlap_ratio((task.get("gold") or {}).get("image_bbox"), (task.get("gold") or {}).get("caption_bbox"), "caption") > args.max_caption_target_overlap:
        flags.append("caption_bbox_overlaps_target_bbox")
    if decision.get("target_box_ok") != "yes":
        flags.append("target_box_not_yes")
    if decision.get("target_box_error") != "none":
        flags.append("target_box_error_" + str(decision.get("target_box_error")))
    if decision.get("caption_target_match") != "yes":
        flags.append("caption_target_match_not_yes")
    if decision.get("object_domain") not in ALLOWED_DOMAINS:
        flags.append("object_domain_not_allowed")
    if decision.get("caption_quality") in {"body_text", "toc_or_index", "wrong_language"}:
        flags.append("caption_quality_bad")
    if decision.get("caption_boundary") not in {"complete", "minor_truncation_correctable"}:
        flags.append("caption_boundary_not_auto_clean")
    if decision.get("accept_for_sft") is not True:
        flags.append("vlm_did_not_accept_for_sft")
    if decision.get("needs_human_review") is True:
        flags.append("vlm_requested_human_review")
    if flags:
        if any(flag.startswith("target_box_error_") for flag in flags) or "caption_quality_bad" in flags:
            return "rejected", flags
        return "needs_human_review", flags
    return "accepted_sft", flags


def apply_caption_caps(
    accepted_base: list[dict[str, Any]],
    accepted_repair: list[dict[str, Any]],
    selected: list[v09.PageCandidate],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in sorted(accepted_base, key=lambda item: int(item.get("selected_index") or 0)):
        idx = int(row.get("selected_index") or 0)
        split = str(row.get("split") or "train")
        caption = dedup.caption_key(str(row.get("caption_text") or selected[idx].caption_text or ""))
        counts[split][caption] += 1
        combined.append(dict(row))
    for row in sorted(accepted_repair, key=lambda item: int(item.get("selected_index") or 0)):
        idx = int(row.get("selected_index") or 0)
        split = str(row.get("split") or "train")
        caption = dedup.caption_key(str(row.get("caption_text") or selected[idx].caption_text or ""))
        cap = args.train_caption_cap if split == "train" else args.eval_caption_cap
        if counts[split][caption] >= cap:
            skipped.append({"task_id": row.get("task_id"), "split": split, "caption_text": row.get("caption_text"), "reason": "caption_cap_exceeded"})
            continue
        counts[split][caption] += 1
        out = dict(row)
        out["visual_audit_status"] = "accepted_sft"
        out["strict_gate_flags"] = []
        combined.append(out)
    return combined, skipped


def normalize_base_rows_for_output(accepted_base: list[dict[str, Any]], base_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in accepted_base:
        out = dict(base_by_id.get(str(row.get("task_id")), row))
        out["source_visual_audit_status"] = "accepted_sft"
        out["repair_type"] = ""
        out["repair_audit_status"] = "accepted_sft"
        rows.append(out)
    return rows


def repair_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("task_id")): {
            "repair_type": row.get("repair_type"),
            "repair_meta": row.get("repair_meta"),
            "review_model": row.get("review_model"),
            "decision": row.get("decision"),
        }
        for row in rows
    }


def build_summary(
    args: argparse.Namespace,
    build_args: argparse.Namespace,
    base_manifest: dict[str, Any],
    scan_summary: dict[str, Any],
    filter_rows: list[dict[str, Any]],
    repair_specs_all: list[RepairSpec],
    repair_specs: list[RepairSpec],
    repair_reviewed: list[dict[str, Any]],
    accepted_repair: list[dict[str, Any]],
    second_pass_reviewed: list[dict[str, Any]],
    accepted_second_pass: list[dict[str, Any]],
    cap_skipped: list[dict[str, Any]],
    skipped_repairs: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    builder_errors: list[dict[str, Any]],
    gold_eval_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    split_counts = Counter(str(task.get("split") or "train") for task in tasks)
    sft_split_counts = Counter(str(row.get("split") or "train") for row in sft_rows)
    status_counts = Counter(row.get("repair_audit_status") for row in repair_reviewed)
    second_status_counts = Counter(row.get("repair_audit_status") for row in second_pass_reviewed)
    all_types = Counter(spec.repair_type for spec in repair_specs_all)
    selected_types = Counter(spec.repair_type for spec in repair_specs)
    accepted_types = Counter(row.get("repair_type") for row in accepted_repair)
    task_repair_ids = {row.get("task_id") for row in accepted_repair}
    fields = []
    for row in sft_rows:
        action = row.get("action") or {}
        if action.get("action") == "write_claims_chunk":
            fields.extend(claim.get("field") for claim in action.get("claims") or [])
            fields.extend(item.get("field") for item in action.get("abstains") or [])
    return {
        "created_at": now_cst(),
        "dataset_version": "v1.0.5_auto_repaired_sft",
        "builder": "scripts/build_v1_0_5_auto_repair_sft.py",
        "mode": args.mode,
        "output_dir": str(output_dir),
        "base_dir": args.base_dir,
        "base_visual_audit_counts": base_manifest.get("visual_audit_status_counts"),
        "provider": args.provider,
        "model": args.model if args.provider == "dashscope" else "offline_rules",
        "fallback_models": [item.strip() for item in args.fallback_models.split(",") if item.strip()],
        "repair_candidate_counts_all": dict(all_types),
        "repair_candidate_counts_selected": dict(selected_types),
        "repair_audit_status_counts": dict(status_counts),
        "accepted_repair_counts": dict(accepted_types),
        "corrected_caption_second_pass_enabled": bool(args.enable_corrected_caption_second_pass),
        "corrected_caption_second_pass_reviewed": len(second_pass_reviewed),
        "corrected_caption_second_pass_status_counts": dict(second_status_counts),
        "corrected_caption_second_pass_accepted": len(accepted_second_pass),
        "accepted_repair_tasks": len(task_repair_ids),
        "skipped_repair_count": len(skipped_repairs),
        "caption_cap_skipped_count": len(cap_skipped),
        "split_counts": dict(split_counts),
        "sft_split_counts": dict(sft_split_counts),
        "sft_rows_total": len(sft_rows),
        "field_set": sorted({field for field in fields if field}),
        "review_model_counts": dict(Counter(row.get("review_model") or "unknown" for row in repair_reviewed)),
        "repair_gate_flag_counts": dict(Counter(flag for row in repair_reviewed for flag in row.get("repair_gate_flags") or []).most_common(30)),
        "scan_summary": scan_summary,
        "filter_summary": {
            "decisions": len(filter_rows),
            "accepted_by_rule": sum(1 for row in filter_rows if row.get("keep")),
            "rejected_by_rule": sum(1 for row in filter_rows if not row.get("keep")),
        },
        "caption_cap_policy": {"train_caption_cap": args.train_caption_cap, "eval_caption_cap": args.eval_caption_cap},
        "builder_error_count": len(builder_errors),
        "builder_errors_preview": builder_errors[:20],
        "gold_eval_core4": gold_eval_summary,
        "artifacts": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "sft_all": str(output_dir / "sft" / "all.jsonl"),
            "repair_audit_stream": str(output_dir / "review" / "repair_audit_stream.jsonl"),
            "repair_audit_reviewed": str(output_dir / "review" / "repair_audit_reviewed.jsonl"),
            "corrected_caption_second_pass_reviewed": str(output_dir / "review" / "corrected_caption_second_pass_reviewed.jsonl"),
            "accepted_repair_review": str(output_dir / "review" / "accepted_repair_review.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
        "args": vars(args),
        "base_build_args": vars(build_args),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.5 Auto-Repaired Core4 SFT 构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 基础目录：`{summary['base_dir']}`",
        f"- 构建模式：`{summary['mode']}`",
        f"- VLM provider/model：`{summary['provider']}` / `{summary['model']}`",
        "",
        "## 结论",
        "",
        "- 本版本保留 v1.0.5 已自动接受样本，并尝试自动修复 caption 截断和 bbox 包含图注/多图的候选。",
        "- 修复候选必须重新生成 overlay/crop 并通过 VLM 复审后才进入 SFT。",
        "- caption 补全优先来自 PDF 页面 text_blocks；大模型只做复审，不单独充当 gold 来源。",
        "",
        "## 规模",
        "",
        f"- base visual audit：`{json.dumps(summary['base_visual_audit_counts'], ensure_ascii=False)}`",
        f"- repair_candidate_counts_all：`{json.dumps(summary['repair_candidate_counts_all'], ensure_ascii=False)}`",
        f"- repair_candidate_counts_selected：`{json.dumps(summary['repair_candidate_counts_selected'], ensure_ascii=False)}`",
        f"- repair_audit_status_counts：`{json.dumps(summary['repair_audit_status_counts'], ensure_ascii=False)}`",
        f"- accepted_repair_counts：`{json.dumps(summary['accepted_repair_counts'], ensure_ascii=False)}`",
        f"- corrected_caption_second_pass_enabled：{summary['corrected_caption_second_pass_enabled']}",
        f"- corrected_caption_second_pass_reviewed：{summary['corrected_caption_second_pass_reviewed']}",
        f"- corrected_caption_second_pass_status_counts：`{json.dumps(summary['corrected_caption_second_pass_status_counts'], ensure_ascii=False)}`",
        f"- corrected_caption_second_pass_accepted：{summary['corrected_caption_second_pass_accepted']}",
        f"- caption_cap_skipped_count：{summary['caption_cap_skipped_count']}",
        f"- split_counts：`{json.dumps(summary['split_counts'], ensure_ascii=False)}`",
        f"- sft_split_counts：`{json.dumps(summary['sft_split_counts'], ensure_ascii=False)}`，total={summary['sft_rows_total']}",
        f"- field_set：`{json.dumps(summary['field_set'], ensure_ascii=False)}`",
        "",
        "## VLM 复审",
        "",
        f"- review_model_counts：`{json.dumps(summary['review_model_counts'], ensure_ascii=False)}`",
        f"- repair_gate_flag_counts：`{json.dumps(summary['repair_gate_flag_counts'], ensure_ascii=False)}`",
        f"- builder_error_count：{summary['builder_error_count']}",
        "",
        "## 产物",
        "",
    ]
    for key, value in summary["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## 使用建议",
            "",
            "- probe 版本只用于检查修复策略质量；full 版本通过后再用于 fresh Core4 SFT / continued repair SFT 对照。",
            "- 若 bbox repair 的 rejected 比例较高，应优先改几何裁剪规则，不要放宽 VLM 审核门槛。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def base_build_args(base_manifest: dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    base_args = dict(base_manifest.get("args") or {})
    base_args.update(
        {
            "output_dir": "",
            "output_root": args.output_root,
            "provider": args.provider,
            "model": args.model,
            "fallback_models": args.fallback_models,
            "dotenv": args.dotenv,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "request_timeout": args.request_timeout,
            "image_max_side": args.image_max_side,
            "crop_max_side": args.crop_max_side,
            "sleep": args.sleep,
            "min_confidence": args.min_confidence,
            "max_caption_target_overlap": args.max_caption_target_overlap,
            "train_caption_cap": args.train_caption_cap,
            "eval_caption_cap": args.eval_caption_cap,
            "review_package_rows": args.review_package_rows,
            "resume": args.resume,
            "overwrite": args.overwrite,
            "retry_failed": args.retry_failed,
        }
    )
    return SimpleNamespace(**base_args)


def load_selected_candidates(args: argparse.Namespace, rng: random.Random) -> tuple[list[v09.PageCandidate], dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    candidates, scan_summary, filter_rows = dedup.collect_dedup_candidates(Path(args.candidate_cache_dir), args)
    split_docs = dedup.choose_doc_splits(candidates, args, rng)
    selected = dedup.select_candidates(candidates, split_docs, args, rng)
    return selected, scan_summary, filter_rows, split_docs


def load_existing_repair_reviews(path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not args.resume or not path.exists():
        return {}
    out = {}
    for row in v105.read_jsonl(path):
        task_id = str(row.get("task_id") or "")
        if task_id and not (args.retry_failed and not row.get("ok")):
            out[task_id] = row
    return out


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(args.output_root) / f"agentbench_v1_0_5_auto_repaired_sft_{stamp}"


def prepare_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists; use --resume or --overwrite")
    for child in ["pages", "crops", "overlays", "sft", "episodes", "review", "gold_eval"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bbox(item: dict[str, Any]) -> list[int]:
    return [int(v) for v in item.get("bbox") or [0, 0, 0, 0]]


def union_bbox(boxes: list[list[int]]) -> list[int]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def bbox_area(box: list[int]) -> float:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def bbox_overlap_ratio(a: Any, b: Any, denominator: str = "caption") -> float:
    return v105.bbox_overlap_ratio(a, b, denominator)


def bbox_iou(a: list[int], b: list[int]) -> float:
    return v09.bbox_iou(a, b)


def valid_image_bbox(box: list[int], candidate: v09.PageCandidate) -> bool:
    if len(box) != 4:
        return False
    x1, y1, x2, y2 = [int(v) for v in box]
    width = x2 - x1
    height = y2 - y1
    if width < 60 or height < 60:
        return False
    if x1 < 0 or y1 < 0 or x2 > candidate.page_width or y2 > candidate.page_height:
        return False
    area_ratio = bbox_area(box) / max(1.0, float(candidate.page_width * candidate.page_height))
    return 0.01 <= area_ratio <= 0.70


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def is_caption_start(text: str) -> bool:
    return core4.caption_starts_marker(text)


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


if __name__ == "__main__":
    raise SystemExit(main())
