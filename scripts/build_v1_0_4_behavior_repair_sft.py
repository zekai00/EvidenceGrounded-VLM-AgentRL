#!/usr/bin/env python3
"""Build v1.0.4 behavior-repair SFT rows.

The patch targets current runtime failures:
- write_claims_chunk must not be empty.
- non-abstain claims must carry evidence_ids.
- remaining_fields must be consumed by write_claims_chunk before finish.
- disallowed evidence/field pairs should abstain instead of producing an unsupported claim.

It only uses the existing train/val splits for training/validation rows. Test and
GoldEval test are intentionally excluded.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615"
)
DEFAULT_OVERLAY_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_llm_overlay_20260611_0222"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
CORE_FIELDS = ["caption_text", "image_scope", "depicted_work_title", "displayed_region", "object_type"]
NO_CLAIM_ROLES = {"toc", "bibliography", "front_matter", "back_matter", "ocr_noise", "low_value_background"}
HARD_NEGATIVE_ROLES = NO_CLAIM_ROLES | {"teaching_overview"}
FIELD_ALIASES = {"title": "depicted_work_title"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--overlay-index", default=str(DEFAULT_OVERLAY_INDEX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dataset-suffix", default="")
    parser.add_argument("--max-patch-train", type=int, default=480)
    parser.add_argument("--max-patch-val", type=int, default=128)
    parser.add_argument("--replay-ratio", type=float, default=0.70)
    parser.add_argument("--max-replay-rows", type=int, default=1400)
    parser.add_argument("--max-boundary-train", type=int, default=120)
    parser.add_argument("--max-boundary-val", type=int, default=32)
    parser.add_argument("--max-supported-train", type=int, default=0)
    parser.add_argument("--max-abstain-train", type=int, default=0)
    parser.add_argument("--max-finish-train", type=int, default=0)
    parser.add_argument("--max-supported-val", type=int, default=0)
    parser.add_argument("--max-abstain-val", type=int, default=0)
    parser.add_argument("--max-finish-val", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    source_root = Path(args.source_root)
    overlay_index = Path(args.overlay_index)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / default_output_name(args)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    (output_dir / "sft").mkdir(parents=True)

    tasks_by_split = load_tasks_by_split(source_root)
    sft_rows_by_split = load_sft_rows_by_split(source_root)
    write_rows_by_split = {
        split: [row for row in rows if action_name(row) == "write_claims_chunk"]
        for split, rows in sft_rows_by_split.items()
    }
    finish_rows_by_split = {
        split: [row for row in rows if action_name(row) == "finish"]
        for split, rows in sft_rows_by_split.items()
    }
    write_row_index = index_write_rows(sft_rows_by_split)
    policies = load_overlay_policies(overlay_index)
    chunks_by_source = group_chunks_by_source(policies)

    continuation_train = build_continuation_rows(write_rows_by_split.get("train", []), rng)
    continuation_val = build_continuation_rows(write_rows_by_split.get("val", []), rng)
    finish_train = build_finish_rows(finish_rows_by_split.get("train", []), rng)
    finish_val = build_finish_rows(finish_rows_by_split.get("val", []), rng)
    boundary_train = build_boundary_rows(
        tasks_by_split.get("train", []),
        write_row_index,
        chunks_by_source,
        rng,
        limit=args.max_boundary_train,
    )
    boundary_val = build_boundary_rows(
        tasks_by_split.get("val", []),
        write_row_index,
        chunks_by_source,
        rng,
        limit=args.max_boundary_val,
    )

    patch_train = build_patch_mix(
        continuation_rows=continuation_train,
        finish_rows=finish_train,
        boundary_rows=boundary_train,
        max_total=args.max_patch_train,
        supported_limit=args.max_supported_train,
        abstain_limit=args.max_abstain_train,
        boundary_limit=args.max_boundary_train,
        finish_limit=args.max_finish_train,
        rng=rng,
    )
    patch_val = build_patch_mix(
        continuation_rows=continuation_val,
        finish_rows=finish_val,
        boundary_rows=boundary_val,
        max_total=args.max_patch_val,
        supported_limit=args.max_supported_val,
        abstain_limit=args.max_abstain_val,
        boundary_limit=args.max_boundary_val,
        finish_limit=args.max_finish_val,
        rng=rng,
    )
    replay_rows = collect_replay_rows(
        sft_rows_by_split.get("train", []),
        target_patch_count=len(patch_train),
        replay_ratio=args.replay_ratio,
        max_rows=args.max_replay_rows,
        rng=rng,
    )
    mixed_train = replay_rows + patch_train
    rng.shuffle(mixed_train)
    rng.shuffle(patch_val)

    write_jsonl(output_dir / "behavior_repair_patch_rows.jsonl", patch_train + patch_val)
    write_jsonl(output_dir / "sft" / "patch_train.jsonl", patch_train)
    write_jsonl(output_dir / "sft" / "patch_val.jsonl", patch_val)
    write_jsonl(output_dir / "sft" / "replay_train.jsonl", replay_rows)
    write_jsonl(output_dir / "sft" / "train.jsonl", mixed_train)
    write_jsonl(output_dir / "sft" / "val.jsonl", patch_val)

    manifest = {
        "created_at": now(),
        "dataset_version": "v1.0.4_behavior_repair_sft",
        "source_root": str(source_root),
        "overlay_index": str(overlay_index),
        "output_dir": str(output_dir),
        "policy": {
            "no_test_split_used": True,
            "patch_goal": [
                "non_abstain_claim_requires_evidence_ids",
                "remaining_fields_nonempty_requires_write_claims_chunk_not_finish",
                "finish_only_when_remaining_fields_empty",
                "disallowed_evidence_field_pair_abstains",
            ],
            "replay_ratio": args.replay_ratio,
            "quota_sampling_enabled": quota_sampling_enabled(args),
            "quotas": {
                "max_supported_train": args.max_supported_train,
                "max_abstain_train": args.max_abstain_train,
                "max_boundary_train": args.max_boundary_train,
                "max_finish_train": args.max_finish_train,
                "max_supported_val": args.max_supported_val,
                "max_abstain_val": args.max_abstain_val,
                "max_boundary_val": args.max_boundary_val,
                "max_finish_val": args.max_finish_val,
            },
        },
        "candidate_counts": {
            "continuation_train": len(continuation_train),
            "finish_train": len(finish_train),
            "boundary_train": len(boundary_train),
            "continuation_val": len(continuation_val),
            "finish_val": len(finish_val),
            "boundary_val": len(boundary_val),
        },
        "counts": {
            "patch_train_rows": len(patch_train),
            "patch_val_rows": len(patch_val),
            "replay_rows": len(replay_rows),
            "mixed_train_rows": len(mixed_train),
        },
        "action_counts": {
            "patch_train": dict(action_counter(patch_train)),
            "patch_val": dict(action_counter(patch_val)),
            "replay": dict(action_counter(replay_rows)),
            "mixed_train": dict(action_counter(mixed_train)),
        },
        "patch_type_counts": {
            "patch_train": dict(patch_type_counter(patch_train)),
            "patch_val": dict(patch_type_counter(patch_val)),
        },
        "field_counts": {
            "patch_train": dict(field_counter(patch_train)),
            "patch_val": dict(field_counter(patch_val)),
        },
        "artifacts": {
            "patch_rows": str(output_dir / "behavior_repair_patch_rows.jsonl"),
            "train": str(output_dir / "sft" / "train.jsonl"),
            "val": str(output_dir / "sft" / "val.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_continuation_rows(source_rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        state = source.get("claim_state") or {}
        remaining = [str(item) for item in state.get("remaining_fields") or []]
        if not remaining:
            continue
        action = source.get("action") or {}
        claims = [item for item in action.get("claims") or [] if isinstance(item, dict)]
        abstains = [item for item in action.get("abstains") or [] if isinstance(item, dict)]
        if claims:
            if any(not (claim.get("evidence_ids") or []) for claim in claims):
                continue
            patch_type = "supported_claim_requires_evidence_ids"
            phase_hint = (
                "remaining_fields 非空，禁止 finish；下一步必须用 write_claims_chunk 处理一个 remaining field。"
                "非 abstain claim 必须带非空 evidence_ids，且 evidence_id 必须来自当前可见 evidence。"
            )
        elif abstains:
            patch_type = "abstain_remaining_field"
            phase_hint = (
                "remaining_fields 非空，禁止 finish；如果当前可见证据不足以支持该字段，"
                "下一步必须用 write_claims_chunk 对该字段 abstain，并给出简短 reason。"
            )
        else:
            continue
        rows.append(
            patch_clone(
                source,
                patch_type=patch_type,
                phase_name="v1_0_4_behavior_repair_claim_continuation",
                phase_hint=phase_hint,
                available_actions=["write_claims_chunk"],
            )
        )
    rng.shuffle(rows)
    return rows


def build_finish_rows(source_rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        state = source.get("claim_state") or {}
        remaining = [str(item) for item in state.get("remaining_fields") or []]
        if remaining:
            continue
        rows.append(
            patch_clone(
                source,
                patch_type="finish_ready_only_after_all_fields_done",
                phase_name="v1_0_4_behavior_repair_finish_ready",
                phase_hint=(
                    "claim_state.remaining_fields 为空，目标字段均已写入或 abstain；"
                    "此时才允许 finish，下一步输出 {\"action\":\"finish\",\"status\":\"done\"}。"
                ),
                available_actions=["finish"],
            )
        )
    rng.shuffle(rows)
    return rows


def build_boundary_rows(
    tasks: list[dict[str, Any]],
    write_row_index: dict[tuple[str, str], dict[str, Any]],
    chunks_by_source: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    task_order = list(tasks)
    rng.shuffle(task_order)
    for task in task_order:
        task_id = str(task.get("task_id"))
        source_chunks = list(chunks_by_source.get(str(task.get("source_file") or ""), []))
        rng.shuffle(source_chunks)
        for claim in (task.get("gold") or {}).get("claims") or []:
            field = str(claim.get("field") or "")
            if field not in CORE_FIELDS:
                continue
            base_row = write_row_index.get((task_id, field))
            if not base_row:
                continue
            disallowed = [chunk for chunk in source_chunks if is_hard_negative(chunk) or not policy_allows_field(chunk, field)]
            if not disallowed:
                continue
            chunk = choose_boundary_chunk(disallowed, field, rng)
            candidates.append(make_boundary_row(base_row, task, field, chunk))
            if len(candidates) >= limit * 3:
                break
        if len(candidates) >= limit * 3:
            break
    return balanced_sample(candidates, limit, rng)


def patch_clone(
    source: dict[str, Any],
    *,
    patch_type: str,
    phase_name: str,
    phase_hint: str,
    available_actions: list[str],
) -> dict[str, Any]:
    row = deepcopy(source)
    row["messages"] = []
    row["label_source"] = f"v1_0_4_behavior_repair_{patch_type}"
    row["tool_schema_version"] = "v1.0.4_behavior_repair_sft_no_select"
    row["variant"] = "v1_0_4_behavior_repair_sft"
    row["phase_name"] = phase_name
    row["phase_hint"] = phase_hint
    row["available_actions"] = available_actions
    row["patch_meta"] = {
        "patch_type": patch_type,
        "source_split": source.get("split"),
        "source_task_id": source.get("task_id"),
        "action": action_name(source),
        "fields": fields_from_action(source.get("action") or {}),
    }
    return row


def make_boundary_row(base_row: dict[str, Any], task: dict[str, Any], field: str, chunk: dict[str, Any]) -> dict[str, Any]:
    evidence_id = str(chunk.get("evidence_id") or "")
    row = patch_clone(
        base_row,
        patch_type="disallowed_evidence_field_abstain",
        phase_name="v1_0_4_behavior_repair_field_boundary",
        phase_hint=(
            "当前打开的 evidence 已被裁决为不能支持该字段；remaining_fields 非空时禁止 finish。"
            "下一步必须对该字段 abstain，不能硬写 unsupported claim。"
        ),
        available_actions=["write_claims_chunk"],
    )
    row["history"] = make_boundary_history(base_row, task, chunk)
    row["tool_results"] = make_boundary_tool_results(base_row, task, chunk)
    row["selected_evidence_ids"] = [evidence_id]
    row["draft_claims"] = []
    row["claim_state"] = {
        "target_fields": [field],
        "written_fields": [],
        "abstained_fields": [],
        "remaining_fields": [field],
        "claim_count": 0,
        "abstain_count": 0,
        "evidence_ids": [],
    }
    row["action"] = {
        "action": "write_claims_chunk",
        "claims": [],
        "abstains": [{"field": field, "reason": boundary_reason(chunk, field)}],
    }
    row["patch_meta"].update(
        {
            "field": field,
            "evidence_id": evidence_id,
            "adjudicated_evidence_role": chunk.get("adjudicated_evidence_role"),
            "adjudication_status": chunk.get("adjudication_status"),
            "adjudicated_claim_allowed_fields": claim_allowed_fields(chunk),
            "usable_for_claim_by_adjudication": chunk.get("usable_for_claim_by_adjudication"),
        }
    )
    return row


def make_boundary_history(base_row: dict[str, Any], task: dict[str, Any], chunk: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for action in base_row.get("history") or []:
        if isinstance(action, dict) and action.get("action") in {"inspect_page", "crop_target"}:
            history.append(action)
    query = build_query(task, chunk)
    history.append({"action": "retrieve_evidence", "query": query, "scope": "same_document", "top_k": 5})
    history.append({"action": "open_evidence", "evidence_id": chunk.get("evidence_id")})
    return history[-6:]


def make_boundary_tool_results(base_row: dict[str, Any], task: dict[str, Any], chunk: dict[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for result in base_row.get("tool_results") or []:
        if isinstance(result, dict) and result.get("tool") in {"inspect_page", "crop_target"}:
            results.append(result)
    evidence = public_evidence(chunk)
    query = build_query(task, chunk)
    results.append(
        {
            "tool": "retrieve_evidence",
            "query": query,
            "scope": "same_document",
            "anchor": {
                "source_file": task.get("source_file"),
                "page": task.get("page"),
                "bbox": (task.get("gold") or {}).get("target_region_bbox") or (task.get("gold") or {}).get("image_bbox"),
            },
            "results": [evidence],
            "hit_evidence_ids": [],
        }
    )
    open_result = dict(evidence)
    open_result["tool"] = "open_evidence"
    open_result["text"] = evidence.get("display_snippet")
    results.append(open_result)
    return results[-6:]


def public_evidence(chunk: dict[str, Any]) -> dict[str, Any]:
    snippet = str(
        chunk.get("display_snippet")
        or chunk.get("evidence_summary")
        or chunk.get("clean_text")
        or chunk.get("text")
        or ""
    )
    return {
        "evidence_id": chunk.get("evidence_id"),
        "source_file": chunk.get("source_file"),
        "page_start": chunk.get("page_start"),
        "page_end": chunk.get("page_end"),
        "authority_level": chunk.get("authority_level"),
        "citation_level": chunk.get("citation_level"),
        "source_quality": chunk.get("source_quality"),
        "clean_evidence_type": chunk.get("clean_evidence_type"),
        "adjudicated_evidence_role": chunk.get("adjudicated_evidence_role"),
        "adjudication_status": chunk.get("adjudication_status"),
        "adjudicated_claim_allowed_fields": claim_allowed_fields(chunk),
        "usable_for_claim_by_adjudication": chunk.get("usable_for_claim_by_adjudication"),
        "display_snippet": truncate_text(snippet, 520),
    }


def collect_replay_rows(
    source_rows: list[dict[str, Any]],
    *,
    target_patch_count: int,
    replay_ratio: float,
    max_rows: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if target_patch_count <= 0 or replay_ratio <= 0:
        return []
    target = int(round(target_patch_count * replay_ratio / max(1e-6, 1.0 - replay_ratio)))
    target = min(max_rows, target)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        name = action_name(row)
        if name not in {"inspect_page", "crop_target", "open_evidence", "retrieve_evidence", "write_claims_chunk", "finish"}:
            continue
        replay = deepcopy(row)
        replay["messages"] = []
        replay["label_source"] = "v1_0_3_no_select_replay_for_v1_0_4_behavior_repair"
        replay["tool_schema_version"] = "v1.0.4_behavior_repair_replay_no_select"
        buckets[name].append(replay)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    actions = sorted(buckets)
    cursor = 0
    while len(selected) < target and any(buckets.values()):
        action = actions[cursor % len(actions)]
        cursor += 1
        if buckets[action]:
            selected.append(buckets[action].pop())
    return selected


def quota_sampling_enabled(args: argparse.Namespace) -> bool:
    return any(
        int(getattr(args, key, 0) or 0) > 0
        for key in [
            "max_supported_train",
            "max_abstain_train",
            "max_finish_train",
            "max_supported_val",
            "max_abstain_val",
            "max_finish_val",
        ]
    )


def build_patch_mix(
    *,
    continuation_rows: list[dict[str, Any]],
    finish_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    max_total: int,
    supported_limit: int,
    abstain_limit: int,
    boundary_limit: int,
    finish_limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if supported_limit <= 0 and abstain_limit <= 0 and finish_limit <= 0:
        return balanced_sample(continuation_rows + finish_rows + boundary_rows, max_total, rng)

    selected: list[dict[str, Any]] = []
    selected.extend(sample_patch_type(continuation_rows, "supported_claim_requires_evidence_ids", supported_limit, rng))
    selected.extend(sample_patch_type(continuation_rows, "abstain_remaining_field", abstain_limit, rng))
    selected.extend(balanced_sample(boundary_rows, boundary_limit, rng))
    selected.extend(sample_patch_type(finish_rows, "finish_ready_only_after_all_fields_done", finish_limit, rng))

    selected_object_ids = {id(row) for row in selected}
    if len(selected) < max_total:
        remaining = [
            row
            for row in continuation_rows + finish_rows + boundary_rows
            if id(row) not in selected_object_ids
        ]
        selected.extend(balanced_sample(remaining, max_total - len(selected), rng))
    rng.shuffle(selected)
    if max_total > 0 and len(selected) > max_total:
        selected = balanced_sample(selected, max_total, rng)
    return selected


def sample_patch_type(rows: list[dict[str, Any]], patch_type: str, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return balanced_sample([row for row in rows if get_patch_type(row) == patch_type], limit, rng)


def balanced_sample(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        out = list(rows)
        rng.shuffle(out)
        return out
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        patch_type = str((row.get("patch_meta") or {}).get("patch_type") or row.get("label_source") or "")
        fields = fields_from_action(row.get("action") or {})
        field = fields[0] if fields else ""
        buckets[(patch_type, field)].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        cursor += 1
        bucket = buckets.get(key) or []
        if bucket:
            selected.append(bucket.pop())
        if not bucket:
            buckets.pop(key, None)
            keys = sorted(buckets)
            cursor = 0
    return selected


def get_patch_type(row: dict[str, Any]) -> str:
    return str((row.get("patch_meta") or {}).get("patch_type") or row.get("label_source") or "")


def index_write_rows(sft_rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for _split, source_rows in sft_rows_by_split.items():
        for row in source_rows:
            if action_name(row) != "write_claims_chunk":
                continue
            for field in fields_from_action(row.get("action") or {}):
                if field in CORE_FIELDS:
                    rows.setdefault((str(row.get("task_id")), field), row)
    return rows


def load_tasks_by_split(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val"]:
        path = source_root / f"{split}_tasks.jsonl"
        if path.exists():
            out[split] = list(iter_jsonl(path))
    return out


def load_sft_rows_by_split(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val"]:
        path = source_root / "sft" / f"{split}.jsonl"
        if path.exists():
            out[split] = list(iter_jsonl(path))
    return out


def load_overlay_policies(index_dir: Path) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    path = index_dir / "corpus_chunks.jsonl"
    if not path.exists():
        return policies
    for row in iter_jsonl(path):
        if not row.get("adjudication_status") and not row.get("adjudicated_evidence_role"):
            continue
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id:
            policies[evidence_id] = row
    return policies


def group_chunks_by_source(policies: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in policies.values():
        out[str(row.get("source_file") or "")].append(row)
    return out


def choose_boundary_chunk(chunks: list[dict[str, Any]], field: str, rng: random.Random) -> dict[str, Any]:
    ranked = sorted(chunks, key=lambda item: boundary_preference(item, field), reverse=True)
    return rng.choice(ranked[: min(4, len(ranked))])


def boundary_preference(chunk: dict[str, Any], field: str) -> tuple[int, int, int]:
    role = str(chunk.get("adjudicated_evidence_role") or "")
    hard_negative = 3 if role in HARD_NEGATIVE_ROLES else 0
    accepted = 1 if chunk.get("adjudication_status") == "accepted_auto" else 0
    wrong_field = 2 if accepted and not policy_allows_field(chunk, field) else 0
    return (hard_negative, wrong_field, accepted)


def policy_allows_field(policy: dict[str, Any], field: str) -> bool:
    status = str(policy.get("adjudication_status") or "")
    if status and status != "accepted_auto":
        return False
    role = str(policy.get("adjudicated_evidence_role") or policy.get("evidence_role") or "")
    if role in NO_CLAIM_ROLES:
        return False
    if policy.get("usable_for_claim_by_adjudication") is False:
        return False
    return normalize_field(field) in {normalize_field(item) for item in claim_allowed_fields(policy)}


def is_hard_negative(policy: dict[str, Any]) -> bool:
    role = str(policy.get("adjudicated_evidence_role") or "")
    return (
        role in HARD_NEGATIVE_ROLES
        or policy.get("adjudication_status") != "accepted_auto"
        or policy.get("usable_for_claim_by_adjudication") is False
    )


def claim_allowed_fields(policy: dict[str, Any]) -> list[str]:
    fields = policy.get("adjudicated_claim_allowed_fields") or policy.get("claim_allowed_fields") or []
    out: list[str] = []
    for item in fields:
        field = normalize_field(item)
        if field and field not in out:
            out.append(field)
    return out


def normalize_field(field: Any) -> str:
    return FIELD_ALIASES.get(str(field), str(field))


def build_query(task: dict[str, Any], chunk: dict[str, Any]) -> str:
    source = str(task.get("source_stem") or task.get("source_file") or "").replace(".pdf", "")
    role = str(chunk.get("adjudicated_evidence_role") or "")
    return truncate_text(f"{source} {role} 山水画 图像 证据", 120)


def boundary_reason(chunk: dict[str, Any], field: str) -> str:
    role = chunk.get("adjudicated_evidence_role")
    status = chunk.get("adjudication_status")
    allowed = claim_allowed_fields(chunk)
    if status != "accepted_auto":
        return f"当前 evidence 的裁决状态为 {status}，不能作为强证据支持 {field}。"
    if chunk.get("usable_for_claim_by_adjudication") is False:
        return f"当前 evidence role={role} 被裁决为不可支持 claim，不能支持 {field}。"
    return f"当前 evidence role={role} 只允许支持 {allowed}，不能支持字段 {field}。"


def fields_from_action(action: dict[str, Any]) -> list[str]:
    fields: list[str] = []
    for item in action.get("claims") or []:
        field = str(item.get("field") or "")
        if field:
            fields.append(field)
    for item in action.get("abstains") or []:
        field = str(item.get("field") or "")
        if field:
            fields.append(field)
    field = str(action.get("field") or "")
    if field:
        fields.append(field)
    return fields


def action_name(row: dict[str, Any]) -> str:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    return str(action.get("action") or "")


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(action_name(row) for row in rows)


def patch_type_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("patch_meta") or {}).get("patch_type") or "") for row in rows)


def field_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        for field in fields_from_action(row.get("action") or {}):
            counts[field] += 1
    return counts


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Behavior Repair SFT 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- dataset_version: {manifest['dataset_version']}",
        f"- source_root: `{manifest['source_root']}`",
        f"- overlay_index: `{manifest['overlay_index']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## 构建原则",
        "",
        "- 不使用 test split，也不使用 `test_gold_100`。",
        "- patch 用于修 agent 行为，不用于注入新知识。",
        "- replay 用于保持原 no-select 工具流程能力。",
        "- 非 abstain claim 必须带 evidence_ids；remaining_fields 非空时必须继续 write_claims_chunk，不能 finish。",
        "",
        "## 规模",
        "",
    ]
    for key, value in manifest["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Patch 类型", ""])
    for split, counts in manifest["patch_type_counts"].items():
        lines.append(f"### {split}")
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## Action 分布", ""])
    for split, counts in manifest["action_counts"].items():
        lines.append(f"### {split}")
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## 字段分布", ""])
    for split, counts in manifest["field_counts"].items():
        lines.append(f"### {split}")
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def default_output_name(args: argparse.Namespace) -> str:
    suffix = f"_{args.dataset_suffix}" if args.dataset_suffix else f"_replay{int(round(args.replay_ratio * 100))}"
    return f"agentbench_v1_0_4_behavior_repair_sft{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
