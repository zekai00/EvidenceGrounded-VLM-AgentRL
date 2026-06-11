#!/usr/bin/env python3
"""Build v1.0.4 field-claim SFT patch rows from adjudicated overlay evidence.

The patch teaches the no-select agent to respect evidence role and allowed
claim fields. It creates synthetic write_claims_chunk states from existing
v1.0.3 no-select rows, replacing the visible evidence with adjudicated overlay
chunks.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
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

SCHEMA_FIELDS = {
    "caption_text",
    "image_scope",
    "depicted_work_title",
    "displayed_region",
    "object_type",
    "artist",
    "dynasty",
    "visual_elements",
    "technique",
    "composition",
    "medium_dimensions",
    "collection",
}
NO_CLAIM_ROLES = {"toc", "bibliography", "front_matter", "back_matter", "ocr_noise", "low_value_background"}
HARD_NEGATIVE_ROLES = NO_CLAIM_ROLES | {"teaching_overview"}
FIELD_ALIASES = {"title": "depicted_work_title"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--overlay-index", default=str(DEFAULT_OVERLAY_INDEX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-positive-train", type=int, default=450)
    parser.add_argument("--max-negative-train", type=int, default=450)
    parser.add_argument("--max-positive-val", type=int, default=96)
    parser.add_argument("--max-negative-val", type=int, default=96)
    parser.add_argument("--max-replay-rows", type=int, default=700)
    parser.add_argument("--replay-ratio", type=float, default=0.45)
    parser.add_argument("--max-per-evidence", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    source_root = Path(args.source_root)
    overlay_index = Path(args.overlay_index)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / default_output_name()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    (output_dir / "sft").mkdir(parents=True)

    policies = load_overlay_policies(overlay_index)
    chunks_by_source = group_chunks_by_source(policies)
    tasks_by_split = load_tasks_by_split(source_root)
    sft_rows_by_split = load_sft_rows_by_split(source_root)
    write_rows = index_write_rows(sft_rows_by_split)

    positive_candidates = build_positive_candidates(tasks_by_split, write_rows, policies, rng)
    negative_candidates = build_negative_candidates(tasks_by_split, write_rows, policies, chunks_by_source, rng)
    patch_train = balanced_sample(
        [row for row in positive_candidates if row.get("split") == "train"],
        args.max_positive_train,
        args.max_per_evidence,
        rng,
    ) + balanced_sample(
        [row for row in negative_candidates if row.get("split") == "train"],
        args.max_negative_train,
        args.max_per_evidence,
        rng,
    )
    patch_val = balanced_sample(
        [row for row in positive_candidates if row.get("split") == "val"],
        args.max_positive_val,
        args.max_per_evidence,
        rng,
    ) + balanced_sample(
        [row for row in negative_candidates if row.get("split") == "val"],
        args.max_negative_val,
        args.max_per_evidence,
        rng,
    )
    rng.shuffle(patch_train)
    rng.shuffle(patch_val)

    replay_rows = collect_replay_rows(
        sft_rows_by_split.get("train", []),
        target_patch_count=len(patch_train),
        replay_ratio=args.replay_ratio,
        max_rows=args.max_replay_rows,
        rng=rng,
    )
    mixed_train = replay_rows + patch_train
    rng.shuffle(mixed_train)

    write_jsonl(output_dir / "field_claim_patch_rows.jsonl", patch_train + patch_val)
    write_jsonl(output_dir / "sft" / "patch_train.jsonl", patch_train)
    write_jsonl(output_dir / "sft" / "patch_val.jsonl", patch_val)
    write_jsonl(output_dir / "sft" / "replay_train.jsonl", replay_rows)
    write_jsonl(output_dir / "sft" / "train.jsonl", mixed_train)
    write_jsonl(output_dir / "sft" / "val.jsonl", patch_val)

    manifest = {
        "created_at": now(),
        "dataset_version": "v1.0.4_field_claim_sft_patch",
        "source_root": str(source_root),
        "overlay_index": str(overlay_index),
        "output_dir": str(output_dir),
        "candidate_counts": {
            "positive": len(positive_candidates),
            "negative": len(negative_candidates),
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
        "patch_type_counts": dict(meta_counter(patch_train + patch_val, "patch_type")),
        "role_counts": dict(meta_counter(patch_train + patch_val, "adjudicated_evidence_role")),
        "field_counts": dict(field_counter(patch_train + patch_val)),
        "label_source_counts": dict(Counter(str(row.get("label_source")) for row in patch_train + patch_val)),
        "artifacts": {
            "field_claim_patch_rows": str(output_dir / "field_claim_patch_rows.jsonl"),
            "train": str(output_dir / "sft" / "train.jsonl"),
            "val": str(output_dir / "sft" / "val.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def load_overlay_policies(index_dir: Path) -> dict[str, dict[str, Any]]:
    policies: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(index_dir / "corpus_chunks.jsonl"):
        if not row.get("adjudication_status") and not row.get("adjudicated_evidence_role"):
            continue
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id:
            policies[evidence_id] = row
    if not policies:
        raise RuntimeError(f"no adjudicated policies found in {index_dir}")
    return policies


def group_chunks_by_source(policies: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in policies.values():
        out[str(row.get("source_file") or "")].append(row)
    return out


def load_tasks_by_split(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for split in ["train", "val", "test"]:
        path = source_root / f"{split}_tasks.jsonl"
        if path.exists():
            out[split] = list(iter_jsonl(path))
    return out


def load_sft_rows_by_split(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for split in ["train", "val", "test"]:
        path = source_root / "sft" / f"{split}.jsonl"
        if path.exists():
            out[split] = list(iter_jsonl(path))
    return out


def index_write_rows(sft_rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for source_split, source_rows in sft_rows_by_split.items():
        for row in source_rows:
            action = row.get("action") if isinstance(row.get("action"), dict) else {}
            if action.get("action") != "write_claims_chunk":
                continue
            fields = [str(item.get("field")) for item in action.get("claims") or [] if isinstance(item, dict)]
            fields.extend(str(item.get("field")) for item in action.get("abstains") or [] if isinstance(item, dict))
            for field in fields:
                if field in SCHEMA_FIELDS:
                    rows.setdefault((str(row.get("task_id")), field), row)
    return rows


def build_positive_candidates(
    tasks_by_split: dict[str, list[dict[str, Any]]],
    write_rows: dict[tuple[str, str], dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        for task in tasks_by_split.get(split, []):
            task_id = str(task.get("task_id"))
            for claim in (task.get("gold") or {}).get("claims") or []:
                field = str(claim.get("field") or "")
                if field not in SCHEMA_FIELDS or claim.get("abstain") or claim.get("value") is None:
                    continue
                base_row = write_rows.get((task_id, field))
                if not base_row:
                    continue
                evidence_ids = candidate_evidence_ids(claim)
                usable = [policies[eid] for eid in evidence_ids if eid in policies and policy_allows_field(policies[eid], field)]
                if not usable:
                    continue
                chunk = choose_policy_chunk(usable, field, rng)
                candidates.append(make_patch_row(base_row, task, claim, chunk, "positive_allowed_field"))
    return candidates


def build_negative_candidates(
    tasks_by_split: dict[str, list[dict[str, Any]]],
    write_rows: dict[tuple[str, str], dict[str, Any]],
    policies: dict[str, dict[str, Any]],
    chunks_by_source: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for split in ["train", "val"]:
        for task in tasks_by_split.get(split, []):
            task_id = str(task.get("task_id"))
            source_chunks = chunks_by_source.get(str(task.get("source_file") or ""), [])
            rng.shuffle(source_chunks)
            for claim in (task.get("gold") or {}).get("claims") or []:
                field = str(claim.get("field") or "")
                if field not in SCHEMA_FIELDS:
                    continue
                base_row = write_rows.get((task_id, field))
                if not base_row:
                    continue
                disallowed: list[dict[str, Any]] = []
                for eid in candidate_evidence_ids(claim):
                    if eid in policies and not policy_allows_field(policies[eid], field):
                        disallowed.append(policies[eid])
                if not disallowed:
                    disallowed = [
                        chunk for chunk in source_chunks if is_hard_negative(chunk) or not policy_allows_field(chunk, field)
                    ][:4]
                if not disallowed:
                    continue
                chunk = choose_policy_chunk(disallowed, field, rng)
                candidates.append(make_patch_row(base_row, task, claim, chunk, "negative_field_boundary"))
    return candidates


def make_patch_row(
    base_row: dict[str, Any],
    task: dict[str, Any],
    claim: dict[str, Any],
    chunk: dict[str, Any],
    patch_type: str,
) -> dict[str, Any]:
    row = json.loads(json.dumps(base_row, ensure_ascii=False))
    field = str(claim.get("field") or "")
    evidence_id = str(chunk.get("evidence_id") or "")
    role = str(chunk.get("adjudicated_evidence_role") or "")
    allowed_fields = claim_allowed_fields(chunk)
    row["split"] = "val" if str(task.get("split")) == "val" or str(base_row.get("split")) == "val" else "train"
    row["messages"] = []
    row["label_source"] = f"v1_0_4_field_claim_{patch_type}"
    row["tool_schema_version"] = "v1.0.4_field_claim_sft_patch_no_select"
    row["variant"] = "v1_0_4_field_claim_sft_patch"
    row["phase_name"] = "no_select_v1_0_4_field_claim_write"
    row["phase_hint"] = (
        "根据当前可见 evidence 的 adjudicated_evidence_role 与 adjudicated_claim_allowed_fields 写入或 abstain；"
        "只能用 allowed_fields 中包含当前字段且 accepted_auto 的 evidence 支持 claim。"
    )
    row["available_actions"] = ["write_claims_chunk"]
    row["selected_evidence_ids"] = [evidence_id]
    row["history"] = make_history(base_row, task, chunk)
    row["tool_results"] = make_tool_results(base_row, task, claim, chunk)
    row["claim_state"] = {
        "target_fields": [field],
        "written_fields": [],
        "abstained_fields": [],
        "remaining_fields": [field],
        "claim_count": 0,
        "abstain_count": 0,
        "evidence_ids": [],
    }
    row["draft_claims"] = []
    if patch_type == "positive_allowed_field":
        row["action"] = {
            "action": "write_claims_chunk",
            "claims": [
                {
                    "field": field,
                    "value": claim.get("value"),
                    "evidence_ids": [evidence_id],
                    "visual_bbox": claim.get("visual_bbox"),
                    "confidence": min(0.9, max(0.62, float(claim.get("confidence") or 0.76))),
                }
            ],
            "abstains": [],
        }
    else:
        row["action"] = {
            "action": "write_claims_chunk",
            "claims": [],
            "abstains": [
                {
                    "field": field,
                    "reason": negative_reason(chunk, field),
                }
            ],
        }
    row["patch_meta"] = {
        "patch_type": patch_type,
        "field": field,
        "evidence_id": evidence_id,
        "adjudicated_evidence_role": role,
        "adjudication_status": chunk.get("adjudication_status"),
        "adjudicated_claim_allowed_fields": allowed_fields,
        "usable_for_claim_by_adjudication": chunk.get("usable_for_claim_by_adjudication"),
        "source_file": task.get("source_file"),
        "source_split": base_row.get("split"),
    }
    return row


def make_history(base_row: dict[str, Any], task: dict[str, Any], chunk: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    for action in base_row.get("history") or []:
        if not isinstance(action, dict):
            continue
        if action.get("action") in {"inspect_page", "crop_target"}:
            history.append(action)
    query = build_query(task, chunk)
    history.append({"action": "retrieve_evidence", "query": query, "scope": "same_document", "top_k": 5})
    history.append({"action": "open_evidence", "evidence_id": chunk.get("evidence_id")})
    return history[-6:]


def make_tool_results(
    base_row: dict[str, Any],
    task: dict[str, Any],
    claim: dict[str, Any],
    chunk: dict[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for result in base_row.get("tool_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("tool") in {"inspect_page", "crop_target"}:
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
            "hit_evidence_ids": [chunk.get("evidence_id")] if str(chunk.get("evidence_id")) in candidate_evidence_ids(claim) else [],
        }
    )
    open_result = dict(evidence)
    open_result["tool"] = "open_evidence"
    open_result["text"] = evidence["display_snippet"]
    results.append(open_result)
    return results[-6:]


def public_evidence(chunk: dict[str, Any]) -> dict[str, Any]:
    snippet = str(chunk.get("display_snippet") or chunk.get("evidence_summary") or chunk.get("clean_text") or chunk.get("text") or "")
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


def choose_policy_chunk(chunks: list[dict[str, Any]], field: str, rng: random.Random) -> dict[str, Any]:
    ranked = sorted(chunks, key=lambda item: policy_preference(item, field), reverse=True)
    top = ranked[: min(3, len(ranked))]
    return rng.choice(top)


def policy_preference(chunk: dict[str, Any], field: str) -> tuple[int, int, int]:
    role = str(chunk.get("adjudicated_evidence_role") or "")
    preferred_role = 0
    if field in {"artist", "dynasty", "depicted_work_title", "medium_dimensions", "collection"} and role == "object_metadata":
        preferred_role = 3
    elif field in {"caption_text", "image_scope", "displayed_region", "object_type"} and role == "caption_or_plate":
        preferred_role = 3
    elif field in {"visual_elements", "technique", "composition"} and role == "style_analysis":
        preferred_role = 3
    hard_negative = 2 if is_hard_negative(chunk) else 0
    accepted = 1 if chunk.get("adjudication_status") == "accepted_auto" else 0
    return (preferred_role, hard_negative, accepted)


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


def candidate_evidence_ids(claim: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for key in ["evidence_ids", "candidate_evidence_ids"]:
        for item in claim.get(key) or []:
            evidence_id = str(item)
            if evidence_id and evidence_id not in seen:
                seen.add(evidence_id)
                out.append(evidence_id)
    return out


def build_query(task: dict[str, Any], chunk: dict[str, Any]) -> str:
    source = str(task.get("source_stem") or task.get("source_file") or "").replace(".pdf", "")
    role = str(chunk.get("adjudicated_evidence_role") or "")
    return truncate_text(f"{source} {role} 山水画 图像 证据", 120)


def negative_reason(chunk: dict[str, Any], field: str) -> str:
    role = chunk.get("adjudicated_evidence_role")
    status = chunk.get("adjudication_status")
    allowed = claim_allowed_fields(chunk)
    if status != "accepted_auto":
        return f"当前 evidence 的裁决状态为 {status}，不能作为强证据支持 {field}。"
    if chunk.get("usable_for_claim_by_adjudication") is False:
        return f"当前 evidence role={role} 被裁决为不可支持 claim，不能支持 {field}。"
    return f"当前 evidence role={role} 只允许支持 {allowed}，不能支持字段 {field}。"


def balanced_sample(
    rows: list[dict[str, Any]],
    limit: int,
    max_per_evidence: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        meta = row.get("patch_meta") or {}
        key = (
            str(meta.get("patch_type")),
            str(meta.get("adjudicated_evidence_role")),
            str(meta.get("field")),
        )
        buckets[key].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    evidence_counts: Counter[str] = Counter()
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < limit and any(buckets.values()):
        key = keys[cursor % len(keys)]
        cursor += 1
        bucket = buckets[key]
        while bucket:
            row = bucket.pop()
            evidence_id = str((row.get("patch_meta") or {}).get("evidence_id") or "")
            if evidence_counts[evidence_id] < max_per_evidence:
                selected.append(row)
                evidence_counts[evidence_id] += 1
                break
        if not bucket:
            buckets.pop(key, None)
            keys = sorted(buckets)
            cursor = 0
            if not keys:
                break
    return selected


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
        action = str((row.get("action") or {}).get("action") or "")
        if action not in {"inspect_page", "crop_target", "open_evidence", "retrieve_evidence", "write_claims_chunk", "finish"}:
            continue
        replay = dict(row)
        replay["label_source"] = "v1_0_3_no_select_replay_for_v1_0_4_field_claim"
        replay["tool_schema_version"] = "v1.0.4_field_claim_replay_no_select"
        buckets[action].append(replay)
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


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("action") or {}).get("action") or "") for row in rows)


def meta_counter(rows: list[dict[str, Any]], key: str) -> Counter[str]:
    return Counter(str((row.get("patch_meta") or {}).get(key) or "") for row in rows)


def field_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        action = row.get("action") or {}
        for item in action.get("claims") or []:
            counts[str(item.get("field"))] += 1
        for item in action.get("abstains") or []:
            counts[str(item.get("field"))] += 1
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
        "# v1.0.4 Field-Claim SFT Patch 构建报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- 源 no-select 数据：`{manifest['source_root']}`",
        f"- overlay index：`{manifest['overlay_index']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        "",
        "## 数据量",
        "",
    ]
    for key, value in manifest["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Patch 类型", ""])
    for key, value in sorted(manifest["patch_type_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Evidence Role 分布", ""])
    for key, value in sorted(manifest["role_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 字段分布", ""])
    for key, value in sorted(manifest["field_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "- 该数据需要用 `train_trajectory_sft_lora.py --prompt-mode compact` 训练。",
            "- patch row 的 `messages` 为空，避免 original prompt 模式误用旧上下文。",
            "- 正例要求模型只用 accepted_auto 且 allowed_fields 覆盖当前字段的 evidence 写 claim。",
            "- 负例要求模型看到不可用或字段不匹配 evidence 时 abstain。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def default_output_name() -> str:
    return "agentbench_v1_0_4_field_claim_sft_patch_" + datetime.now().strftime("%Y%m%d_%H%M")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
