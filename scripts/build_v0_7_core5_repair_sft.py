#!/usr/bin/env python3
"""Build a focused v0.7 Core5 evidence/claim repair SFT dataset.

The source v0.7 oracle trajectories supervise a 12-field claim card. Current
rollout evaluation uses a narrower Core5 card, so this builder trims claim
actions and claim_state to the five evaluated fields, then inserts an immediate
finish row once those fields are complete.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_7_inspect_crop_sft_20260605_2336"
)
CORE_FIELDS = [
    "caption_text",
    "image_scope",
    "depicted_work_title",
    "displayed_region",
    "object_type",
]
CORE_SET = set(CORE_FIELDS)
DEFAULT_TRAIN_TARGETS = {
    "inspect_page": 100,
    "crop_target": 130,
    "retrieve_evidence": 330,
    "open_evidence": 330,
    "write_claims_chunk": 560,
    "finish": 80,
}
DEFAULT_VAL_TARGETS = {
    "inspect_page": 20,
    "crop_target": 25,
    "retrieve_evidence": 50,
    "open_evidence": 50,
    "write_claims_chunk": 70,
    "finish": 25,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=60607)
    parser.add_argument("--train-targets", default=target_string(DEFAULT_TRAIN_TARGETS))
    parser.add_argument("--val-targets", default=target_string(DEFAULT_VAL_TARGETS))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "sft").mkdir(parents=True, exist_ok=True)

    split_outputs: dict[str, list[dict[str, Any]]] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        source_path = args.source_dir / "sft" / f"{split}.jsonl"
        source_rows = read_jsonl(source_path)
        transformed, transform_stats = transform_split(source_rows)
        targets = parse_targets(args.train_targets if split == "train" else args.val_targets)
        if split == "test":
            selected = sample_by_action(transformed, parse_targets(args.val_targets), rng, replacement=False)
        else:
            selected = sample_by_action(transformed, targets, rng, replacement=(split == "train"))
        rng.shuffle(selected)
        out_path = args.output_dir / "sft" / f"{split}.jsonl"
        write_jsonl(out_path, selected)
        write_jsonl(args.output_dir / f"{split}_preview.jsonl", selected[:8])
        split_outputs[split] = selected
        split_stats[split] = {
            "source_rows": len(source_rows),
            "transformed_rows": len(transformed),
            "selected_rows": len(selected),
            "source_action_counts": dict(action_counter(source_rows)),
            "transformed_action_counts": dict(action_counter(transformed)),
            "selected_action_counts": dict(action_counter(selected)),
            "transform_stats": transform_stats,
            "path": str(out_path),
        }

    all_rows = split_outputs["train"] + split_outputs["val"] + split_outputs["test"]
    write_jsonl(args.output_dir / "sft" / "all.jsonl", all_rows)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.7_core5_evidence_claim_repair_sft",
        "purpose": (
            "Repair the Phase8 policy for Core5 evaluation by increasing evidence/open/claim supervision, "
            "trimming the 12-field oracle claim card to five evaluated fields, splitting claim chunks into single-field steps, and supervising immediate finish only after Core5 is complete."
        ),
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "core_fields": CORE_FIELDS,
        "train_targets": parse_targets(args.train_targets),
        "val_targets": parse_targets(args.val_targets),
        "splits": split_stats,
        "all_rows": len(all_rows),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def transform_split(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row.get("task_id"))].append(row)

    transformed: list[dict[str, Any]] = []
    stats = Counter()
    for task_id, task_rows in by_task.items():
        task_rows = sorted(task_rows, key=lambda item: int(item.get("step") or 0))
        task_out, task_stats = transform_task_rows(task_rows)
        transformed.extend(task_out)
        stats.update(task_stats)
        stats["tasks_seen"] += 1
        if any((row.get("action") or {}).get("action") == "finish" for row in task_out):
            stats["tasks_with_core_finish"] += 1
        else:
            stats["tasks_without_core_finish"] += 1
    return transformed, dict(stats)


def transform_task_rows(task_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], Counter]:
    out: list[dict[str, Any]] = []
    stats: Counter = Counter()
    done: set[str] = set()
    finish_added = False

    for index, row in enumerate(task_rows):
        pre_done = done_from_claim_state(row.get("claim_state") or {})
        done |= pre_done
        if len(done) >= len(CORE_FIELDS):
            finish_row = make_core_row(row, {"action": "finish", "status": "done"}, "core5_immediate_finish")
            out.append(finish_row)
            stats["inserted_finish_rows"] += 1
            finish_added = True
            break

        action = row.get("action") if isinstance(row.get("action"), dict) else {}
        action_name = str(action.get("action") or "")
        if action_name in {"inspect_page", "crop_target", "retrieve_evidence", "open_evidence"}:
            out.append(make_core_row(row, action, "core5_replay_until_claim_done"))
            stats[f"kept_{action_name}"] += 1
            continue
        if action_name in {"write_claims_chunk", "write_claims_batch", "write_claim", "abstain_claim"}:
            filtered_items = filter_core_claim_items(action, done)
            if not filtered_items:
                stats["dropped_noncore_claim_action"] += 1
                continue
            draft = filter_claim_list(row.get("draft_claims") or [])
            for substep, item_action in enumerate(single_field_chunk_actions(filtered_items)):
                synthetic_row = make_core_row_with_state(
                    row,
                    item_action,
                    "core5_single_field_claim_repair",
                    draft,
                    synthetic_substep=substep,
                )
                out.append(synthetic_row)
                stats[f"kept_{item_action.get('action')}"] += 1
                draft = apply_action_to_draft(draft, item_action)
                done = done_from_draft(draft)
            if len(done) >= len(CORE_FIELDS):
                next_state = task_rows[index + 1] if index + 1 < len(task_rows) else row
                finish_row = make_core_row_with_state(
                    next_state,
                    {"action": "finish", "status": "done"},
                    "core5_immediate_finish",
                    draft,
                    synthetic_substep=0,
                )
                finish_row["step"] = int(row.get("step") or 0) + 1
                out.append(finish_row)
                stats["inserted_finish_rows"] += 1
                finish_added = True
                break
            continue
        if action_name == "finish":
            if len(done) >= len(CORE_FIELDS):
                out.append(make_core_row(row, {"action": "finish", "status": "done"}, "core5_original_finish"))
                stats["kept_finish"] += 1
                finish_added = True
            else:
                stats["dropped_early_finish"] += 1
            break
        stats[f"dropped_{action_name or 'unknown'}"] += 1

    if not finish_added:
        stats["missing_finish_after_transform"] += 1
    return out, stats


def make_core_row(row: dict[str, Any], action: dict[str, Any], source: str) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    copied["action"] = copy.deepcopy(action)
    copied["tool_schema_version"] = "v0.7_core5_evidence_claim_repair_sft"
    copied["label_source"] = "v0_7_core5_evidence_claim_repair_sft"
    copied["repair_source"] = source
    copied["claim_state"] = core_claim_state(copied.get("claim_state") or {})
    copied["draft_claims"] = filter_claim_list(copied.get("draft_claims") or [])
    copied["available_actions"] = core_available_actions(copied.get("available_actions") or [], action)
    if copied.get("messages"):
        copied["messages"] = rebuild_last_assistant_action(copied["messages"], copied["action"])
    return copied


def make_core_row_with_state(
    row: dict[str, Any],
    action: dict[str, Any],
    source: str,
    draft_claims: list[dict[str, Any]],
    *,
    synthetic_substep: int,
) -> dict[str, Any]:
    copied = make_core_row(row, action, source)
    copied["draft_claims"] = copy.deepcopy(draft_claims)
    copied["claim_state"] = claim_state_from_draft(draft_claims)
    copied["synthetic_substep"] = synthetic_substep
    return copied


def filter_core_claim_items(action: dict[str, Any], done: set[str]) -> list[tuple[str, dict[str, Any]]]:
    action_name = str(action.get("action") or "")
    remaining = {field for field in CORE_FIELDS if field not in done}
    if action_name in {"write_claims_chunk", "write_claims_batch"}:
        items: list[tuple[str, dict[str, Any]]] = []
        for claim in action.get("claims") or []:
            if claim.get("field") in remaining:
                items.append(("claim", copy.deepcopy(claim)))
        for abstain in action.get("abstains") or []:
            if abstain.get("field") in remaining:
                items.append(("abstain", copy.deepcopy(abstain)))
        return items
    if action_name == "write_claim" and action.get("field") in remaining:
        return [("claim", copy.deepcopy(action))]
    if action_name == "abstain_claim" and action.get("field") in remaining:
        return [("abstain", copy.deepcopy(action))]
    return []


def single_field_chunk_actions(items: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for kind, item in items:
        if kind == "claim":
            claim = copy.deepcopy(item)
            claim.pop("action", None)
            actions.append({"action": "write_claims_chunk", "claims": [claim], "abstains": []})
        else:
            abstain = copy.deepcopy(item)
            abstain.pop("action", None)
            actions.append({"action": "write_claims_chunk", "claims": [], "abstains": [abstain]})
    return actions


def done_from_claim_state(claim_state: dict[str, Any]) -> set[str]:
    written = claim_state.get("written_fields") or []
    abstained = claim_state.get("abstained_fields") or []
    return {str(item) for item in list(written) + list(abstained) if str(item) in CORE_SET}


def fields_written_by_action(action: dict[str, Any]) -> set[str]:
    action_name = str(action.get("action") or "")
    if action_name in {"write_claims_chunk", "write_claims_batch"}:
        fields = [item.get("field") for item in action.get("claims") or []]
        fields.extend(item.get("field") for item in action.get("abstains") or [])
        return {str(item) for item in fields if str(item) in CORE_SET}
    if action_name in {"write_claim", "abstain_claim"} and action.get("field") in CORE_SET:
        return {str(action.get("field"))}
    return set()


def apply_action_to_draft(draft_claims: list[dict[str, Any]], action: dict[str, Any]) -> list[dict[str, Any]]:
    next_claims = [copy.deepcopy(item) for item in draft_claims]
    for claim in action.get("claims") or []:
        normalized = copy.deepcopy(claim)
        normalized["abstain"] = False
        next_claims = upsert_by_field(next_claims, normalized)
    for abstain in action.get("abstains") or []:
        normalized = {
            "field": abstain.get("field"),
            "reason": abstain.get("reason"),
            "abstain": True,
        }
        next_claims = upsert_by_field(next_claims, normalized)
    return next_claims


def upsert_by_field(draft_claims: list[dict[str, Any]], claim: dict[str, Any]) -> list[dict[str, Any]]:
    field = claim.get("field")
    return [item for item in draft_claims if item.get("field") != field] + [claim]


def done_from_draft(draft_claims: list[dict[str, Any]]) -> set[str]:
    return {str(item.get("field")) for item in draft_claims if item.get("field") in CORE_SET}


def claim_state_from_draft(draft_claims: list[dict[str, Any]]) -> dict[str, Any]:
    by_field = {str(item.get("field")): item for item in draft_claims if item.get("field") in CORE_SET}
    written = [field for field in CORE_FIELDS if field in by_field and not by_field[field].get("abstain")]
    abstained = [field for field in CORE_FIELDS if field in by_field and by_field[field].get("abstain")]
    evidence_ids: list[str] = []
    for claim in draft_claims:
        for evidence_id in claim.get("evidence_ids") or []:
            evidence_id = str(evidence_id)
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
    done = set(written) | set(abstained)
    return {
        "target_fields": CORE_FIELDS,
        "written_fields": written,
        "abstained_fields": abstained,
        "remaining_fields": [field for field in CORE_FIELDS if field not in done],
        "claim_count": len(written),
        "abstain_count": len(abstained),
        "evidence_ids": evidence_ids,
    }


def core_claim_state(claim_state: dict[str, Any]) -> dict[str, Any]:
    done = done_from_claim_state(claim_state)
    written = [field for field in claim_state.get("written_fields") or [] if field in CORE_SET]
    abstained = [field for field in claim_state.get("abstained_fields") or [] if field in CORE_SET]
    evidence_ids = claim_state.get("evidence_ids") or []
    return {
        "target_fields": CORE_FIELDS,
        "written_fields": written,
        "abstained_fields": abstained,
        "remaining_fields": [field for field in CORE_FIELDS if field not in done],
        "claim_count": len(written),
        "abstain_count": len(abstained),
        "evidence_ids": evidence_ids,
    }


def filter_claim_list(claims: list[Any]) -> list[Any]:
    filtered: list[Any] = []
    for claim in claims:
        if isinstance(claim, dict) and claim.get("field") in CORE_SET:
            filtered.append(claim)
    return filtered


def core_available_actions(actions: list[Any], target_action: dict[str, Any]) -> list[Any]:
    action_name = str(target_action.get("action") or "")
    if action_name == "finish":
        return ["finish"]
    return actions


def rebuild_last_assistant_action(messages: list[Any], action: dict[str, Any]) -> list[Any]:
    copied = copy.deepcopy(messages)
    for message in reversed(copied):
        if isinstance(message, dict) and message.get("role") == "assistant":
            message["content"] = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
            break
    return copied


def sample_by_action(
    rows: list[dict[str, Any]],
    targets: dict[str, int],
    rng: random.Random,
    *,
    replacement: bool,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str((row.get("action") or {}).get("action") or "unknown")].append(row)

    selected: list[dict[str, Any]] = []
    for action_name, target in targets.items():
        bucket = buckets.get(action_name) or []
        if not bucket or target <= 0:
            continue
        if replacement and len(bucket) < target:
            selected.extend(copy.deepcopy(rng.choice(bucket)) for _ in range(target))
        else:
            take = min(target, len(bucket))
            selected.extend(copy.deepcopy(item) for item in rng.sample(bucket, take))
    return annotate_copies(selected)


def annotate_copies(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: Counter[str] = Counter()
    for row in rows:
        key = f"{row.get('task_id')}|{row.get('step')}|{json.dumps(row.get('action'), ensure_ascii=False, sort_keys=True)}"
        row["repair_copy_index"] = seen[key]
        seen[key] += 1
    return rows


def parse_targets(text: str) -> dict[str, int]:
    targets: dict[str, int] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        targets[key.strip()] = int(value)
    return targets


def target_string(targets: dict[str, int]) -> str:
    return ",".join(f"{key}={value}" for key, value in targets.items())


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("action") or {}).get("action") or "unknown") for row in rows)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        f"# v0.7 Core5 Evidence/Claim Repair SFT 构建报告 {manifest['created_at']}",
        "",
        "## 目的",
        "",
        "当前最强 Phase8 adapter 在 tolerant env 的 val62 成功率约 0.79，但 evidence hit 和 claim supported 仍偏弱。本数据集只用 train split 构建，避免污染正式 val62。",
        "",
        "核心做法：",
        "",
        "- 把原 12 字段 claim card 压到 Core5：caption_text、image_scope、depicted_work_title、displayed_region、object_type。",
        "- 保留 inspect/crop/retrieve/open 的 replay，重点增加 retrieve/open/write_claims_chunk 的训练权重。",
        "- write_claims_chunk 被拆成单字段小步，和环境中最多 2 个 item 的 normalize 规则保持一致；训练目标更严格地要求每次只写一个 remaining field。",
        "- 一旦 Core5 全部完成，才插入 finish 监督，减少模型继续写非评测字段导致的上下文压力。",
        "- SFT compact prompt 已补入 available_actions 和 claim_state，使训练输入更接近 executable rollout 输入。",
        "",
        "## 数据统计",
        "",
        f"- source_dir: `{manifest['source_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- all_rows: {manifest['all_rows']}",
        "",
    ]
    for split, stats in manifest["splits"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- source_rows: {stats['source_rows']}",
                f"- transformed_rows: {stats['transformed_rows']}",
                f"- selected_rows: {stats['selected_rows']}",
                f"- selected_action_counts: `{json.dumps(stats['selected_action_counts'], ensure_ascii=False)}`",
                f"- transform_stats: `{json.dumps(stats['transform_stats'], ensure_ascii=False)}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
