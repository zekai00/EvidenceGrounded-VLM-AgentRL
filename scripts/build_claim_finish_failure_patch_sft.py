#!/usr/bin/env python3
"""Build v0.7 claim/finish failure patch SFT rows.

This patch set targets three rollout failures observed after CaptionRank:
- premature finish while claim_state.remaining_fields is non-empty;
- oversized write_claims_chunk actions that are likely to be truncated;
- claim items accidentally placed in abstains.

Rows are rebuilt with the current executable-env prompt so the model sees the
latest final_action_guard, target caption hints, and 1-2 field chunk protocol.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env import EvidenceAgentEnv  # noqa: E402
from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation, build_prompt_text  # noqa: E402


DEFAULT_TASKS = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_7_inspect_crop_sft_20260605_2336/tasks_all.jsonl"
)
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)
DEFAULT_CLAIM_SFT = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/v0_7_claim_grounding_phase_mask_patch_sft_20260606_1632/sft"
)
DEFAULT_FINISH_SFT = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/v0_7_finish_ready_patch_sft_20260606_0356/sft"
)
DEFAULT_FINISH_VIOLATION_SFT = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/v0_7_finish_violation_patch_sft_20260606_1850/sft"
)

TARGET_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--evidence-index", type=Path, default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--claim-sft-dir", type=Path, default=DEFAULT_CLAIM_SFT)
    parser.add_argument("--finish-sft-dir", type=Path, default=DEFAULT_FINISH_SFT)
    parser.add_argument("--finish-violation-sft-dir", type=Path, default=DEFAULT_FINISH_VIOLATION_SFT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", action="append", default=[])
    parser.add_argument("--failure-oversample", type=int, default=48)
    parser.add_argument("--claim-rows", type=int, default=360)
    parser.add_argument("--finish-ready-rows", type=int, default=96)
    parser.add_argument("--finish-violation-rows", type=int, default=160)
    parser.add_argument("--val-claim-rows", type=int, default=96)
    parser.add_argument("--val-finish-ready-rows", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--seed", type=int, default=60606)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "sft").mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(args.tasks)
    task_by_id = {str(task["task_id"]): task for task in tasks}
    prompt_config = PromptConfig(
        max_history_actions=8,
        max_tool_results=6,
        max_evidence_per_result=3,
        snippet_chars=180,
        max_text_chars=16000,
        head_text_chars=4000,
        coordinate_info=True,
        tool_schema="inspect_crop",
        compact_claim_state=False,
        region_selection_hint=True,
        strict_claim_phase_hint=True,
    )

    failure_rows = collect_failure_rows(args, task_by_id, prompt_config)
    claim_train = collect_regular_claim_rows(args.claim_sft_dir / "train.jsonl", task_by_id, prompt_config, args.claim_rows, rng)
    finish_violation_train = collect_regular_claim_rows(
        args.finish_violation_sft_dir / "train.jsonl",
        task_by_id,
        prompt_config,
        args.finish_violation_rows,
        rng,
    )
    finish_ready_train = collect_finish_ready_rows(
        args.finish_sft_dir / "train.jsonl",
        task_by_id,
        prompt_config,
        args.finish_ready_rows,
        rng,
    )
    claim_val = collect_regular_claim_rows(args.claim_sft_dir / "val.jsonl", task_by_id, prompt_config, args.val_claim_rows, rng)
    finish_ready_val = collect_finish_ready_rows(
        args.finish_sft_dir / "val.jsonl",
        task_by_id,
        prompt_config,
        args.val_finish_ready_rows,
        rng,
    )

    train_rows: list[dict[str, Any]] = []
    for row in failure_rows:
        for copy_index in range(args.failure_oversample):
            copied = deep_copy(row)
            copied["claim_finish_patch"]["copy_index"] = copy_index
            copied["claim_finish_patch"]["oversample"] = args.failure_oversample
            train_rows.append(copied)
    train_rows.extend(claim_train)
    train_rows.extend(finish_violation_train)
    train_rows.extend(finish_ready_train)
    rng.shuffle(train_rows)

    val_rows = [deep_copy(row) for row in failure_rows]
    val_rows.extend(claim_val)
    val_rows.extend(finish_ready_val)
    rng.shuffle(val_rows)

    write_jsonl(args.output_dir / "sft" / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "sft" / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "failure_rows.jsonl", failure_rows)

    manifest = {
        "created_at": now(),
        "dataset_version": "v0.7_claim_finish_failure_patch_sft",
        "purpose": "Repair claim_continuation premature finish, oversized chunks, and claim/abstain schema slips.",
        "tasks": str(args.tasks),
        "evidence_index": str(args.evidence_index),
        "rollout_jsonl": args.rollout_jsonl,
        "sources": {
            "claim_sft_dir": str(args.claim_sft_dir),
            "finish_sft_dir": str(args.finish_sft_dir),
            "finish_violation_sft_dir": str(args.finish_violation_sft_dir),
        },
        "output_dir": str(args.output_dir),
        "failure_rows": len(failure_rows),
        "failure_oversample": args.failure_oversample,
        "claim_train_rows": len(claim_train),
        "finish_violation_train_rows": len(finish_violation_train),
        "finish_ready_train_rows": len(finish_ready_train),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_action_counts": dict(action_counter(train_rows)),
        "val_action_counts": dict(action_counter(val_rows)),
        "train_patch_kind_counts": dict(kind_counter(train_rows)),
        "val_patch_kind_counts": dict(kind_counter(val_rows)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "构建报告.md", manifest, failure_rows)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def collect_failure_rows(
    args: argparse.Namespace,
    task_by_id: dict[str, dict[str, Any]],
    prompt_config: PromptConfig,
) -> list[dict[str, Any]]:
    env = EvidenceAgentEnv(
        args.tasks,
        args.evidence_index,
        args.output_dir / "_replay_env",
        max_steps=args.max_steps,
        include_gold_regions=False,
        phase_aware_mask=True,
        enforce_tool_mask=True,
        tool_schema="inspect_crop",
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path_str in args.rollout_jsonl:
        for trajectory in read_jsonl(Path(path_str)):
            for row in failure_rows_from_trajectory(env, trajectory, task_by_id, prompt_config):
                key = "|".join(
                    [
                        str(row.get("task_id")),
                        str(row.get("failure_step")),
                        json.dumps(row.get("action"), ensure_ascii=False, sort_keys=True),
                    ]
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    return rows


def failure_rows_from_trajectory(
    env: EvidenceAgentEnv,
    trajectory: dict[str, Any],
    task_by_id: dict[str, dict[str, Any]],
    prompt_config: PromptConfig,
) -> list[dict[str, Any]]:
    task_id = str(trajectory.get("task_id") or "")
    task = task_by_id.get(task_id)
    if not task:
        return []
    rows: list[dict[str, Any]] = []
    obs = env.reset(task_id=task_id)
    for step in trajectory.get("steps") or []:
        parsed = step.get("parsed_action")
        action_name = str(parsed.get("action")) if isinstance(parsed, dict) else ""
        phase = str((step.get("tool_mask") or {}).get("phase") or "")
        result = step.get("result") or {}
        if bool(step.get("mask_violation")) and action_name == "finish" and phase == "claim_continuation":
            target_action = build_next_claim_action(task, obs, max_items=2)
            rows.append(make_row(task, obs, target_action, prompt_config, step, "premature_finish_claim_continuation"))
        elif action_name in {"", "invalid"} or result.get("error"):
            if phase in {"claim_continuation", "claim_ready", "evidence_opening"}:
                target_action = build_next_claim_action(task, obs, max_items=2)
                rows.append(make_row(task, obs, target_action, prompt_config, step, "claim_schema_or_tool_error"))
        if not isinstance(parsed, dict):
            break
        obs, _, terminated, _ = env.step(parsed)
        if terminated:
            break
    return rows


def collect_regular_claim_rows(
    path: Path,
    task_by_id: dict[str, dict[str, Any]],
    prompt_config: PromptConfig,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row_action(row) in {"write_claims_chunk", "write_claim", "abstain_claim"}]
    rng.shuffle(rows)
    selected = rows[:limit] if limit > 0 else rows
    result: list[dict[str, Any]] = []
    for row in selected:
        task = task_by_id.get(str(row.get("task_id")))
        if not task:
            continue
        action = normalize_training_action(row.get("action") or {}, max_items=2)
        if not action:
            continue
        obs = obs_from_row(row, task)
        result.append(make_regular_row(row, task, obs, action, prompt_config, "regular_claim_small_chunk"))
    return result


def collect_finish_ready_rows(
    path: Path,
    task_by_id: dict[str, dict[str, Any]],
    prompt_config: PromptConfig,
    limit: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    rows = [row for row in read_jsonl(path) if row_action(row) == "finish"]
    ready_rows = [
        row
        for row in rows
        if not ((row.get("claim_state") or {}).get("remaining_fields") or [])
        or str((row.get("tool_mask") or {}).get("phase")) == "finish_ready"
    ]
    rng.shuffle(ready_rows)
    selected = ready_rows[:limit] if limit > 0 else ready_rows
    result: list[dict[str, Any]] = []
    for row in selected:
        task = task_by_id.get(str(row.get("task_id")))
        if not task:
            continue
        action = {"action": "finish", "status": "done"}
        obs = obs_from_row(row, task)
        obs["available_actions"] = ["finish"]
        obs["tool_mask"] = {
            "enabled": True,
            "phase": "finish_ready",
            "allowed_actions": ["finish"],
            "blocked_actions": [],
            "reason": "claim_state has no remaining fields; finish the trajectory.",
            "step": len(obs.get("history") or []),
            "tool_schema": "inspect_crop",
        }
        result.append(make_regular_row(row, task, obs, action, prompt_config, "finish_ready_positive"))
    return result


def build_next_claim_action(task: dict[str, Any], obs: dict[str, Any], max_items: int = 2) -> dict[str, Any]:
    remaining = list((obs.get("claim_state") or {}).get("remaining_fields") or [])
    if not remaining:
        return {"action": "finish", "status": "done"}
    visible_ids = set(map(str, obs.get("visible_evidence_ids") or []))
    local_ids = [evidence_id for evidence_id in visible_ids if evidence_id.startswith("local_caption_")]
    gold_by_field = {str(claim.get("field")): claim for claim in (task.get("gold") or {}).get("claims") or []}
    claims: list[dict[str, Any]] = []
    abstains: list[dict[str, Any]] = []
    for field in remaining:
        if len(claims) + len(abstains) >= max_items:
            break
        gold = gold_by_field.get(str(field)) or {}
        if gold.get("abstain"):
            abstains.append({"field": field, "reason": str(gold.get("reason") or "当前证据不足，不能写成确定 claim")})
            continue
        evidence_ids = [str(item) for item in gold.get("evidence_ids") or [] if str(item) in visible_ids]
        if not evidence_ids and field in {
            "caption_text",
            "image_scope",
            "depicted_work_title",
            "displayed_region",
            "object_type",
            "artist",
            "dynasty",
            "medium_dimensions",
            "collection",
        }:
            evidence_ids = local_ids[:1]
        if "value" in gold and evidence_ids:
            claims.append(
                {
                    "field": field,
                    "value": simplify_value(gold.get("value")),
                    "evidence_ids": evidence_ids[:3],
                    "visual_bbox": gold.get("visual_bbox"),
                    "confidence": float(gold.get("confidence") or 0.8),
                }
            )
        else:
            abstains.append({"field": field, "reason": "当前可见证据不足，不能写成确定 claim"})
    return {"action": "write_claims_chunk", "claims": claims, "abstains": abstains}


def normalize_training_action(action: dict[str, Any], max_items: int = 2) -> dict[str, Any] | None:
    name = str(action.get("action") or "")
    if name == "write_claim":
        repaired = dict(action)
        repaired["value"] = simplify_value(repaired.get("value"))
        return repaired
    if name == "abstain_claim":
        return dict(action)
    if name not in {"write_claims_chunk", "write_claims_batch"}:
        return None
    claims = []
    abstains = []
    for item in action.get("claims") or []:
        if len(claims) + len(abstains) >= max_items:
            break
        copied = dict(item)
        copied["value"] = simplify_value(copied.get("value"))
        claims.append(copied)
    for item in action.get("abstains") or []:
        if len(claims) + len(abstains) >= max_items:
            break
        if isinstance(item, dict) and "reason" not in item and ("value" in item or "evidence_ids" in item):
            copied = dict(item)
            copied["value"] = simplify_value(copied.get("value"))
            claims.append(copied)
        else:
            abstains.append(dict(item))
    if not claims and not abstains:
        return None
    return {"action": "write_claims_chunk", "claims": claims, "abstains": abstains}


def simplify_value(value: Any) -> Any:
    if isinstance(value, list):
        deduped = []
        for item in value:
            if item not in deduped:
                deduped.append(item)
        return deduped[:5]
    if isinstance(value, str) and len(value) > 220:
        return value[:217] + "..."
    return value


def obs_from_row(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "goal": task.get("goal"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "images": [{"role": "page_image" if i == 0 else "last_crop", "path": path} for i, path in enumerate(row.get("images") or [])],
        "history": row.get("history") or [],
        "tool_results": row.get("tool_results") or [],
        "draft_claims": row.get("draft_claims") or [],
        "claim_state": row.get("claim_state") or claim_state_from_drafts(row.get("draft_claims") or []),
        "regions": row.get("regions") or [],
        "available_region_ids": row.get("available_region_ids") or [],
        "selected_evidence_ids": row.get("selected_evidence_ids") or [],
        "visible_evidence_ids": row.get("visible_evidence_ids") or visible_evidence_ids(row),
        "target_evidence_hints": target_evidence_hints(task),
        "valid_crop_count": row.get("valid_crop_count") or 0,
        "tool_schema": "inspect_crop",
        "available_actions": row.get("available_actions") or [],
        "tool_mask": row.get("tool_mask") or {},
    }


def make_regular_row(
    source_row: dict[str, Any],
    task: dict[str, Any],
    obs: dict[str, Any],
    action: dict[str, Any],
    prompt_config: PromptConfig,
    kind: str,
) -> dict[str, Any]:
    step = source_row.get("step")
    row = {
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "split": task.get("split"),
        "variant": (task.get("candidate_augmentation") or {}).get("variant"),
        "step": step,
        "tool_schema_version": "v0.7_claim_finish_failure_patch",
        "action": action,
        "history": obs.get("history") or [],
        "tool_results": obs.get("tool_results") or [],
        "draft_claims": obs.get("draft_claims") or [],
        "claim_state": obs.get("claim_state") or {},
        "selected_evidence_ids": obs.get("selected_evidence_ids") or [],
        "visible_evidence_ids": obs.get("visible_evidence_ids") or [],
        "available_actions": obs.get("available_actions") or [],
        "tool_mask": obs.get("tool_mask") or {},
        "available_region_ids": obs.get("available_region_ids") or [],
        "regions": obs.get("regions") or [],
        "valid_crop_count": obs.get("valid_crop_count") or 0,
        "images": [item.get("path") for item in obs.get("images") or [] if isinstance(item, dict) and item.get("path")],
        "label_source": "v0_7_claim_finish_failure_patch_sft",
        "claim_finish_patch": {"kind": kind},
    }
    row["prompt_text"] = build_prompt_text(obs, prompt_config)
    row["messages"] = build_messages_from_observation(obs, prompt_config, include_assistant_action=action)
    return row


def make_row(
    task: dict[str, Any],
    obs: dict[str, Any],
    action: dict[str, Any],
    prompt_config: PromptConfig,
    failure_step: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    row = make_regular_row(
        {
            "step": failure_step.get("step"),
            "history": obs.get("history") or [],
            "tool_results": obs.get("tool_results") or [],
            "draft_claims": obs.get("draft_claims") or [],
            "claim_state": obs.get("claim_state") or {},
            "selected_evidence_ids": obs.get("selected_evidence_ids") or [],
            "visible_evidence_ids": obs.get("visible_evidence_ids") or [],
            "available_actions": obs.get("available_actions") or [],
            "tool_mask": obs.get("tool_mask") or {},
            "available_region_ids": obs.get("available_region_ids") or [],
            "regions": obs.get("regions") or [],
            "valid_crop_count": obs.get("valid_crop_count") or 0,
            "images": [item.get("path") for item in obs.get("images") or [] if isinstance(item, dict) and item.get("path")],
        },
        task,
        obs,
        action,
        prompt_config,
        kind,
    )
    row["failure_step"] = failure_step.get("step")
    row["claim_finish_patch"].update(
        {
            "bad_action": failure_step.get("parsed_action"),
            "bad_raw_text": failure_step.get("raw_text"),
            "bad_phase": (failure_step.get("tool_mask") or {}).get("phase"),
            "bad_result": failure_step.get("result"),
            "remaining_fields": (obs.get("claim_state") or {}).get("remaining_fields") or [],
        }
    )
    return row


def target_evidence_hints(task: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    for item in task.get("local_evidence") or []:
        evidence_id = str(item.get("evidence_id") or "")
        snippet = item.get("display_snippet") or item.get("text") or ""
        if evidence_id and snippet:
            hints.append(
                {
                    "evidence_id": evidence_id,
                    "source_file": item.get("source_file"),
                    "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
                    "citation_level": item.get("citation_level"),
                    "display_snippet": snippet,
                }
            )
    return hints[:3]


def visible_evidence_ids(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for evidence_id in row.get("selected_evidence_ids") or []:
        add_unique(ids, str(evidence_id))
    for result in row.get("tool_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("tool") == "retrieve_evidence":
            for item in result.get("results") or []:
                add_unique(ids, str(item.get("evidence_id") or ""))
        elif result.get("tool") == "open_evidence":
            add_unique(ids, str(result.get("evidence_id") or ""))
    return ids


def claim_state_from_drafts(drafts: list[dict[str, Any]]) -> dict[str, Any]:
    by_field = {str(item.get("field")): item for item in drafts if isinstance(item, dict) and item.get("field")}
    return {
        "target_fields": TARGET_FIELDS,
        "written_fields": [field for field in TARGET_FIELDS if field in by_field and not by_field[field].get("abstain")],
        "abstained_fields": [field for field in TARGET_FIELDS if field in by_field and by_field[field].get("abstain")],
        "remaining_fields": [field for field in TARGET_FIELDS if field not in by_field],
        "claim_count": sum(1 for field in TARGET_FIELDS if field in by_field and not by_field[field].get("abstain")),
        "abstain_count": sum(1 for field in TARGET_FIELDS if field in by_field and by_field[field].get("abstain")),
    }


def add_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def row_action(row: dict[str, Any]) -> str:
    action = row.get("action") or {}
    return str(action.get("action") or "") if isinstance(action, dict) else ""


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row_action(row) for row in rows)


def kind_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("claim_finish_patch") or {}).get("kind") or "") for row in rows)


def deep_copy(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False))


def write_report(path: Path, manifest: dict[str, Any], failure_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.7 Claim/Finish Failure Patch SFT 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- failure_rows: {manifest['failure_rows']}",
        f"- train_rows: {manifest['train_rows']}",
        f"- val_rows: {manifest['val_rows']}",
        f"- train_action_counts: `{json.dumps(manifest['train_action_counts'], ensure_ascii=False)}`",
        f"- train_patch_kind_counts: `{json.dumps(manifest['train_patch_kind_counts'], ensure_ascii=False)}`",
        "",
        "## 设计",
        "",
        "- 真实 rollout 失败状态 oversample，用于修 premature finish 和 claim schema/tool error。",
        "- 常规 claim rows 统一缩小到每次 1-2 个字段，降低 JSON 截断概率。",
        "- 混入 finish-ready 正例，避免模型误以为 claim 阶段永远不能 finish。",
        "",
        "## 失败样本预览",
        "",
    ]
    for row in failure_rows[:20]:
        patch = row.get("claim_finish_patch") or {}
        lines.append(
            f"- task_id: `{row.get('task_id')}`; step: {row.get('failure_step')}; kind: {patch.get('kind')}; "
            f"remaining_fields: `{json.dumps(patch.get('remaining_fields'), ensure_ascii=False)}`; "
            f"target: `{json.dumps(row.get('action'), ensure_ascii=False)}`"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
