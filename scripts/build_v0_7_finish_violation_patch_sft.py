#!/usr/bin/env python3
"""Build a small v0.7 patch set for premature finish in claim_continuation."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--evidence-index", type=Path, default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--claim-sft-dir", type=Path, default=DEFAULT_CLAIM_SFT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rollout-jsonl", action="append", default=[])
    parser.add_argument("--group-jsonl", action="append", default=[])
    parser.add_argument("--failure-oversample", type=int, default=24)
    parser.add_argument("--regular-train-rows", type=int, default=320)
    parser.add_argument("--regular-val-rows", type=int, default=96)
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
        max_history_actions=6,
        max_tool_results=5,
        max_evidence_per_result=2,
        snippet_chars=120,
        max_text_chars=10000,
        head_text_chars=3000,
        coordinate_info=True,
        tool_schema="inspect_crop",
        compact_claim_state=True,
        region_selection_hint=True,
        strict_claim_phase_hint=False,
    )

    failure_rows = collect_failure_rows(args, task_by_id, prompt_config)
    regular_train = sample_regular_rows(args.claim_sft_dir / "train.jsonl", args.regular_train_rows, rng)
    regular_val = sample_regular_rows(args.claim_sft_dir / "val.jsonl", args.regular_val_rows, rng)

    train_rows: list[dict[str, Any]] = []
    for row in failure_rows:
        for copy_index in range(args.failure_oversample):
            copied = json.loads(json.dumps(row, ensure_ascii=False))
            copied["finish_violation_patch"]["copy_index"] = copy_index
            copied["finish_violation_patch"]["oversample"] = args.failure_oversample
            train_rows.append(copied)
    train_rows.extend(regular_train)
    rng.shuffle(train_rows)

    val_rows = [json.loads(json.dumps(row, ensure_ascii=False)) for row in failure_rows]
    val_rows.extend(regular_val)
    rng.shuffle(val_rows)

    write_jsonl(args.output_dir / "sft" / "train.jsonl", train_rows)
    write_jsonl(args.output_dir / "sft" / "val.jsonl", val_rows)
    write_jsonl(args.output_dir / "failure_rows.jsonl", failure_rows)

    manifest = {
        "created_at": now(),
        "dataset_version": "v0.7_finish_violation_patch_sft",
        "purpose": "Repair premature finish in claim_continuation by replaying model failure states and supervising write_claims_chunk/abstain.",
        "tasks": str(args.tasks),
        "evidence_index": str(args.evidence_index),
        "claim_sft_dir": str(args.claim_sft_dir),
        "rollout_jsonl": args.rollout_jsonl,
        "group_jsonl": args.group_jsonl,
        "output_dir": str(args.output_dir),
        "failure_rows": len(failure_rows),
        "failure_oversample": args.failure_oversample,
        "regular_train_rows": len(regular_train),
        "regular_val_rows": len(regular_val),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_action_counts": dict(action_counter(train_rows)),
        "val_action_counts": dict(action_counter(val_rows)),
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
    for trajectory in iter_trajectories(args):
        row = failure_row_from_trajectory(env, trajectory, task_by_id, prompt_config)
        if not row:
            continue
        key = f"{row['task_id']}|{row['failure_step']}|{json.dumps(row['action'], ensure_ascii=False, sort_keys=True)}"
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def iter_trajectories(args: argparse.Namespace):
    for path_str in args.rollout_jsonl:
        for row in read_jsonl(Path(path_str)):
            yield row
    for path_str in args.group_jsonl:
        for group in read_jsonl(Path(path_str)):
            task_id = str(group.get("task_id") or "")
            for sample in group.get("samples") or []:
                yield {
                    "task_id": task_id,
                    "split": group.get("split"),
                    "steps": sample.get("steps") or [],
                    "sample_index": sample.get("sample_index"),
                    "group_reward": sample.get("group_reward"),
                }


def failure_row_from_trajectory(
    env: EvidenceAgentEnv,
    trajectory: dict[str, Any],
    task_by_id: dict[str, dict[str, Any]],
    prompt_config: PromptConfig,
) -> dict[str, Any] | None:
    task_id = str(trajectory.get("task_id") or "")
    task = task_by_id.get(task_id)
    if not task:
        return None
    obs = env.reset(task_id=task_id)
    for step in trajectory.get("steps") or []:
        parsed = step.get("parsed_action")
        action_name = str(parsed.get("action")) if isinstance(parsed, dict) else ""
        phase = str((step.get("tool_mask") or {}).get("phase") or "")
        if bool(step.get("mask_violation")) and action_name == "finish" and phase == "claim_continuation":
            target_action = build_target_action(task, obs)
            return make_row(task, obs, target_action, prompt_config, step)
        if not isinstance(parsed, dict):
            return None
        obs, _, terminated, info = env.step(parsed)
        if terminated or info.get("result", {}).get("error"):
            return None
    return None


def build_target_action(task: dict[str, Any], obs: dict[str, Any]) -> dict[str, Any]:
    remaining = list((obs.get("claim_state") or {}).get("remaining_fields") or [])
    gold_by_field = {str(claim.get("field")): claim for claim in (task.get("gold") or {}).get("claims") or []}
    claims: list[dict[str, Any]] = []
    abstains: list[dict[str, Any]] = []
    for field in remaining[:4]:
        gold = gold_by_field.get(str(field)) or {}
        if gold.get("abstain"):
            abstains.append(
                {
                    "field": field,
                    "reason": str(gold.get("reason") or "当前证据不足，不能写成确定 claim"),
                }
            )
            continue
        if "value" in gold:
            claim = {
                "field": field,
                "value": gold.get("value"),
                "evidence_ids": list(gold.get("evidence_ids") or []),
                "visual_bbox": gold.get("visual_bbox"),
                "confidence": float(gold.get("confidence") or 0.8),
            }
            claims.append(claim)
    if claims or abstains:
        return {"action": "write_claims_chunk", "claims": claims, "abstains": abstains}
    field = str(remaining[0]) if remaining else "collection"
    return {"action": "abstain_claim", "field": field, "reason": "claim_state 仍有待写字段，不能提前 finish；当前证据不足。"}


def make_row(
    task: dict[str, Any],
    obs: dict[str, Any],
    action: dict[str, Any],
    prompt_config: PromptConfig,
    failure_step: dict[str, Any],
) -> dict[str, Any]:
    messages = build_messages_from_observation(obs, prompt_config, include_assistant_action=action)
    return {
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "split": task.get("split"),
        "variant": (task.get("candidate_augmentation") or {}).get("variant"),
        "step": failure_step.get("step"),
        "failure_step": failure_step.get("step"),
        "tool_schema_version": "v0.7_finish_violation_patch",
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
        "regions": obs.get("regions") or {},
        "valid_crop_count": obs.get("valid_crop_count") or 0,
        "images": [item.get("path") for item in obs.get("images") or [] if isinstance(item, dict) and item.get("path")],
        "prompt_text": build_prompt_text(obs, prompt_config),
        "messages": messages,
        "label_source": "v0_7_finish_violation_patch_sft",
        "finish_violation_patch": {
            "kind": "premature_finish_claim_continuation",
            "bad_action": failure_step.get("parsed_action"),
            "bad_raw_text": failure_step.get("raw_text"),
            "bad_phase": (failure_step.get("tool_mask") or {}).get("phase"),
            "remaining_fields": (obs.get("claim_state") or {}).get("remaining_fields") or [],
        },
    }


def sample_regular_rows(path: Path, limit: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = read_jsonl(path) if path.exists() else []
    if len(rows) <= limit:
        selected = rows
    else:
        selected = rng.sample(rows, limit)
    result = []
    for row in selected:
        copied = json.loads(json.dumps(row, ensure_ascii=False))
        copied["finish_violation_patch"] = {"kind": "regular_claim_replay"}
        result.append(copied)
    return result


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("action") or {}).get("action") or "") for row in rows)


def write_report(path: Path, manifest: dict[str, Any], failure_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.7 Finish-Violation Patch SFT 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- failure_rows: {manifest['failure_rows']}",
        f"- train_rows: {manifest['train_rows']}",
        f"- val_rows: {manifest['val_rows']}",
        f"- train_action_counts: `{json.dumps(manifest['train_action_counts'], ensure_ascii=False)}`",
        "",
        "## 失败类型",
        "",
        "模型在 `claim_continuation` 阶段还有 remaining fields 时提前输出 `finish`。这时 `finish` 被 phase-aware mask 禁止，正确行为应该是继续 `write_claims_chunk` 或对证据不足字段 `abstain`。",
        "",
        "## 失败样本",
        "",
    ]
    for row in failure_rows[:20]:
        patch = row.get("finish_violation_patch") or {}
        lines.extend(
            [
                f"- task_id: `{row.get('task_id')}`; step: {row.get('failure_step')}; remaining_fields: `{json.dumps(patch.get('remaining_fields'), ensure_ascii=False)}`; target: `{json.dumps(row.get('action'), ensure_ascii=False)}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
