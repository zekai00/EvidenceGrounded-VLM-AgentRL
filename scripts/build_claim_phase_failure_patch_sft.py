#!/usr/bin/env python3
"""Build claim-phase failure-state SFT patch data from executable rollouts."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation  # noqa: E402
from evidence_agent_env.tools.claim_tools import apply_claim_write, claim_state, normalize_abstain, normalize_claim  # noqa: E402


DEFAULT_SFT_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_6_chunked_claim_sft_20260604_1650/sft"
)
DEFAULT_TASKS = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_6_chunked_claim_sft_20260604_1650/tasks_all.jsonl"
)
DEFAULT_ROLLOUTS = (
    "outputs/rollout_v0_6_250step_val8_no_regionhint_20260605_0156/rollouts.jsonl,"
    "outputs/rollout_v0_6_250step_val8_regionhint_20260605_0156/rollouts.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", default=DEFAULT_ROLLOUTS, help="Comma-separated rollout JSONL files.")
    parser.add_argument("--tasks", default=str(DEFAULT_TASKS), help="Comma-separated task JSONL files.")
    parser.add_argument("--base-sft-dir", type=Path, default=DEFAULT_SFT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-train-rows", type=int, default=2000)
    parser.add_argument("--replay-val-rows", type=int, default=256)
    parser.add_argument("--replay-test-rows", type=int, default=256)
    parser.add_argument("--patch-oversample", type=int, default=4)
    parser.add_argument("--max-patches-per-task", type=int, default=4)
    parser.add_argument("--claim-chunk-size", type=int, default=4)
    parser.add_argument("--region-selection-hint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-claim-phase-hint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(paths_from_csv(args.tasks))
    patch_rows, patch_stats = build_patch_rows(args, tasks)

    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.6_claim_phase_failure_patch",
        "rollouts": [str(path) for path in paths_from_csv(args.rollouts)],
        "tasks": [str(path) for path in paths_from_csv(args.tasks)],
        "base_sft_dir": str(args.base_sft_dir),
        "output_dir": str(args.output_dir),
        "replay_train_rows": args.replay_train_rows,
        "replay_val_rows": args.replay_val_rows,
        "replay_test_rows": args.replay_test_rows,
        "patch_oversample": args.patch_oversample,
        "max_patches_per_task": args.max_patches_per_task,
        "claim_chunk_size": args.claim_chunk_size,
        "region_selection_hint": args.region_selection_hint,
        "strict_claim_phase_hint": args.strict_claim_phase_hint,
        "seed": args.seed,
        "patch_stats": patch_stats,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        replay_limit = {
            "train": args.replay_train_rows,
            "val": args.replay_val_rows,
            "test": args.replay_test_rows,
        }[split]
        replay_rows = sample_replay(args.base_sft_dir / f"{split}.jsonl", replay_limit, args.seed)
        rows = list(replay_rows)
        if split == "train":
            for row in patch_rows:
                for copy_index in range(max(1, args.patch_oversample)):
                    copied = copy.deepcopy(row)
                    copied["patch_source"]["copy_index"] = copy_index
                    copied["patch_source"]["copies"] = max(1, args.patch_oversample)
                    rows.append(copied)
        random.shuffle(rows)
        out_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        manifest["splits"][split] = {
            "file": str(out_path),
            "rows_written": len(rows),
            "replay_rows": len(replay_rows),
            "patch_rows_before_oversample": len(patch_rows) if split == "train" else 0,
            "action_counts": dict(sorted(action_counts(rows).items())),
        }
        print(f"[{split}] wrote {len(rows)} rows -> {out_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "构建报告.md", manifest)
    print(f"manifest -> {manifest_path}")
    return 0


def build_patch_rows(args: argparse.Namespace, tasks: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    per_task: Counter[str] = Counter()
    prompt_config = PromptConfig(
        max_history_actions=8,
        max_tool_results=6,
        max_evidence_per_result=3,
        snippet_chars=180,
        max_text_chars=24000,
        head_text_chars=5000,
        coordinate_info=True,
        tool_schema="chunked_claim",
        compact_claim_state=False,
        region_selection_hint=args.region_selection_hint,
        strict_claim_phase_hint=args.strict_claim_phase_hint,
    )

    for rollout_path in paths_from_csv(args.rollouts):
        if not rollout_path.exists():
            stats["missing_rollout_files"] += 1
            continue
        for record in read_jsonl(rollout_path):
            task_id = str(record.get("task_id", ""))
            task = tasks.get(task_id)
            if not task:
                stats["missing_tasks"] += 1
                continue
            state = RuntimeState.from_task(task)
            for step in record.get("steps") or []:
                phase = str((step.get("tool_mask") or {}).get("phase", ""))
                is_claim_phase = phase in {"claim_writing", "claim_continuation"}
                pred_action = step.get("parsed_action")
                pred_name = str(pred_action.get("action", "")) if isinstance(pred_action, dict) else "invalid"
                is_bad = (
                    is_claim_phase
                    and (
                        bool(step.get("mask_violation"))
                        or pred_name in {"open_evidence", "retrieve_evidence", "select_evidence", "invalid"}
                    )
                )
                if is_bad:
                    stats["claim_phase_failures_seen"] += 1
                    if per_task[task_id] < max(1, args.max_patches_per_task):
                        target_action = build_target_action(task, state.draft_claims, args.claim_chunk_size)
                        obs = state.to_observation(step.get("tool_mask") or {}, step.get("available_actions") or [])
                        messages = build_messages_from_observation(
                            obs,
                            prompt_config,
                            include_assistant_action=target_action,
                        )
                        row = make_sft_row(
                            task=task,
                            state=state,
                            step=step,
                            messages=messages,
                            action=target_action,
                            rollout_path=rollout_path,
                        )
                        rows.append(row)
                        per_task[task_id] += 1
                        stats[f"target::{target_action.get('action')}"] += 1
                    else:
                        stats["skipped_max_patches_per_task"] += 1
                state.apply_step(step)

    stats["patch_rows"] = len(rows)
    stats["tasks_with_patches"] = len(per_task)
    return rows, dict(sorted(stats.items()))


def build_target_action(task: dict[str, Any], draft_claims: list[dict[str, Any]], chunk_size: int) -> dict[str, Any]:
    current_state = claim_state(draft_claims)
    remaining = list(current_state.get("remaining_fields") or [])
    if not remaining:
        return {"action": "finish", "status": "done"}
    gold_by_field = {
        str(item.get("field")): item
        for item in (task.get("gold") or {}).get("claims", [])
        if isinstance(item, dict) and item.get("field")
    }
    claims: list[dict[str, Any]] = []
    abstains: list[dict[str, Any]] = []
    for field in remaining[: max(1, chunk_size)]:
        gold = gold_by_field.get(str(field))
        if gold and not gold.get("abstain") and "value" in gold:
            evidence_ids = gold.get("evidence_ids") or gold.get("candidate_evidence_ids") or (task.get("gold") or {}).get("evidence_ids") or []
            claims.append(
                {
                    "field": field,
                    "value": gold.get("value"),
                    "evidence_ids": list(evidence_ids),
                    "visual_bbox": gold.get("visual_bbox"),
                    "confidence": float(gold.get("confidence", 0.8) or 0.8),
                }
            )
        else:
            reason = str(gold.get("reason") if gold else "当前证据不足，暂不写入该字段")
            abstains.append({"field": field, "reason": reason})
    return {"action": "write_claims_chunk", "claims": claims, "abstains": abstains}


def make_sft_row(
    *,
    task: dict[str, Any],
    state: "RuntimeState",
    step: dict[str, Any],
    messages: list[dict[str, Any]],
    action: dict[str, Any],
    rollout_path: Path,
) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "source_task_id": task.get("task_id"),
        "split": "train",
        "step": int(step.get("step", 0) or 0),
        "variant": 0,
        "label_source": "rollout_claim_phase_failure_patch",
        "tool_schema_version": "v0.6_claim_phase_failure_patch",
        "images": [item.get("path") for item in state.images() if item.get("path")],
        "history": list(state.history),
        "tool_results": list(state.tool_results),
        "draft_claims": list(state.draft_claims),
        "claim_state": claim_state(state.draft_claims),
        "selected_evidence_ids": list(state.selected_evidence_ids),
        "messages": messages,
        "prompt_text": first_user_text(messages),
        "action": action,
        "patch_source": {
            "kind": "v0_6_claim_phase_failure_state",
            "rollout": str(rollout_path),
            "failed_step": int(step.get("step", 0) or 0),
            "failed_action": step.get("parsed_action"),
            "mask_phase": (step.get("tool_mask") or {}).get("phase"),
            "mask_reason": (step.get("tool_mask") or {}).get("reason"),
            "copy_index": 0,
            "copies": 1,
        },
    }


class RuntimeState:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.history: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.draft_claims: list[dict[str, Any]] = []
        self.selected_evidence_ids: list[str] = []
        self.last_crop: str | None = None

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> "RuntimeState":
        return cls(task)

    def images(self) -> list[dict[str, str]]:
        images = [{"role": "page_image", "path": str(self.task.get("page_image", ""))}]
        if self.last_crop:
            images.append({"role": "last_crop", "path": self.last_crop})
        return images

    def to_observation(self, tool_mask: dict[str, Any], available_actions: list[str]) -> dict[str, Any]:
        return {
            "task_id": self.task.get("task_id"),
            "goal": self.task.get("goal"),
            "source_file": self.task.get("source_file"),
            "page": self.task.get("page"),
            "images": self.images(),
            "history": list(self.history),
            "tool_results": list(self.tool_results),
            "draft_claims": list(self.draft_claims),
            "claim_state": claim_state(self.draft_claims),
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "available_actions": list(available_actions),
            "tool_mask": dict(tool_mask),
        }

    def apply_step(self, step: dict[str, Any]) -> None:
        action = step.get("parsed_action")
        if isinstance(action, dict):
            self.history.append(action)
        else:
            self.history.append({"raw": step.get("raw_text", "")})
        result = step.get("result") or {}
        self.tool_results.append(result)
        if result.get("tool") in {"crop_region", "crop_image"} and result.get("crop_path"):
            self.last_crop = str(result.get("crop_path"))
        if result.get("tool") == "select_evidence":
            for evidence_id in result.get("selected_evidence_ids") or []:
                evidence_id = str(evidence_id)
                if evidence_id not in self.selected_evidence_ids:
                    self.selected_evidence_ids.append(evidence_id)
        if result.get("tool") in {"write_claims_chunk", "write_claims_batch"}:
            self.draft_claims = apply_claim_write(
                self.draft_claims,
                claims=result.get("claims") or [],
                abstains=result.get("abstains") or [],
            )
        elif result.get("tool") == "write_claim" and isinstance(result.get("claim"), dict):
            self.draft_claims = apply_claim_write(self.draft_claims, claims=[result["claim"]], abstains=[])
        elif result.get("tool") == "abstain_claim" and isinstance(result.get("claim"), dict):
            self.draft_claims = apply_claim_write(self.draft_claims, claims=[], abstains=[result["claim"]])


def sample_replay(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit <= 0 or limit >= len(rows):
        return rows
    name_offset = sum((index + 1) * ord(char) for index, char in enumerate(path.name)) % 10000
    rng = random.Random(seed + name_offset)
    rows = list(rows)
    rng.shuffle(rows)
    return rows[:limit]


def load_tasks(paths: list[Path]) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            task_id = str(row.get("task_id", ""))
            if task_id:
                tasks[task_id] = row
    return tasks


def action_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        action = row.get("action")
        name = str(action.get("action", "")) if isinstance(action, dict) else ""
        counts[name] += 1
    return counts


def paths_from_csv(value: str) -> list[Path]:
    return [Path(item.strip()) for item in str(value or "").split(",") if item.strip()]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
    return ""


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.6 Claim Phase Failure Patch SFT 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- base_sft_dir: `{manifest['base_sft_dir']}`",
        f"- rollouts: `{manifest['rollouts']}`",
        "",
        "## Patch Stats",
        "",
    ]
    for key, value in manifest.get("patch_stats", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Splits", ""])
    for split, stats in manifest.get("splits", {}).items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- rows_written: {stats['rows_written']}",
                f"- replay_rows: {stats['replay_rows']}",
                f"- patch_rows_before_oversample: {stats['patch_rows_before_oversample']}",
                f"- action_counts: `{json.dumps(stats['action_counts'], ensure_ascii=False)}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
