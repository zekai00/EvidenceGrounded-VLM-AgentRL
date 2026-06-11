#!/usr/bin/env python3
"""Build a no-select failure-patch SFT dataset from executable rollouts.

The patch rows are not model outputs. They are oracle no-select SFT rows for
tasks that failed or scored low in executable rollout, with emphasis on
evidence opening, retrieval, chunked claim writing, and finish readiness.
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
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DEFAULT_ROLLOUTS = [
    "outputs/no_select_executable_val32_balanced_final_20260608_1230_part0/rollouts.jsonl",
    "outputs/no_select_executable_val32_balanced_final_20260608_1230_part1/rollouts.jsonl",
]
PATCH_ACTIONS = {"open_evidence", "retrieve_evidence", "write_claims_chunk", "finish"}
REPLAY_ACTIONS = {"inspect_page", "crop_target", "open_evidence", "retrieve_evidence", "write_claims_chunk", "finish"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--rollouts", nargs="*", default=DEFAULT_ROLLOUTS)
    parser.add_argument("--claim-supported-threshold", type=float, default=0.45)
    parser.add_argument("--evidence-recall-threshold", type=float, default=0.45)
    parser.add_argument("--include-task-id", action="append", default=[])
    parser.add_argument("--patch-actions", default=",".join(sorted(PATCH_ACTIONS)))
    parser.add_argument("--replay-ratio", type=float, default=0.7)
    parser.add_argument("--max-replay-rows", type=int, default=700)
    parser.add_argument("--val-patch-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / default_output_name()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    (output_dir / "sft").mkdir(parents=True)

    rollout_records = load_rollout_records([Path(item) for item in args.rollouts])
    selected_tasks, selection_reasons = select_patch_tasks(rollout_records, args)
    patch_actions = {item.strip() for item in args.patch_actions.split(",") if item.strip()}

    source_rows_by_split = {
        split: read_jsonl(source_root / "sft" / f"{split}.jsonl")
        for split in ["train", "val", "test"]
        if (source_root / "sft" / f"{split}.jsonl").exists()
    }
    patch_rows = collect_patch_rows(source_rows_by_split, selected_tasks, selection_reasons, patch_actions)
    if not patch_rows:
        raise RuntimeError("no patch rows selected")

    rng = random.Random(args.seed)
    rng.shuffle(patch_rows)
    val_count = max(1, int(round(len(patch_rows) * args.val_patch_fraction))) if len(patch_rows) > 5 else 0
    patch_val = patch_rows[:val_count]
    patch_train = patch_rows[val_count:]

    replay_rows = collect_replay_rows(
        source_rows_by_split.get("train", []),
        exclude_task_ids=selected_tasks,
        max_rows=args.max_replay_rows,
        replay_ratio=args.replay_ratio,
        patch_train_count=len(patch_train),
        rng=rng,
    )
    mixed_train = replay_rows + patch_train
    rng.shuffle(mixed_train)

    write_jsonl(output_dir / "failure_patch_rows.jsonl", patch_rows)
    write_jsonl(output_dir / "sft" / "patch_train.jsonl", patch_train)
    write_jsonl(output_dir / "sft" / "patch_val.jsonl", patch_val)
    write_jsonl(output_dir / "sft" / "train.jsonl", mixed_train)
    write_jsonl(output_dir / "sft" / "val.jsonl", patch_val if patch_val else patch_train[: min(32, len(patch_train))])

    manifest = {
        "created_at": now(),
        "dataset_version": "v1.0.3_no_select_failure_patch_sft",
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "rollouts": args.rollouts,
        "thresholds": {
            "claim_supported_rate": args.claim_supported_threshold,
            "evidence_recall": args.evidence_recall_threshold,
        },
        "selected_task_count": len(selected_tasks),
        "selected_tasks": sorted(selected_tasks),
        "selection_reasons": selection_reasons,
        "patch_actions": sorted(patch_actions),
        "counts": {
            "patch_rows": len(patch_rows),
            "patch_train_rows": len(patch_train),
            "patch_val_rows": len(patch_val),
            "replay_rows": len(replay_rows),
            "mixed_train_rows": len(mixed_train),
        },
        "action_counts": {
            "patch": dict(action_counter(patch_rows)),
            "patch_train": dict(action_counter(patch_train)),
            "patch_val": dict(action_counter(patch_val)),
            "replay": dict(action_counter(replay_rows)),
            "mixed_train": dict(action_counter(mixed_train)),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def load_rollout_records(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    row["_rollout_path"] = str(path)
                    rows.append(row)
    return rows


def select_patch_tasks(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[set[str], dict[str, list[str]]]:
    selected = {str(item) for item in args.include_task_id}
    reasons: dict[str, list[str]] = defaultdict(list)
    for task_id in selected:
        reasons[task_id].append("manual_include")
    for record in records:
        task_id = str(record.get("task_id") or "")
        metrics = record.get("trajectory_metrics") or {}
        if not task_id:
            continue
        if not bool(metrics.get("trajectory_success")):
            selected.add(task_id)
            reasons[task_id].append("trajectory_failed")
        if float(metrics.get("claim_supported_rate") or 0.0) <= args.claim_supported_threshold:
            selected.add(task_id)
            reasons[task_id].append("low_claim_supported")
        if float(metrics.get("evidence_recall") or 0.0) <= args.evidence_recall_threshold:
            selected.add(task_id)
            reasons[task_id].append("low_evidence_recall")
        if int(metrics.get("invalid_steps") or 0) > 0:
            selected.add(task_id)
            reasons[task_id].append("invalid_steps")
    return selected, {task_id: sorted(set(items)) for task_id, items in reasons.items()}


def collect_patch_rows(
    source_rows_by_split: dict[str, list[dict[str, Any]]],
    selected_tasks: set[str],
    selection_reasons: dict[str, list[str]],
    patch_actions: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for split, source_rows in source_rows_by_split.items():
        for row in source_rows:
            task_id = str(row.get("task_id") or "")
            action_name = action_type(row)
            if task_id not in selected_tasks or action_name not in patch_actions:
                continue
            key = (task_id, int(row.get("step") or -1), action_name)
            if key in seen:
                continue
            seen.add(key)
            patch = dict(row)
            patch["split"] = "train"
            patch["source_split"] = split
            patch["label_source"] = "v1_0_3_no_select_failure_patch_oracle"
            patch["tool_schema_version"] = "v1.0.3_no_select_failure_patch"
            patch["patch_meta"] = {
                "patch_type": patch_type_for_action(action_name),
                "selection_reasons": selection_reasons.get(task_id, []),
                "source_split": split,
            }
            rows.append(patch)
    return rows


def collect_replay_rows(
    source_rows: list[dict[str, Any]],
    *,
    exclude_task_ids: set[str],
    max_rows: int,
    replay_ratio: float,
    patch_train_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if patch_train_count <= 0 or replay_ratio <= 0.0:
        return []
    target = int(round(patch_train_count * replay_ratio / max(1e-6, 1.0 - replay_ratio)))
    target = min(max_rows, max(0, target))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        if str(row.get("task_id") or "") in exclude_task_ids:
            continue
        action_name = action_type(row)
        if action_name not in REPLAY_ACTIONS:
            continue
        replay = dict(row)
        replay["label_source"] = "v1_0_3_no_select_replay"
        replay["tool_schema_version"] = "v1.0.3_no_select_replay"
        buckets[action_name].append(replay)
    for items in buckets.values():
        rng.shuffle(items)
    selected: list[dict[str, Any]] = []
    action_names = sorted(buckets)
    cursor = 0
    while len(selected) < target and any(buckets.values()):
        name = action_names[cursor % len(action_names)]
        cursor += 1
        if buckets[name]:
            selected.append(buckets[name].pop())
    return selected


def action_type(row: dict[str, Any]) -> str:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    return str(action.get("action") or "")


def patch_type_for_action(action_name: str) -> str:
    if action_name == "open_evidence":
        return "evidence_opening"
    if action_name == "retrieve_evidence":
        return "evidence_recall"
    if action_name == "write_claims_chunk":
        return "claim_grounding_or_abstain"
    if action_name == "finish":
        return "finish_ready"
    return "tool_protocol"


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(action_type(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    reason_names = {
        "invalid_steps": "存在无效步骤",
        "low_claim_supported": "claim 支持率偏低",
        "low_evidence_recall": "证据召回偏低",
        "trajectory_failed": "轨迹未成功",
    }
    lines = [
        "# v1.0.3 No-Select Failure-Patch SFT 构建报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- 源数据目录：`{manifest['source_root']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        f"- 选中任务数：{manifest['selected_task_count']}",
        "",
        "## 数据量统计",
        "",
    ]
    for key, value in manifest["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Patch 动作分布", ""])
    for key, value in manifest["action_counts"]["patch"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 选中任务", ""])
    for task_id in manifest["selected_tasks"]:
        reasons = "，".join(
            reason_names.get(reason, reason)
            for reason in manifest["selection_reasons"].get(task_id, [])
        )
        lines.append(f"- `{task_id}`: {reasons}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_name() -> str:
    return "v1_0_3_no_select_failure_patch_sft_" + datetime.now().strftime("%Y%m%d_%H%M")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
