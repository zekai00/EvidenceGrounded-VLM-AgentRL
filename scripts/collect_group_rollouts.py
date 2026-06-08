#!/usr/bin/env python3
"""Collect K executable trajectories per task and compute GRPO-style advantages."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env import EvidenceAgentEnv, QwenVLSftPolicy  # noqa: E402
from evidence_agent_env.data import read_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--samples-per-task", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument(
        "--advantage-reward",
        choices=["final_reward", "total_reward", "shaped"],
        default="final_reward",
        help="Reward used for within-group advantage normalization.",
    )
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--image-max-pixels", type=int, default=131072)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--tool-schema",
        choices=["highlighted_direct", "region", "evidence_select", "chunked_claim", "inspect_crop"],
        default="evidence_select",
    )
    parser.add_argument("--max-history-actions", type=int, default=8)
    parser.add_argument("--max-tool-results", type=int, default=6)
    parser.add_argument("--max-evidence-per-result", type=int, default=3)
    parser.add_argument("--snippet-chars", type=int, default=180)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--head-text-chars", type=int, default=3000)
    parser.add_argument("--coordinate-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-gold-regions", action="store_true")
    parser.add_argument("--phase-aware-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-tool-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--region-selection-hint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-claim-phase-hint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--compact-claim-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to true for chunked_claim.",
    )
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--print-steps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)

    tasks = select_tasks(read_jsonl(args.tasks), args)
    prompt_config = PromptConfig(
        max_history_actions=args.max_history_actions,
        max_tool_results=args.max_tool_results,
        max_evidence_per_result=args.max_evidence_per_result,
        snippet_chars=args.snippet_chars,
        max_text_chars=args.max_text_chars,
        head_text_chars=args.head_text_chars,
        coordinate_info=args.coordinate_info,
        tool_schema=args.tool_schema,
        compact_claim_state=args.compact_claim_state
        if args.compact_claim_state is not None
        else args.tool_schema in {"chunked_claim", "inspect_crop"},
        region_selection_hint=args.region_selection_hint,
        strict_claim_phase_hint=args.strict_claim_phase_hint,
    )
    policy = QwenVLSftPolicy(
        args.model,
        args.adapter,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        image_max_pixels=args.image_max_pixels,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
        prompt_config=prompt_config,
    )
    env = EvidenceAgentEnv(
        args.tasks,
        args.evidence_index,
        output_dir,
        max_steps=args.max_steps,
        include_gold_regions=args.include_gold_regions,
        phase_aware_mask=args.phase_aware_mask,
        enforce_tool_mask=args.enforce_tool_mask,
        tool_schema=args.tool_schema,
    )

    groups: list[dict[str, Any]] = []
    groups_path = output_dir / "rollout_groups.jsonl"
    with groups_path.open("w", encoding="utf-8") as f:
        for task_index, task in enumerate(tasks):
            group = collect_group(env, policy, task, output_dir, task_index, args)
            groups.append(group)
            f.write(json.dumps(group, ensure_ascii=False) + "\n")
            print(json.dumps(progress_record(groups, task_index + 1, len(tasks)), ensure_ascii=False), flush=True)

    summary = build_summary(args, policy, groups, groups_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "group_rollout_report.md", summary, groups)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def select_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task_id:
        wanted = set(args.task_id)
        selected = [task for task in tasks if str(task.get("task_id")) in wanted]
    else:
        end = len(tasks) if args.max_tasks <= 0 else min(len(tasks), args.start_index + args.max_tasks)
        selected = tasks[args.start_index : end]
    if not selected:
        raise ValueError("no tasks selected")
    return selected


def collect_group(
    env: EvidenceAgentEnv,
    policy: QwenVLSftPolicy,
    task: dict[str, Any],
    output_dir: Path,
    task_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    trajectories = []
    rewards = []
    for sample_index in range(args.samples_per_task):
        record = run_one(env, policy, task, output_dir, task_index, sample_index, args)
        trajectories.append(record)
        rewards.append(sample_reward(record, args.advantage_reward))
    advantages = normalized_advantages(rewards)
    for record, reward, advantage in zip(trajectories, rewards, advantages):
        record["group_reward"] = reward
        record["advantage"] = advantage
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "samples": trajectories,
        "reward_mean": sum(rewards) / max(1, len(rewards)),
        "reward_std": std(rewards),
        "best_reward": max(rewards) if rewards else 0.0,
        "best_sample_index": int(max(range(len(rewards)), key=lambda idx: rewards[idx])) if rewards else None,
    }


def run_one(
    env: EvidenceAgentEnv,
    policy: QwenVLSftPolicy,
    task: dict[str, Any],
    output_dir: Path,
    task_index: int,
    sample_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    obs = env.reset(task_id=str(task.get("task_id")))
    steps: list[dict[str, Any]] = []
    terminated = False
    for step_index in range(args.max_steps):
        tool_mask = obs.get("tool_mask") or {}
        available_actions = list(obs.get("available_actions") or [])
        prediction = policy.act(obs)
        pred_action_name = (
            str(prediction["action"].get("action", ""))
            if isinstance(prediction.get("action"), dict)
            else "invalid"
        )
        mask_violation = bool(available_actions and pred_action_name not in set(available_actions))
        action = prediction["action"] if prediction["action"] is not None else prediction["raw_text"]
        obs, reward, terminated, info = env.step(action)
        result = info.get("result") or {}
        steps.append(
            {
                "step": step_index,
                "raw_text": prediction["raw_text"],
                "parsed_action": prediction["action"],
                "available_actions": available_actions,
                "tool_mask": tool_mask,
                "mask_violation": mask_violation,
                "reward": reward,
                "result": result,
                "total_reward": info.get("total_reward"),
                "terminated": terminated,
            }
        )
        if args.print_steps:
            action_name = prediction["action"].get("action", "invalid") if isinstance(prediction["action"], dict) else "invalid"
            print(
                json.dumps(
                    {
                        "time": now(),
                        "task_id": task.get("task_id"),
                        "sample": sample_index,
                        "step": step_index,
                        "action": action_name,
                        "mask_phase": tool_mask.get("phase"),
                        "mask_violation": mask_violation,
                        "reward": reward,
                        "total_reward": info.get("total_reward"),
                        "terminated": terminated,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if terminated:
            break
    trajectory_path = output_dir / "trajectories" / f"{task_index:04d}_{sample_index:02d}_{task.get('task_id')}.json"
    env.dump_trajectory(trajectory_path)
    metrics = env.trajectory_metrics()
    return {
        "sample_index": sample_index,
        "num_steps": len(steps),
        "terminated": terminated,
        "total_reward": env.total_reward,
        "trajectory_metrics": metrics,
        "steps": steps,
        "trajectory": str(trajectory_path),
    }


def normalized_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    mean = sum(rewards) / len(rewards)
    sigma = std(rewards)
    if sigma < 1e-6:
        return [0.0 for _ in rewards]
    return [round((reward - mean) / sigma, 6) for reward in rewards]


def std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def sample_reward(sample: dict[str, Any], mode: str) -> float:
    metrics = sample.get("trajectory_metrics") or {}
    if mode == "total_reward":
        return float(sample.get("total_reward", 0.0) or 0.0)
    if mode == "shaped":
        final_reward = float(metrics.get("final_reward", 0.0) or 0.0)
        finish = float(bool(metrics.get("finish")))
        crop = float(bool(metrics.get("crop_success")))
        success = float(bool(metrics.get("trajectory_success")))
        invalid = float(metrics.get("invalid_step_rate", 0.0) or 0.0)
        steps = int(metrics.get("steps", sample.get("num_steps", 0)) or 0)
        step_penalty = max(0, steps - 10) * 0.02
        return final_reward + 0.20 * finish + 0.15 * crop + 0.20 * success - 0.40 * invalid - step_penalty
    return float(metrics.get("final_reward", 0.0) or 0.0)


def build_summary(
    args: argparse.Namespace,
    policy: QwenVLSftPolicy,
    groups: list[dict[str, Any]],
    groups_path: Path,
) -> dict[str, Any]:
    samples = [sample for group in groups for sample in group.get("samples") or []]
    n = max(1, len(samples))
    action_counts: Counter[str] = Counter()
    for sample in samples:
        for step in sample.get("steps") or []:
            action = step.get("parsed_action")
            action_counts[str(action.get("action", "invalid")) if isinstance(action, dict) else "invalid"] += 1
    return {
        "created_at": now(),
        "tasks_path": args.tasks,
        "evidence_index": args.evidence_index,
        "output_dir": args.output_dir,
        "tasks_used": len(groups),
        "samples_per_task": args.samples_per_task,
        "advantage_reward": args.advantage_reward,
        "samples_total": len(samples),
        "policy": policy.metadata(),
        "metrics": {
            "trajectory_success_rate": sum(bool(s["trajectory_metrics"].get("trajectory_success")) for s in samples) / n,
            "finish_rate": sum(bool(s["trajectory_metrics"].get("finish")) for s in samples) / n,
            "crop_success_rate": sum(bool(s["trajectory_metrics"].get("crop_success")) for s in samples) / n,
            "mean_final_reward": sum(float(s["trajectory_metrics"].get("final_reward", 0.0)) for s in samples) / n,
            "mean_evidence_recall": sum(float(s["trajectory_metrics"].get("evidence_recall", 0.0)) for s in samples) / n,
            "mean_claim_supported_rate": sum(float(s["trajectory_metrics"].get("claim_supported_rate", 0.0)) for s in samples) / n,
            "mean_invalid_step_rate": sum(float(s["trajectory_metrics"].get("invalid_step_rate", 0.0)) for s in samples) / n,
            "mean_group_reward_std": sum(float(group.get("reward_std", 0.0)) for group in groups) / max(1, len(groups)),
            "nonzero_advantage_group_rate": sum(float(group.get("reward_std", 0.0)) > 1e-6 for group in groups) / max(1, len(groups)),
            "action_counts": dict(action_counts),
        },
        "rollout_groups": str(groups_path),
    }


def progress_record(groups: list[dict[str, Any]], done: int, total: int) -> dict[str, Any]:
    samples = [sample for group in groups for sample in group.get("samples") or []]
    n = max(1, len(samples))
    return {
        "time": now(),
        "done": done,
        "total": total,
        "samples": len(samples),
        "mean_final_reward": sum(float(s["trajectory_metrics"].get("final_reward", 0.0)) for s in samples) / n,
        "trajectory_success_rate": sum(bool(s["trajectory_metrics"].get("trajectory_success")) for s in samples) / n,
    }


def write_report(path: Path, summary: dict[str, Any], groups: list[dict[str, Any]]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# EvidenceGrounded Group Rollout Report",
        "",
        f"- created_at: {summary['created_at']}",
        f"- tasks_used: {summary['tasks_used']}",
        f"- samples_per_task: {summary['samples_per_task']}",
        f"- advantage_reward: {summary.get('advantage_reward', 'final_reward')}",
        f"- samples_total: {summary['samples_total']}",
        f"- model: {summary['policy']['model']}",
        f"- adapter: {summary['policy']['adapter']}",
        "",
        "## Metrics",
        "",
        f"- trajectory_success_rate: {metrics['trajectory_success_rate']:.3f}",
        f"- finish_rate: {metrics['finish_rate']:.3f}",
        f"- crop_success_rate: {metrics['crop_success_rate']:.3f}",
        f"- mean_final_reward: {metrics['mean_final_reward']:.3f}",
        f"- mean_evidence_recall: {metrics['mean_evidence_recall']:.3f}",
        f"- mean_claim_supported_rate: {metrics['mean_claim_supported_rate']:.3f}",
        f"- mean_invalid_step_rate: {metrics['mean_invalid_step_rate']:.3f}",
        f"- mean_group_reward_std: {metrics['mean_group_reward_std']:.3f}",
        f"- nonzero_advantage_group_rate: {metrics['nonzero_advantage_group_rate']:.3f}",
        f"- action_counts: `{json.dumps(metrics['action_counts'], ensure_ascii=False)}`",
        "",
        "## Groups",
        "",
    ]
    for group in groups:
        lines.extend(
            [
                f"### {group.get('task_id')}",
                "",
                f"- reward_mean: {float(group.get('reward_mean', 0.0)):.3f}",
                f"- reward_std: {float(group.get('reward_std', 0.0)):.3f}",
                f"- best_reward: {float(group.get('best_reward', 0.0)):.3f}",
            ]
        )
        for sample in group.get("samples") or []:
            metrics_one = sample.get("trajectory_metrics", {})
            lines.append(
                f"- sample {sample.get('sample_index')}: group_reward={sample.get('group_reward')} final_reward={metrics_one.get('final_reward')} advantage={sample.get('advantage')} success={metrics_one.get('trajectory_success')} steps={sample.get('num_steps')}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
