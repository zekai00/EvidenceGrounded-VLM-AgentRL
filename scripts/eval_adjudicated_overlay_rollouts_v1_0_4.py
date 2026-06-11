#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.verifier import EvidenceVerifier  # noqa: E402


DEFAULT_TASKS = "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615/val_tasks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 clean/LLM overlay verifier 复算已有 rollout 指标。")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--clean-index", required=True)
    parser.add_argument("--overlay-index", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-steps", type=int, default=14)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"adjudicated_overlay_rollout_replay_v1_0_4_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(Path(args.tasks))
    rollouts = list(iter_rollouts(args.rollouts))

    clean_verifier = EvidenceVerifier(args.clean_index)
    overlay_verifier = EvidenceVerifier(args.overlay_index)
    rows: list[dict[str, Any]] = []
    for rollout in rollouts:
        task = tasks.get(str(rollout.get("task_id")))
        if not task:
            continue
        history, tool_results, draft_claims = rollout_parts(rollout)
        clean_metrics = clean_verifier.trajectory_metrics(task, history, tool_results, draft_claims, max_steps=args.max_steps)
        overlay_metrics = overlay_verifier.trajectory_metrics(task, history, tool_results, draft_claims, max_steps=args.max_steps)
        rows.append(
            {
                "task_id": rollout.get("task_id"),
                "source_file": rollout.get("source_file"),
                "page": rollout.get("page"),
                "clean_metrics": clean_metrics,
                "overlay_metrics": overlay_metrics,
                "delta": metric_delta(clean_metrics, overlay_metrics),
            }
        )

    manifest = {
        "created_at": now(),
        "tasks": args.tasks,
        "rollouts": args.rollouts,
        "clean_index": args.clean_index,
        "overlay_index": args.overlay_index,
        "evaluated_rollouts": len(rows),
        "clean_summary": summarize(rows, "clean_metrics"),
        "overlay_summary": summarize(rows, "overlay_metrics"),
        "delta_summary": summarize_delta(rows),
        "artifacts": {
            "per_task": str(output_dir / "per_task_metrics.jsonl"),
            "report": str(output_dir / "评估报告.md"),
        },
    }
    write_jsonl(output_dir / "per_task_metrics.jsonl", rows)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "评估报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def rollout_parts(rollout: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for step in rollout.get("steps") or []:
        action = step.get("parsed_action") or step.get("model_parsed_action")
        if isinstance(action, dict):
            history.append(action)
            tool_results.append(step.get("result") or {})
    draft_claims = rollout.get("final_claims") or []
    return history, tool_results, draft_claims


def metric_delta(clean: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "final_reward",
        "claim_supported_rate",
        "supported_claim_count",
        "unsupported_claim_count",
        "core_supported_count",
        "core_unsupported_count",
        "evidence_recall",
        "invalid_step_rate",
    ]
    out: dict[str, Any] = {}
    for key in keys:
        try:
            out[key] = round(float(overlay.get(key, 0.0)) - float(clean.get(key, 0.0)), 6)
        except Exception:
            continue
    return out


def summarize(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not rows:
        return {}
    metrics = [row[key] for row in rows]
    action_counts: Counter[str] = Counter()
    for metric in metrics:
        action_counts.update(metric.get("action_counts") or {})
    return {
        "trajectory_success_rate": mean_bool(metrics, "trajectory_success"),
        "finish_rate": mean_bool(metrics, "finish"),
        "crop_success_rate": mean_bool(metrics, "crop_success"),
        "mean_final_reward": mean_float(metrics, "final_reward"),
        "mean_claim_supported_rate": mean_float(metrics, "claim_supported_rate"),
        "mean_supported_claim_count": mean_float(metrics, "supported_claim_count"),
        "mean_unsupported_claim_count": mean_float(metrics, "unsupported_claim_count"),
        "mean_core_supported_count": mean_float(metrics, "core_supported_count"),
        "mean_core_unsupported_count": mean_float(metrics, "core_unsupported_count"),
        "mean_evidence_recall": mean_float(metrics, "evidence_recall"),
        "mean_invalid_step_rate": mean_float(metrics, "invalid_step_rate"),
        "action_counts": dict(action_counts),
    }


def summarize_delta(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    deltas = [row["delta"] for row in rows]
    return {
        "mean_final_reward_delta": mean_float(deltas, "final_reward"),
        "mean_claim_supported_rate_delta": mean_float(deltas, "claim_supported_rate"),
        "mean_supported_claim_count_delta": mean_float(deltas, "supported_claim_count"),
        "mean_unsupported_claim_count_delta": mean_float(deltas, "unsupported_claim_count"),
        "tasks_with_reward_drop": sum(float(row["delta"].get("final_reward", 0.0)) < 0 for row in rows),
        "tasks_with_supported_claim_drop": sum(float(row["delta"].get("supported_claim_count", 0.0)) < 0 for row in rows),
    }


def mean_float(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(float(row.get(key, 0.0) or 0.0) for row in rows) / max(1, len(rows)), 6)


def mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum(bool(row.get(key)) for row in rows) / max(1, len(rows)), 6)


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("task_id")): row for row in iter_jsonl(path)}


def iter_rollouts(paths: list[str]) -> Iterable[dict[str, Any]]:
    for path in paths:
        yield from iter_jsonl(Path(path))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    clean = manifest["clean_summary"]
    overlay = manifest["overlay_summary"]
    delta = manifest["delta_summary"]
    lines = [
        "# v1.0.4 Adjudicated Overlay Rollout Replay 评估报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- tasks：`{manifest['tasks']}`",
        f"- clean index：`{manifest['clean_index']}`",
        f"- overlay index：`{manifest['overlay_index']}`",
        f"- evaluated rollouts：{manifest['evaluated_rollouts']}",
        "",
        "## 指标对比",
        "",
        "| metric | clean | overlay | delta |",
        "|---|---:|---:|---:|",
    ]
    for key in [
        "mean_final_reward",
        "mean_claim_supported_rate",
        "mean_supported_claim_count",
        "mean_unsupported_claim_count",
        "mean_core_supported_count",
        "mean_core_unsupported_count",
        "mean_evidence_recall",
        "mean_invalid_step_rate",
    ]:
        delta_key = key.replace("mean_", "")
        lines.append(f"| {key} | {clean.get(key)} | {overlay.get(key)} | {delta.get('mean_' + delta_key + '_delta', '')} |")
    lines.extend(
        [
            "",
            "## delta summary",
            "",
            f"- tasks_with_reward_drop：{delta.get('tasks_with_reward_drop')}",
            f"- tasks_with_supported_claim_drop：{delta.get('tasks_with_supported_claim_drop')}",
            "",
            "## 产物",
            "",
        ]
    )
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    raise SystemExit(main())
