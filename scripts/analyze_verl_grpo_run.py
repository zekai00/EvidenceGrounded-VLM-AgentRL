#!/usr/bin/env python3
"""Summarize verl GRPO run logs, agent-loop debug traces, and training curves."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
METRIC_RE = re.compile(r"([A-Za-z0-9_./@-]+):(-?\d+(?:\.\d+)?(?:e[-+]?\d+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tmp-dir", type=Path, required=True)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title", default="GRPO run report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = args.log_file or args.run_dir.with_suffix(".log")
    log_rows = parse_log_metrics(log_file)
    debug_groups = group_debug_records(args.run_dir, args.tmp_dir)
    summary = summarize(args.run_dir, args.tmp_dir, log_file, log_rows, debug_groups, args.title)

    write_json(args.out_dir / "log_metrics.json", log_rows)
    write_json(args.out_dir / "debug_group_summary.json", debug_groups)
    write_json(args.out_dir / "summary.json", summary)
    plot_curves(args.out_dir, log_rows, debug_groups)
    write_report(args.out_dir / "report.md", summary)
    print(json.dumps({"out_dir": str(args.out_dir), "summary": summary["headline"]}, ensure_ascii=False, indent=2))
    return 0


def parse_log_metrics(log_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not log_file.exists():
        return rows
    for raw in log_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = ANSI_RE.sub("", raw)
        if "training/global_step:" not in line or " - " not in line:
            continue
        metrics: dict[str, Any] = {}
        for key, value in METRIC_RE.findall(line):
            try:
                numeric = float(value)
            except ValueError:
                continue
            metrics[key] = int(numeric) if numeric.is_integer() else numeric
        step = metrics.get("training/global_step") or metrics.get("step")
        if step is None:
            continue
        metrics["step"] = int(step)
        rows.append(metrics)
    deduped: dict[int, dict[str, Any]] = {}
    for row in rows:
        deduped[int(row["step"])] = row
    return [deduped[step] for step in sorted(deduped)]


def group_debug_records(run_dir: Path, tmp_dir: Path) -> list[dict[str, Any]]:
    debug_paths = sorted(tmp_dir.glob("*/trajectory_debug.json"), key=lambda path: path.stat().st_mtime)
    debug_records = [read_json(path) | {"debug_path": str(path), "mtime": path.stat().st_mtime} for path in debug_paths]
    sequence: list[tuple[str, int, int]] = []
    rollout_dir = run_dir / "rollout_data"
    validation_dir = run_dir / "validation_data"
    rollout_steps = sorted(int(path.stem) for path in rollout_dir.glob("*.jsonl") if path.stem.isdigit())
    validation_steps = {int(path.stem) for path in validation_dir.glob("*.jsonl") if path.stem.isdigit()}
    for step in rollout_steps:
        sequence.append(("train_like", step, count_jsonl(rollout_dir / f"{step}.jsonl")))
        if step in validation_steps:
            sequence.append(("validation_like", step, count_jsonl(validation_dir / f"{step}.jsonl")))

    groups: list[dict[str, Any]] = []
    cursor = 0
    for split, step, count in sequence:
        records = debug_records[cursor : cursor + count]
        cursor += count
        groups.append(summarize_debug_group(split, step, records, expected_count=count))
    if cursor != len(debug_records):
        groups.append(
            summarize_debug_group("unassigned", -1, debug_records[cursor:], expected_count=len(debug_records) - cursor)
        )
    return groups


def summarize_debug_group(
    split: str,
    step: int,
    records: list[dict[str, Any]],
    *,
    expected_count: int,
) -> dict[str, Any]:
    metrics = [record.get("trajectory_metrics") or {} for record in records]
    repair_events = [event for record in records for event in (record.get("repair_events") or [])]
    invalid_reasons = [reason for record in records for reason in (record.get("invalid_reasons") or [])]
    action_counts: Counter[str] = Counter()
    task_ids: list[str] = []
    worst: list[dict[str, Any]] = []
    multi_json_turns = 0
    malformed_turns = 0
    for record in records:
        task_id = str(record.get("task_id") or "")
        if task_id:
            task_ids.append(task_id)
        action_counts.update(str(item) for item in (record.get("step_actions") or []))
        multi_json_turns += sum(1 for raw in (record.get("raw_actions") or []) if str(raw).count('"action"') > 1)
        malformed_turns += sum(1 for item in (record.get("parsed_actions") or []) if item is None)
        worst.append(
            {
                "task_id": task_id,
                "score": record.get("score"),
                "final_reward": (record.get("trajectory_metrics") or {}).get("final_reward"),
                "finish": (record.get("trajectory_metrics") or {}).get("finish"),
                "claim_support_f1": (record.get("trajectory_metrics") or {}).get("claim_support_f1"),
                "abstain_f1": (record.get("trajectory_metrics") or {}).get("abstain_f1"),
                "unsupported_claim_count": (record.get("trajectory_metrics") or {}).get("unsupported_claim_count"),
                "premature_finish_count": (record.get("trajectory_metrics") or {}).get("premature_finish_count"),
                "invalid_reasons": record.get("invalid_reasons") or [],
                "repair_events": record.get("repair_events") or [],
                "debug_path": record.get("debug_path"),
            }
        )
    worst = sorted(worst, key=lambda item: float(item.get("score") or 0.0))[:5]
    repair_key_counts: Counter[str] = Counter()
    repair_reason_counts: Counter[str] = Counter()
    for event in repair_events:
        repair_key_counts.update(str(item) for item in (event.get("keys") or []))
        repair_reason_counts.update(str(item) for item in (event.get("reasons") or []))

    return {
        "split": split,
        "step": step,
        "expected_count": expected_count,
        "debug_count": len(records),
        "task_ids": task_ids,
        "score": numeric_summary([record.get("score") for record in records]),
        "schema_repair_penalty_total": numeric_summary(
            [record.get("schema_repair_penalty_total") for record in records]
        ),
        "metrics": {key: numeric_summary([metric.get(key) for metric in metrics]) for key in metric_keys(metrics)},
        "action_counts": dict(action_counts),
        "repair_event_count": len(repair_events),
        "repair_key_counts": dict(repair_key_counts),
        "repair_reason_counts": dict(repair_reason_counts),
        "invalid_reason_counts": dict(Counter(invalid_reasons)),
        "multi_json_turns": multi_json_turns,
        "malformed_turns": malformed_turns,
        "worst": worst,
    }


def metric_keys(metrics: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "final_reward",
        "trajectory_success",
        "finish",
        "crop_success",
        "claim_support_precision",
        "claim_support_recall",
        "claim_support_f1",
        "claim_supported_rate",
        "abstain_precision",
        "abstain_recall",
        "abstain_f1",
        "unsupported_claim_count",
        "core_unsupported_count",
        "placeholder_claim_count",
        "premature_finish_count",
        "premature_finish_rate",
        "invalid_steps",
        "invalid_step_rate",
        "steps",
        "predicted_claim_count",
        "predicted_non_abstain_claim_count",
        "predicted_abstain_count",
        "correct_abstain_count",
    ]
    present = {key for metric in metrics for key, value in metric.items() if is_number(value)}
    return [key for key in preferred if key in present]


def numeric_summary(values: list[Any]) -> dict[str, Any]:
    nums = [float(value) for value in values if is_number(value)]
    if not nums:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(nums),
        "mean": sum(nums) / len(nums),
        "min": min(nums),
        "max": max(nums),
    }


def summarize(
    run_dir: Path,
    tmp_dir: Path,
    log_file: Path,
    log_rows: list[dict[str, Any]],
    debug_groups: list[dict[str, Any]],
    title: str,
) -> dict[str, Any]:
    final_train = latest_group(debug_groups, "train_like")
    final_val = latest_group(debug_groups, "validation_like")
    final_log = log_rows[-1] if log_rows else {}
    headline = {
        "final_step": final_log.get("step"),
        "final_train_score_mean": get_nested(final_train, "score", "mean"),
        "final_val_score_mean": get_nested(final_val, "score", "mean"),
        "final_val_finish_rate": get_nested(final_val, "metrics", "finish", "mean"),
        "final_val_claim_support_f1": get_nested(final_val, "metrics", "claim_support_f1", "mean"),
        "final_val_abstain_f1": get_nested(final_val, "metrics", "abstain_f1", "mean"),
        "clip_ratio_final": final_log.get("response_length/clip_ratio"),
        "max_memory_allocated_gb_final": final_log.get("actor/perf/max_memory_allocated_gb"),
        "max_memory_reserved_gb_final": final_log.get("actor/perf/max_memory_reserved_gb"),
    }
    return {
        "title": title,
        "run_dir": str(run_dir),
        "tmp_dir": str(tmp_dir),
        "log_file": str(log_file),
        "headline": headline,
        "log_steps": len(log_rows),
        "final_log_metrics": final_log,
        "final_train_like": final_train,
        "final_validation_like": final_val,
        "all_validation_like": [group for group in debug_groups if group.get("split") == "validation_like"],
    }


def plot_curves(out_dir: Path, log_rows: list[dict[str, Any]], debug_groups: list[dict[str, Any]]) -> None:
    if not log_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["step"]) for row in log_rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    plot_metric(axes[0], steps, log_rows, "actor/loss", "actor/loss")
    plot_metric(axes[0], steps, log_rows, "actor/pg_loss", "pg_loss")
    axes[0].set_ylabel("loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)
    plot_metric(axes[1], steps, log_rows, "critic/score/mean", "score")
    plot_metric(axes[1], steps, log_rows, "response_length/clip_ratio", "clip_ratio")
    axes[1].set_ylabel("score / clip")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.25)
    plot_metric(axes[2], steps, log_rows, "actor/perf/max_memory_allocated_gb", "allocated GB")
    plot_metric(axes[2], steps, log_rows, "actor/perf/max_memory_reserved_gb", "reserved GB")
    axes[2].set_ylabel("GPU memory")
    axes[2].set_xlabel("global step")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_score_memory_curves.png", dpi=180)
    plt.close(fig)

    train_groups = [group for group in debug_groups if group.get("split") == "train_like"]
    val_groups = [group for group in debug_groups if group.get("split") == "validation_like"]
    if train_groups or val_groups:
        fig, ax = plt.subplots(figsize=(10, 5))
        for label, groups in [("train_like", train_groups), ("validation_like", val_groups)]:
            xs = [int(group["step"]) for group in groups if group.get("step", -1) >= 0]
            ys = [get_nested(group, "score", "mean") for group in groups if group.get("step", -1) >= 0]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{label} score")
        ax.set_xlabel("global step")
        ax.set_ylabel("agent-loop score")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "debug_score_curves.png", dpi=180)
        plt.close(fig)


def plot_metric(ax: Any, steps: list[int], rows: list[dict[str, Any]], key: str, label: str) -> None:
    vals = [row.get(key) for row in rows]
    if not any(is_number(value) for value in vals):
        return
    ax.plot(steps, [float(value) if is_number(value) else math.nan for value in vals], linewidth=1.4, label=label)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    h = summary["headline"]
    final_val = summary.get("final_validation_like") or {}
    final_train = summary.get("final_train_like") or {}
    lines = [
        f"# {summary['title']}",
        "",
        "## 核心结论",
        "",
        f"- final step: `{h.get('final_step')}`",
        f"- train-like score mean: `{fmt(h.get('final_train_score_mean'))}`",
        f"- validation-like score mean: `{fmt(h.get('final_val_score_mean'))}`",
        f"- validation finish_rate: `{fmt(h.get('final_val_finish_rate'))}`",
        f"- validation claim_support_f1: `{fmt(h.get('final_val_claim_support_f1'))}`",
        f"- validation abstain_f1: `{fmt(h.get('final_val_abstain_f1'))}`",
        f"- final response clip_ratio: `{fmt(h.get('clip_ratio_final'))}`",
        f"- final actor memory allocated/reserved GB: `{fmt(h.get('max_memory_allocated_gb_final'))}` / `{fmt(h.get('max_memory_reserved_gb_final'))}`",
        "",
        "## 曲线",
        "",
        "![loss_score_memory_curves](loss_score_memory_curves.png)",
        "",
        "![debug_score_curves](debug_score_curves.png)",
        "",
        "## Final Train-Like",
        "",
        metric_table(final_train),
        "",
        "## Final Validation-Like",
        "",
        metric_table(final_val),
        "",
        "## 最差样例",
        "",
        worst_table(final_val.get("worst") or []),
        "",
        "## 路径",
        "",
        f"- run_dir: `{summary['run_dir']}`",
        f"- tmp_dir: `{summary['tmp_dir']}`",
        f"- log_file: `{summary['log_file']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def metric_table(group: dict[str, Any]) -> str:
    if not group:
        return "_无数据_"
    keys = [
        ("score", ("score",)),
        ("finish_rate", ("metrics", "finish")),
        ("trajectory_success", ("metrics", "trajectory_success")),
        ("claim_support_precision", ("metrics", "claim_support_precision")),
        ("claim_support_recall", ("metrics", "claim_support_recall")),
        ("claim_support_f1", ("metrics", "claim_support_f1")),
        ("abstain_f1", ("metrics", "abstain_f1")),
        ("unsupported_claim_count", ("metrics", "unsupported_claim_count")),
        ("premature_finish_count", ("metrics", "premature_finish_count")),
        ("invalid_steps", ("metrics", "invalid_steps")),
        ("steps", ("metrics", "steps")),
        ("repair_events", ("repair_event_count",)),
        ("multi_json_turns", ("multi_json_turns",)),
        ("malformed_turns", ("malformed_turns",)),
    ]
    rows = ["| metric | mean/count | min | max |", "|---|---:|---:|---:|"]
    for label, path in keys:
        value = get_nested(group, *path)
        if isinstance(value, dict):
            rows.append(f"| {label} | {fmt(value.get('mean'))} | {fmt(value.get('min'))} | {fmt(value.get('max'))} |")
        else:
            rows.append(f"| {label} | {fmt(value)} |  |  |")
    return "\n".join(rows)


def worst_table(items: list[dict[str, Any]]) -> str:
    if not items:
        return "_无数据_"
    rows = ["| task_id | score | finish | support_f1 | abstain_f1 | unsupported | premature | invalid |", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for item in items:
        rows.append(
            "| {task_id} | {score} | {finish} | {support} | {abstain} | {unsupported} | {premature} | {invalid} |".format(
                task_id=item.get("task_id"),
                score=fmt(item.get("score")),
                finish=item.get("finish"),
                support=fmt(item.get("claim_support_f1")),
                abstain=fmt(item.get("abstain_f1")),
                unsupported=item.get("unsupported_claim_count"),
                premature=item.get("premature_finish_count"),
                invalid="<br>".join(str(reason) for reason in item.get("invalid_reasons") or []),
            )
        )
    return "\n".join(rows)


def latest_group(groups: list[dict[str, Any]], split: str) -> dict[str, Any]:
    candidates = [group for group in groups if group.get("split") == split]
    if not candidates:
        return {}
    return sorted(candidates, key=lambda group: int(group.get("step") or -1))[-1]


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def get_nested(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) or isinstance(value, bool)


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1.0000" if value else "0.0000"
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
