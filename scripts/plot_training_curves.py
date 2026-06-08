#!/usr/bin/env python3
"""Plot training loss and learning-rate curves from saved train_log.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/root/models/evidence_grounded_vlm_agentrl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name-filter", action="append", default=[])
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = discover_runs(args.model_root, args.name_filter)
    if args.max_runs > 0:
        runs = runs[: args.max_runs]
    if not runs:
        raise SystemExit("no train_log.jsonl files found")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries: list[dict[str, Any]] = []
    for run in runs:
        rows = read_jsonl(run / "train_log.jsonl")
        if not rows:
            continue
        config = read_json(run / "run_config.json")
        train_summary = read_json(run / "summary.json")
        summary = summarize_run(run, rows, config, train_summary)
        summaries.append(summary)
        plot_single_run(plt, run.name, rows, args.output_dir)

    plot_combined(plt, summaries, args.output_dir, metric="lr")
    plot_combined(plt, summaries, args.output_dir, metric="loss")
    (args.output_dir / "training_curve_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "runs": len(summaries)}, ensure_ascii=False, indent=2))
    return 0


def discover_runs(root: Path, filters: list[str]) -> list[Path]:
    runs = sorted(path.parent for path in root.glob("*/train_log.jsonl"))
    if filters:
        runs = [run for run in runs if any(item in run.name for item in filters)]
    return runs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(
    run: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    train_summary: dict[str, Any],
) -> dict[str, Any]:
    args = config.get("args") or {}
    losses = [float(row.get("loss", 0.0)) for row in rows if "loss" in row]
    lrs = [float(row.get("lr", 0.0)) for row in rows if "lr" in row]
    steps = [int(row.get("global_step", 0)) for row in rows if "global_step" in row]
    return {
        "name": run.name,
        "path": str(run),
        "train_log": str(run / "train_log.jsonl"),
        "run_config": str(run / "run_config.json"),
        "logged_points": len(rows),
        "first_step": min(steps) if steps else None,
        "last_step": max(steps) if steps else None,
        "peak_lr": max(lrs) if lrs else None,
        "final_logged_lr": lrs[-1] if lrs else None,
        "mean_logged_loss": sum(losses) / max(1, len(losses)),
        "final_logged_loss": losses[-1] if losses else None,
        "train_rows_used": train_summary.get("train_rows_used", config.get("train_rows_used")),
        "val_rows_used": train_summary.get("val_rows_used", config.get("val_rows_used")),
        "optimizer_steps": train_summary.get("optimizer_steps", config.get("optimizer_steps")),
        "final_val_loss": train_summary.get("final_val_loss", config.get("final_val_loss")),
        "learning_rate_arg": args.get("learning_rate"),
        "warmup_ratio": args.get("warmup_ratio"),
        "gradient_accumulation_steps": args.get("gradient_accumulation_steps"),
        "batch_size": args.get("batch_size"),
        "max_seq_length": args.get("max_seq_length"),
    }


def plot_single_run(plt: Any, name: str, rows: list[dict[str, Any]], output_dir: Path) -> None:
    steps = [int(row.get("global_step", 0)) for row in rows]
    losses = [float(row.get("loss", 0.0)) for row in rows]
    lrs = [float(row.get("lr", 0.0)) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(steps, losses, linewidth=1.8)
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title(name)
    axes[1].plot(steps, lrs, linewidth=1.8, color="#b45309")
    axes[1].set_ylabel("learning rate")
    axes[1].set_xlabel("optimizer step")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{safe_name(name)}_loss_lr.png", dpi=160)
    plt.close(fig)


def plot_combined(plt: Any, summaries: list[dict[str, Any]], output_dir: Path, metric: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for summary in summaries:
        rows = read_jsonl(Path(summary["train_log"]))
        steps = [int(row.get("global_step", 0)) for row in rows]
        values = [float(row.get(metric, 0.0)) for row in rows]
        ax.plot(steps, values, linewidth=1.4, label=short_label(summary["name"]))
    ax.set_xlabel("optimizer step")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"combined_{metric}.png", dpi=180)
    plt.close(fig)


def short_label(name: str) -> str:
    for prefix in ["qwen25vl3b_", "evidence_grounded_"]:
        name = name.replace(prefix, "")
    return name[:58]


def safe_name(name: str) -> str:
    keep = []
    for char in name:
        keep.append(char if char.isalnum() or char in {"-", "_"} else "_")
    return "".join(keep)


if __name__ == "__main__":
    raise SystemExit(main())
