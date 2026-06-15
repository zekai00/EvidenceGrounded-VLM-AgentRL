#!/usr/bin/env python3
"""Summarize an SFT training run with loss and GPU memory plots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--title", default="")
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    assets_dir = run_dir / "training_assets"
    assets_dir.mkdir(exist_ok=True)

    summary = read_json(run_dir / "summary.json")
    run_config = read_json(run_dir / "run_config.json")
    train_log = list(iter_jsonl(run_dir / "train_log.jsonl"))
    gpu_log_path = run_dir / "gpu_memory_monitor.jsonl"
    gpu_log = list(iter_jsonl(gpu_log_path)) if gpu_log_path.exists() else []

    loss_plot = assets_dir / "loss_curve.png"
    gpu_plot = assets_dir / "gpu_memory_curve.png"
    plot_loss(train_log, summary, loss_plot)
    gpu_stats = plot_gpu(gpu_log, gpu_plot) if gpu_log else {}

    record_path = run_dir / "训练记录.md"
    write_markdown(record_path, args.title or run_dir.name, args.notes, summary, run_config, train_log, gpu_stats, loss_plot, gpu_plot if gpu_log else None)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "record": str(record_path),
                "loss_plot": str(loss_plot),
                "gpu_plot": str(gpu_plot) if gpu_log else None,
                "gpu_stats": gpu_stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def plot_loss(train_log: list[dict[str, Any]], summary: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [int(row["global_step"]) for row in train_log if "global_step" in row and "loss" in row]
    losses = [float(row["loss"]) for row in train_log if "global_step" in row and "loss" in row]
    lrs = [float(row.get("lr") or 0.0) for row in train_log if "global_step" in row and "loss" in row]

    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)
    if steps:
        ax1.plot(steps, losses, marker="o", linewidth=1.8, label="train loss")
    ax1.set_xlabel("optimizer step")
    ax1.set_ylabel("logged train loss")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    if steps and lrs:
        ax2.plot(steps, lrs, color="#777777", linestyle="--", linewidth=1.4, label="lr")
    ax2.set_ylabel("learning rate")
    final_val = summary.get("final_val_loss")
    if final_val is not None and steps:
        ax1.axhline(float(final_val), color="#c44e52", linestyle=":", linewidth=1.5, label=f"final val loss {float(final_val):.4f}")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def plot_gpu(gpu_log: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    series: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    stats: dict[str, Any] = {}
    sample_idx = 0
    for record in gpu_log:
        if not record.get("pid_alive"):
            continue
        for gpu in record.get("gpus") or []:
            try:
                gpu_index = int(gpu["gpu_index"])
                series[gpu_index].append((sample_idx, int(gpu["memory_used_mib"]), int(gpu.get("utilization_gpu_pct") or 0)))
                stat = stats.setdefault(f"gpu{gpu_index}", {})
                if "max_memory_allocated_mib" in gpu:
                    stat["max_torch_allocated_mib"] = max(
                        float(stat.get("max_torch_allocated_mib") or 0.0),
                        float(gpu.get("max_memory_allocated_mib") or 0.0),
                    )
                if "max_memory_reserved_mib" in gpu:
                    stat["max_torch_reserved_mib"] = max(
                        float(stat.get("max_torch_reserved_mib") or 0.0),
                        float(gpu.get("max_memory_reserved_mib") or 0.0),
                    )
            except Exception:
                continue
        sample_idx += 1

    fig, ax1 = plt.subplots(figsize=(9, 4.8), dpi=150)
    for gpu_index, values in sorted(series.items()):
        if not values:
            continue
        xs = [item[0] for item in values]
        mem = [item[1] for item in values]
        util = [item[2] for item in values]
        ax1.plot(xs, mem, linewidth=1.8, label=f"GPU{gpu_index} memory MiB")
        stat = stats.setdefault(f"gpu{gpu_index}", {})
        stat.update(
            {
                "max_memory_mib": max(mem),
                "mean_memory_mib": round(mean(mem), 2),
                "max_utilization_pct": max(util),
                "mean_utilization_pct": round(mean(util), 2),
                "samples": len(mem),
            }
        )
    ax1.set_xlabel("sample index")
    ax1.set_ylabel("memory used MiB")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return stats


def write_markdown(
    path: Path,
    title: str,
    notes: str,
    summary: dict[str, Any],
    run_config: dict[str, Any],
    train_log: list[dict[str, Any]],
    gpu_stats: dict[str, Any],
    loss_plot: Path,
    gpu_plot: Path | None,
) -> None:
    loss_rows = [row for row in train_log if "global_step" in row and "loss" in row]
    rel_loss = loss_plot.relative_to(path.parent)
    rel_gpu = gpu_plot.relative_to(path.parent) if gpu_plot else None
    lines = [
        f"# {title} 训练记录",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 输出目录：`{summary.get('output_dir', path.parent)}`",
        f"- 初始 adapter：`{summary.get('base_or_initial_adapter', run_config.get('adapter', ''))}`",
        f"- 训练数据：`{run_config.get('train_jsonl', '')}`",
        f"- 验证数据：`{run_config.get('val_jsonl', '')}`",
        "",
        "## 训练概要",
        "",
        f"- optimizer steps：{summary.get('optimizer_steps')}",
        f"- micro steps：{summary.get('micro_steps')}",
        f"- train rows used：{summary.get('train_rows_used')}",
        f"- val rows used：{summary.get('val_rows_used')}",
        f"- skipped batches：{summary.get('skipped_batches')}",
        f"- mean_train_loss：{summary.get('mean_train_loss')}",
        f"- final_val_loss：{summary.get('final_val_loss')}",
        "",
        "## Loss 变化",
        "",
        f"![loss curve]({rel_loss})",
        "",
        "| step | loss | lr |",
        "|---:|---:|---:|",
    ]
    for row in loss_rows:
        lines.append(f"| {row.get('global_step')} | {float(row.get('loss')):.6f} | {float(row.get('lr') or 0.0):.8g} |")
    lines.extend(["", "## 显存记录", ""])
    if rel_gpu:
        lines.append(f"![gpu memory]({rel_gpu})")
        lines.append("")
    if gpu_stats:
        lines.extend(
            [
                "| GPU | max sampled memory MiB | mean sampled memory MiB | max torch allocated MiB | max torch reserved MiB | max util % | mean util % | samples |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for gpu, stat in sorted(gpu_stats.items()):
            lines.append(
                f"| {gpu} | {stat.get('max_memory_mib')} | {stat.get('mean_memory_mib')} | "
                f"{stat.get('max_torch_allocated_mib', '')} | {stat.get('max_torch_reserved_mib', '')} | "
                f"{stat.get('max_utilization_pct')} | {stat.get('mean_utilization_pct')} | {stat.get('samples')} |"
            )
    else:
        lines.append("- 未发现 `gpu_memory_monitor.jsonl`，本次没有可用显存时间序列。")
    if notes:
        lines.extend(["", "## 备注", "", notes])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
