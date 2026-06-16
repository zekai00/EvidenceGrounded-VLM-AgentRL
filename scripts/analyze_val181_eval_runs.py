#!/usr/bin/env python3
"""Aggregate val181 evaluation runs and generate a Markdown report.

The script intentionally depends only on the Python standard library plus PIL,
because the training environment used for these runs does not include
matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from PIL import Image, ImageDraw, ImageFont


RUNS = {
    "base": {
        "display": "Base Qwen2.5-VL-3B-Instruct",
        "short": "Base",
        "kind": "base model",
        "model_path": "/root/models/Qwen2.5-VL-3B-Instruct",
        "jsonl": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_base_qwen25vl3b_20260616_1046/validation_data/0.jsonl",
        "debug_glob": "/root/Workspace/VLM/tmp/evidence_grounded_v1_3_1_val181_base_20260616_1046/*/trajectory_debug.json",
        "log": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_base_qwen25vl3b_20260616_1046.log",
        "output_dir": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_base_qwen25vl3b_20260616_1046",
    },
    "sft": {
        "display": "v1.3.1 Continued-B SFT LoRA",
        "short": "SFT",
        "kind": "LoRA adapter",
        "model_path": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/v1_3_1_continued_from_v13best_sft_qwen25vl3b_full_save250_20260614_1652/adapter",
        "jsonl": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_sft_continuedB_20260616_1105/validation_data/0.jsonl",
        "debug_glob": "/root/Workspace/VLM/tmp/evidence_grounded_v1_3_1_val181_sft_20260616_1105/*/trajectory_debug.json",
        "log": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_sft_continuedB_20260616_1105.log",
        "output_dir": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_sft_continuedB_20260616_1105",
    },
    "rl": {
        "display": "Stage A5.3 GRPO 160-step LoRA",
        "short": "RL",
        "kind": "GRPO checkpoint",
        "model_path": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/v1_3_1_trajectory_grpo_stageA5_3_scopefix_claimalias_4gpu_160step_6144_keep_20260616_0355/global_step_160",
        "jsonl": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_rl_stageA5_3_160step_20260616_1143/validation_data/160.jsonl",
        "debug_glob": "/root/Workspace/VLM/tmp/evidence_grounded_v1_3_1_val181_rl_20260616_1143/*/trajectory_debug.json",
        "log": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_rl_stageA5_3_160step_20260616_1143.log",
        "output_dir": "/root/EvidenceGrounded-VLM-AgentRL-Outputs/outputs/val181_eval_rl_stageA5_3_160step_20260616_1143",
    },
}


PRIMARY_METRICS = [
    "score",
    "finish",
    "trajectory_success",
    "crop_success",
    "evidence_f1",
    "cited_evidence_f1",
    "claim_support_f1",
    "claim_support_precision",
    "claim_support_recall",
    "abstain_f1",
    "abstain_precision",
    "abstain_recall",
    "abstain_accuracy",
    "unsupported_claim_count",
    "premature_finish_count",
    "invalid_steps",
    "steps",
]


SAMPLE_TASKS_FALLBACK = ["v13_t_000744", "v13_t_001253", "v13_t_000125"]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for name in names:
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return None
        return float(value)
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            gts = json.loads(row["gts"])
            row["_task_id"] = gts["task_id"]
            row["_gts"] = gts
            rows.append(row)
    return rows


def read_debug(glob_pattern: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    debug_by_task: dict[str, dict[str, Any]] = {}
    seen: Counter[str] = Counter()
    duplicates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(Path().glob(glob_pattern) if not glob_pattern.startswith("/") else Path("/").glob(glob_pattern[1:])):
        data = json.loads(path.read_text(encoding="utf-8"))
        task_id = data["task_id"]
        seen[task_id] += 1
        data["_debug_path"] = str(path)
        duplicates[task_id].append(data)
        if task_id not in debug_by_task:
            debug_by_task[task_id] = data
    dup_summary = {
        task_id: len(items)
        for task_id, items in duplicates.items()
        if len(items) > 1
    }
    return debug_by_task, {"debug_files": sum(seen.values()), "debug_unique_tasks": len(seen), "duplicates": dup_summary}


def parse_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    out: dict[str, Any] = {}
    start = re.search(r"START ([^\n]+)", text)
    end = re.search(r"END ([^\n]+) rc=(\d+)", text)
    if start:
        out["start"] = start.group(1).strip()
    if end:
        out["end"] = end.group(1).strip()
        out["rc"] = int(end.group(2))
    if start and end:
        try:
            s = dt.datetime.fromisoformat(out["start"])
            e = dt.datetime.fromisoformat(out["end"])
            out["duration_minutes"] = round((e - s).total_seconds() / 60.0, 2)
        except ValueError:
            pass
    reward = re.findall(r"reward/mean@1:([\-0-9.]+)", text)
    if reward:
        out["logged_reward_mean"] = float(reward[-1])
    turns = re.findall(r"val-aux/num_turns/min:(\d+) - val-aux/num_turns/max:(\d+) - val-aux/num_turns/mean:([0-9.]+)", text)
    if turns:
        out["num_turns_min"] = int(turns[-1][0])
        out["num_turns_max"] = int(turns[-1][1])
        out["num_turns_mean"] = float(turns[-1][2])
    return out


def aggregate_run(run_key: str, config: dict[str, str]) -> dict[str, Any]:
    rows = read_jsonl(Path(config["jsonl"]))
    debug_by_task, debug_meta = read_debug(config["debug_glob"])
    task_ids = [row["_task_id"] for row in rows]
    combined: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = row["_task_id"]
        debug = debug_by_task.get(task_id, {})
        metrics = dict(debug.get("trajectory_metrics") or {})
        metrics["score"] = float(row["score"])
        combined[task_id] = {
            "task_id": task_id,
            "row": row,
            "debug": debug,
            "metrics": metrics,
        }

    metric_values: dict[str, list[float]] = defaultdict(list)
    action_sums: Counter[str] = Counter()
    for task in combined.values():
        metrics = task["metrics"]
        for key, value in metrics.items():
            if key == "action_counts":
                for action, count in (value or {}).items():
                    action_sums[action] += int(count)
                continue
            number = to_float(value)
            if number is not None:
                metric_values[key].append(number)

    summary: dict[str, Any] = {
        "run_key": run_key,
        "display": config["display"],
        "short": config["short"],
        "kind": config["kind"],
        "model_path": config["model_path"],
        "jsonl": config["jsonl"],
        "debug_glob": config["debug_glob"],
        "log": config["log"],
        "output_dir": config["output_dir"],
        "n_rows": len(rows),
        "n_unique_tasks": len(set(task_ids)),
        **debug_meta,
        **parse_log(Path(config["log"])),
    }
    for key, values in metric_values.items():
        if values:
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_min"] = min(values)
            summary[f"{key}_max"] = max(values)
            summary[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
    scores = metric_values["score"]
    summary["score_full_count"] = sum(1 for value in scores if value >= 0.999)
    summary["score_ge_0_9_count"] = sum(1 for value in scores if value >= 0.9)
    summary["score_lt_0_5_count"] = sum(1 for value in scores if value < 0.5)
    summary["action_avg"] = {action: count / len(combined) for action, count in sorted(action_sums.items())}
    return {"summary": summary, "tasks": combined}


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def draw_bar_chart(path: Path, title: str, labels: list[str], values: list[float], ylabel: str, colors: list[tuple[int, int, int]]) -> None:
    width, height = 1100, 650
    margin_left, margin_right, margin_top, margin_bottom = 110, 55, 95, 120
    img = Image.new("RGB", (width, height), (250, 251, 253))
    draw = ImageDraw.Draw(img)
    font_title = load_font(30, bold=True)
    font_label = load_font(22)
    font_small = load_font(18)
    font_value = load_font(20, bold=True)
    draw.text((margin_left, 28), title, fill=(28, 35, 45), font=font_title)
    draw.text((margin_left, 65), ylabel, fill=(88, 97, 110), font=font_small)

    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    max_val = max(values) if values else 1.0
    if max_val <= 1.05:
        y_max = 1.0
    else:
        y_max = max_val * 1.15
    y_max = max(y_max, 0.1)
    x0, y0 = margin_left, margin_top
    x1, y1 = width - margin_right, height - margin_bottom

    for i in range(6):
        y = y1 - plot_h * i / 5
        value = y_max * i / 5
        draw.line((x0, y, x1, y), fill=(224, 229, 236), width=1)
        draw.text((32, y - 11), f"{value:.2f}", fill=(97, 110, 124), font=font_small)
    draw.line((x0, y0, x0, y1), fill=(150, 158, 170), width=2)
    draw.line((x0, y1, x1, y1), fill=(150, 158, 170), width=2)

    gap = 70
    bar_w = (plot_w - gap * (len(values) + 1)) / max(len(values), 1)
    for idx, (label, value) in enumerate(zip(labels, values)):
        left = x0 + gap + idx * (bar_w + gap)
        right = left + bar_w
        bar_h = plot_h * (value / y_max)
        top = y1 - bar_h
        color = colors[idx % len(colors)]
        draw.rounded_rectangle((left, top, right, y1), radius=8, fill=color)
        value_text = f"{value:.3f}" if abs(value) < 10 else f"{value:.1f}"
        tw = draw.textlength(value_text, font=font_value)
        draw.text((left + (bar_w - tw) / 2, top - 30), value_text, fill=(28, 35, 45), font=font_value)
        lw = draw.textlength(label, font=font_label)
        draw.text((left + (bar_w - lw) / 2, y1 + 22), label, fill=(28, 35, 45), font=font_label)

    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def copy_sample_images(task_id: str, task: dict[str, Any], assets_dir: Path) -> dict[str, str]:
    gts = task["row"]["_gts"]
    tasks_path = Path(gts["tasks_path"])
    record = None
    with tasks_path.open(encoding="utf-8") as fh:
        for line in fh:
            data = json.loads(line)
            if data.get("task_id") == task_id:
                record = data
                break
    if not record:
        return {}
    sample_dir = assets_dir / "samples" / task_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key in ["overlay_image", "artwork_image", "caption_image", "page_image"]:
        src = record.get(key)
        if not src:
            continue
        src_path = Path(src)
        if not src_path.exists():
            continue
        suffix = src_path.suffix.lower() or ".jpg"
        dst = sample_dir / f"{key}{suffix}"
        shutil.copy2(src_path, dst)
        copied[key] = str(dst)
    copied["_task_record"] = str(tasks_path)
    return copied


def compact_action_text(text: Any, max_chars: int = 900) -> str:
    cleaned = " ".join(str(text).splitlines())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_chars:
        return cleaned[: max_chars - 18].rstrip() + " ...[已截断]"
    return cleaned


def action_lines(task: dict[str, Any], limit: int | None = None) -> list[str]:
    debug = task.get("debug") or {}
    raw = list(debug.get("raw_actions") or [])
    out = []
    total = len(raw)
    for idx, text in enumerate(raw[: limit or total], start=1):
        out.append(f"{idx:02d}. {compact_action_text(text)}")
    if limit and total > limit:
        out.append(f"... ({total - limit} more actions)")
    return out


def invalid_reason_lines(task: dict[str, Any]) -> str:
    reasons = list((task.get("debug") or {}).get("invalid_reasons") or [])
    if not reasons:
        return "非法/修复原因：无"
    lines = ["非法/修复原因："]
    for reason in reasons:
        lines.append(f"- {reason}")
    return "\n".join(lines)


def select_samples(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    base = results["base"]["tasks"]
    sft = results["sft"]["tasks"]
    rl = results["rl"]["tasks"]
    candidates = []
    for task_id in sorted(rl):
        b = base[task_id]["metrics"]["score"]
        s = sft[task_id]["metrics"]["score"]
        r = rl[task_id]["metrics"]["score"]
        candidates.append((task_id, b, s, r))

    samples = []
    strict_rescue = []
    loose_rescue = []
    for task_id, b, s, r in candidates:
        metrics = rl[task_id]["metrics"]
        ok_rl = (
            r >= 0.99
            and bool(metrics.get("trajectory_success"))
            and metrics.get("unsupported_claim_count") == 0
            and metrics.get("invalid_steps") == 0
        )
        if not ok_rl:
            continue
        item = (task_id, b, s, r)
        if b < 0.8 and s < 0.8:
            strict_rescue.append(item)
        elif s < 0.8 or b < 0.8:
            loose_rescue.append(item)
    rescue = sorted(strict_rescue or loose_rescue, key=lambda x: (x[3] - min(x[1], x[2]), x[3] - x[2]), reverse=True)
    for task_id, b, s, r in rescue:
        samples.append({"task_id": task_id, "reason": "RL 修复：RL 成功完成且无 unsupported/invalid，base 与 SFT 至少一组明显较低"})
        break
    best_all = sorted(candidates, key=lambda x: (x[3], x[2], x[1]), reverse=True)
    for task_id, b, s, r in best_all:
        if task_id not in {s["task_id"] for s in samples} and r >= 0.99 and s >= 0.99:
            samples.append({"task_id": task_id, "reason": "稳定成功：SFT 与 RL 都能完成"})
            break
    worst_rl = sorted(candidates, key=lambda x: x[3])
    for task_id, b, s, r in worst_rl:
        if task_id not in {s["task_id"] for s in samples}:
            samples.append({"task_id": task_id, "reason": "残余失败：RL 得分最低，需要继续分析"})
            break

    used = {sample["task_id"] for sample in samples}
    for task_id in SAMPLE_TASKS_FALLBACK:
        if len(samples) >= 3:
            break
        if task_id in rl and task_id not in used:
            samples.append({"task_id": task_id, "reason": "固定回归样例"})
            used.add(task_id)
    return samples


def write_csv(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    keys = ["run", "display", "n_rows", "score_mean", "score_min", "score_max", "finish_mean", "trajectory_success_mean", "claim_support_f1_mean", "abstain_f1_mean", "unsupported_claim_count_mean", "premature_finish_count_mean", "invalid_steps_mean", "steps_mean", "num_turns_mean", "duration_minutes"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for run_key, summary in summaries.items():
            writer.writerow({key: summary.get(key if key != "run" else "run_key") for key in keys})


def build_report(results: dict[str, dict[str, Any]], report_dir: Path, assets_dir: Path) -> str:
    summaries = {key: value["summary"] for key, value in results.items()}
    samples = select_samples(results)
    rows = []
    for key in ["base", "sft", "rl"]:
        summary = summaries[key]
        rows.append([
            summary["short"],
            fmt(summary.get("score_mean")),
            fmt(summary.get("score_min")),
            fmt(summary.get("score_max")),
            fmt(summary.get("finish_mean")),
            fmt(summary.get("trajectory_success_mean")),
            fmt(summary.get("claim_support_f1_mean")),
            fmt(summary.get("abstain_f1_mean")),
            fmt(summary.get("unsupported_claim_count_mean")),
            fmt(summary.get("invalid_steps_mean")),
            fmt(summary.get("num_turns_mean"), 2),
            fmt(summary.get("duration_minutes"), 2),
        ])
    summary_table = md_table(
        ["方案", "奖励均值", "最低分", "最高分", "Finish率", "轨迹成功率", "Claim F1", "Abstain F1", "Unsupported均值", "非法步数", "平均轮数", "耗时(分钟)"],
        rows,
    )

    delta_rows = []
    pairs = [("SFT - Base", "sft", "base"), ("RL - SFT", "rl", "sft"), ("RL - Base", "rl", "base")]
    for label, a, b in pairs:
        sa, sb = summaries[a], summaries[b]
        delta_rows.append([
            label,
            fmt(sa.get("score_mean", 0) - sb.get("score_mean", 0)),
            fmt(sa.get("finish_mean", 0) - sb.get("finish_mean", 0)),
            fmt(sa.get("trajectory_success_mean", 0) - sb.get("trajectory_success_mean", 0)),
            fmt(sa.get("claim_support_f1_mean", 0) - sb.get("claim_support_f1_mean", 0)),
            fmt(sa.get("abstain_f1_mean", 0) - sb.get("abstain_f1_mean", 0)),
            fmt(sa.get("unsupported_claim_count_mean", 0) - sb.get("unsupported_claim_count_mean", 0)),
        ])
    delta_table = md_table(
        ["对比", "奖励变化", "Finish变化", "轨迹成功变化", "Claim F1变化", "Abstain F1变化", "Unsupported变化"],
        delta_rows,
    )

    action_rows = []
    for key in ["base", "sft", "rl"]:
        action_avg = summaries[key].get("action_avg", {})
        action_rows.append([
            summaries[key]["short"],
            fmt(action_avg.get("inspect_page", 0), 2),
            fmt(action_avg.get("crop_target", 0), 2),
            fmt(action_avg.get("open_evidence", 0), 2),
            fmt(action_avg.get("retrieve_evidence", 0), 2),
            fmt(action_avg.get("write_claim", 0), 2),
            fmt(action_avg.get("abstain_claim", 0), 2),
            fmt(action_avg.get("finish", 0), 2),
        ])
    action_table = md_table(
        ["方案", "inspect", "crop", "open", "retrieve", "write", "abstain", "finish"],
        action_rows,
    )

    plot_reward = assets_dir / "val181_reward_mean.png"
    plot_core = assets_dir / "val181_core_rates.png"
    plot_errors = assets_dir / "val181_error_counts.png"
    colors = [(74, 111, 165), (213, 122, 47), (65, 145, 103)]
    labels = [summaries[k]["short"] for k in ["base", "sft", "rl"]]
    draw_bar_chart(plot_reward, "val181 Reward Mean", labels, [summaries[k].get("score_mean", 0) for k in ["base", "sft", "rl"]], "higher is better", colors)
    draw_bar_chart(plot_core, "Core Success Rates", labels, [summaries[k].get("trajectory_success_mean", 0) for k in ["base", "sft", "rl"]], "trajectory_success mean", colors)
    draw_bar_chart(plot_errors, "Unsupported Claims", labels, [summaries[k].get("unsupported_claim_count_mean", 0) for k in ["base", "sft", "rl"]], "lower is better", colors)

    sample_sections = []
    for sample in samples:
        task_id = sample["task_id"]
        image_paths = copy_sample_images(task_id, results["rl"]["tasks"][task_id], assets_dir)
        score_bits = []
        for run_key in ["base", "sft", "rl"]:
            task = results[run_key]["tasks"][task_id]
            metrics = task["metrics"]
            score_bits.append(
                f"{summaries[run_key]['short']}：得分={fmt(metrics.get('score'))}，finish={fmt(metrics.get('finish'))}，"
                f"claim_f1={fmt(metrics.get('claim_support_f1'))}，abstain_f1={fmt(metrics.get('abstain_f1'))}，"
                f"unsupported={fmt(metrics.get('unsupported_claim_count'))}"
            )
        image_md = []
        for key, label in [("overlay_image", "页面标注图"), ("artwork_image", "目标裁剪图"), ("caption_image", "Caption 裁剪图")]:
            path = image_paths.get(key)
            if path:
                rel = Path(path).relative_to(report_dir)
                image_md.append(f"![{label}]({rel.as_posix()})")

        actions_blocks = []
        for run_key in ["base", "sft", "rl"]:
            task = results[run_key]["tasks"][task_id]
            actions = "\n".join(action_lines(task))
            invalids = invalid_reason_lines(task)
            actions_blocks.append(
                f"<details><summary>{summaries[run_key]['short']} 完整动作序列</summary>\n\n"
                f"{invalids}\n\n"
                f"```text\n{actions}\n```\n\n</details>"
            )

        sample_sections.append(
            "\n".join(
                [
                    f"### {task_id} - {sample['reason']}",
                    "\n".join(image_md),
                    "",
                    "\n".join(f"- {bit}" for bit in score_bits),
                    "",
                    "\n\n".join(actions_blocks),
                ]
            )
        )

    paths_table = md_table(
        ["方案", "模型/Checkpoint", "Validation JSONL", "日志"],
        [
            [summaries[k]["short"], summaries[k]["model_path"], summaries[k]["jsonl"], summaries[k]["log"]]
            for k in ["base", "sft", "rl"]
        ],
    )

    report = f"""# val181 三组模型对照评测报告（Base / SFT / RL）

生成时间：{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 结论摘要

- 这次完整评测覆盖 `181` 个 trajectory-level validation tasks，三组使用同一套 agent loop、reward 和确定性 validation 设置。
- Base 奖励均值 = `{fmt(summaries['base'].get('score_mean'))}`；SFT 奖励均值 = `{fmt(summaries['sft'].get('score_mean'))}`；RL 奖励均值 = `{fmt(summaries['rl'].get('score_mean'))}`。
- SFT 在完整 val181 上低于 base，主要说明 continued-B SFT adapter 会模仿轨迹格式，但泛化到全量 validation 时有更多错误 claim / abstain 策略偏差。
- Stage A5.3 GRPO 160-step 在 val181 上相对 SFT 提升 `{fmt(summaries['rl'].get('score_mean', 0) - summaries['sft'].get('score_mean', 0))}`，相对 base 提升 `{fmt(summaries['rl'].get('score_mean', 0) - summaries['base'].get('score_mean', 0))}`。
- RL 的主要收益来自 `finish_rate`、`trajectory_success`、`claim_support_f1` 和 `invalid_steps`；`unsupported_claim_count` 仍是残余问题，RL 均值 `{fmt(summaries['rl'].get('unsupported_claim_count_mean'))}` 略高于 SFT `{fmt(summaries['sft'].get('unsupported_claim_count_mean'))}` 和 base `{fmt(summaries['base'].get('unsupported_claim_count_mean'))}`。
- 用户给出的 `...20260616_0410` RL 目录为空；本次使用实际完整 checkpoint：`{summaries['rl']['model_path']}`。

## 评测设置

- 数据切分：`/root/datasets/evidence_grounded_vlm_agentrl/rlvr_v1_3_1_trajectory_level_latest/verl/val.parquet`
- 样本规模：`181` 行，`181` 个唯一 task ID。
- 运行模式：`trainer.val_only=true`，只做 validation，不更新 optimizer，不保存新 checkpoint。
- 生成设置：`MAX_RESPONSE_LENGTH=6144`，`max_model_len=16384`，`temperature=0`，`do_sample=false`，`n=1`。
- 并行设置：`N_GPUS_PER_NODE=4`，`AGENT_NUM_WORKERS=4`；debug trace 会 pad 到 `184` 个文件，下方指标已按 181 个唯一 validation task 去重。
- 启动脚本：`scripts/run_verl_v1_3_1_trajectory_val_eval.sh`。

## 主指标

{summary_table}

![奖励均值]({plot_reward.relative_to(report_dir).as_posix()})

![轨迹成功率]({plot_core.relative_to(report_dir).as_posix()})

![Unsupported claim 均值]({plot_errors.relative_to(report_dir).as_posix()})

## 增量对比

{delta_table}

## 行为分布

平均每条 trajectory 的 action 次数：

{action_table}

## 指标解释

- 奖励均值：verl validation JSONL 中的 `score` 均值，是本次主对比指标。
- Finish率：trajectory 最终是否合法调用 `finish`。
- 轨迹成功率：reward 侧综合成功标记，通常要求定位、证据、claim/abstain 和 finish 都基本满足。
- Claim F1：支持性 claim 的 field/evidence 支持匹配 F1，越高说明写出的 claim 更能被证据支撑。
- Abstain F1：对无证据字段是否正确显式 abstain 的 F1。
- Unsupported均值：平均每条轨迹中 unsupported claim 数，越低越好。
- 非法步数：平均每条轨迹非法 action/解析失败步数，越低越好。
- 平均轮数：verl 日志里的响应轮数统计；它和 `steps` 不完全等价，会包含 agent loop 的 tool/assistant 交互轮。

## 样例轨迹

下面样例包含图像、三组得分和动作序列。为保证 Markdown 可读性，单步原始输出会被规范成单行展示；极长的异常输出会标记为 `...[已截断]`。完整原始输出仍可从后面的 JSONL 和 debug trace 路径追溯。

{(chr(10) * 2).join(sample_sections)}

## 运行路径

{paths_table}

## 数据文件

- 聚合 CSV: `val181_metrics_summary.csv`
- 聚合 JSON: `val181_metrics_summary.json`
- 图片资产目录: `assets/`

## 中文总结

本报告比较了 base Qwen2.5-VL-3B-Instruct、continued-B SFT LoRA 和 Stage A5.3 GRPO 160-step checkpoint 在完整 `val181` trajectory-level validation split 上的表现。三组使用相同 agent loop、reward、确定性 validation 参数和长度上限。结果显示，RL checkpoint 显著提升主 reward、finish rate、trajectory success、claim support F1，并明显降低 invalid steps。仍需继续优化的是 unsupported claim：RL 的 unsupported claim 均值略高于 SFT/base。SFT-only adapter 在完整 validation 上低于 base，说明单纯模仿能学到动作格式，但还不足以稳定学会证据不足时 abstain、以及避免 unsupported claim。
"""
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", default="/root/EvidenceGrounded-VLM-AgentRL-Outputs/docs/03_实验报告")
    parser.add_argument("--report-name", default="20260616_val181_base_sft_rl对照评测报告.md")
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = report_dir / "assets" / "20260616_val181_base_sft_rl"
    assets_dir.mkdir(parents=True, exist_ok=True)

    results = {key: aggregate_run(key, config) for key, config in RUNS.items()}
    summaries = {key: value["summary"] for key, value in results.items()}

    (report_dir / "val181_metrics_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(report_dir / "val181_metrics_summary.csv", summaries)
    report = build_report(results, report_dir, assets_dir)
    report_path = report_dir / args.report_name
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
