#!/usr/bin/env python3
"""Build a v0.6 hard-negative patch SFT dataset for write_claims_chunk.

The patch keeps all original rows as replay, adds a short phase hint to
write_claims_chunk rows, and oversamples chunk-writing states. Rows observed as
failures in train-set diagnostics can receive a stronger oversampling factor.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_6_chunked_claim_sft_20260604_1650/sft"
)

DEFAULT_PHASE_HINT = (
    "阶段提示：当前通常已经完成候选区域选择、裁剪、证据检索和证据打开；"
    "如果 claim_state 显示仍有待写字段，并且已有 selected_evidence_ids 或工具返回中的 evidence，"
    "优先调用 write_claims_chunk 写入下一组结构化 claim。"
    "除非明确缺少必要证据，否则不要继续 open_evidence 或 finish。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-action", default="write_claims_chunk")
    parser.add_argument("--train-target-oversample", type=int, default=2)
    parser.add_argument("--train-hard-oversample", type=int, default=6)
    parser.add_argument("--eval-target-oversample", type=int, default=1)
    parser.add_argument("--failure-predictions", default="", help="Comma-separated train prediction JSONL files.")
    parser.add_argument("--low-f1-threshold", type=float, default=0.5)
    parser.add_argument("--phase-hint", default=DEFAULT_PHASE_HINT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    failure_keys, failure_stats = load_failure_keys(
        prediction_paths(args.failure_predictions),
        low_f1_threshold=args.low_f1_threshold,
        target_action=args.target_action,
    )
    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.6_chunked_claim_hardnegative_patch",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "target_action": args.target_action,
        "train_target_oversample": args.train_target_oversample,
        "train_hard_oversample": args.train_hard_oversample,
        "eval_target_oversample": args.eval_target_oversample,
        "failure_predictions": [str(path) for path in prediction_paths(args.failure_predictions)],
        "failure_key_count": len(failure_keys),
        "failure_stats": failure_stats,
        "phase_hint": args.phase_hint,
        "seed": args.seed,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        src = args.input_dir / f"{split}.jsonl"
        rows = read_jsonl(src)
        patched, split_stats = patch_split(
            rows,
            split=split,
            target_action=args.target_action,
            failure_keys=failure_keys if split == "train" else set(),
            train_target_oversample=args.train_target_oversample,
            train_hard_oversample=args.train_hard_oversample,
            eval_target_oversample=args.eval_target_oversample,
            phase_hint=args.phase_hint,
        )
        random.shuffle(patched)
        out_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(out_path, patched)
        split_stats["file"] = str(out_path)
        manifest["splits"][split] = split_stats
        print(
            f"[{split}] {split_stats['source_rows']} -> {split_stats['rows_written']} rows; "
            f"{args.target_action}={split_stats['patched_action_counts'].get(args.target_action, 0)}; "
            f"hard={split_stats['hard_rows_seen']}"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "构建报告.md", manifest)
    print(f"manifest -> {manifest_path}")
    return 0


def patch_split(
    rows: list[dict[str, Any]],
    *,
    split: str,
    target_action: str,
    failure_keys: set[tuple[str, int]],
    train_target_oversample: int,
    train_hard_oversample: int,
    eval_target_oversample: int,
    phase_hint: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    hard_rows_seen = 0
    hinted_rows_seen = 0
    copy_counter: Counter[int] = Counter()
    for row in rows:
        action = action_name(row)
        key = row_key(row)
        is_target = action == target_action
        is_hard = split == "train" and is_target and key in failure_keys
        if is_hard:
            copies = max(1, train_hard_oversample)
            hard_rows_seen += 1
        elif is_target:
            copies = max(1, train_target_oversample if split == "train" else eval_target_oversample)
        else:
            copies = 1
        copy_counter[copies] += 1

        for copy_index in range(copies):
            copied = copy.deepcopy(row)
            if is_target:
                add_phase_hint(copied, phase_hint)
                hinted_rows_seen += int(copy_index == 0)
            copied["patch_source"] = {
                "kind": "v0_6_chunk_hardnegative_phase_hint",
                "split": split,
                "is_target_action": is_target,
                "is_hard_negative": is_hard,
                "copy_index": copy_index,
                "copies": copies,
                "row_key": {"task_id": key[0], "step": key[1]},
            }
            patched.append(copied)

    return patched, {
        "source_rows": len(rows),
        "rows_written": len(patched),
        "source_action_counts": dict(sorted(action_counts(rows).items())),
        "patched_action_counts": dict(sorted(action_counts(patched).items())),
        "target_rows_seen": sum(1 for row in rows if action_name(row) == target_action),
        "hard_rows_seen": hard_rows_seen,
        "hinted_target_rows_seen": hinted_rows_seen,
        "copy_count_distribution": {str(k): v for k, v in sorted(copy_counter.items())},
    }


def load_failure_keys(
    paths: list[Path],
    *,
    low_f1_threshold: float,
    target_action: str,
) -> tuple[set[tuple[str, int]], dict[str, Any]]:
    keys: set[tuple[str, int]] = set()
    stats: Counter[str] = Counter()
    for path in paths:
        if not path.exists():
            stats["missing_prediction_files"] += 1
            continue
        for row in read_jsonl(path):
            gold = row.get("gold_action") or {}
            if gold.get("action") != target_action:
                continue
            stats["target_predictions_seen"] += 1
            pred = row.get("pred_action")
            result = row.get("result") or {}
            pred_action = pred.get("action") if isinstance(pred, dict) else None
            is_failure = False
            if not result.get("valid_json") or not result.get("valid_action"):
                stats["invalid_json_or_action"] += 1
                is_failure = True
            if pred_action != target_action:
                stats[f"wrong_action::{pred_action}"] += 1
                is_failure = True
            if float(result.get("batch_claim_field_f1", 0.0) or 0.0) < low_f1_threshold:
                stats["low_claim_field_f1"] += 1
                is_failure = True
            if is_failure:
                keys.add((str(row.get("task_id", "")), int(row.get("step", 0) or 0)))
    stats["failure_keys"] = len(keys)
    return keys, dict(sorted(stats.items()))


def add_phase_hint(row: dict[str, Any], hint: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            if hint not in content:
                message["content"] = f"{hint}\n\n{content}"
            return
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", ""))
                    if hint not in text:
                        item["text"] = f"{hint}\n\n{text}"
                    return
            content.append({"type": "text", "text": hint})
            return


def prediction_paths(raw: str) -> list[Path]:
    return [Path(item.strip()) for item in str(raw or "").split(",") if item.strip()]


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("task_id", "")), int(row.get("step", 0) or 0)


def action_name(row: dict[str, Any]) -> str:
    action = row.get("action")
    return str(action.get("action", "")) if isinstance(action, dict) else ""


def action_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(action_name(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.6 Chunk Hard-Negative Patch SFT 构建报告",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## 1. 目标",
        "",
        "本数据集用于修复 v0.6 250-step adapter 在 claim 写入阶段容易继续 `open_evidence` 或提前 `finish` 的问题。",
        "做法是在 `write_claims_chunk` 样本中加入阶段提示，并对训练集写入阶段样本做适度过采样；训练集诊断中已经失败的状态会获得更高过采样权重。",
        "",
        "## 2. 数据位置",
        "",
        f"- 输入：`{manifest['input_dir']}`",
        f"- 输出：`{manifest['output_dir']}`",
        "",
        "## 3. 失败诊断",
        "",
        f"- failure prediction files：{len(manifest['failure_predictions'])}",
        f"- failure key count：{manifest['failure_key_count']}",
        "",
        "```json",
        json.dumps(manifest["failure_stats"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 4. Split 统计",
        "",
        "| split | source rows | rows written | target rows | hard rows | target written |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, stats in manifest["splits"].items():
        target_written = stats["patched_action_counts"].get(manifest["target_action"], 0)
        lines.append(
            f"| {split} | {stats['source_rows']} | {stats['rows_written']} | "
            f"{stats['target_rows_seen']} | {stats['hard_rows_seen']} | {target_written} |"
        )
    lines.extend(
        [
            "",
            "## 5. 阶段提示",
            "",
            "```text",
            manifest["phase_hint"],
            "```",
            "",
            "## 6. 注意事项",
            "",
            "- val/test 没有使用失败预测做过采样，避免把评测失败标签泄漏进训练。",
            "- 这是 SFT patch 数据，不是 on-policy RL 数据。",
            "- 后续应从 v0.6 250-step adapter 继续做小步训练，并优先评测 `write_claims_chunk` action-level 指标。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
