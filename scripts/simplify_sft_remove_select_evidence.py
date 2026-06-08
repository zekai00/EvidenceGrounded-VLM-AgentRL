#!/usr/bin/env python3
"""Create a simplified SFT dataset without the redundant select_evidence step.

In v1.0.2 trajectories, `select_evidence` is usually a bookkeeping action that
registers the local caption evidence id before immediately opening it. The
current 3B policy consistently confuses this phase with retrieve_evidence.
This script keeps the evidence id visible in state, removes the extra action
from history/tool-results, and trains/evaluates the more natural sequence:

inspect_page -> crop_target -> open_evidence(local_caption) -> retrieve_evidence
-> open_evidence(retrieved evidence) -> write_claims_chunk -> finish
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE_ACTIONS = {
    "inspect_page": ["inspect_page"],
    "crop_target": ["crop_target"],
    "open_evidence": ["open_evidence"],
    "retrieve_evidence": ["retrieve_evidence"],
    "write_claims_chunk": ["write_claims_chunk"],
    "finish": ["finish"],
}

PHASE_HINTS = {
    "inspect_page": "先读取页面布局候选，不要直接裁剪、检索或写 claim。",
    "crop_target": "从 inspect_page 返回的目标候选区域中裁剪目标图像。",
    "open_evidence": "打开当前页面中可见的 local_caption evidence_id，或打开上一步检索返回的 evidence_id。",
    "retrieve_evidence": "基于目标图像和图注构造检索 query，补充同文档或语料库证据。",
    "write_claims_chunk": "一次只写入或 abstain 一个 remaining_fields 中的字段。",
    "finish": "只有 remaining_fields 为空时才能结束。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(output_root)
        shutil.rmtree(output_root)
    shutil.copytree(input_root, output_root, ignore=shutil.ignore_patterns("sft"))
    (output_root / "sft").mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "created_at": now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "policy": "remove select_evidence rows and remove select_evidence from history/tool_results",
        "splits": {},
    }
    for split in ["train", "val", "test"]:
        src = input_root / "sft" / f"{split}.jsonl"
        dst = output_root / "sft" / f"{split}.jsonl"
        if not src.exists():
            continue
        summary["splits"][split] = simplify_file(src, dst)
    (output_root / "no_select_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def simplify_file(src: Path, dst: Path) -> dict[str, Any]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_actions: Counter[str] = Counter()
    output_actions: Counter[str] = Counter()
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id") or "")
            action_type = get_action_type(row)
            input_actions[action_type] += 1
            if action_type == "select_evidence":
                continue
            rows_by_task[task_id].append(clean_row(row))

    output_rows = 0
    with dst.open("w", encoding="utf-8") as f:
        for task_id in sorted(rows_by_task):
            rows = rows_by_task[task_id]
            for new_step, row in enumerate(rows):
                row["step"] = new_step
                action_type = get_action_type(row)
                row["available_actions"] = PHASE_ACTIONS.get(action_type, [action_type] if action_type else [])
                row["phase_name"] = f"no_select_{action_type}"
                row["phase_hint"] = PHASE_HINTS.get(action_type, "")
                output_actions[action_type] += 1
                output_rows += 1
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    return {
        "input_rows": sum(input_actions.values()),
        "output_rows": output_rows,
        "removed_select_evidence_rows": input_actions.get("select_evidence", 0),
        "input_action_counts": dict(input_actions),
        "output_action_counts": dict(output_actions),
        "tasks": len(rows_by_task),
    }


def clean_row(row: dict[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row["history"] = [
        item for item in (row.get("history") or []) if isinstance(item, dict) and item.get("action") != "select_evidence"
    ]
    row["tool_results"] = [
        item for item in (row.get("tool_results") or []) if isinstance(item, dict) and item.get("tool") != "select_evidence"
    ]
    return row


def get_action_type(row: dict[str, Any]) -> str:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    return str(action.get("action") or "")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
