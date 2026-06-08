#!/usr/bin/env python3
"""Build highlighted runtime tasks for executable SFT rollout."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402


DEFAULT_DATASET = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_3_template_highlighted_sft_20260531_0504"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(dataset_dir / "tasks_all.jsonl")
    crop_targets = load_crop_targets(dataset_dir)
    runtime_tasks: list[dict[str, Any]] = []
    missing: list[str] = []

    for task in tasks:
        task_id = str(task.get("task_id"))
        target = crop_targets.get(task_id)
        if not target:
            missing.append(task_id)
            continue
        item = copy.deepcopy(task)
        gold = copy.deepcopy(item.get("gold") or {})
        gold["original_image_bbox"] = gold.get("image_bbox")
        gold["image_bbox"] = target["bbox"]
        item["gold"] = gold
        item["page_image"] = target["page_image"]
        item["runtime_source"] = "v0_3_3_highlighted_sft"
        item["runtime_mode"] = "highlighted_direct_bbox"
        item["highlighted_page_image"] = target["page_image"]
        item["corrected_image_bbox"] = target["bbox"]
        item["sft_step0_action"] = target["action"]
        runtime_tasks.append(item)

    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for item in runtime_tasks:
        by_split.setdefault(str(item.get("split")), []).append(item)

    write_jsonl(output_dir / "tasks_all.jsonl", runtime_tasks)
    for split, rows in sorted(by_split.items()):
        if rows:
            write_jsonl(output_dir / f"{split}_tasks.jsonl", rows)

    manifest = {
        "created_at": now(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "runtime_mode": "highlighted_direct_bbox",
        "tasks_total": len(tasks),
        "tasks_written": len(runtime_tasks),
        "missing_step0_crop_targets": len(missing),
        "split_counts": dict(Counter(str(item.get("split")) for item in runtime_tasks)),
        "files": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if missing:
        (output_dir / "missing_task_ids.json").write_text(json.dumps(missing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def load_crop_targets(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        path = dataset_dir / "sft" / f"{split}.jsonl"
        for row in read_jsonl(path):
            action = row.get("action") or {}
            if row.get("step") != 0 or action.get("action") != "crop_image":
                continue
            images = row.get("images") or []
            if not images:
                continue
            targets[str(row.get("task_id"))] = {
                "split": split,
                "bbox": action.get("bbox"),
                "page_image": images[0],
                "action": action,
            }
    return targets


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(f"/root/datasets/evidence_grounded_vlm_agentrl/runtime_tasks_v0_3_3_highlighted_{stamp}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
