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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evidence_agent_env.data import build_anchor_profile


DEFAULT_DATASET = "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 v1.0.4 figure anchor 表。")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--version", default="figure_anchor_table_v1_0_4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.version}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    anchors: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    figure_label_count = 0
    caption_count = 0

    for split in ("train", "val", "test"):
        path = dataset_root / f"{split}_tasks.jsonl"
        if not path.exists():
            continue
        for task in iter_jsonl(path):
            row = anchor_row(task, split)
            anchors.append(row)
            split_counts[split] += 1
            source_counts[str(row["source_file"])] += 1
            if row["figure_labels"]:
                figure_label_count += 1
            if row["caption_texts"]:
                caption_count += 1

    write_jsonl(output_dir / "anchors.jsonl", anchors)
    manifest = {
        "created_at": now(),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "anchor_rows": len(anchors),
        "split_counts": dict(split_counts),
        "source_file_count": len(source_counts),
        "rows_with_caption": caption_count,
        "rows_with_figure_label": figure_label_count,
        "artifacts": {
            "anchors": str(output_dir / "anchors.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "构建报告.md", manifest, source_counts)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def anchor_row(task: dict[str, Any], split: str) -> dict[str, Any]:
    profile = build_anchor_profile(task)
    target_region = first_target_visible_region(task)
    caption_evidence_ids = [str(item.get("evidence_id")) for item in task.get("local_evidence") or [] if item.get("evidence_id")]
    return {
        "task_id": task.get("task_id"),
        "split": split,
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "target_region_id": target_region.get("region_id"),
        "target_bbox": target_region.get("bbox"),
        "caption_evidence_ids": caption_evidence_ids,
        "caption_texts": profile["captions"],
        "figure_labels": profile["figure_labels"],
        "caption_terms": profile["caption_terms"],
        "entity_terms": profile["entity_terms"],
        "source_type": task.get("source_type"),
        "dataset_version": task.get("dataset_version"),
    }


def first_target_visible_region(task: dict[str, Any]) -> dict[str, Any]:
    regions = task.get("region_candidates") or []
    for region in regions:
        if region.get("target_region_rank") in {1, "1"} and region.get("type") == "figure_candidate":
            return region
    for region in regions:
        if region.get("type") == "figure_candidate":
            return region
    return {}


def write_report(path: Path, manifest: dict[str, Any], source_counts: Counter[str]) -> None:
    lines = [
        "# v1.0.4 Figure Anchor 表构建报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- 源数据集：`{manifest['dataset_root']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        f"- anchor rows：{manifest['anchor_rows']}",
        f"- source files：{manifest['source_file_count']}",
        f"- 有 caption 的行：{manifest['rows_with_caption']}",
        f"- 有 figure label 的行：{manifest['rows_with_figure_label']}",
        "",
        "## Split 分布",
        "",
    ]
    for key, value in sorted(manifest["split_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 文档分布 Top 20", ""])
    for source_file, count in source_counts.most_common(20):
        lines.append(f"- `{source_file}`: {count}")
    lines.extend(["", "## 产物", ""])
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


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
