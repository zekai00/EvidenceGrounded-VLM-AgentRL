#!/usr/bin/env python3
"""Add deterministic phase-aware tool masks to stepwise SFT rows.

The v1.0.x datasets already store a staged oracle action sequence, but some
rows do not expose `available_actions` to the model. This script creates a
copy of an AgentBench dataset and annotates each SFT row with the legal tool(s)
for that phase. It does not change images, target boxes, captions, evidence,
or the supervised action payload.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PHASE_ACTIONS: dict[str, list[str]] = {
    "inspect_page": ["inspect_page"],
    "propose_regions": ["propose_regions"],
    "crop_region": ["crop_region"],
    "crop_target": ["crop_target"],
    "crop_image": ["crop_image"],
    "select_evidence": ["select_evidence"],
    "open_evidence": ["open_evidence"],
    "retrieve_evidence": ["retrieve_evidence"],
    "write_claim": ["write_claim"],
    "abstain_claim": ["abstain_claim"],
    "write_claims_chunk": ["write_claims_chunk"],
    "write_claims_batch": ["write_claims_batch"],
    "finish": ["finish"],
}

PHASE_NAMES: dict[str, str] = {
    "inspect_page": "inspect_layout",
    "propose_regions": "propose_candidate_regions",
    "crop_region": "crop_selected_region",
    "crop_target": "crop_target_region",
    "crop_image": "crop_free_bbox",
    "select_evidence": "select_existing_local_evidence",
    "open_evidence": "open_selected_or_retrieved_evidence",
    "retrieve_evidence": "retrieve_additional_text_evidence",
    "write_claim": "write_single_claim",
    "abstain_claim": "abstain_single_claim",
    "write_claims_chunk": "write_one_claim_chunk",
    "write_claims_batch": "write_claim_batch",
    "finish": "finish_when_claims_complete",
}

PHASE_HINTS: dict[str, str] = {
    "inspect_page": "先读取页面布局候选，不要直接裁剪、检索或写 claim。",
    "crop_target": "从 inspect_page 返回的目标候选区域中裁剪目标图像。",
    "crop_region": "从 propose_regions 返回的候选区域中裁剪目标图像。",
    "select_evidence": "从当前已出现的本地图注或候选证据里登记 evidence_id，不要发起新的 retrieve_evidence。",
    "open_evidence": "打开已经选中或刚检索到的 evidence_id，读取证据文本。",
    "retrieve_evidence": "基于目标图像和图注构造检索 query，补充同文档或语料库证据。",
    "write_claims_chunk": "一次只写入或 abstain 一个 remaining_fields 中的字段。",
    "finish": "只有 remaining_fields 为空时才能结束。",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    if not input_root.exists():
        raise FileNotFoundError(input_root)
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)

    shutil.copytree(input_root, output_root, ignore=shutil.ignore_patterns("sft"))
    (output_root / "sft").mkdir(parents=True, exist_ok=True)

    totals: dict[str, Any] = {
        "created_at": now(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "splits": {},
        "phase_action_policy": PHASE_ACTIONS,
        "note": "Only SFT rows are annotated. Tasks, page images, crops, bboxes, captions, and evidence payloads are copied unchanged.",
    }
    for split in ["train", "val", "test"]:
        src = input_root / "sft" / f"{split}.jsonl"
        dst = output_root / "sft" / f"{split}.jsonl"
        if not src.exists():
            continue
        stats = annotate_file(src, dst)
        totals["splits"][split] = stats

    (output_root / "phase_mask_summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))
    return 0


def annotate_file(src: Path, dst: Path) -> dict[str, Any]:
    action_counts: Counter[str] = Counter()
    available_counts: Counter[str] = Counter()
    rows = 0
    with src.open("r", encoding="utf-8") as f_in, dst.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            row = json.loads(line)
            action = row.get("action") if isinstance(row.get("action"), dict) else {}
            action_type = str(action.get("action") or "")
            available = PHASE_ACTIONS.get(action_type, [action_type] if action_type else [])
            row["available_actions"] = available
            row["phase_name"] = PHASE_NAMES.get(action_type, action_type)
            row["phase_hint"] = PHASE_HINTS.get(action_type, "")
            action_counts[action_type] += 1
            available_counts[",".join(available)] += 1
            rows += 1
            f_out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {
        "rows": rows,
        "action_counts": dict(action_counts),
        "available_action_counts": dict(available_counts),
    }


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
