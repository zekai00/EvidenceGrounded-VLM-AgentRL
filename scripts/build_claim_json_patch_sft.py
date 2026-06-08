#!/usr/bin/env python3
"""Build a write_claims_chunk-focused SFT patch set.

The stepwise GRPO smoke after environment masking reached the right crop/retrieve
phase, but still produced malformed or empty JSON around write_claims_chunk.
This builder keeps enough replay for earlier tools while making claim writing
the dominant supervised action.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TRAIN_TARGET_COUNTS = {
    "write_claims_chunk": 704,
    "retrieve_evidence": 96,
    "open_evidence": 80,
    "crop_region": 40,
    "propose_regions": 40,
    "select_evidence": 32,
    "finish": 32,
}
VAL_TARGET_COUNTS = {
    "write_claims_chunk": 80,
    "retrieve_evidence": 16,
    "open_evidence": 12,
    "crop_region": 6,
    "propose_regions": 6,
    "select_evidence": 4,
    "finish": 4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-sft-dir",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_chunked_claim_sft_20260604_1650/sft",
    )
    parser.add_argument(
        "--phase-patch-dir",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_evidence_phase_patch_v2_sft_20260605_1141",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Default: /root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_claim_json_patch_sft_<timestamp>",
    )
    parser.add_argument("--seed", type=int, default=60605)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_claim_json_patch_sft_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)

    sources = {
        "main_sft": Path(args.main_sft_dir),
        "phase_patch": Path(args.phase_patch_dir),
    }
    train_pool = collect_rows(sources, "train")
    val_pool = collect_rows(sources, "val")
    train_rows = sample_by_action(train_pool, TRAIN_TARGET_COUNTS, args.seed)
    val_rows = sample_by_action(val_pool, VAL_TARGET_COUNTS, args.seed + 17)

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "SFT patch for malformed/empty write_claims_chunk outputs observed in stepwise executable rollout.",
        "sources": {name: str(path) for name, path in sources.items()},
        "seed": args.seed,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_target_counts": TRAIN_TARGET_COUNTS,
        "val_target_counts": VAL_TARGET_COUNTS,
        "train_action_distribution": dict(action_counter(train_rows)),
        "val_action_distribution": dict(action_counter(val_rows)),
        "notes": [
            "Rows preserve original messages and images.",
            "Train set intentionally makes write_claims_chunk about two thirds of the update stream.",
            "Use prompt_mode=original and sample_strategy=first/random when training so the intended action mix is preserved.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0


def collect_rows(sources: dict[str, Path], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source_name, source_dir in sources.items():
        path = source_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            action = row_action(row)
            if not action:
                continue
            key = stable_row_key(row)
            if key in seen:
                continue
            seen.add(key)
            copied = json.loads(json.dumps(row, ensure_ascii=False))
            copied["claim_json_patch_source"] = source_name
            rows.append(copied)
    return rows


def stable_row_key(row: dict[str, Any]) -> str:
    action = row.get("action") or {}
    return "|".join(
        [
            str(row.get("task_id", "")),
            str(row.get("step", "")),
            str(row.get("label_source", "")),
            str(row.get("claim_json_patch_source", "")),
            json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))[:256],
        ]
    )


def sample_by_action(rows: list[dict[str, Any]], target_counts: dict[str, int], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row_action(row)].append(row)
    selected: list[dict[str, Any]] = []
    for action, count in target_counts.items():
        bucket = list(buckets.get(action, []))
        if not bucket:
            continue
        rng.shuffle(bucket)
        if len(bucket) >= count:
            selected.extend(bucket[:count])
        else:
            selected.extend(bucket)
            selected.extend(rng.choices(bucket, k=count - len(bucket)))
    rng.shuffle(selected)
    return selected


def row_action(row: dict[str, Any]) -> str:
    action = row.get("action") or {}
    if isinstance(action, dict):
        return str(action.get("action") or "")
    return ""


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(row_action(row) for row in rows)


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
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
