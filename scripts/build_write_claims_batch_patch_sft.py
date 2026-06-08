#!/usr/bin/env python3
"""Build a conservative write_claims_batch patch SFT dataset.

The patch dataset keeps the original EvidenceGrounded v0.5 rows unchanged, but
oversamples rows whose gold action is write_claims_batch. This preserves replay
coverage for all other tools while increasing exposure to the weak final claim
writing action.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_5_evidence_selection_sft_20260601_1839/sft"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-action", default="write_claims_batch")
    parser.add_argument("--train-oversample", type=int, default=5)
    parser.add_argument("--eval-oversample", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": now(),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "target_action": args.target_action,
        "train_oversample": args.train_oversample,
        "eval_oversample": args.eval_oversample,
        "seed": args.seed,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        rows = read_jsonl(args.input_dir / f"{split}.jsonl")
        oversample = args.train_oversample if split == "train" else args.eval_oversample
        patched = patch_rows(rows, args.target_action, oversample)
        random.shuffle(patched)
        out_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(out_path, patched)

        original_counts = action_counts(rows)
        patched_counts = action_counts(patched)
        manifest["splits"][split] = {
            "source_rows": len(rows),
            "rows_written": len(patched),
            "oversample": oversample,
            "source_action_counts": dict(sorted(original_counts.items())),
            "patched_action_counts": dict(sorted(patched_counts.items())),
            "target_action_source_rows": original_counts.get(args.target_action, 0),
            "target_action_written_rows": patched_counts.get(args.target_action, 0),
            "target_action_fraction": patched_counts.get(args.target_action, 0) / max(1, len(patched)),
            "file": str(out_path),
        }
        print(f"[{split}] {len(rows)} -> {len(patched)} rows; {args.target_action}={patched_counts.get(args.target_action, 0)}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return 0


def patch_rows(rows: list[dict[str, Any]], target_action: str, oversample: int) -> list[dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    for row in rows:
        count = oversample if action_name(row) == target_action else 1
        for copy_index in range(max(1, count)):
            copied = json.loads(json.dumps(row, ensure_ascii=False))
            copied["patch_source"] = {
                "kind": "write_claims_batch_oversample",
                "copy_index": copy_index,
                "oversample": count,
            }
            patched.append(copied)
    return patched


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


def action_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(action_name(row) for row in rows)


def action_name(row: dict[str, Any]) -> str:
    action = row.get("action")
    return str(action.get("action", "")) if isinstance(action, dict) else ""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
