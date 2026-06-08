#!/usr/bin/env python3
"""Convert EvidenceGrounded v0.5 SFT JSONL rows to verl multi-turn SFT parquet.

verl 0.8.0 uses MultiTurnSFTDataset for SFT. The dataset expects a parquet
column named `messages`; for multimodal rows, message text contains `<image>`
placeholders and the real image payloads are stored in the `images` column.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_INPUT_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_5_evidence_selection_sft_20260601_1839/sft"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split names.")
    parser.add_argument("--max-rows-per-split", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--preview-rows", type=int, default=3)
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=0,
        help="Optional max_pixels added to each image dict for qwen-vl-utils resizing.",
    )
    parser.add_argument(
        "--image-min-pixels",
        type=int,
        default=0,
        help="Optional min_pixels added to each image dict for qwen-vl-utils resizing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    manifest: dict[str, Any] = {
        "created_at": now(),
        "source_format": "EvidenceGrounded v0.5 step-level SFT JSONL",
        "target_format": "verl 0.8.0 MultiTurnSFTDataset parquet",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "messages_key": "messages",
        "image_key": "images",
        "image_placeholder": "<image>",
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "splits": {},
    }

    for split in splits:
        src = args.input_dir / f"{split}.jsonl"
        if not src.exists():
            raise FileNotFoundError(src)

        rows = []
        action_counts: Counter[str] = Counter()
        image_count_counts: Counter[int] = Counter()
        source_count = 0
        for row_index, raw in enumerate(read_jsonl(src)):
            if args.max_rows_per_split and len(rows) >= args.max_rows_per_split:
                break
            source_count += 1
            converted = convert_row(
                raw,
                row_index=row_index,
                image_max_pixels=args.image_max_pixels,
                image_min_pixels=args.image_min_pixels,
            )
            rows.append(converted)
            action_counts[converted["action_name"]] += 1
            image_count_counts[len(converted["images"])] += 1

        df = pd.DataFrame(rows)
        out_parquet = args.output_dir / f"{split}.parquet"
        df.to_parquet(out_parquet, index=False)

        preview_path = args.output_dir / f"{split}_preview.jsonl"
        with preview_path.open("w", encoding="utf-8") as f:
            for row in rows[: args.preview_rows]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest["splits"][split] = {
            "source_jsonl": str(src),
            "output_parquet": str(out_parquet),
            "preview_jsonl": str(preview_path),
            "source_rows_seen": source_count,
            "rows_written": len(rows),
            "action_counts": dict(sorted(action_counts.items())),
            "image_count_distribution": {str(k): v for k, v in sorted(image_count_counts.items())},
            "columns": list(df.columns),
        }
        print(f"[{split}] wrote {len(rows)} rows -> {out_parquet}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return 0


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_row(
    row: dict[str, Any],
    row_index: int,
    image_max_pixels: int = 0,
    image_min_pixels: int = 0,
) -> dict[str, Any]:
    messages, images = convert_messages(
        row.get("messages", []),
        fallback_images=row.get("images", []),
        image_max_pixels=image_max_pixels,
        image_min_pixels=image_min_pixels,
    )
    action = row.get("action", {})
    action_name = action.get("action", "") if isinstance(action, dict) else ""
    response = find_last_assistant_text(messages)
    prompt = find_first_user_text(messages)

    return {
        "data_source": "evidence_grounded_vlm_agentrl_v0_5",
        "messages": messages,
        "images": images,
        "prompt": prompt,
        "response": response,
        "task_id": row.get("task_id", ""),
        "source_task_id": row.get("source_task_id", ""),
        "split": row.get("split", ""),
        "variant": int(row.get("variant", 0) or 0),
        "step": int(row.get("step", 0) or 0),
        "tool_schema_version": row.get("tool_schema_version", ""),
        "label_source": row.get("label_source", ""),
        "action_name": action_name,
        "action_json": json.dumps(action, ensure_ascii=False, separators=(",", ":")),
        "row_index": row_index,
        "extra_info": {
            "history_len": len(row.get("history", []) or []),
            "tool_results_len": len(row.get("tool_results", []) or []),
            "draft_claims_len": len(row.get("draft_claims", []) or []),
            "selected_evidence_ids": list(row.get("selected_evidence_ids", []) or []),
            "original_images": list(row.get("images", []) or []),
        },
    }


def convert_messages(
    messages: list[dict[str, Any]],
    fallback_images: list[str],
    image_max_pixels: int = 0,
    image_min_pixels: int = 0,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    converted: list[dict[str, str]] = []
    images: list[dict[str, str]] = []

    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        converted_content, new_images = convert_content(
            content,
            image_max_pixels=image_max_pixels,
            image_min_pixels=image_min_pixels,
        )
        images.extend(new_images)
        converted.append({"role": role, "content": converted_content})

    if not images and fallback_images:
        images = [
            make_image_payload(str(path), image_max_pixels=image_max_pixels, image_min_pixels=image_min_pixels)
            for path in fallback_images
        ]
        for message in converted:
            if message["role"] == "user":
                message["content"] = "\n".join(["<image>"] * len(images)) + "\n" + message["content"]
                break

    return converted, images


def convert_content(
    content: Any,
    image_max_pixels: int = 0,
    image_min_pixels: int = 0,
) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(content, str):
        return content, []

    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False), []

    pieces: list[str] = []
    images: list[dict[str, Any]] = []
    for item in content:
        if not isinstance(item, dict):
            pieces.append(str(item))
            continue

        item_type = item.get("type")
        if item_type == "image":
            image_path = item.get("image") or item.get("path") or item.get("url")
            if image_path:
                images.append(
                    make_image_payload(
                        str(image_path),
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                    )
                )
                pieces.append("<image>")
        elif item_type == "image_url":
            image_url = item.get("image_url")
            if isinstance(image_url, dict):
                image_url = image_url.get("url")
            if image_url:
                images.append(
                    make_image_payload(
                        str(image_url),
                        image_max_pixels=image_max_pixels,
                        image_min_pixels=image_min_pixels,
                    )
                )
                pieces.append("<image>")
        elif item_type == "text":
            pieces.append(str(item.get("text", "")))
        else:
            pieces.append(json.dumps(item, ensure_ascii=False))

    return "\n".join(piece for piece in pieces if piece), images


def make_image_payload(image: str, image_max_pixels: int = 0, image_min_pixels: int = 0) -> dict[str, Any]:
    payload: dict[str, Any] = {"image": image}
    if image_max_pixels > 0:
        payload["max_pixels"] = int(image_max_pixels)
    if image_min_pixels > 0:
        payload["min_pixels"] = int(image_min_pixels)
    return payload


def find_first_user_text(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return message.get("content", "")
    return messages[0].get("content", "") if messages else ""


def find_last_assistant_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
