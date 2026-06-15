#!/usr/bin/env python3
"""Audit possible bbox coordinate-mode mistakes in v1.3.1 annotations.

The remote VLM prompt asks for norm1000 boxes, but some raw responses look like
page pixel coordinates. This script does a non-destructive val/test audit and
creates before/after overlays for suspicious targets.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_DATASET = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v1_3_1_remote_vlm_evidence_sft_20260614_1335"
)
DEFAULT_OUT = Path(
    "/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/"
    "docs/03_实验报告/assets_20260615_1240/bbox_coord_audit_val_test"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-per-split", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--require-caption-text-match",
        action="store_true",
        help="Only flag cases where raw-px caption bbox covers caption text blocks better than current norm bbox.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw or "{}")
    except Exception:
        match = re.search(r"\{.*\}", raw or "", re.S)
        if not match:
            return {}
        try:
            return json.loads(re.sub(r",\s*([}\]])", r"\1", match.group(0)))
        except Exception:
            return {}


def bbox_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        out = [float(v) for v in value]
    except Exception:
        return None
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def clamp_px(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        max(0, min(width, round(x1))),
        max(0, min(height, round(y1))),
        max(0, min(width, round(x2))),
        max(0, min(height, round(y2))),
    ]


def norm_to_px(box: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = box
    return [
        max(0, min(width, round(x1 * width / 1000))),
        max(0, min(height, round(y1 * height / 1000))),
        max(0, min(width, round(x2 * width / 1000))),
        max(0, min(height, round(y2 * height / 1000))),
    ]


def valid_px(box: list[int] | None) -> bool:
    return bool(box and len(box) == 4 and box[2] > box[0] and box[3] > box[1])


def area_ratio(box: list[int] | None, width: int, height: int) -> float:
    if not valid_px(box):
        return 0.0
    return ((box[2] - box[0]) * (box[3] - box[1])) / max(1, width * height)


def iou(a: list[int] | None, b: list[int] | None) -> float:
    if not valid_px(a) or not valid_px(b):
        return 0.0
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(1, area_a + area_b - inter)


def coverage(box: list[int] | None, text_box: list[int] | None) -> float:
    if not valid_px(box) or not valid_px(text_box):
        return 0.0
    ix1 = max(box[0], text_box[0])
    iy1 = max(box[1], text_box[1])
    ix2 = min(box[2], text_box[2])
    iy2 = min(box[3], text_box[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    text_area = (text_box[2] - text_box[0]) * (text_box[3] - text_box[1])
    return inter / max(1, text_area)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def text_match_score(caption: str, block_text: str) -> float:
    cap = compact_text(caption)
    block = compact_text(block_text)
    if not cap or not block:
        return 0.0
    if block in cap or cap in block:
        return min(1.0, len(block) / max(1, len(cap)))
    # Character overlap is coarse, but robust for OCR/PDF spacing differences.
    cap_chars = set(cap)
    block_chars = set(block)
    return len(cap_chars & block_chars) / max(1, len(cap_chars | block_chars))


def caption_box_text_score(box: list[int] | None, caption_text: str, text_blocks: list[dict[str, Any]]) -> float:
    best = 0.0
    for block in text_blocks:
        text = str(block.get("text") or "")
        match = text_match_score(caption_text, text)
        if match < 0.25:
            continue
        b = block.get("bbox")
        if not isinstance(b, list) or len(b) != 4:
            continue
        try:
            text_box = [round(float(v)) for v in b]
        except Exception:
            continue
        spatial = max(iou(box, text_box), coverage(box, text_box))
        best = max(best, match * spatial)
    return round(best, 4)


def raw_can_be_px(box: list[float] | None, width: int, height: int) -> bool:
    if not box:
        return False
    return 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height


def is_suspicious(target_raw: list[float] | None, caption_raw: list[float] | None, width: int, height: int) -> tuple[bool, dict[str, Any]]:
    target_norm_px = norm_to_px(target_raw, width, height) if target_raw else None
    caption_norm_px = norm_to_px(caption_raw, width, height) if caption_raw else None
    target_raw_px = clamp_px(target_raw, width, height) if target_raw and raw_can_be_px(target_raw, width, height) else None
    caption_raw_px = clamp_px(caption_raw, width, height) if caption_raw and raw_can_be_px(caption_raw, width, height) else None
    norm_area = area_ratio(target_norm_px, width, height)
    raw_area = area_ratio(target_raw_px, width, height)
    cap_norm_area = area_ratio(caption_norm_px, width, height)
    cap_raw_area = area_ratio(caption_raw_px, width, height)

    reasons = []
    if target_raw_px and norm_area > 0.35 and raw_area > 0.01 and norm_area > raw_area * 1.45:
        reasons.append("target_norm_area_much_larger_than_raw_px")
    if caption_raw_px and cap_raw_area > 0 and cap_norm_area > cap_raw_area * 1.45 and cap_norm_area > 0.01:
        reasons.append("caption_norm_area_much_larger_than_raw_px")
    if target_raw_px and caption_raw_px and target_raw_px[1] >= caption_raw_px[3] and target_norm_px and caption_norm_px:
        # raw-px interpretation has common caption-above-target geometry.
        if not (target_norm_px[1] >= caption_norm_px[3]):
            reasons.append("raw_px_geometry_more_plausible")
    return bool(reasons), {
        "target_norm_px": target_norm_px,
        "caption_norm_px": caption_norm_px,
        "target_raw_px": target_raw_px,
        "caption_raw_px": caption_raw_px,
        "target_norm_area": round(norm_area, 4),
        "target_raw_px_area": round(raw_area, 4),
        "caption_norm_area": round(cap_norm_area, 4),
        "caption_raw_px_area": round(cap_raw_area, 4),
        "reasons": reasons,
    }


def text_match_prefer_raw_px(
    *,
    target_raw: list[float] | None,
    caption_raw: list[float] | None,
    width: int,
    height: int,
    caption_text: str,
    text_blocks: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    target_norm_px = norm_to_px(target_raw, width, height) if target_raw else None
    caption_norm_px = norm_to_px(caption_raw, width, height) if caption_raw else None
    target_raw_px = clamp_px(target_raw, width, height) if target_raw and raw_can_be_px(target_raw, width, height) else None
    caption_raw_px = clamp_px(caption_raw, width, height) if caption_raw and raw_can_be_px(caption_raw, width, height) else None
    norm_score = caption_box_text_score(caption_norm_px, caption_text, text_blocks)
    raw_score = caption_box_text_score(caption_raw_px, caption_text, text_blocks)
    target_raw_area = area_ratio(target_raw_px, width, height)
    target_norm_area = area_ratio(target_norm_px, width, height)
    flag = bool(caption_raw_px and raw_score >= 0.25 and raw_score > max(0.15, norm_score * 1.5))
    reasons = []
    if flag:
        reasons.append("raw_px_caption_matches_caption_text_block")
    if flag and target_norm_area > target_raw_area * 1.25:
        reasons.append("target_norm_area_larger_than_raw_px")
    return flag, {
        "target_norm_px": target_norm_px,
        "caption_norm_px": caption_norm_px,
        "target_raw_px": target_raw_px,
        "caption_raw_px": caption_raw_px,
        "target_norm_area": round(target_norm_area, 4),
        "target_raw_px_area": round(target_raw_area, 4),
        "caption_norm_text_score": norm_score,
        "caption_raw_px_text_score": raw_score,
        "reasons": reasons,
    }


def draw_overlay(page_image: Path, target_box: list[int] | None, caption_box: list[int] | None, out: Path) -> None:
    with Image.open(page_image).convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        if valid_px(target_box):
            draw.rectangle(target_box, outline=(220, 0, 0), width=5)
        if valid_px(caption_box):
            draw.rectangle(caption_box, outline=(0, 185, 210), width=5)
        image.save(out, quality=92)


def crop(page_image: Path, box: list[int] | None, out: Path) -> None:
    if not valid_px(box):
        return
    with Image.open(page_image).convert("RGB") as image:
        image.crop(tuple(box)).save(out, quality=92)


def side_by_side(left: Path, right: Path, out: Path) -> None:
    with Image.open(left).convert("RGB") as a, Image.open(right).convert("RGB") as b:
        h = max(a.height, b.height)
        aw = round(a.width * h / a.height)
        bw = round(b.width * h / b.height)
        a2 = a.resize((aw, h))
        b2 = b.resize((bw, h))
        canvas = Image.new("RGB", (aw + bw, h), "white")
        canvas.paste(a2, (0, 0))
        canvas.paste(b2, (aw, 0))
        canvas.save(out, quality=92)


def split_task_ids(dataset: Path, split: str) -> set[str]:
    ids = set()
    for row in read_jsonl(dataset / "sft" / f"{split}.jsonl"):
        if row.get("task_id"):
            ids.add(str(row["task_id"]))
    return ids


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ann_by_page = {}
    for row in read_jsonl(args.dataset / "page_level_vlm_annotations.jsonl"):
        raw = parse_json(row.get("raw_response") or "")
        ann_by_page[str(row.get("page_id"))] = {"row": row, "raw_detections": raw.get("detections") or []}
    text_blocks_by_page: dict[str, list[dict[str, Any]]] = {}
    for block in read_jsonl(args.dataset / "page_text_blocks.jsonl"):
        text_blocks_by_page.setdefault(str(block.get("page_id")), []).append(block)

    target_rows = read_jsonl(args.dataset / "figure_targets.jsonl")
    split_ids = {split: split_task_ids(args.dataset, split) for split in ["val", "test"]}
    suspicious: list[dict[str, Any]] = []

    for target in target_rows:
        tid = str(target.get("target_id"))
        split = next((name for name, ids in split_ids.items() if tid in ids), "")
        if not split:
            continue
        page_id = str(target.get("page_id"))
        page_info = ann_by_page.get(page_id)
        if not page_info:
            continue
        dets = page_info["row"].get("detections") or []
        raw_dets = page_info["raw_detections"]
        match_idx = None
        for idx, det in enumerate(dets):
            if det.get("target_bbox_norm1000") == target.get("target_bbox_norm1000") and (
                (det.get("caption_text") or "") == ((target.get("base_fields") or {}).get("caption_text") or "")
            ):
                match_idx = idx
                break
        if match_idx is None:
            continue
        raw_det = raw_dets[match_idx] if match_idx < len(raw_dets) and isinstance(raw_dets[match_idx], dict) else dets[match_idx]
        width = int(target.get("width") or page_info["row"].get("width") or 0)
        height = int(target.get("height") or page_info["row"].get("height") or 0)
        target_raw = bbox_list(raw_det.get("target_bbox_norm1000"))
        caption_raw = bbox_list(raw_det.get("caption_bbox_norm1000"))
        caption_text = (target.get("base_fields") or {}).get("caption_text") or ""
        if args.require_caption_text_match:
            flag, detail = text_match_prefer_raw_px(
                target_raw=target_raw,
                caption_raw=caption_raw,
                width=width,
                height=height,
                caption_text=caption_text,
                text_blocks=text_blocks_by_page.get(page_id, []),
            )
        else:
            flag, detail = is_suspicious(target_raw, caption_raw, width, height)
        if not flag:
            continue
        suspicious.append(
            {
                "split": split,
                "target_id": tid,
                "page_id": page_id,
                "source_file": target.get("source_file"),
                "page_num": target.get("page_num"),
                "page_image": target.get("page_image"),
                "caption_text": caption_text,
                "raw_target_bbox": target_raw,
                "raw_caption_bbox": caption_raw,
                "current_target_bbox_px": target.get("target_bbox_px"),
                "current_caption_bbox_px": target.get("caption_bbox_px"),
                **detail,
            }
        )

    sampled: list[dict[str, Any]] = []
    for split in ["val", "test"]:
        rows = [row for row in suspicious if row["split"] == split]
        rows.sort(key=lambda row: (-row["target_norm_area"], row["target_id"]))
        sampled.extend(rows[: args.max_per_split])

    for idx, row in enumerate(sampled, start=1):
        stem = f"{idx:02d}_{row['split']}_{row['target_id']}"
        before = args.out_dir / f"{stem}_before_overlay.jpg"
        after = args.out_dir / f"{stem}_after_rawpx_overlay.jpg"
        compare = args.out_dir / f"{stem}_compare_overlay.jpg"
        page = Path(row["page_image"])
        draw_overlay(page, row["current_target_bbox_px"], row["current_caption_bbox_px"], before)
        draw_overlay(page, row["target_raw_px"], row["caption_raw_px"], after)
        side_by_side(before, after, compare)
        crop(page, row["current_target_bbox_px"], args.out_dir / f"{stem}_before_target.jpg")
        crop(page, row["target_raw_px"], args.out_dir / f"{stem}_after_target.jpg")
        crop(page, row["current_caption_bbox_px"], args.out_dir / f"{stem}_before_caption.jpg")
        crop(page, row["caption_raw_px"], args.out_dir / f"{stem}_after_caption.jpg")
        row["asset_stem"] = stem
        row["compare_overlay"] = str(compare)

    summary = {
        "dataset": str(args.dataset),
        "strategy": "caption_text_match" if args.require_caption_text_match else "area_heuristic",
        "splits": {split: len(ids) for split, ids in split_ids.items()},
        "suspicious_counts": {
            split: sum(1 for row in suspicious if row["split"] == split) for split in ["val", "test"]
        },
        "sampled": len(sampled),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.out_dir / "suspicious_val_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in suspicious:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out_dir / "sampled_val_test.jsonl").open("w", encoding="utf-8") as handle:
        for row in sampled:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
