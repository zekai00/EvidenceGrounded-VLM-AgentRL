#!/usr/bin/env python3
"""Build a highlighted-page copy of EvidenceGrounded SFT JSONL files.

The original v0.3.1 SFT rows know the target crop bbox, but many page images do
not visibly mark the target figure. That makes the first `crop_image` action
under-specified for a VLM. This script draws a red rectangle on the page image
for each task and rewrites image references to the highlighted page.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248",
    )
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--version", default="agentbench_v0_3_3_template_highlighted_sft")
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--line-width", type=int, default=8)
    parser.add_argument("--bbox-source", choices=["template", "action"], default="template")
    parser.add_argument("--min-template-score", type=float, default=0.45)
    parser.add_argument("--match-max-dim", type=int, default=900)
    parser.add_argument("--match-scale-steps", type=int, default=25)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_root) / f"{args.version}_{stamp}"
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(output_dir)
    (output_dir / "sft").mkdir(parents=True, exist_ok=True)
    highlighted_dir = output_dir / "highlighted_pages"
    highlighted_dir.mkdir(parents=True, exist_ok=True)

    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    all_rows_by_split = {split: read_jsonl(input_dir / "sft" / f"{split}.jsonl") for split in splits}
    task_targets = collect_task_targets(all_rows_by_split, args)

    image_cache: dict[tuple[str, str], str] = {}
    split_stats: dict[str, Any] = {}
    for split, rows in all_rows_by_split.items():
        out_rows = []
        replaced = 0
        missing_target = 0
        for row in rows:
            target = task_targets.get(str(row.get("task_id")))
            if not target:
                missing_target += 1
                out_rows.append(row)
                continue
            new_row = json.loads(json.dumps(row, ensure_ascii=False))
            rewrite_bboxes(new_row, target)
            images = list(new_row.get("images") or [])
            if images:
                old_page = images[0]
                cache_key = (str(new_row.get("task_id")), old_page)
                if cache_key not in image_cache:
                    image_cache[cache_key] = make_highlighted_page(
                        Path(old_page),
                        target["page_bbox"],
                        highlighted_dir,
                        str(new_row.get("task_id")),
                        args.line_width,
                    )
                images[0] = image_cache[cache_key]
                new_row["images"] = images
                rewrite_message_images(new_row, old_page, images[0])
                replaced += 1
            out_rows.append(new_row)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", out_rows)
        split_stats[split] = {
            "rows": len(rows),
            "rows_with_highlighted_page": replaced,
            "rows_missing_task_target": missing_target,
            "action_distribution": dict(Counter(str((row.get("action") or {}).get("action", "unknown")) for row in rows)),
        }

    copy_sidecar_files(input_dir, output_dir)
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "version": args.version,
        "task_targets": len(task_targets),
        "highlighted_page_files": len(image_cache),
        "splits": split_stats,
        "template_score": summarize_template_scores(task_targets),
        "note": "Page image[0] is rewritten to a red-rectangle highlighted page. In template mode, step-0 crop bbox is corrected by matching the crop image back to the page PNG.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps({"builder": Path(__file__).name, **summary}, ensure_ascii=False, indent=2), encoding="utf-8")
    report = [
        "# EvidenceGrounded v0.3.2 Highlighted SFT 构建报告",
        "",
        f"- 构建时间：{summary['created_at']}",
        f"- 输入：`{input_dir}`",
        f"- 输出：`{output_dir}`",
        f"- 任务目标框：{len(task_targets)}",
        f"- 高亮页面文件：{len(image_cache)}",
        "",
        "## Split 统计",
        "",
        "| split | rows | highlighted_rows | missing_target |",
        "|---|---:|---:|---:|",
    ]
    for split, stat in split_stats.items():
        report.append(f"| {split} | {stat['rows']} | {stat['rows_with_highlighted_page']} | {stat['rows_missing_task_target']} |")
    report.extend(
        [
            "",
            "## 说明",
            "",
            "v0.3.1 的 step-0 `crop_image` 监督目标并不稳定等同于当前 page PNG 像素坐标。",
            "本版本先用 crop 图在 page PNG 上做模板匹配，反推出真实页面像素 bbox，再绘制红色矩形，并同步修正 `crop_image.bbox` 监督。",
        ]
    )
    (output_dir / "构建报告.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def collect_task_targets(all_rows_by_split: dict[str, list[dict[str, Any]]], args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for rows in all_rows_by_split.values():
        for row in rows:
            action = row.get("action") or {}
            if action.get("action") == "crop_image" and "bbox" in action:
                task_id = str(row.get("task_id"))
                page_path = (row.get("images") or [None])[0]
                targets[task_id] = {
                    "original_bbox": action.get("bbox"),
                    "page_bbox": action.get("bbox"),
                    "page_path": page_path,
                    "crop_path": find_crop_path(row),
                }
    for rows in all_rows_by_split.values():
        for row in rows:
            task_id = str(row.get("task_id"))
            if task_id in targets and not targets[task_id].get("crop_path"):
                crop_path = find_crop_path(row)
                if crop_path:
                    targets[task_id]["crop_path"] = crop_path
    if args.bbox_source == "template":
        for target in targets.values():
            page_path = target.get("page_path")
            crop_path = target.get("crop_path")
            if page_path and crop_path:
                located = locate_crop_on_page(Path(page_path), Path(crop_path), args.match_max_dim, args.match_scale_steps)
                if located and located["score"] >= args.min_template_score:
                    target["page_bbox"] = located["bbox"]
                    target["template_score"] = located["score"]
                    target["template_scale"] = located["scale"]
                else:
                    target["template_score"] = located["score"] if located else None
                    target["template_failed"] = True
            else:
                target["template_score"] = None
                target["template_failed"] = True
    return targets


def find_crop_path(row: dict[str, Any]) -> str | None:
    images = row.get("images") or []
    if len(images) >= 2:
        return images[1]
    for result in row.get("tool_results") or []:
        if isinstance(result, dict) and result.get("tool") == "crop_image" and result.get("crop_path"):
            return str(result.get("crop_path"))
    return None


def locate_crop_on_page(page_path: Path, crop_path: Path, match_max_dim: int, scale_steps: int) -> dict[str, Any] | None:
    import cv2
    import numpy as np

    page = cv2.imread(str(page_path), cv2.IMREAD_GRAYSCALE)
    crop = cv2.imread(str(crop_path), cv2.IMREAD_GRAYSCALE)
    if page is None or crop is None:
        return None
    resize_factor = 1.0
    if match_max_dim and max(page.shape[:2]) > match_max_dim:
        resize_factor = match_max_dim / max(page.shape[:2])
        page = cv2.resize(
            page,
            (int(page.shape[1] * resize_factor), int(page.shape[0] * resize_factor)),
            interpolation=cv2.INTER_AREA,
        )
        crop = cv2.resize(
            crop,
            (max(1, int(crop.shape[1] * resize_factor)), max(1, int(crop.shape[0] * resize_factor))),
            interpolation=cv2.INTER_AREA,
        )
    best: tuple[float, float, tuple[int, int], tuple[int, int]] | None = None
    for scale in np.linspace(0.25, 1.25, max(5, int(scale_steps))):
        h, w = crop.shape[:2]
        new_w, new_h = int(w * float(scale)), int(h * float(scale))
        if new_w < 20 or new_h < 20 or new_w > page.shape[1] or new_h > page.shape[0]:
            continue
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        template = cv2.resize(crop, (new_w, new_h), interpolation=interpolation)
        result = cv2.matchTemplate(page, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(result)
        if best is None or score > best[0]:
            best = (float(score), float(scale), loc, (new_w, new_h))
    if best is None:
        return None
    score, scale, loc, size = best
    x, y = loc
    w, h = size
    if resize_factor != 1.0:
        inv = 1.0 / resize_factor
        bbox = [int(round(x * inv)), int(round(y * inv)), int(round((x + w) * inv)), int(round((y + h) * inv))]
    else:
        bbox = [x, y, x + w, y + h]
    return {"score": score, "scale": scale, "bbox": bbox}


def rewrite_bboxes(row: dict[str, Any], target: dict[str, Any]) -> None:
    old_bbox = target.get("original_bbox")
    new_bbox = target.get("page_bbox")
    if not old_bbox or not new_bbox:
        return
    rewrite_action_bbox(row.get("action"), old_bbox, new_bbox)
    for action in row.get("history") or []:
        rewrite_action_bbox(action, old_bbox, new_bbox)
    for result in row.get("tool_results") or []:
        if isinstance(result, dict) and result.get("tool") == "crop_image":
            if same_bbox(result.get("bbox"), old_bbox):
                result["bbox"] = new_bbox


def rewrite_action_bbox(action: Any, old_bbox: Any, new_bbox: Any) -> None:
    if isinstance(action, dict) and action.get("action") == "crop_image" and same_bbox(action.get("bbox"), old_bbox):
        action["bbox"] = new_bbox


def same_bbox(a: Any, b: Any) -> bool:
    try:
        return [int(round(float(x))) for x in a] == [int(round(float(x))) for x in b]
    except Exception:
        return False


def summarize_template_scores(task_targets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores = [float(item["template_score"]) for item in task_targets.values() if item.get("template_score") is not None]
    failed = sum(1 for item in task_targets.values() if item.get("template_failed"))
    if not scores:
        return {"count": 0, "failed": failed}
    return {
        "count": len(scores),
        "failed": failed,
        "min": min(scores),
        "mean": sum(scores) / len(scores),
        "max": max(scores),
    }


def make_highlighted_page(page_path: Path, bbox: Any, output_dir: Path, task_id: str, line_width: int) -> str:
    if not page_path.exists():
        raise FileNotFoundError(page_path)
    image = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
    width = max(3, int(line_width))
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=(255, 0, 0))
    safe_name = safe_filename(f"{task_id}_{page_path.stem}_highlighted.png")
    out_path = output_dir / safe_name
    image.save(out_path)
    return str(out_path)


def rewrite_message_images(row: dict[str, Any], old_page: str, new_page: str) -> None:
    for message in row.get("messages") or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "image" and item.get("image") == old_page:
                item["image"] = new_page


def copy_sidecar_files(input_dir: Path, output_dir: Path) -> None:
    for name in [
        "tasks_all.jsonl",
        "train_tasks.jsonl",
        "val_tasks.jsonl",
        "test_tasks.jsonl",
        "claim_gold.jsonl",
        "evidence_links.jsonl",
        "quality_report.json",
    ]:
        src = input_dir / name
        if src.exists():
            dst = output_dir / name
            dst.write_bytes(src.read_bytes())


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
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


if __name__ == "__main__":
    raise SystemExit(main())
