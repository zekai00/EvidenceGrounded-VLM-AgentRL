#!/usr/bin/env python3
"""Create a 30-task manual audit sheet for v0.4.2 batch-claims data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


DEFAULT_DATASET = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_4_2_batch_claims_sft_20260601_1511"
)
DEFAULT_DOCS_DIR = Path("/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/docs/03_实验报告")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    report_path = output_dir / f"v0.4.2BatchClaims人工抽检30条_{stamp}.md"
    assets_dir = output_dir / f"v0.4.2BatchClaims人工抽检30条_{stamp}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(dataset_dir / "tasks_all.jsonl")
    episodes = {row["task_id"]: row for row in read_jsonl(dataset_dir / "episodes" / "oracle_episodes.jsonl")}
    selected = select_tasks(tasks, args.n, args.seed)
    lines = build_report(selected, episodes, report_path, assets_dir, dataset_dir, args.seed)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": str(dataset_dir),
        "report_path": str(report_path),
        "assets_dir": str(assets_dir),
        "sample_count": len(selected),
        "task_ids": [task["task_id"] for task in selected],
    }
    (assets_dir / "sample_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_tasks(tasks: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    by_scope: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        by_scope.setdefault(image_scope(task), []).append(task)
    for rows in by_scope.values():
        rng.shuffle(rows)

    quotas = [
        ("full_work", 10),
        ("partial_detail", 6),
        ("ABSTAIN", 6),
        ("album_leaf", 3),
        ("scroll_section", 3),
    ]
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scope, quota in quotas:
        take_diverse(by_scope.get(scope, []), quota, selected, seen)

    # Force coverage for caption visibility edge cases.
    force_categories = [
        ("page_caption_bbox", 4),
        ("no_visible_caption_evidence", 2),
        ("caption_text_abstain", 2),
    ]
    for category, quota in force_categories:
        rows = [task for task in tasks if category_match(task, category)]
        rng.shuffle(rows)
        take_diverse(rows, quota, selected, seen, replace=True)

    if len(selected) < n:
        remaining = [task for task in tasks if task["task_id"] not in seen]
        rng.shuffle(remaining)
        take_diverse(remaining, n - len(selected), selected, seen)
    return selected[:n]


def take_diverse(
    candidates: list[dict[str, Any]],
    quota: int,
    selected: list[dict[str, Any]],
    seen: set[str],
    *,
    replace: bool = False,
) -> None:
    source_seen = {task.get("source_file") for task in selected}
    source_task_seen = {source_task_key(task) for task in selected}
    added = 0
    for task in candidates:
        if task["task_id"] in seen:
            continue
        if source_task_key(task) in source_task_seen:
            continue
        if not replace and added < max(1, quota // 2) and task.get("source_file") in source_seen:
            continue
        selected.append(task)
        seen.add(task["task_id"])
        source_seen.add(task.get("source_file"))
        source_task_seen.add(source_task_key(task))
        added += 1
        if added >= quota:
            return
    for task in candidates:
        if added >= quota:
            return
        if task["task_id"] in seen:
            continue
        if source_task_key(task) in source_task_seen:
            continue
        selected.append(task)
        seen.add(task["task_id"])
        source_task_seen.add(source_task_key(task))
        added += 1


def source_task_key(task: dict[str, Any]) -> str:
    if task.get("source_task_id"):
        return str(task["source_task_id"])
    task_id = str(task.get("task_id", ""))
    if "_r" in task_id:
        return task_id.rsplit("_r", 1)[0]
    return task_id


def category_match(task: dict[str, Any], category: str) -> bool:
    if category == "page_caption_bbox":
        return caption_source(task) == "page_caption_bbox"
    if category == "no_visible_caption_evidence":
        return bool(task.get("local_evidence")) and caption_source(task) == "no_visible_caption_evidence"
    if category == "caption_text_abstain":
        claim = claims_by_field(task).get("caption_text", {})
        return bool(claim.get("abstain"))
    return False


def build_report(
    tasks: list[dict[str, Any]],
    episodes: dict[str, dict[str, Any]],
    report_path: Path,
    assets_dir: Path,
    dataset_dir: Path,
    seed: int,
) -> list[str]:
    scope_counts = Counter(image_scope(task) for task in tasks)
    caption_source_counts = Counter(caption_source(task) for task in tasks)
    lines: list[str] = [
        "# v0.4.2 Batch Claims 人工抽检 30 条",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} CST",
        f"数据集：`{dataset_dir}`",
        f"抽样 seed：`{seed}`",
        "",
        "## 你需要检查什么",
        "",
        "每条样本建议按下面顺序看：",
        "",
        "1. **目标图像定位**：红框是否框住真正要理解的山水画/局部图；如果同页有多图，是否选错图。",
        "2. **图注对应关系**：蓝框/青框是否是红框目标图的图注，不是其他图、正文、页眉页脚。",
        "3. **caption_text**：图注文本是否完整，有没有 OCR 错字、跨行漏字、串到别的图注。",
        "4. **image_scope**：`full_work / partial_detail / scroll_section / album_leaf / abstain` 是否合理。局部图不能标成完整作品。",
        "5. **作品与展示区域**：`depicted_work_title` 是原作品名；`displayed_region` 是当前展示区域，例如 `《匡庐图》局部`。",
        "6. **作者/朝代/技法/材质尺寸/馆藏**：必须由图注或证据支持；`图片来源` 不等于 `馆藏`。",
        "7. **visual_elements / composition**：是否真的能从图像或证据看出来；不要把其他图例的内容写进来。",
        "8. **证据链**：claim 的 `evidence_ids` 是否打开过；来自图注的字段应优先引用 `local_caption_*`。",
        "9. **batch 动作**：`write_claims_batch` 里 claims 和 abstains 是否互相冲突；abstain 是否是真的证据不足。",
        "",
        "建议你在每条下面的 `人工检查记录` 里写：`OK`、`小问题`、`严重问题`，并标明字段名。",
        "",
        "## 抽样覆盖",
        "",
        f"- image_scope 分布：`{json.dumps(scope_counts, ensure_ascii=False)}`",
        f"- caption evidence 可见来源：`{json.dumps(caption_source_counts, ensure_ascii=False)}`",
        "",
    ]
    for idx, task in enumerate(tasks, start=1):
        lines.extend(render_task(idx, task, episodes.get(task["task_id"], {}), report_path, assets_dir))
    return lines


def render_task(index: int, task: dict[str, Any], episode: dict[str, Any], report_path: Path, assets_dir: Path) -> list[str]:
    prefix = f"{index:02d}_{safe_name(task['task_id'])}"
    page_annotated = assets_dir / f"{prefix}_page_annotated.jpg"
    target_crop = assets_dir / f"{prefix}_target_crop.jpg"
    create_assets(task, page_annotated, target_crop)
    page_rel = page_annotated.relative_to(report_path.parent)
    crop_rel = target_crop.relative_to(report_path.parent)
    claims = claims_by_field(task)
    batch = next((action for action in episode.get("actions", []) if action.get("action") == "write_claims_batch"), {})
    local_evidence = task.get("local_evidence") or []
    lines = [
        f"## {index:02d}. `{task['task_id']}`",
        "",
        f"- split：`{task.get('split')}`",
        f"- source_file：`{task.get('source_file')}`",
        f"- page：`{task.get('page')}`",
        f"- image_scope：`{image_scope(task)}`",
        f"- caption_source：`{caption_source(task)}`",
        f"- caption_text：{claims.get('caption_text', {}).get('value') if not claims.get('caption_text', {}).get('abstain') else '**ABSTAIN**'}",
        "",
        f"![page annotated]({page_rel})",
        "",
        f"![target crop]({crop_rel})",
        "",
        "图中标注：红框=目标图像 bbox；蓝框=gold caption_bbox；青框=可见 caption candidate；灰框=其他 region candidate。",
        "",
        "### Local Evidence",
        "",
    ]
    if local_evidence:
        for item in local_evidence:
            lines.append(f"- `{item.get('evidence_id')}`：{item.get('display_snippet')}")
    else:
        lines.append("- 无 local caption evidence。")
    lines.extend(["", "### Claims", "", "| 字段 | 值 / abstain | evidence_ids |", "|---|---|---|"])
    for field in [
        "caption_text",
        "image_scope",
        "depicted_work_title",
        "displayed_region",
        "object_type",
        "artist",
        "dynasty",
        "visual_elements",
        "technique",
        "composition",
        "medium_dimensions",
        "collection",
    ]:
        claim = claims.get(field, {})
        if claim.get("abstain"):
            value = "ABSTAIN: " + str(claim.get("reason", ""))
        else:
            value = json.dumps(claim.get("value"), ensure_ascii=False)
        evidence_ids = ", ".join(map(str, claim.get("evidence_ids") or []))
        lines.append(f"| `{field}` | {escape_table(value)} | {escape_table(evidence_ids)} |")
    lines.extend(
        [
            "",
            "### Batch Action",
            "",
            f"- claims：{len(batch.get('claims') or [])}",
            f"- abstains：{len(batch.get('abstains') or [])}",
            f"- trajectory steps：{len(episode.get('actions') or [])}",
            "",
            "### 人工检查记录",
            "",
            "- 目标图像定位：",
            "- 图注对应关系：",
            "- 字段/证据问题：",
            "- 结论：",
            "",
        ]
    )
    return lines


def create_assets(task: dict[str, Any], page_out: Path, crop_out: Path) -> None:
    page = Path(task["page_image"])
    with Image.open(page) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        for region in task.get("region_candidates") or []:
            bbox = region.get("bbox")
            if not valid_bbox(bbox):
                continue
            color = (80, 80, 80)
            width = 2
            label = str(region.get("region_id", "r"))
            if region.get("caption_evidence_id"):
                color = (0, 180, 220)
                width = 4
                label = f"{label} caption"
            draw_box(draw, bbox, color, label, width)
        caption_bbox = caption_bbox_for_page(task, image.size)
        if valid_bbox(caption_bbox):
            draw_box(draw, caption_bbox, (40, 90, 255), "caption_bbox", 4)
        target_bbox = task.get("gold", {}).get("image_bbox") or task.get("gold", {}).get("target_region_bbox")
        if valid_bbox(target_bbox):
            draw_box(draw, target_bbox, (255, 30, 30), "target", 5)
            crop = image.crop(tuple(int(v) for v in target_bbox))
            crop.save(crop_out, quality=92)
        elif task.get("artwork_image") and Path(task["artwork_image"]).exists():
            shutil.copyfile(task["artwork_image"], crop_out)
        else:
            crop = image.copy()
            crop.thumbnail((600, 600))
            crop.save(crop_out, quality=92)
        image.thumbnail((1200, 1600))
        image.save(page_out, quality=90)


def draw_box(draw: ImageDraw.ImageDraw, bbox: Any, color: tuple[int, int, int], label: str, width: int) -> None:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
    text_y = max(0, y1 - 14)
    draw.rectangle([x1, text_y, x1 + max(50, len(label) * 7), text_y + 14], fill=color)
    draw.text((x1 + 2, text_y + 1), label, fill=(255, 255, 255))


def caption_bbox_for_page(task: dict[str, Any], size: tuple[int, int]) -> list[int] | None:
    bbox = task.get("gold", {}).get("caption_bbox")
    if not valid_bbox(bbox):
        return None
    width, height = size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    coord = task.get("gold", {}).get("caption_bbox_coordinate")
    legacy = coord == "legacy_0_1000" or (
        coord is None and max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1000 and (width != 1000 or height != 1000)
    )
    if legacy:
        return [
            max(0, min(width, int(round(x1 / 1000.0 * width)))),
            max(0, min(height, int(round(y1 / 1000.0 * height)))),
            max(0, min(width, int(round(x2 / 1000.0 * width)))),
            max(0, min(height, int(round(y2 / 1000.0 * height)))),
        ]
    return [
        max(0, min(width, int(round(x1)))),
        max(0, min(height, int(round(y1)))),
        max(0, min(width, int(round(x2)))),
        max(0, min(height, int(round(y2)))),
    ]


def claims_by_field(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(claim.get("field")): claim for claim in task.get("gold", {}).get("claims", [])}


def image_scope(task: dict[str, Any]) -> str:
    claim = claims_by_field(task).get("image_scope", {})
    if claim.get("abstain"):
        return "ABSTAIN"
    return str(claim.get("value") or "")


def caption_source(task: dict[str, Any]) -> str:
    found_local = bool(task.get("local_evidence"))
    for region in task.get("region_candidates") or []:
        if region.get("caption_evidence_id"):
            return str(region.get("source") or "caption_region")
    return "no_visible_caption_evidence" if found_local else "no_local_caption_evidence"


def valid_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def safe_name(value: str) -> str:
    keep = []
    for ch in value:
        if ch.isalnum() or ch in {"_", "-"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)[:80]


def escape_table(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())
