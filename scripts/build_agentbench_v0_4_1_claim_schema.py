#!/usr/bin/env python3
"""Build v0.4.1 with a finer claim schema for partial/detail figures."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import EvidenceIndex, read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation, build_prompt_text, image_size  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_4_region_no_highlight_sft_20260601_1325"
)
DEFAULT_EVIDENCE_INDEX_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)

NEW_FIELDS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)
    tasks = read_jsonl(source_dir / "tasks_all.jsonl")
    sft_by_task = load_sft_by_task(source_dir)

    new_tasks: list[dict[str, Any]] = []
    schema_notes: list[dict[str, Any]] = []
    for task in tasks:
        new_task = copy.deepcopy(task)
        new_task["dataset_version"] = "v0.4.1_region_claim_schema"
        new_task["claim_schema_version"] = "v0.4.1"
        normalize_legacy_caption_bbox(new_task)
        enrich_caption_from_page_text(new_task)
        old_claims = task.get("gold", {}).get("claims", [])
        new_claims, notes, local_evidence = build_claim_schema(
            old_claims,
            new_task.get("gold", {}).get("caption_text") or "",
            new_task,
        )
        new_task["gold"]["legacy_claim_fields_v0_4"] = [claim.get("field") for claim in old_claims]
        new_task["gold"]["claims"] = new_claims
        new_task["gold"]["claim_schema_fields"] = NEW_FIELDS
        new_task["gold"]["claim_schema_notes"] = notes
        if local_evidence:
            new_task["local_evidence"] = [local_evidence]
            annotate_caption_region_candidates(new_task, local_evidence)
        new_tasks.append(new_task)
        schema_notes.append({"task_id": new_task["task_id"], **notes})

    new_sft_rows: list[dict[str, Any]] = []
    new_episodes: list[dict[str, Any]] = []
    tasks_by_id = {str(task["task_id"]): task for task in new_tasks}
    for task in new_tasks:
        task_rows = sft_by_task[str(task["task_id"])]
        rows, actions = rebuild_sft_rows(task, task_rows)
        new_sft_rows.extend(rows)
        new_episodes.append(
            {
                "task_id": task["task_id"],
                "source_task_id": task.get("source_task_id"),
                "split": task.get("split"),
                "variant": task.get("candidate_augmentation", {}).get("variant"),
                "actions": actions,
            }
        )

    write_outputs(output_dir, new_tasks, new_episodes, new_sft_rows)
    quality = summarize(new_tasks, new_sft_rows, schema_notes)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.4.1_region_claim_schema",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "claim_schema_fields": NEW_FIELDS,
        "quality": quality,
        "files": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "sft_train": str(output_dir / "sft" / "train.jsonl"),
            "sft_val": str(output_dir / "sft" / "val.jsonl"),
            "sft_test": str(output_dir / "sft" / "test.jsonl"),
            "oracle_episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def load_sft_by_task(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ["train", "val", "test"]:
        for row in read_jsonl(source_dir / "sft" / f"{split}.jsonl"):
            rows_by_task[str(row["task_id"])].append(row)
    for rows in rows_by_task.values():
        rows.sort(key=lambda item: int(item.get("step", 0)))
    return rows_by_task


def build_claim_schema(
    old_claims: list[dict[str, Any]],
    caption: str,
    task: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None]:
    by_field = {str(claim.get("field")): claim for claim in old_claims}
    caption_claim = by_field.get("caption_text") or {}
    old_title_claim = by_field.get("title") or {}
    local_evidence = make_local_caption_evidence(task, caption)
    local_caption_source = {
        "evidence_ids": [local_evidence["evidence_id"]] if local_evidence else [],
        "candidate_evidence_ids": [],
        "support_type": "page_caption_text",
    }
    extracted_title = extract_title(caption)
    old_title = clean_title(old_title_claim.get("value")) if not old_title_claim.get("abstain") else ""
    title = clean_title(extracted_title or old_title)
    evidence_source = local_caption_source if local_evidence else (caption_claim or old_title_claim)
    image_scope = infer_image_scope(caption, title)
    object_type = infer_object_type(caption, image_scope, title)
    medium_dimensions = infer_medium_dimensions(caption)
    collection = infer_collection(caption)
    inferred_dynasty = infer_dynasty_from_caption(caption)
    inferred_artist = infer_artist_from_caption(caption, inferred_dynasty)
    inferred_technique = infer_technique_from_caption(caption)

    claims: list[dict[str, Any]] = []
    claims.append(value_or_abstain("caption_text", choose_caption_text(caption_claim.get("value"), caption), evidence_source, "图注文本为空或不可读"))
    claims.append(value_or_abstain("image_scope", image_scope, evidence_source, "图注或证据不足，无法可靠判断当前图像是全幅、局部还是卷/册页片段"))
    title_evidence_source = evidence_source if extracted_title else (old_title_claim or evidence_source)
    claims.append(value_or_abstain("depicted_work_title", title, title_evidence_source, "图注和证据片段未明确给出原作品名"))
    claims.append(
        value_or_abstain(
            "displayed_region",
            infer_displayed_region(title, caption, image_scope),
            evidence_source,
            "图注和证据片段未明确说明当前展示区域",
        )
    )
    claims.append(value_or_abstain("object_type", object_type, evidence_source, "图注和证据片段未明确对象类型"))
    claims.append(copy_or_inferred_claim(by_field.get("artist") or {}, "artist", inferred_artist, evidence_source))
    claims.append(copy_or_inferred_claim(by_field.get("dynasty") or {}, "dynasty", inferred_dynasty, evidence_source))
    claims.append(copy_claim(by_field.get("visual_elements") or {}, "visual_elements"))
    claims.append(copy_or_inferred_claim(by_field.get("technique") or {}, "technique", inferred_technique, evidence_source))
    claims.append(copy_claim(by_field.get("composition") or {}, "composition"))
    claims.append(value_or_abstain("medium_dimensions", medium_dimensions, evidence_source, "图注未明确材质或尺寸"))
    claims.append(value_or_abstain("collection", collection, evidence_source, "图注未明确馆藏信息"))
    notes = {
        "caption": caption,
        "title_from_caption": bool(extracted_title),
        "old_title_value": old_title,
        "new_depicted_work_title": title or None,
        "image_scope": image_scope or None,
        "object_type": object_type or None,
        "medium_dimensions": medium_dimensions or None,
        "collection": collection or None,
        "artist_from_caption": bool(inferred_artist and (by_field.get("artist") or {}).get("abstain")),
        "dynasty_from_caption": bool(inferred_dynasty and (by_field.get("dynasty") or {}).get("abstain")),
        "technique_from_caption": bool(inferred_technique and (by_field.get("technique") or {}).get("abstain")),
    }
    return claims, notes, local_evidence


def make_local_caption_evidence(task: dict[str, Any], caption: str) -> dict[str, Any] | None:
    caption = normalize_spaces(caption)
    if not caption:
        return None
    evidence_id = f"local_caption_{task['task_id']}"
    return {
        "evidence_id": evidence_id,
        "source_file": task.get("source_file"),
        "page_start": task.get("page"),
        "page_end": task.get("page"),
        "authority_level": "B",
        "citation_level": "page_caption_region",
        "source_quality": "pdf_text_block_or_page_caption",
        "display_snippet": caption,
    }


def annotate_caption_region_candidates(task: dict[str, Any], local_evidence: dict[str, Any]) -> None:
    caption = normalize_spaces(local_evidence.get("display_snippet"))
    if not caption:
        return
    title = extract_title(caption)
    best_index = -1
    best_score = 0
    caption_chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", caption))
    for index, region in enumerate(task.get("region_candidates") or []):
        text = normalize_spaces(region.get("nearby_text"))
        if not text:
            continue
        score = len(caption_chars & set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text)))
        if title and title in text:
            score += 50
        if score > best_score:
            best_score = score
            best_index = index
    threshold = max(3, min(6, len(caption_chars) // 2))
    if best_index >= 0 and best_score >= threshold:
        region = task["region_candidates"][best_index]
        caption_bbox = task.get("gold", {}).get("caption_bbox")
        if is_valid_bbox(caption_bbox):
            region["bbox"] = [int(v) for v in caption_bbox]
            region["nearby_text"] = caption
        region["caption_evidence_id"] = local_evidence["evidence_id"]
        region["caption_hint"] = caption[:160]
        return
    caption_bbox = task.get("gold", {}).get("caption_bbox")
    if is_valid_bbox(caption_bbox):
        existing_ids = {str(item.get("region_id")) for item in task.get("region_candidates") or []}
        index = 0
        while f"r_caption_{index}" in existing_ids:
            index += 1
        task.setdefault("region_candidates", []).append(
            {
                "bbox": [int(v) for v in caption_bbox],
                "source": "page_caption_bbox",
                "type": "text_or_caption_candidate",
                "score": 0.06,
                "nearby_text": caption,
                "hint": "页面图注候选区域；可用 caption_evidence_id 打开本页图注证据",
                "region_id": f"r_caption_{index}",
                "caption_evidence_id": local_evidence["evidence_id"],
                "caption_hint": caption[:160],
            }
        )


def normalize_legacy_caption_bbox(task: dict[str, Any]) -> None:
    """Convert legacy 0-1000 caption bbox into current page-pixel coordinates."""
    gold = task.get("gold") or {}
    raw_bbox = gold.get("caption_bbox")
    if not is_valid_bbox(raw_bbox):
        return
    size = image_size(task.get("page_image"))
    width = size.get("width")
    height = size.get("height")
    if not width or not height:
        return
    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    legacy_normalized = max_coord <= 1000 and (width != 1000 or height != 1000)
    gold["legacy_caption_bbox_v0_3"] = [int(round(v)) for v in raw_bbox]
    if legacy_normalized:
        converted = [
            int(round(x1 / 1000.0 * width)),
            int(round(y1 / 1000.0 * height)),
            int(round(x2 / 1000.0 * width)),
            int(round(y2 / 1000.0 * height)),
        ]
        gold["caption_bbox"] = clamp_bbox(converted, int(width), int(height))
        gold["caption_bbox_coordinate"] = "page_pixels_from_legacy_0_1000"
    else:
        gold["caption_bbox"] = clamp_bbox([int(round(v)) for v in raw_bbox], int(width), int(height))
        gold["caption_bbox_coordinate"] = "page_pixels"


def enrich_caption_from_page_text(task: dict[str, Any]) -> None:
    gold = task.get("gold") or {}
    caption = normalize_spaces(gold.get("caption_text") or "")
    caption_bbox = gold.get("caption_bbox")
    if not caption or not is_valid_bbox(caption_bbox):
        return
    pdf_path = resolve_pdf_path(task)
    if not pdf_path:
        return
    size = image_size(task.get("page_image"))
    width = size.get("width")
    height = size.get("height")
    if not width or not height:
        return
    try:
        with fitz.open(str(pdf_path)) as doc:
            page_index = int(task.get("page") or 0) - 1
            if page_index < 0 or page_index >= len(doc):
                return
            page = doc[page_index]
            scale_x = float(width) / page.rect.width
            scale_y = float(height) / page.rect.height
            blocks = page_text_blocks(page, scale_x, scale_y, int(width), int(height))
    except Exception:
        return
    if not blocks:
        return
    image_bbox = gold.get("image_bbox") or gold.get("target_region_bbox")
    start_index = find_caption_start_block(blocks, image_bbox, caption_bbox, caption)
    if start_index < 0:
        return
    selected = merge_caption_blocks(blocks, start_index)
    if not selected:
        return
    merged_text = join_caption_lines([item["text"] for item in selected])
    if not should_replace_caption(caption, merged_text):
        return
    merged_bbox = union_bboxes([item["bbox"] for item in selected])
    if not is_valid_bbox(merged_bbox):
        return
    gold["legacy_caption_text_v0_3"] = caption
    gold["caption_text"] = merged_text
    gold["caption_bbox"] = merged_bbox
    gold["caption_bbox_coordinate"] = "page_pixels_from_pdf_text_multiline"
    gold["caption_text_source"] = "pdf_text_multiline_merge"


def resolve_pdf_path(task: dict[str, Any]) -> Path | None:
    source_path = task.get("source_path")
    if source_path and Path(source_path).exists():
        return Path(source_path)
    source_file = str(task.get("source_file") or "")
    if not source_file:
        return None
    roots = [
        Path("/root/datasets/chinese_landscape_authority_corpus/raw_pdfs"),
        Path("/root/Workspace/ShanshuiAgent/data/raw_pdfs"),
        Path("/root/Workspace/ChineseLandscape/data/raw_pdfs"),
    ]
    for root in roots:
        if not root.exists():
            continue
        direct = root / source_file
        if direct.exists():
            return direct
        matches = list(root.rglob(source_file))
        if matches:
            return matches[0]
    return None


def page_text_blocks(page: fitz.Page, scale_x: float, scale_y: float, width: int, height: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines") or []:
            line_text = "".join(str(span.get("text") or "") for span in line.get("spans") or [])
            line_text = normalize_spaces(line_text)
            if not line_text:
                continue
            bbox = scale_pdf_bbox(line.get("bbox"), scale_x, scale_y, width, height)
            if not is_valid_bbox(bbox):
                continue
            blocks.append({"bbox": bbox, "text": line_text})
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return blocks


def scale_pdf_bbox(value: Any, scale_x: float, scale_y: float, width: int, height: int) -> list[int] | None:
    if not value or len(value) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value]
    return clamp_bbox(
        [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        ],
        width,
        height,
    )


def find_caption_start_block(blocks: list[dict[str, Any]], image_bbox: Any, caption_bbox: list[int], caption: str) -> int:
    caption_chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", caption))
    best_index = -1
    best_score = -999.0
    for index, block in enumerate(blocks):
        bbox = block["bbox"]
        text = normalize_spaces(block["text"])
        if looks_like_body_text(text):
            continue
        legacy_overlap = bbox_iou(caption_bbox, bbox)
        image_score = caption_image_relation_score(bbox, image_bbox)
        char_score = len(caption_chars & set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text))) / max(1, len(caption_chars))
        prefix_bonus = 1.0 if text and (caption.startswith(text[: min(len(text), 12)]) or text.startswith(caption[: min(len(caption), 12)])) else 0.0
        marker_bonus = 2.5 if looks_like_caption_start(text) else 0.0
        score = image_score + marker_bonus + char_score + prefix_bonus + legacy_overlap * 0.5
        if score > best_score:
            best_score = score
            best_index = index
    return best_index if best_score >= 1.0 else -1


def caption_image_relation_score(caption_bbox: list[int], image_bbox: Any) -> float:
    if not is_valid_bbox(image_bbox):
        return 0.0
    ix1, iy1, ix2, iy2 = [float(v) for v in image_bbox]
    cx1, cy1, cx2, cy2 = [float(v) for v in caption_bbox]
    overlap = horizontal_overlap_ratio(caption_bbox, [int(ix1), int(iy1), int(ix2), int(iy2)])
    image_center = (ix1 + ix2) / 2.0
    caption_center = (cx1 + cx2) / 2.0
    center_penalty = min(abs(caption_center - image_center) / max(1.0, ix2 - ix1), 1.5)
    below_gap = cy1 - iy2
    above_gap = iy1 - cy2
    relation = -0.5
    if 0 <= below_gap <= 170:
        relation = 3.0 - below_gap / 170.0
    elif 0 <= above_gap <= 130:
        relation = 2.2 - above_gap / 130.0
    return relation + overlap * 2.0 - center_penalty


def looks_like_caption_start(text: str) -> bool:
    return bool(re.search(r"(?:^|[〔【（(])\s*(?:图|圖|Figure|Fig\.?)\s*[一二三四五六七八九十百千万〇零\d]+", text, flags=re.I)) or "《" in text


def merge_caption_blocks(blocks: list[dict[str, Any]], start_index: int) -> list[dict[str, Any]]:
    selected = [blocks[start_index]]
    start_bbox = blocks[start_index]["bbox"]
    prev_bbox = start_bbox
    start_center = (start_bbox[0] + start_bbox[2]) / 2.0
    for block in blocks[start_index + 1 : start_index + 10]:
        bbox = block["bbox"]
        text = normalize_spaces(block["text"])
        if not text or looks_like_body_text(text):
            break
        gap = bbox[1] - prev_bbox[3]
        center = (bbox[0] + bbox[2]) / 2.0
        horizontal_ok = abs(center - start_center) <= max(220, (start_bbox[2] - start_bbox[0]) * 0.8) or horizontal_overlap_ratio(bbox, start_bbox) >= 0.15
        if gap < -5:
            continue
        if gap > 55:
            break
        if not horizontal_ok:
            continue
        if re.match(r"^(?:图|Figure|Fig\.?)\s*\d", text, flags=re.I):
            break
        selected.append(block)
        prev_bbox = bbox
    return selected


def looks_like_body_text(text: str) -> bool:
    if len(text) > 95:
        return True
    if re.match(r"^第?\s*\d+\s*[页章节]", text):
        return True
    return False


def should_replace_caption(old_caption: str, new_caption: str) -> bool:
    old_caption = normalize_spaces(old_caption)
    new_caption = normalize_spaces(new_caption)
    if not new_caption:
        return False
    if len(new_caption) > len(old_caption) + 4 and compact_text(old_caption) in compact_text(new_caption):
        return True
    if looks_like_caption_start(new_caption) and not looks_like_caption_start(old_caption):
        return True
    if looks_like_caption_start(new_caption) and len(new_caption) >= max(8, len(old_caption)):
        return True
    return False


def join_caption_lines(lines: list[str]) -> str:
    result = ""
    for raw in lines:
        line = normalize_spaces(raw)
        if not line:
            continue
        if not result:
            result = line
            continue
        sep = " " if re.match(r"^[A-Za-z0-9]", line) or re.search(r"[A-Za-z0-9]$", result) else ""
        result += sep + line
    return normalize_spaces(result)


def union_bboxes(bboxes: list[list[int]]) -> list[int] | None:
    valid = [bbox for bbox in bboxes if is_valid_bbox(bbox)]
    if not valid:
        return None
    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]


def bbox_iou(a: Any, b: Any) -> float:
    if not is_valid_bbox(a) or not is_valid_bbox(b):
        return 0.0
    ax1, ay1, ax2, ay2 = [float(v) for v in a]
    bx1, by1, bx2, by2 = [float(v) for v in b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1.0, area_a + area_b - inter)


def horizontal_overlap_ratio(a: list[int], b: list[int]) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    return overlap / max(1, min(a[2] - a[0], b[2] - b[0]))


def choose_caption_text(claim_value: Any, caption: str) -> str:
    claim_text = normalize_spaces(str(claim_value or ""))
    caption = normalize_spaces(caption)
    if not claim_text:
        return caption
    if looks_like_caption_start(caption) and not looks_like_caption_start(claim_text):
        return caption
    if len(caption) > len(claim_text) + 4 and compact_text(claim_text) in compact_text(caption):
        return caption
    if looks_like_caption_start(caption) and len(caption) >= max(8, len(claim_text)):
        return caption
    return claim_text


def compact_text(text: str) -> str:
    return re.sub(r"\s+", "", normalize_spaces(text))


def clamp_bbox(bbox: list[int], width: int, height: int) -> list[int]:
    return [
        max(0, min(width, int(bbox[0]))),
        max(0, min(height, int(bbox[1]))),
        max(0, min(width, int(bbox[2]))),
        max(0, min(height, int(bbox[3]))),
    ]


def is_valid_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [int(v) for v in value]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def copy_claim(claim: dict[str, Any], field: str, fallback_value: Any = None) -> dict[str, Any]:
    if claim.get("abstain"):
        return {
            "claim_id": field,
            "field": field,
            "value": None,
            "abstain": True,
            "reason": claim.get("reason") or "证据不足",
            "evidence_ids": [],
            "candidate_evidence_ids": claim.get("candidate_evidence_ids") or [],
            "support_type": claim.get("support_type", "text"),
        }
    value = claim.get("value")
    if value in [None, "", []] and fallback_value not in [None, "", []]:
        value = fallback_value
    if value in [None, "", []]:
        return {
            "claim_id": field,
            "field": field,
            "value": None,
            "abstain": True,
            "reason": "证据不足",
            "evidence_ids": [],
            "candidate_evidence_ids": claim.get("candidate_evidence_ids") or [],
            "support_type": claim.get("support_type", "text"),
        }
    copied = copy.deepcopy(claim)
    copied["claim_id"] = field
    copied["field"] = field
    copied["value"] = value
    copied["abstain"] = False
    copied.setdefault("confidence", 0.85)
    copied.setdefault("evidence_ids", [])
    copied.setdefault("candidate_evidence_ids", [])
    return copied


def copy_or_inferred_claim(claim: dict[str, Any], field: str, inferred_value: Any, evidence_source: dict[str, Any]) -> dict[str, Any]:
    if inferred_value not in [None, "", []]:
        if not claim or claim.get("abstain") or values_equivalent(claim.get("value"), inferred_value):
            return value_or_abstain(field, inferred_value, evidence_source, "图注和证据片段未明确给出该字段")
    if claim and not claim.get("abstain") and claim.get("value") not in [None, "", []]:
        return copy_claim(claim, field)
    if inferred_value not in [None, "", []]:
        return value_or_abstain(field, inferred_value, evidence_source, "图注和证据片段未明确给出该字段")
    return copy_claim(claim or {}, field)


def values_equivalent(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        left_items = {normalize_spaces(item) for item in (left if isinstance(left, list) else [left]) if item not in [None, ""]}
        right_items = {normalize_spaces(item) for item in (right if isinstance(right, list) else [right]) if item not in [None, ""]}
        return bool(left_items & right_items)
    return normalize_spaces(left) == normalize_spaces(right)


def value_or_abstain(field: str, value: Any, evidence_source: dict[str, Any], reason: str) -> dict[str, Any]:
    if value in [None, "", []]:
        return {
            "claim_id": field,
            "field": field,
            "value": None,
            "abstain": True,
            "reason": reason,
            "evidence_ids": [],
            "candidate_evidence_ids": evidence_source.get("candidate_evidence_ids") or [],
            "support_type": "text",
        }
    evidence_ids = evidence_source.get("evidence_ids") or []
    candidate_ids = evidence_source.get("candidate_evidence_ids") or []
    if not evidence_ids and candidate_ids:
        evidence_ids = candidate_ids[:1]
    return {
        "claim_id": field,
        "field": field,
        "value": value,
        "abstain": False,
        "evidence_ids": evidence_ids,
        "candidate_evidence_ids": candidate_ids,
        "support_type": "text",
        "confidence": 0.85,
    }


def extract_title(text: str) -> str:
    matches = re.findall(r"《([^》]{1,80})》", text or "")
    if matches:
        return matches[0].strip()
    return ""


def clean_title(value: Any) -> str:
    text = str(value or "").strip()
    text = text.strip("《》<>")
    return text


def infer_image_scope(caption: str, title: str) -> str:
    text = caption or ""
    if any(key in text for key in ["局部", "截取", "基础上进行截取"]):
        return "partial_detail"
    if any(key in text for key in ["之一", "其一", "一段"]):
        return "scroll_section"
    if any(key in text for key in ["图册", "册页"]):
        return "album_leaf"
    if is_generic_caption(text):
        return ""
    if title:
        return "full_work"
    return ""


def infer_displayed_region(title: str, caption: str, image_scope: str) -> str:
    if title and image_scope == "partial_detail":
        return f"《{title}》局部"
    if title and image_scope == "scroll_section":
        return f"《{title}》之一"
    if title and image_scope == "album_leaf":
        return f"《{title}》册页/图册页"
    if title and image_scope == "full_work":
        return f"《{title}》"
    return ""


def infer_object_type(caption: str, image_scope: str, title: str) -> str:
    text = caption or ""
    if image_scope == "partial_detail":
        return "detail_crop"
    if image_scope == "scroll_section":
        return "scroll_section"
    if image_scope == "album_leaf":
        return "album_leaf"
    if "轴" in text:
        return "hanging_scroll"
    if "卷" in text:
        return "handscroll"
    if title:
        return "painting_or_figure"
    return ""


def infer_medium_dimensions(caption: str) -> str:
    text = normalize_spaces(caption)
    materials = []
    for pattern in ["绢本设色", "绢本水墨", "纸本设色", "纸本水墨", "卷绢"]:
        if pattern in text:
            materials.append(pattern)
    if not any(pattern in text for pattern in ["绢本设色", "绢本水墨"]) and "绢本" in text:
        materials.append("绢本")
    if not any(pattern in text for pattern in ["纸本设色", "纸本水墨"]) and "纸本" in text:
        materials.append("纸本")
    if not any(pattern in text for pattern in ["绢本水墨", "纸本水墨"]) and "水墨" in text:
        materials.append("水墨")
    if not any(pattern in text for pattern in ["绢本设色", "纸本设色"]) and "设色" in text:
        materials.append("设色")
    dims = re.findall(r"\d+(?:\.\d+)?\s*[xX×]\s*\d+(?:\.\d+)?\s*(?:cm|厘米)", text)
    pieces = dedupe(materials + [normalize_spaces(dim) for dim in dims])
    return "；".join(pieces)


def infer_collection(caption: str) -> str:
    text = normalize_spaces(caption)
    text = re.split(r"(?:图片来源|图源|来源|取像|取图|链接|http|www)", text, maxsplit=1)[0]
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text)
    candidates: list[str] = []
    suffix = r"(?:博物馆|博物院|美术馆|艺术馆|纪念馆|书画院|画院)"
    for source in [text, compact]:
        candidates.extend(re.findall(rf"([\u4e00-\u9fffA-Za-z0-9（）()·\s]{{2,60}}?{suffix})(?:藏|收藏)?", source))
    for raw in candidates:
        value = clean_collection_name(raw)
        if value:
            return value
    return ""


def clean_collection_name(value: str) -> str:
    value = normalize_spaces(value)
    value = re.sub(r"(?:藏|收藏)$", "", value)
    value = re.split(r"(?:图\s*\d+(?:[-—]\d+)?|Figure|Fig\.?)", value, flags=re.I)[-1]
    value = re.sub(r"^.*《[^》]{1,80}》", "", value)
    value = re.sub(
        r"^.*?(?:厘米|cm|CM|mm|MM|\d+(?:\.\d+)?\s*[xX×*]\s*\d+(?:\.\d+)?|"
        r"绢本设色|绢本水墨|纸本设色|纸本水墨|绢本|纸本|设色|水墨|浅绛|青绿|卷绢|卷|轴|册页)",
        "",
        value,
    )
    value = re.sub(r"^(?:现藏于?|藏于|收藏于|藏|于|在|：|:|，|,|。|；|;|\s)+", "", value)
    value = value.strip(" ，,。；;：:（）()[]【】")
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9（）()·]{2,30}(?:博物馆|博物院|美术馆|艺术馆|纪念馆|书画院|画院))$", value)
    if not match:
        return ""
    value = match.group(1)
    if value in {"中华珍宝馆"}:
        return ""
    return value


DYNASTIES = [
    "五代十国",
    "魏晋南北朝",
    "南北朝",
    "五代",
    "北宋",
    "南宋",
    "元代",
    "明代",
    "清代",
    "宋代",
    "唐代",
    "晋代",
    "隋代",
    "元",
    "明",
    "清",
    "宋",
    "唐",
    "晋",
    "隋",
]


def infer_dynasty_from_caption(caption: str) -> str:
    text = normalize_spaces(caption)
    for item in DYNASTIES:
        if item in text:
            return item
    return ""


def infer_artist_from_caption(caption: str, dynasty: str) -> str:
    if not dynasty:
        return ""
    text = normalize_spaces(caption)
    if dynasty not in text:
        return ""
    after = text.split(dynasty, 1)[1]
    after = re.sub(r"《[^》]{1,80}》", " ", after)
    after = re.split(r"(?:绢本|纸本|设色|水墨|浅绛|青绿|厘米|cm|图片来源|取像|藏|\d)", after, maxsplit=1)[0]
    after = re.sub(r"[（）()【】\[\]：:，,。；;、\s]", "", after)
    if 2 <= len(after) <= 6 and after not in DYNASTIES and after not in {"佚名", "局部"}:
        return after
    if after == "佚名":
        return after
    return ""


def infer_technique_from_caption(caption: str) -> list[str]:
    text = normalize_spaces(caption)
    values: list[str] = []
    for item in ["工笔", "写意", "青绿", "浅绛", "水墨", "设色"]:
        if item in text:
            values.append(item)
    return values


def is_generic_caption(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return bool(re.fullmatch(r"图\d+(?:[.-]\d+)?", compact))


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def rebuild_sft_rows(task: dict[str, Any], old_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_config = PromptConfig(tool_schema="region", coordinate_info=True)
    prefix_rows = []
    first_claim_row = None
    for row in old_rows:
        action_name = (row.get("action") or {}).get("action")
        if action_name in {"write_claim", "abstain_claim", "finish"}:
            first_claim_row = row
            break
        prefix_rows.append(row)
    if first_claim_row is None:
        first_claim_row = old_rows[-1]

    new_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for row in prefix_rows:
        action = copy.deepcopy(row.get("action") or {})
        new_rows.append(make_sft_row(task, row, action, prompt_config, int(row.get("step", len(new_rows)))))
        actions.append(action)

    state = {
        "history": copy.deepcopy(first_claim_row.get("history") or []),
        "tool_results": copy.deepcopy(first_claim_row.get("tool_results") or []),
        "draft_claims": copy.deepcopy(first_claim_row.get("draft_claims") or []),
        "images": copy.deepcopy(first_claim_row.get("images") or []),
    }
    state["history"] = retag_history_actions(state["history"], actions)
    step = len(new_rows)
    opened_ids = opened_evidence_ids(state)
    for evidence_id in required_evidence_ids(task):
        if evidence_id in opened_ids:
            continue
        action = {"action": "open_evidence", "evidence_id": evidence_id}
        row_state = {
            "task_id": task["task_id"],
            "split": task.get("split"),
            "step": step,
            "history": copy.deepcopy(state["history"]),
            "tool_results": copy.deepcopy(state["tool_results"]),
            "draft_claims": copy.deepcopy(state["draft_claims"]),
            "images": copy.deepcopy(state["images"]),
        }
        new_rows.append(make_sft_row(task, row_state, action, prompt_config, step))
        actions.append(action)
        apply_open_evidence_action(task, state, action)
        opened_ids.add(evidence_id)
        step += 1
    for claim in task.get("gold", {}).get("claims", []):
        action = claim_to_action(claim)
        row_state = {
            "task_id": task["task_id"],
            "split": task.get("split"),
            "step": step,
            "history": copy.deepcopy(state["history"]),
            "tool_results": copy.deepcopy(state["tool_results"]),
            "draft_claims": copy.deepcopy(state["draft_claims"]),
            "images": copy.deepcopy(state["images"]),
        }
        new_rows.append(make_sft_row(task, row_state, action, prompt_config, step))
        actions.append(action)
        apply_claim_action(state, action)
        step += 1
    finish = {"action": "finish", "status": "done"}
    finish_state = {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "step": step,
        "history": copy.deepcopy(state["history"]),
        "tool_results": copy.deepcopy(state["tool_results"]),
        "draft_claims": copy.deepcopy(state["draft_claims"]),
        "images": copy.deepcopy(state["images"]),
    }
    new_rows.append(make_sft_row(task, finish_state, finish, prompt_config, step))
    actions.append(finish)
    return new_rows, actions


def required_evidence_ids(task: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for claim in task.get("gold", {}).get("claims", []):
        if claim.get("abstain"):
            continue
        for evidence_id in claim.get("evidence_ids") or []:
            evidence_id = str(evidence_id)
            if evidence_id and evidence_id not in ids:
                ids.append(evidence_id)
    return ids


def opened_evidence_ids(state: dict[str, Any]) -> set[str]:
    opened: set[str] = set()
    for action in state.get("history") or []:
        if isinstance(action, dict) and action.get("action") == "open_evidence" and action.get("evidence_id"):
            opened.add(str(action["evidence_id"]))
    for result in state.get("tool_results") or []:
        if isinstance(result, dict) and result.get("tool") == "open_evidence" and result.get("evidence_id") and not result.get("error"):
            opened.add(str(result["evidence_id"]))
    return opened


_INDEX_CACHE: dict[str, EvidenceIndex] = {}


def evidence_index_for(task: dict[str, Any]) -> EvidenceIndex:
    index_dir = str((task.get("evidence_index") or {}).get("path") or DEFAULT_EVIDENCE_INDEX_DIR)
    if index_dir not in _INDEX_CACHE:
        _INDEX_CACHE[index_dir] = EvidenceIndex(index_dir)
    return _INDEX_CACHE[index_dir]


def apply_open_evidence_action(task: dict[str, Any], state: dict[str, Any], action: dict[str, Any]) -> None:
    evidence_id = str(action.get("evidence_id"))
    item = local_evidence_by_id(task, evidence_id) or evidence_index_for(task).open(evidence_id)
    if not item:
        result = {"tool": "open_evidence", "evidence_id": evidence_id, "error": "evidence_id_not_found"}
    else:
        result = {
            "tool": "open_evidence",
            "evidence_id": item.get("evidence_id"),
            "source_file": item.get("source_file"),
            "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
            "page_end": item.get("page_end"),
            "authority_level": item.get("authority_level"),
            "citation_level": item.get("citation_level"),
            "display_snippet": item.get("display_snippet") or item.get("evidence_summary") or str(item.get("text", ""))[:600],
        }
    state["history"].append(copy.deepcopy(action))
    state["tool_results"].append(result)


def local_evidence_by_id(task: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    for item in task.get("local_evidence") or []:
        if str(item.get("evidence_id")) == str(evidence_id):
            return item
    return None


def retag_history_actions(history: list[Any], prefix_actions: list[dict[str, Any]]) -> list[Any]:
    if len(history) >= len(prefix_actions):
        return history
    return prefix_actions[:]


def claim_to_action(claim: dict[str, Any]) -> dict[str, Any]:
    field = claim.get("field")
    if claim.get("abstain"):
        return {"action": "abstain_claim", "field": field, "reason": claim.get("reason", "证据不足")}
    return {
        "action": "write_claim",
        "field": field,
        "value": claim.get("value"),
        "evidence_ids": claim.get("evidence_ids") or [],
        "visual_bbox": claim.get("visual_bbox"),
        "confidence": claim.get("confidence", 0.85),
    }


def apply_claim_action(state: dict[str, Any], action: dict[str, Any]) -> None:
    state["history"].append(copy.deepcopy(action))
    if action["action"] == "write_claim":
        claim = {
            "field": action.get("field"),
            "value": action.get("value"),
            "evidence_ids": action.get("evidence_ids") or [],
            "visual_bbox": action.get("visual_bbox"),
            "confidence": action.get("confidence"),
            "abstain": False,
        }
        upsert_claim(state["draft_claims"], claim)
        state["tool_results"].append({"tool": "write_claim", "claim": claim})
    elif action["action"] == "abstain_claim":
        claim = {"field": action.get("field"), "reason": action.get("reason"), "abstain": True}
        upsert_claim(state["draft_claims"], claim)
        state["tool_results"].append({"tool": "abstain_claim", "claim": claim})


def upsert_claim(claims: list[dict[str, Any]], claim: dict[str, Any]) -> None:
    field = claim.get("field")
    claims[:] = [item for item in claims if item.get("field") != field]
    claims.append(claim)


def make_sft_row(
    task: dict[str, Any],
    row_state: dict[str, Any],
    action: dict[str, Any],
    prompt_config: PromptConfig,
    step: int,
) -> dict[str, Any]:
    obs = make_obs(task, row_state)
    return {
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "split": task.get("split"),
        "variant": task.get("candidate_augmentation", {}).get("variant"),
        "step": step,
        "tool_schema_version": "v0.4.1_region_claim_schema",
        "action": action,
        "history": copy.deepcopy(row_state.get("history") or []),
        "tool_results": copy.deepcopy(row_state.get("tool_results") or []),
        "draft_claims": copy.deepcopy(row_state.get("draft_claims") or []),
        "selected_evidence_ids": copy.deepcopy(row_state.get("selected_evidence_ids") or []),
        "images": copy.deepcopy(row_state.get("images") or []),
        "prompt_text": build_prompt_text(obs, prompt_config),
        "messages": build_messages_from_observation(obs, prompt_config, include_assistant_action=action),
        "label_source": "v0_4_region_selection_claim_schema_v0_4_1",
    }


def make_obs(task: dict[str, Any], row_state: dict[str, Any]) -> dict[str, Any]:
    image_paths = row_state.get("images") or [task.get("page_image")]
    images = []
    for index, path in enumerate(image_paths):
        images.append({"role": "page_image" if index == 0 else "last_crop", "path": path})
    return {
        "task_id": task["task_id"],
        "goal": task.get("goal"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "images": images,
        "page_size": image_size(task["page_image"]),
        "history": row_state.get("history") or [],
        "tool_results": row_state.get("tool_results") or [],
        "draft_claims": row_state.get("draft_claims") or [],
        "claim_state": row_state.get("claim_state"),
        "selected_evidence_ids": row_state.get("selected_evidence_ids") or [],
    }


def write_outputs(output_dir: Path, tasks: list[dict[str, Any]], episodes: list[dict[str, Any]], sft_rows: list[dict[str, Any]]) -> None:
    write_jsonl(output_dir / "tasks_all.jsonl", tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)
    for split in ["train", "val", "test"]:
        split_tasks = [task for task in tasks if task.get("split") == split]
        split_episodes = [episode for episode in episodes if episode.get("split") == split]
        split_sft = [row for row in sft_rows if row.get("split") == split]
        write_jsonl(output_dir / f"{split}_tasks.jsonl", split_tasks)
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", split_episodes)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", split_sft)
    write_jsonl(output_dir / "sft" / "all.jsonl", sft_rows)


def summarize(tasks: list[dict[str, Any]], sft_rows: list[dict[str, Any]], schema_notes: list[dict[str, Any]]) -> dict[str, Any]:
    task_split_counts = Counter(str(task.get("split")) for task in tasks)
    sft_split_counts = Counter(str(row.get("split")) for row in sft_rows)
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in sft_rows)
    field_counts = Counter(str((row.get("action") or {}).get("field")) for row in sft_rows if (row.get("action") or {}).get("field"))
    claim_field_counts = Counter()
    image_scope_counts = Counter()
    object_type_counts = Counter()
    for task in tasks:
        for claim in task.get("gold", {}).get("claims", []):
            claim_field_counts[str(claim.get("field"))] += 1
            if claim.get("field") == "image_scope" and not claim.get("abstain"):
                image_scope_counts[str(claim.get("value"))] += 1
            if claim.get("field") == "object_type" and not claim.get("abstain"):
                object_type_counts[str(claim.get("value"))] += 1
    return {
        "tasks_total": len(tasks),
        "task_split_counts": dict(task_split_counts),
        "sft_rows_total": len(sft_rows),
        "sft_split_counts": dict(sft_split_counts),
        "sft_action_counts": dict(action_counts),
        "sft_field_action_counts": dict(field_counts),
        "claim_field_counts": dict(claim_field_counts),
        "image_scope_distribution": dict(image_scope_counts),
        "object_type_distribution": dict(object_type_counts),
        "title_from_caption_count": sum(bool(item.get("title_from_caption")) for item in schema_notes),
        "medium_dimensions_count": sum(bool(item.get("medium_dimensions")) for item in schema_notes),
        "collection_count": sum(bool(item.get("collection")) for item in schema_notes),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    quality = manifest["quality"]
    lines = [
        "# EvidenceGrounded AgentBench v0.4.1 Claim Schema 修正报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 修正点",
        "",
        "v0.4.1 不再把局部图直接粗略写入 `title` 字段，而是拆分为：",
        "",
        "```text",
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
        "```",
        "",
        "其中 `image_scope=partial_detail` 表示当前 crop 是原作品局部；`depicted_work_title` 表示原作品名；`displayed_region` 表示当前显示的区域。",
        "",
        "## 规模",
        "",
        f"- tasks_total：{quality['tasks_total']}",
        f"- task_split_counts：`{json.dumps(quality['task_split_counts'], ensure_ascii=False)}`",
        f"- sft_rows_total：{quality['sft_rows_total']}",
        f"- sft_split_counts：`{json.dumps(quality['sft_split_counts'], ensure_ascii=False)}`",
        f"- sft_action_counts：`{json.dumps(quality['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 字段统计",
        "",
        f"- claim_field_counts：`{json.dumps(quality['claim_field_counts'], ensure_ascii=False)}`",
        f"- image_scope_distribution：`{json.dumps(quality['image_scope_distribution'], ensure_ascii=False)}`",
        f"- object_type_distribution：`{json.dumps(quality['object_type_distribution'], ensure_ascii=False)}`",
        f"- title_from_caption_count：{quality['title_from_caption_count']}",
        f"- medium_dimensions_count：{quality['medium_dimensions_count']}",
        f"- collection_count：{quality['collection_count']}",
        "",
        "## 实际注意事项",
        "",
        "- `image_scope` 目前主要由图注规则推断，仍是 silver label，不是人工精标。",
        "- 对只有“图 1.12”这类弱图注的样本，作品名、展示范围、材质、馆藏可能仍需要 abstain。",
        "- 有些 PDF image block 是整页扫描或由多个 tile 组成，region proposal 在其他语料上可能没有当前 416 个样本这么高召回。",
        "- legacy evidence chunk 部分没有严格页码，citation 质量仍需后续升级。",
        "- 当前 oracle trajectory 是模板化专家示范，后续 rollout/RL 时需要 anti-loop verifier 防止重复 open/retrieve。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(f"/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_4_1_region_claim_schema_sft_{stamp}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
