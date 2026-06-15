#!/usr/bin/env python3
"""Build a clean v1.0.4 Core4 no-select SFT dataset from v1.0.3 tasks.

The builder removes the noisy displayed_region field, rebuilds Core4 claims,
adds a local_visual evidence anchor for object_type, and regenerates oracle SFT
rows with Core4 claim_state and prompts.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615"
)
DEFAULT_GOLD_EVAL_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/gold_eval_v1_0_4_caption_corrected_20260611_1830"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")

CORE4_FIELDS = ["caption_text", "depicted_work_title", "image_scope", "object_type"]
AVAILABLE_TOOLS = [
    "inspect_page",
    "crop_target",
    "retrieve_evidence",
    "open_evidence",
    "write_claims_chunk",
    "finish",
]
IMAGE_SCOPE_VALUES = {"full_work", "figure_or_plate", "partial_detail", "album_leaf", "scroll_section"}
OBJECT_TYPE_VALUES = {
    "landscape_painting",
    "painting",
    "calligraphy",
    "diagram",
    "architectural_detail",
    "artifact_or_object",
    "text_page_or_caption",
}
FIELD_SPEC = "caption_text|depicted_work_title|image_scope|object_type"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--gold-eval-dir", default=str(DEFAULT_GOLD_EVAL_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-review-per-reason", type=int, default=80)
    parser.add_argument(
        "--quality-mode",
        choices=["loose", "strict_single_caption"],
        default="loose",
        help=(
            "loose keeps the original v1.0.3 scale; strict_single_caption keeps only caption-marker "
            "tasks with a single figure label and writes the rest to filtered_out.jsonl."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / default_output_name(args.quality_mode)
    if not source_root.exists():
        raise FileNotFoundError(source_root)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    (output_dir / "sft").mkdir(parents=True)
    (output_dir / "episodes").mkdir(parents=True)
    (output_dir / "gold_eval").mkdir(parents=True)

    old_replay = load_old_replay(source_root)
    caption_overrides = load_caption_overrides(Path(args.gold_eval_dir))

    all_tasks: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    all_sft_rows: list[dict[str, Any]] = []
    all_filtered_out: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        src_path = source_root / f"{split}_tasks.jsonl"
        tasks = read_jsonl(src_path)
        split_tasks: list[dict[str, Any]] = []
        split_episodes: list[dict[str, Any]] = []
        split_sft: list[dict[str, Any]] = []
        split_filtered: list[dict[str, Any]] = []
        for task in tasks:
            transformed, task_reviews = transform_task(task, caption_overrides)
            quality = quality_gate(transformed, args.quality_mode)
            transformed["quality_gate"] = quality
            if not quality["keep"]:
                filtered = filtered_out_row(transformed, quality)
                split_filtered.append(filtered)
                all_filtered_out.append(filtered)
                review_rows.append(
                    review(
                        transformed,
                        "all",
                        "quality_filtered_" + str(quality.get("primary_reason") or "unknown"),
                        caption_from_task(transformed),
                    )
                )
                continue
            actions = build_oracle_actions(transformed, old_replay)
            sft_rows = build_sft_rows(transformed, actions, old_replay)
            split_tasks.append(transformed)
            split_episodes.append(
                {
                    "task_id": transformed["task_id"],
                    "source_task_id": transformed.get("source_task_id"),
                    "split": transformed.get("split"),
                    "variant": transformed.get("candidate_augmentation") or 0,
                    "actions": actions,
                }
            )
            split_sft.extend(sft_rows)
            review_rows.extend(task_reviews)
        write_jsonl(output_dir / f"{split}_tasks.jsonl", split_tasks)
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", split_episodes)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", split_sft)
        write_jsonl(output_dir / f"{split}_filtered_out.jsonl", split_filtered)
        all_tasks.extend(split_tasks)
        all_episodes.extend(split_episodes)
        all_sft_rows.extend(split_sft)

    write_jsonl(output_dir / "tasks_all.jsonl", all_tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", all_episodes)
    write_jsonl(output_dir / "sft" / "all.jsonl", all_sft_rows)
    write_jsonl(output_dir / "filtered_out.jsonl", all_filtered_out)

    gold_eval_summary = build_gold_eval_core4(Path(args.gold_eval_dir), output_dir / "gold_eval", caption_overrides)
    review_rows = cap_review_rows(review_rows, args.max_review_per_reason)
    write_jsonl(output_dir / "review_queue.jsonl", review_rows)

    summary = build_summary(
        args,
        source_root,
        output_dir,
        all_tasks,
        all_sft_rows,
        review_rows,
        all_filtered_out,
        gold_eval_summary,
    )
    write_json(output_dir / "manifest.json", summary)
    write_report(output_dir / "构建报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def default_output_name(quality_mode: str) -> str:
    suffix = "_strict_single" if quality_mode == "strict_single_caption" else ""
    return f"agentbench_v1_0_4_core4_clean_sft{suffix}_{datetime.now().strftime('%Y%m%d_%H%M')}"


def transform_task(
    task: dict[str, Any],
    caption_overrides: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = copy.deepcopy(task)
    task_id = str(row.get("task_id") or "")
    row["dataset_version"] = "v1.0.4_core4_clean"
    row["tool_schema_version"] = "v1.0.4_no_select_core4"
    row["goal"] = (
        "Inspect the PDF page, crop the target Chinese landscape-related figure, "
        "retrieve/open evidence, and write Core4 evidence-grounded claims."
    )
    row["available_tools"] = list(AVAILABLE_TOOLS)

    gold = row.setdefault("gold", {})
    caption = str(caption_overrides.get(task_id) or gold.get("caption_text") or local_caption_text(row) or "").strip()
    if caption:
        gold["caption_text"] = caption
        patch_local_caption(row, caption)
        patch_region_caption_hints(row, caption)
    ensure_local_visual(row)
    claims, reviews = build_core4_claims(row, caption)
    gold["claims"] = claims
    gold["target_claim_fields"] = list(CORE4_FIELDS)
    gold["claim_schema_fields"] = list(CORE4_FIELDS)
    gold["evidence_ids"] = dedupe(eid for claim in claims for eid in claim.get("evidence_ids", []))
    gold["candidate_evidence_ids"] = dedupe(eid for claim in claims for eid in claim.get("candidate_evidence_ids", []))
    gold["evidence_query"] = build_evidence_query(row, caption, claims)
    gold["core4_schema_version"] = "v1.0.4_core4_clean"
    gold["removed_fields"] = ["displayed_region"]
    return row, reviews


def build_core4_claims(task: dict[str, Any], caption: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reviews: list[dict[str, Any]] = []
    task_id = str(task.get("task_id") or "")
    local_caption_id = local_caption_evidence_id(task)
    local_visual_id = local_visual_evidence_id(task)
    candidate_ids = base_candidate_evidence_ids(task, local_caption_id, local_visual_id)

    claims: list[dict[str, Any]] = []
    if caption:
        claims.append(
            supported_claim("caption_text", caption, [local_caption_id], candidate_ids, 0.90, "target_local_caption")
        )
    else:
        claims.append(abstain_claim("caption_text", "页面文本层未给出可靠目标图注"))
        reviews.append(review(task, "caption_text", "missing_caption_text", caption))

    title, title_reason = infer_depicted_work_title(caption)
    if title:
        claims.append(
            supported_claim(
                "depicted_work_title",
                title,
                [local_caption_id],
                candidate_ids,
                0.84 if title_reason == "single_chinese_title" else 0.72,
                title_reason,
            )
        )
    else:
        claims.append(abstain_claim("depicted_work_title", "图注未明确给出可绑定目标图的作品名"))
        if title_reason:
            reviews.append(review(task, "depicted_work_title", title_reason, caption))

    scope, scope_reason = infer_image_scope(caption)
    if scope:
        claims.append(
            supported_claim("image_scope", scope, [local_caption_id], candidate_ids, 0.82, scope_reason)
        )
    else:
        claims.append(abstain_claim("image_scope", "证据未明确说明整图、图版、局部、册页或卷本片段"))
        reviews.append(review(task, "image_scope", "scope_uncertain", caption))

    object_type, object_reason, evidence_ids, confidence = infer_object_type(caption, local_caption_id, local_visual_id)
    if object_type:
        claims.append(
            supported_claim(
                "object_type",
                object_type,
                evidence_ids,
                candidate_ids,
                confidence,
                object_reason,
                visual_bbox=(task.get("gold") or {}).get("image_bbox") if local_visual_id in evidence_ids else None,
            )
        )
    else:
        claims.append(abstain_claim("object_type", "文本和规则视觉线索不足以可靠判断对象类型"))
        reviews.append(review(task, "object_type", object_reason or "object_type_uncertain", caption))

    for claim in claims:
        claim["core4_schema_version"] = "v1.0.4_core4_clean"
    if not local_caption_id:
        reviews.append(review(task, "all", "missing_local_caption_evidence_id", caption))
    if not local_visual_id:
        reviews.append(review(task, "object_type", "missing_local_visual_evidence_id", caption))
    if len(figure_labels(caption)) > 1:
        reviews.append(review(task, "caption_text", "multi_figure_label_caption_needs_vlm", caption))
    return claims, reviews


def infer_depicted_work_title(caption: str) -> tuple[str, str]:
    titles = [normalize_space(item) for item in re.findall(r"《([^》]{1,80})》", caption or "")]
    titles = [item for item in titles if item]
    valid_titles = [item for item in titles if len(re.sub(r"\s+", "", item)) >= 2]
    if titles and not valid_titles:
        return "", "invalid_short_title"
    titles = valid_titles
    unique = dedupe(titles)
    if len(unique) == 1:
        return unique[0], "single_chinese_title"
    if len(unique) > 1:
        return "", "multiple_titles_need_target_alignment"
    return "", "no_explicit_title"


def infer_image_scope(caption: str) -> tuple[str, str]:
    text = normalize_space(caption)
    low = text.lower()
    if re.search(r"(局部|细部|細部|部分|detail|details|partial|part of)", text, flags=re.I):
        return "partial_detail", "caption_explicit_partial_detail"
    if re.search(r"(全图|全圖|全幅|整幅|全卷|全本|whole|complete|entire)", text, flags=re.I):
        return "full_work", "caption_explicit_full_work"
    if re.search(r"(册页|冊頁|册之|冊之|图册|圖冊|album leaf|album)", text, flags=re.I):
        return "album_leaf", "caption_explicit_album_leaf"
    if re.search(r"(卷本|手卷|长卷|長卷|handscroll|scroll section)", text, flags=re.I):
        return "scroll_section", "caption_explicit_scroll_section"
    if caption_like(text):
        return "figure_or_plate", "caption_figure_or_plate_marker"
    if low.startswith(("fig", "figure", "plate")):
        return "figure_or_plate", "caption_figure_or_plate_marker"
    return "", ""


def infer_object_type(caption: str, local_caption_id: str, local_visual_id: str) -> tuple[str, str, list[str], float]:
    text = normalize_space(caption)
    evidence_caption = [local_caption_id] if local_caption_id else []
    evidence_visual = [local_visual_id] if local_visual_id else []
    both = dedupe(evidence_caption + evidence_visual)
    if re.search(r"(图式|圖式|示意图|示意圖|样式|樣式|结构图|結構圖|diagram|schema)", text, flags=re.I):
        return "diagram", "caption_diagram_terms", evidence_caption or both, 0.86
    if re.search(r"(桥栏|橋欄|栏杆|欄杆|斗拱|建筑构件|建築構件|建筑细部|建築細部)", text):
        return "architectural_detail", "caption_architectural_detail_terms", both, 0.80
    if re.search(r"(书法|書法|法帖|行书|行書|楷书|楷書|草书|草書|隶书|隸書|calligraphy)", text, flags=re.I):
        return "calligraphy", "caption_calligraphy_terms", evidence_caption or both, 0.84
    if re.search(r"(山水画|山水畫|山水图|山水圖|landscape painting|landscape)", text, flags=re.I):
        return "landscape_painting", "caption_landscape_painting_terms", both, 0.88
    if re.search(r"(hanging scroll|handscroll|album leaf|ink on|ink and color|color on silk|ink on paper|绢本|絹本|纸本|紙本|设色|設色)", text, flags=re.I):
        return "painting", "caption_painting_medium_terms", both, 0.80
    title, _ = infer_depicted_work_title(text)
    if title and re.search(r"(图|圖|画|畫|山水|松|溪|山|水|林|峰|壑|行旅|园|園)", title):
        return "painting", "title_artwork_visual_terms", evidence_visual or both, 0.72
    if caption_like(text):
        return "painting", "visual_anchor_from_target_crop_needs_spotcheck", evidence_visual or evidence_caption, 0.62
    return "", "object_type_uncertain", [], 0.0


def quality_gate(task: dict[str, Any], mode: str) -> dict[str, Any]:
    caption = caption_from_task(task)
    labels = figure_labels(caption)
    starts_marker = caption_starts_marker(caption)
    has_title = bool(re.search(r"《[^》]{1,80}》", caption or ""))
    body_terms = bool(
        re.search(r"(参考文献|本章小结|不足与展望|目录|总结|主要结论|创新之处)", caption or "")
    )
    reasons: list[str] = []
    if not caption:
        reasons.append("missing_caption_text")
    if body_terms:
        reasons.append("bad_section_terms")
    if mode == "strict_single_caption":
        if not starts_marker:
            reasons.append("caption_not_start_marker")
        if len(labels) > 1:
            reasons.append("multi_figure_caption")
        if caption_number_only(caption):
            reasons.append("caption_number_only")
        if caption_body_after_marker(caption):
            reasons.append("caption_body_after_marker")
        if caption_too_short_after_marker(caption):
            reasons.append("caption_too_short_after_marker")
    keep = not reasons
    return {
        "mode": mode,
        "keep": keep,
        "primary_reason": reasons[0] if reasons else "accepted",
        "reasons": reasons,
        "caption_starts_marker": starts_marker,
        "figure_label_count": len(labels),
        "figure_labels": labels,
        "caption_has_title": has_title,
        "caption_length": len(normalize_space(caption)),
    }


def filtered_out_row(task: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "caption_text": caption_from_task(task),
        "quality_gate": quality,
        "page_image": task.get("page_image"),
        "artwork_image": task.get("artwork_image"),
        "overlay_image": task.get("overlay_image"),
    }


def caption_from_task(task: dict[str, Any]) -> str:
    gold = task.get("gold") or {}
    caption = gold.get("caption_text")
    if caption:
        return str(caption)
    for claim in gold.get("claims") or []:
        if claim.get("field") == "caption_text" and not claim.get("abstain"):
            return str(claim.get("value") or "")
    return local_caption_text(task)


def supported_claim(
    field: str,
    value: Any,
    evidence_ids: list[str],
    candidate_ids: list[str],
    confidence: float,
    support_type: str,
    visual_bbox: Any = None,
) -> dict[str, Any]:
    evidence_ids = [eid for eid in dedupe(evidence_ids) if eid]
    return {
        "claim_id": field,
        "field": field,
        "value": value,
        "abstain": False,
        "evidence_ids": evidence_ids,
        "candidate_evidence_ids": dedupe(candidate_ids),
        "support_type": support_type,
        "confidence": round(float(confidence), 3),
        "visual_bbox": visual_bbox,
    }


def abstain_claim(field: str, reason: str) -> dict[str, Any]:
    return {"claim_id": field, "field": field, "value": None, "abstain": True, "reason": reason, "confidence": 0.72}


def build_oracle_actions(task: dict[str, Any], old_replay: dict[str, Any]) -> list[dict[str, Any]]:
    gold = task.get("gold") or {}
    local_caption_id = local_caption_evidence_id(task)
    local_visual_id = local_visual_evidence_id(task)
    target_region_id = target_region(task).get("region_id") or "r0"
    actions: list[dict[str, Any]] = [
        {"action": "inspect_page", "top_k": min(10, max(6, len(task.get("region_candidates") or [])))},
        {"action": "crop_target", "region_id": target_region_id},
    ]
    if local_caption_id:
        actions.append({"action": "open_evidence", "evidence_id": local_caption_id})
    if local_visual_id and any(local_visual_id in (claim.get("evidence_ids") or []) for claim in gold.get("claims") or []):
        actions.append({"action": "open_evidence", "evidence_id": local_visual_id})
    actions.append(
        {
            "action": "retrieve_evidence",
            "query": gold.get("evidence_query") or build_evidence_query(task, gold.get("caption_text") or "", gold.get("claims") or []),
            "scope": "same_document",
            "top_k": 5,
        }
    )
    external_id = first_old_external_open_id(str(task.get("task_id")), old_replay)
    if external_id:
        actions.append({"action": "open_evidence", "evidence_id": external_id})
    claim_by_field = {str(claim.get("field")): claim for claim in gold.get("claims") or []}
    for field in CORE4_FIELDS:
        claim = claim_by_field.get(field)
        if not claim:
            actions.append({"action": "write_claims_chunk", "claims": [], "abstains": [abstain_claim(field, "字段 gold 缺失")]})
        elif claim.get("abstain"):
            actions.append(
                {
                    "action": "write_claims_chunk",
                    "claims": [],
                    "abstains": [{"field": field, "reason": claim.get("reason") or "证据不足"}],
                }
            )
        else:
            actions.append(
                {
                    "action": "write_claims_chunk",
                    "claims": [
                        {
                            "field": field,
                            "value": claim.get("value"),
                            "evidence_ids": claim.get("evidence_ids") or [],
                            "visual_bbox": claim.get("visual_bbox"),
                            "confidence": claim.get("confidence", 0.75),
                        }
                    ],
                    "abstains": [],
                }
            )
    actions.append({"action": "finish", "status": "done"})
    return actions


def build_sft_rows(task: dict[str, Any], actions: list[dict[str, Any]], old_replay: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    draft_claims: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    images = [task.get("page_image")]
    for step, action in enumerate(actions):
        row = {
            "task_id": task.get("task_id"),
            "source_task_id": task.get("source_task_id"),
            "split": task.get("split"),
            "variant": task.get("candidate_augmentation") or 0,
            "step": step,
            "tool_schema_version": "v1.0.4_no_select_core4",
            "action": copy.deepcopy(action),
            "history": copy.deepcopy(history),
            "tool_results": copy.deepcopy(tool_results),
            "draft_claims": copy.deepcopy(draft_claims),
            "selected_evidence_ids": copy.deepcopy(selected_ids),
            "images": [item for item in images if item],
            "label_source": "v1_0_4_core4_clean_rule_sft",
            "claim_state": claim_state(draft_claims),
            "available_actions": phase_actions(action),
            "phase_name": "core4_" + str(action.get("action")),
            "phase_hint": phase_hint(action),
        }
        row["prompt_text"] = build_prompt_text(task, row)
        row["messages"] = build_messages(row)
        rows.append(row)
        result = result_for_action(task, action, old_replay, draft_claims)
        history.append(copy.deepcopy(action))
        tool_results.append(copy.deepcopy(result))
        if action.get("action") == "crop_target":
            crop = result.get("crop_path") or task.get("artwork_image")
            images = [task.get("page_image"), crop]
        if action.get("action") == "write_claims_chunk":
            draft_claims = apply_claim_write(draft_claims, action.get("claims") or [], action.get("abstains") or [])
    return rows


def result_for_action(
    task: dict[str, Any],
    action: dict[str, Any],
    old_replay: dict[str, Any],
    draft_claims: list[dict[str, Any]],
) -> dict[str, Any]:
    name = action.get("action")
    task_id = str(task.get("task_id"))
    if name == "inspect_page":
        return {
            "tool": "inspect_page",
            "page_image": task.get("page_image"),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "regions": public_region_candidates(task),
            "layout_regions": public_region_candidates(task),
        }
    if name == "crop_target":
        region = target_region(task)
        bbox = region.get("bbox") or (task.get("gold") or {}).get("image_bbox")
        return {
            "tool": "crop_target",
            "region_id": action.get("region_id"),
            "crop_mode": "region_id",
            "bbox": bbox,
            "crop_path": task.get("artwork_image"),
            "bbox_iou": 1.0,
        }
    if name == "open_evidence":
        evidence_id = str(action.get("evidence_id") or "")
        local = open_local_evidence(task, evidence_id)
        if local:
            return local
        return (old_replay.get("open_results", {}).get((task_id, evidence_id)) or {"tool": "open_evidence", "evidence_id": evidence_id, "error": "evidence not found in replay cache"})
    if name == "retrieve_evidence":
        old = old_replay.get("retrieve_results", {}).get(task_id)
        if old:
            old = copy.deepcopy(old)
            old["query"] = action.get("query")
            old["scope"] = action.get("scope")
            return old
        return {"tool": "retrieve_evidence", "query": action.get("query"), "scope": action.get("scope"), "results": [], "hit_evidence_ids": []}
    if name == "write_claims_chunk":
        next_claims = apply_claim_write(draft_claims, action.get("claims") or [], action.get("abstains") or [])
        return {
            "tool": "write_claims_chunk",
            "claims": action.get("claims") or [],
            "abstains": action.get("abstains") or [],
            "claim_state": claim_state(next_claims),
        }
    if name == "finish":
        return {"tool": "finish", "status": action.get("status", "done"), "draft_claims": draft_claims}
    return {"tool": name}


def build_prompt_text(task: dict[str, Any], row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "你是 evidence-grounded figure understanding 的 VLM tool-call agent。",
            "目标：先检查 PDF 页面布局，再裁剪目标图像，之后直接打开/检索可见证据，并写出 Core4 证据支撑 claim；本协议不使用 select_evidence。",
            f"task_id：{row.get('task_id')}；step：{row.get('step')}",
            f"source_file：{task.get('source_file', '')}；page：{task.get('page', '')}",
            f"输入图像：{len(row.get('images') or [])} 张。第 1 张通常是 PDF 页面；第 2 张通常是已裁剪目标图。",
            "Core4 target_fields：caption_text、depicted_work_title、image_scope、object_type。",
            "字段定义：caption_text=目标图注原文；depicted_work_title=目标作品名；image_scope=full_work|figure_or_plate|partial_detail|album_leaf|scroll_section；object_type=landscape_painting|painting|calligraphy|diagram|architectural_detail|artifact_or_object|text_page_or_caption。",
            "约束：只输出一个 JSON 对象；不要输出 markdown；不要编造事实；证据不足必须 abstain；每次 write_claims_chunk 最多处理一个 remaining field；remaining_fields 非空时禁止 finish。",
            '工具格式：{"action":"inspect_page","top_k":整数}；{"action":"crop_target","region_id":"r0"}；{"action":"open_evidence","evidence_id":"local_caption_xxx"}；{"action":"retrieve_evidence","query":"...","scope":"same_document","top_k":5}；{"action":"write_claims_chunk","claims":[{"field":"' + FIELD_SPEC + '","value":值,"evidence_ids":["完整 evidence_id"],"visual_bbox":null,"confidence":0到1}],"abstains":[{"field":"字段名","reason":"证据不足原因"}]}；{"action":"finish","status":"done"}',
            "历史动作：",
            json.dumps(row.get("history") or [], ensure_ascii=False, separators=(",", ":")),
            "工具返回摘要：",
            json.dumps(simplify_tool_results(row.get("tool_results") or []), ensure_ascii=False, separators=(",", ":")),
            "当前阶段允许的工具：",
            json.dumps(row.get("available_actions") or [], ensure_ascii=False, separators=(",", ":")),
            f"阶段提示：{row.get('phase_hint') or ''}",
            "当前 claim_state：",
            json.dumps(row.get("claim_state") or {}, ensure_ascii=False, separators=(",", ":")),
            "当前 claims：",
            json.dumps(row.get("draft_claims") or [], ensure_ascii=False, separators=(",", ":")),
            "当前可见 evidence_ids 包括 local_caption 和 local_visual；claim 的 evidence_ids 必须从可见或已打开/已检索结果中逐字符复制。",
            "请根据当前状态选择下一步工具调用。只输出一个 JSON 对象。",
        ]
    )


def build_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image in row.get("images") or []:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": row.get("prompt_text") or ""})
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": json.dumps(row.get("action") or {}, ensure_ascii=False, separators=(",", ":"))},
    ]


def simplify_tool_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for result in results[-6:]:
        if not isinstance(result, dict):
            continue
        item = {k: result.get(k) for k in ["tool", "evidence_id", "source_file", "page", "page_start", "status", "error"] if k in result}
        if result.get("tool") == "inspect_page":
            item["regions"] = result.get("regions", [])[:10]
        if result.get("tool") == "retrieve_evidence":
            item["query"] = result.get("query")
            item["scope"] = result.get("scope")
            item["results"] = [
                {
                    "evidence_id": r.get("evidence_id"),
                    "source_file": r.get("source_file"),
                    "page_start": r.get("page_start") if r.get("page_start") is not None else r.get("page"),
                    "display_snippet": truncate(str(r.get("display_snippet") or r.get("text") or ""), 180),
                }
                for r in (result.get("results") or [])[:3]
                if isinstance(r, dict)
            ]
        if result.get("display_snippet"):
            item["display_snippet"] = truncate(str(result.get("display_snippet")), 240)
        if result.get("claim_state"):
            item["claim_state"] = result.get("claim_state")
        out.append(item)
    return out


def build_gold_eval_core4(
    gold_eval_dir: Path,
    output_dir: Path,
    caption_overrides: dict[str, str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {"input_dir": str(gold_eval_dir), "splits": {}, "available": gold_eval_dir.exists()}
    if not gold_eval_dir.exists():
        return summary
    for filename in ["val_gold_50.jsonl", "test_gold_100.jsonl"]:
        path = gold_eval_dir / filename
        if not path.exists():
            continue
        rows = []
        review_rows = []
        for task in read_jsonl(path):
            transformed, reviews = transform_task(task, caption_overrides)
            rows.append(transformed)
            review_rows.extend(reviews)
        write_jsonl(output_dir / filename, rows)
        write_jsonl(output_dir / filename.replace(".jsonl", "_review_queue.jsonl"), review_rows)
        summary["splits"][filename] = {
            "tasks": len(rows),
            "review_rows": len(review_rows),
            "field_counts": field_summary(rows),
        }
    return summary


def load_old_replay(source_root: Path) -> dict[str, Any]:
    out = {"retrieve_results": {}, "open_results": {}, "external_open_ids": defaultdict(list)}
    for split in ["train", "val", "test"]:
        path = source_root / "sft" / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            task_id = str(row.get("task_id") or "")
            for result in row.get("tool_results") or []:
                if not isinstance(result, dict):
                    continue
                if result.get("tool") == "retrieve_evidence" and task_id not in out["retrieve_results"]:
                    out["retrieve_results"][task_id] = result
                if result.get("tool") == "open_evidence" and result.get("evidence_id") and not result.get("error"):
                    eid = str(result.get("evidence_id"))
                    out["open_results"][(task_id, eid)] = result
                    if not eid.startswith("local_caption_") and eid not in out["external_open_ids"][task_id]:
                        out["external_open_ids"][task_id].append(eid)
    return out


def load_caption_overrides(gold_eval_dir: Path) -> dict[str, str]:
    overrides: dict[str, str] = {}
    if not gold_eval_dir.exists():
        return overrides
    for filename in ["val_gold_50.jsonl", "test_gold_100.jsonl"]:
        path = gold_eval_dir / filename
        if not path.exists():
            continue
        for row in read_jsonl(path):
            caption = (row.get("gold") or {}).get("caption_text")
            if caption:
                overrides[str(row.get("task_id"))] = str(caption)
    return overrides


def first_old_external_open_id(task_id: str, old_replay: dict[str, Any]) -> str | None:
    ids = old_replay.get("external_open_ids", {}).get(task_id) or []
    return ids[0] if ids else None


def local_caption_text(task: dict[str, Any]) -> str:
    for item in task.get("local_evidence") or []:
        if str(item.get("evidence_id") or "").startswith("local_caption_"):
            return str(item.get("display_snippet") or item.get("text") or "")
    return ""


def patch_local_caption(task: dict[str, Any], caption: str) -> None:
    for item in task.get("local_evidence") or []:
        if str(item.get("evidence_id") or "").startswith("local_caption_"):
            item["display_snippet"] = caption
            item["text"] = caption


def patch_region_caption_hints(task: dict[str, Any], caption: str) -> None:
    local_id = local_caption_evidence_id(task)
    for region in task.get("region_candidates") or []:
        if local_id and region.get("caption_evidence_id") == local_id:
            for key in ["caption_hint", "linked_caption_text", "nearby_text"]:
                if key in region and region.get(key):
                    region[key] = caption


def ensure_local_visual(task: dict[str, Any]) -> None:
    task_id = str(task.get("task_id") or "")
    eid = f"local_visual_{task_id}"
    evidence = task.setdefault("local_evidence", [])
    if any(str(item.get("evidence_id")) == eid for item in evidence):
        return
    evidence.append(
        {
            "evidence_id": eid,
            "source_file": task.get("source_file"),
            "page_start": task.get("page"),
            "page_end": task.get("page"),
            "authority_level": "visual",
            "citation_level": "target_crop_visual",
            "source_quality": "target_crop_visual_anchor",
            "source_role": "local_visual",
            "evidence_type": "target_crop_visual",
            "display_snippet": "目标裁剪图像的本地视觉证据；可用于支持 object_type，以及在图像范围明确时辅助 image_scope。",
            "bbox": (task.get("gold") or {}).get("image_bbox"),
            "image_path": task.get("artwork_image"),
            "adjudicated_claim_allowed_fields": ["object_type", "image_scope"],
            "usable_for_claim_by_adjudication": True,
        }
    )


def local_caption_evidence_id(task: dict[str, Any]) -> str:
    for item in task.get("local_evidence") or []:
        eid = str(item.get("evidence_id") or "")
        if eid.startswith("local_caption_"):
            return eid
    return ""


def local_visual_evidence_id(task: dict[str, Any]) -> str:
    for item in task.get("local_evidence") or []:
        eid = str(item.get("evidence_id") or "")
        if eid.startswith("local_visual_"):
            return eid
    return ""


def base_candidate_evidence_ids(task: dict[str, Any], local_caption_id: str, local_visual_id: str) -> list[str]:
    ids: list[str] = [local_caption_id, local_visual_id]
    gold = task.get("gold") or {}
    ids.extend(str(eid) for eid in gold.get("candidate_evidence_ids") or [])
    for claim in gold.get("claims") or []:
        ids.extend(str(eid) for eid in claim.get("candidate_evidence_ids") or [])
        ids.extend(str(eid) for eid in claim.get("evidence_ids") or [])
    return [eid for eid in dedupe(ids) if eid]


def build_evidence_query(task: dict[str, Any], caption: str, claims: list[dict[str, Any]]) -> str:
    values = [str(task.get("source_stem") or ""), caption]
    for claim in claims:
        if not claim.get("abstain") and claim.get("field") in {"depicted_work_title", "image_scope", "object_type"}:
            values.append(str(claim.get("value")))
    return " ".join(item for item in dedupe(normalize_space(v) for v in values) if item)[:500]


def target_region(task: dict[str, Any]) -> dict[str, Any]:
    regions = task.get("region_candidates") or []
    for item in regions:
        if item.get("is_target") or item.get("target_region_rank") == 1:
            return item
    return regions[0] if regions else {"region_id": "r0", "bbox": (task.get("gold") or {}).get("image_bbox")}


def public_region_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    hidden = {"is_target", "target_iou", "gold_iou", "source_task_id", "source_gold_bbox", "debug_reason"}
    return [{key: value for key, value in item.items() if key not in hidden} for item in task.get("region_candidates") or []]


def open_local_evidence(task: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    for item in task.get("local_evidence") or []:
        if str(item.get("evidence_id")) != evidence_id:
            continue
        return {
            "tool": "open_evidence",
            "evidence_id": evidence_id,
            "source_file": item.get("source_file"),
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "authority_level": item.get("authority_level"),
            "citation_level": item.get("citation_level"),
            "source_role": item.get("source_role"),
            "adjudicated_claim_allowed_fields": item.get("adjudicated_claim_allowed_fields"),
            "usable_for_claim_by_adjudication": item.get("usable_for_claim_by_adjudication"),
            "display_snippet": item.get("display_snippet") or item.get("text"),
        }
    return None


def apply_claim_write(
    draft_claims: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    abstains: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    next_claims = [copy.deepcopy(item) for item in draft_claims]
    for claim in claims:
        item = {
            "field": claim.get("field"),
            "value": claim.get("value"),
            "evidence_ids": claim.get("evidence_ids") or [],
            "visual_bbox": claim.get("visual_bbox"),
            "confidence": claim.get("confidence", 0.7),
            "abstain": False,
        }
        next_claims = [old for old in next_claims if old.get("field") != item["field"]] + [item]
    for abstain in abstains:
        item = {"field": abstain.get("field"), "reason": abstain.get("reason"), "abstain": True}
        next_claims = [old for old in next_claims if old.get("field") != item["field"]] + [item]
    return next_claims


def claim_state(draft_claims: list[dict[str, Any]]) -> dict[str, Any]:
    by_field = {str(item.get("field")): item for item in draft_claims if item.get("field")}
    written = [field for field in CORE4_FIELDS if field in by_field and not by_field[field].get("abstain")]
    abstained = [field for field in CORE4_FIELDS if field in by_field and by_field[field].get("abstain")]
    evidence_ids = dedupe(
        str(eid) for item in draft_claims for eid in (item.get("evidence_ids") or []) if isinstance(item, dict)
    )
    return {
        "target_fields": list(CORE4_FIELDS),
        "written_fields": written,
        "abstained_fields": abstained,
        "remaining_fields": [field for field in CORE4_FIELDS if field not in by_field],
        "claim_count": len(written),
        "abstain_count": len(abstained),
        "evidence_ids": evidence_ids,
    }


def phase_actions(action: dict[str, Any]) -> list[str]:
    name = str(action.get("action") or "")
    return [name] if name else []


def phase_hint(action: dict[str, Any]) -> str:
    return {
        "inspect_page": "先读取页面布局候选，不要直接裁剪、检索或写 claim。",
        "crop_target": "根据 inspect_page 返回的目标候选区域裁剪目标图像。",
        "open_evidence": "打开已经可见或已检索到的 evidence_id。",
        "retrieve_evidence": "根据目标图注和已裁剪图像构造检索 query，补充同文档证据。",
        "write_claims_chunk": "一次只写入或 abstain 一个 Core4 remaining field。",
        "finish": "只有 Core4 remaining_fields 为空时才能结束。",
    }.get(str(action.get("action") or ""), "")


def caption_like(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return bool(caption_starts_marker(normalized))


def caption_starts_marker(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    return bool(
        re.match(
            r"^(?:〔?\[?【?(?:图|圖)[一二三四五六七八九十百〇零0-9]+|"
            r"Fig\.?[A-Za-z]?[0-9IVXivx]+|Figure[A-Za-z]?[0-9IVXivx]+|Plate[A-Za-z]?[0-9IVXivx]+|"
            r"PLATE[A-Za-z]?[0-9IVXivx]+)",
            normalized,
            flags=re.I,
        )
    )


def caption_number_only(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or "")).strip("。.;；")
    if not normalized:
        return False
    return bool(
        re.fullmatch(
            r"(?:〔?\[?【?(?:图|圖)[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：][一二三四五六七八九十百〇零0-9]+)*[a-zA-Z]?[〕\]】]?|"
            r"(?:Fig\.?|Figure|Plate)[A-Za-z]?[0-9IVXivx]+(?:[.\-．:：][0-9IVXivx]+)*[a-zA-Z]?)",
            normalized,
            flags=re.I,
        )
    )


def caption_too_short_after_marker(text: str) -> bool:
    remainder = caption_remainder_after_first_label(text)
    if not remainder:
        return True
    compact = re.sub(r"\s+", "", remainder)
    if len(compact) <= 3:
        return True
    # OCR tails such as "JOb" after "Figure 16." are labels, not captions.
    if len(compact) <= 6 and re.fullmatch(r"[A-Za-z0-9IVXivx.\-．:：]+", compact):
        return True
    return False


def caption_remainder_after_first_label(text: str) -> str:
    compact = re.sub(r"\s+", "", str(text or ""))
    pattern = (
        r"^(?:〔?\[?【?(?:图|圖)[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：][一二三四五六七八九十百〇零0-9]+)*[a-zA-Z]?[〕\]】]?|"
        r"(?:Fig\.?|Figure|Plate)[A-Za-z]?[0-9IVXivx]+(?:[.\-．:：][0-9IVXivx]+)*[a-zA-Z]?)"
    )
    return re.sub(pattern, "", compact, count=1, flags=re.I).strip("：:.-．、，。;；")


def caption_body_after_marker(text: str) -> bool:
    compact = re.sub(r"\s+", "", str(text or ""))
    return bool(
        re.match(
            r"^〔?\[?【?(?:图|圖)[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：][一二三四五六七八九十百〇零0-9]+)*[〕\]】]?[。。，，；;]",
            compact,
        )
    )


def figure_labels(text: str) -> list[str]:
    labels = []
    pattern = (
        r"(?:图|圖)\s*[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：]\s*[一二三四五六七八九十百〇零0-9]+)*"
        r"|(?:fig\.?|figure|plate)\s*[A-Za-z]?\s*[0-9IVXivx]+(?:[.\-．:：]\s*[0-9IVXivx]+)*"
    )
    for match in re.finditer(pattern, text or "", flags=re.I):
        labels.append(re.sub(r"\s+", "", match.group(0).lower()).replace("圖", "图"))
    return dedupe(labels)


def review(task: dict[str, Any], field: str, reason: str, caption: str) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "field": field,
        "reason": reason,
        "caption_text": caption,
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "page_image": task.get("page_image"),
        "artwork_image": task.get("artwork_image"),
        "overlay_image": task.get("overlay_image"),
    }


def cap_review_rows(rows: list[dict[str, Any]], max_per_reason: int) -> list[dict[str, Any]]:
    if max_per_reason <= 0:
        return rows
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("reason") or "")].append(row)
    out: list[dict[str, Any]] = []
    for reason in sorted(buckets):
        out.extend(buckets[reason][:max_per_reason])
    return out


def build_summary(
    args: argparse.Namespace,
    source_root: Path,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    filtered_out: list[dict[str, Any]],
    gold_eval_summary: dict[str, Any],
) -> dict[str, Any]:
    split_counts = Counter(str(task.get("split")) for task in tasks)
    sft_split_counts = Counter(str(row.get("split")) for row in sft_rows)
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in sft_rows)
    field_counts = field_summary(tasks)
    caption_quality = caption_quality_summary(tasks)
    doc_by_split: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        doc_by_split[str(task.get("split"))].add(str(task.get("source_file")))
    review_counts = Counter(str(row.get("reason")) for row in review_rows)
    filtered_counts = Counter(str((row.get("quality_gate") or {}).get("primary_reason") or "unknown") for row in filtered_out)
    filtered_split_counts = Counter(str(row.get("split")) for row in filtered_out)
    return {
        "created_at": now(),
        "dataset_version": "v1.0.4_core4_clean_sft",
        "quality_mode": args.quality_mode,
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "gold_eval_input_dir": str(args.gold_eval_dir),
        "target_claim_fields": list(CORE4_FIELDS),
        "removed_fields": ["displayed_region"],
        "image_scope_values": sorted(IMAGE_SCOPE_VALUES),
        "object_type_values": sorted(OBJECT_TYPE_VALUES),
        "split_counts": dict(split_counts),
        "doc_counts_by_split": {split: len(docs) for split, docs in doc_by_split.items()},
        "doc_split_violations": doc_split_violations(tasks),
        "sft_rows_total": len(sft_rows),
        "sft_split_counts": dict(sft_split_counts),
        "sft_action_counts": dict(action_counts),
        "field_counts": field_counts,
        "caption_quality_summary": caption_quality,
        "filtered_out_rows": len(filtered_out),
        "filtered_out_split_counts": dict(filtered_split_counts),
        "filtered_out_reason_counts": dict(filtered_counts),
        "review_queue_rows": len(review_rows),
        "review_reason_counts": dict(review_counts),
        "gold_eval_core4": gold_eval_summary,
        "artifacts": {
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
            "sft_train": str(output_dir / "sft" / "train.jsonl"),
            "sft_val": str(output_dir / "sft" / "val.jsonl"),
            "sft_test": str(output_dir / "sft" / "test.jsonl"),
            "review_queue": str(output_dir / "review_queue.jsonl"),
            "filtered_out": str(output_dir / "filtered_out.jsonl"),
            "gold_eval_dir": str(output_dir / "gold_eval"),
            "report": str(output_dir / "构建报告.md"),
        },
        "notes": [
            "本轮不在线调用 LLM/VLM；规则高置信构建 Core4，疑难样本进入 review_queue。",
            "displayed_region 已从 target_claim_fields 移除。",
            "object_type 不再使用低信息值“图像”；优先使用受控枚举，证据不足则 abstain。",
            "新增 local_visual evidence anchor，使视觉判断字段不必伪装为 local_caption 文本支持。",
        ],
    }


def caption_quality_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    lengths: list[int] = []
    for task in tasks:
        caption = caption_from_task(task)
        labels = figure_labels(caption)
        counts["total"] += 1
        counts["caption_starts_marker"] += bool(caption_starts_marker(caption))
        counts["caption_has_title"] += bool(re.search(r"《[^》]{1,80}》", caption or ""))
        counts["multi_figure_caption"] += len(labels) > 1
        counts["caption_len_gt_150"] += len(normalize_space(caption)) > 150
        lengths.append(len(normalize_space(caption)))
    return {
        **dict(counts),
        "caption_len_min": min(lengths) if lengths else 0,
        "caption_len_max": max(lengths) if lengths else 0,
        "caption_len_avg": round(sum(lengths) / max(1, len(lengths)), 2),
    }


def field_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter()
    non = Counter()
    abst = Counter()
    values: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_prefix: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        for claim in (task.get("gold") or {}).get("claims") or []:
            field = str(claim.get("field") or "")
            if not field:
                continue
            total[field] += 1
            if claim.get("abstain"):
                abst[field] += 1
            else:
                non[field] += 1
                values[field][str(claim.get("value"))] += 1
                for eid in claim.get("evidence_ids") or []:
                    evidence_prefix[field][str(eid).split("_", 2)[0] + "_" + str(eid).split("_", 2)[1] if "_" in str(eid) else str(eid)] += 1
    return {
        "total": dict(total),
        "non_abstain": dict(non),
        "abstain": dict(abst),
        "top_values": {field: values[field].most_common(20) for field in sorted(values)},
        "evidence_prefix_counts": {field: dict(evidence_prefix[field]) for field in sorted(evidence_prefix)},
    }


def doc_split_violations(tasks: list[dict[str, Any]]) -> dict[str, list[str]]:
    by_doc: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        by_doc[str(task.get("source_file"))].add(str(task.get("split")))
    return {doc: sorted(splits) for doc, splits in by_doc.items() if len(splits) > 1}


def write_report(path: Path, summary: dict[str, Any]) -> None:
    strict = summary.get("quality_mode") == "strict_single_caption"
    quality_line = (
        "- 已启用 strict_single_caption 质量门控：只保留以图/Fig/Figure/Plate 编号开头、且只包含一个图注编号的主训练样本。"
        if strict
        else "- 已构建 loose 全量 Core4 clean SFT 数据集，保留现有 1500/200/200 量级。"
    )
    lines = [
        "# v1.0.4 Core4 Clean SFT 构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 来源：`{summary['source_root']}`",
        f"- quality_mode：`{summary.get('quality_mode')}`",
        "",
        "## 结论",
        "",
        quality_line,
        "- 已移除 `displayed_region`。",
        "- `object_type` 改为受控枚举，不再写 `图像`。",
        "- 本轮未在线调用 LLM/VLM，疑难样本写入 `review_queue.jsonl`；质量门控过滤样本写入 `filtered_out.jsonl`。",
        "",
        "## 规模",
        "",
        f"- split_counts: `{json.dumps(summary['split_counts'], ensure_ascii=False)}`",
        f"- filtered_out_rows: {summary.get('filtered_out_rows', 0)}",
        f"- filtered_out_split_counts: `{json.dumps(summary.get('filtered_out_split_counts', {}), ensure_ascii=False)}`",
        f"- filtered_out_reason_counts: `{json.dumps(summary.get('filtered_out_reason_counts', {}), ensure_ascii=False)}`",
        f"- sft_rows_total: {summary['sft_rows_total']}",
        f"- sft_split_counts: `{json.dumps(summary['sft_split_counts'], ensure_ascii=False)}`",
        f"- sft_action_counts: `{json.dumps(summary['sft_action_counts'], ensure_ascii=False)}`",
        f"- doc_counts_by_split: `{json.dumps(summary['doc_counts_by_split'], ensure_ascii=False)}`",
        f"- doc_split_violations: `{json.dumps(summary['doc_split_violations'], ensure_ascii=False)}`",
        "",
        "## 字段分布",
        "",
        f"- total: `{json.dumps(summary['field_counts']['total'], ensure_ascii=False)}`",
        f"- non_abstain: `{json.dumps(summary['field_counts']['non_abstain'], ensure_ascii=False)}`",
        f"- abstain: `{json.dumps(summary['field_counts']['abstain'], ensure_ascii=False)}`",
        f"- caption_quality_summary: `{json.dumps(summary.get('caption_quality_summary', {}), ensure_ascii=False)}`",
        "",
        "### Top Values",
        "",
    ]
    for field, vals in summary["field_counts"]["top_values"].items():
        lines.append(f"- {field}: `{json.dumps(vals[:10], ensure_ascii=False)}`")
    lines.extend(
        [
            "",
            "## Review Queue",
            "",
            f"- rows: {summary['review_queue_rows']}",
            f"- reason_counts: `{json.dumps(summary['review_reason_counts'], ensure_ascii=False)}`",
            "",
            "## GoldEval Core4",
            "",
            f"`{json.dumps(summary['gold_eval_core4'], ensure_ascii=False, indent=2)}`",
            "",
            "## 后续建议",
            "",
            "1. 先抽查 `review_queue.jsonl` 中 object_type 与多图注样本。",
            "2. 对 review queue 做小规模 VLM/LLM 裁决补丁，而不是全量重跑。",
            "3. 用该数据集训练 fresh Core4 LoRA，并用当前 best adapter 做 continued Core4 repair 对照。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def truncate(text: str, n: int) -> str:
    text = normalize_space(text)
    return text[:n] + ("..." if len(text) > n else "")


def dedupe(items: Any) -> list[Any]:
    out = []
    seen = set()
    for item in items:
        if item is None or item == "":
            continue
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
