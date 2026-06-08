#!/usr/bin/env python3
"""Build v0.4 no-highlight region-selection AgentBench and trajectory SFT data."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.actions import bbox_iou  # noqa: E402
from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation, build_prompt_text  # noqa: E402
from evidence_agent_env.tools.crop import crop_image, image_size  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_3_template_highlighted_sft_20260531_0504"
)
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--evidence-index", default=str(DEFAULT_EVIDENCE_INDEX))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-augmentations", type=int, default=3)
    parser.add_argument("--eval-augmentations", type=int, default=1)
    parser.add_argument("--top-k-regions", type=int, default=8)
    parser.add_argument("--max-text-distractors", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max-steps", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)

    base_tasks = read_jsonl(source_dir / "tasks_all.jsonl")
    corrected = load_step0_crop_targets(source_dir)
    source_sft = load_source_sft_by_task(source_dir)

    runtime_tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []

    for source_index, task in enumerate(base_tasks):
        task_id = str(task.get("task_id"))
        if task_id not in corrected or task_id not in source_sft:
            continue
        aug_count = args.train_augmentations if task.get("split") == "train" else args.eval_augmentations
        for aug_index in range(aug_count):
            rng = random.Random(args.seed + source_index * 997 + aug_index * 31)
            runtime_task, quality = build_runtime_task(task, corrected[task_id], args, aug_index, rng)
            runtime_tasks.append(runtime_task)
            quality_rows.append(quality)
            episodes.append(
                {
                    "task_id": runtime_task["task_id"],
                    "source_task_id": task_id,
                    "split": runtime_task.get("split"),
                    "variant": aug_index,
                    "actions": build_oracle_actions(task, runtime_task, source_sft[task_id]),
                }
            )

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episodes_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in runtime_tasks:
        by_split[str(task.get("split"))].append(task)
    for episode in episodes:
        episodes_by_split[str(episode.get("split"))].append(episode)

    write_jsonl(output_dir / "tasks_all.jsonl", runtime_tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)
    for split, rows in sorted(by_split.items()):
        write_jsonl(output_dir / f"{split}_tasks.jsonl", rows)
    for split, rows in sorted(episodes_by_split.items()):
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", rows)

    sft_rows = build_sft_rows(output_dir, runtime_tasks, source_sft)
    sft_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sft_rows:
        sft_by_split[str(row.get("split"))].append(row)
    for split, rows in sorted(sft_by_split.items()):
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "sft" / "all.jsonl", sft_rows)

    quality = summarize_quality(runtime_tasks, episodes, sft_rows, quality_rows)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.4_region_no_highlight",
        "source_dir": str(source_dir),
        "evidence_index": args.evidence_index,
        "output_dir": str(output_dir),
        "args": vars(args),
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


def load_step0_crop_targets(source_dir: Path) -> dict[str, dict[str, Any]]:
    targets: dict[str, dict[str, Any]] = {}
    for split in ["train", "val", "test"]:
        for row in read_jsonl(source_dir / "sft" / f"{split}.jsonl"):
            action = row.get("action") or {}
            if row.get("step") == 0 and action.get("action") == "crop_image":
                targets[str(row.get("task_id"))] = {"bbox": action.get("bbox"), "split": split}
    return targets


def load_source_sft_by_task(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split in ["train", "val", "test"]:
        for row in read_jsonl(source_dir / "sft" / f"{split}.jsonl"):
            rows_by_task[str(row.get("task_id"))].append(row)
    for rows in rows_by_task.values():
        rows.sort(key=lambda item: int(item.get("step", 0)))
    return rows_by_task


def build_runtime_task(
    task: dict[str, Any],
    corrected: dict[str, Any],
    args: argparse.Namespace,
    aug_index: int,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_task_id = str(task.get("task_id"))
    gold_bbox = corrected["bbox"]
    candidates_raw = propose_pdf_regions(task, gold_bbox, args.max_text_distractors)
    candidates = assign_region_ids(candidates_raw, gold_bbox, args.top_k_regions, rng)
    best = max(candidates, key=lambda item: item.get("target_iou", -1.0))
    item = copy.deepcopy(task)
    item["task_id"] = f"{source_task_id}_r{aug_index:02d}"
    item["source_task_id"] = source_task_id
    item["dataset_version"] = "v0.4_region_no_highlight"
    item["runtime_mode"] = "no_highlight_region_selection"
    item["goal"] = (
        "Use the original PDF page, proposed regions, cropped target figure, scoped evidence retrieval, "
        "and opened evidence snippets to write evidence-grounded claims for the target Chinese landscape figure."
    )
    item["page_image"] = task["page_image"]
    item["highlighted_page_image"] = None
    item["region_candidates"] = candidates
    item["gold"] = copy.deepcopy(task.get("gold") or {})
    item["gold"]["v0_3_original_image_bbox"] = item["gold"].get("image_bbox")
    item["gold"]["image_bbox"] = gold_bbox
    item["gold"]["target_region_id"] = best["region_id"]
    item["gold"]["target_region_iou"] = best.get("target_iou")
    item["gold"]["target_region_bbox"] = best.get("bbox")
    item["candidate_augmentation"] = {
        "variant": aug_index,
        "shuffle_seeded": True,
        "top_k_regions": args.top_k_regions,
    }
    quality = {
        "task_id": item["task_id"],
        "source_task_id": source_task_id,
        "split": item.get("split"),
        "candidate_count": len(candidates),
        "target_region_id": best["region_id"],
        "target_iou": best.get("target_iou"),
        "page_image_has_highlight": "highlight" in str(item.get("page_image", "")).lower(),
        "pdf_image_candidates": sum(1 for cand in candidates if cand.get("source") == "pdf_image_block"),
    }
    return item, quality


def propose_pdf_regions(task: dict[str, Any], gold_bbox: list[int], max_text_distractors: int) -> list[dict[str, Any]]:
    page_image = Path(task["page_image"])
    with Image.open(page_image) as image:
        page_width, page_height = image.size
    image_blocks: list[dict[str, Any]] = []
    text_blocks: list[dict[str, Any]] = []
    try:
        doc = fitz.open(str(task["source_path"]))
        page = doc[int(task["page"]) - 1]
        scale_x = page_width / page.rect.width
        scale_y = page_height / page.rect.height
        for block_index, block in enumerate(page.get_text("dict").get("blocks", [])):
            bbox = scale_bbox(block.get("bbox"), scale_x, scale_y, page_width, page_height)
            if not bbox:
                continue
            if block.get("type") == 1:
                image_blocks.append(
                    {
                        "bbox": bbox,
                        "source": "pdf_image_block",
                        "type": "figure_candidate",
                        "score": round(area(bbox) / max(1, page_width * page_height), 6),
                        "hint": "PDF 原生 image block；可能是插图、图像、logo 或扫描图块",
                    }
                )
            elif block.get("type") == 0:
                text = block_text(block)
                if text:
                    text_blocks.append(
                        {
                            "bbox": bbox,
                            "source": "pdf_text_block",
                            "type": "text_or_caption_candidate",
                            "score": round(area(bbox) / max(1, page_width * page_height), 6),
                            "nearby_text": text[:160],
                            "hint": "PDF 文本块；可能是标题、正文、图注或页眉页脚",
                        }
                    )
    except Exception:
        pass

    candidates = dedupe_regions(image_blocks, iou_threshold=0.92)
    if not candidates:
        candidates.append(
            {
                "bbox": gold_bbox,
                "source": "silver_gold_fallback",
                "type": "figure_candidate",
                "score": 1.0,
                "hint": "PDF image block extraction failed; fallback used only to keep the dataset executable",
            }
        )

    text_blocks.sort(key=lambda item: distance_to_bbox(item["bbox"], gold_bbox))
    candidates.extend(text_blocks[:max(0, max_text_distractors)])
    candidates.extend(grid_distractors(page_width, page_height, gold_bbox))
    for cand in candidates:
        cand["target_iou"] = bbox_iou(cand.get("bbox"), gold_bbox)
        cand["is_target"] = cand["target_iou"] >= 0.5
    return candidates


def assign_region_ids(
    candidates: list[dict[str, Any]],
    gold_bbox: list[int],
    top_k: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates = sorted(candidates, key=lambda item: (item.get("target_iou", -1.0), item.get("score", 0.0)), reverse=True)
    best = copy.deepcopy(candidates[0])
    rest = [copy.deepcopy(item) for item in candidates[1:]]
    rng.shuffle(rest)
    selected = [best] + rest[: max(0, top_k - 1)]
    rng.shuffle(selected)
    selected = sorted(selected, key=lambda item: item.get("source") == "pdf_text_block")
    for index, item in enumerate(selected):
        item["region_id"] = f"r{index}"
        item["gold_iou"] = bbox_iou(item.get("bbox"), gold_bbox)
    return selected


def build_oracle_actions(
    source_task: dict[str, Any],
    runtime_task: dict[str, Any],
    source_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_region_id = runtime_task["gold"]["target_region_id"]
    corrected_bbox = runtime_task["gold"]["image_bbox"]
    original_bbox = runtime_task["gold"].get("v0_3_original_image_bbox")
    actions: list[dict[str, Any]] = [
        {"action": "propose_regions", "top_k": len(runtime_task.get("region_candidates") or [])},
        {"action": "crop_region", "region_id": target_region_id},
    ]
    for row in source_rows:
        action = copy.deepcopy(row.get("action") or {})
        if action.get("action") == "crop_image":
            continue
        rewrite_action_bbox(action, original_bbox, corrected_bbox)
        actions.append(action)
    return actions


def rewrite_action_bbox(action: dict[str, Any], original_bbox: Any, corrected_bbox: list[int]) -> None:
    if action.get("visual_bbox") == original_bbox:
        action["visual_bbox"] = corrected_bbox
    anchor = action.get("anchor")
    if isinstance(anchor, dict) and anchor.get("bbox") == original_bbox:
        anchor["bbox"] = corrected_bbox


def build_sft_rows(
    output_dir: Path,
    runtime_tasks: list[dict[str, Any]],
    source_sft: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prompt_config = PromptConfig(tool_schema="region", coordinate_info=True)
    for runtime_task in runtime_tasks:
        source_task_id = str(runtime_task["source_task_id"])
        source_rows = source_sft[source_task_id]
        source_step0 = source_rows[0]
        corrected_bbox = runtime_task["gold"]["image_bbox"]
        original_bbox = runtime_task["gold"].get("v0_3_original_image_bbox")
        propose_action = {"action": "propose_regions", "top_k": len(runtime_task.get("region_candidates") or [])}
        crop_action = {"action": "crop_region", "region_id": runtime_task["gold"]["target_region_id"]}
        propose_result = {"tool": "propose_regions", "regions": public_region_candidates(runtime_task)}
        crop_path = output_dir / "crops" / f"{runtime_task['task_id']}.jpg"
        crop_result = crop_image(runtime_task["page_image"], runtime_task["gold"]["target_region_bbox"], crop_path)
        crop_result = {
            "tool": "crop_region",
            "region_id": runtime_task["gold"]["target_region_id"],
            **crop_result,
            "bbox_iou": bbox_iou(corrected_bbox, crop_result["bbox"]),
        }

        obs0 = make_obs(runtime_task, [], [], [], None)
        rows.append(make_sft_row(runtime_task, 0, propose_action, obs0, prompt_config))

        obs1 = make_obs(runtime_task, [propose_action], [propose_result], [], None)
        rows.append(make_sft_row(runtime_task, 1, crop_action, obs1, prompt_config))

        for source_row in source_rows[1:]:
            action = copy.deepcopy(source_row.get("action") or {})
            rewrite_action_bbox(action, original_bbox, corrected_bbox)
            history = transform_history(source_row.get("history") or [], propose_action, crop_action, original_bbox, corrected_bbox)
            tool_results = transform_tool_results(
                source_row.get("tool_results") or [],
                propose_result,
                crop_result,
                original_bbox,
                corrected_bbox,
            )
            draft_claims = transform_claims(source_row.get("draft_claims") or [], original_bbox, corrected_bbox)
            obs = make_obs(runtime_task, history, tool_results, draft_claims, str(crop_path))
            rows.append(make_sft_row(runtime_task, int(source_row.get("step", 0)) + 1, action, obs, prompt_config))
    return rows


def make_obs(
    task: dict[str, Any],
    history: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    draft_claims: list[dict[str, Any]],
    crop_path: str | None,
) -> dict[str, Any]:
    images = [{"role": "page_image", "path": task["page_image"]}]
    if crop_path:
        images.append({"role": "last_crop", "path": crop_path})
    return {
        "task_id": task["task_id"],
        "goal": task.get("goal"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "images": images,
        "page_size": image_size(task["page_image"]),
        "history": history[-8:],
        "tool_results": tool_results[-6:],
        "draft_claims": draft_claims,
    }


def make_sft_row(
    task: dict[str, Any],
    step: int,
    action: dict[str, Any],
    obs: dict[str, Any],
    prompt_config: PromptConfig,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "source_task_id": task["source_task_id"],
        "split": task.get("split"),
        "variant": task.get("candidate_augmentation", {}).get("variant"),
        "step": step,
        "tool_schema_version": "v0.4_region_no_highlight",
        "action": action,
        "history": copy.deepcopy(obs.get("history") or []),
        "tool_results": copy.deepcopy(obs.get("tool_results") or []),
        "draft_claims": copy.deepcopy(obs.get("draft_claims") or []),
        "images": [item.get("path") for item in obs.get("images") or [] if item.get("path")],
        "prompt_text": build_prompt_text(obs, prompt_config),
        "messages": build_messages_from_observation(obs, prompt_config, include_assistant_action=action),
        "label_source": "v0_3_3_corrected_bbox_to_v0_4_pdf_region_selection",
    }


def public_region_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    hidden = {"is_target", "target_iou", "gold_iou", "source_task_id", "source_gold_bbox", "debug_reason"}
    return [{key: value for key, value in item.items() if key not in hidden} for item in task.get("region_candidates") or []]


def transform_history(
    source_history: list[Any],
    propose_action: dict[str, Any],
    crop_action: dict[str, Any],
    original_bbox: Any,
    corrected_bbox: list[int],
) -> list[dict[str, Any]]:
    transformed = [copy.deepcopy(propose_action), copy.deepcopy(crop_action)]
    for action in source_history[1:]:
        item = copy.deepcopy(action)
        rewrite_action_bbox(item, original_bbox, corrected_bbox)
        transformed.append(item)
    return transformed


def transform_tool_results(
    source_results: list[Any],
    propose_result: dict[str, Any],
    crop_result: dict[str, Any],
    original_bbox: Any,
    corrected_bbox: list[int],
) -> list[dict[str, Any]]:
    transformed = [copy.deepcopy(propose_result), copy.deepcopy(crop_result)]
    for result in source_results[1:]:
        item = copy.deepcopy(result)
        rewrite_result_bbox(item, original_bbox, corrected_bbox)
        transformed.append(item)
    return transformed


def transform_claims(claims: list[Any], original_bbox: Any, corrected_bbox: list[int]) -> list[Any]:
    transformed = copy.deepcopy(claims)
    for claim in transformed:
        if isinstance(claim, dict) and claim.get("visual_bbox") == original_bbox:
            claim["visual_bbox"] = corrected_bbox
    return transformed


def rewrite_result_bbox(result: dict[str, Any], original_bbox: Any, corrected_bbox: list[int]) -> None:
    if result.get("bbox") == original_bbox:
        result["bbox"] = corrected_bbox
    if result.get("visual_bbox") == original_bbox:
        result["visual_bbox"] = corrected_bbox
    for key in ["anchor", "claim"]:
        item = result.get(key)
        if isinstance(item, dict):
            rewrite_result_bbox(item, original_bbox, corrected_bbox)


def summarize_quality(
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    task_split_counts = Counter(str(task.get("split")) for task in tasks)
    sft_split_counts = Counter(str(row.get("split")) for row in sft_rows)
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in sft_rows)
    target_ious = [float(row["target_iou"]) for row in quality_rows if row.get("target_iou") is not None]
    candidate_counts = [int(row["candidate_count"]) for row in quality_rows]
    return {
        "tasks_total": len(tasks),
        "task_split_counts": dict(task_split_counts),
        "episodes_total": len(episodes),
        "sft_rows_total": len(sft_rows),
        "sft_split_counts": dict(sft_split_counts),
        "sft_action_counts": dict(action_counts),
        "no_highlight_page_rate": sum(not row["page_image_has_highlight"] for row in quality_rows) / max(1, len(quality_rows)),
        "target_region_recall_iou_0_5": sum(iou >= 0.5 for iou in target_ious) / max(1, len(target_ious)),
        "target_region_recall_iou_0_9": sum(iou >= 0.9 for iou in target_ious) / max(1, len(target_ious)),
        "target_region_iou_mean": sum(target_ious) / max(1, len(target_ious)),
        "target_region_iou_min": min(target_ious) if target_ious else None,
        "candidate_count_mean": sum(candidate_counts) / max(1, len(candidate_counts)),
        "candidate_count_min": min(candidate_counts) if candidate_counts else None,
        "candidate_count_max": max(candidate_counts) if candidate_counts else None,
        "pdf_image_candidate_rate": sum(row["pdf_image_candidates"] > 0 for row in quality_rows) / max(1, len(quality_rows)),
    }


def scale_bbox(value: Any, scale_x: float, scale_y: float, width: int, height: int) -> list[int] | None:
    if not value or len(value) != 4:
        return None
    x1, y1, x2, y2 = value
    box = [
        max(0, min(width, int(round(float(x1) * scale_x)))),
        max(0, min(height, int(round(float(y1) * scale_y)))),
        max(0, min(width, int(round(float(x2) * scale_x)))),
        max(0, min(height, int(round(float(y2) * scale_y)))),
    ]
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def block_text(block: dict[str, Any]) -> str:
    chunks: list[str] = []
    for line in block.get("lines") or []:
        for span in line.get("spans") or []:
            text = str(span.get("text") or "").strip()
            if text:
                chunks.append(text)
    return " ".join(chunks)


def dedupe_regions(regions: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(regions, key=lambda region: area(region["bbox"]), reverse=True):
        if all(bbox_iou(item["bbox"], other["bbox"]) < iou_threshold for other in kept):
            kept.append(item)
    return kept


def grid_distractors(width: int, height: int, gold_bbox: list[int]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    for index, box in enumerate(
        [
            [0, 0, int(width * 0.3), int(height * 0.18)],
            [int(width * 0.7), 0, width, int(height * 0.18)],
            [0, int(height * 0.82), int(width * 0.35), height],
            [int(width * 0.65), int(height * 0.82), width, height],
        ]
    ):
        if bbox_iou(box, gold_bbox) < 0.05:
            regions.append(
                {
                    "bbox": box,
                    "source": "layout_distractor",
                    "type": "non_target_page_region",
                    "score": 0.05,
                    "hint": f"页眉页脚/边角布局干扰区域 {index}",
                }
            )
    return regions


def area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def distance_to_bbox(a: list[int], b: list[int]) -> float:
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return (ax - bx) ** 2 + (ay - by) ** 2


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    quality = manifest["quality"]
    lines = [
        "# EvidenceGrounded AgentBench v0.4 构建报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 定义",
        "",
        "v0.4 是无红框 region-selection agent 数据。模型输入为原始 PDF 页面，不再包含 highlighted page。",
        "",
        "核心轨迹：",
        "",
        "```text",
        "propose_regions -> crop_region(region_id) -> retrieve_evidence -> open_evidence -> write_claim / abstain_claim -> finish",
        "```",
        "",
        "## 规模",
        "",
        f"- tasks_total：{quality['tasks_total']}",
        f"- task_split_counts：`{json.dumps(quality['task_split_counts'], ensure_ascii=False)}`",
        f"- sft_rows_total：{quality['sft_rows_total']}",
        f"- sft_split_counts：`{json.dumps(quality['sft_split_counts'], ensure_ascii=False)}`",
        f"- sft_action_counts：`{json.dumps(quality['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 质量",
        "",
        f"- no_highlight_page_rate：{quality['no_highlight_page_rate']:.3f}",
        f"- target_region_recall_iou_0_5：{quality['target_region_recall_iou_0_5']:.3f}",
        f"- target_region_recall_iou_0_9：{quality['target_region_recall_iou_0_9']:.3f}",
        f"- target_region_iou_mean：{quality['target_region_iou_mean']:.3f}",
        f"- candidate_count_mean：{quality['candidate_count_mean']:.2f}",
        f"- pdf_image_candidate_rate：{quality['pdf_image_candidate_rate']:.3f}",
        "",
        "## 输出文件",
        "",
        f"- tasks_all：`{manifest['files']['tasks_all']}`",
        f"- oracle_episodes：`{manifest['files']['oracle_episodes']}`",
        f"- sft_train：`{manifest['files']['sft_train']}`",
        f"- sft_val：`{manifest['files']['sft_val']}`",
        f"- sft_test：`{manifest['files']['sft_test']}`",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(f"/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_4_region_no_highlight_sft_{stamp}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
