#!/usr/bin/env python3
"""Build v1.0.6 BaseLocate4 expansion and Metadata5 probe.

This builder intentionally uses only local, traceable evidence sources:

1. existing v1.0.5 auto-repaired silver tasks;
2. page-level VLM high-confidence detections;
3. local PDF/evidence-index chunks for Metadata5 support.

LLM/VLM memory is not used as a gold source in this script.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import EvidenceIndex  # noqa: E402


DEFAULT_OLD_SILVER_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_5_auto_repaired_secondpass_sft_20260613_0005"
)
DEFAULT_HYBRID_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_5_hybrid_silver_pagelevel_full_20260613_0725"
)
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_llm_overlay_20260611_0222"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DEFAULT_DOCS_DIR = REPO_ROOT / "docs" / "02_指标与数据"

BASE_FIELDS = ["caption_text", "depicted_work_title", "image_scope", "object_type"]
META_FIELDS = [
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
]
ALL_FIELDS = BASE_FIELDS + META_FIELDS
FIELD_SPEC = "|".join(ALL_FIELDS)
BASE_FIELD_SPEC = "|".join(BASE_FIELDS)
ALLOWED_PL_CAPTION_QUALITY = {"title_like", "descriptive"}
ALLOWED_OBJECT_DOMAINS = {"landscape_painting", "landscape_detail"}

DYNASTY_PATTERN = re.compile(
    r"(北宋|南宋|宋代|宋|元代|元|明代|明|清代|清|唐代|唐|五代|辽|遼|金代|金|民国|民國|"
    r"Northern Song|Southern Song|Song dynasty|Yuan dynasty|Ming dynasty|Qing dynasty|"
    r"Tang dynasty|Five Dynasties|[0-9]{1,2}(?:st|nd|rd|th)?[-–—]?[0-9]{0,2}(?:st|nd|rd|th)? century|"
    r"[0-9]{3,4}s|"
    r"ca\.\s*[0-9]{3,4}(?:[-–—][0-9]{2,4})?)",
    flags=re.I,
)
DIMENSION_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:×|x|X|\*)\s*\d+(?:\.\d+)?(?:\s*(?:×|x|X|\*)\s*\d+(?:\.\d+)?)?\s*(?:厘米|公分|cm|CM))"
)
MEDIUM_PATTERNS = [
    (re.compile(r"纸本|紙本"), "纸本"),
    (re.compile(r"绢本|絹本"), "绢本"),
    (re.compile(r"设色|設色"), "设色"),
    (re.compile(r"水墨"), "水墨"),
    (re.compile(r"ink and color on silk", re.I), "ink and color on silk"),
    (re.compile(r"ink and color on paper", re.I), "ink and color on paper"),
    (re.compile(r"ink on silk", re.I), "ink on silk"),
    (re.compile(r"ink on paper", re.I), "ink on paper"),
    (re.compile(r"color on silk", re.I), "color on silk"),
    (re.compile(r"color on paper", re.I), "color on paper"),
    (re.compile(r"album leaf,?\s*ink[^,;.]{0,40}", re.I), None),
    (re.compile(r"hanging scroll,?\s*ink[^,;.]{0,40}", re.I), None),
    (re.compile(r"handscroll,?\s*ink[^,;.]{0,40}", re.I), None),
]
INSTITUTION_PATTERNS = [
    re.compile(r"((?:北京|台北|臺北|南京|上海|辽宁|遼寧)?故宫博物院)"),
    re.compile(r"([\u4e00-\u9fff]{2,18}(?:博物院|博物馆|美术馆|藝術館|艺术馆))"),
    re.compile(r"(The Metropolitan Museum of Art)", re.I),
    re.compile(r"(Metropolitan Museum of Art)", re.I),
    re.compile(r"(National Palace Museum(?:,?\s*Taipei)?)", re.I),
    re.compile(r"(Palace Museum(?:,?\s*Beijing)?)", re.I),
    re.compile(r"(C\.?\s*C\.?\s*Wang family collection(?:,?\s*New York)?)", re.I),
    re.compile(r"(C C Wang family collection(?:,?\s*New York)?)", re.I),
    re.compile(r"(Museum of Fine Arts,?\s*Boston)", re.I),
    re.compile(r"(Freer Gallery of Art)", re.I),
    re.compile(r"(Cleveland Museum of Art)", re.I),
]
FIGURE_LABEL_PATTERN = re.compile(
    r"((?:图|圖)\s*[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：][一二三四五六七八九十百〇零0-9]+)*[a-zA-Z]?|"
    r"(?:Fig\.?|Figure|Plate)\s*[A-Za-z]?[0-9IVXivx]+(?:[.\-．:：][0-9IVXivx]+)*[a-zA-Z]?)",
    flags=re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1.0.6 BaseLocate4 + Metadata5 probe.")
    parser.add_argument("--old-silver-dir", default=str(DEFAULT_OLD_SILVER_DIR))
    parser.add_argument("--hybrid-dir", default=str(DEFAULT_HYBRID_DIR))
    parser.add_argument("--evidence-index-dir", default=str(DEFAULT_EVIDENCE_INDEX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR))
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--train-caption-cap", type=int, default=2)
    parser.add_argument("--eval-caption-cap", type=int, default=1)
    parser.add_argument("--max-new-pagelevel", type=int, default=0, help="0 means no explicit limit after filters.")
    parser.add_argument("--retrieve-top-k", type=int, default=8)
    parser.add_argument("--probe-train-per-class", type=int, default=40)
    parser.add_argument("--probe-val-per-class", type=int, default=15)
    parser.add_argument("--probe-test-per-class", type=int, default=20)
    parser.add_argument("--sample-per-class", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"agentbench_v1_0_6_baselocate4_metadata5_probe_{stamp}"
    prepare_output_dir(output_dir, args.overwrite)

    old_dir = Path(args.old_silver_dir)
    hybrid_dir = Path(args.hybrid_dir)
    evidence_index = EvidenceIndex(args.evidence_index_dir, anchor_rerank=True)

    old_tasks = read_old_silver_tasks(old_dir)
    pl_rows = select_pagelevel_rows(hybrid_dir / "page_level_auto_usable.jsonl", args, rng)
    pagelevel_tasks = build_pagelevel_tasks(pl_rows, output_dir, len(old_tasks))
    expanded_tasks, skipped_cap = merge_with_caption_caps(old_tasks, pagelevel_tasks, args)
    write_baselocate4_dataset(output_dir, expanded_tasks)

    metadata_rows, classified = build_metadata_pool_and_probe(expanded_tasks, evidence_index, args, rng)
    write_jsonl(output_dir / "metadata_evidence_pool.jsonl", metadata_rows)
    probe_tasks = select_probe_tasks(classified, args, rng)
    write_probe_dataset(output_dir, probe_tasks)
    sample_package = write_abc_sample_package(output_dir, probe_tasks, args, rng)

    summary = build_summary(
        output_dir=output_dir,
        args=args,
        stamp=stamp,
        old_tasks=old_tasks,
        pl_rows=pl_rows,
        pagelevel_tasks=pagelevel_tasks,
        expanded_tasks=expanded_tasks,
        skipped_cap=skipped_cap,
        metadata_rows=metadata_rows,
        classified=classified,
        probe_tasks=probe_tasks,
        sample_package=sample_package,
    )
    write_json(output_dir / "manifest.json", summary)
    report_path = output_dir / "构建报告.md"
    write_report(report_path, summary)
    docs_path = Path(args.docs_dir) / f"{stamp}_v1.0.6BaseLocate4_Metadata5前三阶段构建报告.md"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(report_path.read_text(encoding="utf-8"), encoding="utf-8")
    summary["artifacts"]["docs_report"] = str(docs_path)
    write_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists; use --overwrite")
    for child in ["baselocate4", "baselocate4/sft", "baselocate4/episodes", "metadata_probe", "metadata_probe/sft", "metadata_probe/episodes", "review"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def read_old_silver_tasks(old_dir: Path) -> list[dict[str, Any]]:
    tasks = read_jsonl(old_dir / "tasks_all.jsonl")
    out = []
    for task in tasks:
        copied = copy.deepcopy(task)
        copied["source_stage"] = "old_v1_0_5_auto_repaired_secondpass"
        copied["dataset_version"] = "v1.0.6_baselocate4_expanded"
        copied["tool_schema_version"] = "v1.0.6_no_select_baselocate4"
        copied.setdefault("gold", {})["target_claim_fields"] = list(BASE_FIELDS)
        copied.setdefault("gold", {})["claim_schema_fields"] = list(BASE_FIELDS)
        copied.setdefault("gold", {})["label_source"] = "v1_0_6_baselocate4_old_silver_replay"
        repair_old_silver_base_claims(copied)
        out.append(copied)
    return out


def repair_old_silver_base_claims(task: dict[str, Any]) -> None:
    caption = normalize_space((task.get("gold") or {}).get("caption_text") or "")
    title = extract_title(caption)
    if not title:
        return
    claims = task.setdefault("gold", {}).setdefault("claims", [])
    for idx, claim in enumerate(claims):
        if claim.get("field") != "depicted_work_title":
            continue
        if not claim.get("abstain") and normalize_space(claim.get("value")):
            return
        claims[idx] = supported_claim(
            "depicted_work_title",
            title,
            [local_caption_evidence_id(task)] if local_caption_evidence_id(task) else [],
            dedupe([local_caption_evidence_id(task), local_visual_evidence_id(task)]),
            0.84,
            "v1_0_6_repaired_old_silver_caption_title_extract",
        )
        return
    claims.insert(
        1,
        supported_claim(
            "depicted_work_title",
            title,
            [local_caption_evidence_id(task)] if local_caption_evidence_id(task) else [],
            dedupe([local_caption_evidence_id(task), local_visual_evidence_id(task)]),
            0.84,
            "v1_0_6_repaired_old_silver_caption_title_extract",
        ),
    )


def select_pagelevel_rows(path: Path, args: argparse.Namespace, rng: random.Random) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        if not row.get("auto_usable_probe"):
            continue
        if row.get("object_domain") not in ALLOWED_OBJECT_DOMAINS:
            continue
        if row.get("caption_quality_flag") not in ALLOWED_PL_CAPTION_QUALITY:
            continue
        if row.get("overlaps_silver_secondpass_page"):
            continue
        if not row.get("valid_geometry"):
            continue
        caption = normalize_space(row.get("caption_text"))
        if not caption:
            continue
        if row.get("caption_target_match") not in {"yes", True}:
            continue
        if len(figure_labels(caption)) > 1:
            continue
        rows.append(copy.deepcopy(row))
    rows.sort(key=lambda item: (str(item.get("source_file")), int(item.get("page") or 0), int(item.get("detection_index") or 0), str(item.get("hybrid_item_id"))))
    if args.max_new_pagelevel and len(rows) > args.max_new_pagelevel:
        rows = rng.sample(rows, args.max_new_pagelevel)
        rows.sort(key=lambda item: (str(item.get("source_file")), int(item.get("page") or 0), int(item.get("detection_index") or 0), str(item.get("hybrid_item_id"))))
    return rows


def build_pagelevel_tasks(rows: list[dict[str, Any]], output_dir: Path, start_index: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for offset, row in enumerate(rows):
        task_id = f"v106_pl_{start_index + offset:06d}"
        caption = normalize_space(row.get("caption_text"))
        local_caption_id = f"local_caption_{task_id}"
        local_visual_id = f"local_visual_{task_id}"
        target_bbox = list(row.get("target_bbox_page_px") or row.get("raw_target_bbox_page_px") or [])
        caption_bbox = list(row.get("caption_bbox_page_px") or row.get("raw_caption_bbox_page_px") or [])
        scope = map_image_scope(row.get("image_scope"), caption)
        object_type = map_object_type(row.get("object_domain"))
        title = extract_title(caption)
        region_candidates = [
            {
                "bbox": target_bbox,
                "source": "page_level_vlm",
                "type": "figure_candidate",
                "score": round(float(row.get("confidence") or 0.9) * 10.0, 3),
                "hint": "v1.0.6 page-level VLM high-confidence target",
                "caption_evidence_id": local_caption_id,
                "caption_hint": caption,
                "linked_caption_text": caption,
                "linked_caption_region_id": "r_caption_0",
                "caption_link_score": round(float(row.get("confidence") or 0.9) * 10.0, 3),
                "target_caption_match_score": round(float(row.get("confidence") or 0.9) * 4.0, 3),
                "target_caption_match_reason": "page_level_vlm_caption_target_match",
                "target_region_rank": 1,
                "target_region_sort_score": round(float(row.get("confidence") or 0.9) * 40.0, 3),
                "is_target": True,
                "region_id": "r0",
                "gold_iou": 1.0,
            },
            {
                "bbox": caption_bbox,
                "source": "page_level_vlm_caption",
                "type": "text_or_caption_candidate",
                "score": round(float(row.get("confidence") or 0.9) * 10.0, 3),
                "nearby_text": caption,
                "caption_evidence_id": local_caption_id,
                "caption_hint": caption,
                "hint": "与目标图像匹配的 page-level VLM caption 框",
                "is_target": False,
                "region_id": "r1",
                "gold_iou": 0.0,
            },
        ]
        local_evidence = [
            {
                "evidence_id": local_caption_id,
                "source_file": row.get("source_file"),
                "page_start": row.get("page"),
                "page_end": row.get("page"),
                "authority_level": "B",
                "citation_level": "page_caption_region",
                "source_quality": "page_level_vlm_caption",
                "display_snippet": caption,
                "bbox": caption_bbox,
                "text": caption,
                "claim_allowed_fields": ["caption_text", "depicted_work_title", "creator_or_attribution", "creation_period_or_dynasty", "collection_institution", "dimensions", "medium_material"],
                "usable_for_claim_by_adjudication": True,
            },
            {
                "evidence_id": local_visual_id,
                "source_file": row.get("source_file"),
                "page_start": row.get("page"),
                "page_end": row.get("page"),
                "authority_level": "visual",
                "citation_level": "target_crop_visual",
                "source_quality": "page_level_vlm_target_crop",
                "source_role": "local_visual",
                "evidence_type": "target_crop_visual",
                "display_snippet": "目标裁剪图像的本地视觉证据；可用于支持 object_type，以及在图像范围明确时辅助 image_scope。",
                "bbox": target_bbox,
                "image_path": row.get("crop_image"),
                "adjudicated_claim_allowed_fields": ["object_type", "image_scope"],
                "usable_for_claim_by_adjudication": True,
            },
        ]
        claims = [
            supported_claim("caption_text", caption, [local_caption_id], [local_caption_id, local_visual_id], 0.95, "page_level_caption"),
            supported_claim("image_scope", scope, [local_caption_id, local_visual_id], [local_caption_id, local_visual_id], 0.82, "page_level_scope", None),
            supported_claim("object_type", object_type, [local_visual_id], [local_caption_id, local_visual_id], 0.86, "page_level_object_domain", target_bbox),
        ]
        if title:
            claims.insert(1, supported_claim("depicted_work_title", title, [local_caption_id], [local_caption_id, local_visual_id], 0.86, "caption_title_extract"))
        else:
            claims.insert(1, abstain_claim("depicted_work_title", "图注未明确给出作品题名"))
        task = {
            "task_id": task_id,
            "source_task_id": row.get("hybrid_item_id"),
            "split": row.get("split") or "train",
            "dataset_version": "v1.0.6_baselocate4_expanded",
            "tool_schema_version": "v1.0.6_no_select_baselocate4",
            "task_type": "evidence_grounded_pdf_figure_claim",
            "runtime_mode": "v1_0_6_baselocate4_pagelevel_highconf",
            "source_type": "pdf_page",
            "source_file": row.get("source_file"),
            "source_stem": Path(str(row.get("source_file") or "")).stem,
            "source_path": None,
            "page": row.get("page"),
            "page_image": row.get("page_image"),
            "artwork_image": row.get("crop_image"),
            "overlay_image": row.get("page_level_overlay"),
            "goal": "Inspect the PDF page, crop the target landscape-related figure, open/retrieve evidence, and write BaseLocate4 claims.",
            "available_tools": ["inspect_page", "crop_target", "retrieve_evidence", "open_evidence", "write_claims_chunk", "finish"],
            "region_candidates": region_candidates,
            "local_evidence": local_evidence,
            "gold": {
                "image_bbox": target_bbox,
                "target_region_id": "r0",
                "target_region_bbox": target_bbox,
                "target_region_iou": 1.0,
                "caption_bbox": caption_bbox,
                "caption_text": caption,
                "claims": claims,
                "claim_schema_fields": list(BASE_FIELDS),
                "target_claim_fields": list(BASE_FIELDS),
                "evidence_ids": [local_caption_id, local_visual_id],
                "candidate_evidence_ids": [local_caption_id, local_visual_id],
                "evidence_query": build_evidence_query(row.get("source_file"), caption, title),
                "auto_label": True,
                "needs_review": False,
                "label_source": "v1_0_6_pagelevel_highconf_baselocate4",
            },
            "candidate_meta": {
                "source_kind": "page_level_vlm_detection",
                "source_dataset_dir": row.get("source_dataset_dir"),
                "source_dataset_name": row.get("source_dataset_name"),
                "hybrid_item_id": row.get("hybrid_item_id"),
                "caption_quality_flag": row.get("caption_quality_flag"),
                "object_domain": row.get("object_domain"),
                "confidence": row.get("confidence"),
                "dedup_caption_key": caption_key(caption),
                "v1_0_6_high_conf_filter": True,
            },
            "page_level_vlm": {
                "reason": row.get("reason"),
                "bbox_coord_system": row.get("bbox_coord_system"),
                "raw_target_bbox_page_px": row.get("raw_target_bbox_page_px"),
                "raw_caption_bbox_page_px": row.get("raw_caption_bbox_page_px"),
            },
            "source_stage": "page_level_highconf_v1_0_6",
        }
        tasks.append(task)
    return tasks


def merge_with_caption_caps(
    old_tasks: list[dict[str, Any]],
    pagelevel_tasks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    used_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    caption_home: dict[str, str] = {}
    merged: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for task in old_tasks:
        split = str(task.get("split") or "train")
        key = caption_key((task.get("gold") or {}).get("caption_text") or "")
        if key:
            used_by_split[split][key] += 1
            caption_home.setdefault(key, split)
        merged.append(task)
    for task in pagelevel_tasks:
        split = str(task.get("split") or "train")
        key = caption_key((task.get("gold") or {}).get("caption_text") or "")
        cap = args.train_caption_cap if split == "train" else args.eval_caption_cap
        if key and key in caption_home and caption_home[key] != split:
            skipped.append(skip_row(task, "caption_cross_split_leakage"))
            continue
        if key and used_by_split[split][key] >= cap:
            skipped.append(skip_row(task, "caption_cap_reached"))
            continue
        if key:
            used_by_split[split][key] += 1
            caption_home.setdefault(key, split)
        merged.append(task)
    merged.sort(key=lambda item: (["train", "val", "test"].index(str(item.get("split") or "train")) if str(item.get("split") or "train") in {"train", "val", "test"} else 9, str(item.get("source_file")), int(item.get("page") or 0), str(item.get("task_id"))))
    return merged, skipped


def skip_row(task: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "caption_text": (task.get("gold") or {}).get("caption_text"),
        "skip_reason": reason,
    }


def write_baselocate4_dataset(output_dir: Path, tasks: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_sft: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for task in tasks:
        split = str(task.get("split") or "train")
        by_split[split].append(task)
        actions = build_oracle_actions(task, BASE_FIELDS, include_retrieve=True)
        episode = {"task_id": task.get("task_id"), "split": split, "actions": actions}
        sft_rows = build_sft_rows(task, actions, BASE_FIELDS, "v1_0_6_baselocate4_expanded_sft")
        all_episodes.append(episode)
        all_sft.extend(sft_rows)
    write_jsonl(output_dir / "baselocate4" / "tasks_all.jsonl", tasks)
    write_jsonl(output_dir / "baselocate4" / "episodes" / "all.jsonl", all_episodes)
    write_jsonl(output_dir / "baselocate4" / "sft" / "all.jsonl", all_sft)
    for split in ["train", "val", "test"]:
        split_tasks = by_split.get(split, [])
        write_jsonl(output_dir / "baselocate4" / f"{split}_tasks.jsonl", split_tasks)
        write_jsonl(output_dir / "baselocate4" / "episodes" / f"{split}.jsonl", [ep for ep in all_episodes if ep.get("split") == split])
        write_jsonl(output_dir / "baselocate4" / "sft" / f"{split}.jsonl", [row for row in all_sft if row.get("split") == split])


def build_metadata_pool_and_probe(
    tasks: list[dict[str, Any]],
    index: EvidenceIndex,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_rows: list[dict[str, Any]] = []
    classified: list[dict[str, Any]] = []
    for task in tasks:
        caption = (task.get("gold") or {}).get("caption_text") or ""
        title = title_from_task(task) or extract_title(caption)
        local_meta = extract_metadata(caption, title)
        base_claims = base_claims_from_task(task)
        local_meta_count = len([field for field in META_FIELDS if field in local_meta])
        missing_meta_fields = [field for field in META_FIELDS if field not in local_meta]
        external_supports: dict[str, dict[str, Any]] = {}
        retrieve_result = {"tool": "retrieve_evidence", "query": metadata_query(task, title, caption, missing_meta_fields), "scope": "same_document", "results": [], "hit_evidence_ids": []}
        if missing_meta_fields and local_meta_count < 2:
            retrieval_task = {
                "task_id": task.get("task_id"),
                "source_file": task.get("source_file"),
                "page": task.get("page"),
                "local_evidence": task.get("local_evidence") or [],
                "region_candidates": task.get("region_candidates") or [],
            }
            results = index.search(retrieve_result["query"], "same_document", retrieval_task, args.retrieve_top_k)
            retrieve_result["results"] = results
            for result in results:
                full = index.open(str(result.get("evidence_id") or "")) or result
                if not evidence_can_support_metadata(full):
                    continue
                text = evidence_text(full)
                if not evidence_anchored_to_task(text, title, caption):
                    continue
                focused_text = focus_text_for_target(text, title, caption)
                external_meta = extract_metadata(focused_text, title)
                external_meta = filter_external_metadata(external_meta, focused_text, title, caption)
                if not external_meta:
                    continue
                support_fields = [field for field in missing_meta_fields if field in external_meta]
                if not support_fields:
                    continue
                pool_row = {
                    "evidence_id": full.get("evidence_id"),
                    "source_type": "same_document_chunk",
                    "source_file": full.get("source_file"),
                    "page_start": full.get("page_start") if full.get("page_start") is not None else full.get("page"),
                    "page_end": full.get("page_end"),
                    "authority_level": full.get("authority_level"),
                    "citation_level": full.get("citation_level"),
                    "source_quality": full.get("source_quality"),
                    "clean_evidence_type": full.get("clean_evidence_type"),
                    "text": text,
                    "focused_text": focused_text,
                    "display_snippet": full.get("display_snippet") or text[:500],
                    "linked_task_id": task.get("task_id"),
                    "linked_caption_text": caption,
                    "linked_title": title,
                    "candidate_fields": {field: external_meta[field] for field in support_fields},
                    "support_labels": {field: "support" for field in support_fields},
                    "support_reason": "same_document evidence contains target title/figure anchor and field value",
                }
                pool_rows.append(pool_row)
                for field in support_fields:
                    external_supports.setdefault(
                        field,
                        {
                            "field": field,
                            "value": external_meta[field],
                            "evidence_id": str(full.get("evidence_id")),
                            "evidence": pool_row,
                        },
                    )
        metadata_claims = build_metadata_claims(task, local_meta, external_supports)
        task_class = classify_abc(local_meta, external_supports)
        probe_task = copy.deepcopy(task)
        probe_task["dataset_version"] = "v1.0.6_baselocate4_metadata5_probe"
        probe_task["tool_schema_version"] = "v1.0.6_no_select_baselocate4_metadata5"
        probe_task["runtime_mode"] = "v1_0_6_baselocate4_metadata5_probe"
        probe_task.setdefault("gold", {})["claims"] = base_claims + metadata_claims
        probe_task.setdefault("gold", {})["claim_schema_fields"] = list(ALL_FIELDS)
        probe_task.setdefault("gold", {})["target_claim_fields"] = list(ALL_FIELDS)
        probe_task.setdefault("gold", {})["label_source"] = "v1_0_6_baselocate4_metadata5_rule_evidence_probe"
        probe_task.setdefault("gold", {})["evidence_ids"] = dedupe(
            eid for claim in probe_task["gold"]["claims"] for eid in (claim.get("evidence_ids") or [])
        )
        probe_task["metadata_probe"] = {
            "abc_class": task_class,
            "caption_metadata_fields": sorted(local_meta),
            "external_metadata_fields": sorted(external_supports),
            "missing_metadata_fields_after_caption": missing_meta_fields,
            "retrieve_query": retrieve_result["query"],
            "retrieve_result": retrieve_result,
            "external_evidence": [support["evidence"] for support in external_supports.values()],
            "local_metadata": local_meta,
        }
        classified.append(probe_task)
    return pool_rows, classified


def evidence_can_support_metadata(item: dict[str, Any]) -> bool:
    if item.get("usable_for_retrieval") is False:
        return False
    try:
        if float(item.get("noise_score") or 0.0) >= 0.93:
            return False
    except Exception:
        pass
    role = str(item.get("adjudicated_evidence_role") or "")
    if role in {"ocr_noise", "front_matter", "low_value_background"}:
        return False
    return True


def evidence_text(item: dict[str, Any]) -> str:
    return normalize_space(item.get("display_snippet") or item.get("evidence_summary") or item.get("clean_text") or item.get("text") or "")


def evidence_anchored_to_task(text: str, title: str, caption: str) -> bool:
    text_norm = compact(text)
    title_norm = compact(title)
    if title_norm and len(title_norm) >= 3 and title_norm in text_norm:
        return True
    labels = figure_labels(caption)
    for label in labels:
        if compact(label) and compact(label) in text_norm:
            return True
    return False


def filter_external_metadata(meta: dict[str, str], focused_text: str, title: str, caption: str) -> dict[str, str]:
    filtered: dict[str, str] = {}
    for field, value in meta.items():
        if field == "creator_or_attribution" and not valid_creator_value(value):
            continue
        if field == "creation_period_or_dynasty" and looks_like_bibliography_period(value, focused_text):
            continue
        if field == "collection_institution" and looks_like_bibliography_collection(value, focused_text):
            continue
        if field == "medium_material" and looks_like_title_material_word(value, focused_text):
            continue
        if not value_near_target_anchor(value, focused_text, title, caption):
            continue
        filtered[field] = value
    return filtered


def valid_creator_value(value: str) -> bool:
    value = normalize_space(value)
    if not value:
        return False
    bad_exact = {
        "图版",
        "局部",
        "山水",
        "部分",
        "构图",
        "构图与",
        "画面",
        "本段",
        "本段为",
        "创作",
        "创作了",
        "根据",
        "石涛根据",
        "该图是",
        "聚焦",
        "传世名作",
        "传世名作有",
        "其作品",
    }
    if value in bad_exact:
        return False
    if any(token in value for token in ["本段", "构图", "画面", "创作了", "根据", "来自", "该图", "聚焦", "传世", "名作", "作品"]):
        return False
    if "的" in value:
        return False
    if value.endswith(("与", "和", "及", "为", "了", "根据")):
        return False
    if re.fullmatch(r"[\u4e00-\u9fff]{2,4}", value):
        return True
    if re.fullmatch(r"(?:传|傳)?[\u4e00-\u9fff]{2,4}", value):
        return True
    if re.fullmatch(r"(?:Attributed to|After)\s+[A-Z][A-Za-z'\-\. ]{2,80}", value):
        return True
    if re.fullmatch(r"[A-Z][A-Za-z'\-\. ]{2,80}", value):
        return True
    return False


def looks_like_bibliography_period(value: str, focused_text: str) -> bool:
    value = normalize_space(value)
    text = normalize_space(focused_text)
    if re.fullmatch(r"[唐宋元明清]|\w+ dynasty", value or "", flags=re.I):
        before = text[: max(0, text.find(value))]
        if re.search(r"\[[唐宋元明清]\]|《历代名画记》|出版社|参考文献|Bibliography", before[-80:]):
            return True
    return False


def looks_like_bibliography_collection(value: str, focused_text: str) -> bool:
    pos = focused_text.find(value)
    if pos < 0:
        return False
    after = focused_text[pos + len(value) : pos + len(value) + 24]
    return bool(re.search(r"(图录|圖錄|特展|展览|展覽|院刊|学报|學報|出版社)", after))


def looks_like_title_material_word(value: str, focused_text: str) -> bool:
    if value not in {"设色", "設色", "水墨"}:
        return False
    for match in re.finditer(re.escape(value), focused_text):
        before = focused_text[max(0, match.start() - 8) : match.start()]
        after = focused_text[match.end() : match.end() + 8]
        if "《" in before and "》" in after:
            return True
    return False


def value_near_target_anchor(value: str, focused_text: str, title: str, caption: str) -> bool:
    text = normalize_space(focused_text)
    value = normalize_space(value)
    if not value:
        return False
    value_positions = [m.start() for m in re.finditer(re.escape(value), text)]
    if not value_positions:
        return False
    anchors: list[int] = []
    if title:
        anchors.extend(m.start() for m in re.finditer(re.escape(normalize_space(title)), text))
    for label in figure_labels(caption):
        label_norm = normalize_space(label)
        anchors.extend(m.start() for m in re.finditer(re.escape(label_norm), text))
        compact_label = compact(label_norm)
        compact_text = compact(text)
        compact_pos = compact_text.find(compact_label)
        if compact_pos >= 0:
            anchors.append(approximate_raw_position(text, compact_text, compact_pos))
    if not anchors:
        return False
    return min(abs(v - a) for v in value_positions for a in anchors) <= 180


def build_metadata_claims(
    task: dict[str, Any],
    local_meta: dict[str, str],
    external_supports: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    local_caption_id = local_caption_evidence_id(task)
    claims: list[dict[str, Any]] = []
    for field in META_FIELDS:
        if field in local_meta and local_caption_id:
            claims.append(
                supported_claim(
                    field,
                    local_meta[field],
                    [local_caption_id],
                    [local_caption_id],
                    0.86,
                    "metadata5_local_caption_extract",
                )
            )
        elif field in external_supports:
            support = external_supports[field]
            claims.append(
                supported_claim(
                    field,
                    support["value"],
                    [support["evidence_id"]],
                    [support["evidence_id"], local_caption_id],
                    0.82,
                    "metadata5_same_document_evidence_extract",
                )
            )
        else:
            claims.append(abstain_claim(field, "local caption 与同文档检索 evidence 均未明确支持该 metadata 字段"))
    return claims


def classify_abc(local_meta: dict[str, str], external_supports: dict[str, dict[str, Any]]) -> str:
    local_count = len([field for field in META_FIELDS if field in local_meta])
    external_count = len([field for field in META_FIELDS if field in external_supports])
    if local_count >= 2:
        return "A_caption_has_metadata"
    if external_count >= 1:
        return "B_external_evidence_has_metadata"
    return "C_no_metadata_support"


def select_probe_tasks(
    classified: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in classified:
        by_class[str((task.get("metadata_probe") or {}).get("abc_class"))].append(task)
    for rows in by_class.values():
        rows.sort(key=lambda item: (str(item.get("source_file")), int(item.get("page") or 0), str(item.get("task_id"))))
    selected: list[dict[str, Any]] = []
    quotas = [
        ("train", args.probe_train_per_class),
        ("val", args.probe_val_per_class),
        ("test", args.probe_test_per_class),
    ]
    for cls in ["A_caption_has_metadata", "B_external_evidence_has_metadata", "C_no_metadata_support"]:
        rows = list(by_class.get(cls) or [])
        rng.shuffle(rows)
        cursor = 0
        for split, quota in quotas:
            take = rows[cursor : cursor + quota]
            cursor += len(take)
            for task in take:
                copied = copy.deepcopy(task)
                copied["split"] = split
                copied["metadata_probe"]["probe_split"] = split
                selected.append(copied)
    selected.sort(key=lambda item: (["train", "val", "test"].index(str(item.get("split"))) if str(item.get("split")) in {"train", "val", "test"} else 9, str((item.get("metadata_probe") or {}).get("abc_class")), str(item.get("source_file")), int(item.get("page") or 0), str(item.get("task_id"))))
    return selected


def write_probe_dataset(output_dir: Path, tasks: list[dict[str, Any]]) -> None:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_sft: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    for task in tasks:
        split = str(task.get("split") or "train")
        by_split[split].append(task)
        actions = build_metadata_oracle_actions(task)
        episode = {"task_id": task.get("task_id"), "split": split, "abc_class": (task.get("metadata_probe") or {}).get("abc_class"), "actions": actions}
        sft_rows = build_sft_rows(task, actions, ALL_FIELDS, "v1_0_6_baselocate4_metadata5_probe_sft")
        all_episodes.append(episode)
        all_sft.extend(sft_rows)
    write_jsonl(output_dir / "metadata_probe" / "tasks_all.jsonl", tasks)
    write_jsonl(output_dir / "metadata_probe" / "episodes" / "all.jsonl", all_episodes)
    write_jsonl(output_dir / "metadata_probe" / "sft" / "all.jsonl", all_sft)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / "metadata_probe" / f"{split}_tasks.jsonl", by_split.get(split, []))
        write_jsonl(output_dir / "metadata_probe" / "episodes" / f"{split}.jsonl", [ep for ep in all_episodes if ep.get("split") == split])
        write_jsonl(output_dir / "metadata_probe" / "sft" / f"{split}.jsonl", [row for row in all_sft if row.get("split") == split])


def build_metadata_oracle_actions(task: dict[str, Any]) -> list[dict[str, Any]]:
    metadata_probe = task.get("metadata_probe") or {}
    abc = str(metadata_probe.get("abc_class") or "")
    local_caption_id = local_caption_evidence_id(task)
    local_visual_id = local_visual_evidence_id(task)
    actions: list[dict[str, Any]] = [
        {"action": "inspect_page", "top_k": min(10, max(6, len(task.get("region_candidates") or [])))},
        {"action": "crop_target", "region_id": target_region(task).get("region_id") or "r0"},
    ]
    if local_caption_id:
        actions.append({"action": "open_evidence", "evidence_id": local_caption_id})
    if local_visual_id:
        actions.append({"action": "open_evidence", "evidence_id": local_visual_id})
    if abc in {"B_external_evidence_has_metadata", "C_no_metadata_support"}:
        actions.append(
            {
                "action": "retrieve_evidence",
                "query": metadata_probe.get("retrieve_query") or "",
                "scope": "same_document",
                "top_k": 5,
            }
        )
    if abc == "B_external_evidence_has_metadata":
        opened = []
        for evidence in metadata_probe.get("external_evidence") or []:
            eid = str(evidence.get("evidence_id") or "")
            if eid and eid not in opened:
                actions.append({"action": "open_evidence", "evidence_id": eid})
                opened.append(eid)
            if len(opened) >= 2:
                break
    claim_by_field = {str(claim.get("field")): claim for claim in (task.get("gold") or {}).get("claims") or []}
    for field in ALL_FIELDS:
        claim = claim_by_field.get(field)
        actions.append(claim_action(field, claim))
    actions.append({"action": "finish", "status": "done"})
    return actions


def build_oracle_actions(task: dict[str, Any], target_fields: list[str], include_retrieve: bool) -> list[dict[str, Any]]:
    local_caption_id = local_caption_evidence_id(task)
    local_visual_id = local_visual_evidence_id(task)
    actions: list[dict[str, Any]] = [
        {"action": "inspect_page", "top_k": min(10, max(6, len(task.get("region_candidates") or [])))},
        {"action": "crop_target", "region_id": target_region(task).get("region_id") or "r0"},
    ]
    if local_caption_id:
        actions.append({"action": "open_evidence", "evidence_id": local_caption_id})
    if local_visual_id:
        actions.append({"action": "open_evidence", "evidence_id": local_visual_id})
    if include_retrieve:
        actions.append(
            {
                "action": "retrieve_evidence",
                "query": (task.get("gold") or {}).get("evidence_query") or build_evidence_query(task.get("source_file"), (task.get("gold") or {}).get("caption_text"), title_from_task(task)),
                "scope": "same_document",
                "top_k": 5,
            }
        )
    claim_by_field = {str(claim.get("field")): claim for claim in (task.get("gold") or {}).get("claims") or []}
    for field in target_fields:
        actions.append(claim_action(field, claim_by_field.get(field)))
    actions.append({"action": "finish", "status": "done"})
    return actions


def claim_action(field: str, claim: dict[str, Any] | None) -> dict[str, Any]:
    if not claim:
        return {"action": "write_claims_chunk", "claims": [], "abstains": [{"field": field, "reason": "字段 gold 缺失"}]}
    if claim.get("abstain"):
        return {
            "action": "write_claims_chunk",
            "claims": [],
            "abstains": [{"field": field, "reason": claim.get("reason") or "证据不足"}],
        }
    return {
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


def build_sft_rows(task: dict[str, Any], actions: list[dict[str, Any]], target_fields: list[str], label_source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    draft_claims: list[dict[str, Any]] = []
    images = [task.get("page_image")]
    for step, action in enumerate(actions):
        row = {
            "task_id": task.get("task_id"),
            "source_task_id": task.get("source_task_id"),
            "split": task.get("split"),
            "step": step,
            "tool_schema_version": task.get("tool_schema_version"),
            "action": copy.deepcopy(action),
            "history": copy.deepcopy(history),
            "tool_results": copy.deepcopy(tool_results),
            "draft_claims": copy.deepcopy(draft_claims),
            "selected_evidence_ids": [],
            "images": [item for item in images if item],
            "label_source": label_source,
            "claim_state": claim_state(draft_claims, target_fields),
            "available_actions": [action.get("action")],
            "phase_name": "v106_" + str(action.get("action")),
            "phase_hint": phase_hint(action, target_fields),
        }
        row["prompt_text"] = build_prompt_text(task, row, target_fields)
        row["messages"] = build_messages(row)
        rows.append(row)
        result = result_for_action(task, action, draft_claims, target_fields)
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
    draft_claims: list[dict[str, Any]],
    target_fields: list[str],
) -> dict[str, Any]:
    name = action.get("action")
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
        return {"tool": "crop_target", "region_id": action.get("region_id"), "crop_mode": "region_id", "bbox": bbox, "crop_path": task.get("artwork_image"), "bbox_iou": 1.0}
    if name == "open_evidence":
        eid = str(action.get("evidence_id") or "")
        local = open_local_evidence(task, eid)
        if local:
            return local
        for external in (task.get("metadata_probe") or {}).get("external_evidence") or []:
            if str(external.get("evidence_id") or "") == eid:
                return {
                    "tool": "open_evidence",
                    "evidence_id": eid,
                    "source_file": external.get("source_file"),
                    "page_start": external.get("page_start"),
                    "page_end": external.get("page_end"),
                    "authority_level": external.get("authority_level"),
                    "citation_level": external.get("citation_level"),
                    "source_quality": external.get("source_quality"),
                    "display_snippet": external.get("display_snippet") or external.get("text"),
                }
        return {"tool": "open_evidence", "evidence_id": eid, "error": "evidence not found in v1.0.6 probe cache"}
    if name == "retrieve_evidence":
        cached = (task.get("metadata_probe") or {}).get("retrieve_result")
        if cached:
            result = copy.deepcopy(cached)
            result["query"] = action.get("query")
            result["scope"] = action.get("scope")
            return result
        return {"tool": "retrieve_evidence", "query": action.get("query"), "scope": action.get("scope"), "results": [], "hit_evidence_ids": []}
    if name == "write_claims_chunk":
        next_claims = apply_claim_write(draft_claims, action.get("claims") or [], action.get("abstains") or [])
        return {"tool": "write_claims_chunk", "claims": action.get("claims") or [], "abstains": action.get("abstains") or [], "claim_state": claim_state(next_claims, target_fields)}
    if name == "finish":
        return {"tool": "finish", "status": action.get("status", "done"), "draft_claims": draft_claims}
    return {"tool": name}


def build_prompt_text(task: dict[str, Any], row: dict[str, Any], target_fields: list[str]) -> str:
    fields_line = "、".join(target_fields)
    if target_fields == BASE_FIELDS:
        target_name = "BaseLocate4"
        definition = "caption_text=目标图注原文；depicted_work_title=作品题名；image_scope=full_work|figure_or_plate|partial_detail|album_leaf_or_scroll_section|unknown；object_type=landscape_painting|landscape_detail|painting|diagram_or_chart|text_page_or_caption|unknown。"
    else:
        target_name = "BaseLocate4+Metadata5"
        definition = (
            "BaseLocate4：caption_text=目标图注原文；depicted_work_title=作品题名；image_scope=目标图范围；object_type=目标对象类型。"
            "Metadata5：creator_or_attribution=作者/归属；creation_period_or_dynasty=年代/朝代；collection_institution=藏馆；dimensions=尺寸；medium_material=材质。"
        )
    return "\n".join(
        [
            "你是 evidence-grounded figure understanding 的 VLM tool-call agent。",
            f"目标：先检查 PDF 页面布局，再裁剪目标图像，之后打开/检索可追溯证据，并写出 {target_name} 证据支撑 claim；本协议不使用 select_evidence。",
            f"task_id：{row.get('task_id')}；step：{row.get('step')}",
            f"source_file：{task.get('source_file', '')}；page：{task.get('page', '')}",
            f"输入图像：{len(row.get('images') or [])} 张。第 1 张通常是 PDF 页面；第 2 张通常是已裁剪目标图。",
            f"target_fields：{fields_line}。",
            f"字段定义：{definition}",
            "约束：只输出一个 JSON 对象；不要输出 markdown；不要编造事实；证据不足必须 abstain；每次 write_claims_chunk 处理 1 个字段；remaining_fields 非空时禁止 finish。",
            "Metadata5 证据规则：local caption 只支持图注中明确出现的 metadata；caption 没有的作者/年代/藏馆/尺寸/材质必须 retrieve/open 正文或本地 KB evidence；找不到就 abstain。",
            f'工具格式：{{"action":"inspect_page","top_k":整数}}；{{"action":"crop_target","region_id":"r0"}}；{{"action":"open_evidence","evidence_id":"local_caption_xxx或ev_xxx"}}；{{"action":"retrieve_evidence","query":"...","scope":"same_document","top_k":5}}；{{"action":"write_claims_chunk","claims":[{{"field":"{FIELD_SPEC}","value":值,"evidence_ids":["完整 evidence_id"],"visual_bbox":null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}；{{"action":"finish","status":"done"}}',
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
            "claim 的 evidence_ids 必须从可见、已打开或已检索结果中逐字符复制。",
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
        item = {k: result.get(k) for k in ["tool", "evidence_id", "source_file", "page", "page_start", "page_end", "status", "error"] if k in result}
        if result.get("tool") == "inspect_page":
            item["regions"] = (result.get("regions") or [])[:10]
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


def write_abc_sample_package(
    output_dir: Path,
    probe_tasks: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    package_dir = output_dir / "review" / f"abc_sample_package_{stamp}"
    assets_dir = package_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in probe_tasks:
        by_class[str((task.get("metadata_probe") or {}).get("abc_class"))].append(task)
    sampled: list[dict[str, Any]] = []
    for cls in ["A_caption_has_metadata", "B_external_evidence_has_metadata", "C_no_metadata_support"]:
        rows = list(by_class.get(cls) or [])
        rows.sort(key=lambda item: (str(item.get("source_file")), int(item.get("page") or 0), str(item.get("task_id"))))
        if len(rows) > args.sample_per_class:
            rows = rng.sample(rows, args.sample_per_class)
            rows.sort(key=lambda item: (str(item.get("source_file")), int(item.get("page") or 0), str(item.get("task_id"))))
        sampled.extend(rows)
    write_jsonl(package_dir / "abc_review_samples.jsonl", [sample_record(task) for task in sampled])
    tsv = ["sample_id\tabc_class\tdecision\ttarget_box_ok\tcaption_ok\tmetadata_ok\tnotes"]
    md = [
        "# v1.0.6 BaseLocate4 + Metadata5 A/B/C 抽样包",
        "",
        f"- 数据目录：`{output_dir}`",
        f"- 样本数：{len(sampled)}",
        "- 红框/overlay 来自上游构建；crop 是目标图裁剪。",
        "- A：caption 已包含 metadata；B：caption 不含或不足，但正文/本地 evidence 有；C：caption 和本地检索都未发现 metadata 支持。",
        "",
    ]
    class_titles = {
        "A_caption_has_metadata": "A 类：caption 已包含 metadata",
        "B_external_evidence_has_metadata": "B 类：caption 不含 metadata，但正文/KB 有",
        "C_no_metadata_support": "C 类：caption 和正文/KB 都不支持",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in sampled:
        grouped[str((task.get("metadata_probe") or {}).get("abc_class"))].append(task)
    sample_idx = 0
    for cls in ["A_caption_has_metadata", "B_external_evidence_has_metadata", "C_no_metadata_support"]:
        md.extend([f"## {class_titles[cls]}", ""])
        for task in grouped.get(cls, []):
            sample_idx += 1
            sample_id = f"ABC{sample_idx:03d}"
            overlay_rel = copy_asset(task.get("overlay_image"), assets_dir, f"{sample_id}_{task.get('task_id')}_overlay.jpg")
            crop_rel = copy_asset(task.get("artwork_image"), assets_dir, f"{sample_id}_{task.get('task_id')}_crop.jpg")
            tsv.append("\t".join([sample_id, cls, "", "", "", "", ""]))
            probe = task.get("metadata_probe") or {}
            claims = {claim.get("field"): claim for claim in (task.get("gold") or {}).get("claims") or []}
            meta_table = metadata_table(claims)
            md.extend(
                [
                    f"### {sample_id} {task.get('task_id')}",
                    "",
                    f"- split/source/page：`{task.get('split')}` / `{task.get('source_file')}` / `{task.get('page')}`",
                    f"- caption：{(task.get('gold') or {}).get('caption_text') or ''}",
                    f"- title：`{claim_value(claims.get('depicted_work_title'))}`",
                    f"- 分类理由：local metadata fields=`{probe.get('caption_metadata_fields')}`；external metadata fields=`{probe.get('external_metadata_fields')}`",
                    "",
                    f"![overlay]({overlay_rel})",
                    "",
                    f"![crop]({crop_rel})",
                    "",
                    "Metadata5 标注：",
                    "",
                    meta_table,
                    "",
                ]
            )
            if probe.get("retrieve_query"):
                md.extend([f"- retrieve_query：`{probe.get('retrieve_query')}`", ""])
            if probe.get("external_evidence"):
                md.extend(["外部/正文 evidence：", ""])
                for evidence in probe.get("external_evidence") or []:
                    md.extend(
                        [
                            f"- `{evidence.get('evidence_id')}` page={evidence.get('page_start')}-{evidence.get('page_end')} source=`{evidence.get('source_file')}`",
                            f"  - 支持字段：`{evidence.get('candidate_fields')}`",
                            f"  - snippet：{truncate(evidence.get('display_snippet') or evidence.get('text') or '', 320)}",
                        ]
                    )
                md.append("")
            md.extend(
                [
                    "```json",
                    json.dumps(
                        {
                            "local_metadata": probe.get("local_metadata"),
                            "missing_metadata_fields_after_caption": probe.get("missing_metadata_fields_after_caption"),
                            "metadata_claims": {field: compact_claim(claims.get(field)) for field in META_FIELDS},
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                    "",
                ]
            )
    md_path = package_dir / f"{stamp}_v1.0.6MetadataABC抽样包.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    (package_dir / "review_decisions.tsv").write_text("\n".join(tsv) + "\n", encoding="utf-8")
    return md_path


def metadata_table(claims: dict[str, dict[str, Any]]) -> str:
    lines = ["| field | value/abstain | evidence_ids |", "|---|---|---|"]
    for field in META_FIELDS:
        claim = claims.get(field) or {}
        if claim.get("abstain"):
            value = "ABSTAIN: " + str(claim.get("reason") or "")
        else:
            value = str(claim.get("value") or "")
        lines.append(f"| `{field}` | {value} | `{claim.get('evidence_ids') or []}` |")
    return "\n".join(lines)


def sample_record(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "abc_class": (task.get("metadata_probe") or {}).get("abc_class"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "caption_text": (task.get("gold") or {}).get("caption_text"),
        "overlay_image": task.get("overlay_image"),
        "artwork_image": task.get("artwork_image"),
        "metadata_probe": task.get("metadata_probe"),
        "claims": (task.get("gold") or {}).get("claims"),
    }


def build_summary(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    stamp: str,
    old_tasks: list[dict[str, Any]],
    pl_rows: list[dict[str, Any]],
    pagelevel_tasks: list[dict[str, Any]],
    expanded_tasks: list[dict[str, Any]],
    skipped_cap: list[dict[str, Any]],
    metadata_rows: list[dict[str, Any]],
    classified: list[dict[str, Any]],
    probe_tasks: list[dict[str, Any]],
    sample_package: Path,
) -> dict[str, Any]:
    expanded_split = Counter(task.get("split") for task in expanded_tasks)
    probe_split = Counter(task.get("split") for task in probe_tasks)
    class_counts = Counter((task.get("metadata_probe") or {}).get("abc_class") for task in classified)
    probe_class_counts = Counter((task.get("metadata_probe") or {}).get("abc_class") for task in probe_tasks)
    source_stage_counts = Counter(task.get("source_stage") for task in expanded_tasks)
    baselocate_sft_counts = count_jsonl_by_split(output_dir / "baselocate4" / "sft")
    probe_sft_counts = count_jsonl_by_split(output_dir / "metadata_probe" / "sft")
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "stamp": stamp,
        "dataset_version": "v1.0.6_baselocate4_metadata5_probe",
        "builder": "scripts/build_v1_0_6_baselocate4_metadata5_probe.py",
        "output_dir": str(output_dir),
        "args": vars(args),
        "field_schema": {"base_locate4": BASE_FIELDS, "metadata5": META_FIELDS, "all_fields": ALL_FIELDS},
        "stage1_baselocate4_expansion": {
            "old_silver_tasks": len(old_tasks),
            "pagelevel_selected_rows": len(pl_rows),
            "pagelevel_tasks_built": len(pagelevel_tasks),
            "caption_cap_skipped": len(skipped_cap),
            "expanded_tasks": len(expanded_tasks),
            "expanded_split_counts": dict(expanded_split),
            "source_stage_counts": dict(source_stage_counts),
            "sft_rows": baselocate_sft_counts,
        },
        "stage2_metadata_evidence_pool": {
            "metadata_evidence_rows": len(metadata_rows),
            "supported_field_counts": dict(Counter(field for row in metadata_rows for field in (row.get("candidate_fields") or {}))),
            "source_file_counts_top20": dict(Counter(row.get("source_file") for row in metadata_rows).most_common(20)),
        },
        "stage3_metadata_probe": {
            "classified_task_counts": dict(class_counts),
            "probe_tasks": len(probe_tasks),
            "probe_split_counts": dict(probe_split),
            "probe_class_counts": dict(probe_class_counts),
            "sft_rows": probe_sft_counts,
        },
        "artifacts": {
            "baselocate4_tasks_all": str(output_dir / "baselocate4" / "tasks_all.jsonl"),
            "baselocate4_sft_all": str(output_dir / "baselocate4" / "sft" / "all.jsonl"),
            "metadata_evidence_pool": str(output_dir / "metadata_evidence_pool.jsonl"),
            "metadata_probe_tasks_all": str(output_dir / "metadata_probe" / "tasks_all.jsonl"),
            "metadata_probe_sft_all": str(output_dir / "metadata_probe" / "sft" / "all.jsonl"),
            "abc_sample_package": str(sample_package),
            "report": str(output_dir / "构建报告.md"),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.6 BaseLocate4 + Metadata5 前三阶段构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 构建脚本：`{summary['builder']}`",
        "",
        "## 字段体系",
        "",
        f"- BaseLocate4：`{summary['field_schema']['base_locate4']}`",
        f"- Metadata5：`{summary['field_schema']['metadata5']}`",
        "",
        "## 阶段 1：BaseLocate4 扩展 SFT",
        "",
        f"- old v1.0.5 secondpass tasks：{summary['stage1_baselocate4_expansion']['old_silver_tasks']}",
        f"- page-level high-confidence selected：{summary['stage1_baselocate4_expansion']['pagelevel_selected_rows']}",
        f"- page-level tasks built：{summary['stage1_baselocate4_expansion']['pagelevel_tasks_built']}",
        f"- caption cap skipped：{summary['stage1_baselocate4_expansion']['caption_cap_skipped']}",
        f"- expanded tasks：{summary['stage1_baselocate4_expansion']['expanded_tasks']}",
        f"- split counts：`{json.dumps(summary['stage1_baselocate4_expansion']['expanded_split_counts'], ensure_ascii=False)}`",
        f"- source stage counts：`{json.dumps(summary['stage1_baselocate4_expansion']['source_stage_counts'], ensure_ascii=False)}`",
        f"- SFT rows：`{json.dumps(summary['stage1_baselocate4_expansion']['sft_rows'], ensure_ascii=False)}`",
        "",
        "说明：这一步把 page-level VLM 的 `title_like/descriptive` 高置信候选转成 BaseLocate4 task/SFT rows，并保留旧 secondpass silver 作为稳定基线。",
        "",
        "## 阶段 2：Metadata5 evidence pool",
        "",
        f"- metadata evidence rows：{summary['stage2_metadata_evidence_pool']['metadata_evidence_rows']}",
        f"- supported field counts：`{json.dumps(summary['stage2_metadata_evidence_pool']['supported_field_counts'], ensure_ascii=False)}`",
        f"- top source files：`{json.dumps(summary['stage2_metadata_evidence_pool']['source_file_counts_top20'], ensure_ascii=False)}`",
        "",
        "说明：本阶段只使用 local caption 和本地 evidence index 里的 same-document chunks。大模型内部知识没有作为 evidence source。",
        "",
        "## 阶段 3：BaseLocate4+Metadata5 probe",
        "",
        f"- classified task counts：`{json.dumps(summary['stage3_metadata_probe']['classified_task_counts'], ensure_ascii=False)}`",
        f"- selected probe tasks：{summary['stage3_metadata_probe']['probe_tasks']}",
        f"- probe split counts：`{json.dumps(summary['stage3_metadata_probe']['probe_split_counts'], ensure_ascii=False)}`",
        f"- probe class counts：`{json.dumps(summary['stage3_metadata_probe']['probe_class_counts'], ensure_ascii=False)}`",
        f"- probe SFT rows：`{json.dumps(summary['stage3_metadata_probe']['sft_rows'], ensure_ascii=False)}`",
        "",
        "A/B/C 定义：",
        "",
        "- A：caption 已包含至少 2 个 Metadata5 字段。",
        "- B：caption 不足，但同文档正文/evidence index 可支持至少 1 个缺失 Metadata5 字段。",
        "- C：caption 和同文档检索都未找到 Metadata5 支持。",
        "",
        "## 抽样包",
        "",
        f"- Markdown：`{summary['artifacts']['abc_sample_package']}`",
        "",
        "## 产物",
        "",
    ]
    for key, value in summary["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## 风险和下一步",
            "",
            "- 当前 Metadata5 抽取是规则 + 本地检索 silver，不是人工 gold；抽样包需要人工查看后再决定是否扩大。",
            "- B 类样本尤其需要检查 external evidence 是否确实绑定到当前目标图，而不是同名/相邻图。",
            "- 下一步应根据抽检结果决定是否加入 LLM/VLM 字段级裁决器，提升 `support|no_support|wrong_target` 标签质量。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def count_jsonl_by_split(sft_dir: Path) -> dict[str, int]:
    out = {}
    for split in ["train", "val", "test", "all"]:
        path = sft_dir / f"{split}.jsonl"
        if path.exists():
            out[split] = sum(1 for _ in path.open("r", encoding="utf-8"))
    return out


def base_claims_from_task(task: dict[str, Any]) -> list[dict[str, Any]]:
    claims = []
    by_field = {str(claim.get("field")): claim for claim in (task.get("gold") or {}).get("claims") or []}
    for field in BASE_FIELDS:
        claim = by_field.get(field)
        if claim:
            claims.append(copy.deepcopy(claim))
        else:
            claims.append(abstain_claim(field, "BaseLocate4 字段缺失"))
    return claims


def title_from_task(task: dict[str, Any]) -> str:
    for claim in (task.get("gold") or {}).get("claims") or []:
        if claim.get("field") == "depicted_work_title" and not claim.get("abstain"):
            return str(claim.get("value") or "")
    return ""


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


def target_region(task: dict[str, Any]) -> dict[str, Any]:
    target_id = str((task.get("gold") or {}).get("target_region_id") or "r0")
    for region in task.get("region_candidates") or []:
        if str(region.get("region_id")) == target_id or region.get("is_target"):
            return region
    regions = task.get("region_candidates") or []
    return regions[0] if regions else {"region_id": "r0", "bbox": (task.get("gold") or {}).get("image_bbox")}


def public_region_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    keys = [
        "region_id",
        "bbox",
        "type",
        "source",
        "score",
        "hint",
        "caption_evidence_id",
        "caption_hint",
        "nearby_text",
        "linked_caption_text",
        "linked_caption_region_id",
        "caption_link_score",
        "target_caption_match_score",
        "target_caption_match_reason",
        "target_region_rank",
        "target_region_sort_score",
    ]
    return [{key: region.get(key) for key in keys if key in region} for region in task.get("region_candidates") or []]


def open_local_evidence(task: dict[str, Any], evidence_id: str) -> dict[str, Any] | None:
    for item in task.get("local_evidence") or []:
        if str(item.get("evidence_id") or "") == evidence_id:
            return {
                "tool": "open_evidence",
                "evidence_id": item.get("evidence_id"),
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
                "page_end": item.get("page_end"),
                "authority_level": item.get("authority_level"),
                "citation_level": item.get("citation_level"),
                "source_quality": item.get("source_quality"),
                "display_snippet": item.get("display_snippet") or item.get("text"),
            }
    return None


def extract_metadata(text: str, title: str = "") -> dict[str, str]:
    text = normalize_space(text)
    out: dict[str, str] = {}
    creator = extract_creator(text, title)
    if creator:
        out["creator_or_attribution"] = creator
    period = extract_period(text, title)
    if period:
        out["creation_period_or_dynasty"] = period
    collection = extract_collection(text)
    if collection:
        out["collection_institution"] = collection
    dimensions = extract_dimensions(text)
    if dimensions:
        out["dimensions"] = dimensions
    medium = extract_medium(text)
    if medium:
        out["medium_material"] = medium
    return out


def extract_title(text: str) -> str:
    text = normalize_space(text)
    match = re.search(r"《([^》]{1,80})》", text)
    if match:
        title = normalize_space(match.group(1))
        if 1 <= len(title) <= 80:
            return title
    plain_chinese = re.search(
        r"(?:图|圖)\s*[一二三四五六七八九十百〇零0-9]+(?:[.\-．:：][一二三四五六七八九十百〇零0-9]+)*[a-zA-Z]?\s*[:：]?\s*([\u4e00-\u9fff]{2,30})(?:[（(]|$|\s)",
        text,
    )
    if plain_chinese:
        candidate = normalize_space(plain_chinese.group(1))
        if not re.search(r"^(局部|细部|細部|部分|索桥|栏杆|桥栏|中|第)", candidate):
            return candidate
    english_title = extract_english_title(text)
    if english_title:
        return english_title
    english = re.search(
        r"(?:Attributed to|After)?\s*[A-Z][A-Za-z'\-\. ]{2,80}(?:\([^)]{0,60}\))?,\s*([^,;.]{3,100}),\s*(?:dated|ca\.|[0-9]{3,4}|Northern|Southern|Ming|Song|Yuan|Qing|Hanging|Album|Handscroll)",
        text,
    )
    if english:
        return normalize_space(english.group(1))
    return ""


def extract_english_title(text: str) -> str:
    body = normalize_space(text)
    body = re.sub(r"^(?:Figure|Fig\.?|Plate)\s*[A-Za-z]?[0-9IVXivx]+(?:[.\-．:：][0-9IVXivx]+)*[a-zA-Z]?\.?\s*", "", body, flags=re.I)
    without_artist = re.sub(
        r"^(?:(?:Attributed to|After)\s+)?[A-Z][A-Za-z'\-\. ]{2,80}(?:\s*\([^)]{0,100}\))?,\s*",
        "",
        body,
        count=1,
    )
    if without_artist == body:
        return ""
    cut_patterns = [
        r",\s*signed and dated\s*[0-9]",
        r",\s*(?:dated|ca\.)\s*[0-9]",
        r",\s*[\"'“”]?\s*[0-9]{3,4}s?\b",
        r",\s*(?:Northern|Southern|Song|Yuan|Ming|Qing|Tang)\b",
        r"\.\s*(?:Hanging scroll|Handscroll|Album leaf|Fan leaf|Scroll|Ink|Color)\b",
    ]
    end = len(without_artist)
    for pattern in cut_patterns:
        match = re.search(pattern, without_artist, flags=re.I)
        if match:
            end = min(end, match.start())
    title = without_artist[:end].strip(" ,.;:：\"'“”")
    title = normalize_space(title)
    if title.count('"') % 2 == 1:
        title += '"'
    if 3 <= len(title) <= 120 and not re.search(r"\b(?:Hanging scroll|Handscroll|Album leaf|ink on|ink and color)\b", title, flags=re.I):
        return title
    return ""


def extract_creator(text: str, title: str = "") -> str:
    text = normalize_space(text)
    if title:
        escaped = re.escape(title)
        possessive = re.search(rf"([\u4e00-\u9fff]{{2,4}})的(?:作品|画作|山水画)?\s*《{escaped}》", text)
        if possessive:
            name = possessive.group(1)
            if valid_creator_value(name):
                return name
        match = re.search(rf"(?:\d{{3,4}}年)?(?:北宋|南宋|宋代|宋|元代|元|明代|明|清代|清|唐代|唐|五代|辽|遼|金代|金|民国|民國)?[·\s・]*(传|傳)?([\u4e00-\u9fff]{{2,4}})\s*《{escaped}》", text)
        if match:
            prefix = "传" if match.group(1) else ""
            name = match.group(2)
            if valid_creator_value(prefix + name):
                return prefix + name
    match = re.search(r"(?:Figure|Fig\.?|Plate)\s*[0-9.\-a-zA-Z]+\s+([A-Z][A-Za-z'\-\. ]{2,60})(?:\s*\([^)]{0,80}\))?,", text, flags=re.I)
    if match:
        return normalize_space(match.group(1))
    match = re.search(r"(?:Attributed to|After)\s+([A-Z][A-Za-z'\-\. ]{2,80})", text, flags=re.I)
    if match:
        return normalize_space(match.group(0).rstrip(","))
    return ""


def extract_period(text: str, title: str = "") -> str:
    dated = re.search(r"\bdated\s+([0-9]{3,4})\b", text or "", flags=re.I)
    if dated:
        return "dated " + dated.group(1)
    for match in DYNASTY_PATTERN.finditer(text or ""):
        value = normalize_space(match.group(1))
        if invalid_period_context(text or "", match.start(), match.end(), value):
            continue
        if value in {"唐", "宋", "元", "明", "清"} and not valid_single_char_dynasty_context(text or "", match.start(), value):
            continue
        return value
    return ""


def invalid_period_context(text: str, start: int, end: int, value: str) -> bool:
    after = text[end : end + 8]
    before = text[max(0, start - 12) : start]
    if re.match(r"(?:以来|诗人|畫家|画家|文人|文化|艺术|藝術|时期|時期|风格|風格)", after):
        return True
    if value in {"唐代", "宋代", "元代", "明代", "清代", "唐", "宋", "元", "明", "清"} and re.search(r"《[^》]{0,20}$", before):
        return True
    return False


def valid_single_char_dynasty_context(text: str, pos: int, value: str) -> bool:
    window = text[pos : pos + 16]
    if re.match(rf"{re.escape(value)}[\u4e00-\u9fff]{{1,3}}的(?:作品|画作|山水画)?\s*《", window):
        return False
    if re.match(rf"{re.escape(value)}代", window):
        return True
    if re.match(rf"{re.escape(value)}[·\s・]*[\u4e00-\u9fff]{{2,4}}《", window):
        return True
    before = text[max(0, pos - 4) : pos]
    after = text[pos + len(value) : pos + len(value) + 4]
    if not re.search(r"[\u4e00-\u9fff]", before + after):
        return True
    return False


def extract_collection(text: str) -> str:
    for pattern in INSTITUTION_PATTERNS:
        match = pattern.search(text or "")
        if match:
            value = normalize_space(match.group(1))
            after = (text or "")[match.end() : match.end() + 4]
            if after.startswith(("院刊", "学报", "期刊")):
                continue
            return value
    return ""


def extract_dimensions(text: str) -> str:
    match = DIMENSION_PATTERN.search(text or "")
    return normalize_space(match.group(1)) if match else ""


def extract_medium(text: str) -> str:
    for pattern, fixed in MEDIUM_PATTERNS:
        match = pattern.search(text or "")
        if match:
            return normalize_space(fixed or match.group(0))
    return ""


def focus_text_for_target(text: str, title: str, caption: str) -> str:
    """Return a local text window around the target title/figure label.

    Evidence chunks often contain several adjacent figure captions.  Metadata
    extraction must not scan the whole chunk; otherwise a date or collection
    from a neighboring figure can be incorrectly attached to the target.
    """

    text = normalize_space(text)
    if not text:
        return ""
    labels = figure_labels(caption)
    compact_text = compact(text)
    title_positions: list[int] = []
    title_len = 0
    if title:
        title_compact = compact(title)
        if title_compact:
            search_from = 0
            while True:
                pos = compact_text.find(title_compact, search_from)
                if pos < 0:
                    break
                title_positions.append(approximate_raw_position(text, compact_text, pos))
                search_from = pos + max(1, len(title_compact))
            title_len = len(title)
    label_pos: int | None = None
    label_len = 0
    for label in labels:
        label_compact = compact(label)
        if not label_compact:
            continue
        pos = compact_text.find(label_compact)
        if pos < 0:
            continue
        raw_pos = approximate_raw_position(text, compact_text, pos)
        if label_pos is None or raw_pos < label_pos:
            label_pos = raw_pos
            label_len = len(label)
    if label_pos is not None:
        nearby_titles = [pos for pos in title_positions if abs(label_pos - pos) <= 220]
        if nearby_titles:
            title_pos = min(nearby_titles, key=lambda pos: abs(label_pos - pos))
            start = max(0, title_pos - 12)
            end = min(len(text), label_pos + max(120, label_len) + 360)
            return text[start:end]
        end = min(len(text), label_pos + max(120, label_len) + 420)
        return text[label_pos:end]

    best_pos: int | None = None
    best_len = 0
    for anchor in [title] if title else []:
        anchor_compact = compact(anchor)
        if not anchor_compact:
            continue
        pos = compact_text.find(anchor_compact)
        if pos < 0:
            continue
        raw_pos = approximate_raw_position(text, compact_text, pos)
        if best_pos is None or raw_pos < best_pos:
            best_pos = raw_pos
            best_len = len(anchor)
    if best_pos is None:
        return text[:500]
    start = best_pos
    end = min(len(text), best_pos + max(120, best_len) + 360)
    return text[start:end]


def approximate_raw_position(raw: str, compact_raw: str, compact_pos: int) -> int:
    if compact_pos <= 0:
        return 0
    seen = 0
    for idx, char in enumerate(raw):
        if compact(char):
            if seen >= compact_pos:
                return idx
            seen += 1
    return min(len(raw), compact_pos)


def metadata_query(task: dict[str, Any], title: str, caption: str, missing_fields: list[str]) -> str:
    terms = [title, first_figure_label(caption)]
    field_terms = []
    if "creator_or_attribution" in missing_fields:
        field_terms.append("作者 artist attributed")
    if "creation_period_or_dynasty" in missing_fields:
        field_terms.append("朝代 dynasty century period")
    if "collection_institution" in missing_fields:
        field_terms.append("藏馆 collection museum")
    if "dimensions" in missing_fields:
        field_terms.append("尺寸 dimensions cm")
    if "medium_material" in missing_fields:
        field_terms.append("材质 纸本 绢本 ink silk paper")
    query = " ".join(term for term in terms + field_terms if term)
    return truncate(normalize_space(query), 180)


def build_evidence_query(source_file: Any, caption: Any, title: Any) -> str:
    return truncate(normalize_space(f"{Path(str(source_file or '')).stem} {caption or ''} {title or ''} 山水画 metadata"), 180)


def map_image_scope(value: Any, caption: str) -> str:
    raw = str(value or "")
    caption_l = str(caption or "").lower()
    if raw == "partial_detail" or re.search(r"局部|细部|細部|detail|部分", caption_l):
        return "partial_detail"
    if raw == "full_work":
        return "figure_or_plate"
    if raw == "multi_work_comparison":
        return "figure_or_plate"
    if re.search(r"册|冊|album leaf|handscroll|scroll|卷", caption_l):
        return "album_leaf_or_scroll_section"
    return "figure_or_plate"


def map_object_type(value: Any) -> str:
    if value == "landscape_detail":
        return "landscape_detail"
    if value == "landscape_painting":
        return "landscape_painting"
    return "painting"


def supported_claim(
    field: str,
    value: Any,
    evidence_ids: list[str],
    candidate_ids: list[str],
    confidence: float,
    support_type: str,
    visual_bbox: Any = None,
) -> dict[str, Any]:
    return {
        "claim_id": field,
        "field": field,
        "value": value,
        "abstain": False,
        "evidence_ids": [eid for eid in dedupe(evidence_ids) if eid],
        "candidate_evidence_ids": dedupe(candidate_ids),
        "support_type": support_type,
        "confidence": round(float(confidence), 3),
        "visual_bbox": visual_bbox,
    }


def abstain_claim(field: str, reason: str) -> dict[str, Any]:
    return {"claim_id": field, "field": field, "value": None, "abstain": True, "reason": reason, "confidence": 0.72}


def claim_state(draft_claims: list[dict[str, Any]], target_fields: list[str]) -> dict[str, Any]:
    by_field = {str(item.get("field")): item for item in draft_claims if item.get("field")}
    written = [field for field in target_fields if field in by_field and not by_field[field].get("abstain")]
    abstained = [field for field in target_fields if field in by_field and by_field[field].get("abstain")]
    evidence_ids = dedupe(str(eid) for item in draft_claims for eid in (item.get("evidence_ids") or []) if isinstance(item, dict))
    return {
        "target_fields": list(target_fields),
        "written_fields": written,
        "abstained_fields": abstained,
        "remaining_fields": [field for field in target_fields if field not in by_field],
        "claim_count": len(written),
        "abstain_count": len(abstained),
        "evidence_ids": evidence_ids,
    }


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


def phase_hint(action: dict[str, Any], target_fields: list[str]) -> str:
    suffix = "Metadata5 字段没有证据时必须 abstain。" if target_fields == ALL_FIELDS else ""
    return {
        "inspect_page": "先读取页面布局候选，不要直接裁剪、检索或写 claim。",
        "crop_target": "根据 inspect_page 返回的目标候选区域裁剪目标图像。",
        "open_evidence": "打开已经可见或已检索到的 evidence_id。",
        "retrieve_evidence": "用作品名/图号合并检索作者、年代、藏馆、尺寸、材质等缺失 metadata。",
        "write_claims_chunk": "一次只写入或 abstain 一个 remaining field。" + suffix,
        "finish": "只有 remaining_fields 为空时才能结束。",
    }.get(str(action.get("action") or ""), "")


def figure_labels(text: str) -> list[str]:
    return [normalize_space(match.group(1)) for match in FIGURE_LABEL_PATTERN.finditer(text or "")]


def first_figure_label(text: str) -> str:
    labels = figure_labels(text)
    return labels[0] if labels else ""


def caption_key(text: str) -> str:
    return compact(text)[:160]


def compact(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("圖", "图").replace("（", "(").replace("）", ")").replace("．", ".").replace("·", ".")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def truncate(value: Any, limit: int) -> str:
    text = normalize_space(value)
    return text if len(text) <= limit else text[:limit] + "..."


def dedupe(values: Iterable[Any]) -> list[Any]:
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_asset(path_value: Any, assets_dir: Path, name: str) -> str:
    src = Path(str(path_value or ""))
    dst = assets_dir / sanitize_filename(name)
    if src.exists():
        shutil.copy2(src, dst)
    return "assets/" + dst.name


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def claim_value(claim: dict[str, Any] | None) -> str:
    if not claim:
        return ""
    if claim.get("abstain"):
        return "ABSTAIN"
    return str(claim.get("value") or "")


def compact_claim(claim: dict[str, Any] | None) -> dict[str, Any]:
    if not claim:
        return {}
    return {
        "value": claim.get("value"),
        "abstain": claim.get("abstain"),
        "evidence_ids": claim.get("evidence_ids") or [],
        "reason": claim.get("reason"),
        "support_type": claim.get("support_type"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
