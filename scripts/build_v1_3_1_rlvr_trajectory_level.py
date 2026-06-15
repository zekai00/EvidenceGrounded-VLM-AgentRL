#!/usr/bin/env python3
"""Build native v1.3.1 trajectory-level RLVR data.

This script intentionally does not call the old v0.x data builders. It reads
the v1.3.1 FigureTarget / EvidenceFragment / FieldSupportLabel artifacts and
exports an executable RLVR dataset for the existing verl AgentLoop smoke path.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


V13_FIELDS = [
    "caption_text",
    "depicted_work_title",
    "image_scope",
    "object_type",
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
]

BASELOCATE4 = {"caption_text", "depicted_work_title", "image_scope", "object_type"}
METADATA5 = {
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_3_1_remote_vlm_evidence_sft_20260614_1335",
    )
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--max-train", type=int, default=0, help="0 means all train tasks.")
    parser.add_argument("--max-val", type=int, default=0, help="0 means all val tasks.")
    parser.add_argument("--max-test", type=int, default=0, help="0 means all test tasks.")
    parser.add_argument("--image-max-pixels", type=int, default=196608)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--tool-schema", default="no_select", choices=["no_select", "inspect_crop", "chunked_claim"])
    parser.add_argument("--latest-link", default="rlvr_v1_3_1_trajectory_level_latest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"rlvr_v1_3_1_trajectory_level_{now_tag()}"
    out_dir.mkdir(parents=True, exist_ok=False)
    (out_dir / "tasks").mkdir()
    (out_dir / "evidence_index").mkdir()
    (out_dir / "verl").mkdir()
    (out_dir / "review").mkdir()

    support_rows = read_jsonl(dataset_dir / "field_support_labels.jsonl")
    support_by_target_field, fields_by_fragment = build_support_maps(support_rows)
    all_source_fragments = read_jsonl(dataset_dir / "evidence_fragments.jsonl")
    fragment_by_id = {str(row.get("fragment_id")): row for row in all_source_fragments if row.get("fragment_id")}

    splits = {
        "train": load_split(dataset_dir / "tasks" / "train_tasks.jsonl", args.max_train),
        "val": load_split(dataset_dir / "tasks" / "val_tasks.jsonl", args.max_val),
        "test": load_split(dataset_dir / "tasks" / "test_tasks.jsonl", args.max_test),
    }
    converted: dict[str, list[dict[str, Any]]] = {}
    for split, tasks in splits.items():
        converted[split] = [
            convert_task(task, support_by_target_field, fields_by_fragment, fragment_by_id, split) for task in tasks
        ]
        write_jsonl(out_dir / "tasks" / f"{split}_tasks.jsonl", converted[split])
    all_tasks = [task for split in ["train", "val", "test"] for task in converted[split]]
    write_jsonl(out_dir / "tasks" / "all_tasks.jsonl", all_tasks)

    corpus_chunks = build_corpus_chunks(all_source_fragments, fields_by_fragment)
    write_jsonl(out_dir / "evidence_index" / "corpus_chunks.jsonl", corpus_chunks)

    parquet_paths: dict[str, str] = {}
    for split, tasks in converted.items():
        records = [
            build_verl_record(
                task,
                tasks_path=out_dir / "tasks" / f"{split}_tasks.jsonl",
                evidence_index=out_dir / "evidence_index",
                args=args,
                row_index=index,
            )
            for index, task in enumerate(tasks)
        ]
        parquet_path = out_dir / "verl" / f"{split}.parquet"
        pd.DataFrame(records).to_parquet(parquet_path, index=False)
        parquet_paths[split] = str(parquet_path)
        write_jsonl(out_dir / "verl" / f"{split}_preview.jsonl", records[:5])

    manifest = {
        "created_at": now_human(),
        "dataset_version": "rlvr_v1_3_1_trajectory_level",
        "source_dataset_dir": str(dataset_dir),
        "output_dir": str(out_dir),
        "fields": V13_FIELDS,
        "baselocate4": sorted(BASELOCATE4),
        "metadata5": sorted(METADATA5),
        "tool_schema": args.tool_schema,
        "max_steps": args.max_steps,
        "image_max_pixels": args.image_max_pixels,
        "splits": {
            split: {
                "tasks": len(tasks),
                "trajectory_type_counts": dict(Counter(str(task.get("trajectory_type")) for task in tasks)),
                "tasks_jsonl": str(out_dir / "tasks" / f"{split}_tasks.jsonl"),
                "verl_parquet": parquet_paths[split],
            }
            for split, tasks in converted.items()
        },
        "evidence_index": {
            "chunks": len(corpus_chunks),
            "path": str(out_dir / "evidence_index" / "corpus_chunks.jsonl"),
        },
        "support_label_counts": dict(Counter(str(row.get("label")) for row in support_rows)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_build_report(out_dir / "构建报告.md", manifest)
    write_review_report(out_dir / "人工抽检.md", converted)
    update_latest_link(Path(args.output_root), args.latest_link, out_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def convert_task(
    task: dict[str, Any],
    support_by_target_field: dict[tuple[str, str], list[str]],
    fields_by_fragment: dict[str, set[str]],
    fragment_by_id: dict[str, dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    target_id = str(task.get("task_id"))
    local_evidence = [
        convert_fragment(fragment, fields_by_fragment.get(str(fragment.get("fragment_id")), set()))
        for fragment in task.get("fragments") or []
        if isinstance(fragment, dict)
    ]
    claims = build_gold_claims(task, support_by_target_field)
    evidence_ids = stable_unique(
        [eid for claim in claims for eid in (claim.get("evidence_ids") or claim.get("candidate_evidence_ids") or [])]
    )
    converted = {
        "task_id": target_id,
        "split": split,
        "dataset_version": "rlvr_v1_3_1_trajectory_level",
        "source_dataset_version": task.get("dataset_version"),
        "tool_schema_version": "v1.3.1_rlvr_trajectory_level",
        "task_type": "v1_3_1_trajectory_level_rlvr",
        "trajectory_type": task.get("trajectory_type"),
        "source_file": task.get("source_file"),
        "source_path": task.get("source_path"),
        "page": task.get("page"),
        "page_id": task.get("page_id"),
        "page_image": task.get("page_image"),
        "artwork_image": task.get("artwork_image"),
        "caption_image": task.get("caption_image"),
        "overlay_image": task.get("overlay_image"),
        "target_fields": V13_FIELDS,
        "target_bbox_norm1000": task.get("target_bbox_norm1000"),
        "target_bbox_px": task.get("target_bbox_px"),
        "caption_bbox_norm1000": task.get("caption_bbox_norm1000"),
        "caption_bbox_px": task.get("caption_bbox_px"),
        "local_evidence": local_evidence,
        "region_candidates": build_region_candidates(task, local_evidence),
        "gold_fields": task.get("gold_fields") or {},
        "gold": {
            "image_bbox": task.get("target_bbox_px"),
            "image_bbox_norm1000": task.get("target_bbox_norm1000"),
            "caption_bbox": task.get("caption_bbox_px"),
            "caption_bbox_norm1000": task.get("caption_bbox_norm1000"),
            "claims": claims,
            "evidence_ids": evidence_ids,
            "candidate_evidence_ids": evidence_ids,
        },
        "rlvr_notes": {
            "reward_level": "trajectory",
            "bbox_label_source": "v1.3.1 silver target/caption bbox",
            "field_support_source": "FieldSupportLabel",
            "field_protocol": "BaseLocate4 + Metadata5",
        },
    }
    # Ensure every evidence id referenced by gold claims is locally visible or in the index.
    known = {str(item.get("evidence_id")) for item in local_evidence}
    missing = [eid for eid in evidence_ids if eid not in known and eid not in fragment_by_id]
    if missing:
        converted["rlvr_notes"]["missing_gold_evidence_ids"] = missing
    return converted


def build_gold_claims(
    task: dict[str, Any],
    support_by_target_field: dict[tuple[str, str], list[str]],
) -> list[dict[str, Any]]:
    target_id = str(task.get("task_id"))
    gold_fields = task.get("gold_fields") if isinstance(task.get("gold_fields"), dict) else {}
    claims: list[dict[str, Any]] = []
    for field in V13_FIELDS:
        spec = gold_fields.get(field) if isinstance(gold_fields.get(field), dict) else {}
        support_ids = support_by_target_field.get((target_id, field), [])
        if spec.get("abstain") or not present_value(spec.get("value")):
            claims.append(
                {
                    "field": field,
                    "abstain": True,
                    "reason": spec.get("reason") or "no reliable visible/supporting evidence in v1.3.1 gold_fields",
                    "candidate_evidence_ids": support_ids,
                }
            )
            continue
        evidence_ids = stable_unique([str(eid) for eid in (spec.get("evidence_ids") or [])] + support_ids)
        claims.append(
            {
                "field": field,
                "value": spec.get("value"),
                "evidence_ids": evidence_ids,
                "candidate_evidence_ids": evidence_ids,
                "confidence": 1.0,
                "field_group": "BaseLocate4" if field in BASELOCATE4 else "Metadata5",
            }
        )
    return claims


def convert_fragment(fragment: dict[str, Any], allowed_fields: set[str]) -> dict[str, Any]:
    fragment_id = str(fragment.get("fragment_id"))
    text = (
        fragment.get("display_text")
        or fragment.get("corrected_text")
        or fragment.get("raw_text")
        or ("目标裁剪图像视觉证据" if fragment.get("fragment_type") == "local_visual" else "")
    )
    allowed = sorted(field for field in allowed_fields if field in V13_FIELDS)
    return {
        "evidence_id": fragment_id,
        "fragment_id": fragment_id,
        "target_id": fragment.get("target_id"),
        "fragment_type": fragment.get("fragment_type"),
        "clean_evidence_type": fragment.get("fragment_type"),
        "source_file": fragment.get("source_file"),
        "page": fragment.get("page_num"),
        "page_start": fragment.get("page_num"),
        "page_end": fragment.get("page_num"),
        "source_bbox_norm1000": fragment.get("source_bbox_norm1000"),
        "source_bbox_px": fragment.get("source_bbox_px"),
        "image_path": fragment.get("image_path"),
        "text": text,
        "clean_text": text,
        "display_snippet": truncate(text, 420),
        "evidence_summary": truncate(text, 420),
        "authority_level": "v1.3.1_local_or_same_page",
        "citation_level": fragment.get("fragment_type"),
        "source_quality": "silver_vlm_constructed",
        "adjudication_status": "accepted_auto" if allowed else "no_supported_fields",
        "adjudicated_evidence_role": fragment.get("fragment_type"),
        "claim_allowed_fields": allowed,
        "adjudicated_claim_allowed_fields": allowed,
        "usable_for_claim_by_adjudication": bool(allowed),
        "usable_for_retrieval": fragment.get("fragment_type") in {"same_page_body", "wrong_target_caption", "local_caption_visual"},
    }


def build_region_candidates(task: dict[str, Any], local_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    caption_evidence = next((item for item in local_evidence if item.get("fragment_type") == "local_caption_visual"), None)
    caption_text = str((caption_evidence or {}).get("display_snippet") or "")
    if valid_bbox(task.get("target_bbox_px")):
        regions.append(
            {
                "region_id": "r_target_candidate",
                "bbox": task.get("target_bbox_px"),
                "type": "figure_candidate",
                "source": "v1.3.1_silver_target_bbox",
                "score": 0.95,
                "caption_evidence_id": (caption_evidence or {}).get("evidence_id"),
                "caption_hint": caption_text,
            }
        )
    if valid_bbox(task.get("caption_bbox_px")):
        regions.append(
            {
                "region_id": "r_caption_candidate",
                "bbox": task.get("caption_bbox_px"),
                "type": "text_or_caption_candidate",
                "source": "v1.3.1_silver_caption_bbox",
                "score": 0.90,
                "caption_evidence_id": (caption_evidence or {}).get("evidence_id"),
                "caption_hint": caption_text,
                "nearby_text": caption_text,
            }
        )
    page_bbox = task.get("target_bbox_px") or [0, 0, 1000, 1000]
    if valid_bbox(page_bbox):
        x1, y1, x2, y2 = [int(v) for v in page_bbox]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        regions.append(
            {
                "region_id": "r_context_expand",
                "bbox": [max(0, x1 - width // 5), max(0, y1 - height // 5), x2 + width // 5, y2 + height // 5],
                "type": "figure_candidate",
                "source": "v1.3.1_context_distractor",
                "score": 0.55,
                "hint": "expanded nearby region; may include context or adjacent text",
            }
        )
    return regions


def build_corpus_chunks(
    fragments: list[dict[str, Any]],
    fields_by_fragment: dict[str, set[str]],
) -> list[dict[str, Any]]:
    chunks = []
    for fragment in fragments:
        chunks.append(convert_fragment(fragment, fields_by_fragment.get(str(fragment.get("fragment_id")), set())))
    return chunks


def build_support_maps(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, str], list[str]], dict[str, set[str]]]:
    support_by_target_field: dict[tuple[str, str], list[str]] = defaultdict(list)
    fields_by_fragment: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("label") != "support":
            continue
        target_id = str(row.get("target_id") or "")
        field = str(row.get("field") or "")
        fragment_id = str(row.get("fragment_id") or "")
        if not target_id or field not in V13_FIELDS or not fragment_id:
            continue
        support_by_target_field[(target_id, field)].append(fragment_id)
        fields_by_fragment[fragment_id].add(field)
    return {key: stable_unique(value) for key, value in support_by_target_field.items()}, fields_by_fragment


def build_verl_record(
    task: dict[str, Any],
    *,
    tasks_path: Path,
    evidence_index: Path,
    args: argparse.Namespace,
    row_index: int,
) -> dict[str, Any]:
    prompt_text = build_initial_prompt(task, args)
    ground_truth = {
        "task_id": task.get("task_id"),
        "tasks_path": str(tasks_path),
        "evidence_index": str(evidence_index),
        "max_steps": args.max_steps,
        "phase_aware_mask": True,
        "enforce_tool_mask": True,
        "tool_schema": args.tool_schema,
        "target_claim_fields": V13_FIELDS,
        "reward_mode": "trajectory_level_v1_3_1",
    }
    return {
        "data_source": "evidence_grounded_v1_3_1_trajectory_rlvr",
        "prompt": [{"role": "user", "content": "<image>\n" + prompt_text}],
        "images": [{"image": task.get("page_image"), "max_pixels": args.image_max_pixels}],
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "index": row_index,
            "task_id": task.get("task_id"),
            "split": task.get("split"),
            "trajectory_type": task.get("trajectory_type"),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "agent_name": "evidence_stepwise_agent",
            "reward_level": "trajectory",
        },
    }


def build_initial_prompt(task: dict[str, Any], args: argparse.Namespace) -> str:
    hints = [
        {
            "evidence_id": item.get("evidence_id"),
            "fragment_type": item.get("fragment_type"),
            "snippet": truncate(item.get("display_snippet"), 70),
            "claim_allowed_fields": item.get("claim_allowed_fields"),
        }
        for item in task.get("local_evidence") or []
        if item.get("fragment_type") in {"local_caption_visual", "local_visual"}
    ]
    return "\n".join(
        [
            "EvidenceGrounded-VLM-AgentRL v1.3.1 tool-call agent.",
            "Task: locate the target Chinese landscape figure on this PDF page and complete BaseLocate4+Metadata5. Every non-abstain field needs cited evidence.",
            f"task_id：{task.get('task_id')}；source_file：{task.get('source_file')}；page：{task.get('page')}",
            "fields=" + json.dumps(V13_FIELDS, ensure_ascii=False, separators=(",", ":")),
            "Output exactly one JSON object per turn. No markdown, no prose, no list, no code fence.",
            "First action only: {\"action\":\"inspect_page\",\"top_k\":12}",
            "Later actions: inspect_page,crop_target,open_evidence,retrieve_evidence,write_claim,abstain_claim,finish. No select_evidence.",
            "Use only evidence ids visible in state/tool results. If support is insufficient, abstain_claim. Do not fill artist/period/collection/dimensions/medium from common knowledge.",
            "Schema keys: crop_target(region_id), open_evidence(evidence_id), retrieve_evidence(query,scope,top_k), write_claim(field,value,evidence_ids,confidence), abstain_claim(field,reason), finish(status).",
            "local_hints_not_answers=" + json.dumps(hints[:2], ensure_ascii=False, separators=(",", ":")),
            "Now output only {\"action\":\"inspect_page\",\"top_k\":12}.",
        ]
    )


def write_build_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.3.1 Trajectory-Level RLVR 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- source_dataset_dir: `{manifest['source_dataset_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- tool_schema: `{manifest['tool_schema']}`",
        f"- max_steps: {manifest['max_steps']}",
        "",
        "## Splits",
        "",
        "| split | tasks | trajectory types | parquet |",
        "|---|---:|---|---|",
    ]
    for split, info in manifest["splits"].items():
        lines.append(
            f"| {split} | {info['tasks']} | `{json.dumps(info['trajectory_type_counts'], ensure_ascii=False)}` | `{info['verl_parquet']}` |"
        )
    lines.extend(
        [
            "",
            "## Reward 口径",
            "",
            "- 第一阶段采用 trajectory-level reward。",
            "- 字段协议为 v1.3.1 九字段：BaseLocate4 + Metadata5。",
            "- `FieldSupportLabel.label=support` 被转换为 `claim_allowed_fields`，供 verifier 判断 evidence support。",
            "- bbox 来自 v1.3.1 silver label，第一阶段主要用于 crop/grounding 辅助，不作为唯一主 reward。",
            "",
            "## 训练前说明",
            "",
            "- 本脚本只构建 RLVR 数据和 verl parquet，不启动训练。",
            "- `latest` 软链接会指向本次输出目录，远端 smoke 脚本默认读取 `latest/verl`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_report(path: Path, converted: dict[str, list[dict[str, Any]]]) -> None:
    lines = ["# v1.3.1 Trajectory-Level RLVR 人工抽检", "", f"时间：{now_human()}", ""]
    for split in ["train", "val", "test"]:
        rows = converted.get(split) or []
        if not rows:
            continue
        lines.extend([f"## {split}", ""])
        for task in rows[:3]:
            claims = task.get("gold", {}).get("claims", [])
            supported = [c for c in claims if not c.get("abstain")]
            abstained = [c for c in claims if c.get("abstain")]
            lines.extend(
                [
                    f"### {task.get('task_id')} `{task.get('trajectory_type')}`",
                    "",
                    f"- source: `{task.get('source_file')}` p{task.get('page')}",
                    f"- page_image: `{task.get('page_image')}`",
                    f"- overlay_image: `{task.get('overlay_image')}`",
                    f"- target_bbox_px: `{task.get('target_bbox_px')}`",
                    f"- caption_bbox_px: `{task.get('caption_bbox_px')}`",
                    f"- supported_fields: `{[c.get('field') for c in supported]}`",
                    f"- abstained_fields: `{[c.get('field') for c in abstained]}`",
                    "",
                ]
            )
            if task.get("overlay_image"):
                lines.append(f"![overlay]({task.get('overlay_image')})")
                lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def load_split(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    return rows[:limit] if limit and limit > 0 else rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def update_latest_link(output_root: Path, latest_name: str, out_dir: Path) -> None:
    latest = output_root / latest_name
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    latest.symlink_to(out_dir, target_is_directory=True)


def present_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {"/na", "na", "none", "null"}
    return True


def valid_bbox(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def stable_unique(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def truncate(text: Any, limit: int) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 3] + "..."


def now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


def now_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
