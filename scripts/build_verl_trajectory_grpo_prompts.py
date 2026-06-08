#!/usr/bin/env python3
"""Build single-response trajectory GRPO prompts for verl.

This is a bridge dataset for trajectory-level GRPO smoke tests. The final
interactive setup should let the policy observe tool returns after every step.
For now, each prompt asks the policy to emit a full JSON action list in one
response, so we include compact tool-visible candidates that the first few env
steps would normally reveal.
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CLAIM_FIELD_SPEC = (
    "caption_text|image_scope|depicted_work_title|displayed_region|object_type|artist|dynasty|"
    "visual_elements|technique|composition|medium_dimensions|collection"
)


TOOL_SCHEMA_CHUNKED = [
    '{"action":"propose_regions","top_k":10}',
    '{"action":"select_evidence","evidence_ids":["local_caption_xxx或ev_xxx"]}',
    '{"action":"crop_region","region_id":"r_xxx"}',
    '{"action":"retrieve_evidence","query":"...","scope":"current_page","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
    '{"action":"open_evidence","evidence_id":"ev_xxx或local_caption_xxx"}',
    f'{{"action":"write_claims_chunk","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
    '{"action":"finish","status":"done"}',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-train", type=int, default=50)
    parser.add_argument("--max-val", type=int, default=50)
    parser.add_argument("--max-test", type=int, default=50)
    parser.add_argument("--image-max-pixels", type=int, default=131072)
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--tool-schema", choices=["chunked_claim"], default="chunked_claim")
    parser.add_argument("--region-top-k", type=int, default=10)
    parser.add_argument("--max-local-evidence", type=int, default=6)
    parser.add_argument("--max-candidate-evidence", type=int, default=10)
    parser.add_argument("--max-snippet-chars", type=int, default=180)
    parser.add_argument("--include-static-tool-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase-aware-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-tool-mask", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(args.tasks)
    args.evidence_items = load_evidence_items(args.evidence_index)
    split_limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    manifest: dict[str, Any] = {
        "created_at": now(),
        "source_tasks": args.tasks,
        "evidence_index": args.evidence_index,
        "output_dir": str(out_dir),
        "image_max_pixels": args.image_max_pixels,
        "max_steps": args.max_steps,
        "tool_schema": args.tool_schema,
        "region_top_k": args.region_top_k,
        "max_candidate_evidence": args.max_candidate_evidence,
        "include_static_tool_preview": args.include_static_tool_preview,
        "phase_aware_mask": args.phase_aware_mask,
        "enforce_tool_mask": args.enforce_tool_mask,
        "splits": {},
    }
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        rows = [task for task in tasks if str(task.get("split")) == split]
        limit = split_limits.get(split, -1)
        if limit and limit > 0:
            rows = rows[:limit]
        records = [build_record(task, args) for task in rows]
        path = out_dir / f"{split}.parquet"
        pd.DataFrame(records).to_parquet(path, index=False)
        preview = out_dir / f"{split}_preview.jsonl"
        write_jsonl(preview, records[:5])
        manifest["splits"][split] = {
            "rows": len(records),
            "parquet": str(path),
            "preview": str(preview),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_record(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    image = {"image": task.get("page_image"), "max_pixels": args.image_max_pixels}
    prompt_text = build_prompt_text(task, args)
    prompt = [{"role": "user", "content": "<image>\n" + prompt_text}]
    ground_truth = {
        "task_id": task.get("task_id"),
        "tasks_path": args.tasks,
        "evidence_index": args.evidence_index,
        "max_steps": args.max_steps,
        "phase_aware_mask": bool(args.phase_aware_mask),
        "enforce_tool_mask": bool(args.enforce_tool_mask),
    }
    return {
        "data_source": "evidence_grounded_trajectory",
        "prompt": prompt,
        "images": [image],
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "task_id": task.get("task_id"),
            "split": task.get("split"),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "index": task.get("task_id"),
        },
    }


def build_prompt_text(task: dict[str, Any], args: argparse.Namespace) -> str:
    tools = "\n".join(f"{idx + 1}. {schema}" for idx, schema in enumerate(TOOL_SCHEMA_CHUNKED))
    lines = [
        "你是 evidence-grounded figure understanding 的 VLM tool-call agent。",
        "任务：给定 PDF 页面图像，输出一整条可执行工具调用轨迹，用于定位目标山水画图像、选择证据、写出有证据支撑的结构化 claim，并 finish。",
        "注意：这是 trajectory-level GRPO 的单响应桥接任务。你要一次性输出完整 JSON 数组；真实环境会按数组顺序逐步执行。",
        f"task_id：{task.get('task_id')}",
        f"source_file：{task.get('source_file')}；page：{task.get('page')}",
        f"最多 {args.max_steps} 步。",
        "可用工具 JSON schema：",
        tools,
        "scope 约束：retrieve_evidence.scope 必须只取 current_page、nearby_pages、same_document、corpus 其中一个字符串；禁止输出 current_page|nearby_pages|same_document|corpus。",
        "evidence_id 约束：open_evidence.evidence_id 必须来自 local_evidence 或 retrieve_evidence 返回的 ev_xxx；禁止用 region_id 伪造 ev_r0、ev_r1 等证据 id。",
        "建议轨迹骨架：propose_regions -> select_evidence -> crop_region -> retrieve_evidence(current_page) -> retrieve_evidence(nearby_pages) -> retrieve_evidence(same_document) -> retrieve_evidence(corpus) -> open_evidence 若干次 -> write_claims_chunk 3 次左右 -> finish。",
        "输出格式要求：只输出 JSON 数组；数组中每个元素是一个工具调用 JSON 对象；不要输出 markdown；不要解释。",
        "禁止复制静态工具预览内容到输出；输出中只能出现 action 数组，不能出现 region_candidates/local_evidence/candidate_evidence 等预览字段。",
        "选择 crop_region 时：优先选择 type=figure_candidate；不要把 text_or_caption_candidate 当作图像裁剪；优先选择 local_caption_match_score 高、local_caption_match_rank 靠前且 caption_match_hint 与 local_evidence 作品名/作者/朝代一致的图像候选。",
        "重要约束：不要编造 region_id 或 evidence_id；如果字段证据不足，在 write_claims_chunk.abstains 中说明；每次 write_claims_chunk 写 3-5 个字段；通常用 3 次 write_claims_chunk 覆盖 caption_text 到 collection 的 12 个字段；不要用 write_claims_batch。",
    ]
    if args.include_static_tool_preview:
        preview = build_static_tool_preview(task, args)
        lines.extend(
            [
                "静态工具预览（用于单响应桥接；这些是工具执行后可见的候选摘要，不包含 is_target/gold_iou；请阅读但不要复制到输出）：",
                render_static_tool_preview_text(preview),
            ]
        )
    return "\n".join(lines)


def build_static_tool_preview(task: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    regions = []
    text_regions = [
        item
        for item in (task.get("region_candidates") or [])
        if str(item.get("type")) == "text_or_caption_candidate" and (item.get("caption_hint") or item.get("nearby_text"))
    ]
    local_caption_items = [
        {
            "evidence_id": item.get("evidence_id"),
            "text": item.get("display_snippet") or item.get("evidence_summary") or item.get("text") or "",
        }
        for item in (task.get("local_evidence") or [])[: args.max_local_evidence]
        if str(item.get("evidence_id", "")).startswith("local_caption_")
    ]
    for item in (task.get("region_candidates") or [])[: args.region_top_k]:
        region = {
            "region_id": item.get("region_id"),
            "bbox": item.get("bbox"),
            "source": item.get("source"),
            "type": item.get("type"),
            "score": item.get("score"),
            "hint": truncate(item.get("hint") or item.get("nearby_text") or "", args.max_snippet_chars),
        }
        if item.get("caption_evidence_id"):
            region["caption_evidence_id"] = item.get("caption_evidence_id")
        if item.get("caption_hint"):
            region["caption_hint"] = truncate(item.get("caption_hint"), args.max_snippet_chars)
        if str(item.get("type")) == "figure_candidate":
            local_match = best_local_caption_match(item, text_regions, local_caption_items, args.max_snippet_chars)
            if local_match:
                region["local_caption_match_score"] = round(local_match["score"], 4)
                region["local_caption_match_hint"] = local_match["text"]
                region["local_caption_match_bbox"] = local_match["bbox"]
                region["local_caption_match_evidence_id"] = local_match.get("evidence_id")
            nearby_caption = local_match if local_match and local_match["score"] >= 0.45 else nearest_caption_hint(
                item,
                text_regions,
                args.max_snippet_chars,
            )
            if nearby_caption:
                region["nearby_caption_hint"] = nearby_caption["text"]
                region["nearby_caption_bbox"] = nearby_caption["bbox"]
        regions.append(region)
    figure_regions = [item for item in regions if item.get("type") == "figure_candidate" and "local_caption_match_score" in item]
    figure_regions.sort(key=lambda item: float(item.get("local_caption_match_score") or 0.0), reverse=True)
    for rank, item in enumerate(figure_regions, start=1):
        item["local_caption_match_rank"] = rank
    local_evidence = []
    for item in (task.get("local_evidence") or [])[: args.max_local_evidence]:
        local_evidence.append(
            {
                "evidence_id": item.get("evidence_id"),
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
                "authority_level": item.get("authority_level"),
                "citation_level": item.get("citation_level"),
                "display_snippet": truncate(
                    item.get("display_snippet") or item.get("evidence_summary") or item.get("text") or "",
                    args.max_snippet_chars,
                ),
            }
        )
    return {
        "region_candidates": regions,
        "local_evidence": local_evidence,
        "candidate_evidence": build_candidate_evidence_preview(task, args),
        "allowed_retrieval_scopes": task.get("allowed_retrieval_scopes") or [
            "current_page",
            "nearby_pages",
            "same_document",
            "corpus",
        ],
    }


def render_static_tool_preview_text(preview: dict[str, Any]) -> str:
    lines = ["STATIC_TOOL_PREVIEW_BEGIN"]
    lines.append("REGION_CANDIDATES")
    for item in preview.get("region_candidates") or []:
        parts = [
            f"id={item.get('region_id')}",
            f"type={item.get('type')}",
            f"bbox={item.get('bbox')}",
            f"score={item.get('score')}",
        ]
        if item.get("local_caption_match_score") is not None:
            parts.append(f"local_caption_match_score={item.get('local_caption_match_score')}")
            parts.append(f"local_caption_match_rank={item.get('local_caption_match_rank')}")
        if item.get("local_caption_match_evidence_id"):
            parts.append(f"local_caption_match_evidence_id={item.get('local_caption_match_evidence_id')}")
        hint = item.get("local_caption_match_hint") or item.get("nearby_caption_hint") or item.get("caption_hint") or item.get("hint")
        if hint:
            parts.append(f"caption_match_hint={hint}")
        lines.append("- " + " | ".join(str(part) for part in parts))
    lines.append("LOCAL_EVIDENCE")
    for item in preview.get("local_evidence") or []:
        lines.append(
            "- "
            + " | ".join(
                [
                    f"id={item.get('evidence_id')}",
                    f"authority={item.get('authority_level')}",
                    f"citation={item.get('citation_level')}",
                    f"snippet={item.get('display_snippet')}",
                ]
            )
        )
    lines.append("CANDIDATE_EVIDENCE")
    for item in preview.get("candidate_evidence") or []:
        lines.append(
            "- "
            + " | ".join(
                [
                    f"id={item.get('evidence_id')}",
                    f"authority={item.get('authority_level')}",
                    f"citation={item.get('citation_level')}",
                    f"snippet={item.get('display_snippet')}",
                ]
            )
        )
    lines.append("ALLOWED_RETRIEVAL_SCOPES: " + ", ".join(str(item) for item in preview.get("allowed_retrieval_scopes") or []))
    lines.append("STATIC_TOOL_PREVIEW_END")
    return "\n".join(lines)


def nearest_caption_hint(
    figure_region: dict[str, Any],
    text_regions: list[dict[str, Any]],
    snippet_chars: int,
) -> dict[str, Any] | None:
    box = coerce_bbox(figure_region.get("bbox"))
    if box is None:
        return None
    fx1, fy1, fx2, fy2 = box
    fcx = (fx1 + fx2) / 2
    scored: list[tuple[float, dict[str, Any]]] = []
    for item in text_regions:
        text_box = coerce_bbox(item.get("bbox"))
        if text_box is None:
            continue
        tx1, ty1, tx2, ty2 = text_box
        tcx = (tx1 + tx2) / 2
        vertical_gap = min(abs(ty1 - fy2), abs(fy1 - ty2))
        if ty1 >= fy2:
            vertical_gap = ty1 - fy2
        elif ty2 <= fy1:
            vertical_gap = fy1 - ty2
        else:
            vertical_gap = 0
        if vertical_gap > 420:
            continue
        horizontal_overlap = max(0, min(fx2, tx2) - max(fx1, tx1))
        horizontal_gap = abs(fcx - tcx)
        score = vertical_gap + 0.25 * horizontal_gap - 0.5 * horizontal_overlap
        scored.append((score, item))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0])
    best = scored[0][1]
    text = best.get("caption_hint") or best.get("nearby_text") or ""
    return {"text": truncate(text, snippet_chars), "bbox": best.get("bbox")}


def best_local_caption_match(
    figure_region: dict[str, Any],
    text_regions: list[dict[str, Any]],
    local_caption_items: list[dict[str, Any]],
    snippet_chars: int,
) -> dict[str, Any] | None:
    if not local_caption_items:
        return None
    scored: list[tuple[float, dict[str, Any], dict[str, Any] | None]] = []
    local_by_id = {str(item.get("evidence_id")): item for item in local_caption_items if item.get("evidence_id")}
    for item in text_regions:
        text = item.get("caption_hint") or item.get("nearby_text") or ""
        if not text:
            continue
        caption_id = str(item.get("caption_evidence_id") or "")
        if caption_id and caption_id in local_by_id:
            local_score = 1.0
            matched_local = local_by_id[caption_id]
        else:
            local_score, matched_local = best_text_match(text, local_caption_items)
        if local_score < 0.08:
            continue
        geom_score = caption_geometry_score(figure_region, item)
        if geom_score <= 0.0 and local_score < 0.95:
            continue
        score = 0.55 * local_score + 0.45 * geom_score
        scored.append((score, item, matched_local))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    score, best, matched_local = scored[0]
    text = best.get("caption_hint") or best.get("nearby_text") or ""
    return {
        "score": score,
        "text": truncate(text, snippet_chars),
        "bbox": best.get("bbox"),
        "evidence_id": (matched_local or {}).get("evidence_id") or best.get("caption_evidence_id"),
    }


def best_text_match(text: str, local_caption_items: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    best_score = 0.0
    best_item: dict[str, Any] | None = None
    for item in local_caption_items:
        score = text_similarity(text, str(item.get("text") or ""))
        if score > best_score:
            best_score = score
            best_item = item
    return best_score, best_item


def text_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm in right_norm or right_norm in left_norm:
        return 1.0
    seq_score = SequenceMatcher(None, left_norm[:320], right_norm[:320]).ratio()
    left_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", left_norm))
    right_tokens = set(re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9]+", right_norm))
    token_score = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
    return max(seq_score, token_score)


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。、“”‘’：:；;（）()【】\\[\\]《》<>|/\\\\]", "", text)
    return text


def caption_geometry_score(figure_region: dict[str, Any], text_region: dict[str, Any]) -> float:
    fig = coerce_bbox(figure_region.get("bbox"))
    cap = coerce_bbox(text_region.get("bbox"))
    if fig is None or cap is None:
        return 0.0
    fx1, fy1, fx2, fy2 = fig
    tx1, ty1, tx2, ty2 = cap
    f_width = max(1.0, fx2 - fx1)
    t_width = max(1.0, tx2 - tx1)
    overlap = max(0.0, min(fx2, tx2) - max(fx1, tx1))
    overlap_ratio = overlap / max(1.0, min(f_width, t_width))
    if ty1 >= fy2:
        vertical_gap = ty1 - fy2
        direction_bonus = 0.12
    elif ty2 <= fy1:
        vertical_gap = fy1 - ty2
        direction_bonus = 0.04
    else:
        vertical_gap = 0
        direction_bonus = 0.06
    if vertical_gap > 520 and overlap_ratio < 0.35:
        return 0.0
    gap_score = max(0.0, 1.0 - vertical_gap / 520.0)
    return min(1.0, 0.58 * overlap_ratio + 0.30 * gap_score + direction_bonus)


def build_candidate_evidence_preview(task: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    evidence_items: dict[str, dict[str, Any]] = getattr(args, "evidence_items", {}) or {}
    ordered_ids: list[str] = []
    for claim in task.get("gold", {}).get("claims", []) or []:
        for evidence_id in (claim.get("evidence_ids") or []) + (claim.get("candidate_evidence_ids") or []):
            evidence_id = str(evidence_id)
            if evidence_id.startswith("local_caption_"):
                continue
            if evidence_id and evidence_id not in ordered_ids:
                ordered_ids.append(evidence_id)
    preview: list[dict[str, Any]] = []
    for evidence_id in ordered_ids[: max(0, int(args.max_candidate_evidence))]:
        item = evidence_items.get(evidence_id, {})
        preview.append(
            {
                "evidence_id": evidence_id,
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
                "authority_level": item.get("authority_level"),
                "citation_level": item.get("citation_level"),
                "display_snippet": truncate(
                    item.get("display_snippet") or item.get("evidence_summary") or item.get("text") or "",
                    args.max_snippet_chars,
                ),
            }
        )
    return preview


def load_evidence_items(index_dir: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(index_dir) / "corpus_chunks.jsonl"
    if not path.exists():
        return {}
    return {str(item.get("evidence_id")): item for item in read_jsonl(path) if item.get("evidence_id")}


def coerce_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def truncate(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[: max(0, limit - 1)] + "…"


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
