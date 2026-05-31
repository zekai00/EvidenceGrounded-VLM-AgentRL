#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


FIELDS = ["caption_text", "title", "artist", "dynasty", "visual_elements", "technique", "composition"]
TEXT_FIELDS = {"title", "artist", "dynasty", "technique", "composition"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate audited AgentBench v0.1 tasks to v0.2 retrieval-scope tool schema.")
    parser.add_argument(
        "--v0-1-dir",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_1_vlm_audited_flash_full_20260530_1641",
        help="Audited v0.1 dataset directory.",
    )
    parser.add_argument(
        "--evidence-index-dir",
        default="",
        help="v0.2 evidence index directory. Defaults to latest evidence_index_v0_2_* under output root.",
    )
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--version", default="agentbench_v0_2_retrieval_scope")
    parser.add_argument("--top-k-current-page", type=int, default=5)
    parser.add_argument("--top-k-nearby-pages", type=int, default=6)
    parser.add_argument("--top-k-same-document", type=int, default=8)
    parser.add_argument("--top-k-corpus", type=int, default=8)
    parser.add_argument("--max-open-evidence", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = Path(args.output_root)
    evidence_index_dir = Path(args.evidence_index_dir) if args.evidence_index_dir else latest_dir(output_root, "evidence_index_v0_2_")
    output_dir = output_root / f"{args.version}_{now}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ["episodes", "sft", "review"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    index = load_evidence_index(evidence_index_dir)
    source_tasks = read_jsonl(Path(args.v0_1_dir) / "tasks_all.jsonl")
    tasks = [migrate_task(row, index, i) for i, row in enumerate(source_tasks)]

    episodes: list[dict[str, Any]] = []
    sft_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    retrieval_stats = RetrievalStats()
    for task in tasks:
        episode = build_oracle_episode_v0_2(task, index, args)
        episodes.append(episode)
        samples = episode_to_sft_samples_v0_2(episode, task, index, args, retrieval_stats)
        sft_rows_by_split[task["split"]].extend(samples)

    claim_rows = [row for task in tasks for row in task_to_claim_rows(task)]
    evidence_rows = [row for task in tasks for row in task_to_evidence_rows(task)]
    quality = build_quality_report(tasks, episodes, claim_rows, evidence_rows, sft_rows_by_split, retrieval_stats, index)

    write_jsonl(output_dir / "tasks_all.jsonl", tasks)
    for split in ["train", "val", "test"]:
        split_tasks = [task for task in tasks if task["split"] == split]
        write_jsonl(output_dir / f"{split}_tasks.jsonl", split_tasks)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows_by_split.get(split, []))
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)
    write_jsonl(output_dir / "claim_gold.jsonl", claim_rows)
    write_jsonl(output_dir / "evidence_links.jsonl", evidence_rows)
    review_path = write_review_html(output_dir / "review" / "review.html", tasks[:200])

    summary = build_summary(args, output_dir, evidence_index_dir, source_tasks, tasks, quality, review_path)
    (output_dir / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(build_manifest(args, output_dir, evidence_index_dir, summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def latest_dir(root: Path, prefix: str) -> Path:
    matches = sorted([path for path in root.glob(f"{prefix}*") if path.is_dir()])
    if not matches:
        raise FileNotFoundError(f"No directory matching {prefix}* under {root}")
    return matches[-1]


def load_evidence_index(index_dir: Path) -> dict[str, Any]:
    chunks = read_jsonl(index_dir / "corpus_chunks.jsonl")
    by_id = {}
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        prepared = dict(chunk)
        prepared["_norm_source"] = normalize_source(str(chunk.get("source_file") or chunk.get("source_stem") or ""))
        prepared["_norm_text"] = normalize_text(build_search_text(chunk))
        prepared["_tokens"] = set(tokenize(build_search_text(chunk)))
        by_id[prepared["evidence_id"]] = prepared
        by_source[prepared["_norm_source"]].append(prepared)
    legacy_map_path = index_dir / "legacy_chunk_map.json"
    legacy_map = json.loads(legacy_map_path.read_text(encoding="utf-8")) if legacy_map_path.exists() else {}
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return {
        "dir": index_dir,
        "chunks": chunks,
        "prepared_chunks": list(by_id.values()),
        "by_id": by_id,
        "by_source": by_source,
        "legacy_map": legacy_map,
        "manifest": manifest,
    }


def migrate_task(row: dict[str, Any], index: dict[str, Any], task_index: int) -> dict[str, Any]:
    task = copy.deepcopy(row)
    old_task_id = str(task.get("task_id") or f"v0_1_{task_index:06d}")
    task["source_task_id"] = old_task_id
    task["task_id"] = f"egva_v0_2_scope_{task_index:06d}"
    task["task_type"] = "evidence_grounded_figure_understanding"
    task["tool_schema_version"] = "v0.2_retrieval_scope"
    task["goal"] = (
        "Use page image, cropped figure, scoped evidence retrieval, and opened evidence snippets to write "
        "evidence-grounded claims for the highlighted Chinese landscape figure."
    )
    task["allowed_retrieval_scopes"] = ["current_page", "nearby_pages", "same_document", "corpus"]
    task["available_tools"] = [
        "crop_image",
        "retrieve_evidence",
        "open_evidence",
        "write_claim",
        "abstain_claim",
        "finish",
    ]
    task["evidence_index"] = {
        "path": str(index["dir"]),
        "version": "evidence_index_v0_2",
        "citation_level": "page_range_chunk_or_legacy_chunk",
    }
    task.setdefault("candidate_meta", {})
    task["candidate_meta"]["converted_from"] = "agentbench_v0_1_vlm_audited"
    task["candidate_meta"]["source_task_id"] = old_task_id

    gold = task.setdefault("gold", {})
    legacy_evidence_ids = unique_ids(gold.get("evidence_chunk_ids") or [])
    legacy_candidate_ids = unique_ids(gold.get("candidate_evidence_ids") or [])
    mapped_evidence_ids, missing_evidence = map_legacy_ids(legacy_evidence_ids, index["legacy_map"])
    mapped_candidate_ids, missing_candidates = map_legacy_ids(legacy_candidate_ids, index["legacy_map"])
    gold["legacy_evidence_chunk_ids"] = legacy_evidence_ids
    gold["legacy_candidate_evidence_ids"] = legacy_candidate_ids
    gold["evidence_ids"] = mapped_evidence_ids
    gold["candidate_evidence_ids"] = mapped_candidate_ids
    gold["unmapped_legacy_evidence_ids"] = missing_evidence
    gold["unmapped_legacy_candidate_evidence_ids"] = missing_candidates
    gold["evidence_query"] = build_task_query(task)
    gold["citation_level"] = "page_range_chunk_or_legacy_chunk"
    gold["tool_schema_version"] = "v0.2_retrieval_scope"
    gold["retrieval_scopes"] = ["current_page", "nearby_pages", "same_document", "corpus"]
    gold["label_source"] = "v0_1_vlm_audited_migrated_to_v0_2_retrieval_scope"

    for claim in gold.get("claims") or []:
        legacy_ids = unique_ids(claim.get("evidence_ids") or [])
        legacy_candidates = unique_ids(claim.get("candidate_evidence_ids") or [])
        claim["legacy_evidence_ids"] = legacy_ids
        claim["legacy_candidate_evidence_ids"] = legacy_candidates
        claim["evidence_ids"], missing_claim_ids = map_legacy_ids(legacy_ids, index["legacy_map"])
        claim["candidate_evidence_ids"], missing_claim_candidates = map_legacy_ids(legacy_candidates, index["legacy_map"])
        if missing_claim_ids or missing_claim_candidates:
            claim["unmapped_legacy_ids"] = unique_ids(missing_claim_ids + missing_claim_candidates)
        if claim.get("evidence_status") == "vlm_selected_chunk_support":
            claim["evidence_status"] = "vlm_selected_evidence_support_migrated"
    task["evidence_links"] = [migrate_evidence_link(link, index["legacy_map"]) for link in task.get("evidence_links") or []]
    return task


def migrate_evidence_link(link: dict[str, Any], legacy_map: dict[str, str]) -> dict[str, Any]:
    item = copy.deepcopy(link)
    for key in ["gold_evidence_ids", "candidate_evidence_ids"]:
        legacy_ids = unique_ids(item.get(key) or [])
        mapped, missing = map_legacy_ids(legacy_ids, legacy_map)
        item[f"legacy_{key}"] = legacy_ids
        item[key] = mapped
        if missing:
            item[f"unmapped_legacy_{key}"] = missing
    labels = item.get("support_labels") or {}
    if labels:
        item["legacy_support_labels"] = labels
        item["support_labels"] = {legacy_map.get(old_id, old_id): value for old_id, value in labels.items() if legacy_map.get(old_id, old_id)}
    return item


def build_oracle_episode_v0_2(task: dict[str, Any], index: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    gold = task.get("gold", {})
    anchor = {
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "bbox": gold.get("image_bbox"),
    }
    actions: list[dict[str, Any]] = [{"action": "crop_image", "bbox": gold.get("image_bbox")}]
    caption_query = " ".join(piece for piece in [task.get("source_stem", ""), gold.get("caption_text", "")] if piece).strip()
    if caption_query:
        actions.append(
            {
                "action": "retrieve_evidence",
                "query": caption_query[:500],
                "scope": "current_page",
                "anchor": {**anchor, "bbox": gold.get("caption_bbox") or gold.get("image_bbox")},
                "top_k": args.top_k_current_page,
            }
        )
    actions.append(
        {
            "action": "retrieve_evidence",
            "query": gold.get("evidence_query") or build_task_query(task),
            "scope": "nearby_pages",
            "anchor": anchor,
            "top_k": args.top_k_nearby_pages,
        }
    )
    actions.append(
        {
            "action": "retrieve_evidence",
            "query": gold.get("evidence_query") or build_task_query(task),
            "scope": "same_document",
            "anchor": anchor,
            "top_k": args.top_k_same_document,
        }
    )
    evidence_ids = unique_ids(gold.get("evidence_ids") or [])
    if should_add_corpus_retrieval(task, evidence_ids, index):
        actions.append(
            {
                "action": "retrieve_evidence",
                "query": gold.get("evidence_query") or build_task_query(task),
                "scope": "corpus",
                "anchor": anchor,
                "top_k": args.top_k_corpus,
            }
        )
    for evidence_id in evidence_ids[: args.max_open_evidence]:
        actions.append({"action": "open_evidence", "evidence_id": evidence_id})
    for claim in gold.get("claims") or []:
        if claim.get("abstain"):
            actions.append({"action": "abstain_claim", "field": claim.get("field"), "reason": claim.get("reason", "")})
        else:
            actions.append(
                {
                    "action": "write_claim",
                    "field": claim.get("field"),
                    "value": claim.get("value"),
                    "evidence_ids": claim.get("evidence_ids") or [],
                    "visual_bbox": claim.get("visual_bbox"),
                    "confidence": normalize_confidence(claim),
                }
            )
    actions.append({"action": "finish", "status": "done"})
    return {
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "split": task["split"],
        "tool_schema_version": "v0.2_retrieval_scope",
        "actions": actions,
    }


def should_add_corpus_retrieval(task: dict[str, Any], evidence_ids: list[str], index: dict[str, Any]) -> bool:
    if not evidence_ids:
        return any((not claim.get("abstain")) and claim.get("field") in TEXT_FIELDS for claim in task.get("gold", {}).get("claims") or [])
    task_source = normalize_source(str(task.get("source_file") or task.get("source_stem") or ""))
    for evidence_id in evidence_ids:
        chunk = index["by_id"].get(evidence_id)
        if not chunk:
            return True
        if chunk.get("_norm_source") != task_source:
            return True
        if str(chunk.get("source_quality") or "") == "legacy_milvus":
            return True
    return False


def episode_to_sft_samples_v0_2(
    episode: dict[str, Any],
    task: dict[str, Any],
    index: dict[str, Any],
    args: argparse.Namespace,
    retrieval_stats: "RetrievalStats",
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    draft_claims: list[dict[str, Any]] = []
    rows = []
    for step, action in enumerate(episode["actions"]):
        prompt = build_prompt_v0_2(task, step, history, tool_results, draft_claims)
        images = [task["page_image"]]
        if any(item.get("tool") == "crop_image" for item in tool_results):
            images.append(task["artwork_image"])
        rows.append(
            {
                "task_id": task["task_id"],
                "source_task_id": task.get("source_task_id"),
                "split": task["split"],
                "step": step,
                "tool_schema_version": "v0.2_retrieval_scope",
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))},
                ],
                "action": action,
                "history": copy.deepcopy(history),
                "tool_results": copy.deepcopy(tool_results),
                "draft_claims": copy.deepcopy(draft_claims),
                "images": images,
                "prompt_text": prompt,
                "label_source": "v0_1_vlm_audited_migrated_to_v0_2",
            }
        )
        update_sft_state_v0_2(task, action, history, tool_results, draft_claims, index, retrieval_stats)
    return rows


def build_prompt_v0_2(
    task: dict[str, Any],
    step: int,
    history: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    draft_claims: list[dict[str, Any]],
) -> str:
    return (
        "你是 evidence-grounded figure understanding 的 VLM tool-call agent。"
        "你的目标是根据 PDF 页面图像、局部裁剪图和可追溯证据，写出关于红框山水画图像的 claim。\n"
        f"任务：{task.get('goal')}\n"
        f"task_id：{task.get('task_id')}；source_file：{task.get('source_file')}；page：{task.get('page')}；step：{step}\n"
        "可用工具：\n"
        "1. crop_image(bbox)：裁剪页面中的目标图像区域。\n"
        "2. retrieve_evidence(query, scope, anchor, top_k)：按范围检索证据。scope 只能是 current_page、nearby_pages、same_document、corpus。"
        "current_page 表示只查当前页；nearby_pages 表示查当前页前后一页；same_document 表示查同一篇文献；corpus 表示查全语料。\n"
        "3. open_evidence(evidence_id)：打开已经由 retrieve_evidence 返回的证据片段。\n"
        "4. write_claim(field, value, evidence_ids, visual_bbox, confidence)：写入有证据支撑的字段。\n"
        "5. abstain_claim(field, reason)：证据不足时放弃该字段。\n"
        "6. finish：完成任务。\n"
        "约束：不要凭空编造作品名、画家、朝代、技法；没有可靠证据就 abstain。"
        "不要请求读取任意 PDF 页，只能通过 retrieve_evidence 的 scope 扩大检索范围。\n"
        f"历史动作：{json.dumps(history[-8:], ensure_ascii=False)}\n"
        f"工具返回：{json.dumps(tool_results[-5:], ensure_ascii=False)}\n"
        f"当前 claims：{json.dumps(draft_claims, ensure_ascii=False)}\n"
        "只输出一个 JSON 对象。"
    )


def update_sft_state_v0_2(
    task: dict[str, Any],
    action: dict[str, Any],
    history: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    draft_claims: list[dict[str, Any]],
    index: dict[str, Any],
    retrieval_stats: "RetrievalStats",
) -> None:
    name = action.get("action")
    if name == "crop_image":
        tool_results.append({"tool": "crop_image", "bbox": action.get("bbox"), "crop_path": task.get("artwork_image")})
    elif name == "retrieve_evidence":
        results = retrieve_evidence(task, action, index)
        retrieval_stats.observe(task, action, results)
        tool_results.append(
            {
                "tool": "retrieve_evidence",
                "query": action.get("query"),
                "scope": action.get("scope"),
                "anchor": action.get("anchor"),
                "results": [format_retrieval_result(item) for item in results],
            }
        )
    elif name == "open_evidence":
        chunk = index["by_id"].get(str(action.get("evidence_id")))
        if chunk:
            tool_results.append(
                {
                    "tool": "open_evidence",
                    "evidence_id": action.get("evidence_id"),
                    "source_file": chunk.get("source_file"),
                    "title": chunk.get("title"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "authority_level": chunk.get("authority_level"),
                    "text": str(chunk.get("text") or "")[:1000],
                }
            )
        else:
            tool_results.append({"tool": "open_evidence", "evidence_id": action.get("evidence_id"), "error": "evidence_id_not_found"})
    elif name == "write_claim":
        draft_claims.append({key: action.get(key) for key in ["field", "value", "evidence_ids", "visual_bbox", "confidence"] if key in action})
    elif name == "abstain_claim":
        draft_claims.append({"field": action.get("field"), "abstain": True, "reason": action.get("reason", "")})
    history.append(action)


def retrieve_evidence(task: dict[str, Any], action: dict[str, Any], index: dict[str, Any]) -> list[dict[str, Any]]:
    query = str(action.get("query") or "")
    scope = str(action.get("scope") or "corpus")
    top_k = int(action.get("top_k") or 8)
    source_norm = normalize_source(str(task.get("source_file") or task.get("source_stem") or ""))
    page = safe_int(task.get("page"))
    candidates = candidate_pool_for_scope(scope, source_norm, page, index)
    gold_ids = set(task.get("gold", {}).get("evidence_ids") or [])
    candidate_ids = set(task.get("gold", {}).get("candidate_evidence_ids") or [])
    if scope in {"same_document", "corpus"}:
        for evidence_id in list(candidate_ids | gold_ids):
            chunk = index["by_id"].get(evidence_id)
            if chunk and chunk not in candidates:
                candidates.append(chunk)
    scored = []
    q_tokens = set(tokenize(query))
    important = important_terms(query)
    for chunk in candidates:
        text_norm = chunk.get("_norm_text") or normalize_text(build_search_text(chunk))
        token_overlap = len(q_tokens & set(chunk.get("_tokens") or []))
        phrase_bonus = sum(1 for term in important if term and term in text_norm)
        score = float(token_overlap) + 3.0 * phrase_bonus
        if chunk.get("evidence_id") in candidate_ids:
            score += 4.0
        if chunk.get("evidence_id") in gold_ids:
            score += 8.0
        if chunk.get("_norm_source") == source_norm:
            score += 2.0
        if page and page_in_chunk(page, chunk):
            score += 2.5
        authority_weight = chunk.get("authority_weight")
        if isinstance(authority_weight, (int, float)) and not math.isnan(float(authority_weight)):
            score += min(2.0, float(authority_weight))
        if score <= 0:
            continue
        copied = dict(chunk)
        copied["_score"] = round(score, 4)
        scored.append(copied)
    scored.sort(key=lambda item: (-float(item.get("_score") or 0), item.get("source_file") or "", item.get("page_start") or 0))
    return scored[:top_k]


def candidate_pool_for_scope(scope: str, source_norm: str, page: int | None, index: dict[str, Any]) -> list[dict[str, Any]]:
    if scope == "corpus":
        return list(index["prepared_chunks"])
    source_chunks = list(index["by_source"].get(source_norm, []))
    if scope == "same_document":
        return source_chunks
    if scope == "nearby_pages" and page:
        return [chunk for chunk in source_chunks if chunk_page_distance(page, chunk) <= 1]
    if scope == "current_page" and page:
        return [chunk for chunk in source_chunks if page_in_chunk(page, chunk)]
    return source_chunks


def format_retrieval_result(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "source_file": item.get("source_file"),
        "title": item.get("title"),
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "authority_level": item.get("authority_level"),
        "citation_level": item.get("citation_level"),
        "score": item.get("_score"),
        "snippet": str(item.get("text") or "")[:260],
    }


class RetrievalStats:
    def __init__(self) -> None:
        self.scope_calls = Counter()
        self.scope_nonempty = Counter()
        self.scope_gold_hit = Counter()
        self.calls = 0

    def observe(self, task: dict[str, Any], action: dict[str, Any], results: list[dict[str, Any]]) -> None:
        scope = str(action.get("scope") or "unknown")
        self.calls += 1
        self.scope_calls[scope] += 1
        if results:
            self.scope_nonempty[scope] += 1
        gold_ids = set(task.get("gold", {}).get("evidence_ids") or [])
        result_ids = {str(item.get("evidence_id")) for item in results}
        if gold_ids & result_ids:
            self.scope_gold_hit[scope] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieve_calls": self.calls,
            "scope_calls": dict(self.scope_calls),
            "scope_nonempty": dict(self.scope_nonempty),
            "scope_gold_hit": dict(self.scope_gold_hit),
            "scope_nonempty_rate": {key: self.scope_nonempty[key] / max(1, value) for key, value in self.scope_calls.items()},
            "scope_gold_hit_rate": {key: self.scope_gold_hit[key] / max(1, value) for key, value in self.scope_calls.items()},
        }


def build_quality_report(
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    sft_rows_by_split: dict[str, list[dict[str, Any]]],
    retrieval_stats: RetrievalStats,
    index: dict[str, Any],
) -> dict[str, Any]:
    non_abstain = [row for row in claim_rows if not row.get("abstain")]
    with_evidence = [row for row in non_abstain if row.get("evidence_ids")]
    unmapped_gold = [eid for task in tasks for eid in task.get("gold", {}).get("unmapped_legacy_evidence_ids", [])]
    all_evidence_ids = [eid for row in claim_rows for eid in row.get("evidence_ids", [])]
    found_evidence_ids = [eid for eid in all_evidence_ids if eid in index["by_id"]]
    action_counter = Counter(action.get("action") for episode in episodes for action in episode.get("actions", []))
    split_counter = Counter(task["split"] for task in tasks)
    source_counter = Counter(task["source_file"] for task in tasks)
    field_counter = Counter(row.get("field") for row in claim_rows)
    support_counter = Counter(row.get("evidence_status", "none") for row in non_abstain)
    return {
        "tasks": len(tasks),
        "splits": dict(split_counter),
        "unique_sources": len(source_counter),
        "top_sources": dict(source_counter.most_common(12)),
        "claims": len(claim_rows),
        "claim_fields": dict(field_counter),
        "non_abstain_claims": len(non_abstain),
        "claims_with_evidence": len(with_evidence),
        "claim_evidence_coverage": len(with_evidence) / max(1, len(non_abstain)),
        "support_status": dict(support_counter),
        "evidence_link_rows": len(evidence_rows),
        "gold_evidence_id_mentions": len(all_evidence_ids),
        "mapped_evidence_ids_found_in_index": len(found_evidence_ids),
        "mapped_evidence_id_presence": len(found_evidence_ids) / max(1, len(all_evidence_ids)),
        "unmapped_legacy_gold_evidence_ids": len(unmapped_gold),
        "action_counts": dict(action_counter),
        "episodes": len(episodes),
        "avg_actions_per_episode": sum(len(ep.get("actions", [])) for ep in episodes) / max(1, len(episodes)),
        "sft_rows": {split: len(rows) for split, rows in sft_rows_by_split.items()},
        "retrieval": retrieval_stats.to_dict(),
        "limitations": [
            "v0.2 currently migrates VLM-audited v0.1 gold instead of regenerating every image candidate from the new corpus.",
            "PDF text-layer index is available, but low-text pages still need OCR/VLM fallback before page-level evidence is complete.",
            "Legacy Milvus chunks are preserved for backward compatibility; they remain chunk-level citations when page_start/page_end are null.",
            "The retrieval traces are supervised oracle traces for SFT, not yet an online environment reward trace.",
        ],
    }


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    evidence_index_dir: Path,
    source_tasks: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    quality: dict[str, Any],
    review_path: Path,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "dataset_name": "EvidenceGrounded-AgentBench",
        "version": args.version,
        "output_dir": str(output_dir),
        "source_dataset": args.v0_1_dir,
        "source_tasks": len(source_tasks),
        "final_tasks": len(tasks),
        "evidence_index": str(evidence_index_dir),
        "tool_schema_version": "v0.2_retrieval_scope",
        "outputs": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
            "sft": str(output_dir / "sft"),
            "episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
            "claim_gold": str(output_dir / "claim_gold.jsonl"),
            "evidence_links": str(output_dir / "evidence_links.jsonl"),
            "quality_report": str(output_dir / "quality_report.json"),
            "review_html": str(review_path),
        },
        "quality": quality,
    }


def build_manifest(args: argparse.Namespace, output_dir: Path, evidence_index_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "build_time": summary["created_at"],
        "builder": "scripts/migrate_agentbench_v0_1_to_v0_2.py",
        "args": vars(args),
        "output_dir": str(output_dir),
        "evidence_index": str(evidence_index_dir),
        "summary": summary,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    quality = summary["quality"]
    lines = [
        "# EvidenceGrounded-AgentBench v0.2-retrieval-scope 构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 来源数据：`{summary['source_dataset']}`",
        f"- evidence index：`{summary['evidence_index']}`",
        f"- tool schema：`{summary['tool_schema_version']}`",
        "",
        "## 数据规模",
        "",
        f"- source tasks：{summary['source_tasks']}",
        f"- final tasks：{summary['final_tasks']}",
        f"- splits：`{quality['splits']}`",
        f"- unique sources：{quality['unique_sources']}",
        f"- episodes：{quality['episodes']}",
        f"- avg actions / episode：{quality['avg_actions_per_episode']:.2f}",
        f"- SFT rows：`{quality['sft_rows']}`",
        "",
        "## Claim 与证据质量",
        "",
        f"- claims：{quality['claims']}",
        f"- non-abstain claims：{quality['non_abstain_claims']}",
        f"- claims_with_evidence：{quality['claims_with_evidence']}",
        f"- claim_evidence_coverage：{quality['claim_evidence_coverage']:.4f}",
        f"- gold_evidence_id_mentions：{quality['gold_evidence_id_mentions']}",
        f"- mapped_evidence_id_presence：{quality['mapped_evidence_id_presence']:.4f}",
        f"- unmapped_legacy_gold_evidence_ids：{quality['unmapped_legacy_gold_evidence_ids']}",
        f"- support_status：`{quality['support_status']}`",
        "",
        "## 轨迹动作",
        "",
        f"- action_counts：`{quality['action_counts']}`",
        f"- retrieval：`{quality['retrieval']}`",
        "",
        "## 输出文件",
        "",
    ]
    for name, value in summary["outputs"].items():
        lines.append(f"- {name}：`{value}`")
    lines.extend(["", "## 限制", ""])
    lines.extend(f"- {item}" for item in quality["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_html(path: Path, tasks: list[dict[str, Any]]) -> Path:
    parts = [
        "<html><head><meta charset='utf-8'><title>EvidenceGrounded AgentBench v0.2 Review</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .task{border:1px solid #ccc;padding:16px;margin:16px 0;} img{max-width:560px;border:1px solid #ddd;} code{white-space:pre-wrap;display:block;background:#f7f7f7;padding:8px;}</style>",
        "</head><body><h1>EvidenceGrounded AgentBench v0.2 Review</h1>",
    ]
    for task in tasks:
        parts.append("<div class='task'>")
        parts.append(f"<h2>{html_escape(task['task_id'])} [{html_escape(task['split'])}]</h2>")
        parts.append(f"<p>{html_escape(str(task.get('source_file')))} page {html_escape(str(task.get('page')))}</p>")
        parts.append(f"<img src='file://{html_escape(str(task.get('overlay_image')))}' />")
        preview = {
            "source_task_id": task.get("source_task_id"),
            "tool_schema_version": task.get("tool_schema_version"),
            "gold": task.get("gold"),
        }
        parts.append("<code>" + html_escape(json.dumps(preview, ensure_ascii=False, indent=2)[:5000]) + "</code>")
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def task_to_claim_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for claim in task.get("gold", {}).get("claims", []):
        rows.append(
            {
                "task_id": task["task_id"],
                "source_task_id": task.get("source_task_id"),
                "split": task["split"],
                "source_file": task["source_file"],
                "page": task["page"],
                **claim,
            }
        )
    return rows


def task_to_evidence_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for link in task.get("evidence_links") or []:
        rows.append({"task_id": task["task_id"], "source_task_id": task.get("source_task_id"), "split": task["split"], "source_file": task["source_file"], **link})
    return rows


def build_task_query(task: dict[str, Any]) -> str:
    gold = task.get("gold", {})
    pieces = [task.get("source_stem", ""), gold.get("caption_text", ""), gold.get("title", ""), gold.get("artist", ""), gold.get("dynasty", "")]
    for key in ["visual_elements", "technique", "composition"]:
        value = gold.get(key)
        if isinstance(value, list):
            pieces.extend(str(item) for item in value)
        elif value:
            pieces.append(str(value))
    return " ".join(item for item in unique_strings(pieces) if item).strip()[:700]


def map_legacy_ids(ids: list[str], legacy_map: dict[str, str]) -> tuple[list[str], list[str]]:
    mapped = []
    missing = []
    for item in ids:
        if not item:
            continue
        if str(item).startswith("ev_"):
            mapped.append(str(item))
        elif item in legacy_map:
            mapped.append(legacy_map[item])
        else:
            missing.append(str(item))
    return unique_ids(mapped), unique_ids(missing)


def unique_ids(values: list[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def unique_strings(values: list[Any]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        text = " ".join(str(value).split()).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def normalize_confidence(claim: dict[str, Any]) -> float:
    value = claim.get("confidence")
    if isinstance(value, (int, float)):
        return round(max(0.05, min(0.99, float(value))), 3)
    return 0.8 if claim.get("evidence_ids") else 0.45


def page_in_chunk(page: int, chunk: dict[str, Any]) -> bool:
    start = safe_int(chunk.get("page_start"))
    end = safe_int(chunk.get("page_end")) or start
    if not start:
        return False
    return start <= page <= end


def chunk_page_distance(page: int, chunk: dict[str, Any]) -> int:
    start = safe_int(chunk.get("page_start"))
    end = safe_int(chunk.get("page_end")) or start
    if not start:
        return 999999
    if start <= page <= end:
        return 0
    return min(abs(page - start), abs(page - end))


def safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def build_search_text(chunk: dict[str, Any]) -> str:
    return "\n".join(
        str(chunk.get(key) or "")
        for key in ["title", "author", "category", "source_file", "text", "source_type", "authority_level"]
    )


def normalize_source(value: str) -> str:
    name = Path(str(value)).name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = re.sub(r"^[A-Z]\d+_", "", name)
    name = re.sub(r"^\d+_", "", name)
    return normalize_text(name)


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value).lower())


def tokenize(value: str) -> list[str]:
    norm = normalize_text(value)
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", norm)
    tokens = []
    for word in words:
        tokens.append(word)
        if re.fullmatch(r"[\u4e00-\u9fff]+", word) and len(word) > 2:
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


def important_terms(value: str) -> list[str]:
    terms = re.findall(r"《([^》]{1,32})》|([\u4e00-\u9fff]{2,8})", value)
    flat = []
    for a, b in terms:
        item = a or b
        if item and item not in {"中国", "山水", "山水画", "作品", "图像", "论文", "研究"}:
            flat.append(normalize_text(item))
    return unique_ids(flat)[:24]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    raise SystemExit(main())
