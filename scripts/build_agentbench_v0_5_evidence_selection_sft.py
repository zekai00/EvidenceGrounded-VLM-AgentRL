#!/usr/bin/env python3
"""Build v0.5 SFT data with explicit select_evidence actions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig  # noqa: E402
from evidence_agent_env.tools.region_proposal import propose_regions  # noqa: E402
from build_agentbench_v0_4_1_claim_schema import make_sft_row  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_4_2_batch_claims_caption_linegroup_claimfix_sft_20260601_1731"
)
DEFAULT_TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--region-top-k", type=int, default=DEFAULT_TOP_K)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)

    tasks = read_jsonl(source_dir / "tasks_all.jsonl")
    source_rows = load_sft_by_task(source_dir)
    source_episodes = load_episodes(source_dir)

    new_tasks: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    new_episodes: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        new_task = copy.deepcopy(task)
        select_action = build_select_action(new_task, args.region_top_k)
        new_task["dataset_version"] = "v0.5_evidence_selection"
        new_task["tool_schema_version"] = "v0.5_evidence_selection"
        new_task["evidence_selection"] = evidence_selection_metadata(new_task, args.region_top_k, select_action)
        rows, actions = rebuild_rows(
            new_task,
            source_rows[task_id],
            select_action,
            region_top_k=args.region_top_k,
        )
        new_tasks.append(new_task)
        new_rows.extend(rows)
        episode_actions = rebuild_episode_actions(source_episodes[task_id].get("actions") or [], select_action, args.region_top_k)
        new_episodes.append(
            {
                "task_id": new_task["task_id"],
                "source_task_id": new_task.get("source_task_id"),
                "split": new_task.get("split"),
                "variant": new_task.get("candidate_augmentation", {}).get("variant"),
                "actions": episode_actions,
            }
        )

    write_outputs(output_dir, new_tasks, new_episodes, new_rows)
    quality = summarize(new_tasks, new_rows, new_episodes, source_dir, args.region_top_k)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.5_evidence_selection",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "region_top_k": args.region_top_k,
        "quality": quality,
        "files": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "oracle_episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
            "sft_train": str(output_dir / "sft" / "train.jsonl"),
            "sft_val": str(output_dir / "sft" / "val.jsonl"),
            "sft_test": str(output_dir / "sft" / "test.jsonl"),
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


def load_episodes(source_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in read_jsonl(source_dir / "episodes" / "oracle_episodes.jsonl")}


def build_select_action(task: dict[str, Any], region_top_k: int) -> dict[str, Any] | None:
    local_ids = {str(item.get("evidence_id")) for item in task.get("local_evidence") or [] if item.get("evidence_id")}
    region_caption_ids = {
        str(item.get("caption_evidence_id"))
        for item in (task.get("region_candidates") or [])[:region_top_k]
        if item.get("caption_evidence_id")
    }
    gold_ids: list[str] = []
    for claim in task.get("gold", {}).get("claims", []):
        if claim.get("abstain"):
            continue
        for evidence_id in claim.get("evidence_ids") or []:
            evidence_id = str(evidence_id)
            if evidence_id in local_ids and evidence_id in region_caption_ids and evidence_id not in gold_ids:
                gold_ids.append(evidence_id)
    if not gold_ids:
        return None
    return {"action": "select_evidence", "evidence_ids": gold_ids}


def evidence_selection_metadata(
    task: dict[str, Any],
    region_top_k: int,
    select_action: dict[str, Any] | None,
) -> dict[str, Any]:
    local_ids = [str(item.get("evidence_id")) for item in task.get("local_evidence") or [] if item.get("evidence_id")]
    region_caption_ids = [
        str(item.get("caption_evidence_id"))
        for item in (task.get("region_candidates") or [])[:region_top_k]
        if item.get("caption_evidence_id")
    ]
    candidate_ids = sorted(set(local_ids) | set(region_caption_ids))
    return {
        "region_top_k": region_top_k,
        "candidate_evidence_ids": candidate_ids,
        "region_caption_evidence_ids": region_caption_ids,
        "local_evidence_ids": local_ids,
        "selected_evidence_ids": (select_action or {}).get("evidence_ids") or [],
        "selection_available": bool(select_action),
    }


def rebuild_rows(
    task: dict[str, Any],
    old_rows: list[dict[str, Any]],
    select_action: dict[str, Any] | None,
    *,
    region_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_config = PromptConfig(tool_schema="evidence_select", coordinate_info=True)
    select_result = build_select_result(task, select_action) if select_action else None
    new_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    inserted = False
    for old_row in old_rows:
        old_action = copy.deepcopy(old_row.get("action") or {})
        action = normalize_propose_action(old_action, region_top_k)
        state = transform_row_state(task, old_row, select_action, select_result, region_top_k)
        new_rows.append(retag_row(make_sft_row(task, state, action, prompt_config, len(new_rows))))
        actions.append(action)
        if old_action.get("action") == "propose_regions" and select_action and select_result and not inserted:
            select_state = state_after_propose(task, old_row, action, region_top_k)
            new_rows.append(retag_row(make_sft_row(task, select_state, select_action, prompt_config, len(new_rows))))
            actions.append(copy.deepcopy(select_action))
            inserted = True
    return new_rows, actions


def normalize_propose_action(action: dict[str, Any], region_top_k: int) -> dict[str, Any]:
    if action.get("action") == "propose_regions":
        action["top_k"] = max(region_top_k, int(action.get("top_k", 0) or 0))
    return action


def transform_row_state(
    task: dict[str, Any],
    old_row: dict[str, Any],
    select_action: dict[str, Any] | None,
    select_result: dict[str, Any] | None,
    region_top_k: int,
) -> dict[str, Any]:
    state = {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "step": old_row.get("step"),
        "history": copy.deepcopy(old_row.get("history") or []),
        "tool_results": copy.deepcopy(old_row.get("tool_results") or []),
        "draft_claims": copy.deepcopy(old_row.get("draft_claims") or []),
        "images": copy.deepcopy(old_row.get("images") or []),
        "selected_evidence_ids": [],
    }
    state["history"] = normalize_history(state["history"], region_top_k)
    state["tool_results"] = normalize_tool_results(task, state["tool_results"], region_top_k)
    if select_action and select_result and has_seen_propose(state["history"]):
        state["history"] = insert_after_first_propose(state["history"], copy.deepcopy(select_action))
        state["tool_results"] = insert_after_first_propose_result(state["tool_results"], copy.deepcopy(select_result))
        state["selected_evidence_ids"] = copy.deepcopy(select_action.get("evidence_ids") or [])
    return state


def state_after_propose(
    task: dict[str, Any],
    old_row: dict[str, Any],
    propose_action: dict[str, Any],
    region_top_k: int,
) -> dict[str, Any]:
    return {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "step": old_row.get("step"),
        "history": [copy.deepcopy(propose_action)],
        "tool_results": [propose_result(task, region_top_k)],
        "draft_claims": [],
        "images": copy.deepcopy(old_row.get("images") or [task.get("page_image")]),
        "selected_evidence_ids": [],
    }


def normalize_history(history: list[Any], region_top_k: int) -> list[Any]:
    normalized = copy.deepcopy(history)
    for item in normalized:
        if isinstance(item, dict) and item.get("action") == "propose_regions":
            item["top_k"] = max(region_top_k, int(item.get("top_k", 0) or 0))
            break
    return normalized


def normalize_tool_results(task: dict[str, Any], results: list[Any], region_top_k: int) -> list[Any]:
    normalized = copy.deepcopy(results)
    for index, item in enumerate(normalized):
        if isinstance(item, dict) and item.get("tool") == "propose_regions":
            normalized[index] = propose_result(task, region_top_k)
            break
    return normalized


def has_seen_propose(history: list[Any]) -> bool:
    return any(isinstance(item, dict) and item.get("action") == "propose_regions" for item in history)


def insert_after_first_propose(history: list[Any], item: dict[str, Any]) -> list[Any]:
    if any(isinstance(existing, dict) and existing.get("action") == "select_evidence" for existing in history):
        return history
    result: list[Any] = []
    inserted = False
    for existing in history:
        result.append(existing)
        if not inserted and isinstance(existing, dict) and existing.get("action") == "propose_regions":
            result.append(item)
            inserted = True
    return result


def insert_after_first_propose_result(results: list[Any], item: dict[str, Any]) -> list[Any]:
    if any(isinstance(existing, dict) and existing.get("tool") == "select_evidence" for existing in results):
        return results
    result: list[Any] = []
    inserted = False
    for existing in results:
        result.append(existing)
        if not inserted and isinstance(existing, dict) and existing.get("tool") == "propose_regions":
            result.append(item)
            inserted = True
    return result


def propose_result(task: dict[str, Any], region_top_k: int) -> dict[str, Any]:
    return {"tool": "propose_regions", "regions": propose_regions(task, top_k=region_top_k)}


def build_select_result(task: dict[str, Any], select_action: dict[str, Any] | None) -> dict[str, Any]:
    if not select_action:
        return {}
    selected = [str(item) for item in select_action.get("evidence_ids") or []]
    lookup = local_evidence_lookup(task)
    selected_items = [public_evidence_item(lookup[evidence_id]) for evidence_id in selected if evidence_id in lookup]
    return {
        "tool": "select_evidence",
        "selected_evidence_ids": selected,
        "selected_evidence": selected_items,
    }


def local_evidence_lookup(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("evidence_id")): item
        for item in task.get("local_evidence") or []
        if item.get("evidence_id")
    }


def public_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "evidence_id": item.get("evidence_id"),
        "source_file": item.get("source_file"),
        "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
        "page_end": item.get("page_end"),
        "authority_level": item.get("authority_level"),
        "citation_level": item.get("citation_level"),
        "source_quality": item.get("source_quality"),
        "display_snippet": item.get("display_snippet") or item.get("evidence_summary") or item.get("text", ""),
    }


def rebuild_episode_actions(
    old_actions: list[dict[str, Any]],
    select_action: dict[str, Any] | None,
    region_top_k: int,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    inserted = False
    for old_action in old_actions:
        action = normalize_propose_action(copy.deepcopy(old_action), region_top_k)
        actions.append(action)
        if old_action.get("action") == "propose_regions" and select_action and not inserted:
            actions.append(copy.deepcopy(select_action))
            inserted = True
    return actions


def retag_row(row: dict[str, Any]) -> dict[str, Any]:
    row["tool_schema_version"] = "v0.5_evidence_selection"
    row["label_source"] = "v0_5_evidence_selection"
    return row


def write_outputs(output_dir: Path, tasks: list[dict[str, Any]], episodes: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    write_jsonl(output_dir / "tasks_all.jsonl", tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", [task for task in tasks if task.get("split") == split])
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", [episode for episode in episodes if episode.get("split") == split])
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", [row for row in rows if row.get("split") == split])
    write_jsonl(output_dir / "sft" / "all.jsonl", rows)


def summarize(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    source_dir: Path,
    region_top_k: int,
) -> dict[str, Any]:
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in rows)
    source_rows = 0
    source_action_counts: Counter[str] = Counter()
    source_steps: Counter[str] = Counter()
    source_all = source_dir / "sft" / "all.jsonl"
    if source_all.exists():
        for row in read_jsonl(source_all):
            source_rows += 1
            source_action_counts[str((row.get("action") or {}).get("action"))] += 1
            source_steps[str(row["task_id"])] += 1
    new_steps = Counter(str(row["task_id"]) for row in rows)
    selection_tasks = [task for task in tasks if task.get("evidence_selection", {}).get("selection_available")]
    local_gold_tasks = 0
    region_any_tasks = 0
    region_topk_tasks = 0
    for task in tasks:
        local_ids = {str(item.get("evidence_id")) for item in task.get("local_evidence") or [] if item.get("evidence_id")}
        gold_ids: set[str] = set()
        for claim in task.get("gold", {}).get("claims", []):
            if not claim.get("abstain"):
                gold_ids.update(str(item) for item in claim.get("evidence_ids") or [])
        gold_local = gold_ids & local_ids
        if gold_local:
            local_gold_tasks += 1
        region_any = {
            str(item.get("caption_evidence_id"))
            for item in task.get("region_candidates") or []
            if item.get("caption_evidence_id")
        }
        region_topk = {
            str(item.get("caption_evidence_id"))
            for item in (task.get("region_candidates") or [])[:region_top_k]
            if item.get("caption_evidence_id")
        }
        if gold_local & region_any:
            region_any_tasks += 1
        if gold_local & region_topk:
            region_topk_tasks += 1
    return {
        "tasks_total": len(tasks),
        "task_split_counts": dict(Counter(str(task.get("split")) for task in tasks)),
        "sft_rows_total": len(rows),
        "sft_split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "sft_action_counts": dict(action_counts),
        "avg_steps": sum(new_steps.values()) / max(1, len(new_steps)),
        "min_steps": min(new_steps.values()) if new_steps else 0,
        "max_steps": max(new_steps.values()) if new_steps else 0,
        "source_sft_rows_total": source_rows,
        "source_action_counts": dict(source_action_counts),
        "source_avg_steps": sum(source_steps.values()) / max(1, len(source_steps)),
        "row_increase": len(rows) - source_rows,
        "select_evidence_rows": action_counts.get("select_evidence", 0),
        "selection_task_count": len(selection_tasks),
        "local_gold_task_count": local_gold_tasks,
        "local_gold_in_any_region_count": region_any_tasks,
        "local_gold_in_topk_region_count": region_topk_tasks,
        "selection_recall_over_local_gold": region_topk_tasks / max(1, local_gold_tasks),
        "region_top_k": region_top_k,
        "episode_count": len(episodes),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    q = manifest["quality"]
    lines = [
        "# AgentBench v0.5 Evidence Selection SFT 构建报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 目标",
        "",
        "在 v0.4.2 batch-claims 轨迹中加入显式 `select_evidence` 动作，让模型先选择可信 evidence_id，再裁剪目标图像、检索补充证据并写结构化 claim。",
        "",
        "这一步把训练重点从“必须预测一个完美 caption bbox”转向“从候选证据中选对可追溯 evidence_id”。",
        "",
        "## 数据位置",
        "",
        "```text",
        manifest["output_dir"],
        "```",
        "",
        "## 规模",
        "",
        f"- tasks_total：{q['tasks_total']}",
        f"- task_split_counts：`{json.dumps(q['task_split_counts'], ensure_ascii=False)}`",
        f"- source_sft_rows_total：{q['source_sft_rows_total']}",
        f"- sft_rows_total：{q['sft_rows_total']}",
        f"- row_increase：{q['row_increase']}",
        f"- source_avg_steps：{q['source_avg_steps']:.2f}",
        f"- avg_steps：{q['avg_steps']:.2f}",
        f"- min_steps/max_steps：{q['min_steps']} / {q['max_steps']}",
        "",
        "## 证据选择覆盖",
        "",
        f"- region_top_k：{q['region_top_k']}",
        f"- local_gold_task_count：{q['local_gold_task_count']}",
        f"- local_gold_in_any_region_count：{q['local_gold_in_any_region_count']}",
        f"- local_gold_in_topk_region_count：{q['local_gold_in_topk_region_count']}",
        f"- selection_task_count：{q['selection_task_count']}",
        f"- selection_recall_over_local_gold：{q['selection_recall_over_local_gold']:.4f}",
        "",
        "## 动作分布",
        "",
        f"- sft_action_counts：`{json.dumps(q['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 已知限制",
        "",
        "- 当前 v0.5 只在本地图注 evidence 已进入 top-k region candidates 时插入 `select_evidence`。",
        "- 仍沿用 v0.4.2 的目标图像 region 与 claim schema；红框错误样本尚未全部重标。",
        "- 还没有引入 VLM hard-case 裁决结果；DashScope 图像调用需要单独排查稳定性。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_dir() -> Path:
    return Path(
        "/root/datasets/evidence_grounded_vlm_agentrl/"
        f"agentbench_v0_5_evidence_selection_sft_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
