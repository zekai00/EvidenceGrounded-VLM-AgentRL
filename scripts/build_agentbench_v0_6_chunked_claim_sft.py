#!/usr/bin/env python3
"""Build v0.6 SFT data with chunked claim-writing actions."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_agentbench_v0_4_1_claim_schema import make_sft_row  # noqa: E402
from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig  # noqa: E402
from evidence_agent_env.tools.claim_tools import apply_claim_write, claim_state, claim_write_result  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_5_evidence_selection_sft_20260601_1839"
)
DEFAULT_CHUNK_SIZE = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-history-actions", type=int, default=4)
    parser.add_argument("--max-tool-results", type=int, default=4)
    parser.add_argument("--snippet-chars", type=int, default=140)
    parser.add_argument("--max-text-chars", type=int, default=12000)
    parser.add_argument("--head-text-chars", type=int, default=3500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)

    tasks = read_jsonl(source_dir / "tasks_all.jsonl")
    tasks_by_id = {str(task["task_id"]): retag_task(task) for task in tasks}
    rows_by_task = load_sft_rows(source_dir)
    episodes_by_task = load_episodes(source_dir)

    prompt_config = PromptConfig(
        tool_schema="chunked_claim",
        coordinate_info=True,
        max_history_actions=args.max_history_actions,
        max_tool_results=args.max_tool_results,
        max_evidence_per_result=2,
        snippet_chars=args.snippet_chars,
        max_text_chars=args.max_text_chars,
        head_text_chars=args.head_text_chars,
        compact_claim_state=True,
    )

    all_rows: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    for task_id, task in tasks_by_id.items():
        new_rows, new_actions, note = rebuild_task_rows(
            task,
            rows_by_task[task_id],
            prompt_config,
            chunk_size=max(1, args.chunk_size),
        )
        all_rows.extend(new_rows)
        old_episode = episodes_by_task[task_id]
        all_episodes.append(
            {
                "task_id": task["task_id"],
                "source_task_id": task.get("source_task_id"),
                "split": task.get("split"),
                "variant": task.get("candidate_augmentation", {}).get("variant"),
                "actions": new_actions,
            }
        )
        notes.append(note)

    new_tasks = list(tasks_by_id.values())
    write_outputs(output_dir, new_tasks, all_episodes, all_rows)
    quality = summarize(new_tasks, all_rows, all_episodes, notes, source_dir)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.6_chunked_claim_sft",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "chunk_size": args.chunk_size,
        "prompt_config": {
            "tool_schema": prompt_config.tool_schema,
            "max_history_actions": prompt_config.max_history_actions,
            "max_tool_results": prompt_config.max_tool_results,
            "snippet_chars": prompt_config.snippet_chars,
            "max_text_chars": prompt_config.max_text_chars,
            "head_text_chars": prompt_config.head_text_chars,
            "compact_claim_state": prompt_config.compact_claim_state,
        },
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


def retag_task(task: dict[str, Any]) -> dict[str, Any]:
    new_task = copy.deepcopy(task)
    new_task["dataset_version"] = "v0.6_chunked_claim_sft"
    new_task["tool_schema_version"] = "v0.6_chunked_claim"
    return new_task


def load_sft_rows(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        for row in read_jsonl(source_dir / "sft" / f"{split}.jsonl"):
            rows_by_task.setdefault(str(row["task_id"]), []).append(row)
    for rows in rows_by_task.values():
        rows.sort(key=lambda item: int(item.get("step", 0)))
    return rows_by_task


def load_episodes(source_dir: Path) -> dict[str, dict[str, Any]]:
    return {str(row["task_id"]): row for row in read_jsonl(source_dir / "episodes" / "oracle_episodes.jsonl")}


def rebuild_task_rows(
    task: dict[str, Any],
    old_rows: list[dict[str, Any]],
    prompt_config: PromptConfig,
    *,
    chunk_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    new_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    batch_chunks = 0
    batch_claims = 0
    batch_abstains = 0
    replacement_history: list[dict[str, Any]] | None = None
    replacement_results: list[dict[str, Any]] | None = None
    replacement_claims: list[dict[str, Any]] | None = None

    for old_row in old_rows:
        old_action = copy.deepcopy(old_row.get("action") or {})
        action_name = old_action.get("action")
        if action_name == "write_claims_batch":
            chunks = chunk_batch_action(old_action, chunk_size)
            batch_chunks = len(chunks)
            batch_claims = len(old_action.get("claims") or [])
            batch_abstains = len(old_action.get("abstains") or [])
            base_state = state_from_row(old_row)
            current_history = copy.deepcopy(base_state["history"])
            current_results = copy.deepcopy(base_state["tool_results"])
            current_claims = copy.deepcopy(base_state["draft_claims"])
            for chunk_action in chunks:
                state = {
                    **copy.deepcopy(base_state),
                    "history": copy.deepcopy(current_history),
                    "tool_results": copy.deepcopy(current_results),
                    "draft_claims": copy.deepcopy(current_claims),
                    "claim_state": claim_state(current_claims),
                }
                new_rows.append(retag_row(make_sft_row(task, state, chunk_action, prompt_config, len(new_rows))))
                actions.append(copy.deepcopy(chunk_action))
                current_claims = apply_claim_write(
                    current_claims,
                    claims=chunk_action.get("claims") or [],
                    abstains=chunk_action.get("abstains") or [],
                )
                result = claim_write_result(
                    "write_claims_chunk",
                    current_claims,
                    claims=chunk_action.get("claims") or [],
                    abstains=chunk_action.get("abstains") or [],
                )
                current_history.append(copy.deepcopy(chunk_action))
                current_results.append(result)
            replacement_history = current_history
            replacement_results = current_results
            replacement_claims = current_claims
            continue
        if action_name == "finish" and replacement_history is not None:
            state = state_from_row(old_row)
            state["history"] = copy.deepcopy(replacement_history)
            state["tool_results"] = copy.deepcopy(replacement_results or [])
            state["draft_claims"] = copy.deepcopy(replacement_claims or [])
            state["claim_state"] = claim_state(state["draft_claims"])
            new_rows.append(retag_row(make_sft_row(task, state, old_action, prompt_config, len(new_rows))))
            actions.append(copy.deepcopy(old_action))
            continue
        state = state_from_row(old_row)
        state["claim_state"] = claim_state(state.get("draft_claims") or [])
        new_rows.append(retag_row(make_sft_row(task, state, old_action, prompt_config, len(new_rows))))
        actions.append(copy.deepcopy(old_action))

    return new_rows, actions, {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "batch_chunks": batch_chunks,
        "batch_claims": batch_claims,
        "batch_abstains": batch_abstains,
        "old_steps": len(old_rows),
        "new_steps": len(new_rows),
    }


def state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": row.get("task_id"),
        "split": row.get("split"),
        "step": row.get("step"),
        "history": copy.deepcopy(row.get("history") or []),
        "tool_results": copy.deepcopy(row.get("tool_results") or []),
        "draft_claims": copy.deepcopy(row.get("draft_claims") or []),
        "selected_evidence_ids": copy.deepcopy(row.get("selected_evidence_ids") or []),
        "images": copy.deepcopy(row.get("images") or []),
    }


def chunk_batch_action(action: dict[str, Any], chunk_size: int) -> list[dict[str, Any]]:
    items: list[tuple[str, dict[str, Any]]] = []
    for claim in action.get("claims") or []:
        items.append(("claim", copy.deepcopy(claim)))
    for abstain in action.get("abstains") or []:
        items.append(("abstain", copy.deepcopy(abstain)))
    chunks: list[dict[str, Any]] = []
    for offset in range(0, len(items), chunk_size):
        claim_items: list[dict[str, Any]] = []
        abstain_items: list[dict[str, Any]] = []
        for kind, item in items[offset : offset + chunk_size]:
            if kind == "claim":
                claim_items.append(item)
            else:
                abstain_items.append(item)
        chunks.append({"action": "write_claims_chunk", "claims": claim_items, "abstains": abstain_items})
    return chunks


def retag_row(row: dict[str, Any]) -> dict[str, Any]:
    row["tool_schema_version"] = "v0.6_chunked_claim"
    row["label_source"] = "v0_6_chunked_claim_sft"
    row["claim_state"] = claim_state(row.get("draft_claims") or [])
    return row


def write_outputs(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    write_jsonl(output_dir / "tasks_all.jsonl", tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", [task for task in tasks if task.get("split") == split])
        write_jsonl(
            output_dir / "episodes" / f"{split}_oracle_episodes.jsonl",
            [episode for episode in episodes if episode.get("split") == split],
        )
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", [row for row in rows if row.get("split") == split])
    write_jsonl(output_dir / "sft" / "all.jsonl", rows)


def summarize(
    tasks: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    source_dir: Path,
) -> dict[str, Any]:
    source_rows = read_jsonl(source_dir / "sft" / "all.jsonl")
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in rows)
    source_action_counts = Counter(str((row.get("action") or {}).get("action")) for row in source_rows)
    step_counts = Counter(str(row["task_id"]) for row in rows)
    source_step_counts = Counter(str(row["task_id"]) for row in source_rows)
    prompt_lengths = [len(str(row.get("prompt_text") or "")) for row in rows]
    wcc_rows = [row for row in rows if (row.get("action") or {}).get("action") == "write_claims_chunk"]
    wcc_field_counts = [len((row.get("action") or {}).get("claims") or []) + len((row.get("action") or {}).get("abstains") or []) for row in wcc_rows]
    return {
        "tasks_total": len(tasks),
        "task_split_counts": dict(Counter(str(task.get("split")) for task in tasks)),
        "episode_count": len(episodes),
        "sft_rows_total": len(rows),
        "source_sft_rows_total": len(source_rows),
        "row_increase": len(rows) - len(source_rows),
        "sft_split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "sft_action_counts": dict(action_counts),
        "source_action_counts": dict(source_action_counts),
        "avg_steps": sum(step_counts.values()) / max(1, len(step_counts)),
        "source_avg_steps": sum(source_step_counts.values()) / max(1, len(source_step_counts)),
        "min_steps": min(step_counts.values()) if step_counts else 0,
        "max_steps": max(step_counts.values()) if step_counts else 0,
        "write_claims_chunk_rows": action_counts.get("write_claims_chunk", 0),
        "write_claims_batch_rows": action_counts.get("write_claims_batch", 0),
        "chunk_count_distribution": dict(Counter(str(note["batch_chunks"]) for note in notes)),
        "chunk_fields_avg": sum(wcc_field_counts) / max(1, len(wcc_field_counts)),
        "chunk_fields_min": min(wcc_field_counts) if wcc_field_counts else 0,
        "chunk_fields_max": max(wcc_field_counts) if wcc_field_counts else 0,
        "prompt_chars_avg": sum(prompt_lengths) / max(1, len(prompt_lengths)),
        "prompt_chars_max": max(prompt_lengths) if prompt_lengths else 0,
        "prompt_chars_p95": percentile(prompt_lengths, 0.95),
    }


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[index]


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    q = manifest["quality"]
    lines = [
        "# AgentBench v0.6 Chunked Claim SFT 构建报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 目标",
        "",
        "把 v0.5 中单步超长 `write_claims_batch` 改写成多步 `write_claims_chunk`，同时在 prompt 中使用 compact `claim_state` 控制上下文长度。",
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
        "## 动作分布",
        "",
        f"- source_action_counts：`{json.dumps(q['source_action_counts'], ensure_ascii=False)}`",
        f"- sft_action_counts：`{json.dumps(q['sft_action_counts'], ensure_ascii=False)}`",
        f"- write_claims_chunk_rows：{q['write_claims_chunk_rows']}",
        f"- write_claims_batch_rows：{q['write_claims_batch_rows']}",
        f"- chunk_count_distribution：`{json.dumps(q['chunk_count_distribution'], ensure_ascii=False)}`",
        f"- chunk_fields_avg/min/max：{q['chunk_fields_avg']:.2f} / {q['chunk_fields_min']} / {q['chunk_fields_max']}",
        "",
        "## Prompt 长度",
        "",
        f"- prompt_chars_avg：{q['prompt_chars_avg']:.1f}",
        f"- prompt_chars_p95：{q['prompt_chars_p95']}",
        f"- prompt_chars_max：{q['prompt_chars_max']}",
        "",
        "## 已知限制",
        "",
        "- v0.6 是从 v0.5 oracle trajectory 规则改写得到，不是新人工标注。",
        "- 当前 chunk 大小默认 4 个字段，后续可做 3/4/5 消融。",
        "- 当前 prompt 仍用字符预算近似 token 预算，后续训练前可再接 tokenizer 统计真实 token。",
        "- v0.6 解决的是最终写 claim 的稳定性和上下文控制，不解决 direct bbox 定位问题。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_dir() -> Path:
    return Path(
        "/root/datasets/evidence_grounded_vlm_agentrl/"
        f"agentbench_v0_6_chunked_claim_sft_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())

