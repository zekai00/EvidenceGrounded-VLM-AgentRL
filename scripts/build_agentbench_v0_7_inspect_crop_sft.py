#!/usr/bin/env python3
"""Build v0.7 SFT data with a natural inspect-page then crop-target protocol."""

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
from evidence_agent_env.tools.claim_tools import (  # noqa: E402
    apply_claim_write,
    claim_state,
    normalize_abstain,
    normalize_claim,
)
from evidence_agent_env.tools.crop import image_size  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_chunked_claim_sft_20260604_1650"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--inspect-top-k", type=int, default=10)
    parser.add_argument("--drop-pre-crop-select", action="store_true", default=True)
    parser.add_argument("--max-history-actions", type=int, default=6)
    parser.add_argument("--max-tool-results", type=int, default=5)
    parser.add_argument("--snippet-chars", type=int, default=140)
    parser.add_argument("--max-text-chars", type=int, default=14000)
    parser.add_argument("--head-text-chars", type=int, default=4000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    (output_dir / "sft").mkdir(parents=True, exist_ok=True)
    (output_dir / "episodes").mkdir(parents=True, exist_ok=True)

    tasks = [retag_task(task) for task in read_jsonl(source_dir / "tasks_all.jsonl")]
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    rows_by_task = load_sft_rows(source_dir)

    prompt_config = PromptConfig(
        tool_schema="inspect_crop",
        coordinate_info=True,
        max_history_actions=args.max_history_actions,
        max_tool_results=args.max_tool_results,
        max_evidence_per_result=2,
        snippet_chars=args.snippet_chars,
        max_text_chars=args.max_text_chars,
        head_text_chars=args.head_text_chars,
        compact_claim_state=True,
        region_selection_hint=True,
        strict_claim_phase_hint=True,
    )

    all_rows: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        new_rows, new_actions, note = rebuild_task(
            task,
            rows_by_task.get(task_id, []),
            prompt_config,
            inspect_top_k=args.inspect_top_k,
            drop_pre_crop_select=args.drop_pre_crop_select,
        )
        all_rows.extend(new_rows)
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

    write_outputs(output_dir, list(tasks_by_id.values()), all_episodes, all_rows)
    quality = summarize(tasks, all_rows, notes, source_dir)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.7_inspect_crop_sft",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "protocol": "inspect_page -> crop_target -> retrieve/open/select evidence -> write_claims_chunk/write_claim/abstain_claim -> finish",
        "transform": {
            "propose_regions": "inspect_page",
            "crop_region": "crop_target",
            "drop_pre_crop_select": bool(args.drop_pre_crop_select),
        },
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
    new_task["dataset_version"] = "v0.7_inspect_crop_sft"
    new_task["tool_schema_version"] = "v0.7_inspect_crop"
    new_task["goal"] = (
        "Inspect the original PDF page layout, crop the target Chinese landscape figure, "
        "retrieve/open evidence, and write evidence-grounded structured claims."
    )
    tools = list(new_task.get("available_tools") or [])
    for tool in ["inspect_page", "crop_target"]:
        if tool not in tools:
            tools.append(tool)
    new_task["available_tools"] = tools
    return new_task


def load_sft_rows(source_dir: Path) -> dict[str, list[dict[str, Any]]]:
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    for split in ["train", "val", "test"]:
        for row in read_jsonl(source_dir / "sft" / f"{split}.jsonl"):
            rows_by_task.setdefault(str(row["task_id"]), []).append(row)
    for rows in rows_by_task.values():
        rows.sort(key=lambda item: int(item.get("step", 0)))
    return rows_by_task


def rebuild_task(
    task: dict[str, Any],
    old_rows: list[dict[str, Any]],
    prompt_config: PromptConfig,
    *,
    inspect_top_k: int,
    drop_pre_crop_select: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    new_rows: list[dict[str, Any]] = []
    new_actions: list[dict[str, Any]] = []
    current_history: list[dict[str, Any]] = []
    current_results: list[dict[str, Any]] = []
    current_claims: list[dict[str, Any]] = []
    selected_evidence_ids: list[str] = []
    images = [task.get("page_image")]
    crop_seen = False
    skipped_pre_crop_select = 0

    for index, old_row in enumerate(old_rows):
        old_action = copy.deepcopy(old_row.get("action") or {})
        new_action = transform_action(old_action, inspect_top_k=inspect_top_k)
        action_name = str(new_action.get("action"))

        if (
            drop_pre_crop_select
            and action_name == "select_evidence"
            and not crop_seen
        ):
            skipped_pre_crop_select += 1
            continue

        state = {
            "task_id": task.get("task_id"),
            "split": task.get("split"),
            "step": len(new_rows),
            "history": copy.deepcopy(current_history),
            "tool_results": copy.deepcopy(current_results),
            "draft_claims": copy.deepcopy(current_claims),
            "claim_state": claim_state(current_claims),
            "selected_evidence_ids": copy.deepcopy(selected_evidence_ids),
            "images": [path for path in images if path],
        }
        row = make_sft_row(task, state, new_action, prompt_config, len(new_rows))
        row = retag_row(row)
        new_rows.append(row)
        new_actions.append(copy.deepcopy(new_action))

        result = transformed_result_for_action(task, old_rows, index, old_action, new_action)
        current_history.append(copy.deepcopy(new_action))
        if result:
            current_results.append(result)
            if action_name == "crop_target" and result.get("crop_path"):
                crop_seen = True
                images = [task.get("page_image"), result.get("crop_path")]
        if action_name == "select_evidence":
            for evidence_id in new_action.get("evidence_ids") or []:
                if evidence_id not in selected_evidence_ids:
                    selected_evidence_ids.append(str(evidence_id))
        current_claims = update_claims(current_claims, new_action)

    return new_rows, new_actions, {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "old_steps": len(old_rows),
        "new_steps": len(new_rows),
        "skipped_pre_crop_select": skipped_pre_crop_select,
        "crop_seen": crop_seen,
    }


def transform_action(action: dict[str, Any], *, inspect_top_k: int) -> dict[str, Any]:
    new_action = copy.deepcopy(action)
    name = str(new_action.get("action"))
    if name == "propose_regions":
        return {"action": "inspect_page", "top_k": int(new_action.get("top_k") or inspect_top_k)}
    if name == "crop_region":
        return {"action": "crop_target", "region_id": new_action.get("region_id")}
    return new_action


def transformed_result_for_action(
    task: dict[str, Any],
    old_rows: list[dict[str, Any]],
    index: int,
    old_action: dict[str, Any],
    new_action: dict[str, Any],
) -> dict[str, Any] | None:
    old_result = next_result(old_rows, index)
    old_name = str(old_action.get("action"))
    new_name = str(new_action.get("action"))
    if old_name == "propose_regions":
        regions = copy.deepcopy((old_result or {}).get("regions") or [])
        return {
            "tool": "inspect_page",
            "page_image": task.get("page_image"),
            "page_size": image_size(task.get("page_image")),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "regions": regions,
            "layout_regions": regions,
        }
    if old_name == "crop_region":
        result = copy.deepcopy(old_result or {})
        result["tool"] = "crop_target"
        result["crop_mode"] = "region_id"
        return result
    if new_name == "finish":
        return {"tool": "finish", "status": new_action.get("status", "done"), "draft_claims": []}
    if old_result:
        result = copy.deepcopy(old_result)
        result["tool"] = transform_tool_name(str(result.get("tool") or new_name))
        return result
    return None


def next_result(old_rows: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    if index + 1 >= len(old_rows):
        return None
    current_len = len(old_rows[index].get("tool_results") or [])
    next_results = old_rows[index + 1].get("tool_results") or []
    if len(next_results) > current_len:
        result = next_results[current_len]
        return result if isinstance(result, dict) else None
    if next_results:
        result = next_results[-1]
        return result if isinstance(result, dict) else None
    return None


def transform_tool_name(tool: str) -> str:
    if tool == "propose_regions":
        return "inspect_page"
    if tool == "crop_region":
        return "crop_target"
    return tool


def update_claims(current_claims: list[dict[str, Any]], action: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(action.get("action"))
    if name == "write_claim":
        return apply_claim_write(current_claims, claims=[normalize_claim(action)], abstains=[])
    if name == "abstain_claim":
        return apply_claim_write(current_claims, claims=[], abstains=[normalize_abstain(action)])
    if name in {"write_claims_chunk", "write_claims_batch"}:
        return apply_claim_write(
            current_claims,
            claims=action.get("claims") or [],
            abstains=action.get("abstains") or [],
        )
    return current_claims


def retag_row(row: dict[str, Any]) -> dict[str, Any]:
    row["tool_schema_version"] = "v0.7_inspect_crop"
    row["label_source"] = "v0_7_inspect_crop_sft"
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


def summarize(tasks: list[dict[str, Any]], rows: list[dict[str, Any]], notes: list[dict[str, Any]], source_dir: Path) -> dict[str, Any]:
    source_rows = read_jsonl(source_dir / "sft" / "all.jsonl")
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in rows)
    source_action_counts = Counter(str((row.get("action") or {}).get("action")) for row in source_rows)
    step_counts = Counter(str(row["task_id"]) for row in rows)
    prompt_lengths = [len(str(row.get("prompt_text") or "")) for row in rows]
    return {
        "tasks_total": len(tasks),
        "task_split_counts": dict(Counter(str(task.get("split")) for task in tasks)),
        "sft_rows_total": len(rows),
        "source_sft_rows_total": len(source_rows),
        "row_delta": len(rows) - len(source_rows),
        "sft_split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "sft_action_counts": dict(action_counts),
        "source_action_counts": dict(source_action_counts),
        "avg_steps": sum(step_counts.values()) / max(1, len(step_counts)),
        "min_steps": min(step_counts.values()) if step_counts else 0,
        "max_steps": max(step_counts.values()) if step_counts else 0,
        "skipped_pre_crop_select_total": sum(int(note.get("skipped_pre_crop_select") or 0) for note in notes),
        "tasks_with_skipped_pre_crop_select": sum(1 for note in notes if int(note.get("skipped_pre_crop_select") or 0) > 0),
        "tasks_without_crop_result": sum(1 for note in notes if not note.get("crop_seen")),
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
        "# AgentBench v0.7 Inspect-Crop SFT 构建报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 目标",
        "",
        "把旧的 `propose_regions -> select_evidence -> crop_region` 前置流程改成更自然的 `inspect_page -> crop_target`。",
        "`inspect_page` 表示检查 PDF 页面并返回布局区域；`crop_target` 表示裁剪目标图像区域，可以接 `region_id`，也为后续 direct bbox 留出 `bbox` 接口。",
        "",
        "## 数据位置",
        "",
        "```text",
        manifest["output_dir"],
        "```",
        "",
        "## 规模与动作分布",
        "",
        f"- tasks_total：{q['tasks_total']}",
        f"- task_split_counts：`{json.dumps(q['task_split_counts'], ensure_ascii=False)}`",
        f"- source_sft_rows_total：{q['source_sft_rows_total']}",
        f"- sft_rows_total：{q['sft_rows_total']}",
        f"- row_delta：{q['row_delta']}",
        f"- avg_steps/min/max：{q['avg_steps']:.2f} / {q['min_steps']} / {q['max_steps']}",
        f"- source_action_counts：`{json.dumps(q['source_action_counts'], ensure_ascii=False)}`",
        f"- sft_action_counts：`{json.dumps(q['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 关键质量检查",
        "",
        f"- skipped_pre_crop_select_total：{q['skipped_pre_crop_select_total']}",
        f"- tasks_with_skipped_pre_crop_select：{q['tasks_with_skipped_pre_crop_select']}",
        f"- tasks_without_crop_result：{q['tasks_without_crop_result']}",
        f"- prompt_chars_avg/p95/max：{q['prompt_chars_avg']:.1f} / {q['prompt_chars_p95']} / {q['prompt_chars_max']}",
        "",
        "## 解释",
        "",
        "- 删除 crop 前的 `select_evidence` 是为了让轨迹更接近真实 agent：先看页面和裁剪目标，再检索或选择证据。",
        "- `inspect_page` 的工具返回仍包含候选布局区域，这是可执行环境需要的页面结构信息，不是最终 claim 答案。",
        "- `crop_target(region_id)` 是当前最稳的训练目标；后续可以加入 `crop_target(bbox)` 的 direct bbox 数据，评估 7B/8B 是否能直接定位。",
        "",
        "## 已知限制",
        "",
        "- v0.7 是协议转换版，不是新人工标注版；候选区域质量仍继承 v0.6/v0.5 的 region proposal。",
        "- 当前 `inspect_page` 内部仍调用 deterministic region proposal；这是环境工具实现，不暴露为模型的独立动作。",
        "- 如果后续 direct bbox 能力提升，可以减少 region_id 依赖，但现阶段不建议直接废弃 region_id。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_dir() -> Path:
    return Path(
        "/root/datasets/evidence_grounded_vlm_agentrl/"
        f"agentbench_v0_7_inspect_crop_sft_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
