#!/usr/bin/env python3
"""Build v0.7 SFT rows whose prompts include executable phase masks."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env import EvidenceAgentEnv  # noqa: E402
from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation, build_prompt_text  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_7_inspect_crop_sft_20260605_2336"
)
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--evidence-index", type=Path, default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--max-history-actions", type=int, default=6)
    parser.add_argument("--max-tool-results", type=int, default=5)
    parser.add_argument("--max-evidence-per-result", type=int, default=2)
    parser.add_argument("--snippet-chars", type=int, default=120)
    parser.add_argument("--max-text-chars", type=int, default=10000)
    parser.add_argument("--head-text-chars", type=int, default=3000)
    parser.add_argument("--only-actions", default="")
    parser.add_argument("--max-actions-per-episode", type=int, default=0, help="0 means replay all oracle actions.")
    parser.add_argument("--preview-rows", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    args.output_dir = output_dir
    (output_dir / "sft").mkdir(parents=True, exist_ok=True)
    (output_dir / "validation_rollouts").mkdir(parents=True, exist_ok=True)

    tasks_path = args.source_dir / "tasks_all.jsonl"
    tasks = read_jsonl(tasks_path)
    episodes = read_jsonl(args.source_dir / "episodes" / "oracle_episodes.jsonl")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    episodes_by_split = group_episodes_by_split(episodes)
    allowed_actions = {item.strip() for item in args.only_actions.split(",") if item.strip()}

    prompt_config = PromptConfig(
        max_history_actions=args.max_history_actions,
        max_tool_results=args.max_tool_results,
        max_evidence_per_result=args.max_evidence_per_result,
        snippet_chars=args.snippet_chars,
        max_text_chars=args.max_text_chars,
        head_text_chars=args.head_text_chars,
        coordinate_info=True,
        tool_schema="inspect_crop",
        compact_claim_state=True,
        region_selection_hint=True,
        strict_claim_phase_hint=True,
    )

    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.7_phase_mask_patch_sft",
        "source_dir": str(args.source_dir),
        "tasks": str(tasks_path),
        "evidence_index": str(args.evidence_index),
        "output_dir": str(output_dir),
        "tool_schema": "inspect_crop",
        "prompt_config": prompt_config.__dict__,
        "only_actions": sorted(allowed_actions),
        "splits": {},
    }
    all_rows: list[dict[str, Any]] = []
    for split, split_rows_episodes in episodes_by_split.items():
        rows = build_split_rows(split, split_rows_episodes, task_by_id, tasks_path, args, prompt_config, allowed_actions)
        all_rows.extend(rows)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", rows)
        write_jsonl(output_dir / f"{split}_preview.jsonl", rows[: args.preview_rows])
        manifest["splits"][split] = summarize_rows(rows, output_dir / "sft" / f"{split}.jsonl")

    write_jsonl(output_dir / "sft" / "all.jsonl", all_rows)
    manifest["all"] = summarize_rows(all_rows, output_dir / "sft" / "all.jsonl")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_split_rows(
    split: str,
    episodes: list[dict[str, Any]],
    task_by_id: dict[str, dict[str, Any]],
    tasks_path: Path,
    args: argparse.Namespace,
    prompt_config: PromptConfig,
    allowed_actions: set[str],
) -> list[dict[str, Any]]:
    env = EvidenceAgentEnv(
        tasks_path,
        args.evidence_index,
        args.output_dir or default_output_dir() / "_env",
        max_steps=args.max_steps,
        include_gold_regions=False,
        phase_aware_mask=True,
        enforce_tool_mask=True,
        tool_schema="inspect_crop",
    )
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        task_id = str(episode["task_id"])
        task = task_by_id[task_id]
        obs = env.reset(task_id=task_id)
        for step, action in enumerate(episode.get("actions") or []):
            if args.max_actions_per_episode > 0 and step >= args.max_actions_per_episode:
                break
            action_name = str(action.get("action"))
            if not allowed_actions or action_name in allowed_actions:
                rows.append(make_row(task, split, obs, action, prompt_config, step))
            obs, _, terminated, info = env.step(action)
            if info.get("result", {}).get("error"):
                rows[-1]["oracle_step_error"] = info["result"]["error"] if rows else info["result"]["error"]
                break
            if terminated:
                break
    return rows


def make_row(
    task: dict[str, Any],
    split: str,
    obs: dict[str, Any],
    action: dict[str, Any],
    prompt_config: PromptConfig,
    step: int,
) -> dict[str, Any]:
    messages = build_messages_from_observation(obs, prompt_config, include_assistant_action=action)
    return {
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "split": split,
        "variant": task.get("candidate_augmentation", {}).get("variant"),
        "step": step,
        "tool_schema_version": "v0.7_inspect_crop_phase_mask_patch",
        "action": action,
        "history": obs.get("history") or [],
        "tool_results": obs.get("tool_results") or [],
        "draft_claims": obs.get("draft_claims") or [],
        "claim_state": obs.get("claim_state") or {},
        "selected_evidence_ids": obs.get("selected_evidence_ids") or [],
        "visible_evidence_ids": obs.get("visible_evidence_ids") or [],
        "available_actions": obs.get("available_actions") or [],
        "tool_mask": obs.get("tool_mask") or {},
        "available_region_ids": obs.get("available_region_ids") or [],
        "regions": obs.get("regions") or {},
        "valid_crop_count": obs.get("valid_crop_count") or 0,
        "images": [item.get("path") for item in obs.get("images") or [] if isinstance(item, dict) and item.get("path")],
        "prompt_text": build_prompt_text(obs, prompt_config),
        "messages": messages,
        "label_source": "v0_7_phase_mask_patch_sft",
    }


def group_episodes_by_split(episodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        result.setdefault(str(episode.get("split")), []).append(episode)
    return result


def summarize_rows(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in rows)
    phase_counts = Counter(str((row.get("tool_mask") or {}).get("phase")) for row in rows)
    return {
        "rows": len(rows),
        "path": str(path),
        "action_counts": dict(sorted(action_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.7 Phase-Mask Patch SFT 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- source_dir: `{manifest['source_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- tool_schema: `{manifest['tool_schema']}`",
        "",
        "## 数据内容",
        "",
        "这批数据用真实 EvidenceAgentEnv 重放 oracle episode，在每一步 prompt 中保留 `available_actions` 和 `tool_mask`。",
        "目的不是新增任务知识，而是训练模型服从可执行环境的阶段约束，修复自由 rollout 初始阶段直接 retrieve/open/write 的问题。",
        "",
        "## Split",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['rows']} rows, actions={info['action_counts']}, phases={info['phase_counts']}")
    lines.extend(
        [
            "",
            "## 预期作用",
            "",
            "- 初始阶段只允许 `inspect_page`。",
            "- `inspect_page` 后只允许 `crop_target`。",
            "- 检索预算达到上限后，prompt 明确阻止继续 `retrieve_evidence`。",
            "- claim 阶段优先 `write_claims_chunk`，字段写完后 `finish`。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(f"/root/datasets/evidence_grounded_vlm_agentrl/v0_7_phase_mask_patch_sft_{stamp}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
