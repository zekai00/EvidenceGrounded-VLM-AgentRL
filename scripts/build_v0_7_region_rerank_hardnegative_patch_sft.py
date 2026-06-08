#!/usr/bin/env python3
"""Build v0.7 crop_target region-reranking hard-negative SFT rows."""

from __future__ import annotations

import argparse
import copy
import json
import random
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
REGION_RERANK_HINT = (
    "区域重排提示：当前 inspect_page 已返回完整候选区域。正确目标山水画图像不一定是第一个候选，"
    "也可能排在 r4/r5/r6/r7。不要选择正文、图注、页眉页脚、边角干扰或 text_or_caption_candidate；"
    "优先选择 type=figure_candidate 且与任务目标、图注文本、页面中的山水画图像内容一致的区域。"
    "下一步只能输出 crop_target，region_id 必须来自候选区域列表。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--evidence-index", type=Path, default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--inspect-top-k", type=int, default=10)
    parser.add_argument("--max-history-actions", type=int, default=6)
    parser.add_argument("--max-tool-results", type=int, default=5)
    parser.add_argument("--max-evidence-per-result", type=int, default=5)
    parser.add_argument("--snippet-chars", type=int, default=140)
    parser.add_argument("--max-text-chars", type=int, default=14000)
    parser.add_argument("--head-text-chars", type=int, default=4000)
    parser.add_argument("--train-top1-copies", type=int, default=1)
    parser.add_argument("--train-not-top1-copies", type=int, default=3)
    parser.add_argument("--train-late-rank-copies", type=int, default=5)
    parser.add_argument("--late-rank-threshold", type=int, default=5)
    parser.add_argument("--replay-inspect", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(exist_ok=True)

    tasks_path = args.source_dir / "tasks_all.jsonl"
    tasks = read_jsonl(tasks_path)
    episodes = read_jsonl(args.source_dir / "episodes" / "oracle_episodes.jsonl")
    task_by_id = {str(task["task_id"]): task for task in tasks}
    episodes_by_split = group_by_split(episodes)
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
        strict_claim_phase_hint=False,
    )

    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.7_region_rerank_hardnegative_patch_sft",
        "source_dir": str(args.source_dir),
        "tasks": str(tasks_path),
        "evidence_index": str(args.evidence_index),
        "output_dir": str(output_dir),
        "region_rerank_hint": REGION_RERANK_HINT,
        "prompt_config": prompt_config.__dict__,
        "oversample": {
            "train_top1_copies": args.train_top1_copies,
            "train_not_top1_copies": args.train_not_top1_copies,
            "train_late_rank_copies": args.train_late_rank_copies,
            "late_rank_threshold": args.late_rank_threshold,
        },
        "splits": {},
    }
    all_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        rows, stats = build_split(
            split,
            episodes_by_split.get(split, []),
            task_by_id,
            tasks_path,
            args,
            prompt_config,
        )
        if split == "train":
            random.shuffle(rows)
        all_rows.extend(rows)
        out_path = output_dir / "sft" / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        write_jsonl(output_dir / f"{split}_preview.jsonl", rows[:5])
        stats["path"] = str(out_path)
        manifest["splits"][split] = stats
    write_jsonl(output_dir / "sft" / "all.jsonl", all_rows)
    manifest["all_rows"] = len(all_rows)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def build_split(
    split: str,
    episodes: list[dict[str, Any]],
    task_by_id: dict[str, dict[str, Any]],
    tasks_path: Path,
    args: argparse.Namespace,
    prompt_config: PromptConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = EvidenceAgentEnv(
        tasks_path,
        args.evidence_index,
        (args.output_dir or default_output_dir()) / "_env_replay",
        max_steps=18,
        include_gold_regions=False,
        phase_aware_mask=True,
        enforce_tool_mask=True,
        tool_schema="inspect_crop",
    )
    rows: list[dict[str, Any]] = []
    rank_counts: Counter[str] = Counter()
    copy_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    region_type_counts: Counter[str] = Counter()
    missing_target = 0

    for episode in episodes:
        task_id = str(episode["task_id"])
        task = task_by_id[task_id]
        actions = episode.get("actions") or []
        if len(actions) < 2:
            continue
        inspect_action = {"action": "inspect_page", "top_k": int(args.inspect_top_k)}
        crop_action = first_action(actions, "crop_target")
        if not crop_action:
            continue

        obs0 = env.reset(task_id=task_id)
        if args.replay_inspect:
            row0 = make_row(task, split, obs0, inspect_action, prompt_config, step=0, copy_index=0, copies=1, diagnostic={})
            rows.append(row0)
            action_counts["inspect_page"] += 1
        obs1, _, _, info = env.step(inspect_action)
        if info.get("result", {}).get("error"):
            continue
        diagnostic = crop_diagnostic(obs1, crop_action)
        rank = diagnostic.get("target_rank")
        rank_counts[str(rank if rank is not None else "missing")] += 1
        if diagnostic.get("target_type"):
            region_type_counts[str(diagnostic["target_type"])] += 1
        missing_target += int(rank is None)
        copies = copy_count(split, rank, args)
        copy_counts[str(copies)] += 1
        for copy_index in range(copies):
            rows.append(
                make_row(
                    task,
                    split,
                    obs1,
                    crop_action,
                    prompt_config,
                    step=1,
                    copy_index=copy_index,
                    copies=copies,
                    diagnostic=diagnostic,
                )
            )
            action_counts["crop_target"] += 1
    return rows, {
        "rows": len(rows),
        "episodes": len(episodes),
        "action_counts": dict(sorted(action_counts.items())),
        "target_rank_distribution": dict(sorted(rank_counts.items(), key=lambda item: str(item[0]))),
        "target_type_distribution": dict(sorted(region_type_counts.items())),
        "copy_count_distribution": dict(sorted(copy_counts.items())),
        "missing_target_rows": missing_target,
    }


def make_row(
    task: dict[str, Any],
    split: str,
    obs: dict[str, Any],
    action: dict[str, Any],
    prompt_config: PromptConfig,
    *,
    step: int,
    copy_index: int,
    copies: int,
    diagnostic: dict[str, Any],
) -> dict[str, Any]:
    obs = copy.deepcopy(obs)
    if action.get("action") == "crop_target":
        obs["region_rerank_hint"] = REGION_RERANK_HINT
    messages = build_messages_from_observation(obs, prompt_config, include_assistant_action=action)
    prompt_text = build_prompt_text(obs, prompt_config)
    if action.get("action") == "crop_target":
        messages = add_user_text_prefix(messages, REGION_RERANK_HINT)
        prompt_text = REGION_RERANK_HINT + "\n\n" + prompt_text
    return {
        "task_id": task["task_id"],
        "source_task_id": task.get("source_task_id"),
        "split": split,
        "variant": task.get("candidate_augmentation", {}).get("variant"),
        "step": step,
        "tool_schema_version": "v0.7_inspect_crop_region_rerank_hardnegative_patch",
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
        "prompt_text": prompt_text,
        "messages": messages,
        "label_source": "v0_7_region_rerank_hardnegative_patch_sft",
        "patch_source": {
            "copy_index": copy_index,
            "copies": copies,
            "diagnostic": diagnostic,
        },
    }


def add_user_text_prefix(messages: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    copied = copy.deepcopy(messages)
    for message in copied:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = prefix + "\n\n" + content
            return copied
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = prefix + "\n\n" + str(item.get("text", ""))
                    return copied
    return copied


def first_action(actions: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for action in actions:
        if isinstance(action, dict) and action.get("action") == name:
            return copy.deepcopy(action)
    return None


def crop_diagnostic(obs: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    target_region_id = str(action.get("region_id") or "")
    regions = []
    for result in obs.get("tool_results") or []:
        if isinstance(result, dict) and result.get("tool") == "inspect_page":
            regions = [item for item in result.get("regions") or [] if isinstance(item, dict)]
            break
    target = None
    target_rank = None
    for rank, region in enumerate(regions, start=1):
        if str(region.get("region_id")) == target_region_id:
            target = region
            target_rank = rank
            break
    top1 = regions[0] if regions else {}
    return {
        "target_region_id": target_region_id,
        "target_rank": target_rank,
        "target_bbox": target.get("bbox") if target else None,
        "target_type": target.get("type") if target else None,
        "target_source": target.get("source") if target else None,
        "top1_region_id": top1.get("region_id"),
        "top1_type": top1.get("type"),
        "top1_source": top1.get("source"),
        "candidate_count": len(regions),
    }


def copy_count(split: str, rank: Any, args: argparse.Namespace) -> int:
    if split != "train":
        return 1
    try:
        rank_int = int(rank)
    except Exception:
        return max(1, args.train_late_rank_copies)
    if rank_int <= 1:
        return max(1, args.train_top1_copies)
    if rank_int >= int(args.late_rank_threshold):
        return max(1, args.train_late_rank_copies)
    return max(1, args.train_not_top1_copies)


def group_by_split(episodes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        result.setdefault(str(episode.get("split")), []).append(episode)
    return result


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.7 Region Rerank Hard-Negative Patch SFT 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- source_dir: `{manifest['source_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## 目标",
        "",
        "修复 `crop_target` 选错 region 的问题。候选池 oracle 已经足够，核心是模型需要在完整候选列表中重排并选择目标山水画图像。",
        "",
        "## 构建策略",
        "",
        "- 用真实 `EvidenceAgentEnv` 重放到 `inspect_page` 后的 `region_selection` 状态。",
        "- prompt 中至少展示 10 个候选区域，避免正确 region 排在 r4-r8 时不可见。",
        "- 对训练集中正确 region 不是 top1 的样本过采样；rank 更靠后时更高过采样。",
        "- 不暴露 `is_target/target_iou/gold_iou` 等 hidden label。",
        "",
        "## Split",
        "",
        "| split | rows | episodes | inspect_page | crop_target | missing target | rank dist | copy dist |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for split, stats in manifest["splits"].items():
        actions = stats["action_counts"]
        lines.append(
            f"| {split} | {stats['rows']} | {stats['episodes']} | "
            f"{actions.get('inspect_page', 0)} | {actions.get('crop_target', 0)} | "
            f"{stats['missing_target_rows']} | `{json.dumps(stats['target_rank_distribution'], ensure_ascii=False)}` | "
            f"`{json.dumps(stats['copy_count_distribution'], ensure_ascii=False)}` |"
        )
    lines.extend(["", "## Region Rerank Hint", "", "```text", manifest["region_rerank_hint"], "```"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir() -> Path:
    return Path(
        f"/root/datasets/evidence_grounded_vlm_agentrl/v0_7_region_rerank_hardnegative_patch_sft_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
