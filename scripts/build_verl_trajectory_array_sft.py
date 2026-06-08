#!/usr/bin/env python3
"""Build one-shot trajectory-array SFT data for verl.

The step-wise v0.6 SFT adapter learns "given current state, emit the next tool
call". The trajectory-level GRPO bridge expects "given the task, emit the full
JSON action array". This script converts executable oracle episodes into that
task-level SFT format.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_verl_trajectory_grpo_prompts import build_prompt_text  # noqa: E402
from evidence_agent_env.env import EvidenceAgentEnv  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_chunked_claim_sft_20260604_1650"
)
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)
DEFAULT_MAX_STEPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--evidence-index", type=Path, default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-train", type=int, default=0, help="0 means all train rows.")
    parser.add_argument("--max-val", type=int, default=0, help="0 means all val rows.")
    parser.add_argument("--max-test", type=int, default=0, help="0 means all test rows.")
    parser.add_argument("--image-max-pixels", type=int, default=131072)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--region-top-k", type=int, default=10)
    parser.add_argument("--max-local-evidence", type=int, default=6)
    parser.add_argument("--max-candidate-evidence", type=int, default=10)
    parser.add_argument("--max-snippet-chars", type=int, default=180)
    parser.add_argument("--include-static-tool-preview", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--phase-aware-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-tool-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sanitize-local-caption-open",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop redundant open_evidence(local_caption_xxx) actions after select_evidence, because local captions are already visible/selectable.",
    )
    parser.add_argument("--validate-env", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-limit", type=int, default=50, help="Validate first N rows per split; 0 disables.")
    parser.add_argument("--preview-rows", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(args.source_dir / "tasks_all.jsonl")
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    episodes = read_jsonl(args.source_dir / "episodes" / "oracle_episodes.jsonl")
    split_limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}

    prompt_args = SimpleNamespace(
        max_steps=args.max_steps,
        region_top_k=args.region_top_k,
        max_local_evidence=args.max_local_evidence,
        max_candidate_evidence=args.max_candidate_evidence,
        max_snippet_chars=args.max_snippet_chars,
        include_static_tool_preview=args.include_static_tool_preview,
        evidence_items=load_evidence_items(args.evidence_index),
    )

    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.6_trajectory_array_sft",
        "source_dir": str(args.source_dir),
        "source_tasks": str(args.source_dir / "tasks_all.jsonl"),
        "source_episodes": str(args.source_dir / "episodes" / "oracle_episodes.jsonl"),
        "evidence_index": str(args.evidence_index),
        "output_dir": str(args.output_dir),
        "image_max_pixels": args.image_max_pixels,
        "max_steps": args.max_steps,
        "region_top_k": args.region_top_k,
        "max_candidate_evidence": args.max_candidate_evidence,
        "include_static_tool_preview": args.include_static_tool_preview,
        "sanitize_local_caption_open": args.sanitize_local_caption_open,
        "phase_aware_mask": args.phase_aware_mask,
        "enforce_tool_mask": args.enforce_tool_mask,
        "format": {
            "messages_key": "messages",
            "image_key": "images",
            "supervision": "assistant emits one complete JSON action array",
        },
        "splits": {},
    }

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        split_episodes = [episode for episode in episodes if str(episode.get("split")) == split]
        limit = split_limits.get(split, 0)
        if limit and limit > 0:
            split_episodes = split_episodes[:limit]
        records = [build_record(tasks_by_id[str(ep["task_id"])], ep, prompt_args, args) for ep in split_episodes]
        out_parquet = args.output_dir / f"{split}.parquet"
        pd.DataFrame(records).to_parquet(out_parquet, index=False)
        preview_path = args.output_dir / f"{split}_preview.jsonl"
        write_jsonl(preview_path, records[: args.preview_rows])
        validation = validate_records(split, records, args) if args.validate_env and args.validate_limit else {}
        stats = summarize_records(records)
        manifest["splits"][split] = {
            "rows": len(records),
            "parquet": str(out_parquet),
            "preview": str(preview_path),
            "stats": stats,
            "validation": validation,
        }
        print(f"[{split}] wrote {len(records)} rows -> {out_parquet}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_record(
    task: dict[str, Any],
    episode: dict[str, Any],
    prompt_args: SimpleNamespace,
    args: argparse.Namespace,
) -> dict[str, Any]:
    actions = list(episode.get("actions") or [])
    if args.sanitize_local_caption_open:
        actions = sanitize_actions(task, actions)
    response = json.dumps(actions, ensure_ascii=False, separators=(",", ":"))
    prompt_text = build_prompt_text(task, prompt_args)
    messages = [
        {"role": "user", "content": "<image>\n" + prompt_text},
        {"role": "assistant", "content": response},
    ]
    image = {"image": task.get("page_image"), "max_pixels": args.image_max_pixels}
    action_counts = Counter(str(action.get("action")) for action in actions if isinstance(action, dict))
    return {
        "data_source": "evidence_grounded_vlm_agentrl_v0_6_trajectory_array_sft",
        "messages": messages,
        "images": [image],
        "prompt": messages[0]["content"],
        "response": response,
        "task_id": task.get("task_id"),
        "source_task_id": task.get("source_task_id"),
        "split": task.get("split"),
        "variant": task.get("candidate_augmentation", {}).get("variant"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "action_count": len(actions),
        "action_counts": dict(sorted(action_counts.items())),
        "target_json_chars": len(response),
        "prompt_chars": len(messages[0]["content"]),
        "extra_info": {
            "task_id": task.get("task_id"),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "max_steps": args.max_steps,
            "tool_schema_version": task.get("tool_schema_version"),
        },
    }


def sanitize_actions(task: dict[str, Any], actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make old oracle episodes consistent with the current phase-aware mask.

    Local caption evidence is visible from the task and can be selected directly.
    Opening it after multiple corpus evidence chunks often happens after the
    mask has moved into the claim-writing phase, so it is redundant and blocked.
    """

    local_ids = {str(item.get("evidence_id")) for item in task.get("local_evidence") or [] if item.get("evidence_id")}
    selected_ids: set[str] = set()
    sanitized: list[dict[str, Any]] = []
    for action in actions:
        if not isinstance(action, dict):
            continue
        name = action.get("action")
        if name == "select_evidence":
            selected_ids.update(str(item) for item in action.get("evidence_ids") or [])
            sanitized.append(action)
            continue
        if name == "open_evidence" and str(action.get("evidence_id")) in local_ids and str(action.get("evidence_id")) in selected_ids:
            continue
        sanitized.append(action)
    return sanitized


def validate_records(split: str, records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    limit = min(len(records), max(0, int(args.validate_limit)))
    if limit <= 0:
        return {}
    out_dir = args.output_dir / "validation_rollouts" / split
    env = EvidenceAgentEnv(
        args.source_dir / "tasks_all.jsonl",
        args.evidence_index,
        out_dir,
        max_steps=args.max_steps,
        phase_aware_mask=args.phase_aware_mask,
        enforce_tool_mask=args.enforce_tool_mask,
    )
    metrics: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in records[:limit]:
        actions = json.loads(row["response"])
        obs = env.reset(task_id=str(row["task_id"]))
        terminated = False
        for index, action in enumerate(actions):
            obs, _, terminated, info = env.step(action)
            result = info.get("result") or {}
            if result.get("error"):
                failures.append(
                    {
                        "task_id": row["task_id"],
                        "step": index,
                        "action": action,
                        "error": result.get("error"),
                    }
                )
                break
            if terminated:
                break
        metrics.append(env.trajectory_metrics())
    success_count = sum(1 for item in metrics if item.get("trajectory_success"))
    finish_count = sum(1 for item in metrics if item.get("finish"))
    crop_count = sum(1 for item in metrics if item.get("crop_success"))
    return {
        "validated_rows": limit,
        "trajectory_success_rate": round(success_count / max(1, limit), 6),
        "finish_rate": round(finish_count / max(1, limit), 6),
        "crop_success_rate": round(crop_count / max(1, limit), 6),
        "final_reward_mean": round(mean([float(item.get("final_reward", 0.0)) for item in metrics]), 6),
        "evidence_recall_mean": round(mean([float(item.get("evidence_recall", 0.0)) for item in metrics]), 6),
        "claim_supported_rate_mean": round(mean([float(item.get("claim_supported_rate", 0.0)) for item in metrics]), 6),
        "failures": failures[:20],
        "failure_count": len(failures),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    action_counts = Counter()
    action_lengths = []
    prompt_chars = []
    target_chars = []
    for row in records:
        action_lengths.append(int(row.get("action_count") or 0))
        prompt_chars.append(int(row.get("prompt_chars") or 0))
        target_chars.append(int(row.get("target_json_chars") or 0))
        action_counts.update(row.get("action_counts") or {})
    return {
        "rows": len(records),
        "action_counts": dict(sorted(action_counts.items())),
        "action_len_min": min(action_lengths) if action_lengths else 0,
        "action_len_max": max(action_lengths) if action_lengths else 0,
        "action_len_mean": round(mean(action_lengths), 3),
        "prompt_chars_mean": round(mean(prompt_chars), 3),
        "prompt_chars_p95": percentile(prompt_chars, 0.95),
        "target_json_chars_mean": round(mean(target_chars), 3),
        "target_json_chars_p95": percentile(target_chars, 0.95),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.6 One-Shot Trajectory-Array SFT 数据构建报告",
        "",
        f"时间：{manifest['created_at']} CST",
        "",
        "## 目标",
        "",
        "把 step-wise oracle episodes 合并成 task-level SFT 数据。每条样本输入页面图像和 compact prompt，输出完整 JSON action array，用于 trajectory-level GRPO 前的 warm start。",
        "",
        "## 路径",
        "",
        f"- source_dir：`{manifest['source_dir']}`",
        f"- output_dir：`{manifest['output_dir']}`",
        f"- evidence_index：`{manifest['evidence_index']}`",
        "",
        "## Split 统计",
        "",
        "| split | rows | action_len_mean | action_len_max | target_chars_mean | validation_success | validation_reward |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, info in manifest["splits"].items():
        stats = info.get("stats") or {}
        validation = info.get("validation") or {}
        lines.append(
            "| {split} | {rows} | {alm} | {almax} | {tmean} | {succ} | {reward} |".format(
                split=split,
                rows=info.get("rows", 0),
                alm=stats.get("action_len_mean", 0),
                almax=stats.get("action_len_max", 0),
                tmean=stats.get("target_json_chars_mean", 0),
                succ=validation.get("trajectory_success_rate", ""),
                reward=validation.get("final_reward_mean", ""),
            )
        )
    lines.extend(
        [
            "",
            "## 质量结论",
            "",
            "- 数据监督目标是完整 JSON 数组，不再是单步 JSON 对象。",
            "- 构建时使用真实 `EvidenceAgentEnv` 对前若干条样本进行 oracle 执行验证。",
            "- 如果后续 SFT 后模型仍输出单步 action，说明 task-level warm start 不够，需要增加训练步数或缩短 prompt/target。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_evidence_items(index_dir: Path) -> dict[str, dict[str, Any]]:
    path = index_dir / "corpus_chunks.jsonl"
    if not path.exists():
        return {}
    return {str(item.get("evidence_id")): item for item in read_jsonl(path) if item.get("evidence_id")}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def mean(values: list[float] | list[int]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def percentile(values: list[int], ratio: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * ratio))))
    return int(ordered[index])


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
