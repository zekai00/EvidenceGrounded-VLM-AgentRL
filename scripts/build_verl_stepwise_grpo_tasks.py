#!/usr/bin/env python3
"""Build verl parquet data for executable stepwise EvidenceGrounded GRPO."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env import EvidenceAgentEnv  # noqa: E402
from evidence_agent_env.data import read_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig, build_prompt_text  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--max-train", type=int, default=16)
    parser.add_argument("--max-val", type=int, default=4)
    parser.add_argument("--max-test", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--image-max-pixels", type=int, default=131072)
    parser.add_argument("--prompt-max-text-chars", type=int, default=12000)
    parser.add_argument("--prompt-head-text-chars", type=int, default=3000)
    parser.add_argument("--snippet-chars", type=int, default=160)
    parser.add_argument(
        "--target-claim-fields",
        default="",
        help="Comma-separated fields required before finish is allowed. Empty means the default 12-field claim card.",
    )
    parser.add_argument(
        "--tool-schema",
        choices=["highlighted_direct", "region", "evidence_select", "chunked_claim", "inspect_crop"],
        default="chunked_claim",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(args.tasks)
    env = EvidenceAgentEnv(
        args.tasks,
        args.evidence_index,
        out_dir / "_env_preview",
        max_steps=args.max_steps,
        include_gold_regions=False,
        phase_aware_mask=True,
        enforce_tool_mask=True,
        tool_schema=args.tool_schema,
        target_claim_fields=parse_target_claim_fields(args.target_claim_fields),
    )
    prompt_config = PromptConfig(
        max_history_actions=8,
        max_tool_results=6,
        max_evidence_per_result=3,
        snippet_chars=args.snippet_chars,
        max_text_chars=args.prompt_max_text_chars,
        head_text_chars=args.prompt_head_text_chars,
        coordinate_info=True,
        tool_schema=args.tool_schema,
        compact_claim_state=True,
        region_selection_hint=True,
        strict_claim_phase_hint=True,
    )

    split_limits = {"train": args.max_train, "val": args.max_val, "test": args.max_test}
    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.6_stepwise_executable_grpo",
        "tasks": args.tasks,
        "evidence_index": args.evidence_index,
        "output_dir": str(out_dir),
        "max_steps": args.max_steps,
        "image_max_pixels": args.image_max_pixels,
        "prompt_config": prompt_config.__dict__,
        "target_claim_fields": parse_target_claim_fields(args.target_claim_fields),
        "tool_schema": args.tool_schema,
        "splits": {},
    }

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        rows = [task for task in tasks if str(task.get("split")) == split]
        limit = split_limits.get(split, -1)
        if limit and limit > 0:
            rows = rows[:limit]
        records = [build_record(task, env, prompt_config, args) for task in rows]
        parquet_path = out_dir / f"{split}.parquet"
        preview_path = out_dir / f"{split}_preview.jsonl"
        pd.DataFrame(records).to_parquet(parquet_path, index=False)
        write_jsonl(preview_path, records[:5])
        manifest["splits"][split] = {
            "rows": len(records),
            "parquet": str(parquet_path),
            "preview": str(preview_path),
        }

    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(out_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_record(
    task: dict[str, Any],
    env: EvidenceAgentEnv,
    prompt_config: PromptConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    obs = env.reset(task_id=str(task.get("task_id")))
    prompt_text = build_prompt_text(obs, prompt_config)
    ground_truth = {
        "task_id": task.get("task_id"),
        "tasks_path": args.tasks,
        "evidence_index": args.evidence_index,
        "max_steps": args.max_steps,
        "phase_aware_mask": True,
        "enforce_tool_mask": True,
        "tool_schema": args.tool_schema,
        "target_claim_fields": parse_target_claim_fields(args.target_claim_fields),
        "reward_mode": "shaped",
    }
    return {
        "data_source": "evidence_grounded_stepwise",
        "prompt": [{"role": "user", "content": "<image>\n" + prompt_text}],
        "images": [{"image": task.get("page_image"), "max_pixels": args.image_max_pixels}],
        "reward_model": {"style": "rule", "ground_truth": json.dumps(ground_truth, ensure_ascii=False)},
        "extra_info": {
            "index": task.get("task_id"),
            "task_id": task.get("task_id"),
            "split": task.get("split"),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "agent_name": "evidence_stepwise_agent",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_target_claim_fields(value: str) -> list[str] | None:
    fields = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return fields or None


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.6 Stepwise Executable GRPO 数据构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- tasks: `{manifest['tasks']}`",
        f"- evidence_index: `{manifest['evidence_index']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- max_steps: {manifest['max_steps']}",
        "",
        "## Splits",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['rows']} rows, parquet=`{info['parquet']}`")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "这版数据不是 one-shot action array，而是给 verl 自定义 agent loop 的初始任务输入。",
            "rollout 时模型每一步只输出一个 JSON action；本地 EvidenceAgentEnv 执行动作并把新状态追加回上下文。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
