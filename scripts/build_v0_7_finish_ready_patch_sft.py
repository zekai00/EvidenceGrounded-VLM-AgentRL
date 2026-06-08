#!/usr/bin/env python3
"""Build v0.7 finish-ready SFT rows with explicit available_actions=['finish'].""" 

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402
from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation, build_prompt_text  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_7_inspect_crop_sft_20260605_2336"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-history-actions", type=int, default=6)
    parser.add_argument("--max-tool-results", type=int, default=5)
    parser.add_argument("--snippet-chars", type=int, default=120)
    parser.add_argument("--max-text-chars", type=int, default=10000)
    parser.add_argument("--head-text-chars", type=int, default=3000)
    parser.add_argument("--preview-rows", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(exist_ok=True)

    tasks = {str(task["task_id"]): task for task in read_jsonl(args.source_dir / "tasks_all.jsonl")}
    prompt_config = PromptConfig(
        max_history_actions=args.max_history_actions,
        max_tool_results=args.max_tool_results,
        max_evidence_per_result=2,
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
        "dataset_version": "v0.7_finish_ready_patch_sft",
        "source_dir": str(args.source_dir),
        "output_dir": str(output_dir),
        "prompt_config": prompt_config.__dict__,
        "splits": {},
    }
    all_rows: list[dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        source_path = args.source_dir / "sft" / f"{split}.jsonl"
        rows = [make_finish_row(row, tasks[str(row["task_id"])], split, prompt_config) for row in read_jsonl(source_path) if is_finish_row(row)]
        all_rows.extend(rows)
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", rows)
        write_jsonl(output_dir / f"{split}_preview.jsonl", rows[: args.preview_rows])
        manifest["splits"][split] = summarize(rows, output_dir / "sft" / f"{split}.jsonl")
    write_jsonl(output_dir / "sft" / "all.jsonl", all_rows)
    manifest["all"] = summarize(all_rows, output_dir / "sft" / "all.jsonl")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def is_finish_row(row: dict[str, Any]) -> bool:
    return str((row.get("action") or {}).get("action")) == "finish"


def make_finish_row(
    row: dict[str, Any],
    task: dict[str, Any],
    split: str,
    prompt_config: PromptConfig,
) -> dict[str, Any]:
    obs = {
        "task_id": row.get("task_id"),
        "goal": task.get("goal"),
        "step": row.get("step"),
        "source_file": task.get("source_file") or meta_from_prompt(row.get("prompt_text", "")).get("source_file", ""),
        "page": task.get("page") or meta_from_prompt(row.get("prompt_text", "")).get("page"),
        "images": [{"role": "page_image" if index == 0 else "last_crop", "path": path} for index, path in enumerate(row.get("images") or [])],
        "history": row.get("history") or [],
        "tool_results": row.get("tool_results") or [],
        "draft_claims": row.get("draft_claims") or [],
        "claim_state": row.get("claim_state") or {},
        "selected_evidence_ids": row.get("selected_evidence_ids") or [],
        "visible_evidence_ids": row.get("visible_evidence_ids") or [],
        "available_actions": ["finish"],
        "tool_mask": {
            "enabled": True,
            "phase": "finish_ready",
            "allowed_actions": ["finish"],
            "blocked_actions": [
                "inspect_page",
                "select_evidence",
                "crop_target",
                "retrieve_evidence",
                "open_evidence",
                "write_claim",
                "abstain_claim",
                "write_claims_chunk",
                "write_claims_batch",
            ],
            "reason": "claim_state has no remaining fields; finish the trajectory.",
            "step": len(row.get("history") or []),
            "tool_schema": "inspect_crop",
        },
        "tool_schema": "inspect_crop",
    }
    action = {"action": "finish", "status": "done"}
    new_row = copy.deepcopy(row)
    new_row.update(
        {
            "split": split,
            "tool_schema_version": "v0.7_inspect_crop_finish_ready_patch",
            "action": action,
            "available_actions": ["finish"],
            "tool_mask": obs["tool_mask"],
            "prompt_text": build_prompt_text(obs, prompt_config),
            "messages": build_messages_from_observation(obs, prompt_config, include_assistant_action=action),
            "label_source": "v0_7_finish_ready_patch_sft",
        }
    )
    return new_row


def meta_from_prompt(text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in ["source_file", "page"]:
        marker = f"{key}："
        if marker in text:
            meta[key] = text.split(marker, 1)[1].split("；", 1)[0].split("\n", 1)[0].strip()
    return meta


def summarize(rows: list[dict[str, Any]], path: Path) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "path": str(path),
        "action_counts": dict(Counter(str((row.get("action") or {}).get("action")) for row in rows)),
        "phase_counts": dict(Counter(str((row.get("tool_mask") or {}).get("phase")) for row in rows)),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.7 Finish-Ready Patch SFT 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- source_dir: `{manifest['source_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "这批数据只修一个问题：当 `available_actions` 只有 `finish` 且 phase 为 `finish_ready` 时，模型必须输出 `{\"action\":\"finish\",\"status\":\"done\"}`，不能继续写 claim 或打开证据。",
        "",
        "## Split",
    ]
    for split, info in manifest["splits"].items():
        lines.append(f"- {split}: {info['rows']} rows, actions={info['action_counts']}, phases={info['phase_counts']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_dir() -> Path:
    return Path(f"/root/datasets/evidence_grounded_vlm_agentrl/v0_7_finish_ready_patch_sft_{datetime.now().strftime('%Y%m%d_%H%M')}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
