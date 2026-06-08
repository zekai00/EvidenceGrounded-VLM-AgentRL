#!/usr/bin/env python3
"""Build v0.4.2 SFT data with write_claims_batch actions."""

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
from build_agentbench_v0_4_1_claim_schema import make_sft_row  # noqa: E402


DEFAULT_SOURCE_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_4_1_region_claim_schema_sft_20260601_1458"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "sft").mkdir(exist_ok=True)
    (output_dir / "episodes").mkdir(exist_ok=True)

    tasks = read_jsonl(source_dir / "tasks_all.jsonl")
    sft_by_task = load_sft_by_task(source_dir)

    new_tasks: list[dict[str, Any]] = []
    new_rows: list[dict[str, Any]] = []
    new_episodes: list[dict[str, Any]] = []
    for task in tasks:
        new_task = copy.deepcopy(task)
        new_task["dataset_version"] = "v0.4.2_batch_claims"
        new_task["tool_schema_version"] = "v0.4.2_batch_claims"
        new_task["batch_claims_enabled"] = True
        rows, actions = rebuild_task_rows(new_task, sft_by_task[str(task["task_id"])])
        new_tasks.append(new_task)
        new_rows.extend(rows)
        new_episodes.append(
            {
                "task_id": new_task["task_id"],
                "source_task_id": new_task.get("source_task_id"),
                "split": new_task.get("split"),
                "variant": new_task.get("candidate_augmentation", {}).get("variant"),
                "actions": actions,
            }
        )

    write_outputs(output_dir, new_tasks, new_episodes, new_rows)
    quality = summarize(new_tasks, new_rows, new_episodes, source_dir)
    manifest = {
        "created_at": now(),
        "dataset_version": "v0.4.2_batch_claims",
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "quality": quality,
        "files": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "sft_train": str(output_dir / "sft" / "train.jsonl"),
            "sft_val": str(output_dir / "sft" / "val.jsonl"),
            "sft_test": str(output_dir / "sft" / "test.jsonl"),
            "oracle_episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
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


def rebuild_task_rows(task: dict[str, Any], old_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_config = PromptConfig(tool_schema="region", coordinate_info=True)
    prefix_rows: list[dict[str, Any]] = []
    first_claim_row: dict[str, Any] | None = None
    claim_actions: list[dict[str, Any]] = []
    for row in old_rows:
        action = copy.deepcopy(row.get("action") or {})
        name = action.get("action")
        if name in {"write_claim", "abstain_claim"}:
            if first_claim_row is None:
                first_claim_row = row
            claim_actions.append(action)
        elif name == "finish":
            break
        elif first_claim_row is None:
            prefix_rows.append(row)
    if first_claim_row is None:
        raise ValueError(f"no claim actions for task {task.get('task_id')}")

    new_rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for row in prefix_rows:
        action = copy.deepcopy(row.get("action") or {})
        new_rows.append(retag_row(make_sft_row(task, row, action, prompt_config, len(new_rows))))
        actions.append(action)

    state = {
        "history": copy.deepcopy(first_claim_row.get("history") or []),
        "tool_results": copy.deepcopy(first_claim_row.get("tool_results") or []),
        "draft_claims": copy.deepcopy(first_claim_row.get("draft_claims") or []),
        "images": copy.deepcopy(first_claim_row.get("images") or []),
    }
    state["history"] = copy.deepcopy(actions)
    batch = build_batch_action(claim_actions)
    batch_state = {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "step": len(new_rows),
        "history": copy.deepcopy(state["history"]),
        "tool_results": copy.deepcopy(state["tool_results"]),
        "draft_claims": copy.deepcopy(state["draft_claims"]),
        "images": copy.deepcopy(state["images"]),
    }
    new_rows.append(retag_row(make_sft_row(task, batch_state, batch, prompt_config, len(new_rows))))
    actions.append(batch)
    apply_batch_action(state, batch)

    finish = {"action": "finish", "status": "done"}
    finish_state = {
        "task_id": task["task_id"],
        "split": task.get("split"),
        "step": len(new_rows),
        "history": copy.deepcopy(state["history"]),
        "tool_results": copy.deepcopy(state["tool_results"]),
        "draft_claims": copy.deepcopy(state["draft_claims"]),
        "images": copy.deepcopy(state["images"]),
    }
    new_rows.append(retag_row(make_sft_row(task, finish_state, finish, prompt_config, len(new_rows))))
    actions.append(finish)
    return new_rows, actions


def retag_row(row: dict[str, Any]) -> dict[str, Any]:
    row["tool_schema_version"] = "v0.4.2_batch_claims"
    row["label_source"] = "v0_4_2_batch_claims"
    return row


def build_batch_action(actions: list[dict[str, Any]]) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    abstains: list[dict[str, Any]] = []
    for action in actions:
        if action.get("action") == "write_claim":
            claims.append(
                {
                    "field": action.get("field"),
                    "value": action.get("value"),
                    "evidence_ids": action.get("evidence_ids") or [],
                    "visual_bbox": action.get("visual_bbox"),
                    "confidence": action.get("confidence", 0.85),
                }
            )
        elif action.get("action") == "abstain_claim":
            abstains.append({"field": action.get("field"), "reason": action.get("reason", "证据不足")})
    return {"action": "write_claims_batch", "claims": claims, "abstains": abstains}


def apply_batch_action(state: dict[str, Any], action: dict[str, Any]) -> None:
    state["history"].append(copy.deepcopy(action))
    claims: list[dict[str, Any]] = []
    abstains: list[dict[str, Any]] = []
    for item in action.get("claims") or []:
        claim = {
            "field": item.get("field"),
            "value": item.get("value"),
            "evidence_ids": item.get("evidence_ids") or [],
            "visual_bbox": item.get("visual_bbox"),
            "confidence": item.get("confidence"),
            "abstain": False,
        }
        upsert_claim(state["draft_claims"], claim)
        claims.append(claim)
    for item in action.get("abstains") or []:
        claim = {"field": item.get("field"), "reason": item.get("reason"), "abstain": True}
        upsert_claim(state["draft_claims"], claim)
        abstains.append(claim)
    state["tool_results"].append({"tool": "write_claims_batch", "claims": claims, "abstains": abstains})


def upsert_claim(claims: list[dict[str, Any]], claim: dict[str, Any]) -> None:
    field = claim.get("field")
    claims[:] = [item for item in claims if item.get("field") != field]
    claims.append(claim)


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
) -> dict[str, Any]:
    action_counts = Counter(str((row.get("action") or {}).get("action")) for row in rows)
    old_rows = 0
    old_steps: Counter[str] = Counter()
    old_action_counts: Counter[str] = Counter()
    source_all = source_dir / "sft" / "all.jsonl"
    if source_all.exists():
        for row in read_jsonl(source_all):
            old_rows += 1
            old_steps[str(row["task_id"])] += 1
            old_action_counts[str((row.get("action") or {}).get("action"))] += 1
    new_steps = Counter(str(row["task_id"]) for row in rows)
    batch_sizes = []
    abstain_sizes = []
    for episode in episodes:
        for action in episode.get("actions") or []:
            if action.get("action") == "write_claims_batch":
                batch_sizes.append(len(action.get("claims") or []))
                abstain_sizes.append(len(action.get("abstains") or []))
    return {
        "tasks_total": len(tasks),
        "task_split_counts": dict(Counter(str(task.get("split")) for task in tasks)),
        "sft_rows_total": len(rows),
        "sft_split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "sft_action_counts": dict(action_counts),
        "avg_steps": sum(new_steps.values()) / max(1, len(new_steps)),
        "min_steps": min(new_steps.values()) if new_steps else 0,
        "max_steps": max(new_steps.values()) if new_steps else 0,
        "source_sft_rows_total": old_rows,
        "source_avg_steps": sum(old_steps.values()) / max(1, len(old_steps)),
        "source_action_counts": dict(old_action_counts),
        "row_reduction": old_rows - len(rows),
        "row_reduction_rate": (old_rows - len(rows)) / max(1, old_rows),
        "batch_claim_mean": sum(batch_sizes) / max(1, len(batch_sizes)),
        "batch_abstain_mean": sum(abstain_sizes) / max(1, len(abstain_sizes)),
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    q = manifest["quality"]
    lines = [
        "# AgentBench v0.4.2 Batch Claims SFT 构建报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 目标",
        "",
        "将 v0.4.1 中连续的 `write_claim` / `abstain_claim` 单字段动作合并为一个 `write_claims_batch` 动作，降低轨迹长度和上下文压力。",
        "",
        "## 数据位置",
        "",
        "```text",
        manifest["output_dir"],
        "```",
        "",
        "## 规模变化",
        "",
        f"- source_sft_rows_total：{q['source_sft_rows_total']}",
        f"- sft_rows_total：{q['sft_rows_total']}",
        f"- row_reduction：{q['row_reduction']}",
        f"- row_reduction_rate：{q['row_reduction_rate']:.4f}",
        f"- source_avg_steps：{q['source_avg_steps']:.2f}",
        f"- avg_steps：{q['avg_steps']:.2f}",
        f"- min_steps/max_steps：{q['min_steps']} / {q['max_steps']}",
        "",
        "## 动作分布",
        "",
        f"- sft_action_counts：`{json.dumps(q['sft_action_counts'], ensure_ascii=False)}`",
        f"- batch_claim_mean：{q['batch_claim_mean']:.2f}",
        f"- batch_abstain_mean：{q['batch_abstain_mean']:.2f}",
        "",
        "## 注意事项",
        "",
        "- `write_claims_batch` 不跳过证据：所有 claim 引用的 evidence 仍要求在 batch 前打开。",
        "- batch 内每个 claim/abstain 在环境 verifier 中逐项计分。",
        "- 该版本更适合 trajectory SFT；如果做 step-wise RL，需要把 batch action 当成一个复合决策处理。",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_dir() -> Path:
    return Path(f"/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_4_2_batch_claims_sft_{datetime.now().strftime('%Y%m%d_%H%M')}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
