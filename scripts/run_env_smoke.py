#!/usr/bin/env python3
"""Smoke runner for the executable EvidenceGrounded tool-call environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env import EvidenceAgentEnv  # noqa: E402
from evidence_agent_env.data import read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248/tasks_all.jsonl",
    )
    parser.add_argument(
        "--episodes",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248/episodes/oracle_episodes.jsonl",
    )
    parser.add_argument(
        "--evidence-index",
        default="/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140",
    )
    parser.add_argument("--output-dir", default="outputs/env_smoke_20260601")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--include-gold-regions", action="store_true")
    parser.add_argument("--phase-aware-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-tool-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--tool-schema",
        choices=["highlighted_direct", "region", "evidence_select", "chunked_claim", "inspect_crop"],
        default="chunked_claim",
    )
    parser.add_argument("--oracle", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env = EvidenceAgentEnv(
        args.tasks,
        args.evidence_index,
        args.output_dir,
        max_steps=args.max_steps,
        include_gold_regions=args.include_gold_regions,
        phase_aware_mask=args.phase_aware_mask,
        enforce_tool_mask=args.enforce_tool_mask,
        tool_schema=args.tool_schema,
    )
    obs = env.reset(index=args.index, task_id=args.task_id)
    print(json.dumps({"reset": obs}, ensure_ascii=False, indent=2))
    if args.oracle:
        episode = find_episode(args.episodes, obs["task_id"])
        actions = episode.get("actions") or []
    else:
        actions = [{"action": "inspect_page"}, {"action": "finish"}]
    terminated = False
    for action in actions:
        obs, reward, terminated, info = env.step(action)
        print(json.dumps({"action": action, "reward": reward, "info": info}, ensure_ascii=False))
        if terminated:
            break
    out = Path(args.output_dir) / "trajectory.json"
    env.dump_trajectory(out)
    summary = {
        "task_id": obs["task_id"],
        "steps": env.step_count,
        "terminated": terminated,
        "total_reward": env.total_reward,
        "trajectory": str(out),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def find_episode(path: str, task_id: str) -> dict:
    for row in read_jsonl(path):
        if row.get("task_id") == task_id:
            return row
    raise KeyError(f"episode not found for task_id={task_id}")


if __name__ == "__main__":
    raise SystemExit(main())
