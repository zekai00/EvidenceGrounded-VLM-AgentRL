#!/usr/bin/env python3
"""Build SFT patch data for evidence-opening loops in stepwise rollouts."""

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

from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation  # noqa: E402
from evidence_agent_env.tool_mask import phase_aware_tool_mask  # noqa: E402
from evidence_agent_env.tools.claim_tools import apply_claim_write, claim_state  # noqa: E402


DEFAULT_TASKS = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_6_chunked_claim_sft_20260604_1650/tasks_all.jsonl"
)
DEFAULT_BASE_SFT_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_6_chunked_claim_sft_20260604_1650/sft"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", required=True, help="Comma-separated rollout JSONL files.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--base-sft-dir", type=Path, default=DEFAULT_BASE_SFT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay-train-rows", type=int, default=800)
    parser.add_argument("--replay-val-rows", type=int, default=128)
    parser.add_argument("--replay-test-rows", type=int, default=128)
    parser.add_argument("--patch-oversample", type=int, default=12)
    parser.add_argument("--max-patches-per-task", type=int, default=12)
    parser.add_argument("--claim-chunk-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = {str(row["task_id"]): row for row in read_jsonl(args.tasks)}
    patch_rows, patch_stats = build_patch_rows(args, tasks)

    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.6_evidence_open_loop_patch",
        "rollouts": [str(path) for path in paths_from_csv(args.rollouts)],
        "tasks": str(args.tasks),
        "base_sft_dir": str(args.base_sft_dir),
        "output_dir": str(args.output_dir),
        "replay_train_rows": args.replay_train_rows,
        "replay_val_rows": args.replay_val_rows,
        "replay_test_rows": args.replay_test_rows,
        "patch_oversample": args.patch_oversample,
        "max_patches_per_task": args.max_patches_per_task,
        "claim_chunk_size": args.claim_chunk_size,
        "seed": args.seed,
        "patch_stats": patch_stats,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        replay_limit = {
            "train": args.replay_train_rows,
            "val": args.replay_val_rows,
            "test": args.replay_test_rows,
        }[split]
        rows = sample_replay(args.base_sft_dir / f"{split}.jsonl", replay_limit, args.seed)
        if split == "train":
            for row in patch_rows:
                for copy_index in range(max(1, args.patch_oversample)):
                    copied = copy.deepcopy(row)
                    copied["patch_source"]["copy_index"] = copy_index
                    copied["patch_source"]["copies"] = max(1, args.patch_oversample)
                    rows.append(copied)
        random.shuffle(rows)
        out_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        manifest["splits"][split] = {
            "file": str(out_path),
            "rows_written": len(rows),
            "replay_rows": replay_limit,
            "patch_rows_before_oversample": len(patch_rows) if split == "train" else 0,
            "action_counts": dict(sorted(action_counts(rows).items())),
        }
        print(f"[{split}] wrote {len(rows)} rows -> {out_path}")

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "构建报告.md", manifest)
    print(f"manifest -> {manifest_path}")
    return 0


def build_patch_rows(args: argparse.Namespace, tasks: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    per_task: Counter[str] = Counter()
    prompt_config = PromptConfig(
        max_history_actions=8,
        max_tool_results=6,
        max_evidence_per_result=3,
        snippet_chars=180,
        max_text_chars=24000,
        head_text_chars=5000,
        coordinate_info=True,
        tool_schema="chunked_claim",
        compact_claim_state=True,
        region_selection_hint=True,
        strict_claim_phase_hint=True,
    )
    for rollout_path in paths_from_csv(args.rollouts):
        for record in read_jsonl(rollout_path):
            task_id = str(record.get("task_id", ""))
            task = tasks.get(task_id)
            if not task:
                stats["missing_tasks"] += 1
                continue
            state = RuntimeState(task)
            for step in record.get("steps") or []:
                pred = step.get("parsed_action")
                pred_name = str(pred.get("action", "")) if isinstance(pred, dict) else "invalid"
                phase = str((step.get("tool_mask") or {}).get("phase", ""))
                result = step.get("result") or {}
                if should_patch_bad_phase_action(pred_name, phase, result, state):
                    stats["bad_phase_actions_seen"] += 1
                    if per_task[task_id] < max(1, args.max_patches_per_task):
                        target = build_target_action(task, state, phase, args.claim_chunk_size)
                        obs = state.to_observation()
                        mask = phase_aware_tool_mask(obs)
                        obs["available_actions"] = mask["allowed_actions"]
                        obs["tool_mask"] = mask
                        messages = build_messages_from_observation(
                            obs,
                            prompt_config,
                            include_assistant_action=target,
                        )
                        rows.append(make_sft_row(task, state, step, messages, target, rollout_path))
                        per_task[task_id] += 1
                        stats[f"target::{target.get('action')}"] += 1
                    else:
                        stats["skipped_max_patches_per_task"] += 1
                state.apply_step(step)
    stats["patch_rows"] = len(rows)
    stats["tasks_with_patches"] = len(per_task)
    return rows, dict(sorted(stats.items()))


def should_patch_bad_phase_action(pred_name: str, phase: str, result: dict[str, Any], state: "RuntimeState") -> bool:
    if not state.has_crop():
        return False
    if phase in {"claim_ready", "claim_continuation"} and pred_name in {
        "open_evidence",
        "retrieve_evidence",
        "select_evidence",
        "finish",
        "invalid",
    }:
        return True
    if pred_name != "open_evidence":
        return False
    if phase in {"evidence_retrieval", "evidence_retrieval_after_open", "evidence_opening", "claim_ready"}:
        return True
    return bool(result.get("error"))


def build_target_action(task: dict[str, Any], state: "RuntimeState", phase: str, chunk_size: int) -> dict[str, Any]:
    if phase in {"claim_ready", "claim_continuation"}:
        return build_claim_chunk_action(task, state.draft_claims, chunk_size)
    retrieved_count = state.action_count("retrieve_evidence")
    opened_count = state.action_count("open_evidence")
    if retrieved_count >= 1 and opened_count >= 2:
        return build_claim_chunk_action(task, state.draft_claims, chunk_size)
    return build_retrieve_action(task, state)


def build_retrieve_action(task: dict[str, Any], state: "RuntimeState") -> dict[str, Any]:
    caption = first_caption_text(task)
    title = first_gold_value(task, "depicted_work_title")
    artist = first_gold_value(task, "artist")
    query_parts = [part for part in [caption, title, artist, "山水画 图注 作品"] if part]
    bbox = state.last_crop_bbox() or (task.get("gold") or {}).get("bbox") or [0, 0, 1, 1]
    scope = "current_page" if state.action_count("retrieve_evidence") == 0 else "same_document"
    return {
        "action": "retrieve_evidence",
        "query": " ".join(str(part) for part in query_parts)[:240],
        "scope": scope,
        "anchor": {
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "bbox": bbox,
        },
        "top_k": 5,
    }


def build_claim_chunk_action(task: dict[str, Any], draft_claims: list[dict[str, Any]], chunk_size: int) -> dict[str, Any]:
    current_state = claim_state(draft_claims)
    remaining = list(current_state.get("remaining_fields") or [])
    if not remaining:
        return {"action": "finish", "status": "done"}
    gold_by_field = {
        str(item.get("field")): item
        for item in (task.get("gold") or {}).get("claims", [])
        if isinstance(item, dict) and item.get("field")
    }
    claims: list[dict[str, Any]] = []
    abstains: list[dict[str, Any]] = []
    for field in remaining[: max(1, chunk_size)]:
        gold = gold_by_field.get(str(field))
        if gold and not gold.get("abstain") and gold.get("value") is not None:
            evidence_ids = gold.get("evidence_ids") or gold.get("candidate_evidence_ids") or []
            claims.append(
                {
                    "field": field,
                    "value": gold.get("value"),
                    "evidence_ids": list(evidence_ids),
                    "visual_bbox": gold.get("visual_bbox"),
                    "confidence": float(gold.get("confidence", 0.8) or 0.8),
                }
            )
        else:
            abstains.append({"field": field, "reason": str(gold.get("reason") if gold else "当前证据不足，暂不写入该字段")})
    return {"action": "write_claims_chunk", "claims": claims, "abstains": abstains}


def make_sft_row(
    task: dict[str, Any],
    state: "RuntimeState",
    step: dict[str, Any],
    messages: list[dict[str, Any]],
    action: dict[str, Any],
    rollout_path: Path,
) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "source_task_id": task.get("task_id"),
        "split": "train",
        "step": int(step.get("step", 0) or 0),
        "variant": 0,
        "label_source": "rollout_evidence_open_loop_patch",
        "tool_schema_version": "v0.6_evidence_open_loop_patch",
        "images": [item.get("path") for item in state.images() if item.get("path")],
        "history": list(state.history),
        "tool_results": list(state.tool_results),
        "draft_claims": list(state.draft_claims),
        "claim_state": claim_state(state.draft_claims),
        "selected_evidence_ids": list(state.selected_evidence_ids),
        "messages": messages,
        "prompt_text": first_user_text(messages),
        "action": action,
        "patch_source": {
            "kind": "v0_6_evidence_open_loop_state",
            "rollout": str(rollout_path),
            "failed_step": int(step.get("step", 0) or 0),
            "failed_action": step.get("parsed_action"),
            "mask_phase": (step.get("tool_mask") or {}).get("phase"),
            "mask_reason": (step.get("tool_mask") or {}).get("reason"),
            "copy_index": 0,
            "copies": 1,
        },
    }


class RuntimeState:
    def __init__(self, task: dict[str, Any]) -> None:
        self.task = task
        self.history: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.draft_claims: list[dict[str, Any]] = []
        self.selected_evidence_ids: list[str] = []
        self.last_crop: str | None = None

    def images(self) -> list[dict[str, str]]:
        images = [{"role": "page_image", "path": str(self.task.get("page_image", ""))}]
        if self.last_crop:
            images.append({"role": "last_crop", "path": self.last_crop})
        return images

    def to_observation(self) -> dict[str, Any]:
        return {
            "task_id": self.task.get("task_id"),
            "goal": self.task.get("goal"),
            "source_file": self.task.get("source_file"),
            "page": self.task.get("page"),
            "images": self.images(),
            "history": list(self.history),
            "tool_results": list(self.tool_results),
            "draft_claims": list(self.draft_claims),
            "claim_state": claim_state(self.draft_claims),
            "selected_evidence_ids": list(self.selected_evidence_ids),
            "visible_evidence_ids": sorted(self.visible_evidence_ids()),
        }

    def apply_step(self, step: dict[str, Any]) -> None:
        action = step.get("parsed_action")
        if isinstance(action, dict):
            self.history.append(action)
        else:
            self.history.append({"raw": step.get("raw_text", "")})
        result = step.get("result") or {}
        self.tool_results.append(result)
        if result.get("tool") in {"crop_region", "crop_image"} and result.get("crop_path"):
            self.last_crop = str(result.get("crop_path"))
        if result.get("tool") == "select_evidence":
            for evidence_id in result.get("selected_evidence_ids") or []:
                evidence_id = str(evidence_id)
                if evidence_id not in self.selected_evidence_ids:
                    self.selected_evidence_ids.append(evidence_id)
        if result.get("tool") in {"write_claims_chunk", "write_claims_batch"}:
            self.draft_claims = apply_claim_write(
                self.draft_claims,
                claims=result.get("claims") or [],
                abstains=result.get("abstains") or [],
            )

    def has_crop(self) -> bool:
        return any(result.get("tool") in {"crop_region", "crop_image"} for result in self.tool_results if isinstance(result, dict))

    def action_count(self, name: str) -> int:
        return sum(1 for action in self.history if isinstance(action, dict) and action.get("action") == name)

    def last_crop_bbox(self) -> list[Any] | None:
        for result in reversed(self.tool_results):
            if isinstance(result, dict) and result.get("tool") in {"crop_region", "crop_image"} and result.get("bbox"):
                return result.get("bbox")
        return None

    def visible_evidence_ids(self) -> set[str]:
        ids: set[str] = set()
        for item in self.task.get("local_evidence") or []:
            if item.get("evidence_id"):
                ids.add(str(item["evidence_id"]))
        for result in self.tool_results:
            if not isinstance(result, dict):
                continue
            if result.get("tool") == "propose_regions":
                for region in result.get("regions") or []:
                    if isinstance(region, dict) and region.get("caption_evidence_id"):
                        ids.add(str(region["caption_evidence_id"]))
            if result.get("tool") == "retrieve_evidence":
                for item in result.get("results") or []:
                    if isinstance(item, dict) and item.get("evidence_id"):
                        ids.add(str(item["evidence_id"]))
            if result.get("tool") == "open_evidence" and result.get("evidence_id") and not result.get("error"):
                ids.add(str(result["evidence_id"]))
        return ids


def first_caption_text(task: dict[str, Any]) -> str:
    for item in task.get("local_evidence") or []:
        text = item.get("display_snippet") or item.get("evidence_summary") or item.get("text")
        if text:
            return str(text)
    return str(task.get("goal") or "")


def first_gold_value(task: dict[str, Any], field: str) -> str:
    for claim in (task.get("gold") or {}).get("claims", []) or []:
        if isinstance(claim, dict) and claim.get("field") == field and claim.get("value"):
            return str(claim.get("value"))
    return ""


def first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
    return ""


def paths_from_csv(value: str) -> list[Path]:
    return [Path(item.strip()) for item in str(value).split(",") if item.strip()]


def sample_replay(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if limit <= 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, limit)


def action_name(row: dict[str, Any]) -> str:
    action = row.get("action")
    return str(action.get("action", "")) if isinstance(action, dict) else ""


def action_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(action_name(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.6 Evidence Open Loop Patch SFT 构建报告",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## 目标",
        "",
        "修复 stepwise rollout 中 crop 后反复调用 `open_evidence`、编造 evidence id、迟迟不进入 `retrieve_evidence` 或 `write_claims_chunk` 的问题。",
        "",
        "## 数据",
        "",
        f"- 输出：`{manifest['output_dir']}`",
        f"- 来源 rollouts：`{manifest['rollouts']}`",
        f"- replay：`{manifest['base_sft_dir']}`",
        "",
        "## Patch 统计",
        "",
        "```json",
        json.dumps(manifest["patch_stats"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## Split 统计",
        "",
        "| split | rows | patch rows | action counts |",
        "|---|---:|---:|---|",
    ]
    for split, stats in manifest["splits"].items():
        lines.append(
            f"| {split} | {stats['rows_written']} | {stats['patch_rows_before_oversample']} | "
            f"`{json.dumps(stats['action_counts'], ensure_ascii=False)}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
