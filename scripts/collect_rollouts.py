#!/usr/bin/env python3
"""Collect executable rollouts with a Qwen-VL SFT adapter policy."""

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
from evidence_agent_env.data import read_jsonl  # noqa: E402
from evidence_agent_env.policy import QwenVLSftPolicy  # noqa: E402
from evidence_agent_env.prompting import PromptConfig  # noqa: E402


def bbox_iou(a: Any, b: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(x) for x in a]
        bx1, by1, bx2, by2 = [float(x) for x in b]
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument(
        "--evidence-index",
        default="/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument(
        "--adapter",
        default="outputs/evidence_sft_qwen25vl3b_lora_compact_v2_highlight360_20260531_0510/adapter",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-tasks", type=int, default=8)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--max-steps", type=int, default=16)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--max-seq-length", type=int, default=14336)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--tool-schema",
        choices=["highlighted_direct", "region", "evidence_select", "chunked_claim", "inspect_crop", "no_select"],
        default="evidence_select",
    )
    parser.add_argument("--max-history-actions", type=int, default=8)
    parser.add_argument("--max-tool-results", type=int, default=6)
    parser.add_argument("--max-evidence-per-result", type=int, default=3)
    parser.add_argument("--snippet-chars", type=int, default=180)
    parser.add_argument("--max-text-chars", type=int, default=24000)
    parser.add_argument("--head-text-chars", type=int, default=5000)
    parser.add_argument("--coordinate-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--region-selection-hint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add a non-answer-leaking phase hint before crop_region selection.",
    )
    parser.add_argument(
        "--strict-claim-phase-hint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add a stricter claim-writing phase hint after the tool mask enters claim_writing/claim_continuation.",
    )
    parser.add_argument(
        "--dynamic-tool-schema",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only show currently allowed tool formats in the prompt. Default is false because v0.7 dynamic-schema val8 was a negative result.",
    )
    parser.add_argument(
        "--compact-claim-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use compact claim_state instead of full claim details. Defaults to true for chunked_claim.",
    )
    parser.add_argument("--include-gold-regions", action="store_true")
    parser.add_argument("--phase-aware-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enforce-tool-mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--reward-mode",
        choices=["default", "field_policy_probe"],
        default="default",
        help="default keeps legacy reward; field_policy_probe weights cited/opened evidence and field-level support.",
    )
    parser.add_argument(
        "--field-policy-prompt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Add explicit field/evidence policy hints to rollout prompts and tool-result summaries.",
    )
    parser.add_argument(
        "--target-claim-fields",
        default="",
        help="Comma-separated fields required before finish is allowed. Empty means the default 12-field claim card.",
    )
    parser.add_argument(
        "--target-claim-fields-from-gold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When --target-claim-fields is empty, read the required fields from selected tasks' gold target_claim_fields/claim_schema_fields.",
    )
    parser.add_argument("--system-prompt", default="")
    parser.add_argument("--print-steps", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trajectories").mkdir(exist_ok=True)

    task_rows = read_jsonl(args.tasks)
    selected = select_tasks(task_rows, args)
    target_claim_fields = resolve_target_claim_fields(args, selected)
    prompt_config = PromptConfig(
        max_history_actions=args.max_history_actions,
        max_tool_results=args.max_tool_results,
        max_evidence_per_result=args.max_evidence_per_result,
        snippet_chars=args.snippet_chars,
        max_text_chars=args.max_text_chars,
        head_text_chars=args.head_text_chars,
        coordinate_info=args.coordinate_info,
        tool_schema=args.tool_schema,
        compact_claim_state=args.compact_claim_state
        if args.compact_claim_state is not None
        else args.tool_schema in {"chunked_claim", "inspect_crop", "no_select"},
        region_selection_hint=args.region_selection_hint,
        strict_claim_phase_hint=args.strict_claim_phase_hint,
        dynamic_tool_schema=args.dynamic_tool_schema,
        field_policy_prompt=args.field_policy_prompt,
    )
    policy = QwenVLSftPolicy(
        args.model,
        args.adapter,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=args.torch_dtype,
        image_max_pixels=args.image_max_pixels,
        max_seq_length=args.max_seq_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        system_prompt=args.system_prompt,
        prompt_config=prompt_config,
    )
    env = EvidenceAgentEnv(
        args.tasks,
        args.evidence_index,
        output_dir,
        max_steps=args.max_steps,
        include_gold_regions=args.include_gold_regions,
        phase_aware_mask=args.phase_aware_mask,
        enforce_tool_mask=args.enforce_tool_mask,
        tool_schema=args.tool_schema,
        target_claim_fields=target_claim_fields,
        reward_mode=args.reward_mode,
        field_policy_hints=args.field_policy_prompt,
    )

    rollout_records: list[dict[str, Any]] = []
    trajectories_path = output_dir / "rollouts.jsonl"
    with trajectories_path.open("w", encoding="utf-8") as f:
        for ordinal, task in enumerate(selected):
            record = run_one(env, policy, task, output_dir, ordinal, args)
            rollout_records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(progress_record(ordinal + 1, len(selected), rollout_records), ensure_ascii=False), flush=True)

    summary = build_summary(args, policy, rollout_records, trajectories_path, target_claim_fields)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(output_dir / "rollout_report.md", summary, rollout_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def select_tasks(tasks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.task_id:
        wanted = set(args.task_id)
        selected = [task for task in tasks if str(task.get("task_id")) in wanted]
    else:
        end = len(tasks) if args.max_tasks <= 0 else min(len(tasks), args.start_index + args.max_tasks)
        selected = tasks[args.start_index : end]
    if not selected:
        raise ValueError("no tasks selected")
    return selected


def parse_target_claim_fields(value: str) -> list[str] | None:
    fields = [item.strip() for item in str(value or "").split(",") if item.strip()]
    return fields or None


def resolve_target_claim_fields(args: argparse.Namespace, selected: list[dict[str, Any]]) -> list[str] | None:
    explicit_fields = parse_target_claim_fields(args.target_claim_fields)
    if explicit_fields:
        return explicit_fields
    if not args.target_claim_fields_from_gold:
        return None

    field_sets: list[tuple[str, ...]] = []
    missing_task_ids: list[str] = []
    for task in selected:
        fields = task_gold_target_claim_fields(task)
        if fields:
            field_sets.append(tuple(fields))
        else:
            missing_task_ids.append(str(task.get("task_id")))
    if missing_task_ids:
        raise ValueError(f"selected tasks are missing gold target claim fields: {missing_task_ids[:10]}")
    unique_field_sets = sorted(set(field_sets))
    if not unique_field_sets:
        raise ValueError("selected tasks do not contain gold target claim fields")
    if len(unique_field_sets) != 1:
        raise ValueError(f"selected tasks contain inconsistent gold target claim fields: {unique_field_sets}")
    return list(unique_field_sets[0])


def task_gold_target_claim_fields(task: dict[str, Any]) -> list[str]:
    gold = task.get("gold") or {}
    raw_fields = (
        gold.get("target_claim_fields")
        or gold.get("claim_schema_fields")
        or task.get("target_claim_fields")
        or task.get("claim_schema_fields")
        or []
    )
    return [str(item) for item in raw_fields if str(item)]


def run_one(
    env: EvidenceAgentEnv,
    policy: QwenVLSftPolicy,
    task: dict[str, Any],
    output_dir: Path,
    ordinal: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    obs = env.reset(task_id=str(task.get("task_id")))
    steps: list[dict[str, Any]] = []
    terminated = False
    for step_index in range(args.max_steps):
        tool_mask = obs.get("tool_mask") or {}
        available_actions = list(obs.get("available_actions") or [])
        prediction = policy.act(obs)
        action = prediction["action"] if prediction["action"] is not None else prediction["raw_text"]
        obs, reward, terminated, info = env.step(action)
        result = info.get("result") or {}
        executed_action = env.history[-1] if env.history else None
        parsed_action = executed_action if isinstance(executed_action, dict) and "action" in executed_action else None
        action_name = parsed_action.get("action", "invalid") if isinstance(parsed_action, dict) else "invalid"
        mask_violation = bool(available_actions and action_name not in set(available_actions))
        steps.append(
            {
                "step": step_index,
                "raw_text": prediction["raw_text"],
                "model_parsed_action": prediction["action"],
                "parsed_action": parsed_action,
                "available_actions": available_actions,
                "tool_mask": tool_mask,
                "mask_violation": mask_violation,
                "reward": reward,
                "result": result,
                "total_reward": info.get("total_reward"),
                "terminated": terminated,
            }
        )
        if args.print_steps:
            print(
                json.dumps(
                    {
                        "time": now(),
                        "task_id": task.get("task_id"),
                        "step": step_index,
                        "action": action_name,
                        "mask_phase": tool_mask.get("phase"),
                        "mask_violation": mask_violation,
                        "reward": reward,
                        "total_reward": info.get("total_reward"),
                        "terminated": terminated,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if terminated:
            break
    trajectory_path = output_dir / "trajectories" / f"{ordinal:04d}_{task.get('task_id')}.json"
    env.dump_trajectory(trajectory_path)
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "steps": steps,
        "num_steps": len(steps),
        "terminated": terminated,
        "total_reward": env.total_reward,
        "final_claims": env.draft_claims,
        "trajectory_metrics": env.trajectory_metrics(),
        "trajectory": str(trajectory_path),
        "metrics": {
            **summarize_one(steps, env.draft_claims, env.trajectory_metrics()),
            **process_diagnostics(task, steps),
            **region_diagnostics(task, steps),
        },
    }


def summarize_one(
    steps: list[dict[str, Any]],
    final_claims: list[dict[str, Any]],
    trajectory_metrics: dict[str, Any],
) -> dict[str, Any]:
    action_counts = Counter(
        str((step.get("parsed_action") or {}).get("action", "invalid")) if isinstance(step.get("parsed_action"), dict) else "invalid"
        for step in steps
    )
    mask_violations = sum(int(step.get("mask_violation", False)) for step in steps)
    schema_repair_steps = [step for step in steps if schema_repaired_keys(step)]
    schema_repair_key_counts: Counter[str] = Counter()
    for step in schema_repair_steps:
        schema_repair_key_counts.update(schema_repaired_keys(step))
    crop_ious = [
        float((step.get("result") or {}).get("bbox_iou"))
        for step in steps
        if isinstance((step.get("result") or {}).get("bbox_iou"), (int, float))
    ]
    evidence_hits = []
    for step in steps:
        result = step.get("result") or {}
        evidence_hits.extend(result.get("hit_evidence_ids") or [])
    verifier_evidence_hit_count = trajectory_metrics.get("evidence_hit_count")
    evidence_hit_count = (
        int(verifier_evidence_hit_count)
        if isinstance(verifier_evidence_hit_count, (int, float))
        else len(set(map(str, evidence_hits)))
    )
    return {
        "valid_json_steps": sum(1 for step in steps if isinstance(step.get("parsed_action"), dict)),
        "invalid_json_steps": sum(1 for step in steps if not isinstance(step.get("parsed_action"), dict)),
        "action_counts": dict(action_counts),
        "max_crop_iou": max(crop_ious) if crop_ious else None,
        "crop_success": bool(crop_ious and max(crop_ious) >= 0.5),
        "evidence_hit_count": evidence_hit_count,
        "claim_count": len(final_claims),
        "has_finish_action": action_counts.get("finish", 0) > 0,
        "has_finish": bool(trajectory_metrics.get("finish")),
        "premature_finish_count": int(trajectory_metrics.get("premature_finish_count", 0) or 0),
        "premature_finish_rate": float(trajectory_metrics.get("premature_finish_rate", 0.0) or 0.0),
        "mask_violation_count": mask_violations,
        "mask_violation_rate": mask_violations / max(1, len(steps)),
        "schema_repair_count": len(schema_repair_steps),
        "schema_repair_rate": len(schema_repair_steps) / max(1, len(steps)),
        "schema_repaired_key_counts": dict(schema_repair_key_counts),
        "trajectory_success": bool(trajectory_metrics.get("trajectory_success")),
        "final_reward": trajectory_metrics.get("final_reward"),
        "claim_supported_rate": trajectory_metrics.get("claim_supported_rate"),
        "claim_support_precision": trajectory_metrics.get("claim_support_precision"),
        "claim_support_recall": trajectory_metrics.get("claim_support_recall"),
        "claim_support_f1": trajectory_metrics.get("claim_support_f1"),
        "evidence_recall": trajectory_metrics.get("evidence_recall"),
        "evidence_precision": trajectory_metrics.get("evidence_precision"),
        "evidence_f1": trajectory_metrics.get("evidence_f1"),
        "retrieved_evidence_precision": trajectory_metrics.get("retrieved_evidence_precision"),
        "retrieved_evidence_recall": trajectory_metrics.get("retrieved_evidence_recall"),
        "retrieved_evidence_f1": trajectory_metrics.get("retrieved_evidence_f1"),
        "opened_evidence_precision": trajectory_metrics.get("opened_evidence_precision"),
        "opened_evidence_recall": trajectory_metrics.get("opened_evidence_recall"),
        "opened_evidence_f1": trajectory_metrics.get("opened_evidence_f1"),
        "cited_evidence_precision": trajectory_metrics.get("cited_evidence_precision"),
        "cited_evidence_recall": trajectory_metrics.get("cited_evidence_recall"),
        "cited_evidence_f1": trajectory_metrics.get("cited_evidence_f1"),
        "retrieved_evidence_count": trajectory_metrics.get("retrieved_evidence_count"),
        "opened_evidence_count": trajectory_metrics.get("opened_evidence_count"),
        "cited_evidence_count": trajectory_metrics.get("cited_evidence_count"),
        "retrieved_evidence_hit_count": trajectory_metrics.get("retrieved_evidence_hit_count"),
        "opened_evidence_hit_count": trajectory_metrics.get("opened_evidence_hit_count"),
        "cited_evidence_hit_count": trajectory_metrics.get("cited_evidence_hit_count"),
        "field_policy_selection_score": trajectory_metrics.get("field_policy_selection_score"),
        "invalid_step_rate": trajectory_metrics.get("invalid_step_rate"),
        "core_supported_count": trajectory_metrics.get("core_supported_count"),
        "core_supported_rate": trajectory_metrics.get("core_supported_rate"),
        "core_support_precision": trajectory_metrics.get("core_support_precision"),
        "core_support_recall": trajectory_metrics.get("core_support_recall"),
        "core_support_f1": trajectory_metrics.get("core_support_f1"),
        "abstain_precision": trajectory_metrics.get("abstain_precision"),
        "abstain_recall": trajectory_metrics.get("abstain_recall"),
        "abstain_f1": trajectory_metrics.get("abstain_f1"),
        "core_field_match_count": trajectory_metrics.get("core_field_match_count"),
        "core_field_recall": trajectory_metrics.get("core_field_recall"),
        "local_caption_only_claim_count": trajectory_metrics.get("local_caption_only_claim_count"),
        "local_caption_only_risk_field_claim_count": trajectory_metrics.get(
            "local_caption_only_risk_field_claim_count"
        ),
        "local_caption_only_unsupported_count": trajectory_metrics.get("local_caption_only_unsupported_count"),
    }


def schema_repaired_keys(step: dict[str, Any]) -> list[str]:
    action = step.get("parsed_action") or {}
    result = step.get("result") or {}
    keys: list[str] = []
    for source in [action, result]:
        if not isinstance(source, dict):
            continue
        for key in source.get("_schema_repaired_keys") or source.get("schema_repaired_keys") or []:
            key = str(key)
            if key and key not in keys:
                keys.append(key)
    return keys


def process_diagnostics(task: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    local_evidence_ids = {str(item.get("evidence_id")) for item in task.get("local_evidence") or []}
    retrieve_steps: list[int] = []
    write_steps: list[int] = []
    open_steps: list[int] = []
    local_open_steps: list[int] = []
    external_open_steps: list[int] = []
    negative_retrieve_count = 0
    negative_write_claim_count = 0
    finish_not_ready_count = 0
    premature_mask_finish = False
    claim_continuation_finish_count = 0

    for index, step in enumerate(steps):
        action = step.get("parsed_action") or {}
        action_name = str(action.get("action", "invalid")) if isinstance(action, dict) else "invalid"
        phase = str((step.get("tool_mask") or {}).get("phase") or "")
        reward = step.get("reward")
        if action_name == "retrieve_evidence":
            retrieve_steps.append(index)
            if isinstance(reward, (int, float)) and float(reward) < 0:
                negative_retrieve_count += 1
        if action_name in {"write_claim", "write_claims_chunk", "write_claims_batch", "abstain_claim"}:
            write_steps.append(index)
            if isinstance(reward, (int, float)) and float(reward) < 0:
                negative_write_claim_count += 1
        if action_name == "open_evidence":
            open_steps.append(index)
            evidence_id = str(action.get("evidence_id") or "")
            if evidence_id in local_evidence_ids or evidence_id.startswith("local_"):
                local_open_steps.append(index)
            else:
                external_open_steps.append(index)
        if action_name == "finish":
            if phase != "finish_ready":
                finish_not_ready_count += 1
            if phase == "claim_continuation":
                claim_continuation_finish_count += 1
            if step.get("mask_violation") or (step.get("result") or {}).get("error"):
                premature_mask_finish = True

    first_retrieve_step = min(retrieve_steps) if retrieve_steps else None
    first_write_step = min(write_steps) if write_steps else None
    external_after_retrieve = [
        step_index
        for step_index in external_open_steps
        if first_retrieve_step is not None and step_index > first_retrieve_step
    ]
    return {
        "retrieve_count": len(retrieve_steps),
        "open_evidence_count": len(open_steps),
        "local_open_count": len(local_open_steps),
        "external_open_count": len(external_open_steps),
        "external_open_after_retrieve_count": len(external_after_retrieve),
        "no_retrieve": len(retrieve_steps) == 0,
        "retrieve_without_external_open": bool(retrieve_steps) and not external_after_retrieve,
        "write_before_retrieve": bool(write_steps) and (first_retrieve_step is None or first_write_step < first_retrieve_step),
        "first_retrieve_step": first_retrieve_step,
        "first_write_step": first_write_step,
        "premature_mask_finish": premature_mask_finish,
        "finish_not_ready_count": finish_not_ready_count,
        "claim_continuation_finish_count": claim_continuation_finish_count,
        "negative_retrieve_count": negative_retrieve_count,
        "negative_write_claim_count": negative_write_claim_count,
    }


def region_diagnostics(task: dict[str, Any], steps: list[dict[str, Any]]) -> dict[str, Any]:
    gold_bbox = (task.get("gold") or {}).get("image_bbox")
    if not gold_bbox:
        return {
            "candidate_oracle_iou": None,
            "candidate_oracle_rank": None,
            "candidate_oracle_region_id": None,
            "selected_region_id": None,
            "selected_region_iou": None,
        }

    first_regions: list[dict[str, Any]] = []
    selected_region_id: str | None = None
    selected_region_bbox: Any = None
    selected_region_iou: float | None = None
    crop_region_called = False
    crop_region_error = None
    for step in steps:
        result = step.get("result") or {}
        if result.get("tool") in {"inspect_page", "propose_regions"} and result.get("regions") and not first_regions:
            first_regions = [item for item in (result.get("regions") or []) if isinstance(item, dict)]
        if result.get("tool") in {"crop_region", "crop_target"}:
            crop_region_called = True
            crop_region_error = result.get("error")
            selected_region_id = str(result.get("region_id")) if result.get("region_id") is not None else None
            selected_region_bbox = result.get("bbox")
            if isinstance(result.get("bbox_iou"), (int, float)):
                selected_region_iou = float(result["bbox_iou"])

    scored: list[tuple[float, int, dict[str, Any]]] = []
    for idx, region in enumerate(first_regions):
        scored.append((bbox_iou(region.get("bbox"), gold_bbox), idx + 1, region))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_iou, best_rank, best_region = (scored[0] if scored else (None, None, {}))

    if selected_region_iou is None and selected_region_bbox is not None:
        selected_region_iou = bbox_iou(selected_region_bbox, gold_bbox)

    return {
        "candidate_count": len(first_regions),
        "candidate_oracle_iou": best_iou,
        "candidate_oracle_rank": best_rank,
        "candidate_oracle_region_id": best_region.get("region_id") if best_region else None,
        "candidate_oracle_source": best_region.get("source") if best_region else None,
        "candidate_oracle_type": best_region.get("type") if best_region else None,
        "crop_region_called": crop_region_called,
        "crop_region_error": crop_region_error,
        "selected_region_id": selected_region_id,
        "selected_region_iou": selected_region_iou,
        "selected_region_hit_iou50": bool(selected_region_iou is not None and selected_region_iou >= 0.5),
        "selected_region_hit_iou70": bool(selected_region_iou is not None and selected_region_iou >= 0.7),
        "selected_matches_oracle_region": bool(
            selected_region_id is not None
            and best_region
            and str(best_region.get("region_id")) == str(selected_region_id)
        ),
    }


def build_summary(
    args: argparse.Namespace,
    policy: QwenVLSftPolicy,
    records: list[dict[str, Any]],
    trajectories_path: Path,
    target_claim_fields: list[str] | None,
) -> dict[str, Any]:
    n = max(1, len(records))
    action_counts: Counter[str] = Counter()
    schema_repair_key_counts: Counter[str] = Counter()
    for record in records:
        for step in record.get("steps") or []:
            action = step.get("parsed_action") or {}
            action_counts[str(action.get("action", "invalid")) if isinstance(action, dict) else "invalid"] += 1
            schema_repair_key_counts.update(schema_repaired_keys(step))
    candidate_ious = [
        float(item.get("metrics", {}).get("candidate_oracle_iou"))
        for item in records
        if isinstance(item.get("metrics", {}).get("candidate_oracle_iou"), (int, float))
    ]
    selected_ious = [
        float(item.get("metrics", {}).get("selected_region_iou"))
        for item in records
        if isinstance(item.get("metrics", {}).get("selected_region_iou"), (int, float))
    ]
    return {
        "created_at": now(),
        "tasks_path": args.tasks,
        "evidence_index": args.evidence_index,
        "output_dir": args.output_dir,
        "tasks_used": len(records),
        "env": {
            "max_steps": args.max_steps,
            "include_gold_regions": args.include_gold_regions,
            "phase_aware_mask": args.phase_aware_mask,
            "enforce_tool_mask": args.enforce_tool_mask,
            "reward_mode": args.reward_mode,
            "field_policy_prompt": args.field_policy_prompt,
            "target_claim_fields": target_claim_fields,
            "target_claim_fields_from_gold": bool(args.target_claim_fields_from_gold),
        },
        "policy": policy.metadata(),
        "metrics": {
            "trajectory_success_rate": sum(bool(item.get("trajectory_metrics", {}).get("trajectory_success")) for item in records) / n,
            "terminated_rate": sum(bool(item.get("terminated")) for item in records) / n,
            "finish_action_rate": sum(bool(item.get("metrics", {}).get("has_finish_action")) for item in records) / n,
            "finish_rate": sum(bool(item.get("metrics", {}).get("has_finish")) for item in records) / n,
            "premature_finish_task_rate": sum(
                int(int(item.get("metrics", {}).get("premature_finish_count", 0) or 0) > 0) for item in records
            )
            / n,
            "mean_premature_finish_count": sum(
                int(item.get("metrics", {}).get("premature_finish_count", 0) or 0) for item in records
            )
            / n,
            "max_step_stop_rate": sum(
                int((not item.get("metrics", {}).get("has_finish_action")) and int(item.get("num_steps", 0)) >= args.max_steps)
                for item in records
            )
            / n,
            "crop_success_rate": sum(bool(item.get("metrics", {}).get("crop_success")) for item in records) / n,
            "candidate_oracle_hit_rate_iou50": sum(float(x) >= 0.5 for x in candidate_ious) / max(1, len(candidate_ious)),
            "candidate_oracle_hit_rate_iou70": sum(float(x) >= 0.7 for x in candidate_ious) / max(1, len(candidate_ious)),
            "candidate_oracle_mean_iou": sum(candidate_ious) / max(1, len(candidate_ious)),
            "crop_region_called_rate": sum(bool(item.get("metrics", {}).get("crop_region_called")) for item in records) / n,
            "selected_region_hit_rate_iou50": sum(float(x) >= 0.5 for x in selected_ious) / max(1, len(selected_ious)),
            "selected_region_hit_rate_iou70": sum(float(x) >= 0.7 for x in selected_ious) / max(1, len(selected_ious)),
            "selected_region_mean_iou": sum(selected_ious) / max(1, len(selected_ious)),
            "selected_region_iou_count": len(selected_ious),
            "selected_matches_oracle_region_rate": sum(
                bool(item.get("metrics", {}).get("selected_matches_oracle_region")) for item in records
            )
            / n,
            "mean_total_reward": sum(float(item.get("total_reward", 0.0)) for item in records) / n,
            "mean_steps": sum(int(item.get("num_steps", 0)) for item in records) / n,
            "mean_claim_count": sum(int(item.get("metrics", {}).get("claim_count", 0)) for item in records) / n,
            "any_evidence_hit_rate": sum(
                int(
                    (
                        item.get("trajectory_metrics", {}).get(
                            "evidence_hit_count", item.get("metrics", {}).get("evidence_hit_count", 0)
                        )
                        or 0
                    )
                    > 0
                )
                for item in records
            )
            / n,
            "mean_final_reward": sum(float(item.get("trajectory_metrics", {}).get("final_reward", 0.0)) for item in records) / n,
            "mean_claim_supported_rate": sum(float(item.get("trajectory_metrics", {}).get("claim_supported_rate", 0.0)) for item in records) / n,
            "mean_claim_support_precision": sum(
                float(item.get("trajectory_metrics", {}).get("claim_support_precision", 0.0)) for item in records
            )
            / n,
            "mean_claim_support_recall": sum(
                float(item.get("trajectory_metrics", {}).get("claim_support_recall", 0.0)) for item in records
            )
            / n,
            "mean_claim_support_f1": sum(
                float(item.get("trajectory_metrics", {}).get("claim_support_f1", 0.0)) for item in records
            )
            / n,
            "mean_core_support_precision": sum(
                float(item.get("trajectory_metrics", {}).get("core_support_precision", 0.0)) for item in records
            )
            / n,
            "mean_core_support_recall": sum(
                float(item.get("trajectory_metrics", {}).get("core_support_recall", 0.0)) for item in records
            )
            / n,
            "mean_core_support_f1": sum(
                float(item.get("trajectory_metrics", {}).get("core_support_f1", 0.0)) for item in records
            )
            / n,
            "mean_abstain_precision": sum(
                float(item.get("trajectory_metrics", {}).get("abstain_precision", 0.0)) for item in records
            )
            / n,
            "mean_abstain_recall": sum(
                float(item.get("trajectory_metrics", {}).get("abstain_recall", 0.0)) for item in records
            )
            / n,
            "mean_abstain_f1": sum(
                float(item.get("trajectory_metrics", {}).get("abstain_f1", 0.0)) for item in records
            )
            / n,
            "mean_evidence_precision": sum(
                float(item.get("trajectory_metrics", {}).get("evidence_precision", 0.0)) for item in records
            )
            / n,
            "mean_evidence_recall": sum(float(item.get("trajectory_metrics", {}).get("evidence_recall", 0.0)) for item in records) / n,
            "mean_evidence_f1": sum(float(item.get("trajectory_metrics", {}).get("evidence_f1", 0.0)) for item in records) / n,
            "mean_retrieved_evidence_precision": sum(
                float(item.get("trajectory_metrics", {}).get("retrieved_evidence_precision", 0.0)) for item in records
            )
            / n,
            "mean_retrieved_evidence_recall": sum(
                float(item.get("trajectory_metrics", {}).get("retrieved_evidence_recall", 0.0)) for item in records
            )
            / n,
            "mean_retrieved_evidence_f1": sum(
                float(item.get("trajectory_metrics", {}).get("retrieved_evidence_f1", 0.0)) for item in records
            )
            / n,
            "mean_opened_evidence_precision": sum(
                float(item.get("trajectory_metrics", {}).get("opened_evidence_precision", 0.0)) for item in records
            )
            / n,
            "mean_opened_evidence_recall": sum(
                float(item.get("trajectory_metrics", {}).get("opened_evidence_recall", 0.0)) for item in records
            )
            / n,
            "mean_opened_evidence_f1": sum(
                float(item.get("trajectory_metrics", {}).get("opened_evidence_f1", 0.0)) for item in records
            )
            / n,
            "mean_cited_evidence_precision": sum(
                float(item.get("trajectory_metrics", {}).get("cited_evidence_precision", 0.0)) for item in records
            )
            / n,
            "mean_cited_evidence_recall": sum(
                float(item.get("trajectory_metrics", {}).get("cited_evidence_recall", 0.0)) for item in records
            )
            / n,
            "mean_cited_evidence_f1": sum(
                float(item.get("trajectory_metrics", {}).get("cited_evidence_f1", 0.0)) for item in records
            )
            / n,
            "mean_retrieved_evidence_count": sum(
                int(item.get("trajectory_metrics", {}).get("retrieved_evidence_count", 0) or 0) for item in records
            )
            / n,
            "mean_opened_evidence_count": sum(
                int(item.get("trajectory_metrics", {}).get("opened_evidence_count", 0) or 0) for item in records
            )
            / n,
            "mean_cited_evidence_count": sum(
                int(item.get("trajectory_metrics", {}).get("cited_evidence_count", 0) or 0) for item in records
            )
            / n,
            "mean_retrieved_evidence_hit_count": sum(
                int(item.get("trajectory_metrics", {}).get("retrieved_evidence_hit_count", 0) or 0) for item in records
            )
            / n,
            "mean_opened_evidence_hit_count": sum(
                int(item.get("trajectory_metrics", {}).get("opened_evidence_hit_count", 0) or 0) for item in records
            )
            / n,
            "mean_cited_evidence_hit_count": sum(
                int(item.get("trajectory_metrics", {}).get("cited_evidence_hit_count", 0) or 0) for item in records
            )
            / n,
            "mean_field_policy_selection_score": sum(
                float(item.get("trajectory_metrics", {}).get("field_policy_selection_score", 0.0)) for item in records
            )
            / n,
            "mean_invalid_step_rate": sum(float(item.get("trajectory_metrics", {}).get("invalid_step_rate", 0.0)) for item in records) / n,
            "mean_local_caption_only_claim_count": sum(
                int(item.get("trajectory_metrics", {}).get("local_caption_only_claim_count", 0) or 0) for item in records
            )
            / n,
            "mean_local_caption_only_risk_field_claim_count": sum(
                int(item.get("trajectory_metrics", {}).get("local_caption_only_risk_field_claim_count", 0) or 0)
                for item in records
            )
            / n,
            "mean_local_caption_only_unsupported_count": sum(
                int(item.get("trajectory_metrics", {}).get("local_caption_only_unsupported_count", 0) or 0)
                for item in records
            )
            / n,
            "mean_mask_violation_rate": sum(float(item.get("metrics", {}).get("mask_violation_rate", 0.0)) for item in records) / n,
            "mask_violation_task_rate": sum(int(item.get("metrics", {}).get("mask_violation_count", 0) > 0) for item in records) / n,
            "mean_schema_repair_rate": sum(float(item.get("metrics", {}).get("schema_repair_rate", 0.0)) for item in records) / n,
            "schema_repair_task_rate": sum(int(item.get("metrics", {}).get("schema_repair_count", 0) > 0) for item in records) / n,
            "schema_repaired_key_counts": dict(schema_repair_key_counts),
            "no_retrieve_task_rate": sum(int(bool(item.get("metrics", {}).get("no_retrieve"))) for item in records) / n,
            "retrieve_without_external_open_task_rate": sum(
                int(bool(item.get("metrics", {}).get("retrieve_without_external_open"))) for item in records
            )
            / n,
            "write_before_retrieve_task_rate": sum(
                int(bool(item.get("metrics", {}).get("write_before_retrieve"))) for item in records
            )
            / n,
            "premature_mask_finish_task_rate": sum(
                int(bool(item.get("metrics", {}).get("premature_mask_finish"))) for item in records
            )
            / n,
            "finish_not_ready_task_rate": sum(
                int(int(item.get("metrics", {}).get("finish_not_ready_count", 0) or 0) > 0) for item in records
            )
            / n,
            "mean_retrieve_count": sum(int(item.get("metrics", {}).get("retrieve_count", 0) or 0) for item in records) / n,
            "mean_external_open_count": sum(int(item.get("metrics", {}).get("external_open_count", 0) or 0) for item in records) / n,
            "mean_external_open_after_retrieve_count": sum(
                int(item.get("metrics", {}).get("external_open_after_retrieve_count", 0) or 0) for item in records
            )
            / n,
            "mean_negative_retrieve_count": sum(
                int(item.get("metrics", {}).get("negative_retrieve_count", 0) or 0) for item in records
            )
            / n,
            "mean_negative_write_claim_count": sum(
                int(item.get("metrics", {}).get("negative_write_claim_count", 0) or 0) for item in records
            )
            / n,
            "action_counts": dict(action_counts),
        },
        "rollouts": str(trajectories_path),
    }


def progress_record(done: int, total: int, records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_light_summary(records)
    return {"time": now(), "done": done, "total": total, **summary}


def build_light_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(records))
    return {
        "trajectory_success_rate": sum(bool(item.get("trajectory_metrics", {}).get("trajectory_success")) for item in records) / n,
        "finish_action_rate": sum(bool(item.get("metrics", {}).get("has_finish_action")) for item in records) / n,
        "finish_rate": sum(bool(item.get("metrics", {}).get("has_finish")) for item in records) / n,
        "premature_finish_task_rate": sum(
            int(int(item.get("metrics", {}).get("premature_finish_count", 0) or 0) > 0) for item in records
        )
        / n,
        "crop_success_rate": sum(bool(item.get("metrics", {}).get("crop_success")) for item in records) / n,
        "crop_region_called_rate": sum(bool(item.get("metrics", {}).get("crop_region_called")) for item in records) / n,
        "mean_total_reward": sum(float(item.get("total_reward", 0.0)) for item in records) / n,
        "mean_final_reward": sum(float(item.get("trajectory_metrics", {}).get("final_reward", 0.0)) for item in records) / n,
    }


def write_markdown_report(path: Path, summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    metrics = summary["metrics"]
    lines = [
        "# EvidenceGrounded Highlighted Runtime Rollout Report",
        "",
        f"- created_at: {summary['created_at']}",
        f"- tasks_used: {summary['tasks_used']}",
        f"- model: {summary['policy']['model']}",
        f"- adapter: {summary['policy']['adapter']}",
        f"- runtime tasks: {summary['tasks_path']}",
        f"- phase_aware_mask: {summary.get('env', {}).get('phase_aware_mask')}; enforce_tool_mask: {summary.get('env', {}).get('enforce_tool_mask')}",
        f"- target_claim_fields: `{json.dumps(summary.get('env', {}).get('target_claim_fields'), ensure_ascii=False)}`",
        f"- target_claim_fields_from_gold: {summary.get('env', {}).get('target_claim_fields_from_gold')}",
        "",
        "## Metrics",
        "",
        f"- trajectory_success_rate: {metrics['trajectory_success_rate']:.3f}",
        f"- finish_action_rate: {metrics.get('finish_action_rate', 0.0):.3f}",
        f"- finish_rate: {metrics['finish_rate']:.3f}",
        f"- premature_finish_task_rate: {metrics.get('premature_finish_task_rate', 0.0):.3f}",
        f"- mean_premature_finish_count: {metrics.get('mean_premature_finish_count', 0.0):.3f}",
        f"- max_step_stop_rate: {metrics.get('max_step_stop_rate', 0.0):.3f}",
        f"- crop_success_rate: {metrics['crop_success_rate']:.3f}",
        f"- crop_region_called_rate: {metrics.get('crop_region_called_rate', 0.0):.3f}",
        f"- candidate_oracle_hit_rate_iou50: {metrics.get('candidate_oracle_hit_rate_iou50', 0.0):.3f}",
        f"- candidate_oracle_hit_rate_iou70: {metrics.get('candidate_oracle_hit_rate_iou70', 0.0):.3f}",
        f"- candidate_oracle_mean_iou: {metrics.get('candidate_oracle_mean_iou', 0.0):.3f}",
        f"- selected_region_hit_rate_iou50: {metrics.get('selected_region_hit_rate_iou50', 0.0):.3f}",
        f"- selected_region_hit_rate_iou70: {metrics.get('selected_region_hit_rate_iou70', 0.0):.3f}",
        f"- selected_region_mean_iou: {metrics.get('selected_region_mean_iou', 0.0):.3f}",
        f"- selected_region_iou_count: {metrics.get('selected_region_iou_count', 0)}",
        f"- selected_matches_oracle_region_rate: {metrics.get('selected_matches_oracle_region_rate', 0.0):.3f}",
        f"- any_evidence_hit_rate: {metrics['any_evidence_hit_rate']:.3f}",
        f"- mean_total_reward: {metrics['mean_total_reward']:.3f}",
        f"- mean_final_reward: {metrics['mean_final_reward']:.3f}",
        f"- mean_claim_supported_rate: {metrics['mean_claim_supported_rate']:.3f}",
        f"- mean_claim_support_precision: {metrics.get('mean_claim_support_precision', 0.0):.3f}",
        f"- mean_claim_support_recall: {metrics.get('mean_claim_support_recall', 0.0):.3f}",
        f"- mean_claim_support_f1: {metrics.get('mean_claim_support_f1', 0.0):.3f}",
        f"- mean_core_support_precision: {metrics.get('mean_core_support_precision', 0.0):.3f}",
        f"- mean_core_support_recall: {metrics.get('mean_core_support_recall', 0.0):.3f}",
        f"- mean_core_support_f1: {metrics.get('mean_core_support_f1', 0.0):.3f}",
        f"- mean_abstain_precision: {metrics.get('mean_abstain_precision', 0.0):.3f}",
        f"- mean_abstain_recall: {metrics.get('mean_abstain_recall', 0.0):.3f}",
        f"- mean_abstain_f1: {metrics.get('mean_abstain_f1', 0.0):.3f}",
        f"- mean_evidence_precision: {metrics.get('mean_evidence_precision', 0.0):.3f}",
        f"- mean_evidence_recall: {metrics['mean_evidence_recall']:.3f}",
        f"- mean_evidence_f1: {metrics.get('mean_evidence_f1', 0.0):.3f}",
        f"- mean_retrieved_evidence_precision: {metrics.get('mean_retrieved_evidence_precision', 0.0):.3f}",
        f"- mean_retrieved_evidence_recall: {metrics.get('mean_retrieved_evidence_recall', 0.0):.3f}",
        f"- mean_retrieved_evidence_f1: {metrics.get('mean_retrieved_evidence_f1', 0.0):.3f}",
        f"- mean_opened_evidence_precision: {metrics.get('mean_opened_evidence_precision', 0.0):.3f}",
        f"- mean_opened_evidence_recall: {metrics.get('mean_opened_evidence_recall', 0.0):.3f}",
        f"- mean_opened_evidence_f1: {metrics.get('mean_opened_evidence_f1', 0.0):.3f}",
        f"- mean_cited_evidence_precision: {metrics.get('mean_cited_evidence_precision', 0.0):.3f}",
        f"- mean_cited_evidence_recall: {metrics.get('mean_cited_evidence_recall', 0.0):.3f}",
        f"- mean_cited_evidence_f1: {metrics.get('mean_cited_evidence_f1', 0.0):.3f}",
        f"- mean_retrieved_evidence_count: {metrics.get('mean_retrieved_evidence_count', 0.0):.3f}",
        f"- mean_opened_evidence_count: {metrics.get('mean_opened_evidence_count', 0.0):.3f}",
        f"- mean_cited_evidence_count: {metrics.get('mean_cited_evidence_count', 0.0):.3f}",
        f"- mean_retrieved_evidence_hit_count: {metrics.get('mean_retrieved_evidence_hit_count', 0.0):.3f}",
        f"- mean_opened_evidence_hit_count: {metrics.get('mean_opened_evidence_hit_count', 0.0):.3f}",
        f"- mean_cited_evidence_hit_count: {metrics.get('mean_cited_evidence_hit_count', 0.0):.3f}",
        f"- mean_field_policy_selection_score: {metrics.get('mean_field_policy_selection_score', 0.0):.3f}",
        f"- mean_invalid_step_rate: {metrics['mean_invalid_step_rate']:.3f}",
        f"- mean_local_caption_only_claim_count: {metrics.get('mean_local_caption_only_claim_count', 0.0):.3f}",
        f"- mean_local_caption_only_risk_field_claim_count: {metrics.get('mean_local_caption_only_risk_field_claim_count', 0.0):.3f}",
        f"- mean_local_caption_only_unsupported_count: {metrics.get('mean_local_caption_only_unsupported_count', 0.0):.3f}",
        f"- mean_mask_violation_rate: {metrics.get('mean_mask_violation_rate', 0.0):.3f}",
        f"- mask_violation_task_rate: {metrics.get('mask_violation_task_rate', 0.0):.3f}",
        f"- mean_schema_repair_rate: {metrics.get('mean_schema_repair_rate', 0.0):.3f}",
        f"- schema_repair_task_rate: {metrics.get('schema_repair_task_rate', 0.0):.3f}",
        f"- schema_repaired_key_counts: `{json.dumps(metrics.get('schema_repaired_key_counts', {}), ensure_ascii=False)}`",
        f"- no_retrieve_task_rate: {metrics.get('no_retrieve_task_rate', 0.0):.3f}",
        f"- retrieve_without_external_open_task_rate: {metrics.get('retrieve_without_external_open_task_rate', 0.0):.3f}",
        f"- write_before_retrieve_task_rate: {metrics.get('write_before_retrieve_task_rate', 0.0):.3f}",
        f"- premature_mask_finish_task_rate: {metrics.get('premature_mask_finish_task_rate', 0.0):.3f}",
        f"- finish_not_ready_task_rate: {metrics.get('finish_not_ready_task_rate', 0.0):.3f}",
        f"- mean_retrieve_count: {metrics.get('mean_retrieve_count', 0.0):.3f}",
        f"- mean_external_open_count: {metrics.get('mean_external_open_count', 0.0):.3f}",
        f"- mean_external_open_after_retrieve_count: {metrics.get('mean_external_open_after_retrieve_count', 0.0):.3f}",
        f"- mean_negative_retrieve_count: {metrics.get('mean_negative_retrieve_count', 0.0):.3f}",
        f"- mean_negative_write_claim_count: {metrics.get('mean_negative_write_claim_count', 0.0):.3f}",
        f"- mean_steps: {metrics['mean_steps']:.2f}",
        f"- mean_claim_count: {metrics['mean_claim_count']:.2f}",
        f"- action_counts: `{json.dumps(metrics['action_counts'], ensure_ascii=False)}`",
        "",
        "## Per Task",
        "",
    ]
    for record in records:
        metrics_one = record.get("metrics", {})
        lines.extend(
            [
                f"### {record.get('task_id')}",
                "",
                f"- source: {record.get('source_file')} p.{record.get('page')}",
                f"- steps: {record.get('num_steps')}; total_reward: {float(record.get('total_reward', 0.0)):.3f}; terminated: {record.get('terminated')}",
                f"- crop_success: {metrics_one.get('crop_success')}; max_crop_iou: {metrics_one.get('max_crop_iou')}",
                f"- candidate_oracle_iou: {metrics_one.get('candidate_oracle_iou')}; candidate_oracle_rank: {metrics_one.get('candidate_oracle_rank')}; crop_region_called: {metrics_one.get('crop_region_called')}; selected_region_iou: {metrics_one.get('selected_region_iou')}; selected_region_id: {metrics_one.get('selected_region_id')}",
                f"- evidence_hit_count: {metrics_one.get('evidence_hit_count')}; claim_count: {metrics_one.get('claim_count')}; has_finish_action: {metrics_one.get('has_finish_action')}; has_finish: {metrics_one.get('has_finish')}; premature_finish_count: {metrics_one.get('premature_finish_count')}",
                f"- trajectory_success: {metrics_one.get('trajectory_success')}; final_reward: {metrics_one.get('final_reward')}; claim_supported_rate: {metrics_one.get('claim_supported_rate')}; evidence_recall: {metrics_one.get('evidence_recall')}",
                f"- claim support: precision={metrics_one.get('claim_support_precision')}; recall={metrics_one.get('claim_support_recall')}; f1={metrics_one.get('claim_support_f1')}; core_f1={metrics_one.get('core_support_f1')}; abstain_f1={metrics_one.get('abstain_f1')}",
                f"- evidence prf: precision={metrics_one.get('evidence_precision')}; recall={metrics_one.get('evidence_recall')}; f1={metrics_one.get('evidence_f1')}",
                f"- evidence flow: retrieved_recall={metrics_one.get('retrieved_evidence_recall')}; opened_recall={metrics_one.get('opened_evidence_recall')}; cited_recall={metrics_one.get('cited_evidence_recall')}; retrieved_hits={metrics_one.get('retrieved_evidence_hit_count')}; opened_hits={metrics_one.get('opened_evidence_hit_count')}; cited_hits={metrics_one.get('cited_evidence_hit_count')}",
                f"- evidence flow prf: retrieved_p/r/f1={metrics_one.get('retrieved_evidence_precision')}/{metrics_one.get('retrieved_evidence_recall')}/{metrics_one.get('retrieved_evidence_f1')}; opened_p/r/f1={metrics_one.get('opened_evidence_precision')}/{metrics_one.get('opened_evidence_recall')}/{metrics_one.get('opened_evidence_f1')}; cited_p/r/f1={metrics_one.get('cited_evidence_precision')}/{metrics_one.get('cited_evidence_recall')}/{metrics_one.get('cited_evidence_f1')}",
                f"- field_policy_selection_score: {metrics_one.get('field_policy_selection_score')}",
                f"- local caption use: local_only={metrics_one.get('local_caption_only_claim_count')}; risk_field={metrics_one.get('local_caption_only_risk_field_claim_count')}; unsupported={metrics_one.get('local_caption_only_unsupported_count')}",
                f"- mask_violation_count: {metrics_one.get('mask_violation_count')}; mask_violation_rate: {metrics_one.get('mask_violation_rate')}",
                f"- schema_repair_count: {metrics_one.get('schema_repair_count')}; schema_repair_rate: {metrics_one.get('schema_repair_rate')}; schema_repaired_key_counts: `{json.dumps(metrics_one.get('schema_repaired_key_counts', {}), ensure_ascii=False)}`",
                f"- process: no_retrieve={metrics_one.get('no_retrieve')}; retrieve_without_external_open={metrics_one.get('retrieve_without_external_open')}; write_before_retrieve={metrics_one.get('write_before_retrieve')}; premature_mask_finish={metrics_one.get('premature_mask_finish')}",
                f"- process counts: retrieve={metrics_one.get('retrieve_count')}; external_open={metrics_one.get('external_open_count')}; external_open_after_retrieve={metrics_one.get('external_open_after_retrieve_count')}; negative_retrieve={metrics_one.get('negative_retrieve_count')}; negative_write_claim={metrics_one.get('negative_write_claim_count')}",
                f"- trajectory: `{record.get('trajectory')}`",
                "",
            ]
        )
        for step in (record.get("steps") or [])[:5]:
            action = step.get("parsed_action") or {}
            lines.append(
                f"  - step {step.get('step')}: `{action.get('action', 'invalid') if isinstance(action, dict) else 'invalid'}` phase={step.get('tool_mask', {}).get('phase')} mask_violation={step.get('mask_violation')} reward={step.get('reward')}"
            )
        if len(record.get("steps") or []) > 5:
            lines.append("  - ...")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
