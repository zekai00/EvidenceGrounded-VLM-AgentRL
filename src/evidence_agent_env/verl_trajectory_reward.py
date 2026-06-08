"""Custom reward function for verl single-response trajectory GRPO smoke."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env import EvidenceAgentEnv  # noqa: E402


ACTION_NAMES = {
    "inspect_page",
    "propose_regions",
    "select_evidence",
    "crop_region",
    "crop_target",
    "crop_image",
    "retrieve_evidence",
    "open_evidence",
    "write_claim",
    "abstain_claim",
    "write_claims_chunk",
    "write_claims_batch",
    "finish",
}


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str | dict[str, Any],
    extra_info: dict[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    if data_source != "evidence_grounded_trajectory":
        return {"score": 0.0, "format_reward": 0.0, "env_reward": 0.0, "error": f"unsupported {data_source=}"}
    spec = parse_ground_truth(ground_truth)
    actions, parse_error, parse_mode = parse_action_sequence(solution_str)
    if parse_error:
        return {"score": -0.3, "format_reward": -0.3, "env_reward": 0.0, "error": parse_error}
    output_dir = reward_output_dir(spec, solution_str)
    env = EvidenceAgentEnv(
        spec["tasks_path"],
        spec["evidence_index"],
        output_dir,
        max_steps=int(spec.get("max_steps", 16)),
        include_gold_regions=False,
        phase_aware_mask=bool(spec.get("phase_aware_mask", True)),
        enforce_tool_mask=bool(spec.get("enforce_tool_mask", True)),
        tool_schema=str(spec.get("tool_schema", "chunked_claim")),
    )
    env.reset(task_id=str(spec["task_id"]))
    terminated = False
    for action in actions[: int(spec.get("max_steps", 16))]:
        _, _, terminated, _ = env.step(action)
        if terminated:
            break
    metrics = env.trajectory_metrics()
    format_reward = format_reward_for_parse_mode(parse_mode)
    crop_reward = crop_shaping_reward(metrics, actions)
    score = float(metrics["final_reward"]) + format_reward + crop_reward
    return {
        "score": max(-1.0, min(1.0, score)),
        "format_reward": format_reward,
        "crop_reward": crop_reward,
        "env_reward": float(metrics["final_reward"]),
        "parse_mode": parse_mode,
        "trajectory_success": float(bool(metrics["trajectory_success"])),
        "finish": float(bool(metrics["finish"])),
        "crop_success": float(bool(metrics["crop_success"])),
        "evidence_recall": float(metrics["evidence_recall"]),
        "claim_supported_rate": float(metrics["claim_supported_rate"]),
        "invalid_step_rate": float(metrics["invalid_step_rate"]),
        "mask_violation_rate": float(mask_violation_rate(env.tool_results)),
        "steps": float(metrics["steps"]),
    }


def parse_ground_truth(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return json.loads(value)


def parse_action_sequence(text: str) -> tuple[list[dict[str, Any]], str | None, str]:
    if not text:
        return [], "empty response", "none"
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    parsed = try_parse_json_array(cleaned, require_full=True)
    if parsed is None:
        parsed = try_parse_jsonl(cleaned)
        mode = "jsonl" if parsed is not None else "none"
    else:
        mode = "strict_array"
    if parsed is None:
        parsed = extract_action_objects_lenient(cleaned)
        mode = "lenient" if parsed else "none"
    if parsed is None:
        return [], "response is not a JSON array, JSONL action list, or extractable action stream", mode
    actions = [item for item in parsed if isinstance(item, dict)]
    if not actions:
        return [], "no JSON object actions found", mode
    return actions, None, mode


def try_parse_json_array(text: str, *, require_full: bool = False) -> list[Any] | None:
    starts = [idx for idx, char in enumerate(text) if char == "["]
    decoder = json.JSONDecoder()
    for start in starts:
        try:
            value, end = decoder.raw_decode(text[start:])
        except Exception:
            continue
        if require_full and text[start + end :].strip():
            continue
        if isinstance(value, list):
            return value
    return None


def try_parse_jsonl(text: str) -> list[Any] | None:
    rows = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            return None
    return rows or None


def extract_action_objects_lenient(text: str) -> list[dict[str, Any]] | None:
    decoder = json.JSONDecoder()
    actions: list[dict[str, Any]] = []
    pos = 0
    while pos < len(text):
        candidates = [idx for idx in (text.find("[", pos), text.find("{", pos)) if idx != -1]
        if not candidates:
            break
        start = min(candidates)
        try:
            value, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            pos = start + 1
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and item.get("action") in ACTION_NAMES:
                    actions.append(item)
        elif isinstance(value, dict) and value.get("action") in ACTION_NAMES:
            actions.append(value)
        pos = start + max(1, end)
    deduped: list[dict[str, Any]] = []
    for action in actions:
        if not deduped or action != deduped[-1]:
            deduped.append(action)
    return deduped or None


def format_reward_for_parse_mode(parse_mode: str) -> float:
    if parse_mode == "strict_array":
        return 0.08
    if parse_mode == "jsonl":
        return 0.02
    if parse_mode == "lenient":
        return -0.08
    return -0.3


def crop_shaping_reward(metrics: dict[str, Any], actions: list[dict[str, Any]]) -> float:
    has_crop = any(
        action.get("action") in {"crop_region", "crop_target", "crop_image"}
        for action in actions
        if isinstance(action, dict)
    )
    if bool(metrics.get("crop_success")):
        return 0.15
    if has_crop:
        return -0.12
    return -0.05


def reward_output_dir(spec: dict[str, Any], solution_str: str) -> Path:
    digest = hashlib.sha1(solution_str.encode("utf-8", errors="ignore")).hexdigest()[:12]
    task_id = str(spec.get("task_id", "unknown"))
    return Path("/tmp/evidence_grounded_verl_reward") / f"{task_id}_{os.getpid()}_{digest}"


def mask_violation_rate(tool_results: list[dict[str, Any]]) -> float:
    if not tool_results:
        return 0.0
    violations = sum(
        1
        for result in tool_results
        if isinstance(result, dict) and "blocked by phase-aware tool mask" in str(result.get("error", ""))
    )
    return violations / max(1, len(tool_results))
