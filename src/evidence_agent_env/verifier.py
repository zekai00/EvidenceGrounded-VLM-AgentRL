"""Trajectory verifier for the EvidenceGrounded executable env."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .actions import bbox_iou


CLAIM_FIELDS = {
    "caption_text",
    "image_scope",
    "depicted_work_title",
    "displayed_region",
    "object_type",
    "artist",
    "dynasty",
    "visual_elements",
    "technique",
    "composition",
    "medium_dimensions",
    "collection",
    # Backward compatibility for v0.4 and older datasets.
    "title",
}

CORE_CLAIM_FIELDS = {
    "caption_text",
    "image_scope",
    "depicted_work_title",
    "displayed_region",
    "object_type",
}

NO_CLAIM_EVIDENCE_ROLES = {
    "toc",
    "bibliography",
    "front_matter",
    "back_matter",
    "ocr_noise",
    "low_value_background",
}

FIELD_ALIASES = {
    "title": "depicted_work_title",
}


class EvidenceVerifier:
    def __init__(self, evidence_index_dir: str | Path | None = None) -> None:
        self.evidence_policy_by_id = load_evidence_policy(evidence_index_dir) if evidence_index_dir else {}

    def step_reward(self, task: dict[str, Any], action: dict[str, Any], result: dict[str, Any]) -> float:
        name = action.get("action")
        if result.get("error"):
            return -0.2
        if name in {"crop_image", "crop_region", "crop_target"}:
            return min(0.5, max(0.0, float(result.get("bbox_iou", 0.0)))) if result.get("bbox_iou", -1) >= 0 else 0.0
        if name == "retrieve_evidence":
            if int(result.get("valid_crop_count") or 0) <= 0:
                return -0.05
            hits = set(result.get("hit_evidence_ids") or [])
            return 0.2 if hits else 0.05
        if name == "select_evidence":
            selected = {str(item) for item in action.get("evidence_ids") or []}
            gold = gold_evidence_ids(task)
            if not selected:
                return -0.05
            hits = selected & gold
            if hits and selected <= gold:
                return 0.25
            if hits:
                return 0.15
            return -0.05
        if name == "open_evidence":
            return 0.1 if result.get("evidence_id") in gold_evidence_ids(task) else 0.0
        if name == "write_claim":
            if int(result.get("valid_crop_count") or 0) <= 0:
                return -0.05
            return 0.2 if self._claim_supported(task, action) else -0.05
        if name == "abstain_claim":
            return 0.1 if self._gold_abstains(task, str(action.get("field"))) else -0.05
        if name in {"write_claims_chunk", "write_claims_batch"}:
            if int(result.get("valid_crop_count") or 0) <= 0:
                return -0.05
            reward = 0.0
            for claim in action.get("claims") or []:
                reward += 0.2 if self._claim_supported(task, claim) else -0.05
            for abstain in action.get("abstains") or []:
                reward += 0.1 if self._gold_abstains(task, str(abstain.get("field"))) else -0.05
            return reward
        if name == "finish":
            return self.final_reward(task, result.get("draft_claims") or [])
        return 0.0

    def final_reward(self, task: dict[str, Any], draft_claims: list[dict[str, Any]]) -> float:
        return float(self.trajectory_metrics(task, [], [], draft_claims, max_steps=1)["final_reward"])

    def trajectory_metrics(
        self,
        task: dict[str, Any],
        history: list[dict[str, Any]],
        tool_results: list[dict[str, Any]],
        draft_claims: list[dict[str, Any]],
        *,
        max_steps: int,
    ) -> dict[str, Any]:
        """Score a full executable trajectory.

        This verifier intentionally uses deterministic task-local evidence and
        gold claim links. It is a training/evaluation verifier, not an LLM judge.
        """

        action_counts = Counter(
            str(action.get("action", "invalid")) if isinstance(action, dict) else "invalid" for action in history
        )
        invalid_steps = sum(
            1
            for action, result in zip(history, tool_results)
            if not isinstance(action, dict) or bool(result.get("error"))
        )
        finish_events = [
            (action, result)
            for action, result in zip(history, tool_results)
            if isinstance(action, dict) and action.get("action") == "finish"
        ]
        finish = any(not bool(result.get("error")) for _action, result in finish_events)
        premature_finish_count = sum(1 for _action, result in finish_events if bool(result.get("error")))
        premature_finish_penalty = min(1.0, premature_finish_count)

        crop_ious = [
            float(result.get("bbox_iou"))
            for result in tool_results
            if isinstance(result.get("bbox_iou"), (int, float)) and float(result.get("bbox_iou")) >= 0
        ]
        max_crop_iou = max(crop_ious) if crop_ious else 0.0
        crop_success = max_crop_iou >= 0.5

        gold_ids = gold_evidence_ids(task)
        selected_ids = selected_evidence_ids_from(history, tool_results, draft_claims)
        evidence_hits = selected_ids & gold_ids
        evidence_precision = len(evidence_hits) / max(1, len(selected_ids))
        evidence_recall = len(evidence_hits) / max(1, len(gold_ids))

        claim_metrics = self.claim_metrics(task, draft_claims)
        steps = len(history)
        efficiency = max(0.0, 1.0 - max(0, steps - 8) / max(1, max_steps))
        invalid_penalty = min(1.0, invalid_steps / max(1, steps))
        unsupported_penalty = claim_metrics["unsupported_claim_count"] / max(1, claim_metrics["predicted_claim_count"])
        evidence_hit_bonus = 1.0 if evidence_hits else 0.0
        target_recall = 0.20
        evidence_recall_bonus = min(1.0, evidence_recall / target_recall)
        core_supported_bonus = claim_metrics["core_supported_count"] / max(1, len(CORE_CLAIM_FIELDS))
        core_written_bonus = claim_metrics["core_field_match_count"] / max(1, len(CORE_CLAIM_FIELDS))
        core_unsupported_penalty = claim_metrics["core_unsupported_count"] / max(1, len(CORE_CLAIM_FIELDS))
        finish_quality_bonus = 1.0 if finish and claim_metrics["claim_supported_rate"] >= 0.4 else 0.0

        # Keep the final reward compact and bounded for GRPO-style relative ranking.
        final_reward = (
            0.15 * float(finish)
            + 0.20 * min(1.0, max_crop_iou)
            + 0.12 * evidence_hit_bonus
            + 0.08 * evidence_recall_bonus
            + 0.22 * claim_metrics["claim_supported_rate"]
            + 0.18 * core_supported_bonus
            + 0.04 * core_written_bonus
            + 0.10 * finish_quality_bonus
            + 0.05 * claim_metrics["abstain_accuracy"]
            + 0.05 * efficiency
            - 0.10 * invalid_penalty
            - 0.10 * unsupported_penalty
            - 0.08 * core_unsupported_penalty
            - 0.25 * premature_finish_penalty
        )
        final_reward = max(-1.0, min(1.0, final_reward))

        trajectory_success = (
            finish
            and crop_success
            and bool(evidence_hits)
            and claim_metrics["supported_claim_count"] > 0
            and invalid_steps == 0
        )
        return {
            "final_reward": round(final_reward, 6),
            "trajectory_success": bool(trajectory_success),
            "finish": bool(finish),
            "premature_finish_count": premature_finish_count,
            "premature_finish_rate": round(premature_finish_count / max(1, steps), 6),
            "steps": steps,
            "invalid_steps": invalid_steps,
            "invalid_step_rate": invalid_penalty,
            "action_counts": dict(action_counts),
            "max_crop_iou": round(max_crop_iou, 6),
            "crop_success": bool(crop_success),
            "gold_evidence_count": len(gold_ids),
            "selected_evidence_count": len(selected_ids),
            "evidence_hit_count": len(evidence_hits),
            "evidence_precision": round(evidence_precision, 6),
            "evidence_recall": round(evidence_recall, 6),
            "efficiency": round(efficiency, 6),
            **claim_metrics,
        }

    def claim_metrics(self, task: dict[str, Any], draft_claims: list[dict[str, Any]]) -> dict[str, Any]:
        gold_claims = [
            item for item in task.get("gold", {}).get("claims", []) if str(item.get("field", "")) in CLAIM_FIELDS
        ]
        gold_by_field = {str(item.get("field")): item for item in gold_claims}
        pred_by_field = {str(item.get("field")): item for item in draft_claims if item.get("field")}

        gold_supported = [item for item in gold_claims if not item.get("abstain")]
        gold_abstains = [item for item in gold_claims if item.get("abstain")]
        supported_count = 0
        unsupported_count = 0
        correct_abstains = 0
        field_match_count = 0
        core_supported_count = 0
        core_unsupported_count = 0
        core_field_match_count = 0

        for field, pred in pred_by_field.items():
            gold = gold_by_field.get(field)
            if gold is not None:
                field_match_count += 1
                if field in CORE_CLAIM_FIELDS:
                    core_field_match_count += 1
            if pred.get("abstain"):
                if gold is not None and gold.get("abstain"):
                    correct_abstains += 1
                continue
            if gold is not None and not gold.get("abstain") and evidence_overlap(gold, pred):
                overlap_ids = overlapping_evidence_ids(gold, pred)
                if self._evidence_policy_supports_field(field, overlap_ids):
                    supported_count += 1
                    if field in CORE_CLAIM_FIELDS:
                        core_supported_count += 1
                else:
                    unsupported_count += 1
                    if field in CORE_CLAIM_FIELDS:
                        core_unsupported_count += 1
            else:
                unsupported_count += 1
                if field in CORE_CLAIM_FIELDS:
                    core_unsupported_count += 1

        return {
            "gold_claim_count": len(gold_claims),
            "gold_supported_claim_count": len(gold_supported),
            "gold_abstain_count": len(gold_abstains),
            "predicted_claim_count": len(draft_claims),
            "field_match_count": field_match_count,
            "supported_claim_count": supported_count,
            "unsupported_claim_count": unsupported_count,
            "core_supported_count": core_supported_count,
            "core_unsupported_count": core_unsupported_count,
            "core_field_match_count": core_field_match_count,
            "core_supported_rate": core_supported_count / max(1, len(CORE_CLAIM_FIELDS)),
            "core_field_recall": core_field_match_count / max(1, len(CORE_CLAIM_FIELDS)),
            "correct_abstain_count": correct_abstains,
            "claim_supported_rate": supported_count / max(1, len(gold_supported)),
            "claim_field_recall": field_match_count / max(1, len(gold_claims)),
            "abstain_accuracy": correct_abstains / max(1, len(gold_abstains)),
        }

    def crop_iou(self, task: dict[str, Any], bbox: Any) -> float:
        return bbox_iou(task.get("gold", {}).get("image_bbox"), bbox)

    def _gold_abstains(self, task: dict[str, Any], field: str) -> bool:
        for item in task.get("gold", {}).get("claims", []):
            if item.get("field") == field:
                return bool(item.get("abstain"))
        return False

    def _claim_supported(self, task: dict[str, Any], action: dict[str, Any]) -> bool:
        field = str(action.get("field", ""))
        for item in task.get("gold", {}).get("claims", []):
            if item.get("field") == field:
                if item.get("abstain"):
                    return False
                if not evidence_overlap(item, action):
                    return False
                return self._evidence_policy_supports_field(field, overlapping_evidence_ids(item, action))
        return False

    def _evidence_policy_supports_field(self, field: str, evidence_ids: set[str]) -> bool:
        if not self.evidence_policy_by_id:
            return True
        checked = 0
        for evidence_id in evidence_ids:
            policy = self.evidence_policy_by_id.get(str(evidence_id))
            if not policy:
                # Overlay is intentionally partial; missing policy keeps legacy behavior.
                return True
            checked += 1
            if policy_allows_claim_field(policy, field):
                return True
        return checked == 0


def load_evidence_policy(index_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    if not index_dir:
        return {}
    path = Path(index_dir) / "corpus_chunks.jsonl"
    if not path.exists():
        return {}
    policies: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("adjudication_status") and not row.get("adjudicated_evidence_role"):
                continue
            evidence_id = str(row.get("evidence_id") or "")
            if evidence_id:
                policies[evidence_id] = row
    return policies


def policy_allows_claim_field(policy: dict[str, Any], field: str) -> bool:
    status = str(policy.get("adjudication_status") or "")
    if status and status != "accepted_auto":
        return False
    role = str(policy.get("adjudicated_evidence_role") or policy.get("evidence_role") or "")
    if role in NO_CLAIM_EVIDENCE_ROLES:
        return False
    if policy.get("usable_for_claim_by_adjudication") is False:
        return False
    allowed = policy.get("adjudicated_claim_allowed_fields") or policy.get("claim_allowed_fields") or []
    if not allowed:
        return False
    return claim_field_allowed(field, allowed)


def claim_field_allowed(field: str, allowed_fields: list[Any]) -> bool:
    normalized = FIELD_ALIASES.get(str(field), str(field))
    allowed = {FIELD_ALIASES.get(str(item), str(item)) for item in allowed_fields}
    return normalized in allowed


def evidence_policy_block_reason(policy: dict[str, Any], field: str) -> str | None:
    if not policy.get("adjudication_status") and not policy.get("adjudicated_evidence_role"):
        return None
    if policy_allows_claim_field(policy, field):
        return None
    evidence_id = policy.get("evidence_id")
    role = policy.get("adjudicated_evidence_role") or policy.get("evidence_role")
    status = policy.get("adjudication_status")
    allowed = policy.get("adjudicated_claim_allowed_fields") or policy.get("claim_allowed_fields") or []
    return f"evidence {evidence_id} role={role} status={status} cannot support field={field}; allowed_fields={allowed}"


def gold_evidence_ids(task: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for item in task.get("gold", {}).get("claims", []):
        ids.update(str(eid) for eid in item.get("evidence_ids") or [])
        ids.update(str(eid) for eid in item.get("candidate_evidence_ids") or [])
    ids.update(str(eid) for eid in task.get("gold", {}).get("evidence_ids") or [])
    ids.update(str(eid) for eid in task.get("gold", {}).get("candidate_evidence_ids") or [])
    return ids


def evidence_overlap(gold: dict[str, Any], pred: dict[str, Any]) -> bool:
    return bool(overlapping_evidence_ids(gold, pred))


def overlapping_evidence_ids(gold: dict[str, Any], pred: dict[str, Any]) -> set[str]:
    gold_ids = {str(eid) for eid in gold.get("evidence_ids") or []}
    if not gold_ids:
        gold_ids = {str(eid) for eid in gold.get("candidate_evidence_ids") or []}
    pred_ids = {str(eid) for eid in pred.get("evidence_ids") or []}
    return gold_ids & pred_ids


def selected_evidence_ids_from(
    history: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
    draft_claims: list[dict[str, Any]],
) -> set[str]:
    ids: set[str] = set()
    for action in history:
        if not isinstance(action, dict):
            continue
        if action.get("action") == "select_evidence":
            ids.update(str(eid) for eid in action.get("evidence_ids") or [])
        if action.get("action") in {"open_evidence"} and action.get("evidence_id"):
            ids.add(str(action.get("evidence_id")))
        if action.get("action") == "write_claim":
            ids.update(str(eid) for eid in action.get("evidence_ids") or [])
        if action.get("action") in {"write_claims_chunk", "write_claims_batch"}:
            for claim in action.get("claims") or []:
                if isinstance(claim, dict):
                    ids.update(str(eid) for eid in claim.get("evidence_ids") or [])
    for result in tool_results:
        if not isinstance(result, dict):
            continue
        ids.update(str(eid) for eid in result.get("hit_evidence_ids") or [])
        if result.get("tool") == "open_evidence" and result.get("evidence_id") and not result.get("error"):
            ids.add(str(result.get("evidence_id")))
    for claim in draft_claims:
        ids.update(str(eid) for eid in claim.get("evidence_ids") or [])
    return ids
