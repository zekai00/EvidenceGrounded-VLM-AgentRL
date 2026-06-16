"""Trajectory verifier for the EvidenceGrounded executable env."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from .actions import bbox_iou
from .tools.claim_tools import is_placeholder_claim_value


CLAIM_FIELDS = {
    "caption_text",
    "image_scope",
    "depicted_work_title",
    "displayed_region",
    "object_type",
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
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
    "object_type",
}

METADATA5_CLAIM_FIELDS = {
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
}

LOCAL_CAPTION_RISK_FIELDS = {
    "image_scope",
    "displayed_region",
    "object_type",
    "visual_elements",
    "technique",
    "composition",
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
    "artist": "creator_or_attribution",
    "dynasty": "creation_period_or_dynasty",
    "collection": "collection_institution",
    "medium_dimensions": "dimensions",
}


class EvidenceVerifier:
    def __init__(self, evidence_index_dir: str | Path | None = None, *, reward_mode: str = "default") -> None:
        self.evidence_policy_by_id = load_evidence_policy(evidence_index_dir) if evidence_index_dir else {}
        self.reward_mode = str(reward_mode or "default")

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
            if self.reward_mode == "field_policy_probe":
                return 0.06 if hits else 0.02
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
            if self.reward_mode == "field_policy_probe":
                return 0.14 if result.get("evidence_id") in gold_evidence_ids(task) else 0.01
            return 0.1 if result.get("evidence_id") in gold_evidence_ids(task) else 0.0
        if name == "write_claim":
            if int(result.get("valid_crop_count") or 0) <= 0:
                return -0.05
            if is_placeholder_claim_value(action.get("value")):
                return -0.18 if self.reward_mode == "field_policy_probe" else -0.12
            if self.reward_mode == "field_policy_probe":
                return 0.28 if self._claim_supported(task, action) else -0.14
            return 0.2 if self._claim_supported(task, action) else -0.05
        if name == "abstain_claim":
            return 0.18 if self._gold_abstains(task, str(action.get("field"))) else -0.08
        if name in {"write_claims_chunk", "write_claims_batch"}:
            if int(result.get("valid_crop_count") or 0) <= 0:
                return -0.05
            reward = 0.0
            for claim in action.get("claims") or []:
                if is_placeholder_claim_value(claim.get("value")):
                    reward += -0.18 if self.reward_mode == "field_policy_probe" else -0.12
                elif self.reward_mode == "field_policy_probe":
                    reward += 0.28 if self._claim_supported(task, claim) else -0.12
                else:
                    reward += 0.2 if self._claim_supported(task, claim) else -0.05
            for abstain in action.get("abstains") or []:
                reward += 0.18 if self._gold_abstains(task, str(abstain.get("field"))) else -0.08
            return reward
        if name == "finish":
            history = list(result.get("history") or [])
            tool_results = list(result.get("tool_results") or [])
            history.append(action)
            tool_results.append(result)
            return self.final_reward(
                task,
                result.get("draft_claims") or [],
                history=history,
                tool_results=tool_results,
            )
        return 0.0

    def final_reward(
        self,
        task: dict[str, Any],
        draft_claims: list[dict[str, Any]],
        *,
        history: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
    ) -> float:
        return float(
            self.trajectory_metrics(task, history or [], tool_results or [], draft_claims, max_steps=1)[
                "final_reward"
            ]
        )

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
        evidence_f1 = f1_score(evidence_precision, evidence_recall)
        retrieved_ids = retrieved_evidence_ids_from(tool_results)
        opened_ids = opened_evidence_ids_from(history, tool_results)
        cited_ids = cited_evidence_ids_from(draft_claims)
        retrieved_hits = retrieved_ids & gold_ids
        opened_hits = opened_ids & gold_ids
        cited_hits = cited_ids & gold_ids
        retrieved_precision = len(retrieved_hits) / max(1, len(retrieved_ids))
        retrieved_recall = len(retrieved_hits) / max(1, len(gold_ids))
        retrieved_f1 = f1_score(retrieved_precision, retrieved_recall)
        opened_precision = len(opened_hits) / max(1, len(opened_ids))
        opened_recall = len(opened_hits) / max(1, len(gold_ids))
        opened_f1 = f1_score(opened_precision, opened_recall)
        cited_precision = len(cited_hits) / max(1, len(cited_ids))
        cited_recall = len(cited_hits) / max(1, len(gold_ids))
        cited_f1 = f1_score(cited_precision, cited_recall)

        claim_metrics = self.claim_metrics(task, draft_claims)
        steps = len(history)
        efficiency = max(0.0, 1.0 - max(0, steps - 8) / max(1, max_steps))
        invalid_penalty = min(1.0, invalid_steps / max(1, steps))
        unsupported_penalty = claim_metrics["unsupported_claim_count"] / max(1, claim_metrics["predicted_claim_count"])
        placeholder_penalty = claim_metrics["placeholder_claim_count"] / max(1, claim_metrics["predicted_claim_count"])
        evidence_hit_bonus = 1.0 if evidence_hits else 0.0
        target_recall = 0.20
        evidence_recall_bonus = min(1.0, evidence_recall / target_recall)
        core_supported_bonus = claim_metrics["core_supported_count"] / max(1, len(CORE_CLAIM_FIELDS))
        core_written_bonus = claim_metrics["core_field_match_count"] / max(1, len(CORE_CLAIM_FIELDS))
        core_unsupported_penalty = claim_metrics["core_unsupported_count"] / max(1, len(CORE_CLAIM_FIELDS))
        finish_quality_bonus = 1.0 if finish and claim_metrics["claim_supported_rate"] >= 0.4 else 0.0
        field_policy_selection_score = field_policy_score(
            finish=finish,
            crop_success=crop_success,
            claim_supported_rate=claim_metrics["claim_supported_rate"],
            core_supported_bonus=core_supported_bonus,
            opened_recall=opened_recall,
            cited_recall=cited_recall,
            abstain_accuracy=claim_metrics["abstain_accuracy"],
            invalid_penalty=invalid_penalty,
            unsupported_penalty=unsupported_penalty,
            core_unsupported_penalty=core_unsupported_penalty,
            placeholder_penalty=placeholder_penalty,
            premature_finish_penalty=premature_finish_penalty,
        )

        # Keep the final reward compact and bounded for GRPO-style relative ranking.
        if self.reward_mode == "field_policy_probe":
            final_reward = field_policy_selection_score
        else:
            final_reward = (
                0.15 * float(finish)
                + 0.20 * min(1.0, max_crop_iou)
                + 0.12 * evidence_hit_bonus
                + 0.08 * evidence_recall_bonus
                + 0.22 * claim_metrics["claim_supported_rate"]
                + 0.18 * core_supported_bonus
                + 0.04 * core_written_bonus
                + 0.10 * finish_quality_bonus
                + 0.12 * claim_metrics["abstain_accuracy"]
                + 0.05 * efficiency
                - 0.10 * invalid_penalty
                - 0.10 * unsupported_penalty
                - 0.08 * core_unsupported_penalty
                - 0.14 * placeholder_penalty
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
            "evidence_f1": round(evidence_f1, 6),
            "retrieved_evidence_count": len(retrieved_ids),
            "opened_evidence_count": len(opened_ids),
            "cited_evidence_count": len(cited_ids),
            "retrieved_evidence_hit_count": len(retrieved_hits),
            "opened_evidence_hit_count": len(opened_hits),
            "cited_evidence_hit_count": len(cited_hits),
            "retrieved_evidence_precision": round(retrieved_precision, 6),
            "retrieved_evidence_recall": round(retrieved_recall, 6),
            "retrieved_evidence_f1": round(retrieved_f1, 6),
            "opened_evidence_precision": round(opened_precision, 6),
            "opened_evidence_recall": round(opened_recall, 6),
            "opened_evidence_f1": round(opened_f1, 6),
            "cited_evidence_precision": round(cited_precision, 6),
            "cited_evidence_recall": round(cited_recall, 6),
            "cited_evidence_f1": round(cited_f1, 6),
            "field_policy_selection_score": round(field_policy_selection_score, 6),
            "reward_mode": self.reward_mode,
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
        predicted_non_abstain = [item for item in draft_claims if not item.get("abstain")]
        predicted_abstains = [item for item in draft_claims if item.get("abstain")]
        core_predicted_non_abstain = [
            item for item in predicted_non_abstain if str(item.get("field") or "") in CORE_CLAIM_FIELDS
        ]
        supported_count = 0
        unsupported_count = 0
        correct_abstains = 0
        field_match_count = 0
        core_supported_count = 0
        core_unsupported_count = 0
        core_field_match_count = 0
        local_caption_only_claim_count = 0
        local_caption_only_risk_field_claim_count = 0
        local_caption_only_unsupported_count = 0
        placeholder_claim_count = 0

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
            if is_placeholder_claim_value(pred.get("value")):
                placeholder_claim_count += 1
            evidence_ids = [str(eid) for eid in pred.get("evidence_ids") or []]
            local_caption_only = bool(evidence_ids) and all(eid.startswith("local_caption_") for eid in evidence_ids)
            local_caption_risk_field = local_caption_only and field in LOCAL_CAPTION_RISK_FIELDS
            if local_caption_only:
                local_caption_only_claim_count += 1
            if local_caption_risk_field:
                local_caption_only_risk_field_claim_count += 1
            supported = False
            if gold is not None and not gold.get("abstain") and evidence_overlap(gold, pred):
                overlap_ids = overlapping_evidence_ids(gold, pred)
                if self._evidence_policy_supports_field(field, overlap_ids):
                    supported_count += 1
                    supported = True
                    if field in CORE_CLAIM_FIELDS:
                        core_supported_count += 1
            if not supported:
                unsupported_count += 1
                if local_caption_risk_field:
                    local_caption_only_unsupported_count += 1
                if field in CORE_CLAIM_FIELDS:
                    core_unsupported_count += 1

        claim_support_precision = supported_count / max(1, len(predicted_non_abstain))
        claim_support_recall = supported_count / max(1, len(gold_supported))
        core_support_precision = core_supported_count / max(1, len(core_predicted_non_abstain))
        core_support_recall = core_supported_count / max(1, len(CORE_CLAIM_FIELDS))
        abstain_precision = correct_abstains / max(1, len(predicted_abstains))
        abstain_recall = correct_abstains / max(1, len(gold_abstains))

        return {
            "gold_claim_count": len(gold_claims),
            "gold_supported_claim_count": len(gold_supported),
            "gold_abstain_count": len(gold_abstains),
            "predicted_claim_count": len(draft_claims),
            "predicted_non_abstain_claim_count": len(predicted_non_abstain),
            "predicted_abstain_count": len(predicted_abstains),
            "field_match_count": field_match_count,
            "supported_claim_count": supported_count,
            "unsupported_claim_count": unsupported_count,
            "core_supported_count": core_supported_count,
            "core_unsupported_count": core_unsupported_count,
            "core_field_match_count": core_field_match_count,
            "local_caption_only_claim_count": local_caption_only_claim_count,
            "local_caption_only_risk_field_claim_count": local_caption_only_risk_field_claim_count,
            "local_caption_only_unsupported_count": local_caption_only_unsupported_count,
            "placeholder_claim_count": placeholder_claim_count,
            "claim_support_precision": claim_support_precision,
            "claim_support_recall": claim_support_recall,
            "claim_support_f1": f1_score(claim_support_precision, claim_support_recall),
            "core_support_precision": core_support_precision,
            "core_support_recall": core_support_recall,
            "core_support_f1": f1_score(core_support_precision, core_support_recall),
            "abstain_precision": abstain_precision,
            "abstain_recall": abstain_recall,
            "abstain_f1": f1_score(abstain_precision, abstain_recall),
            "core_supported_rate": core_supported_count / max(1, len(CORE_CLAIM_FIELDS)),
            "core_field_recall": core_field_match_count / max(1, len(CORE_CLAIM_FIELDS)),
            "correct_abstain_count": correct_abstains,
            "claim_supported_rate": claim_support_recall,
            "claim_field_recall": field_match_count / max(1, len(gold_claims)),
            "abstain_accuracy": abstain_recall,
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


def f1_score(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def field_policy_score(
    *,
    finish: bool,
    crop_success: bool,
    claim_supported_rate: float,
    core_supported_bonus: float,
    opened_recall: float,
    cited_recall: float,
    abstain_accuracy: float,
    invalid_penalty: float,
    unsupported_penalty: float,
    core_unsupported_penalty: float,
    placeholder_penalty: float,
    premature_finish_penalty: float,
) -> float:
    """Selection score for field/evidence-policy probes.

    This score intentionally avoids rewarding retrieve hits directly. It
    rewards evidence only after it has been opened or cited in final claims.
    """

    score = (
        0.12 * float(finish)
        + 0.12 * float(crop_success)
        + 0.30 * claim_supported_rate
        + 0.16 * core_supported_bonus
        + 0.14 * min(1.0, opened_recall / 0.20)
        + 0.18 * min(1.0, cited_recall / 0.20)
        + 0.12 * abstain_accuracy
        - 0.10 * invalid_penalty
        - 0.16 * unsupported_penalty
        - 0.12 * core_unsupported_penalty
        - 0.20 * placeholder_penalty
        - 0.25 * premature_finish_penalty
    )
    return max(-1.0, min(1.0, float(score)))


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


def retrieved_evidence_ids_from(tool_results: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for result in tool_results:
        if not isinstance(result, dict) or result.get("tool") != "retrieve_evidence":
            continue
        ids.update(str(eid) for eid in result.get("hit_evidence_ids") or [])
        for item in result.get("results") or []:
            if isinstance(item, dict) and item.get("evidence_id"):
                ids.add(str(item.get("evidence_id")))
    return ids


def opened_evidence_ids_from(history: list[dict[str, Any]], tool_results: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for action, result in zip(history, tool_results):
        if not isinstance(action, dict) or action.get("action") != "open_evidence":
            continue
        evidence_id = action.get("evidence_id")
        if evidence_id and isinstance(result, dict) and not result.get("error"):
            ids.add(str(evidence_id))
    for result in tool_results:
        if (
            isinstance(result, dict)
            and result.get("tool") == "open_evidence"
            and result.get("evidence_id")
            and not result.get("error")
        ):
            ids.add(str(result.get("evidence_id")))
    return ids


def cited_evidence_ids_from(draft_claims: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for claim in draft_claims:
        if isinstance(claim, dict):
            ids.update(str(eid) for eid in claim.get("evidence_ids") or [])
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
