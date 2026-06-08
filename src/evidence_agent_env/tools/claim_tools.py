"""Claim-state helpers for EvidenceGrounded tool-call trajectories."""

from __future__ import annotations

from typing import Any


DEFAULT_CLAIM_FIELDS = [
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
]


def normalize_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except Exception:
        confidence = 0.6
    return max(0.0, min(1.0, confidence))


def normalize_claim(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "field": item.get("field"),
        "value": item.get("value"),
        "evidence_ids": item.get("evidence_ids") or [],
        "visual_bbox": item.get("visual_bbox"),
        "confidence": normalize_confidence(item.get("confidence")),
        "abstain": False,
    }


def normalize_abstain(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "field": item.get("field"),
        "reason": item.get("reason"),
        "abstain": True,
    }


def upsert_claim(draft_claims: list[dict[str, Any]], claim: dict[str, Any]) -> list[dict[str, Any]]:
    field = claim.get("field")
    return [item for item in draft_claims if item.get("field") != field] + [claim]


def apply_claim_write(
    draft_claims: list[dict[str, Any]],
    *,
    claims: list[dict[str, Any]] | None = None,
    abstains: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    next_claims = list(draft_claims)
    for item in claims or []:
        next_claims = upsert_claim(next_claims, normalize_claim(item))
    for item in abstains or []:
        next_claims = upsert_claim(next_claims, normalize_abstain(item))
    return next_claims


def claim_state(
    draft_claims: list[dict[str, Any]],
    *,
    target_fields: list[str] | None = None,
) -> dict[str, Any]:
    target_fields = target_fields or DEFAULT_CLAIM_FIELDS
    by_field = {str(item.get("field")): item for item in draft_claims if item.get("field")}
    written_fields = [field for field in target_fields if field in by_field and not by_field[field].get("abstain")]
    abstained_fields = [field for field in target_fields if field in by_field and by_field[field].get("abstain")]
    remaining_fields = [field for field in target_fields if field not in by_field]
    evidence_ids: list[str] = []
    for item in draft_claims:
        for evidence_id in item.get("evidence_ids") or []:
            evidence_id = str(evidence_id)
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
    return {
        "target_fields": target_fields,
        "written_fields": written_fields,
        "abstained_fields": abstained_fields,
        "remaining_fields": remaining_fields,
        "claim_count": len(written_fields),
        "abstain_count": len(abstained_fields),
        "evidence_ids": evidence_ids,
    }


def claim_write_result(
    tool: str,
    draft_claims: list[dict[str, Any]],
    *,
    claims: list[dict[str, Any]] | None = None,
    abstains: list[dict[str, Any]] | None = None,
    target_fields: list[str] | None = None,
) -> dict[str, Any]:
    normalized_claims = [normalize_claim(item) for item in claims or []]
    normalized_abstains = [normalize_abstain(item) for item in abstains or []]
    return {
        "tool": tool,
        "claims": normalized_claims,
        "abstains": normalized_abstains,
        "claim_state": claim_state(draft_claims, target_fields=target_fields),
    }
