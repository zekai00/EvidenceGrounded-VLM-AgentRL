"""Action parsing and validation for the EvidenceGrounded tool-call env."""

from __future__ import annotations

import json
from typing import Any


ALLOWED_ACTIONS = {
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

REQUIRED_KEYS: dict[str, set[str]] = {
    "inspect_page": set(),
    "propose_regions": set(),
    "select_evidence": {"evidence_ids"},
    "crop_region": {"region_id"},
    "crop_target": set(),
    "crop_image": {"bbox"},
    "retrieve_evidence": {"query", "scope", "top_k"},
    "open_evidence": {"evidence_id"},
    "write_claim": {"field", "value", "evidence_ids"},
    "abstain_claim": {"field", "reason"},
    "write_claims_chunk": {"claims"},
    "write_claims_batch": {"claims"},
    "finish": set(),
}

RETRIEVAL_SCOPES = {"current_page", "nearby_pages", "same_document", "corpus"}


def parse_action(action: str | dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(action, dict):
        return normalize_action_shape(action), None
    if not isinstance(action, str):
        return None, "action must be a JSON string or dict"
    try:
        parsed = json.loads(action)
    except json.JSONDecodeError as exc:
        return None, f"invalid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "action JSON must be an object"
    return normalize_action_shape(parsed), None


def normalize_action_shape(action: dict[str, Any]) -> dict[str, Any]:
    """Repair common schema slips without changing the intended tool call."""

    if action.get("action") == "write_claim":
        return normalize_claim_shape(action)
    if action.get("action") not in {"write_claims_chunk", "write_claims_batch"}:
        return action
    claims = [normalize_claim_shape(item) for item in (action.get("claims") or []) if isinstance(item, dict)]
    repaired_abstains: list[Any] = []
    changed = False
    for item in action.get("abstains") or []:
        if (
            isinstance(item, dict)
            and "reason" not in item
            and ("value" in item or "evidence_ids" in item or "confidence" in item)
        ):
            claims.append(normalize_claim_shape(item))
            changed = True
            continue
        repaired_abstains.append(item)
    bounded_claims = claims[:2]
    bounded_abstains = repaired_abstains[: max(0, 2 - len(bounded_claims))]
    if len(bounded_claims) != len(claims) or len(bounded_abstains) != len(repaired_abstains):
        changed = True
    repaired = dict(action)
    repaired["claims"] = bounded_claims
    repaired["abstains"] = bounded_abstains
    return repaired if changed or repaired != action else action


def normalize_claim_shape(claim: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(claim)
    confidence = repaired.get("confidence")
    try:
        confidence_value = float(confidence)
    except Exception:
        confidence_value = 0.6
    repaired["confidence"] = max(0.0, min(1.0, confidence_value))
    if isinstance(repaired.get("evidence_ids"), list):
        evidence_ids: list[str] = []
        for item in repaired.get("evidence_ids") or []:
            evidence_id = str(item)
            if evidence_id and evidence_id not in evidence_ids:
                evidence_ids.append(evidence_id)
        repaired["evidence_ids"] = evidence_ids[:3]
    value = repaired.get("value")
    if isinstance(value, list):
        values = []
        for item in value:
            if item not in values:
                values.append(item)
        repaired["value"] = values[:5]
    elif isinstance(value, str) and len(value) > 240:
        repaired["value"] = value[:237] + "..."
    return repaired


def validate_action(action: dict[str, Any]) -> tuple[bool, str | None]:
    name = str(action.get("action", ""))
    if name not in ALLOWED_ACTIONS:
        return False, f"unknown action: {name}"
    missing = REQUIRED_KEYS[name] - set(action)
    if missing:
        return False, f"missing keys for {name}: {sorted(missing)}"
    if name in {"crop_image", "crop_target"} and "bbox" in action:
        if coerce_bbox(action.get("bbox")) is None:
            return False, "bbox must be [x1,y1,x2,y2]"
    if name == "crop_target":
        has_region_id = isinstance(action.get("region_id"), str) and bool(action.get("region_id"))
        has_bbox = coerce_bbox(action.get("bbox")) is not None
        if not has_region_id and not has_bbox:
            return False, "crop_target needs either region_id or bbox"
    if name == "retrieve_evidence":
        if str(action.get("scope")) not in RETRIEVAL_SCOPES:
            return False, f"invalid retrieval scope: {action.get('scope')}"
        try:
            int(action.get("top_k"))
        except Exception:
            return False, "top_k must be an integer"
    if name == "select_evidence":
        evidence_ids = action.get("evidence_ids")
        if not isinstance(evidence_ids, list):
            return False, "evidence_ids must be a list"
        if not evidence_ids:
            return False, "select_evidence needs at least one evidence_id"
        if not all(isinstance(item, str) and item for item in evidence_ids):
            return False, "all evidence_ids must be non-empty strings"
    if name == "write_claim":
        if not isinstance(action.get("evidence_ids"), list):
            return False, "evidence_ids must be a list"
    if name in {"write_claims_chunk", "write_claims_batch"}:
        claims = action.get("claims")
        abstains = action.get("abstains", [])
        if not isinstance(claims, list):
            return False, "claims must be a list"
        if not isinstance(abstains, list):
            return False, "abstains must be a list"
        if not claims and not abstains:
            return False, f"{name} needs at least one claim or abstain"
        for index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                return False, f"claims[{index}] must be an object"
            missing = {"field", "value", "evidence_ids"} - set(claim)
            if missing:
                return False, f"claims[{index}] missing keys: {sorted(missing)}"
            if not isinstance(claim.get("evidence_ids"), list):
                return False, f"claims[{index}].evidence_ids must be a list"
        for index, abstain in enumerate(abstains):
            if not isinstance(abstain, dict):
                return False, f"abstains[{index}] must be an object"
            missing = {"field", "reason"} - set(abstain)
            if missing:
                return False, f"abstains[{index}] missing keys: {sorted(missing)}"
    return True, None


def coerce_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(a: Any, b: Any) -> float:
    box_a = coerce_bbox(a)
    box_b = coerce_bbox(b)
    if box_a is None or box_b is None:
        return -1.0
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else -1.0
