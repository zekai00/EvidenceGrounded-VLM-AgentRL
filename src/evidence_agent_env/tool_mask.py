"""Phase-aware tool availability for executable EvidenceGrounded rollouts."""

from __future__ import annotations

from typing import Any

from .actions import ALLOWED_ACTIONS


FIELD_ALIASES = {
    "title": "depicted_work_title",
    "artist": "creator_or_attribution",
    "dynasty": "creation_period_or_dynasty",
    "collection": "collection_institution",
    "medium_dimensions": "dimensions",
}

ACTION_ORDER = [
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
]

SCHEMA_ACTIONS = {
    "inspect_crop": {
        "inspect_page",
        "select_evidence",
        "crop_target",
        "retrieve_evidence",
        "open_evidence",
        "write_claim",
        "abstain_claim",
        "write_claims_chunk",
        "write_claims_batch",
        "finish",
    },
    "no_select": {
        "inspect_page",
        "crop_target",
        "retrieve_evidence",
        "open_evidence",
        "write_claim",
        "abstain_claim",
        "write_claims_chunk",
        "write_claims_batch",
        "finish",
    },
}


def all_actions() -> list[str]:
    return [action for action in ACTION_ORDER if action in ALLOWED_ACTIONS]


def schema_actions(schema: str | None = None) -> list[str]:
    allowed = SCHEMA_ACTIONS.get(str(schema or ""))
    if not allowed:
        return all_actions()
    return [action for action in ACTION_ORDER if action in allowed and action in ALLOWED_ACTIONS]


def phase_aware_tool_mask(obs: dict[str, Any]) -> dict[str, Any]:
    """Return allowed actions for the current observation.

    This is intentionally heuristic and deterministic. It is not a learned
    planner; it prevents obvious phase regressions such as reopening evidence
    after the state is already ready for chunked claim writing.
    """

    history_actions = action_names(obs.get("history") or [])
    tool_names = tool_result_names(obs.get("tool_results") or [])
    claim_state = obs.get("claim_state") or {}
    remaining_fields = list(claim_state.get("remaining_fields") or [])
    written_fields = list(claim_state.get("written_fields") or [])
    abstained_fields = list(claim_state.get("abstained_fields") or [])

    if not remaining_fields:
        return decision(
            "finish_ready",
            ["finish"],
            "claim_state has no remaining fields; finish the trajectory.",
            obs,
        )

    has_regions = (
        bool(obs.get("available_region_ids") or obs.get("regions"))
        or "inspect_page" in history_actions
        or "inspect_page" in tool_names
        or "propose_regions" in history_actions
        or "propose_regions" in tool_names
    )
    has_crop = has_successful_crop(obs.get("tool_results") or []) or int(obs.get("valid_crop_count") or 0) > 0
    has_retrieved = "retrieve_evidence" in history_actions or "retrieve_evidence" in tool_names
    retrieved_count = sum(1 for action in history_actions if action == "retrieve_evidence")
    opened_count = sum(1 for action in history_actions if action == "open_evidence")
    has_opened = opened_count > 0
    has_selected = bool(obs.get("selected_evidence_ids")) or "select_evidence" in history_actions or "select_evidence" in tool_names
    has_evidence = has_selected or has_opened
    has_claims = bool(written_fields or abstained_fields or obs.get("draft_claims"))

    if str(obs.get("tool_schema") or "") == "no_select":
        return no_select_phase_mask(
            obs,
            remaining_fields=remaining_fields,
            has_regions=has_regions,
            has_crop=has_crop,
            has_retrieved=has_retrieved,
            retrieved_count=retrieved_count,
            opened_count=opened_count,
            has_claims=has_claims,
        )

    if has_claims:
        return claim_phase_decision(
            "claim_continuation",
            "claims have already started and remaining fields still exist; write one field at a time or abstain remaining fields before finish.",
            obs,
            remaining_fields,
        )

    if not has_regions:
        return decision(
            "region_discovery",
            ["inspect_page", "propose_regions"],
            "the page has not been inspected and no layout regions are available yet.",
            obs,
        )

    if not has_crop:
        has_available_region_ids = bool(obs.get("available_region_ids") or obs.get("regions"))
        region_actions = ["crop_target", "crop_region"] if has_available_region_ids else [
            "select_evidence",
            "crop_target",
            "crop_region",
            "propose_regions",
        ]
        return decision(
            "region_selection",
            region_actions,
            "layout regions are available; crop the target figure region before retrieval and claim writing.",
            obs,
        )

    if not has_retrieved:
        if opened_count >= 1:
            return claim_phase_decision(
                "evidence_retrieval_after_open",
                "one evidence item has already been opened before retrieval; stop opening repeatedly, then retrieve more evidence or write claims.",
                obs,
                remaining_fields,
                extra_actions=["select_evidence", "retrieve_evidence"],
                keep_extra_when_unsupported=True,
            )
        return decision(
            "evidence_retrieval",
            ["select_evidence", "retrieve_evidence", "open_evidence"],
            "target region is cropped; retrieve evidence or select/open one visible local evidence item before writing claims.",
            obs,
        )

    if retrieved_count < 4:
        return claim_phase_decision(
            "evidence_opening",
            "retrieval results or visible local evidence are available; open/select evidence, continue bounded retrieval if needed, or write if enough evidence is visible.",
            obs,
            remaining_fields,
            extra_actions=["select_evidence", "open_evidence", "retrieve_evidence"],
            keep_extra_when_unsupported=True,
        )

    if opened_count < 8:
        return claim_phase_decision(
            "evidence_opening_after_retrieval_cap",
            "the bounded retrieval budget has been used; open/select visible evidence or write claims instead of retrieving again.",
            obs,
            remaining_fields,
            extra_actions=["select_evidence", "open_evidence"],
            keep_extra_when_unsupported=True,
        )

    if has_selected or opened_count >= 8 or retrieved_count >= 4:
        return claim_phase_decision(
            "claim_ready",
            "enough evidence has been retrieved/opened; select visible evidence if needed, then write claims instead of repeatedly opening evidence.",
            obs,
            remaining_fields,
            extra_actions=["select_evidence"],
        )

    return claim_phase_decision(
        "evidence_opening",
        "retrieval results are available; open/select evidence, or write if enough evidence is already visible.",
        obs,
        remaining_fields,
        extra_actions=["select_evidence", "open_evidence", "retrieve_evidence"],
        keep_extra_when_unsupported=True,
    )


def no_select_phase_mask(
    obs: dict[str, Any],
    *,
    remaining_fields: list[Any],
    has_regions: bool,
    has_crop: bool,
    has_retrieved: bool,
    retrieved_count: int,
    opened_count: int,
    has_claims: bool,
) -> dict[str, Any]:
    """Phase mask for the v1.0.3 no-select protocol.

    The model never calls select_evidence in this protocol. Evidence becomes
    usable when it is visible in the observation, returned by retrieval, or
    opened explicitly.
    """

    if not remaining_fields:
        return decision(
            "finish_ready",
            ["finish"],
            "claim_state has no remaining fields; finish the trajectory.",
            obs,
        )

    if has_claims:
        return claim_phase_decision(
            "claim_continuation",
            "claims have already started and remaining fields still exist; continue writing or abstain remaining fields before finish.",
            obs,
            remaining_fields,
        )

    if not has_regions:
        return decision(
            "region_discovery",
            ["inspect_page"],
            "the page has not been inspected; inspect_page is required before crop_target.",
            obs,
        )

    if not has_crop:
        return decision(
            "region_selection",
            ["crop_target"],
            "layout regions are available; crop_target must crop the target figure region before evidence use.",
            obs,
        )

    visible_count = len(obs.get("visible_evidence_ids") or [])
    if not has_retrieved and opened_count == 0:
        allowed = ["open_evidence", "retrieve_evidence"] if visible_count else ["retrieve_evidence"]
        return decision(
            "local_evidence_opening",
            allowed,
            "target region has been cropped; open one visible local caption evidence if available, otherwise retrieve evidence.",
            obs,
        )

    if not has_retrieved:
        return claim_phase_decision(
            "evidence_retrieval_after_open",
            "one local evidence item has already been opened; retrieve more evidence or write/abstain if the local evidence is sufficient.",
            obs,
            remaining_fields,
            extra_actions=["retrieve_evidence"],
            keep_extra_when_unsupported=True,
        )

    if opened_count >= 1:
        return claim_phase_decision(
            "claim_ready",
            "retrieved evidence has been opened; write or abstain missing fields instead of continuing evidence navigation.",
            obs,
            remaining_fields,
        )

    if retrieved_count < 3 and opened_count < 4:
        return claim_phase_decision(
            "evidence_opening",
            "retrieval results are available; open evidence, optionally retrieve again within budget, or write claims.",
            obs,
            remaining_fields,
            extra_actions=["open_evidence", "retrieve_evidence"],
            keep_extra_when_unsupported=True,
        )

    return claim_phase_decision(
        "claim_ready",
        "enough evidence has been retrieved/opened; write claims instead of continuing broad retrieval.",
        obs,
        remaining_fields,
    )


def action_allowed(action: dict[str, Any] | None, mask: dict[str, Any]) -> bool:
    if not isinstance(action, dict):
        return False
    return str(action.get("action", "")) in set(mask.get("allowed_actions") or [])


def action_names(actions: list[Any]) -> list[str]:
    return [str(action.get("action", "")) for action in actions if isinstance(action, dict)]


def tool_result_names(results: list[Any]) -> list[str]:
    return [str(result.get("tool", "")) for result in results if isinstance(result, dict)]


def has_successful_crop(results: list[Any]) -> bool:
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("tool") not in {"crop_region", "crop_target", "crop_image"}:
            continue
        if result.get("error"):
            continue
        if result.get("crop_path") or result.get("bbox"):
            return True
    return False


def decision(phase: str, allowed: list[str], reason: str, obs: dict[str, Any]) -> dict[str, Any]:
    schema_allowed = set(schema_actions(obs.get("tool_schema")))
    allowed_set = {action for action in allowed if action in ALLOWED_ACTIONS and action in schema_allowed}
    ordered_allowed = [action for action in ACTION_ORDER if action in allowed_set]
    return {
        "enabled": True,
        "phase": phase,
        "allowed_actions": ordered_allowed,
        "blocked_actions": [action for action in schema_actions(obs.get("tool_schema")) if action not in allowed_set],
        "reason": reason,
        "step": len(obs.get("history") or []),
        "tool_schema": obs.get("tool_schema"),
    }


def claim_phase_decision(
    phase: str,
    reason: str,
    obs: dict[str, Any],
    remaining_fields: list[Any],
    *,
    extra_actions: list[str] | None = None,
    keep_extra_when_unsupported: bool = False,
) -> dict[str, Any]:
    next_field = str(remaining_fields[0]) if remaining_fields else ""
    priority_field = caption_text_priority_field(obs, remaining_fields)
    field_for_support = priority_field or next_field
    support = visible_evidence_supports_field(obs, field_for_support) if field_for_support else None
    extra = list(extra_actions or [])
    if priority_field:
        allowed = ["write_claim"]
        reason = (
            reason
            + " caption_text is still missing and supported by current visible/opened evidence; "
            + "write_claim must complete caption_text before other claim fields."
        )
        return decision(phase, allowed, reason, obs)
    if support is True:
        allowed = extra + ["write_claim", "abstain_claim"]
        reason = (
            reason
            + f" next_missing={next_field} is supported by current visible/opened evidence; write_claim is allowed, abstain_claim remains available if the sampled field lacks support."
        )
    elif support is False:
        allowed = (extra if keep_extra_when_unsupported else []) + ["abstain_claim"]
        reason = (
            reason
            + f" next_missing={next_field} has no current visible/opened evidence with allowed_fields support; write_claim is blocked."
        )
    else:
        allowed = extra + ["write_claim", "abstain_claim"]
        reason = (
            reason
            + f" next_missing={next_field}; no field-policy metadata is visible, so write_claim or abstain_claim remain available."
        )
    return decision(phase, allowed, reason, obs)


def caption_text_priority_field(obs: dict[str, Any], remaining_fields: list[Any]) -> str | None:
    remaining = {normalize_claim_field(field) for field in remaining_fields}
    if "caption_text" not in remaining:
        return None
    if visible_evidence_supports_field(obs, "caption_text") is True:
        return "caption_text"
    return None


def visible_evidence_supports_field(obs: dict[str, Any], field: str) -> bool | None:
    normalized = normalize_claim_field(field)
    if not normalized:
        return None
    saw_policy = False
    for item in visible_evidence_policy_items(obs):
        allowed = item.get("adjudicated_claim_allowed_fields") or item.get("claim_allowed_fields") or item.get("allowed_fields")
        if not allowed:
            continue
        saw_policy = True
        allowed_set = {normalize_claim_field(value) for value in allowed}
        if normalized in allowed_set:
            return True
    return False if saw_policy else None


def visible_evidence_policy_items(obs: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in obs.get("visible_evidence") or []:
        if isinstance(item, dict):
            items.append(item)
    for result in obs.get("tool_results") or []:
        if not isinstance(result, dict):
            continue
        if result.get("tool") == "open_evidence" and result.get("evidence_id") and not result.get("error"):
            items.append(result)
        elif result.get("tool") == "select_evidence":
            for item in result.get("selected_evidence") or []:
                if isinstance(item, dict):
                    items.append(item)
    return items


def normalize_claim_field(field: Any) -> str:
    text = str(field or "")
    return FIELD_ALIASES.get(text, text)
