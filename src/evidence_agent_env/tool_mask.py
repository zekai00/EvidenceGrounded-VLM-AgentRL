"""Phase-aware tool availability for executable EvidenceGrounded rollouts."""

from __future__ import annotations

from typing import Any

from .actions import ALLOWED_ACTIONS


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

    if has_claims:
        return decision(
            "claim_continuation",
            ["write_claim", "write_claims_chunk", "abstain_claim"],
            "claims have already started and remaining fields still exist; continue chunk writing or abstain remaining fields before finish.",
            obs,
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
            return decision(
                "evidence_retrieval_after_open",
                ["select_evidence", "retrieve_evidence", "write_claim", "write_claims_chunk", "abstain_claim"],
                "one evidence item has already been opened before retrieval; stop opening repeatedly, then retrieve more evidence or write claims.",
                obs,
            )
        return decision(
            "evidence_retrieval",
            ["select_evidence", "retrieve_evidence", "open_evidence"],
            "target region is cropped; retrieve evidence or select/open one visible local evidence item before writing claims.",
            obs,
        )

    if retrieved_count < 4:
        return decision(
            "evidence_opening",
            ["select_evidence", "open_evidence", "retrieve_evidence", "write_claim", "write_claims_chunk"],
            "retrieval results or visible local evidence are available; open/select evidence, continue bounded retrieval if needed, or write if enough evidence is visible.",
            obs,
        )

    if opened_count < 8:
        return decision(
            "evidence_opening_after_retrieval_cap",
            ["select_evidence", "open_evidence", "write_claim", "write_claims_chunk", "abstain_claim"],
            "the bounded retrieval budget has been used; open/select visible evidence or write claims instead of retrieving again.",
            obs,
        )

    if has_selected or opened_count >= 8 or retrieved_count >= 4:
        return decision(
            "claim_ready",
            ["select_evidence", "write_claim", "write_claims_chunk", "abstain_claim"],
            "enough evidence has been retrieved/opened; select visible evidence if needed, then write claims instead of repeatedly opening evidence.",
            obs,
        )

    return decision(
        "evidence_opening",
        ["select_evidence", "open_evidence", "retrieve_evidence", "write_claim", "write_claims_chunk"],
        "retrieval results are available; open/select evidence, or write if enough evidence is already visible.",
        obs,
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
