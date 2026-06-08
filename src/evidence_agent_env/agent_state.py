"""Compact state objects for EvidenceGrounded agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tools.claim_tools import claim_state


@dataclass
class AgentState:
    task_id: str
    goal: str | None = None
    source_file: str | None = None
    page: int | None = None
    step: int | None = None
    phase: str = "locate"
    images: list[dict[str, Any]] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    draft_claims: list[dict[str, Any]] = field(default_factory=list)
    selected_evidence_ids: list[str] = field(default_factory=list)
    regions: dict[str, Any] = field(default_factory=dict)
    available_region_ids: list[str] = field(default_factory=list)
    visible_evidence_ids: list[str] = field(default_factory=list)
    valid_crop_count: int = 0
    last_crop_path: str | None = None
    available_actions: list[str] = field(default_factory=list)
    tool_mask: dict[str, Any] = field(default_factory=dict)
    tool_schema: str | None = None

    @classmethod
    def from_observation(cls, obs: dict[str, Any]) -> "AgentState":
        return cls(
            task_id=str(obs.get("task_id")),
            goal=obs.get("goal"),
            source_file=obs.get("source_file"),
            page=obs.get("page"),
            step=obs.get("step"),
            phase=infer_phase(obs),
            images=list(obs.get("images") or []),
            history=list(obs.get("history") or []),
            tool_results=list(obs.get("tool_results") or []),
            draft_claims=list(obs.get("draft_claims") or []),
            selected_evidence_ids=[str(item) for item in obs.get("selected_evidence_ids") or []],
            regions=dict(obs.get("regions") or {}),
            available_region_ids=[str(item) for item in obs.get("available_region_ids") or []],
            visible_evidence_ids=[str(item) for item in obs.get("visible_evidence_ids") or []],
            valid_crop_count=int(obs.get("valid_crop_count") or 0),
            last_crop_path=obs.get("last_crop_path"),
            available_actions=[str(item) for item in obs.get("available_actions") or []],
            tool_mask=dict(obs.get("tool_mask") or {}),
            tool_schema=obs.get("tool_schema"),
        )

    def to_observation(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "source_file": self.source_file,
            "page": self.page,
            "step": self.step,
            "phase": self.phase,
            "images": self.images,
            "history": self.history,
            "tool_results": self.tool_results,
            "draft_claims": self.draft_claims,
            "claim_state": claim_state(self.draft_claims),
            "selected_evidence_ids": self.selected_evidence_ids,
            "regions": self.regions,
            "available_region_ids": self.available_region_ids,
            "visible_evidence_ids": self.visible_evidence_ids,
            "valid_crop_count": self.valid_crop_count,
            "last_crop_path": self.last_crop_path,
            "available_actions": self.available_actions,
            "tool_mask": self.tool_mask,
            "tool_schema": self.tool_schema,
        }


def infer_phase(obs: dict[str, Any]) -> str:
    history = [item for item in obs.get("history") or [] if isinstance(item, dict)]
    actions = [str(item.get("action")) for item in history]
    claims = obs.get("draft_claims") or []
    if actions and actions[-1] == "finish":
        return "done"
    if claims:
        return "write"
    if "open_evidence" in actions or "retrieve_evidence" in actions:
        return "evidence"
    if "crop_region" in actions or "crop_image" in actions or "select_evidence" in actions:
        return "crop"
    return "locate"
