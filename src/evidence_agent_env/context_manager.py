"""Budgeted prompt rendering for EvidenceGrounded multi-step agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent_state import AgentState
from .prompting import PromptConfig, build_messages_from_observation, build_prompt_text


@dataclass
class ContextBudget:
    max_history_actions: int = 4
    max_tool_results: int = 4
    max_evidence_per_result: int = 2
    snippet_chars: int = 140
    max_text_chars: int = 12000
    head_text_chars: int = 3500


class ContextManager:
    """Render compact observations without exposing full trajectory history."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        tool_schema: str = "chunked_claim",
        region_selection_hint: bool = True,
        strict_claim_phase_hint: bool = False,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.tool_schema = tool_schema
        self.region_selection_hint = region_selection_hint
        self.strict_claim_phase_hint = strict_claim_phase_hint

    def prompt_config(self) -> PromptConfig:
        return PromptConfig(
            max_history_actions=self.budget.max_history_actions,
            max_tool_results=self.budget.max_tool_results,
            max_evidence_per_result=self.budget.max_evidence_per_result,
            snippet_chars=self.budget.snippet_chars,
            max_text_chars=self.budget.max_text_chars,
            head_text_chars=self.budget.head_text_chars,
            coordinate_info=True,
            tool_schema=self.tool_schema,
            compact_claim_state=True,
            region_selection_hint=self.region_selection_hint,
            strict_claim_phase_hint=self.strict_claim_phase_hint,
        )

    def compact_observation(self, obs: dict[str, Any]) -> dict[str, Any]:
        state = AgentState.from_observation(obs)
        compact = state.to_observation()
        compact["history"] = compact["history"][-self.budget.max_history_actions :]
        compact["tool_results"] = compact["tool_results"][-self.budget.max_tool_results :]
        return compact

    def build_prompt_text(self, obs: dict[str, Any]) -> str:
        return build_prompt_text(self.compact_observation(obs), self.prompt_config())

    def build_messages(
        self,
        obs: dict[str, Any],
        *,
        include_assistant_action: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return build_messages_from_observation(
            self.compact_observation(obs),
            self.prompt_config(),
            include_assistant_action=include_assistant_action,
        )
