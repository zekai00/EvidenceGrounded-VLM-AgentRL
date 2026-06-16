"""verl AgentLoop for executable stepwise EvidenceGrounded rollouts."""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.actions import (  # noqa: E402
    ALLOWED_ACTIONS,
    RETRIEVAL_SCOPES,
    canonical_action_name,
    canonical_retrieval_scope,
)
from evidence_agent_env.env import EvidenceAgentEnv  # noqa: E402
from evidence_agent_env.prompting import (  # noqa: E402
    PromptConfig,
    build_prompt_text,
    simplify_action,
)
from evidence_agent_env.tool_mask import normalize_claim_field, visible_evidence_supports_field  # noqa: E402


class EvidenceStepwiseAgentLoop(AgentLoopBase):
    """Run a VLM policy against EvidenceAgentEnv one JSON action at a time."""

    def __init__(self, *args: Any, prompt_config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length
        prompt_config = prompt_config or {}
        self.prompt_config = PromptConfig(
            max_history_actions=int(prompt_config.get("max_history_actions", 8)),
            max_tool_results=int(prompt_config.get("max_tool_results", 6)),
            max_evidence_per_result=int(prompt_config.get("max_evidence_per_result", 3)),
            snippet_chars=int(prompt_config.get("snippet_chars", 160)),
            max_text_chars=int(prompt_config.get("max_text_chars", 12000)),
            head_text_chars=int(prompt_config.get("head_text_chars", 3000)),
            coordinate_info=bool(prompt_config.get("coordinate_info", True)),
            tool_schema=str(prompt_config.get("tool_schema", "chunked_claim")),
            compact_claim_state=bool(prompt_config.get("compact_claim_state", True)),
            region_selection_hint=bool(prompt_config.get("region_selection_hint", True)),
            strict_claim_phase_hint=bool(prompt_config.get("strict_claim_phase_hint", True)),
            dynamic_tool_schema=bool(prompt_config.get("dynamic_tool_schema", False)),
            field_policy_prompt=bool(prompt_config.get("field_policy_prompt", False)),
            compact_state_update=bool(prompt_config.get("compact_state_update", False)),
        )
        self.state_update_chars = int(prompt_config.get("state_update_chars", kwargs.get("state_update_chars", 900)))
        self.max_action_tokens = int(prompt_config.get("max_action_tokens", kwargs.get("max_action_tokens", 128)))
        self.max_claim_action_tokens = int(
            prompt_config.get("max_claim_action_tokens", kwargs.get("max_claim_action_tokens", 768))
        )
        self.include_tool_images = bool(prompt_config.get("include_tool_images", kwargs.get("include_tool_images", True)))
        self.tool_image_max_pixels = int(
            prompt_config.get("tool_image_max_pixels", kwargs.get("tool_image_max_pixels", 65536))
        )
        self.max_state_regions = int(prompt_config.get("max_state_regions", kwargs.get("max_state_regions", 8)))
        self.auto_finish = bool(prompt_config.get("auto_finish", kwargs.get("auto_finish", True)))
        self.max_steps_override = int(prompt_config.get("max_steps_override", kwargs.get("max_steps_override", 0)))
        self.target_claim_fields = parse_target_claim_fields(prompt_config.get("target_claim_fields"))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        spec = parse_ground_truth((kwargs.get("reward_model") or {}).get("ground_truth"))
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        output_root = Path(os.getenv("EVIDENCE_STEPWISE_VERL_TMP", "/tmp/evidence_grounded_verl_stepwise"))
        output_dir = output_root / f"{spec['task_id']}_{os.getpid()}_{uuid4().hex[:8]}"
        max_steps = max(int(spec.get("max_steps", 12)), self.max_steps_override)
        env = EvidenceAgentEnv(
            spec["tasks_path"],
            spec["evidence_index"],
            output_dir,
            max_steps=max_steps,
            include_gold_regions=False,
            phase_aware_mask=bool(spec.get("phase_aware_mask", True)),
            enforce_tool_mask=bool(spec.get("enforce_tool_mask", True)),
            tool_schema=str(spec.get("tool_schema", self.prompt_config.tool_schema)),
            target_claim_fields=parse_target_claim_fields(spec.get("target_claim_fields")) or self.target_claim_fields,
            reward_mode=str(spec.get("reward_mode", "default")),
            field_policy_hints=bool(spec.get("field_policy_hints", self.prompt_config.field_policy_prompt)),
        )
        obs = env.reset(task_id=str(spec["task_id"]))

        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)
        image_paths_seen = {str(item.get("path")) for item in obs.get("images") or [] if isinstance(item, dict)}

        prompt_ids = await self.apply_chat_template(
            messages,
            images=images,
            videos=videos,
            audios=audios,
            mm_processor_kwargs=mm_processor_kwargs,
        )
        current_ids = list(prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        step_rewards: list[float] = []
        step_actions: list[str] = []
        raw_actions: list[str] = []
        parsed_actions: list[dict[str, Any] | None] = []
        invalid_reasons: list[str] = []
        repair_events: list[dict[str, Any]] = []
        schema_repair_penalty_total = 0.0
        terminated = False
        invalid_streak = 0

        for _step in range(max_steps):
            if len(response_ids) >= self.response_length:
                break
            step_sampling_params = dict(sampling_params)
            phase = (obs.get("tool_mask") or {}).get("phase") if isinstance(obs.get("tool_mask"), dict) else None
            available_actions = set(obs.get("available_actions") or [])
            may_write_claim = bool(available_actions & {"write_claim", "abstain_claim"})
            action_token_limit = (
                self.max_claim_action_tokens
                if phase in {"claim_ready", "claim_continuation"} or may_write_claim
                else self.max_action_tokens
            )
            current_max_tokens = int(step_sampling_params.get("max_tokens") or action_token_limit)
            step_sampling_params["max_tokens"] = min(current_max_tokens, action_token_limit)
            with simple_timer("generate_sequences", metrics):
                llm_output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=current_ids,
                    sampling_params=step_sampling_params,
                    image_data=images,
                    video_data=videos,
                    audio_data=audios,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
            if metrics.get("num_preempted") is None:
                metrics["num_preempted"] = llm_output.num_preempted if llm_output.num_preempted is not None else -1
            else:
                metrics["num_preempted"] += llm_output.num_preempted if llm_output.num_preempted is not None else 0

            raw_generated_ids = list(llm_output.token_ids)
            if not raw_generated_ids:
                break
            raw_text = self.tokenizer.decode(raw_generated_ids, skip_special_tokens=True).strip()
            action, action_text = parse_first_action(raw_text)
            if action is None:
                action = recover_malformed_action_for_current_state(raw_text, obs)
            action = repair_action_for_current_state(action, obs)
            raw_actions.append(raw_text[:1000])
            parsed_actions.append(action if isinstance(action, dict) else None)
            if action_text:
                action_ids = self.tokenizer.encode(action_text + "\n", add_special_tokens=False)
            else:
                action_ids = self.tokenizer.encode(raw_text[:256] + "\n", add_special_tokens=False)
            budget = self.response_length - len(response_ids)
            action_ids = action_ids[:budget]
            if action_ids:
                current_ids += action_ids
                response_ids += action_ids
                response_mask += [1] * len(action_ids)
            action_name = str(action.get("action", "invalid")) if isinstance(action, dict) else "invalid"
            step_actions.append(action_name)

            with simple_timer("tool_calls", metrics):
                obs, reward, terminated, info = env.step(action if action is not None else raw_text)
            repair_keys = list(action.get("_agentloop_repaired_keys") or []) if isinstance(action, dict) else []
            if repair_keys:
                penalty = min(0.20, 0.10 * len(repair_keys))
                reward = float(reward) - penalty
                schema_repair_penalty_total += penalty
                repair_events.append(
                    {
                        "step": len(step_actions),
                        "action": action_name,
                        "keys": repair_keys,
                        "reasons": list(action.get("_agentloop_repair_reasons") or []),
                        "penalty": penalty,
                    }
                )
                result_for_penalty = info.get("result") if isinstance(info, dict) else None
                if isinstance(result_for_penalty, dict):
                    result_for_penalty["_agentloop_repaired_keys"] = repair_keys
                    result_for_penalty["_agentloop_repair_penalty"] = penalty
            step_rewards.append(float(reward))
            result = info.get("result") if isinstance(info, dict) else {}
            if action is None or (isinstance(result, dict) and result.get("error")):
                invalid_streak += 1
                invalid_reasons.append(str(result.get("error") if isinstance(result, dict) else "invalid action"))
            else:
                invalid_streak = 0

            if self.auto_finish and should_auto_finish(obs):
                finish_action = {"action": "finish", "status": "done"}
                finish_text = json.dumps(finish_action, ensure_ascii=False, separators=(",", ":"))
                finish_ids = self.tokenizer.encode(finish_text + "\n", add_special_tokens=False)
                budget = self.response_length - len(response_ids)
                finish_ids = finish_ids[:budget]
                if finish_ids:
                    current_ids += finish_ids
                    response_ids += finish_ids
                    response_mask += [0] * len(finish_ids)
                with simple_timer("tool_calls", metrics):
                    obs, finish_reward, terminated, _finish_info = env.step(finish_action)
                step_rewards.append(float(finish_reward))
                step_actions.append("finish")
                break

            if terminated or invalid_streak >= 2:
                terminated = terminated or invalid_streak >= 2
                break

            state_text = build_state_update_text(
                obs,
                self.prompt_config,
                reward,
                info,
                self.state_update_chars,
                max_state_regions=self.max_state_regions,
            )
            new_images, image_paths_seen = self._new_tool_images(obs, image_paths_seen)
            if new_images:
                state_ids = await self._encode_tool_state_update(state_text, new_images)
                if images is None:
                    images = []
                elif not isinstance(images, list):
                    images = [images]
                images.extend(new_images)
                multi_modal_data["images"] = images
            else:
                state_ids = await self._encode_tool_state_update(state_text, [])
            budget = self.response_length - len(response_ids)
            if budget <= 0:
                break
            state_ids = state_ids[:budget]
            current_ids += state_ids
            response_ids += state_ids
            response_mask += [0] * len(state_ids)
            if response_logprobs:
                response_logprobs += [0.0] * len(state_ids)

        trajectory_metrics = env.trajectory_metrics()
        score = shaped_trajectory_score(trajectory_metrics)
        score = max(-1.0, min(1.0, score - schema_repair_penalty_total))
        debug_record = {
            "task_id": spec.get("task_id"),
            "score": score,
            "step_actions": step_actions,
            "tool_rewards": step_rewards,
            "raw_actions": raw_actions,
            "parsed_actions": parsed_actions,
            "invalid_reasons": invalid_reasons,
            "repair_events": repair_events,
            "schema_repair_penalty_total": schema_repair_penalty_total,
            "terminated": terminated,
            "trajectory_metrics": trajectory_metrics,
            "trajectory_output_dir": str(output_dir),
        }
        debug_path = output_dir / "trajectory_debug.json"
        try:
            debug_path.write_text(json.dumps(debug_record, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            debug_path = None
        metrics_obj = AgentLoopMetrics(
            generate_sequences=float(metrics.get("generate_sequences", 0.0)),
            tool_calls=float(metrics.get("tool_calls", 0.0)),
            compute_score=0.0,
            num_preempted=int(metrics.get("num_preempted", -1)),
        )
        extra_fields = {
            "turn_scores": [score],
            "tool_rewards": step_rewards,
            "step_actions": step_actions,
            "trajectory_metrics": trajectory_metrics,
            "trajectory_output_dir": str(output_dir),
            "trajectory_debug_path": str(debug_path) if debug_path else None,
            "raw_actions": raw_actions,
            "parsed_actions": parsed_actions,
            "invalid_reasons": invalid_reasons,
            "repair_events": repair_events,
            "terminated": terminated,
            "schema_repair_penalty_total": schema_repair_penalty_total,
        }
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=response_logprobs[: self.response_length] if response_logprobs else None,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=mm_processor_kwargs,
            reward_score=score,
            num_turns=max(2, 1 + len(step_actions) * 2),
            metrics=metrics_obj,
            extra_fields=extra_fields,
        )

    def _new_tool_images(
        self,
        obs: dict[str, Any],
        image_paths_seen: set[str],
    ) -> tuple[list[Image.Image], set[str]]:
        if not self.include_tool_images:
            return [], image_paths_seen
        new_images: list[Image.Image] = []
        for image in obs.get("images") or []:
            if not isinstance(image, dict) or image.get("role") != "last_crop":
                continue
            path = image.get("path")
            if not path:
                continue
            key = str(path)
            if key in image_paths_seen:
                continue
            try:
                with Image.open(key) as opened:
                    image = opened.convert("RGB")
                    new_images.append(resize_image_for_max_pixels(image, self.tool_image_max_pixels))
                image_paths_seen.add(key)
            except Exception:
                continue
        return new_images, image_paths_seen

    async def _encode_tool_state_update(self, state_text: str, images: list[Image.Image]) -> list[int]:
        # Close the previous assistant message before appending a tool-state
        # message; otherwise the next vLLM call often emits EOS immediately.
        close_assistant_ids = self.tokenizer.encode("\n<|im_end|>\n", add_special_tokens=False)
        if images:
            content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": state_text})
            state_ids = await self.apply_chat_template(
                [{"role": "tool", "content": content}],
                images=images,
                remove_system_prompt=True,
            )
        else:
            state_ids = await self.apply_chat_template(
                [{"role": "tool", "content": state_text}],
                remove_system_prompt=True,
            )
        return close_assistant_ids + list(state_ids)


def parse_ground_truth(value: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        return json.loads(value)
    raise ValueError("reward_model.ground_truth is required for EvidenceStepwiseAgentLoop")


def parse_target_claim_fields(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        fields = [str(item).strip() for item in value if str(item).strip()]
    else:
        fields = [item.strip() for item in str(value).split(",") if item.strip()]
    return fields or None


def parse_first_action(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Return the first valid action object and its compact JSON text."""
    if not text:
        return None, None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    decoder = json.JSONDecoder()
    starts = [idx for idx, char in enumerate(cleaned) if char == "{"]
    for start in starts:
        try:
            value, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            original_name = value.get("action")
            canonical_name = canonical_action_name(original_name)
            if canonical_name not in ALLOWED_ACTIONS:
                continue
            if canonical_name != original_name:
                repaired = dict(value)
                repaired["action"] = canonical_name
                repaired_keys = list(repaired.get("_agentloop_repaired_keys") or [])
                reasons = list(repaired.get("_agentloop_repair_reasons") or [])
                if "action" not in repaired_keys:
                    repaired_keys.append("action")
                reasons.append(f"action alias normalized from {original_name!r} to {canonical_name!r}")
                repaired["_agentloop_repaired_keys"] = repaired_keys
                repaired["_agentloop_repair_reasons"] = reasons
                value = repaired
            action_text = cleaned[start : start + end]
            return value, action_text
    return None, None


def repair_action_for_current_state(action: dict[str, Any] | None, obs: dict[str, Any]) -> dict[str, Any] | None:
    """Patch narrow argument omissions for an otherwise valid current action.

    The generated text remains unchanged for policy-gradient tokens; this only
    keeps the executable environment from ending a rollout after a recoverable
    schema slip such as {"action":"crop_target","top_k":1}.
    """

    if not isinstance(action, dict):
        return action
    allowed = set(obs.get("available_actions") or [])
    name = str(action.get("action") or "")
    if name not in allowed:
        if name == "finish":
            finish_repair = premature_finish_repair_action(obs)
            if finish_repair:
                return finish_repair
        if name in {"write_claim", "abstain_claim"} and "write_claim" in allowed:
            priority_repair = caption_text_priority_write_action(obs, action)
            if priority_repair:
                return priority_repair
        if name == "write_claim" and "abstain_claim" in allowed:
            abstain_repair = no_support_abstain_action(obs, action)
            if abstain_repair:
                return abstain_repair
        return action
    repaired = dict(action)
    repaired_keys = list(repaired.get("_agentloop_repaired_keys") or [])
    reasons = list(repaired.get("_agentloop_repair_reasons") or [])
    if name == "crop_target" and not repaired.get("region_id") and not repaired.get("bbox"):
        region_id = preferred_region_id_from_obs(obs)
        if region_id:
            repaired["region_id"] = region_id
            repaired_keys.append("region_id")
            reasons.append("missing crop_target.region_id")
    elif name == "open_evidence" and not repaired.get("evidence_id"):
        alias_id = str(repaired.get("id") or repaired.get("evidence") or "").strip()
        evidence_ids = [str(item) for item in (obs.get("visible_evidence_ids") or []) if str(item)]
        if alias_id and alias_id in evidence_ids:
            repaired["evidence_id"] = alias_id
            repaired_keys.append("evidence_id")
            reasons.append("open_evidence id alias")
        elif evidence_ids:
            repaired["evidence_id"] = evidence_ids[0]
            repaired_keys.append("evidence_id")
            reasons.append("missing open_evidence.evidence_id")
    elif name == "retrieve_evidence":
        if not repaired.get("query"):
            repaired["query"] = default_retrieve_query_from_obs(obs)
            repaired_keys.append("query")
            reasons.append("missing retrieve_evidence.query")
        if not repaired.get("scope"):
            repaired["scope"] = "same_document"
            repaired_keys.append("scope")
            reasons.append("missing retrieve_evidence.scope")
        else:
            original_scope = str(repaired.get("scope"))
            if original_scope not in RETRIEVAL_SCOPES:
                canonical_scope = canonical_retrieval_scope(original_scope, repaired)
                repaired["scope"] = canonical_scope
                repaired_keys.append("scope")
                reasons.append(f"invalid retrieve_evidence.scope normalized from {original_scope!r} to {canonical_scope!r}")
        if repaired.get("top_k") is None:
            repaired["top_k"] = 5
            repaired_keys.append("top_k")
            reasons.append("missing retrieve_evidence.top_k")
    elif name == "write_claim":
        canonical_field = normalize_claim_field(repaired.get("field"))
        if canonical_field and canonical_field != repaired.get("field"):
            repaired["field"] = canonical_field
            repaired_keys.append("field")
            reasons.append("write_claim.field canonicalized")
        elif not canonical_field:
            next_field = next_missing_field_from_obs(obs)
            if next_field:
                repaired["field"] = next_field
                repaired_keys.append("field")
                reasons.append("missing write_claim.field filled from next_missing")
        field = str(repaired.get("field") or "")
        next_field = next_missing_field_from_obs(obs)
        priority_repair = caption_text_priority_write_action(obs, repaired)
        if priority_repair and field != "caption_text":
            return priority_repair
        if field and not claim_field_is_remaining(obs, field):
            if (
                next_field
                and "abstain_claim" in allowed
                and visible_evidence_supports_field(obs, next_field) is False
            ):
                repaired = {
                    "action": "abstain_claim",
                    "field": next_field,
                    "reason": "next missing field has no visible/opened evidence support",
                }
                repaired_keys.append("action")
                reasons.append(
                    f"write_claim for completed/non-missing field={field} converted to abstain_claim for next_missing={next_field}"
                )
                repaired["_agentloop_repaired_keys"] = repaired_keys
                repaired["_agentloop_repair_reasons"] = reasons
            return repaired
        if field and visible_evidence_supports_field(obs, field) is False and "abstain_claim" in allowed:
            repaired = {
                "action": "abstain_claim",
                "field": field,
                "reason": "visible/opened evidence does not explicitly support this field",
            }
            repaired_keys.append("action")
            reasons.append("write_claim converted to abstain_claim because no visible evidence supports field")
            repaired["_agentloop_repaired_keys"] = repaired_keys
            repaired["_agentloop_repair_reasons"] = reasons
            return repaired
        evidence_ids = repaired.get("evidence_ids")
        if isinstance(evidence_ids, str):
            evidence_ids = [evidence_ids]
            repaired_keys.append("evidence_ids")
            reasons.append("write_claim.evidence_ids string wrapped as list")
        if isinstance(evidence_ids, list):
            fixed_ids: list[Any] = []
            changed = False
            for item in evidence_ids:
                item_id = str(item)
                if item_id.startswith("r_"):
                    mapped_id = evidence_id_for_region_alias(obs, item_id, str(repaired.get("field") or ""))
                    if mapped_id:
                        fixed_ids.append(mapped_id)
                        changed = True
                        reasons.append(f"write_claim.evidence_ids region alias {item_id}->{mapped_id}")
                        continue
                fixed_ids.append(item)
            if changed:
                repaired["evidence_ids"] = fixed_ids
                repaired_keys.append("evidence_ids")
        supporting_evidence_id = evidence_id_for_field(obs, field) if field else None
        evidence_ids = repaired.get("evidence_ids")
        has_supporting_id = False
        if isinstance(evidence_ids, list):
            has_supporting_id = any(evidence_id_supports_field(obs, str(item), field) for item in evidence_ids)
        if supporting_evidence_id and not has_supporting_id:
            repaired["evidence_ids"] = [supporting_evidence_id]
            repaired_keys.append("evidence_ids")
            reasons.append(f"write_claim.evidence_ids replaced with field-supporting evidence {supporting_evidence_id}")
    elif name == "abstain_claim":
        canonical_field = normalize_claim_field(repaired.get("field"))
        if canonical_field and canonical_field != repaired.get("field"):
            repaired["field"] = canonical_field
            repaired_keys.append("field")
            reasons.append("abstain_claim.field canonicalized")
        elif not canonical_field:
            next_field = next_missing_field_from_obs(obs)
            if next_field:
                repaired["field"] = next_field
                repaired_keys.append("field")
                reasons.append("missing abstain_claim.field filled from next_missing")
        field = str(repaired.get("field") or "")
        next_field = next_missing_field_from_obs(obs)
        priority_repair = caption_text_priority_write_action(obs, repaired)
        if priority_repair:
            return priority_repair
        retarget_field = next_unsupported_field_from_obs(obs)
        if (
            field
            and retarget_field
            and field != retarget_field
            and claim_field_is_remaining(obs, field)
            and visible_evidence_supports_field(obs, retarget_field) is False
        ):
            repaired["field"] = retarget_field
            repaired["reason"] = "current earlier missing field has no visible/opened evidence support"
            repaired_keys.append("field")
            reasons.append(f"abstain_claim field retargeted from field={field} to next_unsupported={retarget_field}")
            field = retarget_field
        if field and not claim_field_is_remaining(obs, field):
            if next_field and visible_evidence_supports_field(obs, next_field) is False:
                repaired["field"] = next_field
                repaired["reason"] = repaired.get("reason") or "next missing field has no visible/opened evidence support"
                repaired_keys.append("field")
                reasons.append(f"abstain_claim field retargeted from completed/non-missing field={field} to next_missing={next_field}")
    if repaired_keys:
        repaired["_agentloop_repaired_keys"] = repaired_keys
        repaired["_agentloop_repair_reasons"] = reasons
    return repaired


def recover_malformed_action_for_current_state(text: str, obs: dict[str, Any]) -> dict[str, Any] | None:
    """Recover narrow malformed claim/finish outputs when the current mask is unambiguous."""

    if not text or '"action"' not in text:
        return None
    allowed = set(obs.get("available_actions") or [])
    if re.search(r'"action"\s*:\s*"finish"', text):
        return premature_finish_repair_action(obs)
    if re.search(r'"action"\s*:\s*"(?:write_claim|claim_one)"', text):
        field = normalize_claim_field(extract_jsonish_string_value(text, "field"))
        original = {"action": "write_claim", "field": field or next_missing_field_from_obs(obs)}
        if "write_claim" in allowed:
            priority_repair = caption_text_priority_write_action(obs, original)
            if priority_repair:
                priority_repair["_agentloop_repair_reasons"] = list(
                    priority_repair.get("_agentloop_repair_reasons") or []
                ) + ["malformed write_claim recovered from raw text"]
                return priority_repair
        if "abstain_claim" in allowed:
            abstain_repair = no_support_abstain_action(obs, original)
            if abstain_repair:
                abstain_repair["_agentloop_repair_reasons"] = list(
                    abstain_repair.get("_agentloop_repair_reasons") or []
                ) + ["malformed write_claim recovered as abstain_claim"]
                return abstain_repair
    if re.search(r'"action"\s*:\s*"abstain_claim"', text):
        field = normalize_claim_field(extract_jsonish_string_value(text, "field"))
        original = {"action": "abstain_claim", "field": field or next_missing_field_from_obs(obs)}
        priority_repair = caption_text_priority_write_action(obs, original)
        if priority_repair:
            priority_repair["_agentloop_repair_reasons"] = list(
                priority_repair.get("_agentloop_repair_reasons") or []
            ) + ["malformed abstain_claim recovered from raw text"]
            return priority_repair
        target_field = field if field and claim_field_is_remaining(obs, field) else next_unsupported_field_from_obs(obs)
        if target_field and "abstain_claim" in allowed and visible_evidence_supports_field(obs, target_field) is False:
            return {
                "action": "abstain_claim",
                "field": target_field,
                "reason": "visible/opened evidence does not explicitly support this field",
                "_agentloop_repaired_keys": ["action", "field"],
                "_agentloop_repair_reasons": ["malformed abstain_claim recovered from raw text"],
            }
    return None


def premature_finish_repair_action(obs: dict[str, Any]) -> dict[str, Any] | None:
    allowed = set(obs.get("available_actions") or [])
    if "finish" in allowed:
        return None
    priority_repair = caption_text_priority_write_action(obs, {"action": "finish"})
    if priority_repair and "write_claim" in allowed:
        priority_repair["_agentloop_repair_reasons"] = list(
            priority_repair.get("_agentloop_repair_reasons") or []
        ) + ["premature finish repaired before claim_state was complete"]
        return priority_repair
    field = next_unsupported_field_from_obs(obs)
    if field and "abstain_claim" in allowed:
        return {
            "action": "abstain_claim",
            "field": field,
            "reason": "visible/opened evidence does not explicitly support this field",
            "_agentloop_repaired_keys": ["action", "field"],
            "_agentloop_repair_reasons": [
                "finish is blocked until claim_state.remaining_fields is empty; converted to abstain_claim"
            ],
        }
    return None


def caption_text_priority_write_action(obs: dict[str, Any], original: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if not caption_text_is_supported_and_missing(obs):
        return None
    evidence_id = evidence_id_for_field(obs, "caption_text")
    value = claim_value_for_field(obs, "caption_text")
    if not evidence_id or not value:
        return None
    original = original or {}
    return {
        "action": "write_claim",
        "field": "caption_text",
        "value": value,
        "evidence_ids": [evidence_id],
        "confidence": safe_confidence(original.get("confidence"), default=0.85),
        "_agentloop_repaired_keys": ["action", "field", "value", "evidence_ids"],
        "_agentloop_repair_reasons": [
            "caption_text is missing and supported; retargeted claim action to write_claim(caption_text)"
        ],
    }


def caption_text_is_supported_and_missing(obs: dict[str, Any]) -> bool:
    claim_state = obs.get("claim_state") if isinstance(obs.get("claim_state"), dict) else {}
    remaining = {normalize_claim_field(item) for item in (claim_state.get("remaining_fields") or [])}
    return "caption_text" in remaining and visible_evidence_supports_field(obs, "caption_text") is True


def safe_confidence(value: Any, *, default: float = 0.85) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, numeric))


def no_support_abstain_action(obs: dict[str, Any], original: dict[str, Any]) -> dict[str, Any] | None:
    canonical_field = normalize_claim_field(original.get("field"))
    next_field = next_missing_field_from_obs(obs)
    field = canonical_field if canonical_field and claim_field_is_remaining(obs, canonical_field) else next_field
    if not field or visible_evidence_supports_field(obs, field) is not False:
        return None
    return {
        "action": "abstain_claim",
        "field": field,
        "reason": "visible/opened evidence does not explicitly support this field",
        "_agentloop_repaired_keys": ["action", "field"],
        "_agentloop_repair_reasons": [
            "write_claim is blocked and current field has no visible/opened evidence support; converted to abstain_claim"
        ],
    }


def preferred_region_id_from_obs(obs: dict[str, Any]) -> str | None:
    regions = slim_regions(obs.get("regions") or [], snippet_chars=0, max_items=1)
    if regions and regions[0].get("region_id"):
        return str(regions[0]["region_id"])
    region_ids = [str(item) for item in (obs.get("available_region_ids") or []) if str(item)]
    return region_ids[0] if region_ids else None


def default_retrieve_query_from_obs(obs: dict[str, Any]) -> str:
    goal = obs.get("goal")
    if isinstance(goal, str) and goal.strip():
        return short_text(goal, 80) or "图注 作品 作者 年代 馆藏"
    return "图注 作品 作者 年代 馆藏"


def evidence_id_for_region_alias(obs: dict[str, Any], alias: str, field: str = "") -> str | None:
    """Map common layout-region aliases to the corresponding visible evidence id."""

    visible = [item for item in (obs.get("visible_evidence") or []) if isinstance(item, dict)]
    if alias == "r_caption_candidate":
        return first_caption_evidence_id(visible)
    if alias == "r_target_candidate":
        return first_visual_evidence_id(visible)
    if alias == "r_context_expand":
        return evidence_id_for_field(obs, field) or first_visible_evidence_id(
            visible,
            ("body", "caption", "local_caption_visual", "same_page_body"),
        )
    return None


def next_missing_field_from_obs(obs: dict[str, Any]) -> str | None:
    claim_state = obs.get("claim_state") if isinstance(obs.get("claim_state"), dict) else {}
    remaining = claim_state.get("remaining_fields") or []
    if not remaining:
        return None
    field = normalize_claim_field(remaining[0])
    return field or None


def next_unsupported_field_from_obs(obs: dict[str, Any]) -> str | None:
    claim_state = obs.get("claim_state") if isinstance(obs.get("claim_state"), dict) else {}
    support_summary = claim_support_summary_for_state(obs, claim_state.get("remaining_fields") or [])
    field = normalize_claim_field(support_summary.get("next_unsupported"))
    return field or None


def claim_field_is_remaining(obs: dict[str, Any], field: str) -> bool:
    normalized = normalize_claim_field(field)
    if not normalized:
        return False
    claim_state = obs.get("claim_state") if isinstance(obs.get("claim_state"), dict) else {}
    remaining = {normalize_claim_field(item) for item in (claim_state.get("remaining_fields") or [])}
    return normalized in remaining


def claim_support_summary_for_state(obs: dict[str, Any], remaining_fields: list[Any]) -> dict[str, Any]:
    supported: list[str] = []
    unsupported: list[str] = []
    unknown: list[str] = []
    for item in remaining_fields:
        field = normalize_claim_field(item)
        if not field:
            continue
        support = visible_evidence_supports_field(obs, field)
        if support is True:
            supported.append(field)
        elif support is False:
            unsupported.append(field)
        else:
            unknown.append(field)
    return {
        "supported_missing": supported,
        "unsupported_missing": unsupported,
        "unknown_support_missing": unknown,
        "next_supported": supported[0] if supported else None,
        "next_unsupported": unsupported[0] if unsupported else None,
    }


def first_caption_evidence_id(visible: list[dict[str, Any]]) -> str | None:
    for item in visible:
        evidence_id = str(item.get("evidence_id") or "")
        citation_level = str(item.get("citation_level") or "")
        role = str(item.get("adjudicated_evidence_role") or "")
        if evidence_id and (
            evidence_id.endswith("_caption")
            or citation_level == "local_caption_visual"
            or role == "local_caption_visual"
        ):
            return evidence_id
    return None


def first_visual_evidence_id(visible: list[dict[str, Any]]) -> str | None:
    for item in visible:
        evidence_id = str(item.get("evidence_id") or "")
        citation_level = str(item.get("citation_level") or "")
        role = str(item.get("adjudicated_evidence_role") or "")
        if evidence_id and (
            evidence_id.endswith("_visual")
            or citation_level == "local_visual"
            or role == "local_visual"
        ):
            return evidence_id
    return None


def first_visible_evidence_id(visible: list[dict[str, Any]], keywords: tuple[str, ...]) -> str | None:
    for item in visible:
        evidence_id = str(item.get("evidence_id") or "")
        citation_level = str(item.get("citation_level") or "")
        role = str(item.get("adjudicated_evidence_role") or "")
        haystack = f"{evidence_id} {citation_level} {role}"
        if evidence_id and any(keyword in haystack for keyword in keywords):
            return evidence_id
    return None


def should_auto_finish(obs: dict[str, Any]) -> bool:
    mask = obs.get("tool_mask") if isinstance(obs.get("tool_mask"), dict) else {}
    available = list(obs.get("available_actions") or [])
    claim_state = obs.get("claim_state") or {}
    return (
        mask.get("phase") == "finish_ready"
        and available == ["finish"]
        and not (claim_state.get("remaining_fields") or [])
    )


def parse_action_object(text: str) -> dict[str, Any] | None:
    action, _action_text = parse_first_action(text)
    return action


def extract_jsonish_string_value(text: str, key: str) -> str | None:
    match = re.search(rf'"{re.escape(key)}"\s*:\s*"(?P<value>[^"]{{1,160}})', text)
    if not match:
        return None
    return match.group("value").strip()


def resize_image_for_max_pixels(image: Image.Image, max_pixels: int) -> Image.Image:
    if max_pixels <= 0:
        return image.copy()
    width, height = image.size
    if width * height <= max_pixels:
        return image.copy()
    scale = (float(max_pixels) / float(width * height)) ** 0.5
    target = (max(1, int(width * scale)), max(1, int(height * scale)))
    resized = image.copy()
    resized.thumbnail(target, Image.Resampling.LANCZOS)
    return resized


def build_state_update_text(
    obs: dict[str, Any],
    config: PromptConfig,
    reward: float,
    info: dict[str, Any],
    max_chars: int,
    *,
    max_state_regions: int = 8,
) -> str:
    recent_results = obs.get("tool_results") or []
    recent_result_raw = recent_results[-1] if recent_results else None
    allowed = obs.get("available_actions") or []
    phase = (obs.get("tool_mask") or {}).get("phase") if isinstance(obs.get("tool_mask"), dict) else None
    last_action = simplify_action((obs.get("history") or [{}])[-1]) if obs.get("history") else None
    claim_state = obs.get("claim_state") or {}
    claim_state = {
        "written_fields": claim_state.get("written_fields") or [],
        "abstained_fields": claim_state.get("abstained_fields") or [],
        "remaining_fields": claim_state.get("remaining_fields") or [],
    }
    support_summary = claim_support_summary_for_state(obs, claim_state.get("remaining_fields") or [])
    valid_crop_count = int(obs.get("valid_crop_count") or 0)
    include_full_regions = phase in {"region_discovery", "region_selection"} or valid_crop_count <= 0
    state_regions = (
        slim_regions(obs.get("regions") or [], snippet_chars=50, max_items=max_state_regions)
        if include_full_regions
        else []
    )
    claim_phase = phase in {"claim_ready", "claim_continuation", "finish_ready"} or (
        isinstance(phase, str) and phase.startswith("evidence_") and may_write_claim_from_allowed(allowed)
    )
    priority_field = "caption_text" if caption_text_is_supported_and_missing(obs) else support_summary.get("next_supported")
    evidence_snippet_chars = 0 if claim_phase else 50
    focus_fields = [priority_field] if priority_field else (support_summary.get("supported_missing") or [])[:1]
    focus_snippet_chars = 140 if claim_phase and focus_fields else 0
    visible_evidence_items = (
        focused_visible_evidence(obs, focus_fields, limit=2)
        if claim_phase
        else [item for item in (obs.get("visible_evidence") or []) if isinstance(item, dict)][:6]
    )
    state = {
        "step": obs.get("step") if obs.get("step") is not None else len(obs.get("history") or []),
        "last_action": last_action,
        "last_reward": round(float(reward), 4),
        "available_actions": allowed,
        "phase": phase,
        "last_tool_result": (
            {"tool": recent_result_raw.get("tool")} if claim_phase and isinstance(recent_result_raw, dict)
            else slim_tool_result(recent_result_raw, snippet_chars=70, max_items=3)
        ),
        "available_regions": state_regions,
        "available_region_ids": [] if claim_phase else (obs.get("available_region_ids") or []),
        "selected_evidence_ids": [] if claim_phase else (obs.get("selected_evidence_ids") or []),
        "visible_evidence_ids": [
            str(item.get("evidence_id")) for item in visible_evidence_items if isinstance(item, dict) and item.get("evidence_id")
        ],
        "visible_evidence": [
            slim_evidence(
                item,
                evidence_snippet_chars,
                focus_fields=focus_fields,
                focus_snippet_chars=focus_snippet_chars,
            )
            for item in visible_evidence_items
        ],
        "valid_crop_count": valid_crop_count,
        "last_crop": compact_crop_ref(obs.get("last_crop_path")),
        "claim_state": claim_state,
        "claim_support": support_summary,
        "claim_priority": priority_field,
    }
    phase_rule = ""
    remaining_fields = claim_state.get("remaining_fields") or []
    if phase in {"claim_ready", "claim_continuation"}:
        if config.compact_state_update:
            phase_rule = "claim_one;no_nav;finish_if_empty"
        else:
            phase_rule = (
                "write or abstain exactly one missing field; never stop while missing is non-empty; prefer next_supported if present, otherwise next_missing; "
                "write_claim.evidence_ids must explicitly support the written field; no retrieve/open/crop; finish only when no missing fields"
            )
        if remaining_fields:
            phase_rule += f"; next={remaining_fields[0]}"
        if priority_field:
            phase_rule += f"; must={priority_field}"
    elif phase in {"region_selection"}:
        phase_rule = "crop one region_id from regions/region_ids; prefer figure_candidate/target_rank; no evidence/write/finish"
    elif phase and str(phase).startswith("evidence"):
        phase_rule = (
            "evidence_or_claim;finish_if_empty"
            if config.compact_state_update
            else "retrieve/open evidence from allowed ids; write a supported missing field when current evidence explicitly supports it; never stop while missing is non-empty"
        )
        if remaining_fields:
            phase_rule += f"; next={remaining_fields[0]}"
        if priority_field:
            phase_rule += f"; must={priority_field}"
    if config.compact_state_update:
        phase_rule += "; ids:r=crop,v13=evidence; no_placeholder"
    else:
        phase_rule = (
            phase_rule
            + "; ID types: r_* are region_ids only for crop_target.region_id; "
            + "v13_t_* are evidence_ids only for open_evidence.evidence_id/write_claim.evidence_ids; "
            + "never put r_* in evidence_ids; "
            + "placeholder values like 无/未注明/不详/unknown/not mentioned/N/A must use abstain_claim, not write_claim"
        )
    state["templates"] = next_action_templates_for_state(obs, claim_state)
    formatter = format_compact_state_update_text if config.compact_state_update else format_state_update_text
    text = formatter(state, phase_rule)
    if len(text) <= max_chars:
        return text
    state["last_tool_result"] = slim_tool_result(recent_result_raw, snippet_chars=40, max_items=2)
    if include_full_regions:
        state["available_regions"] = slim_regions(obs.get("regions") or [], snippet_chars=30, max_items=max_state_regions)
    visible_evidence_items = (
        focused_visible_evidence(obs, focus_fields, limit=2)
        if claim_phase
        else [item for item in (obs.get("visible_evidence") or []) if isinstance(item, dict)][:4]
    )
    state["visible_evidence_ids"] = [
        str(item.get("evidence_id")) for item in visible_evidence_items if isinstance(item, dict) and item.get("evidence_id")
    ][:5]
    fallback_evidence_snippet_chars = 0 if claim_phase else 35
    state["visible_evidence"] = [
        slim_evidence(
            item,
            fallback_evidence_snippet_chars,
            focus_fields=focus_fields,
            focus_snippet_chars=120 if claim_phase and focus_fields else 0,
        )
        for item in visible_evidence_items
    ]
    text = formatter(state, phase_rule)
    if len(text) <= max_chars:
        return text
    state["last_tool_result"] = slim_tool_result(recent_result_raw, snippet_chars=20, max_items=1)
    final_evidence_snippet_chars = 0 if claim_phase else 20
    visible_evidence_items = (
        focused_visible_evidence(obs, focus_fields, limit=1)
        if claim_phase
        else [item for item in (obs.get("visible_evidence") or []) if isinstance(item, dict)][:3]
    )
    state["visible_evidence_ids"] = [
        str(item.get("evidence_id")) for item in visible_evidence_items if isinstance(item, dict) and item.get("evidence_id")
    ][:5]
    state["visible_evidence"] = [
        slim_evidence(
            item,
            final_evidence_snippet_chars,
            focus_fields=focus_fields,
            focus_snippet_chars=90 if claim_phase and focus_fields else 0,
        )
        for item in visible_evidence_items
    ]
    if include_full_regions:
        state["available_regions"] = slim_regions(obs.get("regions") or [], snippet_chars=0, max_items=max_state_regions)
    text = formatter(state, phase_rule)
    if len(text) <= max_chars:
        return text
    state["last_tool_result"] = {"tool": recent_result_raw.get("tool")} if isinstance(recent_result_raw, dict) else None
    if include_full_regions:
        state["available_regions"] = slim_regions(obs.get("regions") or [], snippet_chars=0, max_items=max_state_regions)
    text = formatter(state, phase_rule)
    if len(text) <= max_chars or not config.compact_state_update or not include_full_regions:
        return text
    for keep in [6, 4, 2]:
        state["available_regions"] = slim_regions(
            obs.get("regions") or [],
            snippet_chars=0,
            max_items=min(keep, max_state_regions),
        )
        text = formatter(state, phase_rule)
        if len(text) <= max_chars:
            return text
    state["available_regions"] = []
    ranked_regions = slim_regions(obs.get("regions") or [], snippet_chars=0, max_items=min(6, max_state_regions))
    ranked_ids = [item.get("region_id") for item in ranked_regions if item.get("region_id")]
    state["available_region_ids"] = ranked_ids or (obs.get("available_region_ids") or [])[: min(6, max_state_regions)]
    return formatter(state, phase_rule)


def format_state_update_text(state: dict[str, Any], phase_rule: str) -> str:
    return (
        "\n\n[TOOL_STATE_UPDATE]\n"
        + "硬约束：下一步 action 必须属于 available_actions；只输出一个非空 JSON 对象；不要输出空字符串；done 不是 action，只有 finish 出现在 available_actions 时才能用 {\"action\":\"finish\",\"status\":\"done\"}。\n"
        + f"阶段规则：{phase_rule}\n"
        + json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        + "\n继续执行，只输出一个 JSON 对象。\n"
    )


def may_write_claim_from_allowed(allowed: Any) -> bool:
    return bool(set(allowed or []) & {"write_claim", "abstain_claim"})


def format_compact_state_update_text(state: dict[str, Any], phase_rule: str) -> str:
    claim_state = state.get("claim_state") if isinstance(state.get("claim_state"), dict) else {}
    claim_support = state.get("claim_support") if isinstance(state.get("claim_support"), dict) else {}
    claim_priority = state.get("claim_priority")
    regions = state.get("available_regions") or []
    phase = state.get("phase")
    claimish = phase in {"claim_ready", "claim_continuation", "finish_ready"} or (
        isinstance(phase, str) and phase.startswith("evidence_") and may_write_claim_from_allowed(state.get("available_actions"))
    )
    evidence = state.get("visible_evidence") or []
    if claimish:
        evidence = [
            drop_empty_state_items({"evidence_id": item.get("evidence_id"), "snippet": item.get("snippet")})
            for item in evidence
            if isinstance(item, dict)
        ]
    compact = drop_empty_state_items({
        "step": state.get("step"),
        "phase": phase,
        "allowed": state.get("available_actions") or [],
        "rule": phase_rule,
        "regions": regions,
        "region_ids": [] if regions else (state.get("available_region_ids") or []),
        "evidence_ids": [] if claimish else (state.get("visible_evidence_ids") or []),
        "evidence": evidence,
        "selected": state.get("selected_evidence_ids") or [],
        "crop": state.get("last_crop"),
        "claim": {
            "missing": claim_state.get("remaining_fields") or [],
            "must": claim_priority,
        },
        "templates": state.get("templates") or [],
    })
    return (
        "\n\n[STATE]\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + "\nJSON only. claim.must first. finish only when claim.missing=[]; r_*=region, v13_t_*=evidence.\n"
    )


def compact_last_result_for_state(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    tool = result.get("tool")
    if tool in {"propose_regions", "inspect_page"} and result.get("regions"):
        return {"tool": tool, "region_count": len(result.get("regions") or [])}
    return result


def drop_empty_state_items(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            slim = drop_empty_state_items(item)
            if slim is None or slim == [] or slim == {}:
                continue
            out[key] = slim
        return out
    if isinstance(value, list):
        return [drop_empty_state_items(item) for item in value]
    return value


def slim_tool_result(result: Any, *, snippet_chars: int, max_items: int) -> Any:
    if not isinstance(result, dict):
        return result
    tool = result.get("tool")
    if result.get("error"):
        return {"tool": tool, "error": result.get("error")}
    if tool in {"propose_regions", "inspect_page"} and result.get("regions"):
        regions = []
        for item in (result.get("regions") or [])[:max_items]:
            if not isinstance(item, dict):
                continue
            regions.append(
                {
                    "region_id": item.get("region_id"),
                    "bbox": item.get("bbox"),
                    "type": item.get("type"),
                    "caption_evidence_id": item.get("caption_evidence_id"),
                    "hint": short_text(item.get("caption_hint") or item.get("nearby_text") or item.get("hint"), snippet_chars),
                }
            )
        return {"tool": tool, "regions": regions}
    if tool in {"crop_image", "crop_region", "crop_target"}:
        keep = {
            "tool": tool,
            "bbox": result.get("bbox"),
            "bbox_iou": round(float(result.get("bbox_iou", -1)), 4) if result.get("bbox_iou") is not None else None,
        }
        if "region_id" in result:
            keep["region_id"] = result.get("region_id")
        return keep
    if tool == "retrieve_evidence":
        return {
            "tool": tool,
            "scope": result.get("scope"),
            "query": short_text(result.get("query"), 60),
            "results": [slim_evidence(item, snippet_chars) for item in (result.get("results") or [])[:max_items]],
            "hit_evidence_ids": result.get("hit_evidence_ids") or [],
        }
    if tool == "select_evidence":
        return {
            "tool": tool,
            "selected_evidence_ids": result.get("selected_evidence_ids") or [],
            "selected_evidence": [
                slim_evidence(item, snippet_chars) for item in (result.get("selected_evidence") or [])[:max_items]
            ],
            "rejected_evidence_ids": result.get("rejected_evidence_ids") or [],
        }
    if tool == "open_evidence":
        allowed = result.get("adjudicated_claim_allowed_fields") or result.get("claim_allowed_fields") or []
        out = {
            "tool": tool,
            "evidence_id": result.get("evidence_id"),
            "authority_level": result.get("authority_level"),
            "citation_level": result.get("citation_level"),
            "allowed_fields": allowed,
            "usable": result.get("usable_for_claim_by_adjudication"),
            "snippet": short_text(
                result.get("display_snippet")
                or result.get("evidence_summary")
                or result.get("text")
                or result.get("raw_chunk_text"),
                snippet_chars,
            ),
        }
        return drop_empty_state_items(out)
    if tool in {"write_claims_chunk", "write_claims_batch"}:
        return {
            "tool": tool,
            "claim_state": result.get("claim_state"),
        }
    if tool in {"write_claim", "abstain_claim"}:
        claim = result.get("claim") if isinstance(result.get("claim"), dict) else {}
        return {
            "tool": tool,
            "field": claim.get("field"),
            "claim_state": result.get("claim_state"),
        }
    return {key: result.get(key) for key in list(result)[:6]}


def slim_regions(regions: list[Any], *, snippet_chars: int, max_items: int) -> list[dict[str, Any]]:
    slimmed: list[dict[str, Any]] = []
    ordered = sorted(
        enumerate(regions),
        key=lambda pair: (
            int(pair[1].get("target_region_rank") or 999) if isinstance(pair[1], dict) else 999,
            -float(pair[1].get("target_caption_match_score") or 0.0) if isinstance(pair[1], dict) else 0.0,
            -float(pair[1].get("caption_link_score") or 0.0) if isinstance(pair[1], dict) else 0.0,
            pair[0],
        ),
    )
    for _, item in ordered[: max(0, max_items)]:
        if not isinstance(item, dict):
            continue
        slim: dict[str, Any] = {
            "region_id": item.get("region_id"),
            "bbox": item.get("bbox"),
            "type": item.get("type"),
            "caption_evidence_id": item.get("caption_evidence_id"),
        }
        if item.get("target_region_rank") is not None:
            slim["rank"] = item.get("target_region_rank")
        if item.get("target_caption_match_score") is not None:
            slim["match"] = item.get("target_caption_match_score")
        if item.get("caption_link_score") is not None:
            slim["link"] = item.get("caption_link_score")
        if snippet_chars > 0:
            hint = item.get("caption_hint") or item.get("nearby_text") or item.get("hint")
            slim["hint"] = short_text(hint, snippet_chars)
        slimmed.append(slim)
    return slimmed


def compact_crop_ref(path: Any) -> dict[str, Any] | None:
    if not path:
        return None
    return {"available": True, "role": "last_crop", "file": Path(str(path)).name}


def focused_visible_evidence(obs: dict[str, Any], focus_fields: list[Any], *, limit: int) -> list[dict[str, Any]]:
    visible = [item for item in (obs.get("visible_evidence") or []) if isinstance(item, dict)]
    normalized_focus = {normalize_claim_field(field) for field in focus_fields if normalize_claim_field(field)}
    if not normalized_focus:
        return visible[:limit]
    focused: list[dict[str, Any]] = []
    for item in visible:
        allowed = item.get("adjudicated_claim_allowed_fields") or item.get("claim_allowed_fields") or item.get("allowed_fields") or []
        allowed_set = {normalize_claim_field(field) for field in allowed}
        if allowed_set & normalized_focus:
            focused.append(item)
    return (focused or visible)[:limit]


def slim_evidence(
    item: Any,
    snippet_chars: int,
    *,
    focus_fields: list[Any] | None = None,
    focus_snippet_chars: int = 0,
) -> Any:
    if not isinstance(item, dict):
        return item
    allowed = item.get("adjudicated_claim_allowed_fields") or item.get("claim_allowed_fields") or []
    normalized_allowed = {normalize_claim_field(field) for field in allowed}
    normalized_focus = {normalize_claim_field(field) for field in (focus_fields or [])}
    effective_snippet_chars = snippet_chars
    if focus_snippet_chars > snippet_chars and normalized_allowed & normalized_focus:
        effective_snippet_chars = focus_snippet_chars
    return drop_empty_state_items({
        "evidence_id": item.get("evidence_id"),
        "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
        "citation_level": item.get("citation_level"),
        "allowed_fields": allowed,
        "usable": item.get("usable_for_claim_by_adjudication"),
        "snippet": short_text(
            item.get("display_snippet") or item.get("evidence_summary") or item.get("text"),
            effective_snippet_chars,
        ),
    })


def next_action_templates_for_state(obs: dict[str, Any], claim_state: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = set(obs.get("available_actions") or [])
    phase = (obs.get("tool_mask") or {}).get("phase") if isinstance(obs.get("tool_mask"), dict) else None
    templates: list[dict[str, Any]] = []
    if "crop_target" in allowed:
        region_id = preferred_region_id_from_obs(obs)
        templates.append({"action": "crop_target", "region_id": region_id or "r_target_candidate"})
        return templates
    evidence_ids = [str(item) for item in (obs.get("visible_evidence_ids") or []) if str(item)]
    caption_id = next((item for item in evidence_ids if "caption" in item), evidence_ids[0] if evidence_ids else "")
    if phase == "local_evidence_opening" and "open_evidence" in allowed and caption_id:
        templates.append({"action": "open_evidence", "evidence_id": caption_id})
        return templates[:3]
    priority_caption = caption_text_is_supported_and_missing(obs)
    if priority_caption and "write_claim" in allowed:
        evidence_id = evidence_id_for_field(obs, "caption_text")
        value = claim_value_for_field(obs, "caption_text")
        if evidence_id and value:
            templates.append(
                {
                    "action": "write_claim",
                    "field": "caption_text",
                    "value": value,
                    "evidence_ids": [evidence_id],
                    "confidence": 0.85,
                }
            )
            return templates[:3]
    if "retrieve_evidence" in allowed:
        templates.append({"action": "retrieve_evidence", "query": "图注 作品 作者 年代 馆藏", "scope": "same_document", "top_k": 5})
    remaining = [str(item) for item in (claim_state.get("remaining_fields") or []) if str(item)]
    next_field = remaining[0] if remaining else ""
    support_summary = claim_support_summary_for_state(obs, remaining)
    next_supported = str(support_summary.get("next_supported") or "")
    next_unsupported = str(support_summary.get("next_unsupported") or "")
    write_field = next_supported or next_field
    supporting_evidence_id = evidence_id_for_field(obs, write_field) if write_field else None
    if write_field and "write_claim" in allowed:
        if supporting_evidence_id:
            templates.append(
                {
                    "action": "write_claim",
                    "field": write_field,
                    "value": "...",
                    "evidence_ids": [supporting_evidence_id],
                    "confidence": 0.8,
                }
            )
    abstain_field = next_unsupported or (next_field if not supporting_evidence_id else "")
    if abstain_field and "abstain_claim" in allowed:
        templates.append(
            {
                "action": "abstain_claim",
                "field": abstain_field,
                "reason": "visible/opened evidence does not explicitly support this field",
            }
        )
    if "finish" in allowed and not remaining:
        templates.append({"action": "finish", "status": "done"})
    return templates[:3]


def evidence_id_for_field(obs: dict[str, Any], field: str) -> str | None:
    normalized = normalize_claim_field(field)
    if not normalized:
        return None
    for item in obs.get("visible_evidence") or []:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "")
        allowed = (
            item.get("adjudicated_claim_allowed_fields")
            or item.get("claim_allowed_fields")
            or item.get("allowed_fields")
            or []
        )
        if evidence_id and normalized in {normalize_claim_field(value) for value in allowed}:
            return evidence_id
    return None


def claim_value_for_field(obs: dict[str, Any], field: str) -> str | None:
    normalized = normalize_claim_field(field)
    evidence_id = evidence_id_for_field(obs, normalized)
    if not evidence_id:
        return None
    for item in obs.get("visible_evidence") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("evidence_id") or "") != evidence_id:
            continue
        text = item.get("display_snippet") or item.get("evidence_summary") or item.get("text")
        if normalized == "caption_text":
            return short_text(text, 220)
        return short_text(text, 120)
    return None


def evidence_id_supports_field(obs: dict[str, Any], evidence_id: str, field: str) -> bool:
    normalized = normalize_claim_field(field)
    if not evidence_id or not normalized:
        return False
    for item in obs.get("visible_evidence") or []:
        if not isinstance(item, dict):
            continue
        if str(item.get("evidence_id") or "") != evidence_id:
            continue
        allowed = (
            item.get("adjudicated_claim_allowed_fields")
            or item.get("claim_allowed_fields")
            or item.get("allowed_fields")
            or []
        )
        return normalized in {normalize_claim_field(value) for value in allowed}
    return False


def short_text(value: Any, max_chars: int) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def shaped_trajectory_score(metrics: dict[str, Any]) -> float:
    final_reward = float(metrics.get("final_reward", 0.0))
    finish = 1.0 if metrics.get("finish") else 0.0
    crop = 1.0 if metrics.get("crop_success") else 0.0
    success = 1.0 if metrics.get("trajectory_success") else 0.0
    invalid_rate = float(metrics.get("invalid_step_rate", 0.0))
    premature_finish = float(metrics.get("premature_finish_count", 0.0))
    steps = float(metrics.get("steps", 0.0))
    score = final_reward + 0.20 * finish + 0.15 * crop + 0.20 * success
    score -= 0.40 * invalid_rate
    score -= 0.35 * min(1.0, premature_finish)
    score -= 0.02 * max(0.0, steps - 10.0)
    return max(-1.0, min(1.0, float(score)))
