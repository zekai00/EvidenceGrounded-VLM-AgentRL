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

from evidence_agent_env.actions import ALLOWED_ACTIONS  # noqa: E402
from evidence_agent_env.env import EvidenceAgentEnv  # noqa: E402
from evidence_agent_env.prompting import (  # noqa: E402
    PromptConfig,
    build_prompt_text,
    simplify_action,
)


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
        self.target_claim_fields = parse_target_claim_fields(prompt_config.get("target_claim_fields"))

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        spec = parse_ground_truth((kwargs.get("reward_model") or {}).get("ground_truth"))
        metrics: dict[str, Any] = {}
        request_id = uuid4().hex
        output_root = Path(os.getenv("EVIDENCE_STEPWISE_VERL_TMP", "/tmp/evidence_grounded_verl_stepwise"))
        output_dir = output_root / f"{spec['task_id']}_{os.getpid()}_{uuid4().hex[:8]}"
        env = EvidenceAgentEnv(
            spec["tasks_path"],
            spec["evidence_index"],
            output_dir,
            max_steps=int(spec.get("max_steps", 12)),
            include_gold_regions=False,
            phase_aware_mask=bool(spec.get("phase_aware_mask", True)),
            enforce_tool_mask=bool(spec.get("enforce_tool_mask", True)),
            tool_schema=str(spec.get("tool_schema", self.prompt_config.tool_schema)),
            target_claim_fields=parse_target_claim_fields(spec.get("target_claim_fields")) or self.target_claim_fields,
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
        schema_repair_penalty_total = 0.0
        terminated = False
        invalid_streak = 0

        max_steps = int(spec.get("max_steps", 12))
        for _step in range(max_steps):
            if len(response_ids) >= self.response_length:
                break
            step_sampling_params = dict(sampling_params)
            phase = (obs.get("tool_mask") or {}).get("phase") if isinstance(obs.get("tool_mask"), dict) else None
            available_actions = set(obs.get("available_actions") or [])
            may_write_claim = bool(available_actions & {"write_claims_chunk", "write_claims_batch", "write_claim"})
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
            action = repair_action_for_current_state(action, obs)
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
                result_for_penalty = info.get("result") if isinstance(info, dict) else None
                if isinstance(result_for_penalty, dict):
                    result_for_penalty["_agentloop_repaired_keys"] = repair_keys
                    result_for_penalty["_agentloop_repair_penalty"] = penalty
            step_rewards.append(float(reward))
            result = info.get("result") if isinstance(info, dict) else {}
            if action is None or (isinstance(result, dict) and result.get("error")):
                invalid_streak += 1
            else:
                invalid_streak = 0

            if self.auto_finish and not terminated and should_auto_finish(obs):
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
                state_ids = await self._encode_tool_state_with_images(state_text, new_images)
                if images is None:
                    images = []
                elif not isinstance(images, list):
                    images = [images]
                images.extend(new_images)
                multi_modal_data["images"] = images
            else:
                state_ids = self.tokenizer.encode(state_text, add_special_tokens=False)
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

    async def _encode_tool_state_with_images(self, state_text: str, images: list[Image.Image]) -> list[int]:
        content: list[dict[str, Any]] = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": state_text})
        return await self.apply_chat_template(
            [{"role": "tool", "content": content}],
            images=images,
            remove_system_prompt=True,
        )


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
        if isinstance(value, dict) and value.get("action") in ALLOWED_ACTIONS:
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
        if repaired.get("top_k") is None:
            repaired["top_k"] = 5
            repaired_keys.append("top_k")
            reasons.append("missing retrieve_evidence.top_k")
    if repaired_keys:
        repaired["_agentloop_repaired_keys"] = repaired_keys
        repaired["_agentloop_repair_reasons"] = reasons
    return repaired


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
    valid_crop_count = int(obs.get("valid_crop_count") or 0)
    include_full_regions = phase in {"region_discovery", "region_selection"} or valid_crop_count <= 0
    state_regions = (
        slim_regions(obs.get("regions") or [], snippet_chars=50, max_items=max_state_regions)
        if include_full_regions
        else []
    )
    state = {
        "step": obs.get("step") if obs.get("step") is not None else len(obs.get("history") or []),
        "last_action": last_action,
        "last_reward": round(float(reward), 4),
        "available_actions": allowed,
        "phase": phase,
        "last_tool_result": slim_tool_result(recent_result_raw, snippet_chars=70, max_items=3),
        "available_regions": state_regions,
        "available_region_ids": obs.get("available_region_ids") or [],
        "selected_evidence_ids": obs.get("selected_evidence_ids") or [],
        "visible_evidence_ids": (obs.get("visible_evidence_ids") or [])[:8],
        "valid_crop_count": valid_crop_count,
        "last_crop": compact_crop_ref(obs.get("last_crop_path")),
        "claim_state": claim_state,
    }
    phase_rule = ""
    remaining_fields = claim_state.get("remaining_fields") or []
    if phase in {"claim_ready", "claim_continuation"}:
        phase_rule = "write or abstain one missing field; no retrieve/open/crop; finish only when no missing fields"
        if remaining_fields:
            phase_rule += f"; next_missing={remaining_fields[0]}"
    elif phase in {"region_selection"}:
        phase_rule = "crop one region_id from regions/region_ids; prefer figure_candidate/target_rank; no evidence/write/finish"
    elif phase and str(phase).startswith("evidence"):
        phase_rule = "retrieve/open evidence from allowed ids; when evidence is enough, write one field"
    formatter = format_compact_state_update_text if config.compact_state_update else format_state_update_text
    text = formatter(state, phase_rule)
    if len(text) <= max_chars:
        return text
    state["last_tool_result"] = slim_tool_result(recent_result_raw, snippet_chars=40, max_items=2)
    if include_full_regions:
        state["available_regions"] = slim_regions(obs.get("regions") or [], snippet_chars=30, max_items=max_state_regions)
    state["visible_evidence_ids"] = (obs.get("visible_evidence_ids") or [])[:5]
    text = formatter(state, phase_rule)
    if len(text) <= max_chars:
        return text
    state["last_tool_result"] = slim_tool_result(recent_result_raw, snippet_chars=20, max_items=1)
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


def format_compact_state_update_text(state: dict[str, Any], phase_rule: str) -> str:
    claim_state = state.get("claim_state") if isinstance(state.get("claim_state"), dict) else {}
    regions = state.get("available_regions") or []
    compact = drop_empty_state_items({
        "step": state.get("step"),
        "phase": state.get("phase"),
        "allowed": state.get("available_actions") or [],
        "rule": phase_rule,
        "last": state.get("last_action"),
        "reward": state.get("last_reward"),
        "result": compact_last_result_for_state(state.get("last_tool_result")),
        "regions": regions,
        "region_ids": [] if regions else (state.get("available_region_ids") or []),
        "evidence_ids": state.get("visible_evidence_ids") or [],
        "selected": state.get("selected_evidence_ids") or [],
        "crop": state.get("last_crop"),
        "claim": {
            "written": claim_state.get("written_fields") or [],
            "abstained": claim_state.get("abstained_fields") or [],
            "missing": claim_state.get("remaining_fields") or [],
        },
    })
    return (
        "\n\n[STATE]\n"
        + json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        + "\nNext: output exactly one JSON action from allowed. Use finish only if allowed and claim.missing is empty.\n"
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
        return {
            "tool": tool,
            "evidence_id": result.get("evidence_id"),
            "authority_level": result.get("authority_level"),
            "citation_level": result.get("citation_level"),
            "snippet": short_text(
                result.get("display_snippet")
                or result.get("evidence_summary")
                or result.get("text")
                or result.get("raw_chunk_text"),
                snippet_chars,
            ),
        }
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


def slim_evidence(item: Any, snippet_chars: int) -> Any:
    if not isinstance(item, dict):
        return item
    return {
        "evidence_id": item.get("evidence_id"),
        "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
        "citation_level": item.get("citation_level"),
        "snippet": short_text(item.get("display_snippet") or item.get("evidence_summary") or item.get("text"), snippet_chars),
    }


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
