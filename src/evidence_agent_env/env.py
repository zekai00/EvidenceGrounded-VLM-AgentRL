"""Executable reset/step environment for EvidenceGrounded tasks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .actions import parse_action, validate_action
from .data import EvidenceIndex, read_jsonl
from .tools.crop import crop_image, image_size
from .tools.claim_tools import (
    DEFAULT_CLAIM_FIELDS,
    apply_claim_write,
    claim_state,
    claim_write_result,
    normalize_abstain,
    normalize_claim,
)
from .tools.region_proposal import propose_regions
from .tool_mask import action_allowed, phase_aware_tool_mask, schema_actions
from .verifier import EvidenceVerifier, gold_evidence_ids


class EvidenceAgentEnv:
    def __init__(
        self,
        tasks_path: str | Path,
        evidence_index_dir: str | Path,
        output_dir: str | Path,
        *,
        max_steps: int = 20,
        include_gold_regions: bool = False,
        phase_aware_mask: bool = False,
        enforce_tool_mask: bool = False,
        tool_schema: str = "chunked_claim",
        target_claim_fields: list[str] | None = None,
    ) -> None:
        self.tasks_path = Path(tasks_path)
        self.evidence_index_dir = Path(evidence_index_dir)
        self.output_dir = Path(output_dir)
        self.max_steps = max_steps
        self.include_gold_regions = include_gold_regions
        self.phase_aware_mask = phase_aware_mask
        self.enforce_tool_mask = enforce_tool_mask
        self.tool_schema = tool_schema
        self.target_claim_fields = list(target_claim_fields or DEFAULT_CLAIM_FIELDS)
        self.tasks = read_jsonl(self.tasks_path)
        self.index: EvidenceIndex | None = None
        self.verifier = EvidenceVerifier()
        self.task: dict[str, Any] | None = None
        self.history: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.draft_claims: list[dict[str, Any]] = []
        self.regions: dict[str, dict[str, Any]] = {}
        self.selected_evidence_ids: list[str] = []
        self.last_crop: str | None = None
        self.valid_crop_count = 0
        self.invalid_step_count = 0
        self.step_count = 0
        self.total_reward = 0.0

    def reset(self, index: int = 0, task_id: str | None = None) -> dict[str, Any]:
        if task_id is not None:
            matches = [task for task in self.tasks if task.get("task_id") == task_id]
            if not matches:
                raise KeyError(f"unknown task_id: {task_id}")
            self.task = matches[0]
        else:
            self.task = self.tasks[index]
        self.history = []
        self.tool_results = []
        self.draft_claims = []
        self.regions = {}
        self.selected_evidence_ids = []
        self.last_crop = None
        self.valid_crop_count = 0
        self.invalid_step_count = 0
        self.step_count = 0
        self.total_reward = 0.0
        (self.output_dir / "crops").mkdir(parents=True, exist_ok=True)
        return self.observation()

    def step(self, action_input: str | dict[str, Any]) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if self.task is None:
            raise RuntimeError("call reset before step")
        action, parse_error = parse_action(action_input)
        if parse_error:
            result = {"error": parse_error}
            return self._finish_step(action_input, result, -0.2, False)
        assert action is not None
        ok, validation_error = validate_action(action)
        if not ok:
            result = {"error": validation_error}
            return self._finish_step(action, result, -0.2, False)
        mask = self.current_tool_mask()
        if self.phase_aware_mask and self.enforce_tool_mask and not action_allowed(action, mask):
            result = {
                "tool": action.get("action"),
                "error": f"action {action.get('action')} is blocked by phase-aware tool mask",
                "tool_mask": mask,
            }
            terminal_block = str(action.get("action")) == "finish"
            return self._finish_step(action, result, -0.2, terminal_block)
        try:
            result = self._execute(action)
        except Exception as exc:
            result = {"tool": action.get("action"), "error": str(exc)}
        reward = self.verifier.step_reward(
            self.task,
            action,
            {
                **result,
                "draft_claims": self.draft_claims,
                "valid_crop_count": self.valid_crop_count,
                "invalid_step_count": self.invalid_step_count,
                "selected_evidence_ids": self.selected_evidence_ids,
                "visible_evidence_ids": sorted(self._visible_evidence_items()),
            },
        )
        terminated = action.get("action") == "finish" or self.step_count + 1 >= self.max_steps
        return self._finish_step(action, result, reward, terminated)

    def observation(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("call reset before observation")
        images = [{"role": "page_image", "path": self.task.get("page_image")}]
        if self.last_crop:
            images.append({"role": "last_crop", "path": self.last_crop})
        obs = {
            "task_id": self.task.get("task_id"),
            "goal": self.task.get("goal"),
            "source_file": self.task.get("source_file"),
            "page": self.task.get("page"),
            "images": images,
            "page_size": image_size(self.task["page_image"]),
            "history": self.history[-8:],
            "tool_results": self.tool_results[-6:],
            "draft_claims": self.draft_claims,
            "claim_state": self._claim_state(),
            "regions": self._public_regions(),
            "available_region_ids": sorted(self.regions),
            "selected_evidence_ids": self.selected_evidence_ids,
            "visible_evidence_ids": sorted(self._visible_evidence_items()),
            "target_evidence_hints": self._target_evidence_hints(),
            "valid_crop_count": self.valid_crop_count,
            "last_crop_path": self.last_crop,
            "tool_schema": self.tool_schema,
            "available_actions": schema_actions(self.tool_schema),
        }
        if self.phase_aware_mask:
            mask_obs = {**obs, "history": self.history, "tool_results": self.tool_results}
            mask = phase_aware_tool_mask(mask_obs)
            obs["available_actions"] = mask["allowed_actions"]
            obs["tool_mask"] = mask
        return obs

    def current_tool_mask(self) -> dict[str, Any]:
        if not self.phase_aware_mask:
            return {
                "enabled": False,
                "phase": "unmasked",
                "allowed_actions": schema_actions(self.tool_schema),
                "blocked_actions": [],
                "reason": "phase-aware tool mask is disabled.",
                "step": self.step_count,
                "tool_schema": self.tool_schema,
            }
        return phase_aware_tool_mask(
            {
                **self.observation(),
                "history": self.history,
                "tool_results": self.tool_results,
                "tool_schema": self.tool_schema,
            }
        )

    def _execute(self, action: dict[str, Any]) -> dict[str, Any]:
        assert self.task is not None
        name = action["action"]
        if name == "inspect_page":
            top_k = int(action.get("top_k", 10))
            regions = propose_regions(self.task, top_k=top_k, include_gold=self.include_gold_regions)
            regions = self._annotate_caption_links(regions)
            self.regions = {str(item["region_id"]): item for item in regions}
            return {
                "tool": name,
                "page_image": self.task.get("page_image"),
                "page_size": image_size(self.task["page_image"]),
                "source_file": self.task.get("source_file"),
                "page": self.task.get("page"),
                "regions": regions,
                "layout_regions": regions,
            }
        if name == "propose_regions":
            top_k = int(action.get("top_k", 8))
            regions = propose_regions(self.task, top_k=top_k, include_gold=self.include_gold_regions)
            regions = self._annotate_caption_links(regions)
            self.regions = {str(item["region_id"]): item for item in regions}
            return {"tool": name, "regions": regions}
        if name == "select_evidence":
            return self._select_evidence([str(item) for item in (action.get("evidence_ids") or [])])
        if name == "crop_region":
            region = self.regions.get(str(action.get("region_id")))
            if not region:
                return {"tool": name, "error": f"unknown region_id: {action.get('region_id')}"}
            return self._crop(region["bbox"], name, {"region_id": action.get("region_id")})
        if name == "crop_target":
            region_id = action.get("region_id")
            if region_id is not None:
                region = self.regions.get(str(region_id))
                if not region:
                    return {"tool": name, "error": f"unknown region_id: {region_id}"}
                return self._crop(region["bbox"], name, {"region_id": region_id, "crop_mode": "region_id"})
            return self._crop(action["bbox"], name, {"crop_mode": "bbox"})
        if name == "crop_image":
            return self._crop(action["bbox"], name, {})
        if name == "retrieve_evidence":
            if self.index is None:
                self.index = EvidenceIndex(self.evidence_index_dir)
            results = self.index.search(
                str(action.get("query", "")),
                str(action.get("scope")),
                self.task,
                int(action.get("top_k", 5)),
            )
            hit_ids = sorted({str(item["evidence_id"]) for item in results} & gold_evidence_ids(self.task))
            return {
                "tool": name,
                "query": action.get("query"),
                "scope": action.get("scope"),
                "anchor": action.get("anchor"),
                "results": results,
                "hit_evidence_ids": hit_ids,
            }
        if name == "open_evidence":
            local_item = self._open_local_evidence(str(action.get("evidence_id")))
            if local_item:
                return local_item
            if self.index is None:
                self.index = EvidenceIndex(self.evidence_index_dir)
            item = self.index.open(str(action.get("evidence_id")))
            if not item:
                return {"tool": name, "evidence_id": action.get("evidence_id"), "error": "evidence not found"}
            return {
                "tool": name,
                "evidence_id": item.get("evidence_id"),
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
                "page_end": item.get("page_end"),
                "authority_level": item.get("authority_level"),
                "citation_level": item.get("citation_level"),
                "display_snippet": item.get("display_snippet") or item.get("evidence_summary") or item.get("text", "")[:600],
            }
        if name == "write_claim":
            evidence_error = self._claim_evidence_error([action])
            if evidence_error:
                return {"tool": name, "error": evidence_error}
            claim = normalize_claim(action)
            self._upsert_claim(claim)
            return {"tool": name, "claim": claim, "claim_state": self._claim_state()}
        if name == "abstain_claim":
            claim = normalize_abstain(action)
            self._upsert_claim(claim)
            return {"tool": name, "claim": claim, "claim_state": self._claim_state()}
        if name in {"write_claims_chunk", "write_claims_batch"}:
            evidence_error = self._claim_evidence_error(action.get("claims") or [])
            if evidence_error:
                return {"tool": name, "error": evidence_error}
            self.draft_claims = apply_claim_write(
                self.draft_claims,
                claims=action.get("claims") or [],
                abstains=action.get("abstains") or [],
            )
            return claim_write_result(
                name,
                self.draft_claims,
                claims=action.get("claims") or [],
                abstains=action.get("abstains") or [],
                target_fields=self.target_claim_fields,
            )
        if name == "finish":
            return {"tool": name, "status": action.get("status", "done"), "draft_claims": self.draft_claims}
        raise AssertionError(f"unhandled action: {name}")

    def _open_local_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        assert self.task is not None
        for item in self.task.get("local_evidence") or []:
            if str(item.get("evidence_id")) != evidence_id:
                continue
            return {
                "tool": "open_evidence",
                "evidence_id": item.get("evidence_id"),
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
                "page_end": item.get("page_end"),
                "authority_level": item.get("authority_level"),
                "citation_level": item.get("citation_level"),
                "display_snippet": item.get("display_snippet") or item.get("text", ""),
            }
        return None

    def _select_evidence(self, evidence_ids: list[str]) -> dict[str, Any]:
        visible = self._visible_evidence_items()
        accepted: list[str] = []
        rejected: list[str] = []
        selected_items: list[dict[str, Any]] = []
        for evidence_id in evidence_ids:
            item = visible.get(str(evidence_id))
            if not item:
                rejected.append(str(evidence_id))
                continue
            if str(evidence_id) not in accepted:
                accepted.append(str(evidence_id))
                selected_items.append(item)
        if not accepted:
            return {"tool": "select_evidence", "error": f"no known evidence ids selected: {rejected}"}
        for evidence_id in accepted:
            if evidence_id not in self.selected_evidence_ids:
                self.selected_evidence_ids.append(evidence_id)
        result = {
            "tool": "select_evidence",
            "selected_evidence_ids": accepted,
            "selected_evidence": selected_items,
        }
        if rejected:
            result["rejected_evidence_ids"] = rejected
        return result

    def _visible_evidence_items(self) -> dict[str, dict[str, Any]]:
        assert self.task is not None
        items: dict[str, dict[str, Any]] = {}
        for item in self.task.get("local_evidence") or []:
            evidence_id = str(item.get("evidence_id"))
            if evidence_id:
                items[evidence_id] = self._public_evidence_item(item)
        for region in self.regions.values():
            evidence_id = region.get("caption_evidence_id")
            if not evidence_id:
                continue
            evidence_id = str(evidence_id)
            items.setdefault(
                evidence_id,
                {
                    "evidence_id": evidence_id,
                    "source_file": self.task.get("source_file"),
                    "page_start": self.task.get("page"),
                    "citation_level": "region_candidate_caption",
                    "display_snippet": region.get("caption_hint") or region.get("nearby_text") or region.get("hint"),
                },
            )
        for result in self.tool_results:
            if not isinstance(result, dict):
                continue
            if result.get("tool") == "retrieve_evidence":
                for item in result.get("results") or []:
                    evidence_id = str(item.get("evidence_id"))
                    if evidence_id:
                        items[evidence_id] = self._public_evidence_item(item)
            elif result.get("tool") == "open_evidence" and result.get("evidence_id") and not result.get("error"):
                evidence_id = str(result.get("evidence_id"))
                items[evidence_id] = self._public_evidence_item(result)
        return items

    def _public_evidence_item(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "evidence_id": item.get("evidence_id"),
            "source_file": item.get("source_file"),
            "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
            "page_end": item.get("page_end"),
            "authority_level": item.get("authority_level"),
            "citation_level": item.get("citation_level"),
            "source_quality": item.get("source_quality"),
            "display_snippet": item.get("display_snippet") or item.get("evidence_summary") or item.get("text", ""),
        }

    def _public_regions(self) -> list[dict[str, Any]]:
        public: list[dict[str, Any]] = []
        for region in self.regions.values():
            public.append(
                {
                    "region_id": region.get("region_id"),
                    "bbox": region.get("bbox"),
                    "type": region.get("type"),
                    "source": region.get("source"),
                    "score": region.get("score"),
                    "caption_evidence_id": region.get("caption_evidence_id"),
                    "caption_hint": region.get("caption_hint"),
                    "nearby_text": region.get("nearby_text"),
                    "hint": region.get("hint"),
                    "linked_caption_region_id": region.get("linked_caption_region_id"),
                    "linked_caption_text": region.get("linked_caption_text"),
                    "linked_caption_position": region.get("linked_caption_position"),
                    "linked_caption_gap_px": region.get("linked_caption_gap_px"),
                    "caption_link_score": region.get("caption_link_score"),
                    "target_caption_match_score": region.get("target_caption_match_score"),
                    "target_caption_match_reason": region.get("target_caption_match_reason"),
                    "target_region_rank": region.get("target_region_rank"),
                    "target_region_sort_score": region.get("target_region_sort_score"),
                }
            )
        return public

    def _target_evidence_hints(self) -> list[dict[str, Any]]:
        assert self.task is not None
        hints: list[dict[str, Any]] = []
        for item in self.task.get("local_evidence") or []:
            evidence_id = str(item.get("evidence_id") or "")
            snippet = item.get("display_snippet") or item.get("text") or ""
            if not evidence_id or not snippet:
                continue
            hints.append(
                {
                    "evidence_id": evidence_id,
                    "source_file": item.get("source_file"),
                    "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
                    "citation_level": item.get("citation_level"),
                    "display_snippet": snippet,
                }
            )
        return hints[:3]

    def _annotate_caption_links(self, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        linked_regions = [dict(item) for item in regions]
        captions: list[dict[str, Any]] = []
        for item in linked_regions:
            if item.get("type") != "text_or_caption_candidate":
                continue
            text = str(item.get("caption_hint") or item.get("nearby_text") or "")
            if not self._caption_like(text):
                continue
            captions.append(item)
        if not captions:
            return linked_regions
        for item in linked_regions:
            if item.get("type") != "figure_candidate":
                continue
            best = self._best_caption_link(item, captions)
            if not best:
                continue
            caption, position, gap, score = best
            if score < 1.0:
                continue
            item["linked_caption_region_id"] = caption.get("region_id")
            linked_caption_text = str(caption.get("caption_hint") or caption.get("nearby_text") or "")
            item["linked_caption_text"] = linked_caption_text[:240]
            item["linked_caption_position"] = position
            item["linked_caption_gap_px"] = int(round(gap))
            item["caption_link_score"] = round(float(score), 3)
            match_score, match_reason = self._target_caption_match(linked_caption_text)
            item["target_caption_match_score"] = match_score
            item["target_caption_match_reason"] = match_reason
        has_precomputed_target_rank = any(
            item.get("type") == "figure_candidate" and item.get("target_region_rank") is not None
            for item in linked_regions
        )
        if not has_precomputed_target_rank:
            self._rank_target_regions(linked_regions)
        return linked_regions

    @staticmethod
    def _rank_target_regions(regions: list[dict[str, Any]]) -> None:
        figures = [item for item in regions if item.get("type") == "figure_candidate"]
        if not figures:
            return
        if not any(float(item.get("target_caption_match_score") or 0.0) > 0.0 for item in figures):
            return
        scored: list[tuple[float, float, int, dict[str, Any]]] = []
        for index, item in enumerate(figures):
            target_score = float(item.get("target_caption_match_score") or 0.0)
            link_score = float(item.get("caption_link_score") or 0.0)
            scored.append((target_score, link_score, -index, item))
        scored.sort(reverse=True)
        for rank, (_, _, _, item) in enumerate(scored, start=1):
            item["target_region_rank"] = rank
            item["target_region_sort_score"] = round(
                float(item.get("target_caption_match_score") or 0.0) * 10.0
                + float(item.get("caption_link_score") or 0.0),
                3,
            )

    def _best_caption_link(
        self,
        figure: dict[str, Any],
        captions: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str, float, float] | None:
        fig_bbox = figure.get("bbox") or []
        if len(fig_bbox) != 4:
            return None
        fx1, fy1, fx2, fy2 = [float(value) for value in fig_bbox]
        fwidth = max(1.0, fx2 - fx1)
        fcx = (fx1 + fx2) / 2.0
        best: tuple[dict[str, Any], str, float, float] | None = None
        for caption in captions:
            cap_bbox = caption.get("bbox") or []
            if len(cap_bbox) != 4:
                continue
            cx1, cy1, cx2, cy2 = [float(value) for value in cap_bbox]
            overlap = max(0.0, min(fx2, cx2) - max(fx1, cx1)) / fwidth
            if overlap <= 0.15:
                continue
            ccx = (cx1 + cx2) / 2.0
            center_penalty = min(1.0, abs(fcx - ccx) / max(1.0, fwidth))
            position = "below" if cy1 >= fy2 - 8 else "above" if cy2 <= fy1 + 8 else "overlap"
            if position == "below":
                gap = max(0.0, cy1 - fy2)
                position_bonus = 1.0
            elif position == "above":
                gap = max(0.0, fy1 - cy2)
                position_bonus = -0.35
            else:
                gap = 0.0
                position_bonus = -0.2
            if gap > 420:
                continue
            text = str(caption.get("caption_hint") or caption.get("nearby_text") or "")
            prefix_bonus = 0.35 if self._caption_like(text[:40]) else 0.0
            score = 1.8 * min(1.0, overlap) + position_bonus + prefix_bonus - center_penalty - gap / 500.0
            if best is None or score > best[3]:
                best = (caption, position, gap, score)
        return best

    def _target_caption_match(self, linked_caption_text: str) -> tuple[float, str]:
        assert self.task is not None
        linked = self._normalize_caption_text(linked_caption_text)
        if not linked:
            return 0.0, ""
        best_score = 0.0
        best_reason = ""
        for item in self.task.get("local_evidence") or []:
            target_text = str(item.get("display_snippet") or item.get("text") or "")
            target = self._normalize_caption_text(target_text)
            if not target:
                continue
            score = 0.0
            reasons: list[str] = []
            linked_label = self._figure_label(linked)
            target_label = self._figure_label(target)
            if linked_label and target_label and linked_label == target_label:
                score += 2.0
                reasons.append(f"figure_label={linked_label}")
            linked_title = self._work_title(linked)
            target_title = self._work_title(target)
            if linked_title and target_title and linked_title == target_title:
                score += 2.0
                reasons.append(f"title={linked_title}")
            overlap = self._char_overlap(linked, target)
            if overlap >= 0.55:
                score += overlap
                reasons.append(f"text_overlap={overlap:.2f}")
            if score > best_score:
                best_score = score
                best_reason = ",".join(reasons)
        return round(best_score, 3), best_reason

    @staticmethod
    def _caption_like(text: str) -> bool:
        normalized = re.sub(r"\s+", "", str(text or ""))
        if not normalized:
            return False
        return bool(re.match(r"^(【?图|Fig\\.?|Figure)", normalized, flags=re.IGNORECASE))

    @staticmethod
    def _normalize_caption_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).replace("（", "(").replace("）", ")")

    @staticmethod
    def _figure_label(text: str) -> str:
        match = re.search(r"图[一二三四五六七八九十百0-9]+(?:[-－—][一二三四五六七八九十百0-9]+)?", text)
        return match.group(0) if match else ""

    @staticmethod
    def _work_title(text: str) -> str:
        match = re.search(r"《([^》]{1,40})》", text)
        return match.group(1) if match else ""

    @staticmethod
    def _char_overlap(a: str, b: str) -> float:
        a_chars = {char for char in a if "\u4e00" <= char <= "\u9fff"}
        b_chars = {char for char in b if "\u4e00" <= char <= "\u9fff"}
        if not a_chars or not b_chars:
            return 0.0
        return len(a_chars & b_chars) / max(1, len(a_chars | b_chars))

    def _claim_evidence_error(self, claims: list[Any]) -> str | None:
        visible_ids = set(self._visible_evidence_items())
        unknown: list[str] = []
        empty_fields: list[str] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            evidence_ids = [str(item) for item in (claim.get("evidence_ids") or [])]
            if not evidence_ids:
                empty_fields.append(str(claim.get("field", "")))
                continue
            for evidence_id in evidence_ids:
                if evidence_id not in visible_ids:
                    unknown.append(evidence_id)
        if empty_fields:
            return f"claim evidence_ids must be non-empty for fields: {empty_fields}"
        if unknown:
            return f"unknown evidence_ids in claim: {sorted(set(unknown))}"
        return None

    def _crop(self, bbox: Any, tool: str, extra: dict[str, Any]) -> dict[str, Any]:
        assert self.task is not None
        path = self.output_dir / "crops" / f"{self.task['task_id']}_step{self.step_count:02d}.jpg"
        result = crop_image(self.task["page_image"], bbox, path)
        self.last_crop = result["crop_path"]
        self.valid_crop_count += 1
        iou = self.verifier.crop_iou(self.task, result["bbox"])
        return {"tool": tool, **extra, **result, "bbox_iou": iou}

    def _upsert_claim(self, claim: dict[str, Any]) -> None:
        field = claim.get("field")
        self.draft_claims = [item for item in self.draft_claims if item.get("field") != field]
        self.draft_claims.append(claim)

    def _claim_state(self) -> dict[str, Any]:
        return claim_state(self.draft_claims, target_fields=self.target_claim_fields)

    def _finish_step(
        self,
        action: Any,
        result: dict[str, Any],
        reward: float,
        terminated: bool,
    ) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        record = {"step": self.step_count, "action": action, "result": result, "reward": reward}
        self.history.append(action if isinstance(action, dict) else {"raw": action})
        self.tool_results.append(result)
        if result.get("error"):
            self.invalid_step_count += 1
        self.step_count += 1
        self.total_reward += reward
        info = {
            "step": self.step_count,
            "result": result,
            "total_reward": self.total_reward,
            "terminated": terminated,
        }
        if terminated:
            info["trajectory_metrics"] = self.trajectory_metrics()
        return self.observation(), reward, terminated, info

    def trajectory_metrics(self) -> dict[str, Any]:
        if self.task is None:
            raise RuntimeError("call reset before trajectory_metrics")
        return self.verifier.trajectory_metrics(
            self.task,
            self.history,
            self.tool_results,
            self.draft_claims,
            max_steps=self.max_steps,
        )

    def dump_trajectory(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "task_id": self.task.get("task_id") if self.task else None,
            "history": self.history,
            "tool_results": self.tool_results,
            "draft_claims": self.draft_claims,
            "selected_evidence_ids": self.selected_evidence_ids,
            "valid_crop_count": self.valid_crop_count,
            "invalid_step_count": self.invalid_step_count,
            "total_reward": self.total_reward,
            "phase_aware_mask": self.phase_aware_mask,
            "enforce_tool_mask": self.enforce_tool_mask,
            "tool_schema": self.tool_schema,
            "target_claim_fields": self.target_claim_fields,
            "trajectory_metrics": self.trajectory_metrics() if self.task else None,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
