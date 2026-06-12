#!/usr/bin/env python
"""Build v1.0.4 GoldEval field-fragment support labels.

This script creates a small probe set over corrected GoldEval val tasks.  The
unit is (task, field, evidence/fragment): can this fragment support this field
value for the target figure?
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "gold_eval_v1_0_4_caption_corrected_20260611_1830/val_gold_50.jsonl"
)
DEFAULT_INDEX = (
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "evidence_index_v1_0_4_llm_overlay_20260611_0222"
)
DEFAULT_ROLLOUTS = [
    "baseline_val50=outputs/v1_0_4_gold_eval_caption_corrected_val50_bf16_20260611_2055/rollouts.jsonl",
    "behavior_repair_C_val50=outputs/v1_0_4_behavior_repair_C_pos70_replay90_20step_val50_bf16_20260612_0240/rollouts.jsonl",
    "field_policy_probe_val16=outputs/v1_0_4_field_policy_prompt_reward_probe_val16_bf16_20260612_1146/rollouts.jsonl",
]
CORE_FIELDS = ["caption_text", "image_scope", "depicted_work_title", "displayed_region", "object_type"]
LOCAL_CAPTION_RISK_FIELDS = {"image_scope", "displayed_region", "object_type"}
SUPPORT_LABELS = {"support", "weak_support", "no_support", "contradict", "wrong_target"}
FIG_LABEL_RE = re.compile(r"(?:图|圖|fig\.?|figure)\s*[A-Za-z]?\s*\d+(?:[.\-．-]\d+)*", re.I)
TITLE_RE = re.compile(r"《([^》]{1,80})》")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--evidence-index", default=DEFAULT_INDEX)
    parser.add_argument("--rollout", action="append", default=[], help="name=path. Defaults to baseline/C/probe.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-pairs", type=int, default=420)
    parser.add_argument("--max-llm", type=int, default=260)
    parser.add_argument("--provider", choices=["offline", "dashscope"], default="dashscope")
    parser.add_argument("--dotenv", default=".env")
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument(
        "--fallback-models",
        default=(
            "qwen3.6-plus,glm-5.1,kimi-k2.6,qwen3.5-plus-2026-04-20,"
            "qwen3.6-27b,deepseek-v4-flash,deepseek-v4-pro,qwen3.7-max,"
            "qwen3.7-max-preview,qwen3.7-plus-2026-05-26,qwen3.7-max-2026-06-08"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=44)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(args.dotenv)
    output_dir = Path(args.output_dir or default_output_dir())
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = read_jsonl(args.tasks)
    task_by_id = {str(row.get("task_id")): row for row in tasks}
    evidence_by_id = load_evidence_index(Path(args.evidence_index))
    rollout_specs = parse_rollout_specs(args.rollout or DEFAULT_ROLLOUTS)
    rollout_records = {name: read_jsonl(path) for name, path in rollout_specs.items() if Path(path).exists()}

    candidates = build_candidates(tasks, evidence_by_id, rollout_records)
    selected = select_candidates(candidates, args.max_pairs)
    labels = adjudicate_candidates(selected, args)

    pairs_path = output_dir / "pairs.jsonl"
    labels_path = output_dir / "labels.jsonl"
    write_jsonl(pairs_path, [item.to_row() for item in selected])
    write_jsonl(labels_path, labels)
    summary = build_summary(args, output_dir, tasks, selected, labels, rollout_records)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "构建报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


@dataclass
class Candidate:
    pair_id: str
    task_id: str
    field: str
    gold_value: Any
    gold_abstain: bool
    evidence_id: str
    fragment_id: str
    fragment_text: str
    source_role: str
    source_file: str
    page: Any
    candidate_kind: str
    priority: int
    used_by_runs: list[str]
    cited_by_runs: list[str]
    opened_by_runs: list[str]
    retrieved_by_runs: list[str]
    rule_label: str
    rule_allowed: bool
    rule_confidence: float
    rule_reason: str
    target_caption: str
    target_figure_labels: list[str]
    candidate_figure_labels: list[str]
    extra: dict[str, Any]

    def key(self) -> tuple[str, str, str, str]:
        return (self.task_id, self.field, self.evidence_id, self.fragment_id)

    def to_row(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_candidates(
    tasks: list[dict[str, Any]],
    evidence_by_id: dict[str, dict[str, Any]],
    rollout_records: dict[str, list[dict[str, Any]]],
) -> list[Candidate]:
    rollout_use = collect_rollout_use(rollout_records)
    by_key: dict[tuple[str, str, str, str], Candidate] = {}
    for task in tasks:
        task_id = str(task.get("task_id"))
        gold_claims = {
            str(claim.get("field")): claim for claim in (task.get("gold") or {}).get("claims", []) if claim.get("field")
        }
        fields = [field for field in CORE_FIELDS if field in gold_claims]
        target_caption = str((task.get("gold") or {}).get("caption_text") or "")
        target_labels = figure_labels(target_caption)
        evidence_items = task_evidence_items(task, evidence_by_id, rollout_use.get(task_id, {}))
        for field in fields:
            gold = gold_claims[field]
            field_use = (rollout_use.get(task_id, {}).get("by_field") or {}).get(field, {})
            for item in evidence_items:
                evidence_id = str(item.get("evidence_id"))
                fragments = fragment_evidence(task, item)
                for frag in fragments:
                    candidate_kind = candidate_kind_for(item, field_use, gold)
                    priority = candidate_priority(item, field, field_use, gold, candidate_kind)
                    rule = rule_label(task, field, gold, item, frag, target_caption, target_labels)
                    pair_id = make_pair_id(task_id, field, evidence_id, frag["fragment_id"])
                    cand = Candidate(
                        pair_id=pair_id,
                        task_id=task_id,
                        field=field,
                        gold_value=gold.get("value"),
                        gold_abstain=bool(gold.get("abstain")),
                        evidence_id=evidence_id,
                        fragment_id=frag["fragment_id"],
                        fragment_text=frag["text"],
                        source_role=str(item.get("source_role") or item.get("adjudicated_evidence_role") or ""),
                        source_file=str(item.get("source_file") or task.get("source_file") or ""),
                        page=item.get("page_start") if item.get("page_start") is not None else task.get("page"),
                        candidate_kind=candidate_kind,
                        priority=priority,
                        used_by_runs=sorted(field_use.get("used_by_runs") or []),
                        cited_by_runs=sorted(field_use.get("cited_by_runs") or []),
                        opened_by_runs=sorted((rollout_use.get(task_id, {}).get("opened") or {}).get(evidence_id, [])),
                        retrieved_by_runs=sorted((rollout_use.get(task_id, {}).get("retrieved") or {}).get(evidence_id, [])),
                        rule_label=rule["label"],
                        rule_allowed=bool(rule["allowed"]),
                        rule_confidence=float(rule["confidence"]),
                        rule_reason=str(rule["reason"]),
                        target_caption=target_caption,
                        target_figure_labels=target_labels,
                        candidate_figure_labels=figure_labels(frag["text"]),
                        extra={
                            "bbox": item.get("bbox"),
                            "gold_evidence_ids": gold.get("evidence_ids") or [],
                            "gold_candidate_evidence_ids": gold.get("candidate_evidence_ids") or [],
                            "display_snippet": item.get("display_snippet"),
                            "clean_evidence_type": item.get("clean_evidence_type"),
                            "adjudicated_claim_allowed_fields": item.get("adjudicated_claim_allowed_fields"),
                            "usable_for_claim_by_adjudication": item.get("usable_for_claim_by_adjudication"),
                            "region_id": item.get("region_id"),
                            "is_target": item.get("is_target"),
                        },
                    )
                    key = cand.key()
                    existing = by_key.get(key)
                    if existing is None or cand.priority > existing.priority:
                        by_key[key] = cand
    return list(by_key.values())


def collect_rollout_use(rollout_records: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"retrieved": defaultdict(set), "opened": defaultdict(set), "cited": defaultdict(set), "by_field": defaultdict(dict)}
    )
    for run_name, records in rollout_records.items():
        for record in records:
            task_id = str(record.get("task_id"))
            retrieved, opened = set(), set()
            for step in record.get("steps") or []:
                action = step.get("parsed_action") or {}
                result = step.get("result") or {}
                if action.get("action") == "retrieve_evidence" or result.get("tool") == "retrieve_evidence":
                    for item in result.get("results") or []:
                        eid = str(item.get("evidence_id") or "")
                        if eid:
                            retrieved.add(eid)
                            out[task_id]["retrieved"][eid].add(run_name)
                    for eid in result.get("hit_evidence_ids") or []:
                        out[task_id]["retrieved"][str(eid)].add(run_name)
                if action.get("action") == "open_evidence" or result.get("tool") == "open_evidence":
                    eid = str(action.get("evidence_id") or result.get("evidence_id") or "")
                    if eid and not result.get("error"):
                        opened.add(eid)
                        out[task_id]["opened"][eid].add(run_name)
            for claim in record.get("final_claims") or []:
                if not isinstance(claim, dict) or claim.get("abstain"):
                    continue
                field = str(claim.get("field") or "")
                info = out[task_id]["by_field"].setdefault(
                    field, {"used_by_runs": set(), "cited_by_runs": set(), "cited_ids": set()}
                )
                info["used_by_runs"].add(run_name)
                for eid in claim.get("evidence_ids") or []:
                    eid_s = str(eid)
                    out[task_id]["cited"][eid_s].add(run_name)
                    info["cited_by_runs"].add(run_name)
                    info["cited_ids"].add(eid_s)
    # Convert nested sets later in candidate construction through sorted().
    return out


def task_evidence_items(
    task: dict[str, Any],
    evidence_by_id: dict[str, dict[str, Any]],
    use: dict[str, Any],
) -> list[dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for local in task.get("local_evidence") or []:
        eid = str(local.get("evidence_id"))
        row = dict(local)
        row.setdefault("source_role", "local_caption")
        row.setdefault("retrieval_scope_base", "local")
        row.setdefault("evidence_type", "local_caption")
        if eid:
            items[eid] = row
    for claim in (task.get("gold") or {}).get("claims", []) or []:
        for eid in (claim.get("evidence_ids") or []) + (claim.get("candidate_evidence_ids") or []):
            eid_s = str(eid)
            if eid_s in evidence_by_id:
                items[eid_s] = evidence_by_id[eid_s]
    for group_name in ("retrieved", "opened", "cited"):
        for eid in (use.get(group_name) or {}).keys():
            eid_s = str(eid)
            if eid_s in evidence_by_id:
                items[eid_s] = evidence_by_id[eid_s]
    for region in task.get("region_candidates") or []:
        if region.get("is_target"):
            continue
        text = region.get("nearby_text") or region.get("caption_hint") or region.get("linked_caption_text")
        if not text:
            continue
        eid = f"region_{task.get('task_id')}_{region.get('region_id')}"
        items[eid] = {
            "evidence_id": eid,
            "source_role": "same_page_distractor_caption",
            "evidence_type": "region_caption_fragment",
            "source_file": task.get("source_file"),
            "page_start": task.get("page"),
            "display_snippet": text,
            "text": text,
            "bbox": region.get("bbox"),
            "region_id": region.get("region_id"),
            "is_target": False,
        }
    return list(items.values())


def fragment_evidence(task: dict[str, Any], item: dict[str, Any]) -> list[dict[str, str]]:
    text = str(item.get("display_snippet") or item.get("evidence_summary") or item.get("clean_text") or item.get("text") or "")
    text = normalize_space(text)
    if not text:
        return []
    spans = split_by_figure_labels(text)
    if not spans:
        spans = split_sentences(text)
    if not spans:
        spans = [text[:900]]
    fragments = []
    for idx, span in enumerate(spans[:6]):
        span = normalize_space(span)
        if not span:
            continue
        eid = str(item.get("evidence_id"))
        fragments.append({"fragment_id": f"{eid}#frag{idx:02d}", "text": span[:1200]})
    return fragments


def split_by_figure_labels(text: str) -> list[str]:
    matches = list(FIG_LABEL_RE.finditer(text))
    if len(matches) <= 1:
        return []
    spans = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        spans.append(text[start:end].strip(" ;,，。"))
    return spans


def split_sentences(text: str) -> list[str]:
    if len(text) <= 360:
        return [text]
    parts = re.split(r"(?<=[。！？.!?；;])\s+|\n+", text)
    parts = [normalize_space(part) for part in parts if normalize_space(part)]
    if not parts:
        return [text[:900]]
    out: list[str] = []
    buf = ""
    for part in parts:
        if len(buf) + len(part) <= 460:
            buf = f"{buf} {part}".strip()
        else:
            if buf:
                out.append(buf)
            buf = part
    if buf:
        out.append(buf)
    return out[:6]


def candidate_kind_for(item: dict[str, Any], field_use: dict[str, Any], gold: dict[str, Any]) -> str:
    eid = str(item.get("evidence_id"))
    if eid in (field_use.get("cited_ids") or set()):
        return "model_cited"
    if eid in {str(x) for x in gold.get("evidence_ids") or []}:
        return "gold_evidence"
    if str(item.get("source_role")) == "same_page_distractor_caption":
        return "same_page_distractor"
    if eid.startswith("local_caption_"):
        return "local_caption"
    return "candidate_or_retrieved"


def candidate_priority(item: dict[str, Any], field: str, field_use: dict[str, Any], gold: dict[str, Any], kind: str) -> int:
    priority = {
        "model_cited": 1000,
        "gold_evidence": 900,
        "same_page_distractor": 760,
        "local_caption": 720,
        "candidate_or_retrieved": 500,
    }.get(kind, 100)
    eid = str(item.get("evidence_id"))
    if eid in (field_use.get("cited_ids") or set()):
        priority += 80
    if field in LOCAL_CAPTION_RISK_FIELDS and eid.startswith("local_caption_"):
        priority += 70
    if not gold.get("abstain"):
        priority += 30
    if item.get("adjudicated_claim_allowed_fields"):
        priority += 10
    return priority


def rule_label(
    task: dict[str, Any],
    field: str,
    gold: dict[str, Any],
    item: dict[str, Any],
    fragment: dict[str, str],
    target_caption: str,
    target_labels: list[str],
) -> dict[str, Any]:
    text = fragment["text"]
    norm_text = normalize_for_match(text)
    value = gold.get("value")
    norm_value = normalize_for_match(value)
    labels = figure_labels(text)
    if target_labels and labels and not any(label_match(label, target_labels) for label in labels):
        return rule("wrong_target", False, 0.86, "fragment has figure label different from target caption")
    if item.get("source_role") == "same_page_distractor_caption":
        return rule("wrong_target", False, 0.84, "same-page non-target caption/region candidate")
    if gold.get("abstain"):
        if value is None:
            return rule("no_support", False, 0.74, "gold field abstains; no positive field value to support")
    if not norm_value:
        return rule("no_support", False, 0.70, "empty gold value")
    if field == "caption_text":
        if item.get("evidence_id", "").startswith("local_caption_") and text_overlap(norm_text, normalize_for_match(target_caption)) >= 0.70:
            return rule("support", True, 0.96, "local caption matches corrected target caption")
        if text_overlap(norm_text, normalize_for_match(target_caption)) >= 0.80:
            return rule("support", True, 0.86, "fragment overlaps target caption text")
        return rule("no_support", False, 0.72, "fragment does not reproduce target caption")
    if field == "depicted_work_title":
        titles = [normalize_for_match(t) for t in TITLE_RE.findall(text)]
        if norm_value and norm_value in norm_text:
            return rule("support", True, 0.92, "fragment explicitly contains title/value")
        if titles and norm_value and not any(norm_value in t or t in norm_value for t in titles):
            return rule("wrong_target", False, 0.80, "fragment contains a different titled work")
        return rule("no_support", False, 0.68, "title/value not found in fragment")
    if field == "displayed_region":
        if norm_value and norm_value in norm_text:
            return rule("support", True, 0.90, "fragment explicitly contains displayed region value")
        if any(term in norm_text for term in ["局部", "部分", "detail", "part", "section", "fold"]):
            return rule("weak_support", True, 0.66, "fragment contains region/detail cue but value match is not exact")
        return rule("no_support", False, 0.70, "fragment lacks displayed-region cue")
    if field == "image_scope":
        scope_terms = ["局部", "全图", "全幅", "整幅", "卷本", "册页", "detail", "full", "whole", "part"]
        if norm_value and norm_value in norm_text:
            return rule("support", True, 0.88, "fragment explicitly contains image scope value")
        if any(term in norm_text for term in scope_terms):
            return rule("weak_support", True, 0.64, "fragment contains image-scope cue")
        return rule("no_support", False, 0.72, "fragment lacks image-scope cue")
    if field == "object_type":
        if norm_value and norm_value in norm_text:
            return rule("support", True, 0.84, "fragment explicitly contains object type value")
        if str(item.get("evidence_id", "")).startswith("local_caption_") and re.search(r"\bfig(?:ure)?\.?|\bplate\b|图|圖", text, flags=re.I):
            return rule("support", True, 0.78, "caption/figure marker supports generic object_type=image")
        if any(term in norm_text for term in ["山水", "landscape", "hanging scroll", "scroll", "painting", "画"]):
            return rule("weak_support", True, 0.62, "fragment gives weak visual/object-type cue")
        return rule("no_support", False, 0.68, "fragment lacks object-type cue")
    if norm_value and norm_value in norm_text:
        return rule("support", True, 0.82, "fragment explicitly contains field value")
    return rule("no_support", False, 0.60, "no rule support found")


def rule(label: str, allowed: bool, confidence: float, reason: str) -> dict[str, Any]:
    return {"label": label, "allowed": allowed, "confidence": confidence, "reason": reason}


def select_candidates(candidates: list[Candidate], max_pairs: int) -> list[Candidate]:
    # Keep all model-cited pairs first, then fill a field-balanced probe.
    cited = [c for c in candidates if c.candidate_kind == "model_cited"]
    cited_keys = {c.key() for c in cited}
    rest = [c for c in candidates if c.key() not in cited_keys]
    rest.sort(key=lambda c: (-c.priority, c.task_id, c.field, c.evidence_id, c.fragment_id))
    selected = list(cited)
    by_field = Counter(c.field for c in selected)
    target_per_field = max(20, max_pairs // max(1, len(CORE_FIELDS)))
    for cand in rest:
        if len(selected) >= max_pairs:
            break
        if by_field[cand.field] > target_per_field and cand.priority < 850:
            continue
        selected.append(cand)
        by_field[cand.field] += 1
    if len(selected) < max_pairs:
        selected_keys = {c.key() for c in selected}
        for cand in rest:
            if len(selected) >= max_pairs:
                break
            if cand.key() not in selected_keys:
                selected.append(cand)
                selected_keys.add(cand.key())
    return selected[:max_pairs]


def adjudicate_candidates(candidates: list[Candidate], args: argparse.Namespace) -> list[dict[str, Any]]:
    client = None
    llm_budget = max(0, int(args.max_llm))
    if args.provider == "dashscope" and llm_budget > 0:
        client = DashScopeClient(args)
    labels: list[dict[str, Any]] = []
    llm_used = 0
    for idx, cand in enumerate(candidates):
        row = cand.to_row()
        adjudication: dict[str, Any]
        source = "rule"
        model = None
        raw_response = None
        should_llm = (
            client is not None
            and llm_used < llm_budget
            and (cand.candidate_kind == "model_cited" or cand.rule_confidence < 0.88 or cand.field in LOCAL_CAPTION_RISK_FIELDS)
        )
        if should_llm:
            try:
                adjudication, model, raw_response = client.adjudicate(cand)
                adjudication = normalize_adjudication(adjudication, cand)
                source = "llm"
                llm_used += 1
                if args.sleep > 0:
                    time.sleep(args.sleep)
            except Exception as exc:
                adjudication = {
                    "label": cand.rule_label,
                    "allowed": cand.rule_allowed,
                    "confidence": cand.rule_confidence,
                    "target_link": "unknown",
                    "rationale": f"LLM failed; fallback to rule: {type(exc).__name__}: {exc}",
                    "needs_review": True,
                }
                source = "rule_fallback_after_llm_error"
        else:
            adjudication = {
                "label": cand.rule_label,
                "allowed": cand.rule_allowed,
                "confidence": cand.rule_confidence,
                "target_link": target_link_from_rule(cand.rule_label),
                "rationale": cand.rule_reason,
                "needs_review": cand.rule_confidence < 0.75,
            }
        row.update(
            {
                "final_label": adjudication["label"],
                "allowed": bool(adjudication["allowed"]),
                "support_score": support_score(adjudication["label"], float(adjudication.get("confidence") or 0.0)),
                "target_link": adjudication.get("target_link") or target_link_from_rule(adjudication["label"]),
                "confidence": float(adjudication.get("confidence") or 0.0),
                "needs_review": bool(adjudication.get("needs_review")),
                "rationale": adjudication.get("rationale") or "",
                "adjudication_source": source,
                "adjudication_model": model,
                "raw_response": raw_response,
                "adjudicated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        labels.append(row)
        if (idx + 1) % 25 == 0:
            print(json.dumps({"processed": idx + 1, "llm_used": llm_used}, ensure_ascii=False), flush=True)
    return labels


class DashScopeClient:
    def __init__(self, args: argparse.Namespace) -> None:
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(f"DASHSCOPE_API_KEY is not set. Check {args.dotenv}")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=args.request_timeout,
        )
        self.models = dedupe([args.model] + [m.strip() for m in args.fallback_models.split(",") if m.strip()])
        self.args = args

    def adjudicate(self, cand: Candidate) -> tuple[dict[str, Any], str, str]:
        last_error: Exception | None = None
        for model in self.models:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=build_llm_messages(cand),
                    temperature=self.args.temperature,
                    max_tokens=self.args.max_tokens,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                parsed = parse_json_object(content)
                return parsed, model, content
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"all DashScope models failed: {last_error!r}")


def build_llm_messages(cand: Candidate) -> list[dict[str, str]]:
    system = (
        "你是中国绘画文献的证据支持性裁决器。你只判断给定 fragment 是否能支持 target figure 的指定字段。"
        "必须保守：相关不等于支持；同页另一张图、相邻图注、同画家其他作品都应标 wrong_target 或 no_support。"
    )
    user_obj = {
        "task_id": cand.task_id,
        "field": cand.field,
        "gold_value": cand.gold_value,
        "gold_abstain": cand.gold_abstain,
        "target_caption": cand.target_caption,
        "target_figure_labels": cand.target_figure_labels,
        "candidate_evidence": {
            "evidence_id": cand.evidence_id,
            "fragment_id": cand.fragment_id,
            "source_role": cand.source_role,
            "candidate_kind": cand.candidate_kind,
            "candidate_figure_labels": cand.candidate_figure_labels,
            "text": cand.fragment_text,
        },
        "rule_guess": {
            "label": cand.rule_label,
            "allowed": cand.rule_allowed,
            "confidence": cand.rule_confidence,
            "reason": cand.rule_reason,
        },
    }
    user = f"""请只输出 JSON object：
{{
  "label": "support|weak_support|no_support|contradict|wrong_target",
  "allowed": true,
  "target_link": "target|same_page_other_figure|same_artist_other_work|general_background|unknown",
  "confidence": 0.0,
  "needs_review": false,
  "rationale": "一句话中文理由"
}}

判定标准：
- support：fragment 明确支持该 target figure 的该字段值。
- weak_support：有弱线索，但不足以作为唯一最终证据。
- no_support：相关或背景信息，但不能支持该字段。
- contradict：fragment 与 gold_value 明确冲突。
- wrong_target：fragment 属于同页另一张图、另一件作品、相邻图注或同画家其他作品，不可用于目标图。
- 如果 gold_abstain=true，通常应为 no_support，除非 fragment 说明 gold 可能应修正为非 abstain，此时 needs_review=true。

待裁决样本：
{json.dumps(user_obj, ensure_ascii=False, indent=2)}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_adjudication(obj: dict[str, Any], cand: Candidate) -> dict[str, Any]:
    label = str(obj.get("label") or cand.rule_label)
    if label not in SUPPORT_LABELS:
        label = cand.rule_label
    confidence = obj.get("confidence")
    try:
        confidence_f = max(0.0, min(1.0, float(confidence)))
    except Exception:
        confidence_f = cand.rule_confidence
    allowed = obj.get("allowed")
    if allowed is None:
        allowed = label in {"support", "weak_support"}
    return {
        "label": label,
        "allowed": bool(allowed),
        "target_link": str(obj.get("target_link") or target_link_from_rule(label)),
        "confidence": confidence_f,
        "needs_review": bool(obj.get("needs_review")),
        "rationale": str(obj.get("rationale") or cand.rule_reason),
    }


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    selected: list[Candidate],
    labels: list[dict[str, Any]],
    rollout_records: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "dataset_version": "v1.0.4_gold_fragment_support_probe",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "tasks_path": str(args.tasks),
        "evidence_index": str(args.evidence_index),
        "task_count": len(tasks),
        "pair_count": len(labels),
        "llm_label_count": sum(1 for row in labels if row.get("adjudication_source") == "llm"),
        "rule_label_count": sum(1 for row in labels if row.get("adjudication_source") == "rule"),
        "fallback_label_count": sum(1 for row in labels if row.get("adjudication_source") == "rule_fallback_after_llm_error"),
        "label_counts": dict(Counter(row.get("final_label") for row in labels)),
        "field_counts": dict(Counter(row.get("field") for row in labels)),
        "candidate_kind_counts": dict(Counter(row.get("candidate_kind") for row in labels)),
        "source_role_counts": dict(Counter(row.get("source_role") or "unknown" for row in labels)),
        "cited_pair_count": sum(1 for row in labels if row.get("candidate_kind") == "model_cited"),
        "needs_review_count": sum(1 for row in labels if row.get("needs_review")),
        "rollout_runs": {name: len(records) for name, records in rollout_records.items()},
        "model": args.model if args.provider == "dashscope" else None,
        "provider": args.provider,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Gold Fragment Support Probe 构建报告",
        "",
        "## 摘要",
        "",
        f"- 数据集版本：`{summary['dataset_version']}`",
        f"- 样本数：{summary['pair_count']}",
        f"- GoldEval task 数：{summary['task_count']}",
        f"- LLM 裁决：{summary['llm_label_count']}",
        f"- 规则裁决：{summary['rule_label_count']}",
        f"- LLM 失败回退：{summary['fallback_label_count']}",
        f"- needs_review：{summary['needs_review_count']}",
        "",
        "## Label 分布",
        "",
    ]
    for key, value in sorted(summary["label_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Field 分布", ""])
    for key, value in sorted(summary["field_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Candidate Kind 分布", ""])
    for key, value in sorted(summary["candidate_kind_counts"].items()):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Rollout 覆盖", ""])
    for key, value in sorted(summary["rollout_runs"].items()):
        lines.append(f"- {key}: {value} tasks")
    lines.extend(
        [
            "",
            "## 用途",
            "",
            "该 probe 用于离线回放已有 rollout，计算 per-field claim/evidence support、wrong-target citation、local-caption overgeneralization 和 tool-induced gain/harm。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def support_score(label: str, confidence: float) -> float:
    if label == "support":
        return max(0.75, confidence)
    if label == "weak_support":
        return min(0.74, max(0.35, confidence))
    if label == "no_support":
        return 0.0
    if label == "wrong_target":
        return -0.75
    if label == "contradict":
        return -1.0
    return 0.0


def target_link_from_rule(label: str) -> str:
    if label == "wrong_target":
        return "same_page_other_figure"
    if label in {"support", "weak_support"}:
        return "target"
    return "unknown"


def make_pair_id(task_id: str, field: str, evidence_id: str, fragment_id: str) -> str:
    raw = f"{task_id}|{field}|{evidence_id}|{fragment_id}"
    return "fsp_" + str(abs(hash(raw))).zfill(20)


def load_evidence_index(index_dir: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for name in ["corpus_chunks.jsonl"]:
        path = index_dir / name
        if not path.exists():
            continue
        for row in read_jsonl(path):
            eid = str(row.get("evidence_id") or "")
            if eid:
                by_id[eid] = row
    return by_id


def parse_rollout_specs(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            name = Path(item).parent.name or Path(item).stem
            out[name] = item
            continue
        name, path = item.split("=", 1)
        out[name.strip()] = path.strip()
    return out


def figure_labels(text: str) -> list[str]:
    return dedupe([normalize_for_match(match.group(0)) for match in FIG_LABEL_RE.finditer(str(text or ""))])


def label_match(label: str, targets: list[str]) -> bool:
    norm = normalize_for_match(label)
    return any(norm == target or norm in target or target in norm for target in targets)


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def normalize_for_match(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("．", ".").replace("（", "(").replace("）", ")")
    text = re.sub(r"[《》\[\]【】()（）,，。.;；:：\s'\"“”‘’/_\\|-]+", "", text)
    return text


def text_overlap(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(1, max(len(a), len(b)))
    aset, bset = set(a), set(b)
    return len(aset & bset) / max(1, len(aset | bset))


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM response is not a JSON object")
    return obj


def load_dotenv(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def default_output_dir() -> str:
    return (
        "/root/datasets/evidence_grounded_vlm_agentrl/"
        f"v1_0_4_gold_fragment_support_probe_{datetime.now().strftime('%Y%m%d_%H%M')}"
    )


if __name__ == "__main__":
    main()
