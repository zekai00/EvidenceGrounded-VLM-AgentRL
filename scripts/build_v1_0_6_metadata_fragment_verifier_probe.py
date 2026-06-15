#!/usr/bin/env python3
"""Build a v1.0.6 Metadata5 field-level fragment verifier probe.

The probe tests whether local rules plus a small LLM verifier can judge
whether an evidence fragment supports a specific metadata field for a target
figure/work.  It is intentionally a probe, not a gold builder.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import EvidenceIndex  # noqa: E402


DEFAULT_INPUT_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_6_baselocate4_metadata5_probe_20260613_2244"
)
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_llm_overlay_20260611_0222"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DEFAULT_DOCS_DIR = REPO_ROOT / "docs" / "02_指标与数据"
DEFAULT_MODEL_PATH = Path("/root/models/Qwen3-4B-Instruct-2507")

META_FIELDS = [
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
]

FIELD_ZH = {
    "creator_or_attribution": "作者/传称作者",
    "creation_period_or_dynasty": "创作时期/朝代",
    "collection_institution": "收藏机构/藏馆",
    "dimensions": "尺寸",
    "medium_material": "材质",
}

FIELD_HINT = {
    "creator_or_attribution": "只能是画家、作者、传称作者或 after/attributed-to 对象，不能是“代表作/作品/图版”等谓词。",
    "creation_period_or_dynasty": "只能是朝代、时期、世纪或明确年份语境；五行的金木水火土、金碧、金陵等不是朝代。",
    "collection_institution": "只能是收藏/馆藏机构，例如博物馆、故宫、私人收藏等。",
    "dimensions": "只能是作品尺寸，如 183.6×110.2 厘米、长183.6厘米横110.2厘米。",
    "medium_material": "只能是材质媒介，如纸本、绢本、水墨、设色、ink on silk 等。",
}

LABELS = {"support", "no_support", "wrong_target", "ambiguous"}
FIGURE_LABEL_PATTERN = re.compile(
    r"(?:图|圖|fig\.?|figure)\s*[A-Za-z]?\s*[0-9一二三四五六七八九十百〇零IVXivx]+"
    r"(?:[.\-．:：][0-9一二三四五六七八九十百〇零IVXivx]+)*[a-zA-Z]?",
    flags=re.I,
)
TITLE_PATTERN = re.compile(r"《([^》]{1,80})》")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--evidence-index", type=Path, default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--docs-dir", type=Path, default=DEFAULT_DOCS_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sample-size", type=int, default=300)
    parser.add_argument("--human-review-size", type=int, default=50)
    parser.add_argument("--retrieve-hard-negative-top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260613)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-limit", type=int, default=0, help="0 means adjudicate every sampled row.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--timestamp", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d_%H%M")
    output_dir = args.output_root / f"agentbench_v1_0_6_metadata_fragment_verifier_probe_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    tasks = read_jsonl(args.input_dir / "metadata_probe" / "tasks_all.jsonl")
    index = EvidenceIndex(args.evidence_index)

    all_candidates = build_candidates(tasks, index, args)
    sampled = sample_candidates(all_candidates, args.sample_size, rng)

    if args.use_llm:
        adjudicate_with_llm(sampled, args)
    else:
        for row in sampled:
            row["llm_label"] = None
            row["llm_confidence"] = None
            row["llm_reason"] = "LLM adjudication disabled"
            row["final_label"] = fallback_final_label(row)
            row["verifier_status"] = "rule_only"

    review_rows = sample_human_review(sampled, args.human_review_size, rng)

    write_jsonl(output_dir / "fragment_candidates_all.jsonl", all_candidates)
    write_jsonl(output_dir / "fragment_probe_adjudicated.jsonl", sampled)
    write_jsonl(output_dir / "human_review_50.jsonl", review_rows)
    copy_review_assets(review_rows, output_dir / "review" / "assets")
    review_md = output_dir / "review" / f"{timestamp}_v1.0.6MetadataFragmentVerifierProbe人工抽检50.md"
    write_human_review_md(review_md, review_rows, output_dir)

    summary = build_summary(args, timestamp, output_dir, tasks, all_candidates, sampled, review_rows)
    write_json(output_dir / "manifest.json", summary)
    report = render_report(summary)
    (output_dir / "构建报告.md").write_text(report, encoding="utf-8")
    docs_report = args.docs_dir / f"{timestamp}_v1.0.6MetadataFragmentVerifierProbe报告.md"
    docs_report.parent.mkdir(parents=True, exist_ok=True)
    docs_report.write_text(report, encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_candidates(tasks: list[dict[str, Any]], index: EvidenceIndex, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for task in tasks:
        mp = task.get("metadata_probe") or {}
        caption = caption_text(task)
        title = title_from_task(task) or extract_title(caption)
        local_meta = mp.get("local_metadata") or {}
        missing_fields = mp.get("missing_metadata_fields_after_caption") or []
        local_eid = local_caption_evidence_id(task)

        for field in META_FIELDS:
            candidate_value = str(local_meta.get(field) or extract_field_value(field, caption, title) or "")
            prior = "support_candidate" if candidate_value else "no_support_candidate"
            row = make_candidate(
                task=task,
                field=field,
                evidence_id=local_eid or f"local_caption_{task.get('task_id')}",
                evidence_source="local_caption",
                fragment_text=caption,
                candidate_value=candidate_value,
                rule_prior=prior,
                rule_reason="local caption has extracted field value" if candidate_value else "local caption lacks extracted field value",
                title=title,
                caption=caption,
                source_quality="local_caption",
                citation_level="page_caption_region",
            )
            add_candidate(rows, seen, row)

        external_eids: set[str] = set()
        for evidence in mp.get("external_evidence") or []:
            eid = str(evidence.get("evidence_id") or "")
            if eid:
                external_eids.add(eid)
            fragment = choose_fragment(evidence.get("focused_text") or evidence.get("display_snippet") or evidence.get("text") or "")
            candidate_fields = evidence.get("candidate_fields") or {}
            for field in META_FIELDS:
                candidate_value = str(candidate_fields.get(field) or extract_field_value(field, fragment, title) or "")
                prior, reason = rule_prior_for_fragment(field, candidate_value, fragment, title, caption, field in candidate_fields)
                row = make_candidate(
                    task=task,
                    field=field,
                    evidence_id=eid,
                    evidence_source="external_evidence_pool",
                    fragment_text=fragment,
                    candidate_value=candidate_value,
                    rule_prior=prior,
                    rule_reason=reason,
                    title=title,
                    caption=caption,
                    source_file=evidence.get("source_file"),
                    page_start=evidence.get("page_start"),
                    page_end=evidence.get("page_end"),
                    source_quality=evidence.get("source_quality"),
                    citation_level=evidence.get("citation_level"),
                    clean_evidence_type=evidence.get("clean_evidence_type"),
                    authority_level=evidence.get("authority_level"),
                )
                add_candidate(rows, seen, row)

        retrieve_results = ((mp.get("retrieve_result") or {}).get("results") or [])[: args.retrieve_hard_negative_top_k]
        for result in retrieve_results:
            eid = str(result.get("evidence_id") or "")
            if not eid or eid in external_eids:
                continue
            full = index.open(eid) or result
            fragment = choose_fragment(
                full.get("display_snippet")
                or full.get("focused_text")
                or full.get("clean_text")
                or full.get("text")
                or result.get("display_snippet")
                or ""
            )
            target_fields = missing_fields or META_FIELDS
            for field in target_fields[:3]:
                candidate_value = extract_field_value(field, fragment, title)
                prior, reason = rule_prior_for_fragment(field, candidate_value, fragment, title, caption, False)
                if prior == "support_candidate":
                    prior = "ambiguous_candidate"
                    reason = "retrieved hard negative contains a field-like value but was not selected as support"
                row = make_candidate(
                    task=task,
                    field=field,
                    evidence_id=eid,
                    evidence_source="retrieved_hard_negative",
                    fragment_text=fragment,
                    candidate_value=candidate_value,
                    rule_prior=prior,
                    rule_reason=reason,
                    title=title,
                    caption=caption,
                    source_file=full.get("source_file") or result.get("source_file"),
                    page_start=full.get("page_start") if full.get("page_start") is not None else full.get("page"),
                    page_end=full.get("page_end"),
                    source_quality=full.get("source_quality") or result.get("source_quality"),
                    citation_level=full.get("citation_level") or result.get("citation_level"),
                    clean_evidence_type=full.get("clean_evidence_type") or result.get("clean_evidence_type"),
                    authority_level=full.get("authority_level") or result.get("authority_level"),
                )
                add_candidate(rows, seen, row)
    for idx, row in enumerate(rows, start=1):
        row["candidate_id"] = f"fragcand_{idx:06d}"
    return rows


def add_candidate(rows: list[dict[str, Any]], seen: set[tuple[str, str, str, str]], row: dict[str, Any]) -> None:
    if not row.get("fragment_text"):
        return
    key = (
        str(row.get("task_id")),
        str(row.get("field")),
        str(row.get("evidence_id")),
        str(row.get("fragment_hash")),
    )
    if key in seen:
        return
    seen.add(key)
    rows.append(row)


def make_candidate(
    *,
    task: dict[str, Any],
    field: str,
    evidence_id: str,
    evidence_source: str,
    fragment_text: str,
    candidate_value: Any,
    rule_prior: str,
    rule_reason: str,
    title: str,
    caption: str,
    source_file: Any = None,
    page_start: Any = None,
    page_end: Any = None,
    source_quality: Any = None,
    citation_level: Any = None,
    clean_evidence_type: Any = None,
    authority_level: Any = None,
) -> dict[str, Any]:
    fragment_text = normalize_space(fragment_text)
    candidate_value = normalize_space(candidate_value)
    improved_prior, improved_reason = apply_obvious_rule_overrides(field, candidate_value, fragment_text, title, caption, rule_prior, rule_reason)
    return {
        "candidate_id": "",
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "abc_class": (task.get("metadata_probe") or {}).get("abc_class"),
        "source_file": source_file or task.get("source_file"),
        "page": task.get("page"),
        "title": title,
        "caption_text": caption,
        "field": field,
        "field_zh": FIELD_ZH[field],
        "candidate_value": candidate_value,
        "candidate_value_norm": normalize_field_value(field, candidate_value),
        "evidence_id": evidence_id,
        "evidence_source": evidence_source,
        "source_quality": source_quality,
        "citation_level": citation_level,
        "clean_evidence_type": clean_evidence_type,
        "authority_level": authority_level,
        "page_start": page_start,
        "page_end": page_end,
        "fragment_text": fragment_text,
        "fragment_hash": sha1_text(fragment_text)[:16],
        "rule_prior": improved_prior,
        "rule_reason": improved_reason,
        "overlay_image": task.get("overlay_image"),
        "artwork_image": task.get("artwork_image"),
        "page_image": task.get("page_image"),
    }


def rule_prior_for_fragment(
    field: str,
    candidate_value: Any,
    fragment: str,
    title: str,
    caption: str,
    selected_as_support: bool,
) -> tuple[str, str]:
    candidate_value = normalize_space(candidate_value)
    if selected_as_support:
        return "support_candidate", "upstream Metadata5 builder selected this field as supported"
    if candidate_value:
        if title and not title_or_strict_anchor_matches(fragment, title, caption):
            return "wrong_target_candidate", "field-like value appears but target title/strict anchor is missing"
        return "ambiguous_candidate", "field-like value appears but upstream builder did not select it as support"
    if title and not title_or_strict_anchor_matches(fragment, title, caption):
        return "wrong_target_candidate", "fragment does not contain target title or strict caption anchor"
    return "no_support_candidate", "no field-like value extracted from fragment"


def apply_obvious_rule_overrides(
    field: str,
    candidate_value: str,
    fragment: str,
    title: str,
    caption: str,
    prior: str,
    reason: str,
) -> tuple[str, str]:
    if field == "creator_or_attribution" and is_bad_creator(candidate_value):
        return "obvious_reject", f"creator candidate `{candidate_value}` is a known predicate/non-person phrase"
    if field == "creation_period_or_dynasty" and is_bad_period(candidate_value, fragment):
        return "obvious_reject", f"period candidate `{candidate_value}` occurs in non-dynasty context"
    if title and candidate_value and not title_or_strict_anchor_matches(fragment, title, caption):
        return "wrong_target_candidate", "candidate value appears without target title or strict caption anchor"
    return prior, reason


def sample_candidates(rows: list[dict[str, Any]], target_size: int, rng: random.Random) -> list[dict[str, Any]]:
    if target_size <= 0 or target_size >= len(rows):
        sampled = list(rows)
    else:
        per_field = max(1, target_size // len(META_FIELDS))
        sampled = []
        selected_keys: set[str] = set()
        by_field: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_field[str(row.get("field"))].append(row)
        for field in META_FIELDS:
            field_rows = by_field.get(field, [])
            rng.shuffle(field_rows)
            chosen = stratified_take(field_rows, per_field, rng)
            sampled.extend(chosen)
            selected_keys.update(str(row.get("candidate_id")) for row in chosen)
        if len(sampled) < target_size:
            remaining = [row for row in rows if str(row.get("candidate_id")) not in selected_keys]
            rng.shuffle(remaining)
            sampled.extend(remaining[: target_size - len(sampled)])
        sampled = sampled[:target_size]
    for idx, row in enumerate(sampled, start=1):
        row["probe_id"] = f"mfvp_{idx:04d}"
    return sampled


def stratified_take(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    by_prior: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_prior[str(row.get("rule_prior"))].append(row)
    for group in by_prior.values():
        rng.shuffle(group)
    order = ["support_candidate", "ambiguous_candidate", "wrong_target_candidate", "obvious_reject", "no_support_candidate"]
    out: list[dict[str, Any]] = []
    while len(out) < limit and any(by_prior.values()):
        progressed = False
        for prior in order:
            group = by_prior.get(prior) or []
            if group and len(out) < limit:
                out.append(group.pop())
                progressed = True
        if not progressed:
            break
    return out


def adjudicate_with_llm(rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[LLM] loading model: {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_path),
        torch_dtype=torch.float16,
        device_map=None,
        trust_remote_code=True,
    )
    model.to(args.device)
    model.eval()

    limit = args.llm_limit or len(rows)
    for idx, row in enumerate(rows, start=1):
        if idx > limit:
            row["llm_label"] = None
            row["llm_confidence"] = None
            row["llm_reason"] = "Skipped by llm-limit"
            row["final_label"] = fallback_final_label(row)
            row["verifier_status"] = "skipped_llm"
            continue
        prompt = build_llm_prompt(row)
        try:
            raw = generate_llm_json(model, tokenizer, prompt, args.device, args.max_new_tokens)
            parsed = parse_json_object(raw)
            label = normalize_space(parsed.get("label"))
            if label not in LABELS:
                raise ValueError(f"invalid label: {label}")
            row["llm_label"] = label
            row["llm_confidence"] = parsed.get("confidence")
            row["llm_reason"] = normalize_space(parsed.get("reason"))
            row["llm_fragment_quote"] = normalize_space(parsed.get("fragment_quote"))
            row["llm_normalized_value"] = normalize_space(parsed.get("normalized_value"))
            allowed = parsed.get("allowed_values")
            row["llm_allowed_values"] = allowed if isinstance(allowed, list) else []
            row["llm_raw_response"] = raw
            row["final_label"] = label
            row["verifier_status"] = "llm_ok"
        except Exception as exc:
            row["llm_label"] = None
            row["llm_confidence"] = None
            row["llm_reason"] = f"LLM parse/error: {exc}"
            row["llm_raw_response"] = locals().get("raw", "")
            row["final_label"] = fallback_final_label(row)
            row["verifier_status"] = "llm_error"
        if idx == 1 or idx % 25 == 0:
            print(f"[LLM] adjudicated {idx}/{min(limit, len(rows))}", flush=True)


def build_llm_prompt(row: dict[str, Any]) -> str:
    field = str(row.get("field"))
    payload = {
        "task_id": row.get("task_id"),
        "target_caption": row.get("caption_text"),
        "target_title": row.get("title"),
        "field": field,
        "field_meaning": FIELD_ZH.get(field),
        "candidate_value": row.get("candidate_value") or None,
        "evidence_id": row.get("evidence_id"),
        "evidence_source": row.get("evidence_source"),
        "fragment_text": row.get("fragment_text"),
        "rule_prior": row.get("rule_prior"),
        "rule_reason": row.get("rule_reason"),
    }
    return (
        "你是 EvidenceGrounded-VLM-AgentRL 的字段级证据裁决器。"
        "只能根据给定 fragment_text 判断，不允许使用模型内部知识补全。\n"
        "目标：判断 fragment_text 是否支持 target_caption/target_title 所指目标作品的指定 field。\n"
        "标签定义：\n"
        "- support：fragment 明确支持该字段，并且绑定到目标作品/目标图。\n"
        "- no_support：fragment 相关或不相关，但没有支持该字段。\n"
        "- wrong_target：fragment 里有类似字段信息，但绑定到别的作品、别的图号或别的对象。\n"
        "- ambiguous：可能支持，但目标绑定、字段类型或文本质量不够清楚。\n"
        f"字段说明：{FIELD_HINT.get(field)}\n"
        "如果 candidate_value 为空，但 fragment 明确支持该字段，也应输出 support 并给 normalized_value。\n"
        "如果 candidate_value 是明显错误字段值，即使 fragment 与作品相关，也不要输出 support。\n"
        "只输出一个 JSON 对象，不要 markdown，不要解释 JSON 以外的文本。\n"
        "JSON schema："
        '{"label":"support|no_support|wrong_target|ambiguous","normalized_value":null或字符串,'
        '"allowed_values":[字符串数组],"fragment_quote":"来自 fragment_text 的短引用或空串",'
        '"reason":"一句中文理由","confidence":0到1}\n\n'
        f"输入：\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def generate_llm_json(model: Any, tokenizer: Any, prompt: str, device: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "你是严格的 JSON 输出器。"},
        {"role": "user", "content": prompt},
    ]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    import torch

    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found")
    return json.loads(text[start : end + 1])


def fallback_final_label(row: dict[str, Any]) -> str:
    prior = str(row.get("rule_prior") or "")
    return {
        "support_candidate": "support",
        "wrong_target_candidate": "wrong_target",
        "obvious_reject": "no_support",
        "ambiguous_candidate": "ambiguous",
        "no_support_candidate": "no_support",
    }.get(prior, "ambiguous")


def sample_human_review(rows: list[dict[str, Any]], target_size: int, rng: random.Random) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row.get("final_label"))].append(row)
    for group in by_label.values():
        rng.shuffle(group)
    out: list[dict[str, Any]] = []
    order = ["support", "wrong_target", "no_support", "ambiguous"]
    while len(out) < target_size and any(by_label.values()):
        progressed = False
        for label in order:
            group = by_label.get(label) or []
            if group and len(out) < target_size:
                out.append(group.pop())
                progressed = True
        if not progressed:
            break
    for idx, row in enumerate(out, start=1):
        row["human_review_id"] = f"MFHR{idx:03d}"
    return out


def copy_review_assets(rows: list[dict[str, Any]], assets_dir: Path) -> None:
    assets_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        rid = str(row.get("human_review_id") or row.get("probe_id") or "sample")
        for key, suffix in [("overlay_image", "overlay"), ("artwork_image", "crop")]:
            src = Path(str(row.get(key) or ""))
            if src.exists():
                dst = assets_dir / f"{rid}_{suffix}{src.suffix.lower() or '.jpg'}"
                shutil.copy2(src, dst)
                row[f"{suffix}_asset"] = "assets/" + dst.name


def write_human_review_md(path: Path, rows: list[dict[str, Any]], output_dir: Path) -> None:
    lines = [
        "# v1.0.6 Metadata Fragment Verifier Probe 人工抽检 50",
        "",
        "请重点判断 `final_label` 是否合理，以及 fragment 是否真的支持目标字段。",
        "",
        "标签：`support` / `no_support` / `wrong_target` / `ambiguous`。",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row.get('human_review_id')} {row.get('task_id')} `{row.get('field')}`",
                "",
                f"- split/class：`{row.get('split')}` / `{row.get('abc_class')}`",
                f"- source/page：`{row.get('source_file')}` / `{row.get('page')}`",
                f"- caption：{row.get('caption_text')}",
                f"- title：`{row.get('title') or 'ABSTAIN/空'}`",
                f"- evidence：`{row.get('evidence_id')}` / `{row.get('evidence_source')}` / `{row.get('source_quality')}`",
                f"- candidate_value：`{row.get('candidate_value') or ''}`",
                f"- rule_prior：`{row.get('rule_prior')}`，{row.get('rule_reason')}",
                f"- final_label：`{row.get('final_label')}`",
                f"- llm_label：`{row.get('llm_label')}`，confidence=`{row.get('llm_confidence')}`",
                f"- llm_normalized_value：`{row.get('llm_normalized_value') or ''}`",
                f"- llm_reason：{row.get('llm_reason') or ''}",
                "",
            ]
        )
        if row.get("overlay_asset"):
            lines.extend([f"![overlay]({row.get('overlay_asset')})", ""])
        if row.get("crop_asset"):
            lines.extend([f"![crop]({row.get('crop_asset')})", ""])
        lines.extend(
            [
                "fragment:",
                "",
                "```text",
                str(row.get("fragment_text") or ""),
                "```",
                "",
                "人工裁决：`accept / fix_label / reject / unsure`",
                "",
                "备注：",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(
    args: argparse.Namespace,
    timestamp: str,
    output_dir: Path,
    tasks: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    sampled: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "timestamp": timestamp,
        "builder": "scripts/build_v1_0_6_metadata_fragment_verifier_probe.py",
        "input_dir": str(args.input_dir),
        "evidence_index": str(args.evidence_index),
        "model_path": str(args.model_path),
        "use_llm": bool(args.use_llm),
        "sample_size_requested": args.sample_size,
        "output_dir": str(output_dir),
        "summary": {
            "tasks": len(tasks),
            "all_candidates": len(all_candidates),
            "sampled_candidates": len(sampled),
            "human_review_rows": len(review_rows),
            "sampled_by_field": dict(Counter(row.get("field") for row in sampled)),
            "sampled_by_rule_prior": dict(Counter(row.get("rule_prior") for row in sampled)),
            "sampled_by_final_label": dict(Counter(row.get("final_label") for row in sampled)),
            "sampled_by_llm_status": dict(Counter(row.get("verifier_status") for row in sampled)),
            "sampled_by_source": dict(Counter(row.get("evidence_source") for row in sampled)),
            "sampled_by_source_quality": dict(Counter(row.get("source_quality") for row in sampled)),
            "parse_or_llm_errors": sum(1 for row in sampled if row.get("verifier_status") == "llm_error"),
        },
        "artifacts": {
            "all_candidates": str(output_dir / "fragment_candidates_all.jsonl"),
            "adjudicated": str(output_dir / "fragment_probe_adjudicated.jsonl"),
            "human_review_jsonl": str(output_dir / "human_review_50.jsonl"),
            "human_review_md": str(output_dir / "review" / f"{timestamp}_v1.0.6MetadataFragmentVerifierProbe人工抽检50.md"),
            "report": str(output_dir / "构建报告.md"),
        },
    }


def render_report(summary: dict[str, Any]) -> str:
    s = summary["summary"]
    lines = [
        "# v1.0.6 Metadata Fragment Verifier Probe 报告",
        "",
        f"时间：{summary['created_at']}",
        "",
        "## 目标",
        "",
        "本次 probe 用于验证：把 evidence chunk 拆成 field-level fragment 后，规则预筛 + 本地 Qwen3-4B-Instruct-2507 是否能稳定裁决 `support / no_support / wrong_target / ambiguous`。",
        "",
        "它不是最终 gold 数据集，而是下一步构建 Metadata5 verifier cache 的质量探针。",
        "",
        "## 输入与输出",
        "",
        f"- 输入数据：`{summary['input_dir']}`",
        f"- evidence index：`{summary['evidence_index']}`",
        f"- verifier 模型：`{summary['model_path']}`",
        f"- 输出目录：`{summary['output_dir']}`",
        "",
        "## 统计",
        "",
        f"- tasks：{s['tasks']}",
        f"- all candidate fragments：{s['all_candidates']}",
        f"- sampled/adjudicated fragments：{s['sampled_candidates']}",
        f"- human review rows：{s['human_review_rows']}",
        f"- parse_or_llm_errors：{s['parse_or_llm_errors']}",
        f"- sampled_by_field：`{s['sampled_by_field']}`",
        f"- sampled_by_rule_prior：`{s['sampled_by_rule_prior']}`",
        f"- sampled_by_final_label：`{s['sampled_by_final_label']}`",
        f"- sampled_by_llm_status：`{s['sampled_by_llm_status']}`",
        f"- sampled_by_source：`{s['sampled_by_source']}`",
        f"- sampled_by_source_quality：`{s['sampled_by_source_quality']}`",
        "",
        "## 人工抽检包",
        "",
        f"- `{summary['artifacts']['human_review_md']}`",
        "",
        "人工抽检时建议重点看三类：",
        "",
        "1. `support` 是否真的绑定到目标作品。",
        "2. `wrong_target` 是否准确识别了同图号/相邻作品串扰。",
        "3. `no_support` 是否误杀了可支持字段的 fragment。",
        "",
        "## 当前结论",
        "",
        "本报告只给出构建结果；最终是否扩到 2k-4k val/test gold fragments，需要人工抽检 50 条后决定。",
        "",
    ]
    return "\n".join(lines)


def choose_fragment(text: Any, limit: int = 700) -> str:
    text = normalize_space(text)
    if len(text) <= limit:
        return text
    sentences = split_sentences(text)
    selected: list[str] = []
    for sent in sentences:
        if len(" ".join(selected + [sent])) <= limit:
            selected.append(sent)
        if len(" ".join(selected)) >= limit * 0.75:
            break
    if selected:
        return normalize_space(" ".join(selected))
    return text[:limit]


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；;.!?])\s+|\n+", str(text or ""))
    return [normalize_space(part) for part in parts if normalize_space(part)]


def caption_text(task: dict[str, Any]) -> str:
    for claim in (task.get("gold") or {}).get("claims") or []:
        if claim.get("field") == "caption_text" and not claim.get("abstain"):
            return normalize_space(claim.get("value"))
    for item in task.get("local_evidence") or []:
        eid = str(item.get("evidence_id") or "")
        if eid.startswith("local_caption_"):
            return normalize_space(item.get("display_snippet") or item.get("text"))
    return ""


def title_from_task(task: dict[str, Any]) -> str:
    for claim in (task.get("gold") or {}).get("claims") or []:
        if claim.get("field") == "depicted_work_title" and not claim.get("abstain"):
            return normalize_space(claim.get("value"))
    return ""


def local_caption_evidence_id(task: dict[str, Any]) -> str:
    for item in task.get("local_evidence") or []:
        eid = str(item.get("evidence_id") or "")
        if eid.startswith("local_caption_"):
            return eid
    for claim in (task.get("gold") or {}).get("claims") or []:
        if claim.get("field") == "caption_text":
            for eid in claim.get("evidence_ids") or []:
                if str(eid).startswith("local_caption_"):
                    return str(eid)
    return ""


def extract_title(text: str) -> str:
    text = normalize_space(text)
    match = TITLE_PATTERN.search(text)
    if match:
        return normalize_space(match.group(1))
    english = re.search(
        r"(?:Attributed to|After)?\s*[A-Z][A-Za-z'\-. ]{2,80}(?:\([^)]{0,80}\))?,\s*([^,;.]{3,100}),\s*(?:dated|ca\.|[0-9]{3,4}|Northern|Southern|Ming|Song|Yuan|Qing|Hanging|Album|Handscroll)",
        text,
    )
    if english:
        return normalize_space(english.group(1))
    return ""


def extract_field_value(field: str, text: str, title: str = "") -> str:
    text = normalize_space(text)
    if field == "creator_or_attribution":
        return extract_creator(text, title)
    if field == "creation_period_or_dynasty":
        return extract_period(text)
    if field == "collection_institution":
        return extract_collection(text)
    if field == "dimensions":
        return extract_dimensions(text)
    if field == "medium_material":
        return extract_medium(text)
    return ""


def extract_creator(text: str, title: str = "") -> str:
    if title:
        escaped = re.escape(title)
        patterns = [
            rf"([\u4e00-\u9fff]{{2,4}})的(?:作品|画作|山水画)?[^。；;]{{0,30}}(?:代表作|名作|作品)?\s*《{escaped}》",
            rf"(?:北宋|南宋|宋代|宋|元代|元|明代|明|清代|清|唐代|唐|五代|辽|遼|金代|金|民国|民國)?[·\s・]*(传|傳)?([\u4e00-\u9fff]{{2,4}})\s*《{escaped}》",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                value = "".join(group for group in match.groups() if group)
                if not is_bad_creator(value):
                    return normalize_space(value)
    match = re.search(r"(?:Attributed to|After)\s+([A-Z][A-Za-z'\-. ]{2,80})", text, flags=re.I)
    if match:
        return normalize_space(match.group(0).rstrip(",. "))
    match = re.search(r"(?:Figure|Fig\.?|Plate)\s*[0-9.\-a-zA-Z]+\s+([A-Z][A-Za-z'\-. ]{2,60})(?:\s*\([^)]{0,80}\))?,", text, flags=re.I)
    if match:
        return normalize_space(match.group(1))
    return ""


def extract_period(text: str) -> str:
    period_re = re.compile(
        r"(北宋|南宋|宋代|元代|明代|清代|唐代|金代|辽代|遼代|五代|民国|民國|"
        r"Northern Song|Southern Song|Song dynasty|Yuan dynasty|Ming dynasty|Qing dynasty|Tang dynasty|"
        r"[0-9]{1,2}(?:st|nd|rd|th)?[-–—]?[0-9]{0,2}(?:st|nd|rd|th)? century|ca\.\s*[0-9]{3,4}(?:[-–—][0-9]{2,4})?|[0-9]{3,4}s)",
        flags=re.I,
    )
    match = period_re.search(text or "")
    if match:
        return normalize_space(match.group(1))
    single = re.search(r"(唐|宋|元|明|清|金|辽|遼)(?=代|[·\s・]*[\u4e00-\u9fff]{2,4}《)", text or "")
    return normalize_space(single.group(1)) if single else ""


def extract_collection(text: str) -> str:
    patterns = [
        r"((?:北京)?故宫博物院|台北故宫博物院|臺北故宮博物院|上海博物馆|上海博物館|辽宁省博物馆|遼寧省博物館|"
        r"大都会博物馆|大都會博物館|The Metropolitan Museum of Art|National Palace Museum|Palace Museum|"
        r"Nelson-Atkins Museum of Art|Idemitsu Museum of Arts|C\.?\s*C\.?\s*Wang family collection)",
        r"(?:藏于|藏於|收藏于|收藏於|现藏于|現藏於|collection of|collected by)\s*([^，。；;,.]{2,80}(?:博物馆|博物館|Museum|collection|美术馆|藝術館|Art Gallery))",
    ]
    for pattern in patterns:
        match = re.search(pattern, text or "", flags=re.I)
        if match:
            return normalize_space(match.group(1))
    return ""


def extract_dimensions(text: str) -> str:
    text = normalize_space(text)
    direct = re.search(
        r"(\d+(?:\.\d+)?\s*(?:×|x|X|\*)\s*\d+(?:\.\d+)?(?:\s*(?:×|x|X|\*)\s*\d+(?:\.\d+)?)?\s*(?:厘米|公分|cm|CM|in\.?|英寸))",
        text,
    )
    if direct:
        return normalize_space(direct.group(1))
    prose = re.search(
        r"(?:长|長|纵|縱|高|H\.?)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(厘米|公分|cm|CM|in\.?|英寸)"
        r"[，,、\s]*(?:横|宽|寬|W\.?)\s*[:：]?\s*(\d+(?:\.\d+)?)\s*(厘米|公分|cm|CM|in\.?|英寸)?",
        text,
        flags=re.I,
    )
    if prose:
        unit = prose.group(4) or prose.group(2)
        return normalize_space(f"{prose.group(1)}×{prose.group(3)} {unit}")
    return ""


def extract_medium(text: str) -> str:
    patterns = [
        (r"绢本设色|絹本設色", "绢本设色"),
        (r"纸本设色|紙本設色", "纸本设色"),
        (r"绢本|絹本", "绢本"),
        (r"纸本|紙本", "纸本"),
        (r"设色|設色", "设色"),
        (r"水墨", "水墨"),
        (r"ink and color on silk", "ink and color on silk"),
        (r"ink and color on paper", "ink and color on paper"),
        (r"ink on silk", "ink on silk"),
        (r"ink on paper", "ink on paper"),
        (r"color on silk", "color on silk"),
    ]
    for pattern, value in patterns:
        if re.search(pattern, text or "", flags=re.I):
            return value
    return ""


def title_or_strict_anchor_matches(fragment: str, title: str, caption: str) -> bool:
    frag = compact(fragment)
    title_norm = compact(title)
    if title_norm and len(title_norm) >= 3 and title_norm in frag:
        return True
    labels = [compact(label) for label in FIGURE_LABEL_PATTERN.findall(caption or "")]
    caption_terms = [compact(term) for term in TITLE_PATTERN.findall(caption or "")]
    for label in labels:
        if label and label in frag:
            if caption_terms:
                return any(term and term in frag for term in caption_terms)
            return True
    return False


def is_bad_creator(value: str) -> bool:
    value = normalize_space(value)
    if not value:
        return False
    bad_exact = {
        "代表作",
        "代表作品",
        "作品",
        "画作",
        "名作",
        "图版",
        "局部",
        "部分",
        "本段",
        "该图",
        "其作品",
        "传世名作",
    }
    if value in bad_exact:
        return True
    return bool(re.search(r"(代表|作品|画作|名作|图版|局部|本段|该图|其作品|构图|画面)", value))


def is_bad_period(value: str, fragment: str) -> bool:
    value = normalize_space(value)
    if value in {"金"} and re.search(r"五行|五星|金、木、水、火|金木水火土|水火土|金碧|泥金|金陵", fragment):
        return True
    if value in {"唐", "宋", "元", "明", "清", "金", "辽", "遼"} and not re.search(
        rf"{re.escape(value)}代|{re.escape(value)}[·\s・]*[\u4e00-\u9fff]{{2,4}}《|{re.escape(value)} dynasty",
        fragment,
        flags=re.I,
    ):
        return True
    return False


def normalize_field_value(field: str, value: Any) -> str:
    value = normalize_space(value)
    if not value:
        return ""
    if field == "dimensions":
        return re.sub(r"\s+", "", value.lower()).replace("×", "x").replace("*", "x").replace("厘米", "cm").replace("公分", "cm")
    if field == "creation_period_or_dynasty":
        mapping = {
            "明代": "明",
            "Ming dynasty": "明",
            "宋代": "宋",
            "Song dynasty": "宋",
            "Northern Song": "北宋",
            "Southern Song": "南宋",
            "元代": "元",
            "Yuan dynasty": "元",
            "清代": "清",
            "Qing dynasty": "清",
            "唐代": "唐",
            "Tang dynasty": "唐",
            "金代": "金",
        }
        return mapping.get(value, value)
    return compact(value)


def figure_labels(text: str) -> list[str]:
    return [normalize_space(match.group(0)) for match in FIGURE_LABEL_PATTERN.finditer(text or "")]


def sha1_text(text: str) -> str:
    return hashlib.sha1(normalize_space(text).encode("utf-8")).hexdigest()


def compact(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("圖", "图").replace("（", "(").replace("）", ")").replace("．", ".").replace("·", ".")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
