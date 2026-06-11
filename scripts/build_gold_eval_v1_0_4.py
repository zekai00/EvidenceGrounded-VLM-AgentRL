#!/usr/bin/env python3
"""Build a small VLM-reviewed GoldEval split for v1.0.4."""

from __future__ import annotations

import argparse
import base64
import copy
import html
import json
import os
import random
import re
import tempfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


DEFAULT_DATASET_DIR = "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615"
DEFAULT_OUTPUT_ROOT = "/root/datasets/evidence_grounded_vlm_agentrl"
DEFAULT_DOTENV = "/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/.env"
DEFAULT_MODEL = "qwen3.7-max-2026-06-08"
DEFAULT_FALLBACK_MODELS = (
    "qwen3.7-max,"
    "qwen3.7-max-preview,"
    "qwen3.7-plus-2026-05-26,"
    "qwen3.7-plus,"
    "qwen3.6-plus,"
    "qwen3.5-plus-2026-04-20,"
    "qwen3.6-27b,"
    "glm-5.1,"
    "kimi-k2.6,"
    "deepseek-v4-pro,"
    "deepseek-v4-flash"
)

PROMPT = """你是 EvidenceGrounded-VLM-AgentRL 的 GoldEval 数据复核员。你会看到两张图：
1. PDF 页面 overlay：红框是候选目标图像区域，青色/蓝色框是候选图注或文本区域。
2. 红框裁剪图：候选目标图像。

你的任务不是重写样本，而是判断这条现有 silver task 能否进入小规模评测用 GoldEval。

只输出 JSON 对象，不要输出 Markdown。字段必须为：
{
  "status": "accepted_gold|needs_human_review|rejected",
  "is_valid_target": true,
  "is_landscape_related": true,
  "bbox_quality": "good|partial|too_large|too_small|text_region|non_image|unclear",
  "caption_match": "good|partial|wrong|missing|unclear",
  "caption_quality": "clean|minor_ocr_noise|truncated|body_text|toc_or_index|too_short|wrong_language|unclear",
  "gold_claims_usable": true,
  "core_evidence_ok": true,
  "hard_negative_risk": false,
  "confidence": 0.0,
  "reason": "一句话说明",
  "required_human_checks": [],
  "suggested_caption_text": ""
}

裁决标准：
- accepted_gold：目标是山水画相关图像或明确的山水画局部；红框基本框住目标；图注与目标匹配为 good/partial；当前 Core5 gold claims 可直接用于评测，允许 title/dynasty/displayed_region 等字段 abstain。
- needs_human_review：目标可能可用，但图注截断、OCR 噪声、局部/全幅边界、作品名或证据边界需要人工确认。
- rejected：红框是文字/表格/目录/非图像；目标不是山水画相关；图注明显错配/缺失；或当前 gold claim 会把正文句子、目录文字当作图注。
- 古画可能没有作品名，也可能只是大图的一部分；只要当前样本明确可评价“目标图像/局部 + 图注/证据”，可以 accepted_gold。
- 不要凭空补事实。suggested_caption_text 只在页面图注明显可读且需要轻微纠错时填写，否则留空。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build val_gold_50/test_gold_100 with VLM review.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--version", default="v1.0.4")
    parser.add_argument("--val-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--provider", choices=["dashscope", "offline"], default="dashscope")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--dotenv", default=DEFAULT_DOTENV)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--image-max-side", type=int, default=1200)
    parser.add_argument("--crop-max-side", type=int, default=768)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-review-val", type=int, default=200)
    parser.add_argument("--max-review-test", type=int, default=200)
    parser.add_argument("--min-confidence", type=float, default=0.78)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=0, help="Review only N candidates per split for API smoke tests.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(Path(args.dotenv))
    dataset_dir = Path(args.dataset_dir)
    output_dir = resolve_output_dir(args)
    if output_dir.exists() and args.overwrite:
        for child in output_dir.iterdir():
            if child.is_dir():
                remove_tree(child)
            else:
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "review").mkdir(parents=True, exist_ok=True)

    val_rows = read_jsonl(dataset_dir / "val_tasks.jsonl")
    test_rows = read_jsonl(dataset_dir / "test_tasks.jsonl")
    prior = load_prior_reviews(dataset_dir)
    client = make_client(args)

    split_results: dict[str, dict[str, Any]] = {}
    for split, rows, target_size, max_review in [
        ("val", val_rows, args.val_size, args.max_review_val),
        ("test", test_rows, args.test_size, args.max_review_test),
    ]:
        if args.smoke_limit:
            max_review = min(max_review, args.smoke_limit)
            target_size = min(target_size, args.smoke_limit)
        result = build_split(split, rows, target_size, max_review, prior, client, args, output_dir)
        split_results[split] = result

    summary = build_summary(args, dataset_dir, output_dir, val_rows, test_rows, split_results)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "manifest.json", build_manifest(args, dataset_dir, output_dir, summary))
    write_report(output_dir / "构建报告.md", summary)
    write_review_html(output_dir / "review" / "review.html", split_results)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(args.output_root) / f"gold_eval_v1_0_4_{stamp}"


def build_split(
    split: str,
    rows: list[dict[str, Any]],
    target_size: int,
    max_review: int,
    prior: dict[str, dict[str, Any]],
    client: "ReviewClient",
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    rng = random.Random(args.seed + (17 if split == "test" else 0))
    ordered = stratified_order(rows, prior, rng)
    stream_path = output_dir / "review" / f"{split}_reviewed_stream.jsonl"
    reviewed = load_existing_stream(stream_path) if args.resume else []
    reviewed_by_id = {row.get("task_id"): row for row in reviewed if row.get("task_id")}

    accepted: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    review_count = len(reviewed_by_id)

    for task in ordered:
        task_id = task.get("task_id")
        decision_row = reviewed_by_id.get(task_id)
        if decision_row is None:
            if review_count >= max_review:
                break
            decision_row = review_one(split, task, prior.get(str(task_id)), client, args)
            append_jsonl(stream_path, [decision_row])
            reviewed_by_id[str(task_id)] = decision_row
            reviewed.append(decision_row)
            review_count += 1
            if args.sleep:
                time.sleep(args.sleep)

        status = classify_decision(decision_row, args.min_confidence)
        if status == "accepted_gold":
            accepted.append(annotate_gold_task(task, decision_row, args))
        elif status == "needs_human_review":
            review_queue.append(make_queue_row(task, decision_row, status))
        else:
            rejected.append(make_queue_row(task, decision_row, status))

        if len(accepted) >= target_size:
            break

    accepted = accepted[:target_size]
    short = f"{split}_gold_{target_size}.jsonl"
    write_jsonl(output_dir / short, accepted)
    write_jsonl(output_dir / "review" / f"{split}_review_queue.jsonl", review_queue)
    write_jsonl(output_dir / "review" / f"{split}_rejected.jsonl", rejected)
    write_jsonl(output_dir / "review" / f"{split}_accepted_review.jsonl", [row.get("gold_eval", {}) for row in accepted])

    result = {
        "split": split,
        "target_size": target_size,
        "pool_rows": len(rows),
        "reviewed_rows": review_count,
        "accepted_rows": len(accepted),
        "review_queue_rows": len(review_queue),
        "rejected_rows": len(rejected),
        "output_file": str(output_dir / short),
        "stream_file": str(stream_path),
        "accepted_tasks": accepted,
        "review_queue": review_queue,
        "rejected": rejected,
        "source_counts": dict(Counter(task.get("source_file") for task in accepted)),
        "authority_counts": dict(Counter(((task.get("candidate_meta") or {}).get("source_authority_level") or "unknown") for task in accepted)),
        "candidate_source_counts": dict(Counter(((task.get("candidate_meta") or {}).get("source") or "unknown") for task in accepted)),
        "review_model_counts": dict(Counter(((task.get("gold_eval") or {}).get("review_model") or "unknown") for task in accepted)),
        "caption_quality_counts": dict(Counter(((task.get("gold_eval") or {}).get("decision") or {}).get("caption_quality", "unknown") for task in accepted)),
        "bbox_quality_counts": dict(Counter(((task.get("gold_eval") or {}).get("decision") or {}).get("bbox_quality", "unknown") for task in accepted)),
        "caption_match_counts": dict(Counter(((task.get("gold_eval") or {}).get("decision") or {}).get("caption_match", "unknown") for task in accepted)),
    }
    print(
        json.dumps(
            {
                "split": split,
                "target": target_size,
                "reviewed": review_count,
                "accepted": len(accepted),
                "review_queue": len(review_queue),
                "rejected": len(rejected),
                "output": str(output_dir / short),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return result


def review_one(split: str, task: dict[str, Any], prior: dict[str, Any] | None, client: "ReviewClient", args: argparse.Namespace) -> dict[str, Any]:
    base = {
        "task_id": task.get("task_id"),
        "split": split,
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "caption_text": (task.get("gold") or {}).get("caption_text"),
        "caption_flags": caption_flags((task.get("gold") or {}).get("caption_text") or ""),
        "candidate_source": (task.get("candidate_meta") or {}).get("source"),
        "source_authority_level": (task.get("candidate_meta") or {}).get("source_authority_level"),
        "source_type": (task.get("candidate_meta") or {}).get("source_type"),
        "prior_vlm_decision": compact_prior(prior),
        "reviewed_at": now_cst(),
    }
    try:
        raw, model, input_mode = client.review(task, prior)
        decision = parse_json_object(raw)
        row = {
            **base,
            "ok": True,
            "review_model": model,
            "input_mode": input_mode,
            "decision": normalize_decision(decision),
            "raw_response": raw,
        }
        print(
            json.dumps(
                {
                    "split": split,
                    "task_id": task.get("task_id"),
                    "ok": True,
                    "status": row["decision"].get("status"),
                    "confidence": row["decision"].get("confidence"),
                    "model": model,
                    "input_mode": input_mode,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return row
    except Exception as exc:
        row = {
            **base,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "decision": {
                "status": "needs_human_review",
                "confidence": 0.0,
                "reason": "VLM 复核失败，需人工确认",
                "required_human_checks": ["vlm_review_failed"],
            },
        }
        print(json.dumps({"split": split, "task_id": task.get("task_id"), "ok": False, "error": row["error"]}, ensure_ascii=False), flush=True)
        return row


class ReviewClient:
    def review(self, task: dict[str, Any], prior: dict[str, Any] | None) -> tuple[str, str, str]:
        raise NotImplementedError


class OfflineReviewClient(ReviewClient):
    def review(self, task: dict[str, Any], prior: dict[str, Any] | None) -> tuple[str, str, str]:
        flags = caption_flags((task.get("gold") or {}).get("caption_text") or "")
        severe = any(flag in flags for flag in ["toc_or_index", "body_text", "too_short", "hyphen_end"])
        decision = {
            "status": "needs_human_review" if severe else "accepted_gold",
            "is_valid_target": True,
            "is_landscape_related": True,
            "bbox_quality": "unclear",
            "caption_match": "unclear",
            "caption_quality": "unclear" if severe else "minor_ocr_noise",
            "gold_claims_usable": not severe,
            "core_evidence_ok": not severe,
            "hard_negative_risk": severe,
            "confidence": 0.5 if severe else 0.62,
            "reason": "离线规则仅用于 smoke，不作为最终 GoldEval 采信依据",
            "required_human_checks": ["offline_review"],
            "suggested_caption_text": "",
        }
        return json.dumps(decision, ensure_ascii=False), "offline_rules", "offline"


class DashScopeReviewClient(ReviewClient):
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.models = dedupe([args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()])
        self.args = args

    def review(self, task: dict[str, Any], prior: dict[str, Any] | None) -> tuple[str, str, str]:
        last_error: Exception | None = None
        for model in self.models:
            for mode in image_modes(model):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=build_messages(task, prior, self.args, mode),
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or ""
                    parse_json_object(content)
                    return content, model, mode
                except Exception as exc:
                    last_error = exc
                    continue
        raise RuntimeError(f"all VLM models failed: {last_error!r}")


def make_client(args: argparse.Namespace) -> ReviewClient:
    if args.provider == "offline":
        return OfflineReviewClient()
    return DashScopeReviewClient(args)


def build_messages(task: dict[str, Any], prior: dict[str, Any] | None, args: argparse.Namespace, image_mode: str) -> list[dict[str, Any]]:
    task_info = build_task_info(task, prior)
    prompt = PROMPT + "\n当前样本元数据：\n" + json.dumps(task_info, ensure_ascii=False, indent=2)
    if image_mode == "image":
        content: Any = [
            {"type": "image", "image": image_data_url(task.get("overlay_image"), args.image_max_side)},
            {"type": "image", "image": image_data_url(task.get("artwork_image"), args.crop_max_side)},
            {"type": "text", "text": prompt},
        ]
    else:
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(task.get("overlay_image"), args.image_max_side)}},
            {"type": "image_url", "image_url": {"url": image_data_url(task.get("artwork_image"), args.crop_max_side)}},
            {"type": "text", "text": prompt},
        ]
    return [{"role": "user", "content": content}]


def build_task_info(task: dict[str, Any], prior: dict[str, Any] | None) -> dict[str, Any]:
    gold = task.get("gold") or {}
    claims = []
    for claim in gold.get("claims") or []:
        claims.append(
            {
                "field": claim.get("field"),
                "value": claim.get("value"),
                "abstain": claim.get("abstain"),
                "evidence_ids": claim.get("evidence_ids") or [],
                "support_type": claim.get("support_type"),
                "reason": claim.get("reason"),
            }
        )
    local_evidence = []
    for item in task.get("local_evidence") or []:
        local_evidence.append(
            {
                "evidence_id": item.get("evidence_id"),
                "source_file": item.get("source_file"),
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "citation_level": item.get("citation_level"),
                "display_snippet": item.get("display_snippet"),
                "bbox": item.get("bbox"),
            }
        )
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "source_type": task.get("source_type"),
        "candidate_meta": task.get("candidate_meta"),
        "image_bbox": gold.get("image_bbox"),
        "caption_bbox": gold.get("caption_bbox"),
        "caption_text": gold.get("caption_text"),
        "caption_flags": caption_flags(gold.get("caption_text") or ""),
        "target_claim_fields": gold.get("target_claim_fields"),
        "claims": claims,
        "local_evidence": local_evidence,
        "prior_vlm_review": compact_prior(prior),
    }


def image_data_url(path: Any, max_side: int) -> str:
    image_path = Path(str(path))
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        scale = min(1.0, float(max_side) / max(image.size))
        if scale < 1.0:
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            image.save(tmp_path, quality=88)
            data = base64.b64encode(tmp_path.read_bytes()).decode("ascii")
        finally:
            try:
                tmp_path.unlink()
            except FileNotFoundError:
                pass
    return f"data:image/jpeg;base64,{data}"


def image_modes(model: str) -> list[str]:
    lower = model.lower()
    if "qwen3.7" in lower or "qwen-max" in lower or "max" in lower:
        return ["image", "image_url"]
    return ["image_url", "image"]


def stratified_order(rows: list[dict[str, Any]], prior: dict[str, dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("source_file") or "unknown")].append(row)
    for source, bucket in buckets.items():
        rng.shuffle(bucket)
        bucket.sort(key=lambda row: candidate_sort_key(row, prior.get(str(row.get("task_id")))))
    ordered = []
    active_sources = sorted(buckets, key=lambda src: (-len(buckets[src]), src))
    while active_sources:
        next_sources = []
        for source in active_sources:
            bucket = buckets[source]
            if bucket:
                ordered.append(bucket.pop(0))
            if bucket:
                next_sources.append(source)
        active_sources = next_sources
    return ordered


def candidate_sort_key(row: dict[str, Any], prior: dict[str, Any] | None) -> tuple[float, int, str]:
    decision = (prior or {}).get("decision") or {}
    prior_keep = decision.get("should_keep_for_training")
    prior_caption = decision.get("caption_match")
    prior_conf = safe_float(decision.get("confidence"), 0.0)
    prior_score = 0.0
    if prior_keep is True:
        prior_score -= 4.0 + prior_conf
    elif prior_keep is False:
        prior_score += 4.0 + prior_conf
    if prior_caption == "good":
        prior_score -= 1.0
    elif prior_caption in {"wrong", "missing"}:
        prior_score += 1.0

    flags = caption_flags((row.get("gold") or {}).get("caption_text") or "")
    risk = len(flags)
    severe = {"toc_or_index", "body_text", "too_short", "hyphen_end"}
    risk += 2 * sum(1 for flag in flags if flag in severe)
    meta = row.get("candidate_meta") or {}
    authority = str(meta.get("source_authority_level") or "")
    if authority == "A":
        risk -= 1
    elif authority == "A-":
        risk -= 0.5
    return (prior_score + risk, len(flags), str(row.get("task_id") or ""))


def caption_flags(text: str) -> list[str]:
    s = str(text or "").strip()
    flags = []
    if not s:
        return ["missing"]
    if len(s) < 8:
        flags.append("too_short")
    if s.endswith("-") or s.endswith("—"):
        flags.append("hyphen_end")
    if re.search(r"\.{5,}|…{2,}|\b\d+\.\d+\b", s):
        flags.append("toc_or_index")
    if any(token in s for token in ["本章小结", "不足与展望", "总 结", "总结", "目录"]):
        flags.append("toc_or_index")
    if re.match(r"^[a-z0-9][a-z]{2,}", s):
        flags.append("body_text")
    if len(s) > 20 and not re.search(r"(fig\.?|figure|plate|图|圖|《|landscape|山水|painting|scroll|album)", s, flags=re.I):
        if s[:1].islower() or re.search(r"\b(the|and|or|but|when|while|with|from|this|that)\b", s[:80], flags=re.I):
            flags.append("body_text")
    alpha = len(re.findall(r"[A-Za-z\u4e00-\u9fff]", s))
    if len(s) >= 12 and alpha / max(1, len(s)) < 0.45:
        flags.append("low_alpha_ratio")
    return dedupe(flags)


def classify_decision(row: dict[str, Any], min_confidence: float) -> str:
    if not row.get("ok"):
        return "needs_human_review"
    decision = row.get("decision") or {}
    status = str(decision.get("status") or "needs_human_review")
    confidence = safe_float(decision.get("confidence"), 0.0)
    if status == "accepted_gold":
        if confidence < min_confidence:
            return "needs_human_review"
        if decision.get("is_valid_target") is not True or decision.get("is_landscape_related") is not True:
            return "rejected"
        if decision.get("hard_negative_risk") is True:
            return "needs_human_review"
        if decision.get("gold_claims_usable") is False or decision.get("core_evidence_ok") is False:
            return "needs_human_review"
        if decision.get("bbox_quality") not in {"good", "partial"}:
            return "needs_human_review"
        if decision.get("caption_match") not in {"good", "partial"}:
            return "needs_human_review"
        if decision.get("caption_quality") in {"body_text", "toc_or_index", "too_short", "wrong_language"}:
            return "needs_human_review"
        return "accepted_gold"
    if status == "rejected":
        return "rejected"
    return "needs_human_review"


def annotate_gold_task(task: dict[str, Any], decision_row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = copy.deepcopy(task)
    gold = out.setdefault("gold", {})
    gold_eval = {
        "version": args.version,
        "dataset": "GoldEval",
        "status": "accepted_gold",
        "review_type": "dashscope_vlm" if args.provider == "dashscope" else "offline_rule_smoke",
        "review_model": decision_row.get("review_model"),
        "input_mode": decision_row.get("input_mode"),
        "reviewed_at": decision_row.get("reviewed_at") or now_cst(),
        "min_confidence": args.min_confidence,
        "decision": decision_row.get("decision") or {},
        "caption_flags": decision_row.get("caption_flags") or [],
        "original_gold_auto_label": gold.get("auto_label"),
        "original_gold_needs_review": gold.get("needs_review"),
        "original_label_source": gold.get("label_source"),
        "limitations": [
            "这是 VLM 复核的小规模 GoldEval；若用于论文级结论，仍建议再做人工抽检。",
            "复核主要确认目标/图注/当前 Core5 claims 是否可评测，不做外部知识事实重标。",
        ],
    }
    out["gold_eval"] = gold_eval
    out["gold_eval_status"] = "accepted_gold"
    gold["auto_label"] = False
    gold["needs_review"] = False
    gold["label_source"] = "v1_0_4_gold_eval_vlm_reviewed_from_v1_0_3_silver"
    gold["gold_eval_reviewed"] = True
    gold["gold_eval_review_model"] = decision_row.get("review_model")
    gold["gold_eval_reviewed_at"] = gold_eval["reviewed_at"]
    return out


def make_queue_row(task: dict[str, Any], decision_row: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "caption_text": (task.get("gold") or {}).get("caption_text"),
        "caption_flags": caption_flags((task.get("gold") or {}).get("caption_text") or ""),
        "overlay_image": task.get("overlay_image"),
        "artwork_image": task.get("artwork_image"),
        "status": status,
        "decision": decision_row.get("decision") or {},
        "error": decision_row.get("error"),
        "review_model": decision_row.get("review_model"),
        "input_mode": decision_row.get("input_mode"),
    }


def normalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    result = dict(decision)
    result["status"] = normalize_enum(result.get("status"), {"accepted_gold", "needs_human_review", "rejected"}, "needs_human_review")
    result["bbox_quality"] = normalize_enum(result.get("bbox_quality"), {"good", "partial", "too_large", "too_small", "text_region", "non_image", "unclear"}, "unclear")
    result["caption_match"] = normalize_enum(result.get("caption_match"), {"good", "partial", "wrong", "missing", "unclear"}, "unclear")
    result["caption_quality"] = normalize_enum(
        result.get("caption_quality"),
        {"clean", "minor_ocr_noise", "truncated", "body_text", "toc_or_index", "too_short", "wrong_language", "unclear"},
        "unclear",
    )
    for key in ["is_valid_target", "is_landscape_related", "gold_claims_usable", "core_evidence_ok", "hard_negative_risk"]:
        if not isinstance(result.get(key), bool):
            result[key] = None if result.get(key) is None else bool(result.get(key))
    result["confidence"] = max(0.0, min(1.0, safe_float(result.get("confidence"), 0.0)))
    result["reason"] = str(result.get("reason") or "")[:500]
    checks = result.get("required_human_checks")
    if not isinstance(checks, list):
        checks = []
    result["required_human_checks"] = [str(item)[:120] for item in checks if str(item).strip()][:8]
    result["suggested_caption_text"] = str(result.get("suggested_caption_text") or "")[:300]
    return result


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def parse_json_object(text: str) -> dict[str, Any]:
    text = str(text or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("response is not a JSON object")
    return data


def build_summary(
    args: argparse.Namespace,
    dataset_dir: Path,
    output_dir: Path,
    val_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    split_results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    compact_results = {}
    for split, result in split_results.items():
        compact_results[split] = {key: value for key, value in result.items() if key not in {"accepted_tasks", "review_queue", "rejected"}}
    return {
        "created_at": now_cst(),
        "version": args.version,
        "builder": "scripts/build_gold_eval_v1_0_4.py",
        "source_dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "provider": args.provider,
        "model": args.model if args.provider == "dashscope" else "offline_rules",
        "fallback_models": [item.strip() for item in args.fallback_models.split(",") if item.strip()],
        "seed": args.seed,
        "source_split_rows": {"val": len(val_rows), "test": len(test_rows)},
        "target_sizes": {"val_gold": args.val_size, "test_gold": args.test_size},
        "min_confidence": args.min_confidence,
        "splits": compact_results,
        "outputs": {
            "val_gold_50": str(output_dir / f"val_gold_{args.val_size}.jsonl"),
            "test_gold_100": str(output_dir / f"test_gold_{args.test_size}.jsonl"),
            "review_dir": str(output_dir / "review"),
            "report": str(output_dir / "构建报告.md"),
        },
        "limitations": [
            "GoldEval 是 VLM 复核集，不等同于双人标注仲裁集。",
            "本轮只确认现有 Core5 silver label 是否可用于评测，不从 PDF 或外部知识库重建完整事实标签。",
            "accepted_gold 样本会将 gold.auto_label 置为 false、needs_review 置为 false，但 provenance 保留在 gold_eval 字段。",
        ],
    }


def build_manifest(args: argparse.Namespace, dataset_dir: Path, output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "build_time": summary["created_at"],
        "version": args.version,
        "builder": "scripts/build_gold_eval_v1_0_4.py",
        "args": vars(args),
        "source_dataset_dir": str(dataset_dir),
        "outputs": summary["outputs"],
        "summary": summary,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 GoldEval 构建报告",
        "",
        f"生成时间：{summary['created_at']}",
        "",
        "## 目标",
        "",
        "本轮从现有 v1.0.3 no-select 的 val/test task 中构建小规模复核评测集：`val_gold_50` 和 `test_gold_100`。它们用于替代当前启发式 silver val/test 做更可信的阶段性评测。",
        "",
        "## 输入与输出",
        "",
        f"- 输入数据：`{summary['source_dataset_dir']}`",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- val_gold：`{summary['outputs']['val_gold_50']}`",
        f"- test_gold：`{summary['outputs']['test_gold_100']}`",
        f"- review queue：`{summary['outputs']['review_dir']}`",
        f"- 复核方式：`{summary['provider']}`，主模型 `{summary['model']}`",
        f"- 最低 accepted 置信度：{summary['min_confidence']}",
        "",
        "## 构建结果",
        "",
    ]
    for split, result in summary["splits"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- 来源池：{result['pool_rows']} 条",
                f"- 实际复核：{result['reviewed_rows']} 条",
                f"- accepted_gold：{result['accepted_rows']} 条",
                f"- needs_human_review 队列：{result['review_queue_rows']} 条",
                f"- rejected：{result['rejected_rows']} 条",
                f"- source 分布：`{json.dumps(result['source_counts'], ensure_ascii=False)}`",
                f"- authority 分布：`{json.dumps(result['authority_counts'], ensure_ascii=False)}`",
                f"- candidate_source 分布：`{json.dumps(result['candidate_source_counts'], ensure_ascii=False)}`",
                f"- caption_match 分布：`{json.dumps(result['caption_match_counts'], ensure_ascii=False)}`",
                f"- caption_quality 分布：`{json.dumps(result['caption_quality_counts'], ensure_ascii=False)}`",
                f"- bbox_quality 分布：`{json.dumps(result['bbox_quality_counts'], ensure_ascii=False)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 使用建议",
            "",
            "1. 后续模型评测优先在 `val_gold_50` 上做快速回归，在关键修复后再跑 `test_gold_100`。",
            "2. 当前大 val/test 仍保留为工程回归集，不应再被称为最终 gold 指标。",
            "3. 若要把结果写进论文或正式报告，应从 review queue 中抽样做人工二次核验，尤其检查 `partial` caption 和局部图像样本。",
            "",
            "## 局限",
            "",
        ]
    )
    for item in summary["limitations"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_html(path: Path, split_results: dict[str, dict[str, Any]]) -> None:
    cards = []
    for split, result in split_results.items():
        for task in result.get("accepted_tasks", [])[:80]:
            cards.append(render_card(split, task, "accepted_gold"))
        for row in result.get("review_queue", [])[:80]:
            cards.append(render_queue_card(split, row))
    body = "\n".join(cards)
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>v1.0.4 GoldEval Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f5; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    img {{ width: 100%; height: auto; border: 1px solid #ccc; }}
    .meta {{ font-size: 13px; line-height: 1.45; }}
    .status {{ font-weight: 700; }}
  </style>
</head>
<body>
  <h1>v1.0.4 GoldEval Review</h1>
  <div class="grid">
  {body}
  </div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def render_card(split: str, task: dict[str, Any], status: str) -> str:
    decision = ((task.get("gold_eval") or {}).get("decision") or {})
    caption = (task.get("gold") or {}).get("caption_text") or ""
    return f"""
<div class="card">
  <div class="meta"><span class="status">{html.escape(status)}</span> | {html.escape(split)} | {html.escape(str(task.get('task_id')))} | p.{html.escape(str(task.get('page')))}</div>
  <div class="meta">{html.escape(str(task.get('source_file')))}</div>
  <div class="imgs"><img src="{html.escape(str(task.get('overlay_image')))}"><img src="{html.escape(str(task.get('artwork_image')))}"></div>
  <div class="meta"><b>caption</b>: {html.escape(caption)}</div>
  <div class="meta"><b>decision</b>: {html.escape(json.dumps(decision, ensure_ascii=False))}</div>
</div>
"""


def render_queue_card(split: str, row: dict[str, Any]) -> str:
    return f"""
<div class="card">
  <div class="meta"><span class="status">{html.escape(str(row.get('status')))}</span> | {html.escape(split)} | {html.escape(str(row.get('task_id')))} | p.{html.escape(str(row.get('page')))}</div>
  <div class="meta">{html.escape(str(row.get('source_file')))}</div>
  <div class="imgs"><img src="{html.escape(str(row.get('overlay_image')))}"><img src="{html.escape(str(row.get('artwork_image')))}"></div>
  <div class="meta"><b>caption</b>: {html.escape(str(row.get('caption_text') or ''))}</div>
  <div class="meta"><b>decision</b>: {html.escape(json.dumps(row.get('decision') or {}, ensure_ascii=False))}</div>
</div>
"""


def load_prior_reviews(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    paths = list(dataset_dir.glob("vlm_adjudication*/vlm_adjudication.jsonl"))
    result = {}
    for path in paths:
        for row in read_jsonl(path):
            task_id = row.get("task_id")
            if task_id:
                result[str(task_id)] = row
    return result


def compact_prior(prior: dict[str, Any] | None) -> dict[str, Any] | None:
    if not prior:
        return None
    decision = prior.get("decision") or {}
    return {
        "model": prior.get("model"),
        "input_mode": prior.get("input_mode"),
        "is_valid_target": decision.get("is_valid_target"),
        "is_landscape_related": decision.get("is_landscape_related"),
        "bbox_quality": decision.get("bbox_quality"),
        "caption_match": decision.get("caption_match"),
        "should_keep_for_training": decision.get("should_keep_for_training"),
        "confidence": decision.get("confidence"),
        "reason": decision.get("reason"),
    }


def load_existing_stream(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_jsonl(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
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


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def remove_tree(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir():
            remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def dedupe(values: Iterable[Any]) -> list[Any]:
    result = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
