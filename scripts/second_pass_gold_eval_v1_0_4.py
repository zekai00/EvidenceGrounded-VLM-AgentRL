#!/usr/bin/env python3
"""Second-pass final adjudication for v1.0.4 GoldEval review queues."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_gold_eval_v1_0_4 import (  # noqa: E402
    DEFAULT_DOTENV,
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MODEL,
    annotate_gold_task,
    dedupe,
    image_data_url,
    load_dotenv,
    normalize_decision,
    parse_json_object,
    read_jsonl,
    safe_float,
    write_json,
    write_jsonl,
)


FINAL_PROMPT = """你是 EvidenceGrounded-VLM-AgentRL 的 GoldEval 二次终审员。你会看到两张图：
1. PDF 页面 overlay：红框是候选目标图像区域，青色/蓝色框是候选图注或文本区域。
2. 红框裁剪图：候选目标图像。

这条样本一审被标记为 needs_human_review。现在你必须做最终二选一裁决：
- accepted_gold：可以进入小规模评测 GoldEval。
- rejected：不能进入 GoldEval。

只输出 JSON 对象，不要输出 Markdown。字段必须为：
{
  "status": "accepted_gold|rejected",
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

终审标准：
- 只要红框目标、裁剪图、图注框和当前 Core5 claims 在内部是一致的，允许图注有轻微 OCR 噪声或轻微截断；不要求补全全部作者、年代、尺寸等外部事实。
- 如果当前 gold claims 没有声称未被支持的事实，且主要是 caption_text/object_type 或合理 abstain，可以 accepted_gold。
- 如果图注框其实是正文、目录、另一张图的图注，或者目标不是山水画相关图像，必须 rejected。
- 如果你仍然觉得需要人工确认，默认 rejected；不要输出 needs_human_review。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final second-pass adjudication for a GoldEval split.")
    parser.add_argument("--gold-eval-dir", required=True)
    parser.add_argument("--dataset-dir", default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--target-size", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--dotenv", default=DEFAULT_DOTENV)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--image-max-side", type=int, default=1200)
    parser.add_argument("--crop-max-side", type=int, default=768)
    parser.add_argument("--min-confidence", type=float, default=0.78)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--max-second-pass", type=int, default=52)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(Path(args.dotenv))
    gold_dir = Path(args.gold_eval_dir)
    dataset_dir = Path(args.dataset_dir)
    gold_path = gold_dir / f"{args.split}_gold_{args.target_size}.jsonl"
    current_gold = read_jsonl(gold_path)
    if len(current_gold) >= args.target_size:
        print(json.dumps({"status": "already_full", "rows": len(current_gold), "path": str(gold_path)}, ensure_ascii=False), flush=True)
        return 0

    tasks = {row.get("task_id"): row for row in read_jsonl(dataset_dir / f"{args.split}_tasks.jsonl")}
    queue = read_jsonl(gold_dir / "review" / f"{args.split}_review_queue.jsonl")
    stream_path = gold_dir / "review" / f"{args.split}_second_pass_stream.jsonl"
    completed = {row.get("task_id"): row for row in read_jsonl(stream_path)} if args.resume else {}
    client = DashScopeFinalClient(args)

    accepted_additions: list[dict[str, Any]] = []
    second_pass_rows: list[dict[str, Any]] = list(completed.values())
    current_ids = {row.get("task_id") for row in current_gold}
    needed = args.target_size - len(current_gold)
    candidates = [row for row in sorted(queue, key=second_pass_sort_key) if row.get("task_id") not in current_ids]

    for queue_row in candidates:
        if len(accepted_additions) >= needed:
            break
        if len(second_pass_rows) >= args.max_second_pass:
            break
        task_id = queue_row.get("task_id")
        task = tasks.get(task_id)
        if not task:
            continue
        row = completed.get(task_id)
        if row is None:
            row = review_final(args, client, task, queue_row)
            append_rows(stream_path, [row])
            second_pass_rows.append(row)
            if args.sleep:
                time.sleep(args.sleep)
        if is_second_pass_accept(row, args.min_confidence):
            annotated = annotate_gold_task(task, row, SimpleNamespace(version="v1.0.4", provider="dashscope", min_confidence=args.min_confidence))
            annotated["gold_eval"]["review_type"] = "dashscope_vlm_second_pass"
            annotated["gold_eval"]["first_pass_decision"] = queue_row.get("decision") or {}
            annotated["gold_eval"]["second_pass_decision"] = row.get("decision") or {}
            annotated["gold_eval"]["status"] = "accepted_gold"
            annotated["gold_eval_status"] = "accepted_gold"
            accepted_additions.append(annotated)

    final_gold = current_gold + accepted_additions
    final_gold = final_gold[: args.target_size]
    write_jsonl(gold_path, final_gold)
    write_jsonl(gold_dir / "review" / f"{args.split}_second_pass_accepted_additions.jsonl", accepted_additions)
    update_summary_and_report(gold_dir, args, current_gold, accepted_additions, final_gold, second_pass_rows)
    result = {
        "status": "done",
        "split": args.split,
        "target_size": args.target_size,
        "strict_first_pass_rows": len(current_gold),
        "second_pass_reviewed_rows": len(second_pass_rows),
        "second_pass_accepted_additions": len(accepted_additions),
        "final_rows": len(final_gold),
        "gold_path": str(gold_path),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0


class DashScopeFinalClient:
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.models = dedupe([args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()])
        self.args = args

    def infer(self, task: dict[str, Any], first_pass: dict[str, Any]) -> tuple[str, str, str]:
        last_error: Exception | None = None
        for model in self.models:
            for mode in image_modes(model):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=build_messages(task, first_pass, self.args, mode),
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


def build_messages(task: dict[str, Any], first_pass: dict[str, Any], args: argparse.Namespace, image_mode: str) -> list[dict[str, Any]]:
    gold = task.get("gold") or {}
    claims = [
        {
            "field": claim.get("field"),
            "value": claim.get("value"),
            "abstain": claim.get("abstain"),
            "evidence_ids": claim.get("evidence_ids") or [],
            "support_type": claim.get("support_type"),
            "reason": claim.get("reason"),
        }
        for claim in gold.get("claims") or []
    ]
    info = {
        "task_id": task.get("task_id"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "image_bbox": gold.get("image_bbox"),
        "caption_bbox": gold.get("caption_bbox"),
        "caption_text": gold.get("caption_text"),
        "claims": claims,
        "local_evidence": task.get("local_evidence") or [],
        "first_pass_decision": first_pass.get("decision") or {},
    }
    prompt = FINAL_PROMPT + "\n当前样本与一审结果：\n" + json.dumps(info, ensure_ascii=False, indent=2)
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


def review_final(args: argparse.Namespace, client: DashScopeFinalClient, task: dict[str, Any], first_pass: dict[str, Any]) -> dict[str, Any]:
    try:
        raw, model, mode = client.infer(task, first_pass)
        decision = normalize_decision(parse_json_object(raw))
        if decision.get("status") == "needs_human_review":
            decision["status"] = "rejected"
            decision["reason"] = (decision.get("reason") or "") + "；二审不允许继续 defer，因此按 rejected 处理。"
        row = {
            "task_id": task.get("task_id"),
            "split": args.split,
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "caption_text": (task.get("gold") or {}).get("caption_text"),
            "ok": True,
            "review_model": model,
            "input_mode": mode,
            "decision": decision,
            "first_pass_decision": first_pass.get("decision") or {},
            "raw_response": raw,
        }
        print(
            json.dumps(
                {
                    "split": args.split,
                    "task_id": task.get("task_id"),
                    "second_pass": True,
                    "status": decision.get("status"),
                    "confidence": decision.get("confidence"),
                    "model": model,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return row
    except Exception as exc:
        row = {
            "task_id": task.get("task_id"),
            "split": args.split,
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "caption_text": (task.get("gold") or {}).get("caption_text"),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "decision": {"status": "rejected", "confidence": 0.0, "reason": "二审调用失败"},
            "first_pass_decision": first_pass.get("decision") or {},
        }
        print(json.dumps({"split": args.split, "task_id": task.get("task_id"), "second_pass": True, "ok": False, "error": row["error"]}, ensure_ascii=False), flush=True)
        return row


def second_pass_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    decision = row.get("decision") or {}
    score = 0.0
    if decision.get("is_valid_target") is True:
        score -= 2.0
    if decision.get("is_landscape_related") is True:
        score -= 2.0
    if decision.get("bbox_quality") == "good":
        score -= 2.0
    elif decision.get("bbox_quality") == "partial":
        score -= 1.0
    if decision.get("caption_match") == "good":
        score -= 2.0
    elif decision.get("caption_match") == "partial":
        score -= 1.0
    if decision.get("caption_quality") == "clean":
        score -= 1.5
    elif decision.get("caption_quality") == "minor_ocr_noise":
        score -= 1.0
    elif decision.get("caption_quality") == "truncated":
        score -= 0.4
    if decision.get("gold_claims_usable") is True:
        score -= 1.0
    if decision.get("core_evidence_ok") is True:
        score -= 1.0
    if decision.get("hard_negative_risk") is True:
        score += 3.0
    score -= safe_float(decision.get("confidence"), 0.0)
    return (score, str(row.get("task_id") or ""))


def is_second_pass_accept(row: dict[str, Any], min_confidence: float) -> bool:
    if not row.get("ok"):
        return False
    decision = row.get("decision") or {}
    return (
        decision.get("status") == "accepted_gold"
        and safe_float(decision.get("confidence"), 0.0) >= min_confidence
        and decision.get("is_valid_target") is True
        and decision.get("is_landscape_related") is True
        and decision.get("bbox_quality") in {"good", "partial"}
        and decision.get("caption_match") in {"good", "partial"}
        and decision.get("caption_quality") not in {"body_text", "toc_or_index", "too_short", "wrong_language"}
        and decision.get("gold_claims_usable") is True
        and decision.get("core_evidence_ok") is True
        and decision.get("hard_negative_risk") is not True
    )


def image_modes(model: str) -> list[str]:
    lower = model.lower()
    if "qwen3.7" in lower or "qwen-max" in lower or "max" in lower:
        return ["image", "image_url"]
    return ["image_url", "image"]


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def update_summary_and_report(
    gold_dir: Path,
    args: argparse.Namespace,
    first_pass_gold: list[dict[str, Any]],
    accepted_additions: list[dict[str, Any]],
    final_gold: list[dict[str, Any]],
    second_pass_rows: list[dict[str, Any]],
) -> None:
    summary_path = gold_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    split_summary = (summary.get("splits") or {}).get(args.split, {})
    split_summary["strict_first_pass_accepted_rows"] = len(first_pass_gold)
    split_summary["second_pass_reviewed_rows"] = len(second_pass_rows)
    split_summary["second_pass_accepted_additions"] = len(accepted_additions)
    split_summary["accepted_rows"] = len(final_gold)
    split_summary["second_pass_model_counts"] = dict(Counter(row.get("review_model") for row in second_pass_rows))
    split_summary["source_counts"] = dict(Counter(row.get("source_file") for row in final_gold))
    split_summary["authority_counts"] = dict(Counter(((row.get("candidate_meta") or {}).get("source_authority_level") or "unknown") for row in final_gold))
    split_summary["candidate_source_counts"] = dict(Counter(((row.get("candidate_meta") or {}).get("source") or "unknown") for row in final_gold))
    split_summary["review_model_counts"] = dict(Counter(((row.get("gold_eval") or {}).get("review_model") or "unknown") for row in final_gold))
    split_summary["review_type_counts"] = dict(Counter(((row.get("gold_eval") or {}).get("review_type") or "unknown") for row in final_gold))
    split_summary["caption_match_counts"] = dict(Counter((((row.get("gold_eval") or {}).get("decision") or {}).get("caption_match") or "unknown") for row in final_gold))
    split_summary["caption_quality_counts"] = dict(Counter((((row.get("gold_eval") or {}).get("decision") or {}).get("caption_quality") or "unknown") for row in final_gold))
    split_summary["bbox_quality_counts"] = dict(Counter((((row.get("gold_eval") or {}).get("decision") or {}).get("bbox_quality") or "unknown") for row in final_gold))
    summary.setdefault("splits", {})[args.split] = split_summary
    summary.setdefault("outputs", {})[f"{args.split}_gold_{args.target_size}"] = str(gold_dir / f"{args.split}_gold_{args.target_size}.jsonl")
    write_json(summary_path, summary)

    supplement = [
        "",
        "## 二次终审补充",
        "",
        f"- split：`{args.split}`",
        f"- 一审 accepted：{len(first_pass_gold)}",
        f"- 二审复核：{len(second_pass_rows)}",
        f"- 二审补入 accepted：{len(accepted_additions)}",
        f"- 最终 gold 行数：{len(final_gold)}",
        f"- 二审 accepted 记录：`{gold_dir / 'review' / f'{args.split}_second_pass_accepted_additions.jsonl'}`",
        f"- 二审 stream：`{gold_dir / 'review' / f'{args.split}_second_pass_stream.jsonl'}`",
        "",
        "说明：二审只处理一审 `needs_human_review`，且要求最终二选一；`rejected` 样本不参与补入。",
        "",
    ]
    report_path = gold_dir / "构建报告.md"
    with report_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(supplement))


if __name__ == "__main__":
    raise SystemExit(main())
