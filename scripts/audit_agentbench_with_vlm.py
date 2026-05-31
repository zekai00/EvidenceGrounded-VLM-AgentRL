#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_local_evidence_agentbench import (  # noqa: E402
    build_oracle_episode,
    episode_to_sft_samples,
    html_escape,
    task_to_claim_rows,
    task_to_evidence_rows,
    write_jsonl,
)


FIELDS = ["caption_text", "title", "artist", "dynasty", "visual_elements", "technique", "composition"]
VISUAL_FIELDS = {"caption_text", "visual_elements"}
TEXT_NEEDS_CHUNK = {"title", "artist", "dynasty", "technique", "composition"}

SYSTEM_PROMPT = """你是证据约束的多模态主动取证数据审核员。你会看到：
1. PDF 页面 overlay 图：红框是候选山水画图像区域，蓝框是候选图注区域。
2. 红框裁剪图：候选作品/局部图像。
3. 若干候选 evidence chunk：来自本地文献证据库。

你的任务是把规则生成的 silver label 修正成可训练的 VLM agent 数据。
必须只输出 JSON 对象，不要输出 Markdown。

判断原则：
- 统计图、表格、目录、封面、纯文字截图、公式、流程图、非山水画图片，应标记 is_relevant_artwork=false。
- 作品、山水画局部、构图/皴法/点景建筑/桥梁/留白等山水画相关图像，可标记 true。
- 对 title/artist/dynasty/technique/composition 等事实字段，只有当页面图注或候选 chunk 明确支持时才写 claim；证据不足时 abstain=true。
- visual_elements 可以依据红框裁剪图直接判断，但不要把不清楚的细节写进去。
- evidence_ids 只能从给定 candidate_chunks 里选择 chunk_id；如果没有真正支持该字段的 chunk，则留空。
- JSON 字符串内部不要使用英文双引号；如需引用作品名请用书名号《》或中文引号。

输出 JSON schema：
{
  "is_relevant_artwork": true,
  "caption_text": "修正后的图注，没有则空字符串",
  "title": "作品名，没有则空字符串",
  "artist": "画家，没有则空字符串",
  "dynasty": "朝代，没有则空字符串",
  "visual_elements": ["山","水"],
  "technique": ["水墨"],
  "composition": ["留白"],
  "claims": [
    {"field":"caption_text","value":"...","abstain":false,"evidence_ids":[],"confidence":0.8,"reason":"来自蓝框图注"},
    {"field":"artist","value":"吴冠中","abstain":false,"evidence_ids":["chunk_xxx"],"confidence":0.8,"reason":"chunk 明确提到"},
    {"field":"dynasty","value":null,"abstain":true,"evidence_ids":[],"confidence":0.7,"reason":"图注和 chunk 均不足"}
  ],
  "confidence": 0.0,
  "quality_flags": ["可选问题标记"],
  "notes": "一句话说明主要修正依据"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit EvidenceGrounded-AgentBench tasks with DashScope VLM.")
    parser.add_argument("--tasks", required=True, help="Input tasks_all.jsonl from the deterministic build.")
    parser.add_argument("--provider", choices=["dashscope", "local"], default="dashscope")
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="", help="Exact output directory. Useful for resuming an interrupted audit.")
    parser.add_argument("--version", default="agentbench_v0_1_vlm_audited")
    parser.add_argument("--model", default="qwen3.7-max-2026-05-20")
    parser.add_argument("--local-model", default="/root/models/Qwen3-VL-4B-Instruct")
    parser.add_argument(
        "--fallback-models",
        default="qwen3.7-max-2026-05-17,qwen3.7-max,qwen3.6-35b-a3b,kimi-k2.6,qwen3.6-plus-2026-04-02,qwen3.6-plus,qwen3.6-flash-2026-04-16",
    )
    parser.add_argument(
        "--dashscope-image-format",
        choices=["auto", "image_url", "image", "text_only"],
        default="auto",
        help="DashScope message format for image input. auto uses image for qwen3.7/max-style models and image_url for Qwen3.6/Qwen-VL-style models.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--json-mode", action="store_true", default=True)
    parser.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--image-max-side", type=int, default=1200)
    parser.add_argument("--crop-max-side", type=int, default=768)
    parser.add_argument("--chunk-snippet-chars", type=int, default=360)
    parser.add_argument("--drop-irrelevant", action="store_true", default=True)
    parser.add_argument("--keep-irrelevant", dest="drop_irrelevant", action="store_false")
    parser.add_argument("--drop-audit-errors", action="store_true", default=True)
    parser.add_argument("--keep-audit-errors", dest="drop_audit_errors", action="store_false")
    parser.add_argument("--rebalance-splits", action="store_true", default=True)
    parser.add_argument("--no-rebalance-splits", dest="rebalance_splits", action="store_false")
    parser.add_argument("--dotenv", default="/root/Workspace/VLM/EviTool-VL/.env")
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true", help="When resuming, treat previous vlm_audit_error rows as pending again.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks_path = Path(args.tasks)
    rows = read_jsonl(tasks_path)
    if args.limit:
        rows = rows[: args.limit]

    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir) if args.output_dir else output_root / f"{args.version}_{now}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory already exists and is not empty. Use --resume or --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ["audit", "episodes", "sft", "review"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    load_dotenv(args.dotenv)
    chunk_map = load_evidence_chunks(rows)
    if args.provider == "dashscope":
        model_order = [args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()]
        model_order = dedupe_keep_order(model_order)
        client = DashScopeVLMClient(model_order, args)
    else:
        client = LocalQwenVLClient(args)

    stream_path = output_dir / "audit" / "audited_stream.jsonl"
    completed = load_completed_stream(stream_path, rows) if args.resume else []
    if not args.resume and stream_path.exists():
        stream_path.unlink()
    audited = audit_rows(rows, chunk_map, client, args, stream_path, completed)
    filtered_tasks = [row for row in audited if not (args.drop_audit_errors and row.get("vlm_audit_error"))]
    if args.drop_irrelevant:
        final_tasks = [row for row in filtered_tasks if not is_marked_irrelevant(row)]
    else:
        final_tasks = filtered_tasks
    if args.rebalance_splits:
        rebalance_splits_by_source(final_tasks)

    episodes = []
    sft_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in final_tasks:
        episode = build_oracle_episode(task)
        episodes.append(episode)
        for sample in episode_to_sft_samples(episode, task):
            sample["label_source"] = "vlm_audited"
            sft_rows_by_split[task["split"]].append(sample)

    claim_rows = [row for task in final_tasks for row in task_to_claim_rows(task)]
    evidence_rows = [row for task in final_tasks for row in task_to_evidence_rows(task)]
    errors = [row.get("vlm_audit_error") for row in audited if row.get("vlm_audit_error")]

    write_jsonl(output_dir / "audit" / "audited_all_including_irrelevant.jsonl", strip_audit_indexes(audited))
    write_jsonl(output_dir / "tasks_all.jsonl", strip_audit_indexes(final_tasks))
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", strip_audit_indexes([task for task in final_tasks if task["split"] == split]))
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows_by_split.get(split, []))
    write_jsonl(output_dir / "claim_gold.jsonl", claim_rows)
    write_jsonl(output_dir / "evidence_links.jsonl", evidence_rows)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)

    review_path = write_review_html(output_dir / "review" / "review.html", final_tasks[:200])
    summary = build_summary(args, tasks_path, output_dir, audited, final_tasks, claim_rows, sft_rows_by_split, errors, review_path)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(build_manifest(args, tasks_path, output_dir, summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "VLM审核构建报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def audit_rows(
    rows: list[dict[str, Any]],
    chunk_map: dict[str, dict[str, Any]],
    client: "DashScopeVLMClient",
    args: argparse.Namespace,
    stream_path: Path,
    completed: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any] | None] = [None] * len(rows)
    completed_ids = set()
    index_by_id = task_index_by_id(rows)
    for item in completed:
        task_id = item.get("task_id")
        if task_id is None:
            continue
        if args.retry_failed and item.get("vlm_audit_error"):
            continue
        index = int(item.get("_audit_index", index_by_id.get(task_id, -1)))
        if index >= 0 and index < len(results):
            results[index] = item
            completed_ids.add(task_id)
    pending = [(index, row) for index, row in enumerate(rows) if row.get("task_id") not in completed_ids]
    print(json.dumps({"resume_completed": len(completed_ids), "pending": len(pending), "stream": str(stream_path)}, ensure_ascii=False), flush=True)
    stream_lock = threading.Lock()
    if args.workers <= 1:
        for index, row in pending:
            results[index] = audit_one(index, row, chunk_map, client, args)
            append_stream_row(stream_path, results[index], stream_lock)
            if args.sleep:
                time.sleep(args.sleep)
        return [row for row in results if row is not None]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(audit_one, index, row, chunk_map, client, args): index for index, row in pending}
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                fallback = clone_row(rows[index])
                fallback["vlm_audit_error"] = f"{type(exc).__name__}: {exc}"
                fallback.setdefault("quality_flags", []).append("vlm_audit_exception")
                fallback["_audit_index"] = index
                results[index] = fallback
            append_stream_row(stream_path, results[index], stream_lock)
    return [row for row in results if row is not None]


def audit_one(index: int, row: dict[str, Any], chunk_map: dict[str, dict[str, Any]], client: "DashScopeVLMClient", args: argparse.Namespace) -> dict[str, Any]:
    try:
        raw, model, input_mode = client.infer(row, chunk_map)
        parsed = parse_json_object(raw)
        repaired = apply_vlm_audit(row, parsed, raw, model, input_mode, chunk_map)
        repaired["_audit_index"] = index
        print(json.dumps({"index": index, "task_id": row.get("task_id"), "ok": True, "model": model}, ensure_ascii=False), flush=True)
        return repaired
    except Exception as exc:
        fallback = clone_row(row)
        fallback["vlm_audit_error"] = f"{type(exc).__name__}: {exc}"
        fallback.setdefault("quality_flags", []).append("vlm_audit_failed")
        fallback.setdefault("gold", {})["vlm_audited"] = False
        fallback["_audit_index"] = index
        print(json.dumps({"index": index, "task_id": row.get("task_id"), "ok": False, "error": repr(exc)}, ensure_ascii=False), flush=True)
        return fallback


class DashScopeVLMClient:
    def __init__(self, model_order: list[str], args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set. Check --dotenv.")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.model_order = model_order
        self.args = args

    def infer(self, row: dict[str, Any], chunk_map: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
        last_error: Exception | None = None
        for model in self.model_order:
            for image_mode in dashscope_image_modes(model, self.args.dashscope_image_format):
                try:
                    messages = build_messages(row, chunk_map, self.args, local=False, image_mode=image_mode)
                    kwargs = {
                        "model": model,
                        "messages": messages,
                        "temperature": self.args.temperature,
                        "max_tokens": self.args.max_tokens,
                    }
                    if self.args.json_mode:
                        kwargs["response_format"] = {"type": "json_object"}
                    response = self.client.chat.completions.create(**kwargs)
                    content = response.choices[0].message.content or ""
                    parse_json_object(content)
                    return content, model, f"dashscope_{image_mode}"
                except Exception as exc:
                    last_error = exc
                    continue
        raise RuntimeError(f"all VLM models failed: {last_error!r}")


class LocalQwenVLClient:
    def __init__(self, args: argparse.Namespace):
        import torch
        from transformers import AutoProcessor

        self.torch = torch
        self.args = args
        self.processor = AutoProcessor.from_pretrained(args.local_model, trust_remote_code=True)
        model_cls = resolve_vl_model_class(args.local_model)
        self.model = model_cls.from_pretrained(
            args.local_model,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def infer(self, row: dict[str, Any], chunk_map: dict[str, dict[str, Any]]) -> tuple[str, str, str]:
        messages = build_messages(row, chunk_map, self.args, local=True, image_mode="local_image")
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = inputs.to(device)
        generation_kwargs = {"max_new_tokens": self.args.max_tokens}
        if self.args.temperature > 0:
            generation_kwargs.update({"do_sample": True, "temperature": self.args.temperature})
        with self.torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0], self.args.local_model, "local_image"


def resolve_vl_model_class(model_path: str):
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration

    if "Qwen2.5-VL" in model_path or "Qwen2_5" in model_path:
        return Qwen2_5_VLForConditionalGeneration
    return Qwen3VLForConditionalGeneration


def dashscope_image_modes(model: str, requested: str) -> list[str]:
    if requested != "auto":
        return [requested]
    lower = model.lower()
    if "qwen3.7-max" in lower or "qwen3-max" in lower or "qwen-max" in lower:
        return ["image", "image_url", "text_only"]
    return ["image_url", "image", "text_only"]


def build_messages(row: dict[str, Any], chunk_map: dict[str, dict[str, Any]], args: argparse.Namespace, *, local: bool, image_mode: str) -> list[dict[str, Any]]:
    task_info = build_task_info(row, chunk_map, args)
    prompt = SYSTEM_PROMPT + "\n当前样本与候选证据：\n" + json.dumps(task_info, ensure_ascii=False, indent=2)
    if local:
        content = [
            {"type": "image", "image": str(row.get("overlay_image"))},
            {"type": "image", "image": str(row.get("artwork_image"))},
            {"type": "text", "text": prompt},
        ]
    elif image_mode == "image":
        content = [
            {"type": "image", "image": image_data_url(row.get("overlay_image"), args.image_max_side)},
            {"type": "image", "image": image_data_url(row.get("artwork_image"), args.crop_max_side)},
            {"type": "text", "text": prompt},
        ]
    elif image_mode == "text_only":
        text_only_prompt = (
            prompt
            + "\n\n注意：当前模型接口未接收图片，只能依据 rule_gold、候选 chunk 和已有 OCR/caption 做证据一致性审核；"
            "不要声称看到了红框图像。visual_elements 只保留已有字段或候选文本明确支持的内容。"
        )
        content = text_only_prompt
    else:
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(row.get("overlay_image"), args.image_max_side)}},
            {"type": "image_url", "image_url": {"url": image_data_url(row.get("artwork_image"), args.crop_max_side)}},
            {"type": "text", "text": prompt},
        ]
    return [{"role": "user", "content": content}]


def build_task_info(row: dict[str, Any], chunk_map: dict[str, dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    gold = row.get("gold") or {}
    candidate_ids = collect_candidate_chunk_ids(row)
    chunks = []
    for chunk_id in candidate_ids[:10]:
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            continue
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source_file": chunk.get("source_file"),
                "title": chunk.get("title"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "contextual_prefix": truncate(chunk.get("contextual_prefix") or "", 160),
                "raw_chunk_text": truncate(chunk.get("raw_chunk_text") or "", args.chunk_snippet_chars),
            }
        )
    return {
        "task_id": row.get("task_id"),
        "source_file": row.get("source_file"),
        "page": row.get("page"),
        "red_box_image_bbox_0_1000": gold.get("image_bbox"),
        "blue_box_caption_bbox_0_1000": gold.get("caption_bbox"),
        "rule_gold": {
            "caption_text": gold.get("caption_text", ""),
            "title": gold.get("title", ""),
            "artist": gold.get("artist", ""),
            "dynasty": gold.get("dynasty", ""),
            "visual_elements": gold.get("visual_elements", []),
            "technique": gold.get("technique", []),
            "composition": gold.get("composition", []),
            "claims": gold.get("claims", []),
        },
        "candidate_chunks": chunks,
    }


def apply_vlm_audit(row: dict[str, Any], parsed: dict[str, Any], raw: str, model: str, input_mode: str, chunk_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row = clone_row(row)
    gold = row.setdefault("gold", {})
    old_claims = {claim.get("field"): claim for claim in gold.get("claims") or []}
    candidate_ids = set(collect_candidate_chunk_ids(row))
    available_ids = set(chunk_map)
    candidate_id_list = collect_candidate_chunk_ids(row)
    candidate_ids = set(candidate_id_list)
    allowed_ids = candidate_ids | available_ids

    top_fields = normalize_top_fields(parsed)
    for field, value in top_fields.items():
        gold[field] = value

    claim_by_field = normalize_claims(parsed.get("claims") or [])
    new_claims = []
    evidence_links = []
    for field in FIELDS:
        old = old_claims.get(field) or {}
        claim_data = claim_by_field.get(field, {})
        top_value = top_fields.get(field, old.get("value"))
        abstain = bool(claim_data.get("abstain", False))
        value = claim_data.get("value", top_value)
        value = clean_value_for_field(field, value)
        evidence_ids = filter_evidence_ids(claim_data.get("evidence_ids") or old.get("evidence_ids") or [], allowed_ids)
        if field in TEXT_NEEDS_CHUNK and not evidence_ids:
            abstain = True
        if value in ("", [], None):
            abstain = True
        support_type = "visual_text" if field == "visual_elements" else "text"
        if abstain:
            claim = {
                "claim_id": field,
                "field": field,
                "value": None,
                "abstain": True,
                "reason": str(claim_data.get("reason") or old.get("reason") or "vlm audit found insufficient support"),
                "evidence_ids": [],
                "support_type": support_type,
            }
        else:
            claim = {
                "claim_id": field,
                "field": field,
                "value": value,
                "abstain": False,
                "evidence_ids": evidence_ids[:3],
                "candidate_evidence_ids": candidate_id_list[:8],
                "support_type": support_type,
                "confidence": clamp_float(claim_data.get("confidence"), default=0.75),
                "evidence_status": "vlm_selected_chunk_support" if evidence_ids else "vlm_visual_or_ocr_support",
            }
            if field in {"visual_elements", "composition"}:
                claim["visual_bbox"] = gold.get("image_bbox")
        new_claims.append(claim)
        if not claim.get("abstain"):
            evidence_links.append(
                {
                    "field": field,
                    "value": claim.get("value"),
                    "gold_evidence_ids": claim.get("evidence_ids", []),
                    "candidate_evidence_ids": candidate_id_list[:8],
                    "support_labels": {chunk_id: ("supports" if chunk_id in claim.get("evidence_ids", []) else "candidate") for chunk_id in candidate_id_list[:8]},
                    "label_source": "vlm_audit",
                }
            )

    gold["claims"] = new_claims
    gold["evidence_chunk_ids"] = sorted({eid for claim in new_claims for eid in claim.get("evidence_ids", [])})
    gold["candidate_evidence_ids"] = candidate_id_list[:8]
    gold["auto_label"] = True
    gold["needs_review"] = True
    gold["label_source"] = "vlm_audited_from_scratch_pdf_blocks_with_legacy_milvus_evidence"
    gold["vlm_audited"] = True
    gold["vlm_audit"] = {
        "model": model,
        "input_mode": input_mode,
        "is_relevant_artwork": parsed.get("is_relevant_artwork"),
        "confidence": clamp_float(parsed.get("confidence"), default=0.0),
        "notes": str(parsed.get("notes") or ""),
        "quality_flags": parsed.get("quality_flags") if isinstance(parsed.get("quality_flags"), list) else [],
    }
    row["evidence_links"] = evidence_links
    row["vlm_raw_response"] = raw
    flags = set(row.get("quality_flags") or [])
    for item in gold["vlm_audit"]["quality_flags"]:
        if isinstance(item, str) and item:
            flags.add(item)
    if parsed.get("is_relevant_artwork") is False:
        flags.add("vlm_marked_irrelevant")
    row["quality_flags"] = sorted(flags)
    return row


def normalize_top_fields(parsed: dict[str, Any]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for field in FIELDS:
        if field not in parsed:
            continue
        fields[field] = clean_value_for_field(field, parsed.get(field))
    return fields


def normalize_claims(raw_claims: list[Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        field = item.get("field")
        if field in FIELDS:
            result[field] = item
    return result


def clean_value_for_field(field: str, value: Any) -> Any:
    if field in {"visual_elements", "technique", "composition"}:
        if isinstance(value, str):
            value = [part.strip() for part in re.split(r"[,，、/;；\s]+", value) if part.strip()]
        if not isinstance(value, list):
            return []
        return dedupe_keep_order([str(item).strip() for item in value if str(item).strip()])[:8]
    if value is None:
        return ""
    if isinstance(value, list):
        value = "、".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


def filter_evidence_ids(values: list[Any], allowed_ids: set[str]) -> list[str]:
    result = []
    for value in values:
        chunk_id = str(value).strip()
        if chunk_id and chunk_id in allowed_ids and chunk_id not in result:
            result.append(chunk_id)
    return result


def collect_candidate_chunk_ids(row: dict[str, Any]) -> list[str]:
    ids = []
    gold = row.get("gold") or {}
    ids.extend(gold.get("candidate_evidence_ids") or [])
    ids.extend(gold.get("evidence_chunk_ids") or [])
    for claim in gold.get("claims") or []:
        ids.extend(claim.get("candidate_evidence_ids") or [])
        ids.extend(claim.get("evidence_ids") or [])
    return dedupe_keep_order([str(item) for item in ids if item])


def load_evidence_chunks(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stores = sorted({str((row.get("gold") or {}).get("evidence_store") or "") for row in rows if (row.get("gold") or {}).get("evidence_store")})
    chunk_map: dict[str, dict[str, Any]] = {}
    for store in stores:
        path = Path(store) / "chunks.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            chunk_id = row.get("chunk_id")
            if chunk_id:
                chunk_map[str(chunk_id)] = row
    return chunk_map


def rebalance_splits_by_source(tasks: list[dict[str, Any]]) -> None:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        buckets[str(task.get("source_stem") or task.get("source_file") or "")].append(task)
    total = len(tasks)
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
    current = Counter()
    order = {"train": 0, "val": 1, "test": 2}
    assignment = {}
    for source, bucket in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        split = min(targets, key=lambda name: (current[name] / max(1.0, targets[name]), current[name], order[name]))
        assignment[source] = split
        current[split] += len(bucket)
    for source, bucket in buckets.items():
        for task in bucket:
            task["split"] = assignment[source]


def build_summary(
    args: argparse.Namespace,
    tasks_path: Path,
    output_dir: Path,
    audited: list[dict[str, Any]],
    final_tasks: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    sft_rows_by_split: dict[str, list[dict[str, Any]]],
    errors: list[str],
    review_path: Path,
) -> dict[str, Any]:
    split_counter = Counter(task.get("split") for task in final_tasks)
    relevant_false = sum(1 for row in audited if is_marked_irrelevant(row))
    audit_error_rows = sum(1 for row in audited if row.get("vlm_audit_error"))
    relevant_unknown = sum(
        1
        for row in audited
        if not row.get("vlm_audit_error") and ((row.get("gold") or {}).get("vlm_audit") or {}).get("is_relevant_artwork") is None
    )
    field_nonempty = {field: sum(1 for row in final_tasks if (row.get("gold") or {}).get(field)) for field in FIELDS}
    model_counts = Counter(((row.get("gold") or {}).get("vlm_audit") or {}).get("model", "unknown") for row in final_tasks)
    input_mode_counts = Counter(((row.get("gold") or {}).get("vlm_audit") or {}).get("input_mode", "unknown") for row in final_tasks)
    non_abstain = [row for row in claim_rows if not row.get("abstain")]
    with_chunk = [row for row in non_abstain if row.get("evidence_ids")]
    support_status = Counter(row.get("evidence_status", "none") for row in non_abstain)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "dataset_name": "EvidenceGrounded-AgentBench",
        "version": args.version,
        "source_tasks": str(tasks_path),
        "output_dir": str(output_dir),
        "vlm_used": True,
        "vlm_provider": args.provider,
        "vlm_model_primary": args.model if args.provider == "dashscope" else args.local_model,
        "vlm_fallback_models": [item.strip() for item in args.fallback_models.split(",") if item.strip()],
        "input_tasks": len(audited),
        "final_tasks": len(final_tasks),
        "dropped_audit_errors": audit_error_rows if args.drop_audit_errors else 0,
        "dropped_irrelevant": max(0, len(audited) - audit_error_rows - len(final_tasks)) if args.drop_audit_errors else len(audited) - len(final_tasks),
        "vlm_marked_irrelevant": relevant_false,
        "vlm_relevance_unknown": relevant_unknown,
        "splits": dict(split_counter),
        "unique_sources": len({task.get("source_file") for task in final_tasks}),
        "claims": len(claim_rows),
        "non_abstain_claims": len(non_abstain),
        "claims_with_chunk_evidence": len(with_chunk),
        "claim_chunk_evidence_coverage": len(with_chunk) / max(1, len(non_abstain)),
        "support_status": dict(support_status),
        "field_nonempty": field_nonempty,
        "vlm_model_counts": dict(model_counts),
        "vlm_input_mode_counts": dict(input_mode_counts),
        "sft_rows": {split: len(rows) for split, rows in sft_rows_by_split.items()},
        "review_html": str(review_path),
        "errors": len(errors),
        "error_samples": errors[:10],
        "limitations": [
            "VLM audit is still automatic; val/test should receive targeted human spot-check before being treated as final gold.",
            "Citation remains chunk-level because the legacy evidence store has many null page_start/page_end values.",
            "The audit can judge visual relevance and correct claims, but it cannot recover missing authoritative evidence outside the local evidence store.",
        ],
    }


def build_manifest(args: argparse.Namespace, tasks_path: Path, output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "build_time": summary["created_at"],
        "builder": "scripts/audit_agentbench_with_vlm.py",
        "args": vars(args),
        "source_tasks": str(tasks_path),
        "outputs": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
            "claim_gold": str(output_dir / "claim_gold.jsonl"),
            "evidence_links": str(output_dir / "evidence_links.jsonl"),
            "sft": str(output_dir / "sft"),
            "episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
            "review_html": str(output_dir / "review" / "review.html"),
        },
        "summary": summary,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# EvidenceGrounded-AgentBench v0.1 VLM 审核构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 输入任务：`{summary['source_tasks']}`",
        f"- 使用 VLM：是，主模型 `{summary['vlm_model_primary']}`",
        "",
        "## 数据规模",
        "",
        f"- 输入任务数：{summary['input_tasks']}",
        f"- 最终任务数：{summary['final_tasks']}",
        f"- VLM/API 失败并移除：{summary['dropped_audit_errors']}",
        f"- VLM 标记无关并移除：{summary['dropped_irrelevant']}",
        f"- split：`{summary['splits']}`",
        f"- unique sources：{summary['unique_sources']}",
        "",
        "## Claim 与证据质量",
        "",
        f"- claims：{summary['claims']}",
        f"- non-abstain claims：{summary['non_abstain_claims']}",
        f"- claims_with_chunk_evidence：{summary['claims_with_chunk_evidence']}",
        f"- claim_chunk_evidence_coverage：{summary['claim_chunk_evidence_coverage']:.4f}",
        f"- support_status：`{summary['support_status']}`",
        f"- field_nonempty：`{summary['field_nonempty']}`",
        f"- vlm_model_counts：`{summary['vlm_model_counts']}`",
        f"- vlm_input_mode_counts：`{summary['vlm_input_mode_counts']}`",
        "",
        "## 轨迹 SFT",
        "",
        f"- sft_rows：`{summary['sft_rows']}`",
        "",
        "## 审核入口",
        "",
        f"- review HTML：`{summary['review_html']}`",
        "",
        "## 限制",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_html(path: Path, tasks: list[dict[str, Any]]) -> Path:
    parts = [
        "<html><head><meta charset='utf-8'><title>VLM Audited EvidenceGrounded AgentBench Review</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .task{border:1px solid #ccc;padding:16px;margin:16px 0;} img{max-width:560px;border:1px solid #ddd;} code{white-space:pre-wrap;display:block;background:#f7f7f7;padding:8px;}</style>",
        "</head><body><h1>VLM Audited EvidenceGrounded AgentBench Review</h1>",
    ]
    for task in tasks:
        gold = task.get("gold", {})
        audit = gold.get("vlm_audit") or {}
        parts.append("<div class='task'>")
        parts.append(f"<h2>{task['task_id']} [{task['split']}]</h2>")
        parts.append(f"<p>{html_escape(str(task.get('source_file')))} page {task.get('page')} | relevant={audit.get('is_relevant_artwork')} | conf={audit.get('confidence')}</p>")
        parts.append(f"<img src='file://{task['overlay_image']}' />")
        parts.append("<code>" + html_escape(json.dumps(gold, ensure_ascii=False, indent=2)[:5000]) + "</code>")
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def image_data_url(path: str | None, max_side: int) -> str:
    if not path:
        return ""
    image_path = Path(path)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            image.save(tmp.name, format="JPEG", quality=86, optimize=True)
            data = base64.b64encode(Path(tmp.name).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object found: {text[:240]}")
    return json.loads(text[start : end + 1])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_completed_stream(path: Path, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = task_index_by_id(source_rows)
    latest: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        task_id = row.get("task_id")
        if task_id is None:
            continue
        row.setdefault("_audit_index", order.get(task_id, len(order)))
        latest[task_id] = row
    return sorted(latest.values(), key=lambda item: int(item.get("_audit_index", 10**9)))


def task_index_by_id(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {str(row.get("task_id")): index for index, row in enumerate(rows)}


def append_stream_row(path: Path, row: dict[str, Any] | None, lock: threading.Lock) -> None:
    if row is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def strip_audit_indexes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for row in rows:
        item = clone_row(row)
        item.pop("_audit_index", None)
        clean.append(item)
    return clean


def load_dotenv(path: str) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def is_marked_irrelevant(row: dict[str, Any]) -> bool:
    return ((row.get("gold") or {}).get("vlm_audit") or {}).get("is_relevant_artwork") is False


def clone_row(row: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(row, ensure_ascii=False))


def clamp_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def truncate(value: str, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    return value[:limit]


def dedupe_keep_order(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
