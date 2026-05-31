#!/usr/bin/env python3
"""Generation evaluation for EvidenceGrounded trajectory SFT adapters."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch


IMAGE_SIZE_CACHE: dict[str, tuple[int, int]] = {}

ALLOWED_ACTIONS = {
    "crop_image",
    "retrieve_evidence",
    "open_evidence",
    "write_claim",
    "abstain_claim",
    "finish",
}

REQUIRED_KEYS: dict[str, set[str]] = {
    "crop_image": {"bbox"},
    "retrieve_evidence": {"query", "scope", "anchor", "top_k"},
    "open_evidence": {"evidence_id"},
    "write_claim": {"field", "value", "evidence_ids", "visual_bbox", "confidence"},
    "abstain_claim": {"field", "reason"},
    "finish": set(),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-jsonl",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248/sft/val.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--max-rows", type=int, default=128)
    parser.add_argument("--sample-strategy", choices=["first", "random", "balanced_action"], default="balanced_action")
    parser.add_argument(
        "--prompt-mode",
        choices=["compact", "original"],
        default="compact",
        help="compact rebuilds a shorter state prompt from structured history/tool results; original uses dataset messages.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--torch-dtype",
        default="bf16",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--max-text-chars", type=int, default=24000)
    parser.add_argument("--head-text-chars", type=int, default=5000)
    parser.add_argument("--max-history-actions", type=int, default=8)
    parser.add_argument("--max-tool-results", type=int, default=6)
    parser.add_argument("--max-evidence-per-result", type=int, default=3)
    parser.add_argument("--snippet-chars", type=int, default=180)
    parser.add_argument("--coordinate-info", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-seq-length", type=int, default=14336)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--system-prompt", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_all = read_jsonl(Path(args.eval_jsonl))
    rows = select_rows(rows_all, args.max_rows, args.sample_strategy, args.seed)

    processor, model = load_model_and_processor(args)
    model.eval()

    predictions_path = output_dir / "predictions.jsonl"
    metrics = Metrics()
    with predictions_path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            pred_text = generate_action(model, processor, row, args)
            pred_action = extract_json_object(pred_text)
            gold_action = row.get("action") or {}
            result = score_prediction(gold_action, pred_action)
            metrics.add(row, gold_action, pred_action, pred_text, result)
            record = {
                "index": index,
                "task_id": row.get("task_id"),
                "step": row.get("step"),
                "gold_action": gold_action,
                "pred_text": pred_text,
                "pred_action": pred_action,
                "result": result,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(json.dumps({"time": now(), "seen": index + 1, "total": len(rows), **metrics.brief()}, ensure_ascii=False), flush=True)

    summary = {
        "created_at": now(),
        "eval_jsonl": args.eval_jsonl,
        "rows_total": len(rows_all),
        "rows_used": len(rows),
        "model": args.model,
        "adapter": args.adapter,
        "args": vars(args),
        "metrics": metrics.summary(),
        "predictions": str(predictions_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


class Metrics:
    def __init__(self) -> None:
        self.n = 0
        self.counts: Counter[str] = Counter()
        self.denoms: Counter[str] = Counter()
        self.by_gold_action: dict[str, Counter[str]] = defaultdict(Counter)
        self.bbox_ious: list[float] = []

    def add(
        self,
        row: dict[str, Any],
        gold: dict[str, Any],
        pred: dict[str, Any] | None,
        pred_text: str,
        result: dict[str, Any],
    ) -> None:
        self.n += 1
        gold_type = str(gold.get("action", "unknown"))
        self.counts["valid_json"] += int(result["valid_json"])
        self.counts["valid_action"] += int(result["valid_action"])
        self.counts["action_type_match"] += int(result["action_type_match"])
        self.counts["required_keys_match"] += int(result["required_keys_match"])
        self.counts["exact_match"] += int(result["exact_match"])
        self.counts["field_match"] += int(result["field_match"])
        self.counts["evidence_overlap"] += int(result["evidence_overlap"])
        self.counts["scope_match"] += int(result["scope_match"])
        self.counts["bbox_iou_ge_05"] += int(result["bbox_iou"] >= 0.5)
        if gold_type == "retrieve_evidence":
            self.denoms["retrieve_evidence"] += 1
        if gold_type == "crop_image":
            self.denoms["crop_image"] += 1
        if gold_type in {"write_claim", "abstain_claim", "open_evidence", "retrieve_evidence", "crop_image", "finish"}:
            self.denoms["field_comparable"] += 1
        if result["bbox_iou"] >= 0:
            self.bbox_ious.append(float(result["bbox_iou"]))
        for key in [
            "valid_json",
            "valid_action",
            "action_type_match",
            "required_keys_match",
            "exact_match",
            "field_match",
            "evidence_overlap",
            "scope_match",
            "bbox_iou_ge_05",
        ]:
            if key == "bbox_iou_ge_05":
                self.by_gold_action[gold_type][key] += int(result["bbox_iou"] >= 0.5)
            else:
                self.by_gold_action[gold_type][key] += int(result.get(key, False))
        self.by_gold_action[gold_type]["n"] += 1

    def rate(self, key: str) -> float:
        return self.counts[key] / max(1, self.n)

    def brief(self) -> dict[str, float]:
        return {
            "valid_json_rate": self.rate("valid_json"),
            "valid_action_rate": self.rate("valid_action"),
            "action_type_acc": self.rate("action_type_match"),
            "exact_action_acc": self.rate("exact_match"),
        }

    def summary(self) -> dict[str, Any]:
        by_action: dict[str, Any] = {}
        for action, counts in sorted(self.by_gold_action.items()):
            n = max(1, counts["n"])
            by_action[action] = {key: value / n for key, value in counts.items() if key != "n"}
            by_action[action]["n"] = counts["n"]
        return {
            "n": self.n,
            **self.brief(),
            "required_keys_rate": self.rate("required_keys_match"),
            "field_acc": self.rate("field_match"),
            "evidence_overlap_rate": self.rate("evidence_overlap"),
            "scope_acc_on_retrieve": self.counts["scope_match"] / max(1, self.denoms["retrieve_evidence"]),
            "bbox_iou_ge_05_on_crop": self.counts["bbox_iou_ge_05"] / max(1, self.denoms["crop_image"]),
            "bbox_iou_mean": sum(self.bbox_ious) / max(1, len(self.bbox_ious)),
            "by_gold_action": by_action,
        }


def score_prediction(gold: dict[str, Any], pred: dict[str, Any] | None) -> dict[str, Any]:
    valid_json = isinstance(pred, dict)
    gold_type = str(gold.get("action", ""))
    pred_type = str(pred.get("action", "")) if isinstance(pred, dict) else ""
    valid_action = valid_json and pred_type in ALLOWED_ACTIONS
    action_type_match = valid_action and pred_type == gold_type
    required = REQUIRED_KEYS.get(pred_type, set())
    required_keys_match = valid_action and required.issubset(set(pred or {}))
    exact_match = canonical_action(gold) == canonical_action(pred) if isinstance(pred, dict) else False
    field_match = action_type_match and compare_field(gold, pred)
    evidence_overlap = action_type_match and compare_evidence_ids(gold, pred)
    scope_match = action_type_match and gold_type == "retrieve_evidence" and gold.get("scope") == pred.get("scope")
    if gold_type != "retrieve_evidence":
        scope_match = False
    bbox_iou = -1.0
    if action_type_match and gold_type == "crop_image":
        bbox_iou = iou(gold.get("bbox"), pred.get("bbox"))
    return {
        "valid_json": bool(valid_json),
        "valid_action": bool(valid_action),
        "action_type_match": bool(action_type_match),
        "required_keys_match": bool(required_keys_match),
        "exact_match": bool(exact_match),
        "field_match": bool(field_match),
        "evidence_overlap": bool(evidence_overlap),
        "scope_match": bool(scope_match),
        "bbox_iou": bbox_iou,
    }


def compare_field(gold: dict[str, Any], pred: dict[str, Any] | None) -> bool:
    if not isinstance(pred, dict):
        return False
    action = gold.get("action")
    if action in {"write_claim", "abstain_claim"}:
        return str(gold.get("field", "")) == str(pred.get("field", ""))
    if action == "open_evidence":
        return str(gold.get("evidence_id", "")) == str(pred.get("evidence_id", ""))
    if action == "retrieve_evidence":
        return bool(str(pred.get("query", "")).strip()) and str(gold.get("scope", "")) == str(pred.get("scope", ""))
    if action == "finish":
        return True
    if action == "crop_image":
        return iou(gold.get("bbox"), pred.get("bbox")) >= 0.5
    return False


def compare_evidence_ids(gold: dict[str, Any], pred: dict[str, Any] | None) -> bool:
    if not isinstance(pred, dict):
        return False
    gold_ids = set(map(str, gold.get("evidence_ids") or []))
    pred_ids = set(map(str, pred.get("evidence_ids") or []))
    if not gold_ids and not pred_ids:
        return True
    return bool(gold_ids & pred_ids)


def canonical_action(action: dict[str, Any] | None) -> Any:
    if not isinstance(action, dict):
        return None
    return normalize(action)


def normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, float):
        return round(value, 4)
    return value


def iou(a: Any, b: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(x) for x in a]
        bx1, by1, bx2, by2 = [float(x) for x in b]
    except Exception:
        return -1.0
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else -1.0


def generate_action(model: Any, processor: Any, row: dict[str, Any], args: argparse.Namespace) -> str:
    from qwen_vl_utils import process_vision_info

    if args.prompt_mode == "compact":
        messages = build_compact_messages(row, args, include_assistant=False)
    else:
        messages = clone_messages((row.get("messages") or [])[:-1])
    if args.system_prompt:
        messages = [{"role": "system", "content": args.system_prompt}] + messages
    messages = compact_messages(messages, args.max_text_chars, args.head_text_chars)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=True,
        max_length=args.max_seq_length,
        return_tensors="pt",
    )
    device = infer_input_device(model)
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    do_sample = args.temperature > 0
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    input_len = inputs["input_ids"].shape[1]
    output_ids = generated[:, input_len:]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def extract_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    starts = [i for i, ch in enumerate(cleaned) if ch == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(cleaned)):
            ch = cleaned[index]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            obj = json.loads(cleaned[start : index + 1])
                            return obj if isinstance(obj, dict) else None
                        except Exception:
                            break
    return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def select_rows(rows: list[dict[str, Any]], limit: int, strategy: str, seed: int) -> list[dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        selected = list(rows)
    elif strategy == "first":
        selected = rows[:limit]
    elif strategy == "random":
        rng = random.Random(seed)
        selected = list(rows)
        rng.shuffle(selected)
        selected = selected[:limit]
    elif strategy == "balanced_action":
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str((row.get("action") or {}).get("action", "unknown"))].append(row)
        rng = random.Random(seed)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        selected = []
        names = sorted(buckets)
        cursor = 0
        while len(selected) < limit and any(buckets.values()):
            name = names[cursor % len(names)]
            cursor += 1
            if buckets[name]:
                selected.append(buckets[name].pop())
    else:
        raise ValueError(f"unknown sample strategy: {strategy}")
    return selected


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    dtype = parse_torch_dtype(args.torch_dtype)
    processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.image_max_pixels:
        processor_kwargs["max_pixels"] = args.image_max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    model_kwargs["torch_dtype"] = dtype if dtype != "auto" else "auto"
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if dtype == "auto" else dtype
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.config.use_cache = True
    return processor, model


def parse_torch_dtype(name: str) -> Any:
    normalized = str(name or "auto").lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported torch dtype: {name}")


def clone_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(json.dumps(messages, ensure_ascii=False))


def build_compact_messages(row: dict[str, Any], args: argparse.Namespace, *, include_assistant: bool) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for image_path in row.get("images") or []:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": build_compact_prompt_text(row, args)})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if include_assistant:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(row.get("action") or {}, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return messages


def build_compact_prompt_text(row: dict[str, Any], args: argparse.Namespace) -> str:
    meta = extract_meta_from_prompt(row.get("prompt_text", ""))
    history = [simplify_action(item) for item in (row.get("history") or [])[-args.max_history_actions :]]
    tool_results = [simplify_tool_result(item, args) for item in (row.get("tool_results") or [])[-args.max_tool_results :]]
    draft_claims = row.get("draft_claims") or []
    images = row.get("images") or []
    lines = [
        "你是 evidence-grounded figure understanding 的 VLM tool-call agent。",
        "目标：根据 PDF 页面图像、局部裁剪图和可追溯证据，为红框/目标山水画图像写出有证据支撑的结构化 claim。",
        f"task_id：{row.get('task_id')}；step：{row.get('step')}",
        f"source_file：{meta.get('source_file', '')}；page：{meta.get('page', '')}",
        f"输入图像：{len(images)} 张。第 1 张通常是 PDF 页面；第 2 张通常是已裁剪的目标图。",
        "可用工具：",
        '1. {"action":"crop_image","bbox":[x1,y1,x2,y2]}',
        '2. {"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
        '3. {"action":"open_evidence","evidence_id":"ev_xxx"}',
        '4. {"action":"write_claim","field":"caption_text|title|artist|dynasty|visual_elements|technique|composition","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}',
        '5. {"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}',
        '6. {"action":"finish","status":"done"}',
        "约束：只输出一个 JSON 对象；不要输出 markdown；不要编造作品名、画家、朝代、技法；证据不足就 abstain。",
        "历史动作（保留最近若干步）：",
        json.dumps(history, ensure_ascii=False, separators=(",", ":")),
        "工具返回摘要（保留最近若干条，每条检索只保留前几个候选证据）：",
        json.dumps(tool_results, ensure_ascii=False, separators=(",", ":")),
        "当前 claims：",
        json.dumps(draft_claims, ensure_ascii=False, separators=(",", ":")),
        "请根据当前状态选择下一步工具调用。只输出一个 JSON 对象。",
    ]
    if args.coordinate_info:
        image_info = [{"index": i + 1, "path": path, "size": image_size(path)} for i, path in enumerate(images)]
        lines.insert(5, f"图像尺寸：{json.dumps(image_info, ensure_ascii=False, separators=(',', ':'))}")
        lines.insert(6, "坐标规则：所有 bbox 都使用第 1 张 PDF 页面图像的像素坐标，原点在左上角，格式为 [x1,y1,x2,y2]；如果页面上有红框，crop_image 必须裁剪红框范围。")
    return "\n".join(lines)


def extract_meta_from_prompt(prompt_text: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for key in ["source_file", "page"]:
        marker = f"{key}："
        if marker in prompt_text:
            value = prompt_text.split(marker, 1)[1].split("；", 1)[0].split("\n", 1)[0].strip()
            meta[key] = value
    return meta


def simplify_action(action: Any) -> Any:
    if not isinstance(action, dict):
        return action
    keep = {k: action.get(k) for k in ["action", "bbox", "field", "evidence_id", "scope", "top_k", "value", "reason"] if k in action}
    if "query" in action:
        keep["query"] = truncate_text(str(action.get("query", "")), 120)
    if "anchor" in action:
        keep["anchor"] = action.get("anchor")
    if "evidence_ids" in action:
        keep["evidence_ids"] = action.get("evidence_ids")
    return keep


def simplify_tool_result(result: Any, args: argparse.Namespace) -> Any:
    if not isinstance(result, dict):
        return result
    tool = result.get("tool")
    if tool == "crop_image":
        return {"tool": "crop_image", "bbox": result.get("bbox"), "crop_path": result.get("crop_path")}
    if tool == "retrieve_evidence":
        return {
            "tool": "retrieve_evidence",
            "scope": result.get("scope"),
            "query": truncate_text(str(result.get("query", "")), 120),
            "anchor": result.get("anchor"),
            "results": [simplify_evidence(item, args) for item in (result.get("results") or [])[: args.max_evidence_per_result]],
        }
    if tool == "open_evidence":
        simplified = {"tool": "open_evidence", "evidence_id": result.get("evidence_id")}
        for key in ["source_file", "page_start", "page_end", "authority_level", "citation_level", "source_quality"]:
            if key in result:
                simplified[key] = result.get(key)
        for key in ["display_snippet", "evidence_summary", "text", "raw_chunk_text"]:
            if key in result and result.get(key):
                simplified[key] = truncate_text(str(result.get(key)), args.snippet_chars)
                break
        return simplified
    return {key: result.get(key) for key in list(result)[:8]}


def simplify_evidence(item: Any, args: argparse.Namespace) -> Any:
    if not isinstance(item, dict):
        return item
    snippet = item.get("evidence_summary") or item.get("display_snippet") or item.get("text") or ""
    return {
        "evidence_id": item.get("evidence_id"),
        "source_file": item.get("source_file"),
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "authority_level": item.get("authority_level"),
        "citation_level": item.get("citation_level"),
        "score": item.get("score"),
        "snippet": truncate_text(str(snippet), args.snippet_chars),
    }


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def image_size(path: str) -> dict[str, int | None]:
    if path in IMAGE_SIZE_CACHE:
        w, h = IMAGE_SIZE_CACHE[path]
        return {"width": w, "height": h}
    try:
        from PIL import Image

        with Image.open(path) as image:
            size = image.size
        IMAGE_SIZE_CACHE[path] = size
        return {"width": size[0], "height": size[1]}
    except Exception:
        return {"width": None, "height": None}


def compact_messages(messages: list[dict[str, Any]], max_text_chars: int, head_text_chars: int) -> list[dict[str, Any]]:
    if max_text_chars <= 0:
        return messages
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    item["text"] = compact_text(item["text"], max_text_chars, head_text_chars)
        elif isinstance(content, str) and message.get("role") != "assistant":
            message["content"] = compact_text(content, max_text_chars, head_text_chars)
    return messages


def compact_text(text: str, max_chars: int, head_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_chars = min(max(512, head_chars), max_chars - 512)
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + "\n\n[中间过长的历史/证据返回已为评测截断，保留开头任务定义和结尾当前状态。]\n\n"
        + text[-tail_chars:]
    )


def infer_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
