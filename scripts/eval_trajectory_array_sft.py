#!/usr/bin/env python3
"""Evaluate one-shot trajectory-array SFT models with the executable env."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_trajectory_sft_actions import (  # noqa: E402
    disable_autoawq_dispatch,
    infer_input_device,
    parse_torch_dtype,
)
from evidence_agent_env.env import EvidenceAgentEnv  # noqa: E402


DEFAULT_DATA_DIR = Path("/root/datasets/evidence_grounded_vlm_agentrl/trajectory_array_sft_v0_6_20260605_0350")
DEFAULT_TASKS = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_chunked_claim_sft_20260604_1650/tasks_all.jsonl"
)
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--evidence-index", type=Path, default=DEFAULT_EVIDENCE_INDEX)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default="")
    parser.add_argument("--max-rows", type=int, default=8)
    parser.add_argument("--sample-strategy", choices=["first", "random"], default="first")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--max-seq-length", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--image-max-pixels", type=int, default=131072)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--torch-dtype", default="bf16", choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"])
    parser.add_argument("--phase-aware-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-tool-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-autoawq-dispatch", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.data_dir / f"{args.split}.parquet", args.max_rows, args.sample_strategy, args.seed)
    processor, model = load_model_and_processor(args)
    model.eval()

    env = EvidenceAgentEnv(
        args.tasks,
        args.evidence_index,
        output_dir / "env_rollouts",
        max_steps=args.max_steps,
        phase_aware_mask=args.phase_aware_mask,
        enforce_tool_mask=args.enforce_tool_mask,
    )

    predictions_path = output_dir / "predictions.jsonl"
    records: list[dict[str, Any]] = []
    with predictions_path.open("w", encoding="utf-8") as f:
        for index, row in enumerate(rows):
            pred_text = generate(model, processor, row, args)
            parsed, parse_info = extract_actions(pred_text)
            env_result = execute_actions(env, row, parsed if isinstance(parsed, list) else [])
            gold_actions = json.loads(str(row.get("response") or "[]"))
            record = {
                "index": index,
                "task_id": row.get("task_id"),
                "source_file": row.get("source_file"),
                "page": row.get("page"),
                "pred_text": pred_text,
                "parse_info": parse_info,
                "pred_actions": parsed,
                "gold_action_count": len(gold_actions),
                "pred_action_count": len(parsed) if isinstance(parsed, list) else 0,
                "sequence": sequence_metrics(gold_actions, parsed if isinstance(parsed, list) else []),
                "env": env_result,
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps({"seen": index + 1, "total": len(rows), **brief(records)}, ensure_ascii=False), flush=True)

    summary = {
        "created_at": now(),
        "data_dir": str(args.data_dir),
        "split": args.split,
        "rows_used": len(rows),
        "model": args.model,
        "adapter": args.adapter,
        "args": json_safe(vars(args)),
        "metrics": summarize(records),
        "predictions": str(predictions_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if args.disable_autoawq_dispatch:
        disable_autoawq_dispatch()
    dtype = parse_torch_dtype(args.torch_dtype)
    processor = AutoProcessor.from_pretrained(
        args.model,
        trust_remote_code=True,
        max_pixels=args.image_max_pixels,
    )
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
    kwargs["torch_dtype"] = dtype if dtype != "auto" else "auto"
    if args.load_in_4bit:
        compute_dtype = torch.bfloat16 if dtype == "auto" else dtype
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    model = AutoModelForImageTextToText.from_pretrained(args.model, **kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.config.use_cache = True
    return processor, model


def generate(model: Any, processor: Any, row: dict[str, Any], args: argparse.Namespace) -> str:
    from qwen_vl_utils import process_vision_info

    messages = make_generation_messages(row, args.image_max_pixels)
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
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": processor.tokenizer.eos_token_id,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        generated = model.generate(**inputs, **generation_kwargs)
    output_ids = generated[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def make_generation_messages(row: dict[str, Any], image_max_pixels: int) -> list[dict[str, Any]]:
    messages = to_python(row.get("messages"))
    if not messages:
        raise ValueError(f"row has no messages: {row.get('task_id')}")
    user = dict(messages[0])
    content = str(user.get("content") or "")
    if content.startswith("<image>"):
        content = content[len("<image>") :].lstrip()
    images = to_python(row.get("images")) or []
    image_path = ""
    if images:
        first = images[0]
        image_path = str(first.get("image") if isinstance(first, dict) else first)
    user["content"] = [
        {"type": "image", "image": image_path, "max_pixels": image_max_pixels},
        {"type": "text", "text": content},
    ]
    return [user]


def execute_actions(env: EvidenceAgentEnv, row: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    env.reset(task_id=str(row.get("task_id")))
    errors = []
    terminated = False
    for index, action in enumerate(actions[: env.max_steps]):
        _, _, terminated, info = env.step(action)
        result = info.get("result") or {}
        if result.get("error"):
            errors.append({"step": index, "action": action, "error": result.get("error")})
        if terminated:
            break
    metrics = env.trajectory_metrics()
    return {
        "terminated": terminated,
        "errors": errors[:10],
        "error_count": len(errors),
        "metrics": metrics,
    }


def extract_actions(text: str) -> tuple[Any, dict[str, Any]]:
    cleaned = strip_markdown(text)
    decoder = json.JSONDecoder()
    for start, ch in enumerate(cleaned):
        if ch not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(cleaned[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            return value, {"valid_json": True, "json_type": "array", "trailing_chars": len(cleaned) - start - end}
        if isinstance(value, dict):
            return value, {"valid_json": True, "json_type": "object", "trailing_chars": len(cleaned) - start - end}
    return None, {"valid_json": False, "json_type": "none", "trailing_chars": 0}


def sequence_metrics(gold: list[dict[str, Any]], pred: list[dict[str, Any]]) -> dict[str, Any]:
    gold_names = [str(item.get("action")) for item in gold if isinstance(item, dict)]
    pred_names = [str(item.get("action")) for item in pred if isinstance(item, dict)]
    prefix = 0
    for gold_name, pred_name in zip(gold_names, pred_names):
        if gold_name != pred_name:
            break
        prefix += 1
    return {
        "gold_names": gold_names,
        "pred_names": pred_names,
        "first_action_match": bool(gold_names and pred_names and gold_names[0] == pred_names[0]),
        "prefix_action_match_count": prefix,
        "prefix_action_match_rate": prefix / max(1, len(gold_names)),
        "contains_finish": "finish" in pred_names,
        "ends_with_finish": bool(pred_names and pred_names[-1] == "finish"),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = max(1, len(records))
    parse_types = Counter(str((item.get("parse_info") or {}).get("json_type")) for item in records)
    metrics = [((item.get("env") or {}).get("metrics") or {}) for item in records]
    sequences = [item.get("sequence") or {} for item in records]
    return {
        "n": len(records),
        "valid_json_rate": sum(bool((item.get("parse_info") or {}).get("valid_json")) for item in records) / n,
        "json_array_rate": parse_types["array"] / n,
        "json_object_rate": parse_types["object"] / n,
        "first_action_match_rate": sum(bool(item.get("first_action_match")) for item in sequences) / n,
        "contains_finish_rate": sum(bool(item.get("contains_finish")) for item in sequences) / n,
        "ends_with_finish_rate": sum(bool(item.get("ends_with_finish")) for item in sequences) / n,
        "prefix_action_match_rate_mean": avg([float(item.get("prefix_action_match_rate", 0.0)) for item in sequences]),
        "trajectory_success_rate": avg([float(item.get("trajectory_success", 0.0)) for item in metrics]),
        "finish_rate": avg([float(item.get("finish", 0.0)) for item in metrics]),
        "crop_success_rate": avg([float(item.get("crop_success", 0.0)) for item in metrics]),
        "final_reward_mean": avg([float(item.get("final_reward", 0.0)) for item in metrics]),
        "evidence_recall_mean": avg([float(item.get("evidence_recall", 0.0)) for item in metrics]),
        "claim_supported_rate_mean": avg([float(item.get("claim_supported_rate", 0.0)) for item in metrics]),
        "invalid_step_rate_mean": avg([float(item.get("invalid_step_rate", 0.0)) for item in metrics]),
        "parse_types": dict(parse_types),
    }


def brief(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize(records)
    return {
        "json_array_rate": summary["json_array_rate"],
        "trajectory_success_rate": summary["trajectory_success_rate"],
        "final_reward_mean": summary["final_reward_mean"],
    }


def load_rows(path: Path, limit: int, strategy: str, seed: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    rows = [normalize_row(row.to_dict()) for _, row in df.iterrows()]
    if limit <= 0 or limit >= len(rows):
        return rows
    if strategy == "first":
        return rows[:limit]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit]


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: to_python(value) for key, value in row.items()}


def to_python(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: to_python(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_python(item) for item in value]
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def strip_markdown(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
