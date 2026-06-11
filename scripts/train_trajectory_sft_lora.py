#!/usr/bin/env python3
"""LoRA SFT for EvidenceGrounded VLM tool-call trajectories.

The dataset is a JSONL of chat rows:
  image/text user prompt + assistant JSON action

This script intentionally keeps the training loop small and explicit so each
run can be audited from the saved summary, logs, and adapter directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


IMAGE_SIZE_CACHE: dict[str, tuple[int, int]] = {}

ALLOWED_ACTIONS = {
    "inspect_page",
    "propose_regions",
    "select_evidence",
    "crop_region",
    "crop_target",
    "crop_image",
    "retrieve_evidence",
    "open_evidence",
    "write_claim",
    "abstain_claim",
    "write_claims_chunk",
    "write_claims_batch",
    "finish",
}
CLAIM_FIELD_SPEC = (
    "caption_text|image_scope|depicted_work_title|displayed_region|object_type|artist|dynasty|"
    "visual_elements|technique|composition|medium_dimensions|collection"
)
EVIDENCE_POLICY_KEYS = [
    "clean_evidence_type",
    "adjudicated_evidence_role",
    "adjudication_status",
    "adjudicated_claim_allowed_fields",
    "usable_for_claim_by_adjudication",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-jsonl",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248/sft/train.jsonl",
    )
    parser.add_argument(
        "--val-jsonl",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_3_1_low_text_vlm_full_sft_20260531_0248/sft/val.jsonl",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", default=None, help="Optional existing LoRA adapter to continue training.")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=128)
    parser.add_argument("--include-actions", default=None, help="Optional comma-separated gold action allowlist for training rows.")
    parser.add_argument("--sample-strategy", choices=["first", "random", "balanced_action"], default="balanced_action")
    parser.add_argument(
        "--prompt-mode",
        choices=["compact", "original"],
        default="compact",
        help="compact rebuilds a shorter state prompt from structured history/tool results; original uses dataset messages.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated LoRA target modules. Use all-linear only when the PEFT/quantization stack is known healthy.",
    )
    parser.add_argument(
        "--disable-autoawq-dispatch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable PEFT AutoAWQ LoRA dispatcher. Useful when a broken awq package is installed but the base model is not AWQ.",
    )
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--torch-dtype",
        default="bf16",
        choices=["auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"],
    )
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=14336,
        help="Maximum token length after chat templating. Prompt text is compacted before tokenization when needed.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=24000,
        help="Maximum user text characters before tokenization; keeps head and tail.",
    )
    parser.add_argument("--head-text-chars", type=int, default=5000)
    parser.add_argument("--max-history-actions", type=int, default=8)
    parser.add_argument("--max-tool-results", type=int, default=6)
    parser.add_argument("--max-evidence-per-result", type=int, default=3)
    parser.add_argument("--snippet-chars", type=int, default=180)
    parser.add_argument("--coordinate-info", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=0, help="0 disables interim validation loss.")
    parser.add_argument("--save-every", type=int, default=0, help="0 saves only final adapter.")
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=5.0,
        help="Seconds between nvidia-smi samples. Set <=0 to disable GPU memory logging.",
    )
    parser.add_argument(
        "--training-record",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate 训练记录.md with loss and GPU memory plots after training.",
    )
    parser.add_argument("--training-record-title", default="", help="Optional title for 训练记录.md.")
    parser.add_argument("--training-record-notes", default="", help="Optional notes appended to 训练记录.md.")
    parser.add_argument("--system-prompt", default="")
    return parser.parse_args()


class JsonlDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_path = output_dir / "train_log.jsonl"
    gpu_monitor_path = output_dir / "gpu_memory_monitor.jsonl"
    gpu_monitor = start_gpu_monitor(gpu_monitor_path, args.gpu_monitor_interval)

    train_rows_all = read_jsonl(Path(args.train_jsonl))
    val_rows_all = read_jsonl(Path(args.val_jsonl)) if args.val_jsonl else []
    train_rows_all = filter_rows_by_action(train_rows_all, args.include_actions)
    val_rows_all = filter_rows_by_action(val_rows_all, args.include_actions)
    train_rows = select_rows(train_rows_all, args.max_train_rows, args.sample_strategy, args.seed)
    val_rows = select_rows(val_rows_all, args.max_val_rows, "balanced_action", args.seed + 17) if val_rows_all else []

    processor, model = load_model_and_processor(args)
    collator = VlmSftCollator(processor, args)

    train_loader = DataLoader(
        JsonlDataset(train_rows),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )
    val_loader = (
        DataLoader(JsonlDataset(val_rows), batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        if val_rows
        else None
    )

    total_update_steps = max(1, math.ceil(len(train_loader) * args.epochs / args.gradient_accumulation_steps))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_lr_schedule(total_update_steps, max(0, int(total_update_steps * args.warmup_ratio))),
    )

    run_config = {
        "created_at": now(),
        "model": args.model,
        "adapter": args.adapter,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "train_rows_total": len(train_rows_all),
        "val_rows_total": len(val_rows_all),
        "train_rows_used": len(train_rows),
        "val_rows_used": len(val_rows),
        "train_action_distribution": dict(action_counter(train_rows)),
        "val_action_distribution": dict(action_counter(val_rows)),
        "args": vars(args),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    running_loss = 0.0
    losses: list[float] = []
    skipped_batches = 0

    max_micro_steps = int(math.ceil(len(train_loader) * args.epochs))
    while micro_step < max_micro_steps:
        for batch in train_loader:
            if micro_step >= max_micro_steps:
                break
            micro_step += 1
            if batch is None:
                skipped_batches += 1
                continue
            batch = move_to_device(batch, infer_input_device(model))
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps
            losses.append(float(loss.detach().cpu()) * args.gradient_accumulation_steps)

            if micro_step % args.gradient_accumulation_steps == 0 or micro_step == max_micro_steps:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % max(1, args.log_every) == 0 or global_step == 1:
                    record = {
                        "time": now(),
                        "global_step": global_step,
                        "micro_step": micro_step,
                        "loss": running_loss / max(1, args.log_every),
                        "lr": scheduler.get_last_lr()[0],
                        "skipped_batches": skipped_batches,
                    }
                    append_jsonl(logs_path, record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    running_loss = 0.0

                if args.eval_every and val_loader is not None and global_step % args.eval_every == 0:
                    val_loss = evaluate_loss(model, val_loader, args)
                    record = {"time": now(), "global_step": global_step, "val_loss": val_loss}
                    append_jsonl(logs_path, record)
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    model.train()

                if args.save_every and global_step % args.save_every == 0:
                    save_adapter(model, processor, output_dir / f"checkpoint-{global_step}")

    final_val_loss = evaluate_loss(model, val_loader, args) if val_loader is not None else None
    adapter_dir = output_dir / "adapter"
    save_adapter(model, processor, adapter_dir)
    trainable, total = parameter_counts(model)
    if gpu_monitor is not None:
        gpu_monitor.stop()
        gpu_monitor = None
    summary = {
        "created_at": now(),
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "model": args.model,
        "base_or_initial_adapter": args.adapter,
        "train_rows_used": len(train_rows),
        "val_rows_used": len(val_rows),
        "optimizer_steps": global_step,
        "micro_steps": micro_step,
        "mean_train_loss": sum(losses) / max(1, len(losses)),
        "final_val_loss": final_val_loss,
        "skipped_batches": skipped_batches,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_parameter_ratio": trainable / max(1, total),
        "run_config": str(output_dir / "run_config.json"),
        "train_log": str(logs_path),
        "gpu_memory_monitor": str(gpu_monitor_path) if gpu_monitor_path.exists() else None,
        "training_record": str(output_dir / "训练记录.md") if args.training_record else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.training_record:
        generate_training_record(output_dir, args)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


class GpuMemoryMonitor:
    def __init__(self, output_path: Path, interval: float) -> None:
        self.output_path = output_path
        self.interval = max(0.5, float(interval))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="gpu-memory-monitor", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=self.interval + 2.0)

    def _run(self) -> None:
        with self.output_path.open("a", encoding="utf-8") as f:
            while not self.stop_event.is_set():
                record = {
                    "time": now(),
                    "timestamp": time.time(),
                    "pid_alive": True,
                    "gpus": query_nvidia_smi(),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                self.stop_event.wait(self.interval)


def start_gpu_monitor(output_path: Path, interval: float) -> GpuMemoryMonitor | None:
    if interval <= 0:
        return None
    if shutil.which("nvidia-smi") is None:
        print(
            json.dumps({"time": now(), "warning": "nvidia-smi not found; skip GPU memory monitor"}, ensure_ascii=False),
            flush=True,
        )
        return None
    monitor = GpuMemoryMonitor(output_path, interval)
    monitor.start()
    return monitor


def query_nvidia_smi() -> list[dict[str, Any]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    if proc.returncode != 0:
        return []
    gpus: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                {
                    "gpu_index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(float(parts[2])),
                    "utilization_gpu_pct": int(float(parts[3])),
                }
            )
        except ValueError:
            continue
    return gpus


def generate_training_record(output_dir: Path, args: argparse.Namespace) -> None:
    script = Path(__file__).with_name("summarize_sft_training_run.py")
    if not script.exists():
        print(
            json.dumps({"time": now(), "warning": f"training record script not found: {script}"}, ensure_ascii=False),
            flush=True,
        )
        return
    notes = args.training_record_notes or (
        f"由 `scripts/train_trajectory_sft_lora.py` 自动生成；gpu_monitor_interval={args.gpu_monitor_interval}s。"
    )
    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(output_dir),
        "--title",
        args.training_record_title or output_dir.name,
        "--notes",
        notes,
    ]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(
            json.dumps(
                {
                    "time": now(),
                    "warning": "failed to generate training record",
                    "returncode": proc.returncode,
                    "stderr": proc.stderr[-1000:],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    elif proc.stdout.strip():
        print(proc.stdout.strip(), flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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
        action_names = sorted(buckets)
        cursor = 0
        while len(selected) < limit and any(buckets.values()):
            name = action_names[cursor % len(action_names)]
            cursor += 1
            if buckets[name]:
                selected.append(buckets[name].pop())
    else:
        raise ValueError(f"unknown sample strategy: {strategy}")
    return selected


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("action") or {}).get("action", "unknown")) for row in rows)


def filter_rows_by_action(rows: list[dict[str, Any]], include_actions: str | None) -> list[dict[str, Any]]:
    if not include_actions:
        return rows
    allowed = {item.strip() for item in include_actions.split(",") if item.strip()}
    return [row for row in rows if str((row.get("action") or {}).get("action", "")) in allowed]


class VlmSftCollator:
    def __init__(self, processor: Any, args: argparse.Namespace) -> None:
        self.processor = processor
        self.args = args

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor] | None:
        from qwen_vl_utils import process_vision_info

        prepared = [self.prepare_row(row) for row in rows]
        prepared = [item for item in prepared if item is not None]
        if not prepared:
            return None

        prompt_messages = [item["prompt_messages"] for item in prepared]
        full_messages = [item["full_messages"] for item in prepared]
        prompt_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in prompt_messages
        ]
        full_texts = [
            self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            for messages in full_messages
        ]
        image_inputs, video_inputs = process_vision_info(prompt_messages)
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_length,
            return_tensors="pt",
        )
        full_inputs = self.processor(
            text=full_texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            truncation=True,
            max_length=self.args.max_seq_length,
            return_tensors="pt",
        )
        labels = full_inputs.input_ids.clone()
        if "attention_mask" in full_inputs:
            labels[full_inputs.attention_mask == 0] = -100
        for i in range(labels.shape[0]):
            prompt_len = int(prompt_inputs.input_ids[i].ne(self.processor.tokenizer.pad_token_id).sum().item())
            labels[i, : min(prompt_len, labels.shape[1])] = -100
        if int((labels != -100).sum().item()) == 0:
            return None
        full_inputs["labels"] = labels
        return dict(full_inputs)

    def prepare_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        if self.args.prompt_mode == "compact":
            prompt_messages = build_compact_messages(row, self.args, include_assistant=False)
            full_messages = build_compact_messages(row, self.args, include_assistant=True)
        else:
            messages = row.get("messages") or []
            if len(messages) < 2 or messages[-1].get("role") != "assistant":
                return None
            prompt_messages = clone_messages(messages[:-1])
            full_messages = clone_messages(messages)
        if self.args.system_prompt:
            prompt_messages = [{"role": "system", "content": self.args.system_prompt}] + prompt_messages
            full_messages = [{"role": "system", "content": self.args.system_prompt}] + full_messages
        prompt_messages = compact_messages(prompt_messages, self.args.max_text_chars, self.args.head_text_chars)
        full_messages = compact_messages(full_messages, self.args.max_text_chars, self.args.head_text_chars)
        return {"prompt_messages": prompt_messages, "full_messages": full_messages}


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
    no_select = is_no_select_row(row)
    v0_7 = is_inspect_crop_row(row)
    if v0_7:
        target_line = (
            "目标：先检查 PDF 页面布局，再裁剪目标山水画图像，之后直接打开/检索可见证据并写出有证据支撑的结构化 claim；本协议不使用 select_evidence。"
            if no_select
            else "目标：先检查 PDF 页面布局，再裁剪目标山水画图像，之后检索证据并写出有证据支撑的结构化 claim。"
        )
        tool_lines = [
            '1. {"action":"inspect_page","top_k":整数}',
            '2. {"action":"crop_target","region_id":"r_xxx"} 或 {"action":"crop_target","bbox":[x1,y1,x2,y2]}',
            '3. {"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
            '4. {"action":"open_evidence","evidence_id":"ev_xxx"}',
            f'6. {{"action":"write_claims_chunk","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            f'7. {{"action":"write_claim","field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}',
            '8. {"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}',
            f'9. {{"action":"write_claims_batch","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            '10. {"action":"finish","status":"done"}',
        ]
        if not no_select:
            tool_lines.insert(4, '5. {"action":"select_evidence","evidence_ids":["ev_xxx或local_caption_xxx"]}')
        coordinate_rule = (
            "坐标规则：inspect_page 会返回 layout regions；region bbox 和 crop_target 的 bbox 都使用第 1 张 PDF 原始页面图像的像素坐标，"
            "原点在左上角，格式为 [x1,y1,x2,y2]；先 inspect_page，再 crop_target。"
        )
    else:
        target_line = "目标：根据 PDF 页面、候选区域、候选证据、局部裁剪图和可追溯证据，为目标山水画图像选择可信 evidence_id，并写出有证据支撑的结构化 claim。"
        tool_lines = [
            '1. {"action":"inspect_page"}',
            '2. {"action":"propose_regions","top_k":整数}',
            '3. {"action":"select_evidence","evidence_ids":["ev_xxx或local_caption_xxx"]}',
            '4. {"action":"crop_region","region_id":"r_xxx"}',
            '5. {"action":"crop_target","region_id":"r_xxx"} 或 {"action":"crop_target","bbox":[x1,y1,x2,y2]}',
            '6. {"action":"crop_image","bbox":[x1,y1,x2,y2]}',
            '7. {"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
            '8. {"action":"open_evidence","evidence_id":"ev_xxx"}',
            f'9. {{"action":"write_claim","field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}',
            '10. {"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}',
            f'11. {{"action":"write_claims_batch","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            '12. {"action":"finish","status":"done"}',
        ]
        coordinate_rule = (
            "坐标规则：所有候选 region 和 bbox 都使用第 1 张 PDF 页面图像的像素坐标，原点在左上角，格式为 [x1,y1,x2,y2]；"
            "无红框页面优先用 propose_regions 查看候选区域，下一步必须服从当前阶段允许的工具列表；"
            "如果当前阶段只允许 crop_region，就不能提前 select_evidence。"
        )
    lines = [
        "你是 evidence-grounded figure understanding 的 VLM tool-call agent。",
        target_line,
        f"task_id：{row.get('task_id')}；step：{row.get('step')}",
        f"source_file：{meta.get('source_file', '')}；page：{meta.get('page', '')}",
        f"输入图像：{len(images)} 张。第 1 张通常是 PDF 页面；第 2 张通常是已裁剪的目标图。",
        "可用工具：",
        *tool_lines,
        "约束：只输出一个 JSON 对象；不要输出 markdown；不要编造作品名、画家、朝代、技法；证据不足就 abstain。",
        "证据边界约束：如果证据摘要包含 adjudicated_claim_allowed_fields，只能用该 evidence 支持这些字段；如果 usable_for_claim_by_adjudication=false 或 adjudication_status 不是 accepted_auto，应对该字段 abstain 或继续检索。",
        "Claim 写入约束：write_claims_chunk 每次最多写入或 abstain 1 个 remaining_fields 中的字段；remaining_fields 非空时禁止 finish。",
        "历史动作（保留最近若干步）：",
        json.dumps(history, ensure_ascii=False, separators=(",", ":")),
        "工具返回摘要（保留最近若干条，每条检索只保留前几个候选证据）：",
        json.dumps(tool_results, ensure_ascii=False, separators=(",", ":")),
        "已选择 evidence_ids：",
        json.dumps(row.get("selected_evidence_ids") or [], ensure_ascii=False, separators=(",", ":")),
        "当前阶段允许的工具：",
        json.dumps(row.get("available_actions") or [], ensure_ascii=False, separators=(",", ":")),
        "阶段约束：如果当前阶段允许的工具列表非空，输出 JSON 的 action 必须严格从该列表中选择；总工具表中的其他工具此时禁止使用。",
        f"阶段提示：{row.get('phase_hint') or ''}",
        "当前 claim_state（优先依据 target_fields / remaining_fields 判断需要继续写哪些字段，以及何时可以 finish）：",
        json.dumps(simplify_claim_state(row.get("claim_state") or {}), ensure_ascii=False, separators=(",", ":")),
        "当前 claims：",
        json.dumps(draft_claims, ensure_ascii=False, separators=(",", ":")),
        "请根据当前状态选择下一步工具调用。只输出一个 JSON 对象。",
    ]
    if is_crop_only_region_selection_row(row):
        crop_action = "crop_target" if v0_7 else "crop_region"
        region_source = "inspect_page 返回的 layout regions" if v0_7 else "propose_regions 返回的候选区域"
        lines.insert(
            5,
            f"阶段提示：当前已经看到 {region_source}；当前阶段只允许 {crop_action}，"
            "禁止 select_evidence、open_evidence、retrieve_evidence 或 finish。"
            f"下一步必须输出 {{\"action\":\"{crop_action}\",\"region_id\":\"r0\"}} 这种 JSON，region_id 必须来自候选区域列表。",
        )
        lines.insert(6, "当前阶段允许的工具（必须只从这里选择 action）：")
        lines.insert(7, json.dumps([crop_action], ensure_ascii=False, separators=(",", ":")))
    if args.coordinate_info:
        image_info = [{"index": i + 1, "path": path, "size": image_size(path)} for i, path in enumerate(images)]
        lines.insert(5, f"图像尺寸：{json.dumps(image_info, ensure_ascii=False, separators=(',', ':'))}")
        lines.insert(6, coordinate_rule)
    return "\n".join(lines)


def is_inspect_crop_row(row: dict[str, Any]) -> bool:
    version = str(row.get("tool_schema_version") or "")
    if "v0.7" in version or "inspect_crop" in version:
        return True
    actions = [row.get("action") if isinstance(row.get("action"), dict) else {}]
    actions.extend(item for item in row.get("history") or [] if isinstance(item, dict))
    return any(str(item.get("action")) in {"inspect_page", "crop_target"} for item in actions)


def is_no_select_row(row: dict[str, Any]) -> bool:
    version = str(row.get("tool_schema_version") or "")
    if "no_select" in version:
        return True
    phase_name = str(row.get("phase_name") or "")
    if "no_select" in phase_name:
        return True
    actions = [str((row.get("action") or {}).get("action", ""))]
    actions.extend(str(item.get("action", "")) for item in row.get("history") or [] if isinstance(item, dict))
    return "select_evidence" not in actions and any(action in {"inspect_page", "crop_target"} for action in actions)


def is_crop_only_region_selection_row(row: dict[str, Any]) -> bool:
    action = row.get("action") if isinstance(row.get("action"), dict) else {}
    if action.get("action") not in {"crop_region", "crop_target"}:
        return False
    history_actions = [item.get("action") for item in row.get("history") or [] if isinstance(item, dict)]
    tool_names = [item.get("tool") for item in row.get("tool_results") or [] if isinstance(item, dict)]
    has_regions = any(name in history_actions or name in tool_names for name in ["inspect_page", "propose_regions"])
    return has_regions and not any(
        name in tool_names for name in ["crop_region", "crop_target", "crop_image"]
    )


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
    keep = {
        k: action.get(k)
        for k in ["action", "bbox", "region_id", "field", "evidence_id", "scope", "top_k", "value", "reason", "status"]
        if k in action
    }
    if "query" in action:
        keep["query"] = truncate_text(str(action.get("query", "")), 120)
    if "anchor" in action:
        keep["anchor"] = action.get("anchor")
    if "evidence_ids" in action:
        keep["evidence_ids"] = action.get("evidence_ids")
    if "claims" in action:
        keep["claims"] = action.get("claims")
    if "abstains" in action:
        keep["abstains"] = action.get("abstains")
    return keep


def simplify_tool_result(result: Any, args: argparse.Namespace) -> Any:
    if not isinstance(result, dict):
        return result
    tool = result.get("tool")
    if tool in {"crop_image", "crop_region", "crop_target"}:
        keep = {"tool": tool, "bbox": result.get("bbox"), "crop_path": result.get("crop_path")}
        if "region_id" in result:
            keep["region_id"] = result.get("region_id")
        return keep
    if tool in {"propose_regions", "inspect_page"} and result.get("regions"):
        return {
            "tool": tool,
            "regions": [
                {
                    "region_id": item.get("region_id"),
                    "bbox": item.get("bbox"),
                    "type": item.get("type"),
                    "source": item.get("source"),
                    "score": item.get("score"),
                    "nearby_text": truncate_text(str(item.get("nearby_text", "")), 120) if item.get("nearby_text") else None,
                    "caption_evidence_id": item.get("caption_evidence_id"),
                    "caption_hint": truncate_text(str(item.get("caption_hint", "")), 120) if item.get("caption_hint") else None,
                    "hint": truncate_text(str(item.get("hint", "")), 120) if item.get("hint") else None,
                }
                for item in (result.get("regions") or [])[: args.max_evidence_per_result * 2]
            ],
        }
    if tool == "retrieve_evidence":
        return {
            "tool": "retrieve_evidence",
            "scope": result.get("scope"),
            "query": truncate_text(str(result.get("query", "")), 120),
            "anchor": result.get("anchor"),
            "results": [simplify_evidence(item, args) for item in (result.get("results") or [])[: args.max_evidence_per_result]],
        }
    if tool == "select_evidence":
        return {
            "tool": "select_evidence",
            "selected_evidence_ids": result.get("selected_evidence_ids") or [],
            "selected_evidence": [
                simplify_evidence(item, args)
                for item in (result.get("selected_evidence") or [])[: args.max_evidence_per_result]
            ],
            "rejected_evidence_ids": result.get("rejected_evidence_ids") or [],
        }
    if tool == "open_evidence":
        simplified = {"tool": "open_evidence", "evidence_id": result.get("evidence_id")}
        for key in ["source_file", "page_start", "page_end", "authority_level", "citation_level", "source_quality"]:
            if key in result:
                simplified[key] = result.get(key)
        for key in EVIDENCE_POLICY_KEYS:
            if key in result and result.get(key) is not None:
                simplified[key] = result.get(key)
        for key in ["display_snippet", "evidence_summary", "text", "raw_chunk_text"]:
            if key in result and result.get(key):
                simplified[key] = truncate_text(str(result.get(key)), args.snippet_chars)
                break
        return simplified
    return {key: result.get(key) for key in list(result)[:8]}


def simplify_claim_state(claim_state: Any) -> dict[str, Any]:
    if not isinstance(claim_state, dict):
        return {}
    keep_keys = [
        "target_fields",
        "written_fields",
        "abstained_fields",
        "remaining_fields",
        "claim_count",
        "abstain_count",
        "evidence_ids",
    ]
    return {key: claim_state.get(key) for key in keep_keys if key in claim_state}


def simplify_evidence(item: Any, args: argparse.Namespace) -> Any:
    if not isinstance(item, dict):
        return item
    snippet = item.get("evidence_summary") or item.get("display_snippet") or item.get("text") or ""
    simplified = {
        "evidence_id": item.get("evidence_id"),
        "source_file": item.get("source_file"),
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "authority_level": item.get("authority_level"),
        "citation_level": item.get("citation_level"),
        "score": item.get("score"),
        "snippet": truncate_text(str(snippet), args.snippet_chars),
    }
    for key in EVIDENCE_POLICY_KEYS:
        if key in item and item.get(key) is not None:
            simplified[key] = item.get(key)
    return simplified


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
        + "\n\n[中间过长的历史/证据返回已为训练截断，保留开头任务定义和结尾当前状态。]\n\n"
        + text[-tail_chars:]
    )


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    if args.disable_autoawq_dispatch:
        disable_autoawq_dispatch()

    dtype = parse_torch_dtype(args.torch_dtype)
    processor_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if args.image_max_pixels:
        processor_kwargs["max_pixels"] = args.image_max_pixels
    processor = AutoProcessor.from_pretrained(args.model, **processor_kwargs)
    if getattr(processor, "tokenizer", None) is not None:
        processor.tokenizer.padding_side = "right"
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
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=parse_lora_target_modules(args.lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    model.config.use_cache = False
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return processor, model


def parse_lora_target_modules(value: str) -> str | list[str]:
    value = str(value or "").strip()
    if not value or value == "all-linear":
        return "all-linear"
    return [item.strip() for item in value.split(",") if item.strip()]


def disable_autoawq_dispatch() -> None:
    try:
        import peft.import_utils as peft_import_utils
        import peft.tuners.lora.awq as peft_lora_awq

        peft_import_utils.is_auto_awq_available.cache_clear()
        peft_import_utils.is_auto_awq_available = lambda: False
        peft_lora_awq.is_auto_awq_available = lambda: False
    except Exception:
        return


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


def make_lr_schedule(total_steps: int, warmup_steps: int):
    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-6, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))

    return schedule


def evaluate_loss(model: Any, val_loader: DataLoader | None, args: argparse.Namespace) -> float | None:
    if val_loader is None:
        return None
    losses: list[float] = []
    model.eval()
    device = infer_input_device(model)
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            batch = move_to_device(batch, device)
            outputs = model(**batch)
            losses.append(float(outputs.loss.detach().cpu()))
    return sum(losses) / max(1, len(losses))


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def infer_input_device(model: Any) -> torch.device:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return torch.device(device)
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_adapter(model: Any, processor: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)


def parameter_counts(model: Any) -> tuple[int, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return trainable, total


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
