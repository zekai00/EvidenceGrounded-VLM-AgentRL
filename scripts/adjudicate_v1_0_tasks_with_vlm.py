#!/usr/bin/env python3
"""Adjudicate v1.0 layout-candidate tasks with a DashScope VLM sample."""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402


PROMPT = """你是 EvidenceGrounded-VLM-AgentRL 的数据质量裁决员。你会看到两张图：
1. PDF 页面 overlay：红框是目标候选图像，青色框是候选图注。
2. 红框裁剪图。

请判断这条样本是否适合作为“定位目标山水画相关图像，并基于图注/证据写结构化 claim”的 agent 训练任务。

只输出 JSON，不要输出 Markdown：
{
  "is_valid_target": true,
  "is_landscape_related": true,
  "bbox_quality": "good|partial|too_large|too_small|text_region|non_image|unclear",
  "caption_match": "good|partial|wrong|missing|unclear",
  "should_keep_for_training": true,
  "confidence": 0.0,
  "reason": "一句话说明"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dotenv", default="/root/Workspace/VLM/EviTool-VL/.env")
    parser.add_argument("--model", default="qwen3.7-max-2026-05-20")
    parser.add_argument("--fallback-models", default="qwen3.7-max-2026-05-17,qwen3.7-max,qwen3.6-flash-2026-04-16")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--seed", type=int, default=20260608)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--image-max-side", type=int, default=1200)
    parser.add_argument("--crop-max-side", type=int, default=768)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(Path(args.dotenv))
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.dataset_dir) / f"vlm_adjudication_{datetime.now().strftime('%Y%m%d_%H%M')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = read_jsonl(Path(args.dataset_dir) / "tasks_all.jsonl")
    splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    pool = [task for task in tasks if str(task.get("split")) in splits]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    selected = pool[: args.limit]
    client = DashScopeClient(args)
    decisions: list[dict[str, Any]] = []
    for index, task in enumerate(selected):
        try:
            raw, model, mode = client.infer(task)
            parsed = parse_json_object(raw)
            decision = {
                "index": index,
                "task_id": task.get("task_id"),
                "split": task.get("split"),
                "source_file": task.get("source_file"),
                "page": task.get("page"),
                "candidate_source": (task.get("candidate_meta") or {}).get("source"),
                "model": model,
                "input_mode": mode,
                "decision": parsed,
                "raw": raw,
            }
            print(json.dumps({"index": index, "task_id": task.get("task_id"), "ok": True, "keep": parsed.get("should_keep_for_training"), "model": model}, ensure_ascii=False), flush=True)
        except Exception as exc:
            decision = {
                "index": index,
                "task_id": task.get("task_id"),
                "split": task.get("split"),
                "source_file": task.get("source_file"),
                "page": task.get("page"),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(json.dumps({"index": index, "task_id": task.get("task_id"), "ok": False, "error": decision["error"]}, ensure_ascii=False), flush=True)
        decisions.append(decision)
    write_jsonl(output_dir / "vlm_adjudication.jsonl", decisions)
    summary = summarize(args, output_dir, decisions)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "VLM抽检报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


class DashScopeClient:
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.models = [args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()]
        self.args = args

    def infer(self, task: dict[str, Any]) -> tuple[str, str, str]:
        last_error: Exception | None = None
        for model in dedupe(self.models):
            for mode in image_modes(model):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=build_messages(task, self.args, mode),
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
        raise RuntimeError(f"all models failed: {last_error!r}")


def build_messages(task: dict[str, Any], args: argparse.Namespace, mode: str) -> list[dict[str, Any]]:
    info = {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "candidate_source": (task.get("candidate_meta") or {}).get("source"),
        "caption_text": (task.get("gold") or {}).get("caption_text"),
        "image_bbox": (task.get("gold") or {}).get("image_bbox"),
        "caption_bbox": (task.get("gold") or {}).get("caption_bbox"),
    }
    prompt = PROMPT + "\n样本元数据：\n" + json.dumps(info, ensure_ascii=False, indent=2)
    if mode == "image":
        content: Any = [
            {"type": "image", "image": image_data_url(task.get("overlay_image"), args.image_max_side)},
            {"type": "image", "image": image_data_url(task.get("artwork_image"), args.crop_max_side)},
            {"type": "text", "text": prompt},
        ]
    elif mode == "image_url":
        content = [
            {"type": "image_url", "image_url": {"url": image_data_url(task.get("overlay_image"), args.image_max_side)}},
            {"type": "image_url", "image_url": {"url": image_data_url(task.get("artwork_image"), args.crop_max_side)}},
            {"type": "text", "text": prompt},
        ]
    else:
        content = prompt + "\n注意：当前未接收图片，只能依据元数据判断，应给低置信度。"
    return [{"role": "user", "content": content}]


def image_data_url(path: Any, max_side: int) -> str:
    image_path = Path(str(path))
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        scale = min(1.0, max_side / max(image.size))
        if scale < 1.0:
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))))
        tmp = image_path.with_suffix(".vlm_tmp.jpg")
        image.save(tmp, quality=88)
    try:
        data = base64.b64encode(tmp.read_bytes()).decode("ascii")
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return f"data:image/jpeg;base64,{data}"


def image_modes(model: str) -> list[str]:
    lower = model.lower()
    if "qwen3.7" in lower or "max" in lower:
        return ["image", "image_url"]
    return ["image_url", "image"]


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("VLM response is not a JSON object")
    return data


def summarize(args: argparse.Namespace, output_dir: Path, decisions: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [row for row in decisions if not row.get("error")]
    keep = [row for row in ok if (row.get("decision") or {}).get("should_keep_for_training") is True]
    valid = [row for row in ok if (row.get("decision") or {}).get("is_valid_target") is True]
    landscape = [row for row in ok if (row.get("decision") or {}).get("is_landscape_related") is True]
    caption_good = [row for row in ok if (row.get("decision") or {}).get("caption_match") == "good"]
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": args.dataset_dir,
        "output_dir": str(output_dir),
        "limit": args.limit,
        "sample_count": len(decisions),
        "ok_count": len(ok),
        "error_count": len(decisions) - len(ok),
        "keep_rate": len(keep) / max(1, len(ok)),
        "valid_target_rate": len(valid) / max(1, len(ok)),
        "landscape_related_rate": len(landscape) / max(1, len(ok)),
        "caption_good_rate": len(caption_good) / max(1, len(ok)),
        "models": dedupe([row.get("model") for row in ok if row.get("model")]),
        "errors": [row for row in decisions if row.get("error")][:10],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.1 VLM 抽检报告",
        "",
        f"生成时间：{summary['created_at']} CST",
        "",
        f"- dataset_dir：`{summary['dataset_dir']}`",
        f"- sample_count：{summary['sample_count']}",
        f"- ok_count：{summary['ok_count']}",
        f"- error_count：{summary['error_count']}",
        f"- keep_rate：{summary['keep_rate']:.4f}",
        f"- valid_target_rate：{summary['valid_target_rate']:.4f}",
        f"- landscape_related_rate：{summary['landscape_related_rate']:.4f}",
        f"- caption_good_rate：{summary['caption_good_rate']:.4f}",
        f"- models：`{json.dumps(summary['models'], ensure_ascii=False)}`",
        "",
        "说明：这是小样本 VLM 质量抽检，不是最终人工标注。若 keep_rate 或 caption_good_rate 偏低，应先做候选 rerank/VLM filter，再训练。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def dedupe(values: list[Any]) -> list[Any]:
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
