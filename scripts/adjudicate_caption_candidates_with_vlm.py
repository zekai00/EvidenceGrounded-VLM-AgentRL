#!/usr/bin/env python3
"""Adjudicate hard caption-region cases with rule candidates plus optional DashScope VLM."""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

import build_agentbench_v0_4_1_claim_schema as rules  # noqa: E402
from evidence_agent_env.data import read_jsonl, write_jsonl  # noqa: E402


DEFAULT_DATASET = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_4_1_region_claim_schema_caption_linegroup_claimfix_sft_20260601_1728"
)


PROMPT = """你是 PDF 页面图注裁决员。你会看到一张 PDF 页面图：
- 红框：目标图像。
- 黄色候选框 A/B/C/...：候选图注区域。

任务：选择真正属于红框目标图像的 caption candidate。
注意：
- 如果同一行有左右并排两个图注，只能选和红框目标图像水平对应的那个，不要把旁边图的图注合并进来。
- 如果一个图注分成 2-3 行，可以选择多个连续候选；输出的 caption_text 应合并这些行。
- 不要根据正文段落补全 caption；caption_text 必须来自候选文本。
- 如果没有可靠候选，selected_ids 为空，needs_human_review=true。

只输出 JSON：
{
  "selected_ids": ["A"],
  "caption_text": "...",
  "confidence": 0.0,
  "needs_human_review": false,
  "reason": "简短说明"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dotenv", default="/root/Workspace/VLM/EviTool-VL/.env")
    parser.add_argument("--model", default="qwen3.6-flash-2026-04-16")
    parser.add_argument(
        "--fallback-models",
        default="qwen3.7-max-2026-05-20,qwen3.7-max-2026-05-17,qwen3.7-max",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-all", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "caption_adjudication_assets"
    assets_dir.mkdir(exist_ok=True)

    tasks = read_jsonl(dataset_dir / "tasks_all.jsonl")
    selected: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    client = None if args.dry_run else DashScopeClient(args)

    for task in tasks:
        candidates = build_caption_candidates(task)
        hard = is_hard_case(task, candidates)
        if not args.force_all and not hard:
            selected.append(task)
            continue
        if args.limit and len(decisions) >= args.limit:
            selected.append(task)
            continue
        if not candidates:
            item = copy.deepcopy(task)
            item.setdefault("quality_flags", []).append("caption_no_rule_candidates")
            selected.append(item)
            decisions.append({"task_id": task["task_id"], "status": "no_candidates", "hard": hard})
            continue
        if args.dry_run:
            selected.append(task)
            decisions.append({"task_id": task["task_id"], "status": "dry_run", "hard": hard, "candidates": candidates})
            continue
        overlay = draw_candidate_overlay(task, candidates, assets_dir)
        raw = client.infer(task, candidates, overlay)
        decision = parse_json_object(raw)
        item = apply_decision(task, candidates, decision)
        selected.append(item)
        decisions.append(
            {
                "task_id": task["task_id"],
                "status": "adjudicated",
                "hard": hard,
                "model": client.last_model,
                "decision": decision,
                "candidates": candidates,
                "overlay": str(overlay),
            }
        )

    if not args.dry_run:
        write_outputs(dataset_dir, output_dir, selected)
    write_jsonl(output_dir / "caption_adjudication_decisions.jsonl", decisions)
    summary = {
        "created_at": now(),
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "dry_run": args.dry_run,
        "tasks_total": len(tasks),
        "decisions": len(decisions),
        "adjudicated": sum(row["status"] == "adjudicated" for row in decisions),
        "no_candidates": sum(row["status"] == "no_candidates" for row in decisions),
        "model": args.model,
        "fallback_models": args.fallback_models,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_caption_candidates(task: dict[str, Any]) -> list[dict[str, Any]]:
    pdf = rules.resolve_pdf_path(task)
    size = rules.image_size(task.get("page_image"))
    width = size.get("width")
    height = size.get("height")
    image_bbox = task.get("gold", {}).get("image_bbox") or task.get("gold", {}).get("target_region_bbox")
    if not pdf or not width or not height:
        return candidates_from_regions(task)
    try:
        import fitz

        with fitz.open(str(pdf)) as doc:
            page_index = int(task.get("page") or 0) - 1
            page = doc[page_index]
            blocks = rules.page_text_blocks(
                page,
                float(width) / page.rect.width,
                float(height) / page.rect.height,
                int(width),
                int(height),
            )
    except Exception:
        return candidates_from_regions(task)

    rows: list[dict[str, Any]] = []
    for idx, block in enumerate(blocks):
        text = rules.normalize_spaces(block["text"])
        score = rules.caption_image_relation_score(block["bbox"], image_bbox)
        if rules.looks_like_caption_start(text):
            score += 2.5
        if "《" in text or "图" in text or "圖" in text:
            score += 0.8
        if score < 1.2:
            continue
        rows.append({"source_index": idx, "bbox": block["bbox"], "text": text, "rule_score": round(score, 4)})
    rows.sort(key=lambda item: item["rule_score"], reverse=True)
    candidates = []
    seen = set()
    for item in rows[:8]:
        key = (tuple(item["bbox"]), item["text"])
        if key in seen:
            continue
        seen.add(key)
        candidates.append(item)
    for i, item in enumerate(candidates):
        item["candidate_id"] = chr(ord("A") + i)
    return candidates


def candidates_from_regions(task: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for region in task.get("region_candidates") or []:
        text = rules.normalize_spaces(region.get("nearby_text") or region.get("caption_hint") or "")
        if not text:
            continue
        if region.get("caption_evidence_id") or rules.looks_like_caption_start(text):
            candidates.append(
                {
                    "candidate_id": chr(ord("A") + len(candidates)),
                    "bbox": region.get("bbox"),
                    "text": text,
                    "rule_score": float(region.get("score") or 0.0),
                }
            )
    return candidates[:8]


def is_hard_case(task: dict[str, Any], candidates: list[dict[str, Any]]) -> bool:
    gold = task.get("gold") or {}
    caption = rules.normalize_spaces(gold.get("caption_text") or "")
    if not rules.looks_like_caption_start(caption):
        return True
    if not candidates:
        return True
    if gold.get("caption_bbox_coordinate") in {None, "page_pixels_from_legacy_0_1000"}:
        return True
    caption_markers = sum(1 for cand in candidates if count_caption_markers(cand.get("text") or "") >= 2)
    if caption_markers:
        return True
    best_iou = max(rules.bbox_iou(gold.get("caption_bbox"), cand.get("bbox")) for cand in candidates)
    if best_iou < 0.35:
        return True
    return False


def count_caption_markers(text: str) -> int:
    return len(re.findall(r"(?:图|圖|Figure|Fig\.?)\s*[一二三四五六七八九十百千万〇零\d]+", text, flags=re.I))


def draw_candidate_overlay(task: dict[str, Any], candidates: list[dict[str, Any]], assets_dir: Path) -> Path:
    page = Path(task["page_image"])
    out = assets_dir / f"{safe_name(task['task_id'])}_caption_candidates.jpg"
    with Image.open(page) as image:
        image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        target = task.get("gold", {}).get("image_bbox") or task.get("gold", {}).get("target_region_bbox")
        if rules.is_valid_bbox(target):
            draw_box(draw, target, (255, 40, 40), "target", 5)
        for cand in candidates:
            draw_box(draw, cand["bbox"], (245, 180, 0), cand["candidate_id"], 4)
        image.thumbnail((1400, 1800))
        image.save(out, quality=90)
    return out


def apply_decision(task: dict[str, Any], candidates: list[dict[str, Any]], decision: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(task)
    selected_ids = set(map(str, decision.get("selected_ids") or []))
    selected = [cand for cand in candidates if cand["candidate_id"] in selected_ids]
    if not selected:
        item.setdefault("quality_flags", []).append("caption_vlm_needs_human_review")
        item["caption_adjudication"] = decision
        return item
    selected.sort(key=lambda cand: (cand["bbox"][1], cand["bbox"][0]))
    text = rules.normalize_spaces(str(decision.get("caption_text") or rules.join_caption_lines([cand["text"] for cand in selected])))
    bbox = rules.union_bboxes([cand["bbox"] for cand in selected])
    if not text or not rules.is_valid_bbox(bbox):
        item.setdefault("quality_flags", []).append("caption_vlm_invalid_decision")
        item["caption_adjudication"] = decision
        return item
    evidence_id = f"local_caption_{item['task_id']}"
    item.setdefault("gold", {})["caption_text"] = text
    item["gold"]["caption_bbox"] = bbox
    item["gold"]["caption_bbox_coordinate"] = "page_pixels_vlm_adjudicated"
    item["gold"]["caption_text_source"] = "vlm_caption_adjudication"
    item["local_evidence"] = [
        {
            "evidence_id": evidence_id,
            "source_file": item.get("source_file"),
            "page_start": item.get("page"),
            "page_end": item.get("page"),
            "authority_level": "B",
            "citation_level": "page_caption_region",
            "source_quality": "vlm_adjudicated_pdf_caption_candidate",
            "display_snippet": text,
        }
    ]
    for region in item.get("region_candidates") or []:
        if region.get("caption_evidence_id"):
            region.pop("caption_evidence_id", None)
            region.pop("caption_hint", None)
    item.setdefault("region_candidates", []).append(
        {
            "bbox": bbox,
            "source": "vlm_caption_adjudicated",
            "type": "text_or_caption_candidate",
            "score": float(decision.get("confidence") or 0.0),
            "nearby_text": text,
            "hint": "VLM 裁决后的目标图注候选区域",
            "region_id": f"r_caption_vlm_{len(item.get('region_candidates') or [])}",
            "caption_evidence_id": evidence_id,
            "caption_hint": text[:160],
        }
    )
    item["caption_adjudication"] = decision
    return item


def write_outputs(source_dir: Path, output_dir: Path, tasks: list[dict[str, Any]]) -> None:
    write_jsonl(output_dir / "tasks_all.jsonl", tasks)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", [task for task in tasks if task.get("split") == split])
    # This script changes task labels only. Rebuild SFT/episodes with build_agentbench_v0_4_1_claim_schema.py
    # or build_agentbench_v0_4_2_batch_claims_sft.py after using the output tasks as a source if needed.
    manifest = {
        "created_at": now(),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "tasks_total": len(tasks),
        "files": {"tasks_all": str(output_dir / "tasks_all.jsonl")},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


class DashScopeClient:
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        load_dotenv(Path(args.dotenv))
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(f"DASHSCOPE_API_KEY is not set. Check {args.dotenv}")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.models = [args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()]
        self.last_model = ""

    def infer(self, task: dict[str, Any], candidates: list[dict[str, Any]], overlay: Path) -> str:
        payload = {
            "task_id": task.get("task_id"),
            "source_file": task.get("source_file"),
            "page": task.get("page"),
            "target_bbox": task.get("gold", {}).get("image_bbox") or task.get("gold", {}).get("target_region_bbox"),
            "current_caption_text": task.get("gold", {}).get("caption_text"),
            "candidates": candidates,
        }
        last_error: Exception | None = None
        for model in self.models:
            for mode in image_modes_for_model(model):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": build_content(overlay, payload, mode)}],
                        temperature=0,
                        max_tokens=600,
                        response_format={"type": "json_object"},
                    )
                    text = response.choices[0].message.content or ""
                    parse_json_object(text)
                    self.last_model = f"{model}/{mode}"
                    return text
                except Exception as exc:
                    last_error = exc
        raise RuntimeError(f"all caption adjudication models failed: {last_error!r}")


def image_modes_for_model(model: str) -> list[str]:
    lower = model.lower()
    if "qwen3.7-max" in lower or "qwen-max" in lower:
        return ["image", "image_url"]
    return ["image_url", "image"]


def build_content(overlay: Path, payload: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    prompt = PROMPT + "\n样本信息：\n" + json.dumps(payload, ensure_ascii=False, indent=2)
    if mode == "image_url":
        return [
            {"type": "image_url", "image_url": {"url": image_data_url(overlay)}},
            {"type": "text", "text": prompt},
        ]
    return [
        {"type": "image", "image": image_data_url(overlay)},
        {"type": "text", "text": prompt},
    ]


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("model output is not a JSON object")
    return value


def image_data_url(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def draw_box(draw: ImageDraw.ImageDraw, bbox: Any, color: tuple[int, int, int], label: str, width: int) -> None:
    if not rules.is_valid_bbox(bbox):
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    for offset in range(width):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=color)
    text_y = max(0, y1 - 16)
    draw.rectangle([x1, text_y, x1 + max(60, len(label) * 8), text_y + 16], fill=color)
    draw.text((x1 + 2, text_y + 2), label, fill=(255, 255, 255))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))[:100]


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(f"/root/datasets/evidence_grounded_vlm_agentrl/caption_adjudication_v0_4_3_{stamp}")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
