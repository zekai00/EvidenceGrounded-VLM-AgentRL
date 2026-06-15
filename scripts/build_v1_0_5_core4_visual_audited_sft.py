#!/usr/bin/env python3
"""Build v1.0.5 Core4 SFT with visual audit gates.

This builder reuses the v1.0.4 deduplicated candidate pipeline, but inserts a
VLM adjudication step before Core4 SFT rows are emitted.  The first intended
use is a 100-200 row pilot; the same output directory can then be resumed in
``--mode full`` to review all selected candidates.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import html
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_agentbench_v0_9_fixedsplit_train_multitarget as v09  # noqa: E402
import build_agentbench_v1_0_from_layout_candidates as v10  # noqa: E402
import build_gold_eval_v1_0_4 as gold_review  # noqa: E402
import build_v1_0_4_core4_clean_sft as core4  # noqa: E402
import build_v1_0_4_core4_dedup_expanded_sft as dedup  # noqa: E402
from evidence_agent_env.data import EvidenceIndex  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DEFAULT_CANDIDATE_CACHE = dedup.DEFAULT_CANDIDATE_CACHE
DEFAULT_RAW_PDF_ROOT = dedup.DEFAULT_RAW_PDF_ROOT
DEFAULT_EVIDENCE_INDEX = dedup.DEFAULT_EVIDENCE_INDEX
DEFAULT_GOLD_EVAL_DIR = dedup.DEFAULT_GOLD_EVAL_DIR
DEFAULT_DOTENV = Path("/root/Workspace/VLM/EvidenceGrounded-VLM-AgentRL/.env")
DEFAULT_MODEL = "qwen3.7-max"
DEFAULT_FALLBACK_MODELS = (
    "qwen3.7-max-2026-06-08,"
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

LANDSCAPE_POSITIVE_PATTERN = re.compile(
    r"landscape|mountain|mountains|riverbank|rivers?|stream|streams?|pine|pines|groves?|"
    r"level distance|summer mountains|wintry groves|retreat|hermitage|woods|valleys|rocks?|"
    r"山水|山|水|溪|泉|林|松|峰|壑|谷|江|河|岸|岩|石|云|雲|秋晚|春山|青山|疏林|溪山|万壑|萬壑",
    flags=re.IGNORECASE,
)
LANDSCAPE_STRONG_POSITIVE_PATTERN = re.compile(
    r"landscape|mountains?|rivers?|streams?|riverbank|groves?|level distance|summer mountains|"
    r"wintry groves|retreat|hermitage|woods and valleys|juran|"
    r"山水|山|水|溪|泉|峰|壑|谷|江|河|岸|云|雲|秋晚|春山|青山|溪山|万壑|萬壑",
    flags=re.IGNORECASE,
)
NON_LANDSCAPE_NEGATIVE_PATTERN = re.compile(
    r"musicians?|elephant|mahasattva|jataka|duke wen|horses?|bamboo|"
    r"odes of the state|seventh month|calligraphy|inscription|portrait|figure painting|"
    r"骑象|大象|乐人|樂人|竹\b|墨竹|马|馬|人物故事|本生|书法|書法|题跋|題跋|款识|款識",
    flags=re.IGNORECASE,
)

KNOWN_TRUNCATION_OR_CORRECTABLE_IDS = {
    "egva_v0_9_fixed_000011",
    "egva_v0_9_fixed_000012",
    "egva_v0_9_fixed_000031",
    "egva_v0_9_fixed_000072",
}

KNOWN_SEVERE_REGRESSION_IDS = {
    "egva_v0_9_fixed_000080",
    "egva_v0_9_fixed_000160",
    "egva_v0_9_fixed_000167",
    "egva_v0_9_fixed_000172",
    "egva_v0_9_fixed_000181",
    "egva_v0_9_fixed_000185",
    "egva_v0_9_fixed_000199",
    "egva_v0_9_fixed_000222",
    "egva_v0_9_fixed_000246",
    "egva_v0_9_fixed_000415",
    "egva_v0_9_fixed_000416",
    "egva_v0_9_fixed_000431",
    "egva_v0_9_fixed_000432",
}

VISUAL_AUDIT_PROMPT = """你是 EvidenceGrounded-VLM-AgentRL v1.0.5 visual-audited 数据集的严格视觉裁决员。

你会看到两张图：
1. PDF 页面 overlay：红框是候选目标图像区域，青色/蓝色框是候选图注区域。
2. 红框裁剪图：候选目标图像。

你的任务是判断这条候选能不能自动进入 Core4 SFT 训练集。只输出 JSON 对象，不要输出 Markdown。

必须输出字段：
{
  "target_box_ok": "yes|no|uncertain",
  "target_box_error": "none|includes_caption|includes_body_text|multi_figures|text_only|wrong_object|non_image|too_large|too_small|unclear",
  "caption_boundary": "complete|minor_truncation_correctable|truncated|contains_other_caption|wrong_caption|missing|unclear",
  "caption_target_match": "yes|no|uncertain",
  "object_domain": "landscape_painting|landscape_detail|classical_painting_unclear_landscape|non_landscape_artwork|diagram_or_chart|caption_or_table|text_only|calligraphy_or_inscription|architecture_or_object_photo|other|unclear",
  "caption_quality": "clean|minior_ocr_noise|minor_ocr_noise|ocr_noise|body_text|toc_or_index|wrong_language|unclear",
  "accept_for_sft": false,
  "needs_human_review": true,
  "corrected_caption_text": "",
  "suggested_image_bbox_page_px": null,
  "suggested_caption_bbox_page_px": null,
  "confidence": 0.0,
  "reason": "一句话说明"
}

裁决标准：
- accept_for_sft=true 只用于非常干净的训练样本：红框基本只框住目标图像，不能把图注/正文/多张图一起框入；图注必须与目标匹配；对象必须以山水、自然景观、山水画局部为主体。
- 不能因为它是古典绘画就接受；叙事人物、佛教故事、书法题跋、竹石单幅、动物/骑乘/乐人等如果不是山水景观主体，应拒绝或 needs_human_review。
- 如果红框把图像和图注一起圈起来，target_box_error 必须是 includes_caption，不能接受。
- 如果红框是整页正文、目录、表格、纯文字或没有图像，必须拒绝。
- 如果图注文字只截取了同一图注的一部分，但页面中完整图注明显可读，可以写 corrected_caption_text，并把 caption_boundary 标为 minor_truncation_correctable；否则标 truncated 并 needs_human_review=true。
- 如果 corrected_caption_text 比候选图注多出明显作者、年代、媒材、尺寸、馆藏等信息，caption_boundary 不能标为 complete。
- 如果候选图注把相邻两张图的图注拼在一起，caption_boundary 必须是 contains_other_caption；不要自动接受。
- 如果页面有两张图和两条图注，必须判断红框对应哪一条，不能把另一张图的 caption 当成本图 evidence。
- 如果红框没有圈住目标图像，而是圈住图注/正文，请把 target_box_error 标为 text_only 或 includes_body_text；不要因为 corrected_caption_text 正确就接受。
- 如果红框或青色框不准，但页面上能明确看到正确目标图像框或完整图注框，请在 suggested_image_bbox_page_px / suggested_caption_bbox_page_px 中给出页面像素坐标 [x1,y1,x2,y2]；不能确定则填 null。
- 不要凭空补作品名、作者、朝代、尺寸、馆藏；corrected_caption_text 只能来自页面可见图注。
- confidence 表示你对裁决本身的置信度，不是对作品知识的置信度。
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v1.0.5 Core4 visual-audited SFT.")
    parser.add_argument("--mode", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--candidate-cache-dir", default=str(DEFAULT_CANDIDATE_CACHE))
    parser.add_argument("--raw-pdf-root", default=str(DEFAULT_RAW_PDF_ROOT))
    parser.add_argument("--evidence-index-dir", default=str(DEFAULT_EVIDENCE_INDEX))
    parser.add_argument("--gold-eval-dir", default=str(DEFAULT_GOLD_EVAL_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pilot-size", type=int, default=160)
    parser.add_argument("--train-target", type=int, default=100000)
    parser.add_argument("--val-target", type=int, default=100)
    parser.add_argument("--test-target", type=int, default=100)
    parser.add_argument("--train-caption-cap", type=int, default=2)
    parser.add_argument("--eval-caption-cap", type=int, default=1)
    parser.add_argument("--max-doc-pages-train", type=int, default=120)
    parser.add_argument("--max-doc-pages-eval", type=int, default=40)
    parser.add_argument("--reserve-largest-docs-for-train", type=int, default=5)
    parser.add_argument("--page-dpi", type=int, default=150)
    parser.add_argument("--crop-dpi", type=int, default=200)
    parser.add_argument("--top-k-regions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument("--min-caption-score", type=float, default=-999.0)
    parser.add_argument("--max-caption-chars", type=int, default=260)
    parser.add_argument("--opencv-min-area-ratio", type=float, default=0.018)
    parser.add_argument("--opencv-max-area-ratio", type=float, default=0.65)
    parser.add_argument("--opencv-min-width-ratio", type=float, default=0.10)
    parser.add_argument("--opencv-min-height-ratio", type=float, default=0.07)
    parser.add_argument("--opencv-max-text-overlap", type=float, default=0.18)
    parser.add_argument("--opencv-min-aspect", type=float, default=0.22)
    parser.add_argument("--opencv-max-aspect", type=float, default=7.5)
    parser.add_argument("--keep-nonlandscape-pdf-image-blocks", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--candidate-filter-profile",
        choices=["strict", "visual_audited_expanded"],
        default="strict",
        help="Use visual_audited_expanded only when every selected candidate will be VLM-audited.",
    )
    parser.add_argument("--provider", choices=["dashscope", "offline"], default="dashscope")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-models", default=DEFAULT_FALLBACK_MODELS)
    parser.add_argument("--dotenv", default=str(DEFAULT_DOTENV))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--image-max-side", type=int, default=1400)
    parser.add_argument("--crop-max-side", type=int, default=900)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.78)
    parser.add_argument("--direct-corrected-caption-min-confidence", type=float, default=0.90)
    parser.add_argument("--direct-corrected-caption-min-chars", type=int, default=20)
    parser.add_argument("--corrected-caption-min-similarity", type=float, default=0.58)
    parser.add_argument("--require-located-caption-for-direct-use", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-caption-target-overlap", type=float, default=0.05)
    parser.add_argument("--review-package-rows", type=int, default=200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_review.load_dotenv(Path(args.dotenv))
    output_dir = resolve_output_dir(args)
    prepare_output_dir(output_dir, args)

    rng = random.Random(args.seed)
    candidates, scan_summary, filter_rows = dedup.collect_dedup_candidates(Path(args.candidate_cache_dir), args)
    split_docs = dedup.choose_doc_splits(candidates, args, rng)
    selected = dedup.select_candidates(candidates, split_docs, args, rng)
    write_text_json(output_dir / "_split_map.json", split_docs)
    write_jsonl(output_dir / "filter_decisions.jsonl", filter_rows)
    write_jsonl(output_dir / "selected_candidates.jsonl", candidate_rows(selected, split_docs))

    review_indices = choose_review_indices(selected, split_docs, args)
    client = make_client(args)
    index = EvidenceIndex(str(args.evidence_index_dir))
    page_cache: dict[tuple[str, int], Path] = {}
    task_cache: dict[int, dict[str, Any]] = {}
    stream_path = output_dir / "review" / "visual_audit_stream.jsonl"
    existing = load_existing_reviews(stream_path, args)
    reviewed_rows: list[dict[str, Any]] = []
    builder_errors: list[dict[str, Any]] = []

    for cursor, selected_index in enumerate(review_indices, start=1):
        candidate = selected[selected_index]
        task_id = task_id_for_index(selected_index)
        try:
            core5_task = materialize_core5_task(selected_index, candidate, output_dir, args, index, page_cache, task_cache)
            preview_task, _ = core4.transform_task(core5_task, caption_overrides={})
            row = existing.get(task_id)
            if row is None:
                row = review_one(selected_index, preview_task, candidate, client, args)
                append_jsonl(stream_path, [row])
                if args.sleep:
                    time.sleep(args.sleep)
            row = normalize_review_row(row, selected_index, preview_task, candidate, args)
            reviewed_rows.append(row)
            print(
                json.dumps(
                    {
                        "mode": args.mode,
                        "progress": f"{cursor}/{len(review_indices)}",
                        "task_id": task_id,
                        "status": row.get("visual_audit_status"),
                        "model": row.get("review_model"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            builder_errors.append(
                {
                    "selected_index": selected_index,
                    "task_id": task_id,
                    "source_file": candidate.source_file,
                    "page": candidate.page,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    tasks_by_split, episodes_by_split, sft_by_split, accepted_reviews, review_queue, rejected, sft_errors = build_outputs_from_reviews(
        selected,
        reviewed_rows,
        output_dir,
        args,
        index,
        page_cache,
        task_cache,
    )
    builder_errors.extend(sft_errors)

    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", tasks_by_split.get(split, []))
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", episodes_by_split.get(split, []))
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_by_split.get(split, []))
    all_tasks = [task for split in ["train", "val", "test"] for task in tasks_by_split.get(split, [])]
    all_episodes = [ep for split in ["train", "val", "test"] for ep in episodes_by_split.get(split, [])]
    all_sft = [row for split in ["train", "val", "test"] for row in sft_by_split.get(split, [])]
    write_jsonl(output_dir / "tasks_all.jsonl", all_tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", all_episodes)
    write_jsonl(output_dir / "sft" / "all.jsonl", all_sft)
    write_jsonl(output_dir / "review" / "visual_audit_reviewed.jsonl", reviewed_rows)
    write_jsonl(output_dir / "review" / "accepted_review.jsonl", accepted_reviews)
    write_jsonl(output_dir / "review" / "review_queue.jsonl", review_queue)
    write_jsonl(output_dir / "review" / "rejected.jsonl", rejected)
    write_jsonl(output_dir / "builder_errors.jsonl", builder_errors)

    gold_eval_summary = core4.build_gold_eval_core4(
        Path(args.gold_eval_dir), output_dir / "gold_eval", core4.load_caption_overrides(Path(args.gold_eval_dir))
    )
    summary = build_summary(
        args,
        output_dir,
        scan_summary,
        filter_rows,
        selected,
        review_indices,
        reviewed_rows,
        all_tasks,
        all_sft,
        builder_errors,
        gold_eval_summary,
    )
    write_text_json(output_dir / "manifest.json", summary)
    write_report(output_dir / "构建报告.md", summary)
    package_path = write_review_package(output_dir, reviewed_rows, args.review_package_rows)
    summary["artifacts"]["human_review_package"] = str(package_path)
    write_text_json(output_dir / "manifest.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(args.output_root) / f"agentbench_v1_0_5_core4_visual_audited_sft_{stamp}"


def prepare_output_dir(output_dir: Path, args: argparse.Namespace) -> None:
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists() and not args.resume and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} exists; use --resume or --overwrite")
    for child in ["pages", "crops", "overlays", "sft", "episodes", "review", "gold_eval"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)


def choose_review_indices(
    selected: list[v09.PageCandidate],
    split_docs: dict[str, str],
    args: argparse.Namespace,
) -> list[int]:
    all_indices = list(range(len(selected)))
    if args.mode == "full":
        return all_indices
    target = min(max(0, args.pilot_size), len(selected))
    forced_ids = KNOWN_TRUNCATION_OR_CORRECTABLE_IDS | KNOWN_SEVERE_REGRESSION_IDS
    forced = [idx for idx in all_indices if task_id_for_index(idx) in forced_ids]
    forced = forced[:target]
    chosen: list[int] = list(forced)
    chosen_set = set(chosen)
    split_targets = {
        "train": max(0, int(round(target * 0.62))),
        "val": max(0, int(round(target * 0.19))),
        "test": max(0, target),
    }
    split_targets["test"] = max(0, target - split_targets["train"] - split_targets["val"])
    counts = Counter(split_docs[selected[idx].source_file] for idx in chosen)
    by_split: dict[str, list[int]] = defaultdict(list)
    for idx in all_indices:
        if idx in chosen_set:
            continue
        by_split[split_docs[selected[idx].source_file]].append(idx)
    for split in ["train", "val", "test"]:
        bucket = sorted(by_split.get(split, []), key=lambda idx: candidate_risk_score(selected[idx]), reverse=True)
        need = max(0, split_targets[split] - counts[split])
        for idx in bucket[:need]:
            chosen.append(idx)
            chosen_set.add(idx)
    if len(chosen) < target:
        rest = [idx for idx in all_indices if idx not in chosen_set]
        rest.sort(key=lambda idx: candidate_risk_score(selected[idx]), reverse=True)
        chosen.extend(rest[: target - len(chosen)])
    return sorted(chosen[:target])


def candidate_risk_score(candidate: v09.PageCandidate) -> float:
    caption = str(candidate.caption_text or "")
    score = 0.0
    if str(candidate.target_source) == "opencv_visual_region":
        score += 3.0
    if candidate.area_ratio > 0.30:
        score += 2.5
    elif candidate.area_ratio > 0.18:
        score += 1.2
    if candidate.caption_score < 5:
        score += 1.0
    if len(caption) > 80:
        score += 1.0
    if "《" not in caption:
        score += 0.4
    if bbox_overlap_ratio(candidate.image_bbox, candidate.caption_bbox, denominator="caption") > 0.05:
        score += 4.0
    return score


def make_client(args: argparse.Namespace) -> "VisualAuditClient":
    if args.provider == "offline":
        return OfflineVisualAuditClient()
    return DashScopeVisualAuditClient(args)


class VisualAuditClient:
    def review(self, task: dict[str, Any], candidate: v09.PageCandidate) -> tuple[str, str, str]:
        raise NotImplementedError


class OfflineVisualAuditClient(VisualAuditClient):
    def review(self, task: dict[str, Any], candidate: v09.PageCandidate) -> tuple[str, str, str]:
        overlap = bbox_overlap_ratio((task.get("gold") or {}).get("image_bbox"), (task.get("gold") or {}).get("caption_bbox"), denominator="caption")
        decision = {
            "target_box_ok": "uncertain" if overlap <= 0.05 else "no",
            "target_box_error": "none" if overlap <= 0.05 else "includes_caption",
            "caption_boundary": "unclear",
            "caption_target_match": "uncertain",
            "object_domain": "unclear",
            "caption_quality": "unclear",
            "accept_for_sft": False,
            "needs_human_review": True,
            "corrected_caption_text": "",
            "confidence": 0.4,
            "reason": "offline 仅用于 smoke，不作为 visual-audited 采信依据",
        }
        return json.dumps(decision, ensure_ascii=False), "offline_rules", "offline"


class DashScopeVisualAuditClient(VisualAuditClient):
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is not set")
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1", timeout=args.request_timeout)
        self.models = dedupe_keep_order([args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()])
        self.args = args

    def review(self, task: dict[str, Any], candidate: v09.PageCandidate) -> tuple[str, str, str]:
        last_error: Exception | None = None
        for model in self.models:
            for image_mode in image_modes_for_model(model):
                try:
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=build_vlm_messages(task, candidate, self.args, image_mode),
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    content = response.choices[0].message.content or ""
                    gold_review.parse_json_object(content)
                    return content, model, image_mode
                except Exception as exc:
                    last_error = exc
                    continue
        raise RuntimeError(f"all VLM models failed: {last_error!r}")


def image_modes_for_model(model: str) -> list[str]:
    lower = model.lower()
    if "qwen3.7" in lower or "qwen-max" in lower or "max" in lower:
        return ["image", "image_url"]
    return ["image_url", "image"]


def build_vlm_messages(task: dict[str, Any], candidate: v09.PageCandidate, args: argparse.Namespace, image_mode: str) -> list[dict[str, Any]]:
    info = build_task_info(task, candidate)
    prompt = VISUAL_AUDIT_PROMPT + "\n当前候选元数据：\n" + json.dumps(info, ensure_ascii=False, indent=2)
    overlay = task.get("overlay_image")
    crop = task.get("artwork_image")
    if image_mode == "image":
        content: Any = [
            {"type": "image", "image": gold_review.image_data_url(overlay, args.image_max_side)},
            {"type": "image", "image": gold_review.image_data_url(crop, args.crop_max_side)},
            {"type": "text", "text": prompt},
        ]
    else:
        content = [
            {"type": "image_url", "image_url": {"url": gold_review.image_data_url(overlay, args.image_max_side)}},
            {"type": "image_url", "image_url": {"url": gold_review.image_data_url(crop, args.crop_max_side)}},
            {"type": "text", "text": prompt},
        ]
    return [{"role": "user", "content": content}]


def build_task_info(task: dict[str, Any], candidate: v09.PageCandidate) -> dict[str, Any]:
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
            }
        )
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "page_size_px": [candidate.page_width, candidate.page_height],
        "red_box_image_bbox_page_px": gold.get("image_bbox"),
        "cyan_caption_bbox_page_px": gold.get("caption_bbox"),
        "caption_target_overlap_ratio_by_caption": round(
            bbox_overlap_ratio(gold.get("image_bbox"), gold.get("caption_bbox"), denominator="caption"), 4
        ),
        "candidate_caption_text": gold.get("caption_text"),
        "candidate_source": (task.get("candidate_meta") or {}).get("source"),
        "candidate_area_ratio": (task.get("candidate_meta") or {}).get("area_ratio"),
        "candidate_caption_score": (task.get("candidate_meta") or {}).get("caption_score"),
        "candidate_target_variant": (task.get("candidate_meta") or {}).get("target_variant"),
        "candidate_page_size": [candidate.page_width, candidate.page_height],
        "core4_rule_claims_preview": claims,
        "hard_accept_rule": "accept only if target box excludes caption/body/multi-figures and caption belongs to the target",
    }


def review_one(
    selected_index: int,
    task: dict[str, Any],
    candidate: v09.PageCandidate,
    client: VisualAuditClient,
    args: argparse.Namespace,
) -> dict[str, Any]:
    base = {
        "selected_index": selected_index,
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "caption_text": (task.get("gold") or {}).get("caption_text"),
        "image_bbox": (task.get("gold") or {}).get("image_bbox"),
        "caption_bbox": (task.get("gold") or {}).get("caption_bbox"),
        "overlay_image": task.get("overlay_image"),
        "artwork_image": task.get("artwork_image"),
        "candidate_source": (task.get("candidate_meta") or {}).get("source"),
        "candidate_area_ratio": (task.get("candidate_meta") or {}).get("area_ratio"),
        "candidate_caption_score": (task.get("candidate_meta") or {}).get("caption_score"),
        "caption_target_overlap_ratio_by_caption": bbox_overlap_ratio(
            (task.get("gold") or {}).get("image_bbox"), (task.get("gold") or {}).get("caption_bbox"), denominator="caption"
        ),
        "reviewed_at": now_cst(),
    }
    try:
        raw, model, input_mode = client.review(task, candidate)
        parsed = gold_review.parse_json_object(raw)
        row = {
            **base,
            "ok": True,
            "review_model": model,
            "input_mode": input_mode,
            "decision": normalize_decision(parsed),
            "raw_response": raw,
        }
        return row
    except Exception as exc:
        return {
            **base,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "decision": {
                "target_box_ok": "uncertain",
                "target_box_error": "unclear",
                "caption_boundary": "unclear",
                "caption_target_match": "uncertain",
                "object_domain": "unclear",
                "caption_quality": "unclear",
                "accept_for_sft": False,
                "needs_human_review": True,
                "corrected_caption_text": "",
                "confidence": 0.0,
                "reason": "VLM visual audit failed",
            },
        }


def normalize_review_row(
    row: dict[str, Any],
    selected_index: int,
    task: dict[str, Any],
    candidate: v09.PageCandidate,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = dict(row)
    out["selected_index"] = selected_index
    out["task_id"] = task.get("task_id")
    out["split"] = task.get("split")
    out["source_file"] = task.get("source_file")
    out["page"] = task.get("page")
    out["overlay_image"] = task.get("overlay_image")
    out["artwork_image"] = task.get("artwork_image")
    out["caption_text"] = (task.get("gold") or {}).get("caption_text")
    out["image_bbox"] = (task.get("gold") or {}).get("image_bbox")
    out["caption_bbox"] = (task.get("gold") or {}).get("caption_bbox")
    out["candidate_source"] = (task.get("candidate_meta") or {}).get("source")
    out["candidate_area_ratio"] = (task.get("candidate_meta") or {}).get("area_ratio")
    out["candidate_caption_score"] = (task.get("candidate_meta") or {}).get("caption_score")
    out["caption_target_overlap_ratio_by_caption"] = bbox_overlap_ratio(
        (task.get("gold") or {}).get("image_bbox"), (task.get("gold") or {}).get("caption_bbox"), denominator="caption"
    )
    out["decision"] = normalize_decision(out.get("decision") or {})
    status, flags = classify_visual_decision(out, task, args)
    out["visual_audit_status"] = status
    out["strict_gate_flags"] = flags
    out["known_regression_bucket"] = known_regression_bucket(str(out.get("task_id") or ""))
    return out


def normalize_decision(decision: dict[str, Any]) -> dict[str, Any]:
    result = dict(decision or {})
    result["target_box_ok"] = normalize_enum(result.get("target_box_ok"), {"yes", "no", "uncertain"}, "uncertain")
    result["target_box_error"] = normalize_enum(
        result.get("target_box_error"),
        {"none", "includes_caption", "includes_body_text", "multi_figures", "text_only", "wrong_object", "non_image", "too_large", "too_small", "unclear"},
        "unclear",
    )
    result["caption_boundary"] = normalize_enum(
        result.get("caption_boundary"),
        {"complete", "minor_truncation_correctable", "truncated", "contains_other_caption", "wrong_caption", "missing", "unclear"},
        "unclear",
    )
    result["caption_target_match"] = normalize_enum(result.get("caption_target_match"), {"yes", "no", "uncertain"}, "uncertain")
    result["object_domain"] = normalize_enum(
        result.get("object_domain"),
        {
            "landscape_painting",
            "landscape_detail",
            "classical_painting_unclear_landscape",
            "non_landscape_artwork",
            "diagram_or_chart",
            "caption_or_table",
            "text_only",
            "calligraphy_or_inscription",
            "architecture_or_object_photo",
            "other",
            "unclear",
        },
        "unclear",
    )
    result["caption_quality"] = normalize_enum(
        result.get("caption_quality"),
        {"clean", "minior_ocr_noise", "minor_ocr_noise", "ocr_noise", "body_text", "toc_or_index", "wrong_language", "unclear"},
        "unclear",
    )
    if result["caption_quality"] == "minior_ocr_noise":
        result["caption_quality"] = "minor_ocr_noise"
    result["accept_for_sft"] = coerce_bool(result.get("accept_for_sft"))
    result["needs_human_review"] = coerce_bool(result.get("needs_human_review"))
    result["corrected_caption_text"] = normalize_space(result.get("corrected_caption_text"))[:500]
    result["suggested_image_bbox_page_px"] = normalize_bbox_value(
        result.get("suggested_image_bbox_page_px")
        or result.get("suggested_image_bbox")
        or result.get("suggested_image_bbox_0_1000")
    )
    result["suggested_caption_bbox_page_px"] = normalize_bbox_value(
        result.get("suggested_caption_bbox_page_px")
        or result.get("suggested_caption_bbox")
        or result.get("suggested_caption_bbox_0_1000")
    )
    result["confidence"] = max(0.0, min(1.0, safe_float(result.get("confidence"), 0.0)))
    result["reason"] = str(result.get("reason") or "")[:500]
    return result


def classify_visual_decision(row: dict[str, Any], task: dict[str, Any], args: argparse.Namespace) -> tuple[str, list[str]]:
    flags: list[str] = []
    decision = row.get("decision") or {}
    task_id = str(task.get("task_id") or row.get("task_id") or "")
    known_bucket = known_regression_bucket(task_id)
    direct_corrected_caption = can_use_vlm_corrected_caption_directly(row, task, args)
    has_suggested_image = bool(decision.get("suggested_image_bbox_page_px"))
    if known_bucket:
        flags.append("known_user_review_case")
    if not row.get("ok"):
        flags.append("vlm_audit_failed")
        return "needs_human_review", flags
    if safe_float(decision.get("confidence"), 0.0) < args.min_confidence:
        flags.append("confidence_below_threshold")
    gold = task.get("gold") or {}
    overlap_image_bbox = decision.get("suggested_image_bbox_page_px") or gold.get("image_bbox")
    overlap_caption_bbox = decision.get("suggested_caption_bbox_page_px") or gold.get("caption_bbox")
    overlap = bbox_overlap_ratio(overlap_image_bbox, overlap_caption_bbox, denominator="caption")
    if overlap > args.max_caption_target_overlap:
        flags.append("caption_bbox_overlaps_target_bbox")
    fatal_target_errors = {"includes_caption", "includes_body_text", "multi_figures", "text_only", "wrong_object", "non_image", "too_large", "too_small"}
    if decision.get("target_box_ok") != "yes" and not has_suggested_image:
        flags.append("target_box_not_yes")
    if decision.get("target_box_error") in fatal_target_errors and not has_suggested_image:
        flags.append("target_box_error_" + str(decision.get("target_box_error")))
    elif decision.get("target_box_error") in fatal_target_errors and has_suggested_image:
        flags.append("target_box_repaired_by_vlm_direct")
    if decision.get("caption_target_match") != "yes":
        flags.append("caption_target_match_not_yes")
    allowed_domains = {"landscape_painting", "landscape_detail"}
    if decision.get("object_domain") not in allowed_domains:
        flags.append("object_domain_not_allowed")
    bad_caption_quality = {"body_text", "toc_or_index", "wrong_language"}
    if decision.get("caption_quality") in bad_caption_quality:
        flags.append("caption_quality_bad")
    boundary = str(decision.get("caption_boundary") or "")
    corrected = str(decision.get("corrected_caption_text") or "").strip()
    original_caption = normalize_space((task.get("gold") or {}).get("caption_text") or row.get("caption_text"))
    if caption_has_non_landscape_risk(original_caption, decision):
        flags.append("caption_non_landscape_risk")
    if significant_caption_correction(original_caption, corrected):
        flags.append("vlm_corrected_caption_differs_from_candidate")
        if boundary == "complete" and not direct_corrected_caption:
            flags.append("caption_boundary_inconsistent_with_correction")
    if boundary == "complete":
        pass
    elif boundary == "minor_truncation_correctable" and corrected:
        flags.append("caption_corrected_by_vlm")
    elif direct_corrected_caption:
        flags.append("caption_corrected_by_vlm_direct")
    else:
        flags.append("caption_boundary_not_auto_clean")
    if decision.get("needs_human_review") is True and not direct_corrected_caption:
        flags.append("vlm_requested_human_review")
    if decision.get("accept_for_sft") is not True and not direct_corrected_caption:
        flags.append("vlm_did_not_accept_for_sft")

    if any(flag.startswith("target_box_error_") for flag in flags):
        return "rejected", flags
    if "caption_quality_bad" in flags or decision.get("object_domain") in {"diagram_or_chart", "caption_or_table", "text_only"}:
        return "rejected", flags
    non_blocking = {"caption_corrected_by_vlm", "caption_corrected_by_vlm_direct", "target_box_repaired_by_vlm_direct"}
    if direct_corrected_caption:
        non_blocking.add("vlm_corrected_caption_differs_from_candidate")
    if known_bucket:
        return "needs_human_review", flags
    if not [flag for flag in flags if flag not in non_blocking]:
        return "accepted_sft", flags
    return "needs_human_review", flags


def significant_caption_correction(original: str, corrected: str) -> bool:
    original = normalize_space(original)
    corrected = normalize_space(corrected)
    if not corrected or not original or corrected == original:
        return False
    compact_original = re.sub(r"\s+", "", original).casefold()
    compact_corrected = re.sub(r"\s+", "", corrected).casefold()
    if compact_original and compact_original in compact_corrected and len(compact_corrected) >= len(compact_original) + 12:
        return True
    return abs(len(compact_corrected) - len(compact_original)) >= 18


def caption_has_non_landscape_risk(caption: str, decision: dict[str, Any]) -> bool:
    text = normalize_space(caption)
    if not text:
        return False
    if not NON_LANDSCAPE_NEGATIVE_PATTERN.search(text):
        return False
    if LANDSCAPE_STRONG_POSITIVE_PATTERN.search(text):
        return False
    return str(decision.get("object_domain") or "") != "landscape_detail"


def can_use_vlm_corrected_caption_directly(row: dict[str, Any], task: dict[str, Any], args: argparse.Namespace) -> bool:
    if not row.get("ok"):
        return False
    decision = row.get("decision") or {}
    corrected = normalize_space(decision.get("corrected_caption_text"))
    original = normalize_space((task.get("gold") or {}).get("caption_text") or row.get("caption_text"))
    if len(corrected) < int(args.direct_corrected_caption_min_chars):
        return False
    if safe_float(decision.get("confidence"), 0.0) < float(args.direct_corrected_caption_min_confidence):
        return False
    if decision.get("target_box_ok") != "yes" and not decision.get("suggested_image_bbox_page_px"):
        return False
    if decision.get("target_box_error") not in {"none", "unclear"} and not decision.get("suggested_image_bbox_page_px"):
        return False
    if decision.get("caption_target_match") != "yes":
        return False
    if decision.get("object_domain") not in {"landscape_painting", "landscape_detail"}:
        return False
    if decision.get("caption_quality") not in {"clean", "minor_ocr_noise", "ocr_noise"}:
        return False
    if decision.get("caption_boundary") in {"contains_other_caption", "wrong_caption", "missing"}:
        return False
    gold = task.get("gold") or {}
    overlap_image_bbox = decision.get("suggested_image_bbox_page_px") or gold.get("image_bbox")
    overlap_caption_bbox = decision.get("suggested_caption_bbox_page_px") or gold.get("caption_bbox")
    if bbox_overlap_ratio(overlap_image_bbox, overlap_caption_bbox, denominator="caption") > args.max_caption_target_overlap:
        return False
    if caption_has_non_landscape_risk(corrected, decision):
        return False
    return corrected_caption_matches_original_marker(original, corrected)


def corrected_caption_matches_original_marker(original: str, corrected: str) -> bool:
    original_norm = fuzzy_caption_norm(original)
    corrected_norm = fuzzy_caption_norm(corrected)
    if not original_norm or not corrected_norm:
        return False
    if original_norm in corrected_norm:
        return True
    marker = extract_caption_marker(original)
    if marker and fuzzy_caption_norm(marker) in corrected_norm:
        return True
    if len(original_norm) >= 24:
        return difflib.SequenceMatcher(None, original_norm, corrected_norm).ratio() >= 0.62
    return False


def extract_caption_marker(text: str) -> str:
    text = normalize_space(text)
    patterns = [
        r"(?:Figure|Fig\.?|Plate|Pl\.?)\s*[0-9]+(?:[.\-][0-9]+)?[a-z]?",
        r"(?:图|圖)\s*[0-9一二三四五六七八九十百]+(?:[.\-．、][0-9一二三四五六七八九十百]+)?[a-z]?",
        r"〔图[一二三四五六七八九十百]+〕",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0)
    return ""


def apply_vlm_decision_candidate_overrides(
    candidate: v09.PageCandidate, row: dict[str, Any], args: argparse.Namespace
) -> tuple[v09.PageCandidate, dict[str, Any]]:
    decision = row.get("decision") or {}
    meta: dict[str, Any] = {"applied": False, "changes": []}
    image_bbox = list(candidate.image_bbox)
    caption_bbox = list(candidate.caption_bbox or [])
    caption_text = normalize_space(candidate.caption_text)

    suggested_image_bbox = decision.get("suggested_image_bbox_page_px")
    if valid_page_bbox(suggested_image_bbox, candidate, min_width=24, min_height=24):
        image_bbox = [int(v) for v in suggested_image_bbox]
        meta["applied"] = True
        meta["changes"].append("suggested_image_bbox_page_px")
    elif suggested_image_bbox:
        meta["invalid_suggested_image_bbox_page_px"] = suggested_image_bbox

    corrected = normalize_space(decision.get("corrected_caption_text"))
    task_stub = {"gold": {"image_bbox": image_bbox, "caption_bbox": caption_bbox, "caption_text": caption_text}}
    direct_caption = can_use_vlm_corrected_caption_directly(row, task_stub, args)
    if direct_caption:
        located_text, located_bbox, locate_meta = locate_corrected_caption_bbox(candidate, corrected, args)
        meta["caption_locate"] = locate_meta
        if located_bbox:
            caption_bbox = [int(v) for v in located_bbox]
            caption_text = corrected
            meta["applied"] = True
            meta["changes"].extend(["corrected_caption_text", "located_caption_bbox"])
            meta["page_text_aligned_caption"] = located_text
        else:
            suggested_caption_bbox = decision.get("suggested_caption_bbox_page_px")
            if valid_page_bbox(suggested_caption_bbox, candidate, min_width=12, min_height=8):
                caption_bbox = [int(v) for v in suggested_caption_bbox]
                caption_text = corrected
                meta["applied"] = True
                meta["changes"].extend(["corrected_caption_text", "suggested_caption_bbox_page_px"])
            elif args.require_located_caption_for_direct_use:
                return candidate, {
                    **meta,
                    "required_override_failed": True,
                    "skip_reason": "corrected_caption_not_located_in_page_text",
                    "corrected_caption_text": corrected,
                }

    if caption_bbox and bbox_overlap_ratio(image_bbox, caption_bbox, denominator="caption") > args.max_caption_target_overlap:
        return candidate, {
            **meta,
            "required_override_failed": True,
            "skip_reason": "image_caption_overlap_after_vlm_override",
            "image_bbox": image_bbox,
            "caption_bbox": caption_bbox,
        }

    if not meta["applied"]:
        return candidate, meta
    area_ratio = bbox_area(image_bbox) / max(1.0, float(candidate.page_width * candidate.page_height))
    return (
        replace(
            candidate,
            image_bbox=image_bbox,
            caption_bbox=caption_bbox or candidate.caption_bbox,
            caption_text=caption_text,
            area_ratio=round(area_ratio, 6),
            caption_score=max(float(candidate.caption_score or 0.0), 9.0 if direct_caption else float(candidate.caption_score or 0.0)),
            target_source=str(candidate.target_source or "") + "_vlm_decision_override",
        ),
        meta,
    )


def locate_corrected_caption_bbox(
    candidate: v09.PageCandidate,
    corrected_caption: str,
    args: argparse.Namespace,
) -> tuple[str, list[int] | None, dict[str, Any]]:
    if not candidate.caption_bbox:
        return "", None, {"method": "corrected_caption_fuzzy_prefix", "lines": []}
    cx1, cy1, cx2, _cy2 = [int(v) for v in candidate.caption_bbox]
    cap_width = max(1, cx2 - cx1)
    blocks = sorted((dict(block) for block in candidate.text_blocks), key=lambda item: (bbox(item)[1], bbox(item)[0]))
    lines: list[dict[str, Any]] = []
    for block in blocks:
        bb = bbox(block)
        text = normalize_space(block.get("text"))
        if not text:
            continue
        if bb[1] < cy1 - 20 or bb[1] > cy1 + 360:
            continue
        if same_caption_column(candidate.caption_bbox, bb, cap_width):
            lines.append({"bbox": bb, "text": text})
    if not lines:
        return "", None, {"method": "corrected_caption_fuzzy_prefix", "lines": []}
    start = min(range(len(lines)), key=lambda i: abs(lines[i]["bbox"][1] - cy1) + abs(lines[i]["bbox"][0] - cx1))
    prefix_lines: list[dict[str, Any]] = []
    prev_bottom = None
    for item in lines[start:]:
        if prefix_lines and is_caption_start(item["text"]):
            break
        if prev_bottom is not None and item["bbox"][1] - prev_bottom > 58:
            break
        prefix_lines.append(item)
        prev_bottom = item["bbox"][3]
        if len(prefix_lines) >= 8:
            break
    if not prefix_lines:
        return "", None, {"method": "corrected_caption_fuzzy_prefix", "lines": lines[:8], "skip_reason": "no_prefix_lines"}
    best_score = -1.0
    best_count = 0
    best_text = ""
    corrected_norm = fuzzy_caption_norm(corrected_caption)
    for count in range(1, len(prefix_lines) + 1):
        text = normalize_space(" ".join(line["text"] for line in prefix_lines[:count]))
        score = caption_similarity(text, corrected_caption)
        norm_text = fuzzy_caption_norm(text)
        if corrected_norm and (corrected_norm in norm_text or norm_text in corrected_norm):
            score += 0.12
        if score > best_score:
            best_score = score
            best_count = count
            best_text = text
    if best_score < args.corrected_caption_min_similarity:
        return "", None, {
            "method": "corrected_caption_fuzzy_prefix",
            "lines": prefix_lines,
            "best_score": round(best_score, 4),
            "best_text": best_text,
            "skip_reason": "similarity_below_threshold",
        }
    chosen = prefix_lines[:best_count]
    return best_text, union_bbox([line["bbox"] for line in chosen]), {
        "method": "corrected_caption_fuzzy_prefix",
        "lines": chosen,
        "line_count": len(chosen),
        "best_score": round(best_score, 4),
    }


def same_caption_column(caption_bbox: list[int], block_bbox: list[int], cap_width: int) -> bool:
    cx1, _, cx2, _ = caption_bbox
    bx1, _, bx2, _ = block_bbox
    overlap = max(0, min(cx2, bx2) - max(cx1, bx1))
    overlap_ratio = overlap / max(1, min(cx2 - cx1, bx2 - bx1))
    left_close = abs(bx1 - cx1) <= max(70, int(cap_width * 0.35))
    center_close = abs(((bx1 + bx2) / 2) - ((cx1 + cx2) / 2)) <= max(90, int(cap_width * 0.45))
    return overlap_ratio >= 0.35 or (left_close and center_close)


def is_caption_start(text: str) -> bool:
    return bool(extract_caption_marker(text))


def caption_similarity(left: str, right: str) -> float:
    a = fuzzy_caption_norm(left)
    b = fuzzy_caption_norm(right)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def fuzzy_caption_norm(text: str) -> str:
    text = normalize_space(text).casefold()
    text = text.replace("圖", "图")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("．", ".").replace("·", ".")
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("em", "cm")
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text)


def build_outputs_from_reviews(
    selected: list[v09.PageCandidate],
    reviewed_rows: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    index: EvidenceIndex,
    page_cache: dict[tuple[str, int], Path],
    task_cache: dict[int, dict[str, Any]],
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    tasks_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episodes_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sft_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    accepted_reviews: list[dict[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for row in sorted(reviewed_rows, key=lambda item: int(item.get("selected_index") or 0)):
        status = str(row.get("visual_audit_status") or "")
        selected_index = int(row.get("selected_index") or 0)
        if status != "accepted_sft":
            queue_row = compact_review_row(row)
            if status == "rejected":
                rejected.append(queue_row)
            else:
                review_queue.append(queue_row)
            continue
        try:
            candidate = selected[selected_index]
            candidate_for_task, override_meta = apply_vlm_decision_candidate_overrides(candidate, row, args)
            if override_meta.get("required_override_failed"):
                queue_row = compact_review_row(row)
                queue_row["candidate_override_meta"] = override_meta
                review_queue.append(queue_row)
                continue
            if candidate_for_task != candidate:
                task_cache.pop(selected_index, None)
            core5_task = materialize_core5_task(selected_index, candidate_for_task, output_dir, args, index, page_cache, task_cache)
            override = caption_override_from_decision(core5_task, row, args)
            transformed, reviews = core4.transform_task(core5_task, caption_overrides=override)
            transformed["dataset_version"] = "v1.0.5_core4_visual_audited_sft"
            transformed["runtime_mode"] = "v1_0_5_visual_audited_core4_docsplit_caption_cap"
            transformed["tool_schema_version"] = "v1.0.5_no_select_core4_visual_audited"
            transformed.setdefault("candidate_meta", {})["visual_audit_status"] = status
            transformed.setdefault("candidate_meta", {})["visual_audit_model"] = row.get("review_model")
            transformed.setdefault("candidate_meta", {})["dedup_caption_key"] = dedup.caption_key((transformed.get("gold") or {}).get("caption_text") or "")
            transformed.setdefault("gold", {})["label_source"] = "v1_0_5_visual_audited_core4_sft"
            transformed.setdefault("gold", {})["needs_review"] = False
            transformed.setdefault("gold", {})["visual_audited"] = True
            transformed["visual_audit"] = {
                "version": "v1.0.5",
                "status": status,
                "review_model": row.get("review_model"),
                "input_mode": row.get("input_mode"),
                "reviewed_at": row.get("reviewed_at"),
                "decision": row.get("decision") or {},
                "strict_gate_flags": row.get("strict_gate_flags") or [],
                "caption_override_applied": bool(override),
                "candidate_override_meta": override_meta,
                "original_caption_text": row.get("caption_text"),
                "known_regression_bucket": row.get("known_regression_bucket"),
            }
            replay = dedup.replay_from_task(transformed)
            actions = core4.build_oracle_actions(transformed, replay)
            sft_rows = core4.build_sft_rows(transformed, actions, replay)
            for sft_row in sft_rows:
                sft_row["label_source"] = "v1_0_5_core4_visual_audited_sft"
                sft_row["tool_schema_version"] = "v1.0.5_no_select_core4_visual_audited"
            split = str(transformed.get("split") or "train")
            episode = {
                "task_id": transformed["task_id"],
                "source_task_id": transformed.get("source_task_id"),
                "split": split,
                "variant": transformed.get("candidate_augmentation") or 0,
                "actions": actions,
            }
            tasks_by_split[split].append(transformed)
            episodes_by_split[split].append(episode)
            sft_by_split[split].extend(sft_rows)
            accepted = compact_review_row(row)
            accepted["caption_override_applied"] = bool(override)
            accepted["candidate_override_meta"] = override_meta
            accepted["task_review_rows_from_core4_transform"] = reviews
            accepted_reviews.append(accepted)
        except Exception as exc:
            errors.append(
                {
                    "selected_index": selected_index,
                    "task_id": row.get("task_id"),
                    "source_file": row.get("source_file"),
                    "page": row.get("page"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return tasks_by_split, episodes_by_split, sft_by_split, accepted_reviews, review_queue, rejected, errors


def caption_override_from_decision(task: dict[str, Any], row: dict[str, Any], args: argparse.Namespace) -> dict[str, str]:
    decision = row.get("decision") or {}
    corrected = normalize_space(decision.get("corrected_caption_text"))
    original = normalize_space((task.get("gold") or {}).get("caption_text"))
    if not corrected or corrected == original:
        return {}
    if can_use_vlm_corrected_caption_directly(row, task, args):
        return {str(task.get("task_id")): corrected}
    if decision.get("caption_boundary") == "minor_truncation_correctable":
        if len(corrected) < max(8, len(original) // 2):
            return {}
        return {str(task.get("task_id")): corrected}
    if significant_caption_correction(original, corrected):
        return {}
    if (row.get("decision") or {}).get("caption_quality") not in {"clean", "minor_ocr_noise", "ocr_noise"}:
        return {}
    return {str(task.get("task_id")): corrected}


def materialize_core5_task(
    selected_index: int,
    candidate: v09.PageCandidate,
    output_dir: Path,
    args: argparse.Namespace,
    index: EvidenceIndex,
    page_cache: dict[tuple[str, int], Path],
    task_cache: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    if selected_index in task_cache:
        return task_cache[selected_index]
    build_args = SimpleNamespace(page_dpi=args.page_dpi, crop_dpi=args.crop_dpi, top_k_regions=args.top_k_regions)
    task = v09.build_task(selected_index, candidate, output_dir, build_args, index, page_cache)
    v10.normalize_v1_task(task)
    task_cache[selected_index] = task
    return task


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    scan_summary: dict[str, Any],
    filter_rows: list[dict[str, Any]],
    selected: list[v09.PageCandidate],
    review_indices: list[int],
    reviewed_rows: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    gold_eval_summary: dict[str, Any],
) -> dict[str, Any]:
    status_counts = Counter(row.get("visual_audit_status") for row in reviewed_rows)
    decision_counts = {
        "target_box_error": dict(Counter((row.get("decision") or {}).get("target_box_error") for row in reviewed_rows).most_common()),
        "caption_boundary": dict(Counter((row.get("decision") or {}).get("caption_boundary") for row in reviewed_rows).most_common()),
        "object_domain": dict(Counter((row.get("decision") or {}).get("object_domain") for row in reviewed_rows).most_common()),
        "caption_quality": dict(Counter((row.get("decision") or {}).get("caption_quality") for row in reviewed_rows).most_common()),
    }
    split_counts = Counter(task.get("split") for task in tasks)
    sft_split_counts = Counter(row.get("split") for row in sft_rows)
    sft_action_counts = Counter((row.get("action") or {}).get("action") for row in sft_rows)
    caption_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    docs_by_split: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        split = str(task.get("split") or "")
        caption_counts_by_split[split][dedup.caption_key((task.get("gold") or {}).get("caption_text") or "")] += 1
        docs_by_split[split].add(str(task.get("source_file") or ""))
    cap_violations = {
        split: {
            key: count
            for key, count in counts.items()
            if count > (args.train_caption_cap if split == "train" else args.eval_caption_cap)
        }
        for split, counts in caption_counts_by_split.items()
    }
    cap_violations = {split: vals for split, vals in cap_violations.items() if vals}
    correctable_reviewed = [row for row in reviewed_rows if row.get("task_id") in KNOWN_TRUNCATION_OR_CORRECTABLE_IDS]
    severe_reviewed = [row for row in reviewed_rows if row.get("task_id") in KNOWN_SEVERE_REGRESSION_IDS]
    correctable_accepted = [row for row in correctable_reviewed if row.get("visual_audit_status") == "accepted_sft"]
    severe_accepted = [row for row in severe_reviewed if row.get("visual_audit_status") == "accepted_sft"]
    vlm_failed = [row for row in reviewed_rows if not row.get("ok")]
    known_regression_pass = len(correctable_accepted) == 0 and len(severe_accepted) == 0
    pilot_pass = args.mode == "pilot" and known_regression_pass and len(vlm_failed) / max(1, len(reviewed_rows)) <= 0.05
    summary = {
        "created_at": now_cst(),
        "dataset_version": "v1.0.5_core4_visual_audited_sft",
        "builder": "scripts/build_v1_0_5_core4_visual_audited_sft.py",
        "mode": args.mode,
        "output_dir": str(output_dir),
        "candidate_cache_dir": args.candidate_cache_dir,
        "raw_pdf_root": args.raw_pdf_root,
        "evidence_index_dir": args.evidence_index_dir,
        "gold_eval_dir": args.gold_eval_dir,
        "provider": args.provider,
        "model": args.model if args.provider == "dashscope" else "offline_rules",
        "fallback_models": [item.strip() for item in args.fallback_models.split(",") if item.strip()],
        "caption_cap_policy": {"train_caption_cap": args.train_caption_cap, "eval_caption_cap": args.eval_caption_cap},
        "visual_accept_policy": {
            "min_confidence": args.min_confidence,
            "max_caption_target_overlap": args.max_caption_target_overlap,
            "allowed_object_domains": ["landscape_painting", "landscape_detail"],
        },
        "args": vars(args),
        "scan_summary": scan_summary,
        "filter_summary": {
            "decisions": len(filter_rows),
            "accepted_by_rule": sum(1 for row in filter_rows if row.get("keep")),
            "rejected_by_rule": sum(1 for row in filter_rows if not row.get("keep")),
            "primary_reject_reasons": dict(Counter(row.get("primary_reason") for row in filter_rows if not row.get("keep")).most_common(20)),
        },
        "selected_candidates": len(selected),
        "review_scope": {"indices": len(review_indices), "mode": args.mode, "pilot_size": args.pilot_size},
        "reviewed_rows": len(reviewed_rows),
        "visual_audit_status_counts": dict(status_counts),
        "decision_counts": decision_counts,
        "strict_gate_flag_counts": dict(Counter(flag for row in reviewed_rows for flag in (row.get("strict_gate_flags") or [])).most_common(30)),
        "review_model_counts": dict(Counter(row.get("review_model") or "unknown" for row in reviewed_rows).most_common()),
        "known_regression_check": {
            "correctable_ids_reviewed": sorted(row.get("task_id") for row in correctable_reviewed),
            "correctable_ids_auto_accepted": sorted(row.get("task_id") for row in correctable_accepted),
            "severe_ids_reviewed": sorted(row.get("task_id") for row in severe_reviewed),
            "severe_ids_auto_accepted": sorted(row.get("task_id") for row in severe_accepted),
            "known_regression_pass": known_regression_pass,
            "pilot_pass_for_full_build": pilot_pass,
        },
        "split_counts": dict(split_counts),
        "doc_counts_by_split": {split: len(vals) for split, vals in docs_by_split.items()},
        "unique_caption_by_split": {split: len(vals) for split, vals in caption_counts_by_split.items()},
        "caption_cap_violations": cap_violations,
        "sft_rows_total": len(sft_rows),
        "sft_split_counts": dict(sft_split_counts),
        "sft_action_counts": dict(sft_action_counts),
        "field_counts": core4.field_summary(tasks),
        "builder_error_count": len(errors),
        "builder_errors_preview": errors[:20],
        "gold_eval_core4": gold_eval_summary,
        "artifacts": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "sft_all": str(output_dir / "sft" / "all.jsonl"),
            "visual_audit_stream": str(output_dir / "review" / "visual_audit_stream.jsonl"),
            "visual_audit_reviewed": str(output_dir / "review" / "visual_audit_reviewed.jsonl"),
            "review_queue": str(output_dir / "review" / "review_queue.jsonl"),
            "rejected": str(output_dir / "review" / "rejected.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    mode_label = "试点" if summary.get("mode") == "pilot" else "全量"
    lines = [
        f"# v1.0.5 Core4 Visual-Audited SFT {mode_label}构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 构建模式：`{summary['mode']}`",
        f"- 候选缓存：`{summary['candidate_cache_dir']}`",
        f"- VLM provider/model：`{summary['provider']}` / `{summary['model']}`",
        "",
        "## 结论",
        "",
        "- 本轮从 v1.0.4 去重扩量候选管线重新生成 overlay/crop，并在进入 Core4 SFT 前增加 VLM 视觉裁决。",
        "- 自动进入 SFT 的样本必须同时满足：红框目标干净、图注边界干净或可由 VLM 从页面轻微补全、图注匹配目标、对象属于山水画/山水细部/古典绘画不明显非山水、caption bbox 不明显落入 target bbox。",
        f"- caption cap：train 每个 caption 最多 {summary['caption_cap_policy']['train_caption_cap']} 条，val/test 最多 {summary['caption_cap_policy']['eval_caption_cap']} 条。",
        "",
        "## 规模",
        "",
        f"- 原始候选页：{summary['scan_summary'].get('page_rows')}",
        f"- 原始候选框：{summary['scan_summary'].get('figure_rows')}",
        f"- 规则预筛接受候选：{summary['filter_summary'].get('accepted_by_rule')}",
        f"- selected candidates：{summary['selected_candidates']}",
        f"- 本轮 VLM reviewed：{summary['reviewed_rows']}",
        f"- visual_audit_status_counts：`{json.dumps(summary['visual_audit_status_counts'], ensure_ascii=False)}`",
        f"- split task 数：`{json.dumps(summary['split_counts'], ensure_ascii=False)}`",
        f"- split unique caption：`{json.dumps(summary['unique_caption_by_split'], ensure_ascii=False)}`",
        f"- SFT rows：`{json.dumps(summary['sft_split_counts'], ensure_ascii=False)}`，total={summary['sft_rows_total']}",
        f"- action rows：`{json.dumps(summary['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 视觉裁决分布",
        "",
        f"- target_box_error：`{json.dumps(summary['decision_counts'].get('target_box_error'), ensure_ascii=False)}`",
        f"- caption_boundary：`{json.dumps(summary['decision_counts'].get('caption_boundary'), ensure_ascii=False)}`",
        f"- object_domain：`{json.dumps(summary['decision_counts'].get('object_domain'), ensure_ascii=False)}`",
        f"- strict_gate_flag_counts：`{json.dumps(summary['strict_gate_flag_counts'], ensure_ascii=False)}`",
        f"- review_model_counts：`{json.dumps(summary['review_model_counts'], ensure_ascii=False)}`",
        "",
        "## 已知错误回归检查",
        "",
        f"- severe_ids_reviewed：`{json.dumps(summary['known_regression_check']['severe_ids_reviewed'], ensure_ascii=False)}`",
        f"- correctable_ids_auto_accepted：`{json.dumps(summary['known_regression_check']['correctable_ids_auto_accepted'], ensure_ascii=False)}`",
        f"- severe_ids_auto_accepted：`{json.dumps(summary['known_regression_check']['severe_ids_auto_accepted'], ensure_ascii=False)}`",
        f"- known_regression_pass：`{summary['known_regression_check']['known_regression_pass']}`",
        "",
        "## 字段分布",
        "",
        f"- field_counts：`{json.dumps(summary['field_counts'], ensure_ascii=False)}`",
        "",
        "## 产物",
        "",
    ]
    for key, value in summary["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## 使用建议",
            "",
        ]
    )
    if summary.get("mode") == "pilot":
        lines.append(f"- pilot_pass_for_full_build：`{summary['known_regression_check']['pilot_pass_for_full_build']}`")
        if summary["known_regression_check"]["pilot_pass_for_full_build"]:
            lines.append("- 试点没有出现已知 severe 错误被自动接受，API 错误率也在阈值内；可以用同一 output_dir 加 `--mode full --resume` 继续全量视觉审计。")
        else:
            lines.append("- 试点未通过 full-build 自动放大条件；先检查 `review_queue/rejected` 和人工复核包，再修 prompt 或硬门控。")
    else:
        lines.append("- 该版本可作为 fresh Core4 SFT 的 visual-audited 训练候选；模型选择仍应看 corrected GoldEval val50，test_gold_100 保留最终确认。")
    lines.extend(
        [
            "",
            "## 风险",
            "",
            "- 这仍是 VLM 审计 silver，不等于人工 gold；关键实验结论仍需要 GoldEval 或人工抽检支撑。",
            "- VLM 对小字、多栏图注、低分辨率扫描页可能误判，因此 review package 仍应抽查。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review_package(output_dir: Path, reviewed_rows: list[dict[str, Any]], max_rows: int) -> Path:
    package_dir = output_dir / "review" / f"human_review_package_{datetime.now().strftime('%Y%m%d_%H%M')}"
    assets_dir = package_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(reviewed_rows, key=lambda item: int(item.get("selected_index") or 0))[: max(0, max_rows)]
    tsv_lines = [
        "sample_id\ttask_id\tsplit\tvlm_status\thuman_target_box\thuman_caption_boundary\thuman_overall\tcorrected_caption_text\tnotes"
    ]
    md_lines = [
        "# v1.0.5 Visual-Audited 人工复核样本包",
        "",
        f"- 数据目录：`{output_dir}`",
        f"- 样本数：{len(rows)}",
        "- 红框为 target，青色框为候选 caption。",
        "",
    ]
    for pos, row in enumerate(rows, start=1):
        sample_id = f"VA{pos:03d}"
        overlay_rel = copy_asset(row.get("overlay_image"), assets_dir, f"{sample_id}_{row.get('task_id')}_overlay.jpg")
        crop_rel = copy_asset(row.get("artwork_image"), assets_dir, f"{sample_id}_{row.get('task_id')}_crop.jpg")
        tsv_lines.append(
            "\t".join(
                [
                    sample_id,
                    str(row.get("task_id") or ""),
                    str(row.get("split") or ""),
                    str(row.get("visual_audit_status") or ""),
                    "",
                    "",
                    "",
                    "",
                    "",
                ]
            )
        )
        md_lines.extend(
            [
                f"## {sample_id} {row.get('task_id')} [{row.get('visual_audit_status')}]",
                "",
                f"- split/source/page：`{row.get('split')}` / `{row.get('source_file')}` / `{row.get('page')}`",
                f"- candidate_source：`{row.get('candidate_source')}`，area_ratio：`{row.get('candidate_area_ratio')}`，caption_score：`{row.get('candidate_caption_score')}`",
                f"- caption：{row.get('caption_text') or ''}",
                "",
                f"![overlay]({overlay_rel})",
                "",
                f"![crop]({crop_rel})",
                "",
                "```json",
                json.dumps(
                    {
                        "decision": row.get("decision") or {},
                        "strict_gate_flags": row.get("strict_gate_flags") or [],
                        "known_regression_bucket": row.get("known_regression_bucket"),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    md_path = package_dir / "v1.0.5VisualAudited人工复核样本包.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    (package_dir / "review_decisions.tsv").write_text("\n".join(tsv_lines) + "\n", encoding="utf-8")
    write_review_html(package_dir / "review.html", rows)
    return md_path


def write_review_html(path: Path, rows: list[dict[str, Any]]) -> None:
    cards = []
    for row in rows:
        cards.append(
            f"""
<div class="card">
  <div class="meta"><b>{html.escape(str(row.get('visual_audit_status')))}</b> | {html.escape(str(row.get('task_id')))} | {html.escape(str(row.get('split')))} | p.{html.escape(str(row.get('page')))}</div>
  <div class="meta">{html.escape(str(row.get('source_file')))}</div>
  <div class="imgs"><img src="{html.escape(str(row.get('overlay_image')))}"><img src="{html.escape(str(row.get('artwork_image')))}"></div>
  <div class="meta"><b>caption</b>: {html.escape(str(row.get('caption_text') or ''))}</div>
  <div class="meta"><b>decision</b>: {html.escape(json.dumps(row.get('decision') or {}, ensure_ascii=False))}</div>
</div>
"""
        )
    doc = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>v1.0.5 Visual Audit Review</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f7f7f5; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 6px; padding: 12px; }}
    .imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    img {{ width: 100%; height: auto; border: 1px solid #ccc; }}
    .meta {{ font-size: 13px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>v1.0.5 Visual Audit Review</h1>
  <div class="grid">{''.join(cards)}</div>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def copy_asset(path_value: Any, assets_dir: Path, name: str) -> str:
    src = Path(str(path_value or ""))
    dst = assets_dir / sanitize_filename(name)
    if src.exists():
        shutil.copy2(src, dst)
    return "assets/" + dst.name


def compact_review_row(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "selected_index",
        "task_id",
        "split",
        "source_file",
        "page",
        "caption_text",
        "image_bbox",
        "caption_bbox",
        "overlay_image",
        "artwork_image",
        "candidate_source",
        "candidate_area_ratio",
        "candidate_caption_score",
        "caption_target_overlap_ratio_by_caption",
        "ok",
        "error",
        "review_model",
        "input_mode",
        "visual_audit_status",
        "strict_gate_flags",
        "known_regression_bucket",
        "decision",
    ]
    return {key: copy.deepcopy(row.get(key)) for key in keys if key in row}


def candidate_rows(selected: list[v09.PageCandidate], split_docs: dict[str, str]) -> list[dict[str, Any]]:
    rows = []
    for idx, item in enumerate(selected):
        rows.append(
            {
                "selected_index": idx,
                "task_id": task_id_for_index(idx),
                "split": split_docs[item.source_file],
                "source_file": item.source_file,
                "page": item.page,
                "target_variant": item.target_variant,
                "target_source": item.target_source,
                "image_bbox": item.image_bbox,
                "caption_bbox": item.caption_bbox,
                "area_ratio": item.area_ratio,
                "caption_score": item.caption_score,
                "caption_text": item.caption_text,
                "caption_key": dedup.caption_key(item.caption_text),
                "risk_score": round(candidate_risk_score(item), 3),
            }
        )
    return rows


def load_existing_reviews(path: Path, args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    if not args.resume or not path.exists():
        return {}
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        task_id = str(row.get("task_id") or "")
        if not task_id:
            continue
        if args.retry_failed and not row.get("ok"):
            continue
        out[task_id] = row
    return out


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_text_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def task_id_for_index(index: int) -> str:
    return f"egva_v0_9_fixed_{index:06d}"


def known_regression_bucket(task_id: str) -> str:
    if task_id in KNOWN_SEVERE_REGRESSION_IDS:
        return "known_severe_user_review"
    if task_id in KNOWN_TRUNCATION_OR_CORRECTABLE_IDS:
        return "known_caption_truncation_or_correctable"
    return ""


def bbox_overlap_ratio(a: Any, b: Any, denominator: str = "caption") -> float:
    if not a or not b:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    if denominator == "target":
        return inter / area_a
    if denominator == "min":
        return inter / max(1.0, min(area_a, area_b))
    return inter / area_b


def bbox(block: dict[str, Any]) -> list[int]:
    value = block.get("bbox") or block.get("box") or []
    if len(value) != 4:
        return [0, 0, 0, 0]
    return [int(round(float(v))) for v in value]


def union_bbox(boxes: Iterable[list[int]]) -> list[int]:
    clean = [box for box in boxes if len(box) == 4]
    if not clean:
        return [0, 0, 0, 0]
    return [
        min(box[0] for box in clean),
        min(box[1] for box in clean),
        max(box[2] for box in clean),
        max(box[3] for box in clean),
    ]


def bbox_area(box: Any) -> float:
    if not box or len(box) != 4:
        return 0.0
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return 0.0
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def valid_page_bbox(box: Any, candidate: v09.PageCandidate, min_width: int = 8, min_height: int = 8) -> bool:
    if not box or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return False
    if x2 - x1 < min_width or y2 - y1 < min_height:
        return False
    if x1 < -4 or y1 < -4:
        return False
    if x2 > candidate.page_width + 4 or y2 > candidate.page_height + 4:
        return False
    return True


def normalize_bbox_value(value: Any) -> list[int] | None:
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def normalize_enum(value: Any, allowed: set[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"true", "yes", "1", "y"}:
        return True
    if text in {"false", "no", "0", "n"}:
        return False
    return None


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def dedupe_keep_order(values: Iterable[Any]) -> list[Any]:
    out = []
    for value in values:
        if value and value not in out:
            out.append(value)
    return out


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


if __name__ == "__main__":
    raise SystemExit(main())
