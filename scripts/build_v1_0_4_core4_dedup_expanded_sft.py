#!/usr/bin/env python3
"""Build a larger deduplicated v1.0.4 Core4 no-select SFT dataset.

This builder starts from the full v1.0 layout-candidate cache instead of the
already-augmented v1.0.3 train split. It keeps high-confidence single-caption
figure candidates, uses document-level train/val/test splits, and caps repeated
caption text so bbox jitter or same-page duplicates cannot dominate SFT.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_agentbench_v0_9_fixedsplit_train_multitarget as v09  # noqa: E402
import build_agentbench_v1_0_from_layout_candidates as v10  # noqa: E402
import build_v1_0_4_core4_clean_sft as core4  # noqa: E402
import filter_layout_candidates_v1_0_2 as v102_filter  # noqa: E402
from evidence_agent_env.data import EvidenceIndex, read_jsonl, write_jsonl  # noqa: E402


DEFAULT_CANDIDATE_CACHE = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/layout_candidates_v1_0_pdftext_opencv_20260607_2325"
)
DEFAULT_RAW_PDF_ROOT = Path("/root/datasets/chinese_landscape_authority_corpus")
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_llm_overlay_20260611_0222"
)
DEFAULT_GOLD_EVAL_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/gold_eval_v1_0_4_caption_corrected_20260611_1830"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")

FIGURE_LABEL_PATTERN = re.compile(
    r"(?:图|圖)\s*[一二三四五六七八九十百〇零0-9]+"
    r"(?:\s*[.\-－—．:：]\s*[一二三四五六七八九十百〇零0-9]+)*"
    r"(?:\s*[a-zA-Z])?"
    r"|(?:fig\.?|figure|plate)\s*[A-Za-z]?\s*[0-9IVXivx]+"
    r"(?:\s*[.\-－—．:：]\s*[0-9IVXivx]+)*(?:\s*[a-zA-Z])?",
    flags=re.IGNORECASE,
)
LOOSE_CN_FIGURE_LABEL_PATTERN = re.compile(
    r"(?:图|圖)\s*[一二三四五六七八九十百〇零0-9]+"
    r"(?:\s*[.\-－—．_:：lI]\s*[一二三四五六七八九十百〇零0-9]+)+"
    r"|(?:图|圖)\s*(?:[0-9]+\s+){2,}[0-9]+",
    flags=re.IGNORECASE,
)
BAD_SECTION_PATTERN = re.compile(r"参考文献|本章小结|不足与展望|目录|总结|主要结论|创新之处")
SUBFIGURE_LABEL_PATTERN = re.compile(
    r"\b(?:fig\.?|figure|plate)\s*[0-9ivx]+\s*[a-z]\b"
    r"|(?:图|圖)\s*[一二三四五六七八九十百〇零]+(?:\s*[:：]\s*[一二三四五六七八九十百〇零0-9]+)",
    flags=re.IGNORECASE,
)
STRICT_NON_ART_DIAGRAM_PATTERN = re.compile(
    r"统计图|类型统计|類型統計|结构图|結構圖|示意图|示意圖|流程图|流程圖|"
    r"关系图|關係圖|柱状图|柱狀圖|折线图|折線圖|表格|自制|模式|"
    r"思维|思維|演进|演進|成长图|成長圖|构成图|構成圖|图解|圖解|取像链|取像鏈|"
    r"视线分析|視線分析|分析图|分析圖|做法|平面图|平面圖|剖面|立面|"
    r"颜体字|顏体字|颜體字|字体|字形|笔画|筆畫|"
    r"\bdiagram\b|\bschematic\b",
    flags=re.IGNORECASE,
)
NON_PAINTING_DETAIL_PATTERN = re.compile(
    r"seal|seals|signature|inscription|colophon|rubbing|collector'?s?\s+seal|"
    r"印章|藏印|收藏印|钤印|鈐印|题跋|題跋|款识|款識|签名|簽名",
    flags=re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache-dir", default=str(DEFAULT_CANDIDATE_CACHE))
    parser.add_argument("--raw-pdf-root", default=str(DEFAULT_RAW_PDF_ROOT))
    parser.add_argument("--evidence-index-dir", default=str(DEFAULT_EVIDENCE_INDEX))
    parser.add_argument("--gold-eval-dir", default=str(DEFAULT_GOLD_EVAL_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-target", type=int, default=100000, help="Large default means keep all train candidates after caps.")
    parser.add_argument("--val-target", type=int, default=100)
    parser.add_argument("--test-target", type=int, default=100)
    parser.add_argument("--train-caption-cap", type=int, default=2)
    parser.add_argument("--eval-caption-cap", type=int, default=1)
    parser.add_argument("--max-doc-pages-train", type=int, default=120)
    parser.add_argument("--max-doc-pages-eval", type=int, default=40)
    parser.add_argument(
        "--reserve-largest-docs-for-train",
        type=int,
        default=5,
        help="Keep the largest accepted-candidate PDFs in train before choosing val/test docs.",
    )
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
        help=(
            "strict keeps only rule-clean candidates. visual_audited_expanded keeps "
            "additional marker-caption candidates whose errors can be rejected or repaired by VLM audit."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(Path(args.output_root))
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    for child in ["pages", "crops", "overlays", "sft", "episodes", "review", "gold_eval"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    candidates, scan_summary, filter_rows = collect_dedup_candidates(Path(args.candidate_cache_dir), args)
    split_docs = choose_doc_splits(candidates, args, rng)
    selected = select_candidates(candidates, split_docs, args, rng)
    (output_dir / "_split_map.json").write_text(json.dumps(split_docs, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(output_dir / "filter_decisions.jsonl", filter_rows)

    index = EvidenceIndex(str(args.evidence_index_dir))
    page_cache: dict[tuple[str, int], Path] = {}
    tasks_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    episodes_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sft_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_tasks: list[dict[str, Any]] = []
    all_episodes: list[dict[str, Any]] = []
    all_sft: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, candidate in enumerate(selected):
        try:
            core5_task = v09.build_task(idx, candidate, output_dir, args, index, page_cache)
            v10.normalize_v1_task(core5_task)
            transformed, reviews = core4.transform_task(core5_task, caption_overrides={})
            transformed["dataset_version"] = "v1.0.4_core4_dedup_expanded"
            transformed["runtime_mode"] = "v1_0_4_core4_dedup_expanded_docsplit_caption_cap"
            transformed.setdefault("candidate_meta", {})["dedup_caption_key"] = caption_key(caption_from_candidate(candidate))
            transformed.setdefault("candidate_meta", {})["caption_cap_policy"] = (
                f"train<={args.train_caption_cap}, eval<={args.eval_caption_cap}"
            )
            replay = replay_from_task(transformed)
            actions = core4.build_oracle_actions(transformed, replay)
            sft_rows = core4.build_sft_rows(transformed, actions, replay)
            for row in sft_rows:
                row["label_source"] = "v1_0_4_core4_dedup_expanded_rule_sft"
                row["tool_schema_version"] = "v1.0.4_no_select_core4_dedup_expanded"
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
            all_tasks.append(transformed)
            all_episodes.append(episode)
            all_sft.extend(sft_rows)
            review_rows.extend(reviews)
        except Exception as exc:
            errors.append(
                {
                    "source_file": candidate.source_file,
                    "page": candidate.page,
                    "caption_text": candidate.caption_text,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", tasks_by_split.get(split, []))
        write_jsonl(output_dir / "episodes" / f"{split}_oracle_episodes.jsonl", episodes_by_split.get(split, []))
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_by_split.get(split, []))
    write_jsonl(output_dir / "tasks_all.jsonl", all_tasks)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", all_episodes)
    write_jsonl(output_dir / "sft" / "all.jsonl", all_sft)
    write_jsonl(output_dir / "review_queue.jsonl", core4.cap_review_rows(review_rows, 100))
    write_jsonl(output_dir / "builder_errors.jsonl", errors)

    gold_eval_summary = core4.build_gold_eval_core4(
        Path(args.gold_eval_dir), output_dir / "gold_eval", core4.load_caption_overrides(Path(args.gold_eval_dir))
    )
    summary = build_summary(args, output_dir, scan_summary, filter_rows, selected, all_tasks, all_sft, errors, gold_eval_summary)
    (output_dir / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def default_output_dir(root: Path) -> Path:
    return root / f"agentbench_v1_0_4_core4_dedup_expanded_sft_{datetime.now().strftime('%Y%m%d_%H%M')}"


def collect_dedup_candidates(
    cache_dir: Path,
    args: argparse.Namespace,
) -> tuple[list[v09.PageCandidate], dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(cache_dir / "page_candidates.jsonl")
    candidates: list[v09.PageCandidate] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        raw_figures = row.get("figure_candidates") or []
        if not raw_figures:
            continue
        image_blocks = tuple(v10.normalize_block(fig, row) for fig in raw_figures)
        for variant, block in enumerate(image_blocks):
            decision = decide_candidate(row, block, args)
            decisions.append(decision)
            if not decision["keep"]:
                continue
            candidates.append(
                v09.PageCandidate(
                    source_file=str(row["source_file"]),
                    source_stem=str(row["source_stem"]),
                    source_path=Path(row["source_path"]),
                    page=int(row["page"]),
                    page_count=int(row["page_count"]),
                    bbox_pt=tuple(float(v) for v in (block.get("bbox_pt") or [0, 0, 0, 0])),
                    image_bbox=[int(v) for v in block["bbox"]],
                    area_ratio=float(block.get("area_ratio") or 0.0),
                    caption_bbox=block.get("caption_bbox"),
                    caption_text=str(block.get("caption_text") or ""),
                    caption_score=float(block.get("caption_score") or 0.0),
                    page_width=int(row["page_width"]),
                    page_height=int(row["page_height"]),
                    source_meta=row.get("source_meta") or {},
                    image_blocks=image_blocks,
                    text_blocks=tuple(row.get("text_blocks") or []),
                    target_variant=variant,
                    target_source=str(block.get("source") or "layout_candidate"),
                    augmentation=None,
                )
            )
    cache_summary_path = cache_dir / "summary.json"
    cache_summary = json.loads(cache_summary_path.read_text(encoding="utf-8")) if cache_summary_path.exists() else {}
    scan_summary = {
        "candidate_cache_dir": str(cache_dir),
        "raw_pdf_root": str(args.raw_pdf_root),
        "cache_summary": cache_summary,
        "page_rows": len(rows),
        "figure_rows": sum(len(row.get("figure_candidates") or []) for row in rows),
        "accepted_candidates": len(candidates),
        "accepted_unique_caption": len({caption_key(item.caption_text) for item in candidates}),
        "accepted_docs": len({item.source_file for item in candidates}),
    }
    return candidates, scan_summary, decisions


def decide_candidate(row: dict[str, Any], fig: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    caption = str(fig.get("caption_text") or "").strip()
    reasons: list[str] = []
    labels = figure_labels_strict(caption)
    loose_labels = figure_labels_loose(caption)
    if not caption:
        reasons.append("missing_caption_text")
    if caption and not core4.caption_starts_marker(caption):
        reasons.append("caption_not_start_marker")
    if len(set(labels + loose_labels)) > 1:
        reasons.append("multi_figure_caption")
    if caption and core4.caption_number_only(caption):
        reasons.append("caption_number_only")
    if caption and core4.caption_body_after_marker(caption):
        reasons.append("caption_body_after_marker")
    if caption and core4.caption_too_short_after_marker(caption):
        reasons.append("caption_too_short_after_marker")
    if BAD_SECTION_PATTERN.search(caption):
        reasons.append("bad_section_terms")
    source = str(fig.get("source") or "")
    area_ratio = float(fig.get("area_ratio") or 0.0)
    if source == "opencv_visual_region" and area_ratio > args.opencv_max_area_ratio:
        reasons.append("opencv_area_too_large")
    if SUBFIGURE_LABEL_PATTERN.search(caption):
        reasons.append("subfigure_label_risky")
    if STRICT_NON_ART_DIAGRAM_PATTERN.search(caption):
        reasons.append("non_art_diagram_or_schema_caption")
    if NON_PAINTING_DETAIL_PATTERN.search(caption):
        reasons.append("non_painting_detail_caption")
    v102 = v102_filter.decide_candidate(row, fig, args)
    if not v102["keep"]:
        reasons.extend(reason for reason in v102["reasons"] if reason != "kept_by_v1_0_2_rules")
    profile = str(getattr(args, "candidate_filter_profile", "strict") or "strict")
    if profile == "visual_audited_expanded":
        reasons = relax_reasons_for_visual_audit(caption, reasons)
    key = caption_key(caption)
    return {
        "keep": not reasons,
        "reasons": reasons or ["accepted"],
        "primary_reason": reasons[0] if reasons else "accepted",
        "source_file": row.get("source_file"),
        "page": row.get("page"),
        "candidate_id": fig.get("candidate_id"),
        "source": fig.get("source"),
        "bbox": fig.get("bbox"),
        "caption_text": caption,
        "caption_key": key,
        "caption_score": fig.get("caption_score"),
        "area_ratio": fig.get("area_ratio"),
        "figure_labels": labels,
        "loose_figure_labels": loose_labels,
        "v1_0_2_filter": v102,
        "candidate_filter_profile": profile,
    }


def relax_reasons_for_visual_audit(caption: str, reasons: list[str]) -> list[str]:
    """Keep more figure-caption candidates when a downstream VLM audit is mandatory.

    This profile is intentionally not used by rule-only SFT.  It admits
    candidates with repairable bbox/caption-boundary problems, while still
    rejecting missing captions, non-figure body text, obvious diagrams, TOC-like
    material, seals/inscriptions, and known non-landscape classes.
    """
    if not caption or not core4.caption_starts_marker(caption):
        return list(dict.fromkeys(reasons))
    fatal = {
        "missing_caption_text",
        "caption_not_start_marker",
        "bad_section_terms",
        "caption_page_header",
        "obvious_non_landscape_caption",
        "non_art_diagram_caption",
        "non_art_diagram_or_schema_caption",
        "non_painting_detail_caption",
    }
    repairable = {
        "multi_figure_caption",
        "subfigure_label_risky",
        "caption_body_after_marker",
        "caption_body_fragment",
        "caption_too_long_body_like",
        "caption_not_caption_like",
        "caption_number_only",
        "caption_too_short_after_marker",
        "opencv_text_overlap_high",
        "opencv_area_too_large",
        "opencv_area_too_small",
        "opencv_height_too_small",
        "opencv_width_too_small",
        "opencv_bad_aspect",
    }
    kept: list[str] = []
    for reason in reasons:
        if reason in fatal:
            kept.append(reason)
        elif reason in repairable:
            continue
        else:
            kept.append(reason)
    return list(dict.fromkeys(kept))


def choose_doc_splits(
    candidates: list[v09.PageCandidate],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, str]:
    by_doc: dict[str, list[v09.PageCandidate]] = defaultdict(list)
    for item in candidates:
        by_doc[item.source_file].append(item)
    largest_train_docs = {
        doc
        for doc, _count in sorted(by_doc.items(), key=lambda kv: (len(kv[1]), kv[0]), reverse=True)[
            : max(0, int(args.reserve_largest_docs_for_train))
        ]
    }
    available = set(by_doc) - largest_train_docs
    split_docs: dict[str, str] = {}
    for split, target in [("val", args.val_target), ("test", args.test_target)]:
        picked = pick_eval_docs(by_doc, available, target, args.max_doc_pages_eval, rng)
        for doc in picked:
            split_docs[doc] = split
        available.difference_update(picked)
    for doc in sorted(available):
        split_docs[doc] = "train"
    for doc in sorted(largest_train_docs):
        split_docs[doc] = "train"
    return split_docs


def pick_eval_docs(
    by_doc: dict[str, list[v09.PageCandidate]],
    available: set[str],
    target: int,
    per_doc_cap: int,
    rng: random.Random,
) -> list[str]:
    docs = list(available)
    rng.shuffle(docs)
    docs.sort(key=lambda doc: (min(len(by_doc[doc]), per_doc_cap), len(by_doc[doc]), doc), reverse=True)
    picked: list[str] = []
    current = 0
    for doc in docs:
        contribution = min(len({caption_key(item.caption_text) for item in by_doc[doc]}), per_doc_cap)
        if contribution <= 0:
            continue
        picked.append(doc)
        current += contribution
        if current >= target:
            break
    return picked


def select_candidates(
    candidates: list[v09.PageCandidate],
    split_docs: dict[str, str],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[v09.PageCandidate]:
    by_split_doc: dict[str, dict[str, list[v09.PageCandidate]]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        by_split_doc[split_docs[item.source_file]][item.source_file].append(item)
    selected: list[v09.PageCandidate] = []
    selected.extend(
        capped_round_robin(
            by_split_doc["train"],
            target=args.train_target,
            per_doc_cap=args.max_doc_pages_train,
            caption_cap=args.train_caption_cap,
            rng=rng,
        )
    )
    selected.extend(
        capped_round_robin(
            by_split_doc["val"],
            target=args.val_target,
            per_doc_cap=args.max_doc_pages_eval,
            caption_cap=args.eval_caption_cap,
            rng=rng,
        )
    )
    selected.extend(
        capped_round_robin(
            by_split_doc["test"],
            target=args.test_target,
            per_doc_cap=args.max_doc_pages_eval,
            caption_cap=args.eval_caption_cap,
            rng=rng,
        )
    )
    selected.sort(key=lambda item: ({"train": 0, "val": 1, "test": 2}[split_docs[item.source_file]], item.source_file, item.page, item.target_variant))
    return selected


def capped_round_robin(
    by_doc: dict[str, list[v09.PageCandidate]],
    *,
    target: int,
    per_doc_cap: int,
    caption_cap: int,
    rng: random.Random,
) -> list[v09.PageCandidate]:
    buckets: dict[str, list[v09.PageCandidate]] = {}
    for doc, rows in by_doc.items():
        ordered = sorted(
            rows,
            key=lambda item: (-v09.page_target_score_from_candidate(item), item.page, item.target_variant),
        )
        buckets[doc] = ordered[: max(1, per_doc_cap)]
    docs = sorted(buckets)
    rng.shuffle(docs)
    selected: list[v09.PageCandidate] = []
    caption_counts: Counter[str] = Counter()
    cursor = 0
    while len(selected) < target and docs:
        doc = docs[cursor % len(docs)]
        bucket = buckets[doc]
        picked = None
        for idx, item in enumerate(bucket):
            key = caption_key(item.caption_text)
            if caption_counts[key] < caption_cap:
                picked = bucket.pop(idx)
                break
        if picked is not None:
            selected.append(picked)
            caption_counts[caption_key(picked.caption_text)] += 1
        else:
            bucket.clear()
        docs = [item for item in docs if buckets[item]]
        cursor += 1
    return selected


def replay_from_task(task: dict[str, Any]) -> dict[str, Any]:
    retrieve_result = {
        "tool": "retrieve_evidence",
        "query": (task.get("gold") or {}).get("evidence_query"),
        "scope": "same_document",
        "results": (task.get("gold") or {}).get("_retrieval_results") or [],
        "hit_evidence_ids": [],
    }
    return {
        "retrieve_results": {str(task.get("task_id")): retrieve_result},
        "open_results": {},
        "external_open_ids": defaultdict(list),
    }


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    scan_summary: dict[str, Any],
    filter_rows: list[dict[str, Any]],
    selected: list[v09.PageCandidate],
    tasks: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    gold_eval_summary: dict[str, Any],
) -> dict[str, Any]:
    split_counts = Counter(task.get("split") for task in tasks)
    sft_split_counts = Counter(row.get("split") for row in sft_rows)
    action_counts = Counter((row.get("action") or {}).get("action") for row in sft_rows)
    caption_counts_by_split: dict[str, Counter[str]] = defaultdict(Counter)
    source_page_caption_by_split: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
    docs_by_split: dict[str, set[str]] = defaultdict(set)
    for task in tasks:
        split = str(task.get("split") or "")
        caption = str((task.get("gold") or {}).get("caption_text") or "")
        key = caption_key(caption)
        caption_counts_by_split[split][key] += 1
        source_page_caption_by_split[split].add((str(task.get("source_file")), int(task.get("page") or 0), key))
        docs_by_split[split].add(str(task.get("source_file")))
    cap_violations = {
        split: {
            key: count
            for key, count in counts.items()
            if count > (args.train_caption_cap if split == "train" else args.eval_caption_cap)
        }
        for split, counts in caption_counts_by_split.items()
    }
    cap_violations = {split: vals for split, vals in cap_violations.items() if vals}
    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": "v1.0.4_core4_dedup_expanded_sft",
        "builder": "scripts/build_v1_0_4_core4_dedup_expanded_sft.py",
        "output_dir": str(output_dir),
        "candidate_cache_dir": args.candidate_cache_dir,
        "raw_pdf_root": args.raw_pdf_root,
        "evidence_index_dir": args.evidence_index_dir,
        "gold_eval_dir": args.gold_eval_dir,
        "caption_cap_policy": {
            "train_caption_cap": args.train_caption_cap,
            "eval_caption_cap": args.eval_caption_cap,
        },
        "split_policy": {
            "document_level": True,
            "reserve_largest_docs_for_train": args.reserve_largest_docs_for_train,
        },
        "args": vars(args),
        "scan_summary": scan_summary,
        "filter_summary": {
            "decisions": len(filter_rows),
            "accepted": sum(1 for row in filter_rows if row.get("keep")),
            "rejected": sum(1 for row in filter_rows if not row.get("keep")),
            "primary_reject_reasons": dict(Counter(row.get("primary_reason") for row in filter_rows if not row.get("keep")).most_common(20)),
            "accepted_source_counts": dict(Counter(row.get("source") for row in filter_rows if row.get("keep")).most_common()),
        },
        "split_counts": dict(split_counts),
        "doc_counts_by_split": {split: len(vals) for split, vals in docs_by_split.items()},
        "unique_caption_by_split": {split: len(vals) for split, vals in caption_counts_by_split.items()},
        "unique_source_page_caption_by_split": {split: len(vals) for split, vals in source_page_caption_by_split.items()},
        "caption_cap_violations": cap_violations,
        "selected_candidates": len(selected),
        "sft_rows_total": len(sft_rows),
        "sft_split_counts": dict(sft_split_counts),
        "sft_action_counts": dict(action_counts),
        "target_source_counts": dict(Counter((task.get("candidate_meta") or {}).get("source") for task in tasks).most_common()),
        "field_counts": core4.field_summary(tasks),
        "builder_error_count": len(errors),
        "builder_errors_preview": errors[:20],
        "gold_eval_core4": gold_eval_summary,
        "artifacts": {
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
            "sft_train": str(output_dir / "sft" / "train.jsonl"),
            "sft_val": str(output_dir / "sft" / "val.jsonl"),
            "sft_test": str(output_dir / "sft" / "test.jsonl"),
            "filter_decisions": str(output_dir / "filter_decisions.jsonl"),
            "review_queue": str(output_dir / "review_queue.jsonl"),
            "builder_errors": str(output_dir / "builder_errors.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    return summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Core4 去重扩容 SFT 构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 候选缓存：`{summary['candidate_cache_dir']}`",
        f"- raw PDF 根目录：`{summary['raw_pdf_root']}`",
        "",
        "## 结论",
        "",
        "- 本轮从全量 layout candidate cache 重新抽样，不复用 v1.0.3 中以 bbox jitter 凑数的 train split。",
        "- 已按文档级划分 train/val/test，并限制 caption 重复：train 每个 caption 最多 2 条，val/test 最多 1 条。",
        "- 输出仍是 no-select Core4 SFT，字段为 caption_text、depicted_work_title、image_scope、object_type。",
        "",
        "## 规模",
        "",
        f"- 原始候选页：{summary['scan_summary'].get('page_rows')}",
        f"- 原始候选框：{summary['scan_summary'].get('figure_rows')}",
        f"- 自动接受候选：{summary['filter_summary'].get('accepted')}",
        f"- 自动接受 unique caption：{summary['scan_summary'].get('accepted_unique_caption')}",
        f"- 自动接受 PDF 数：{summary['scan_summary'].get('accepted_docs')}",
        f"- split task 数：`{json.dumps(summary['split_counts'], ensure_ascii=False)}`",
        f"- split unique caption：`{json.dumps(summary['unique_caption_by_split'], ensure_ascii=False)}`",
        f"- split unique (source,page,caption)：`{json.dumps(summary['unique_source_page_caption_by_split'], ensure_ascii=False)}`",
        f"- split PDF 数：`{json.dumps(summary['doc_counts_by_split'], ensure_ascii=False)}`",
        f"- SFT rows：`{json.dumps(summary['sft_split_counts'], ensure_ascii=False)}`，total={summary['sft_rows_total']}",
        f"- action rows：`{json.dumps(summary['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 去重检查",
        "",
        f"- caption cap policy：`{json.dumps(summary['caption_cap_policy'], ensure_ascii=False)}`",
        f"- caption cap violations：`{json.dumps(summary['caption_cap_violations'], ensure_ascii=False)}`",
        "",
        "## 过滤",
        "",
        f"- rejected：{summary['filter_summary'].get('rejected')}",
        f"- primary_reject_reasons：`{json.dumps(summary['filter_summary'].get('primary_reject_reasons'), ensure_ascii=False)}`",
        f"- accepted_source_counts：`{json.dumps(summary['filter_summary'].get('accepted_source_counts'), ensure_ascii=False)}`",
        "",
        "## 字段分布",
        "",
        f"- field_counts：`{json.dumps(summary['field_counts'], ensure_ascii=False)}`",
        "",
        "## 风险",
        "",
        "- 这仍是规则 silver 数据；caption 边界比旧 train 干净，但没有全量人工/VLM 裁决。",
        "- 少数 caption 仍可能是建筑/示意图/多对象说明，后续应通过 review_queue 或 GoldEval 误差再补丁。",
        "- 本轮主要解决 SFT 训练多样性，不替代 val_gold_50/test_gold_100 的最终评测。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def caption_from_candidate(candidate: v09.PageCandidate) -> str:
    return str(candidate.caption_text or "")


def caption_key(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def figure_labels_strict(text: str) -> list[str]:
    return [
        re.sub(r"\s+", "", match.group(0).lower()).replace("圖", "图")
        for match in FIGURE_LABEL_PATTERN.finditer(text or "")
    ]


def figure_labels_loose(text: str) -> list[str]:
    return [
        re.sub(r"\s+", "", match.group(0).lower()).replace("圖", "图").replace("i", "l")
        for match in LOOSE_CN_FIGURE_LABEL_PATTERN.finditer(text or "")
    ]


if __name__ == "__main__":
    raise SystemExit(main())
