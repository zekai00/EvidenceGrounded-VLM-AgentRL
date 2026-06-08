#!/usr/bin/env python3
"""Build AgentBench v1.0 tasks from a layout candidate cache."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_agentbench_v0_9_fixedsplit_train_multitarget as v09  # noqa: E402
from evidence_agent_env.data import EvidenceIndex, read_jsonl  # noqa: E402


DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache-dir", required=True)
    parser.add_argument("--evidence-index-dir", default=str(DEFAULT_EVIDENCE_INDEX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--train-target", type=int, default=1500)
    parser.add_argument("--train-min", type=int, default=1500)
    parser.add_argument("--train-max", type=int, default=1500)
    parser.add_argument("--val-target", type=int, default=200)
    parser.add_argument("--test-target", type=int, default=200)
    parser.add_argument("--page-dpi", type=int, default=150)
    parser.add_argument("--crop-dpi", type=int, default=200)
    parser.add_argument("--top-k-regions", type=int, default=10)
    parser.add_argument("--max-doc-pages-train", type=int, default=120)
    parser.add_argument("--max-doc-pages-eval", type=int, default=30)
    parser.add_argument("--max-val-docs", type=int, default=0, help="Optional cap for number of val PDFs; 0 keeps old behavior.")
    parser.add_argument("--max-test-docs", type=int, default=0, help="Optional cap for number of test PDFs; 0 keeps old behavior.")
    parser.add_argument("--max-train-targets-per-page", type=int, default=6)
    parser.add_argument("--augment-train-to-target", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-caption-text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-caption-score", type=float, default=-999.0)
    parser.add_argument("--jitter-seed-offset", type=int, default=9100)
    parser.add_argument("--seed", type=int, default=20260607)
    # Kept for compatibility with v0.9 selection helpers.
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.86)
    parser.add_argument("--min-width-ratio", type=float, default=0.045)
    parser.add_argument("--min-height-ratio", type=float, default=0.035)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(Path(args.output_root))
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    for child in ["pages", "crops", "overlays", "sft", "episodes", "review"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    candidates, scan_summary = collect_candidates_from_cache(Path(args.candidate_cache_dir), args)
    split_docs = choose_doc_splits_v1(candidates, args, rng)
    selected = select_split_candidates_v1(candidates, split_docs, args, rng)
    (output_dir / "_split_map.json").write_text(json.dumps(split_docs, ensure_ascii=False, indent=2), encoding="utf-8")

    index = EvidenceIndex(args.evidence_index_dir)
    page_cache: dict[tuple[str, int], Path] = {}
    tasks: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for idx, candidate in enumerate(selected):
        try:
            task = v09.build_task(idx, candidate, output_dir, args, index, page_cache)
            normalize_v1_task(task)
            episode = v09.build_episode(task)
            rows = v09.build_sft_rows(task, episode["actions"], output_dir, args)
            for row in rows:
                row["tool_schema_version"] = "v1.0_inspect_crop_core5"
                row["label_source"] = "v1_0_layout_candidate_core5_sft"
            tasks.append(task)
            episodes.append(episode)
            sft_rows.extend(rows)
            quality_rows.append(v09.task_quality(task))
        except Exception as exc:
            errors.append(
                {
                    "source_file": candidate.source_file,
                    "page": candidate.page,
                    "image_bbox": candidate.image_bbox,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    v09.write_outputs(output_dir, tasks, episodes, sft_rows)
    quality = v09.summarize(args, output_dir, candidates, selected, tasks, sft_rows, quality_rows, scan_summary, errors)
    quality["dataset_version"] = "v1.0_layout_candidate_core5"
    quality["candidate_cache_dir"] = str(args.candidate_cache_dir)
    quality["notes"] = [
        "Gold labels are silver labels from v1.0 layout candidate cache: PDF image blocks plus OpenCV visual regions.",
        "Current cache does not yet include successful external OCR because PaddleOCR/EasyOCR runtime was blocked in smoke.",
        "Val/test are document-level unseen-PDF splits and strict page-capped.",
    ]
    manifest = {
        "created_at": now(),
        "dataset_version": "v1.0_layout_candidate_core5",
        "builder": "scripts/build_agentbench_v1_0_from_layout_candidates.py",
        "candidate_cache_dir": str(args.candidate_cache_dir),
        "evidence_index_dir": str(args.evidence_index_dir),
        "output_dir": str(output_dir),
        "page_cap_policy": "val/test strict: at most one selected task per (source_file,page); train may contain multiple targets per page",
        "split_policy": "fixed document-level split: each source_file appears in exactly one split",
        "target_claim_fields": v09.CORE5_FIELDS,
        "args": vars(args),
        "quality": quality,
        "files": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
            "oracle_episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
            "sft_train": str(output_dir / "sft" / "train.jsonl"),
            "sft_val": str(output_dir / "sft" / "val.jsonl"),
            "sft_test": str(output_dir / "sft" / "test.jsonl"),
            "review_html": str(output_dir / "review" / "review.html"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8")
    v09.write_review_html(output_dir / "review" / "review.html", tasks[:120])
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


def default_output_dir(root: Path) -> Path:
    return root / f"agentbench_v1_0_layout_candidate_sft_{datetime.now().strftime('%Y%m%d_%H%M')}"


def collect_candidates_from_cache(cache_dir: Path, args: argparse.Namespace) -> tuple[list[v09.PageCandidate], dict[str, Any]]:
    pages_path = cache_dir / "page_candidates.jsonl"
    if not pages_path.exists():
        raise FileNotFoundError(pages_path)
    rows = read_jsonl(pages_path)
    candidates: list[v09.PageCandidate] = []
    for row in rows:
        figs = filter_figures(row.get("figure_candidates") or [], args)
        if not figs:
            continue
        image_blocks = tuple(normalize_block(fig, row) for fig in figs)
        best = image_blocks[0]
        candidates.append(
            v09.PageCandidate(
                source_file=str(row["source_file"]),
                source_stem=str(row["source_stem"]),
                source_path=Path(row["source_path"]),
                page=int(row["page"]),
                page_count=int(row["page_count"]),
                bbox_pt=tuple(float(v) for v in (best.get("bbox_pt") or [0, 0, 0, 0])),
                image_bbox=list(best["bbox"]),
                area_ratio=float(best.get("area_ratio") or 0.0),
                caption_bbox=best.get("caption_bbox"),
                caption_text=str(best.get("caption_text") or ""),
                caption_score=float(best.get("caption_score") or 0.0),
                page_width=int(row["page_width"]),
                page_height=int(row["page_height"]),
                source_meta=row.get("source_meta") or {},
                image_blocks=image_blocks,
                text_blocks=tuple(row.get("text_blocks") or []),
                target_variant=0,
                target_source=str(best.get("source") or "layout_candidate"),
            )
        )
    source_counts = Counter(item.source_file for item in candidates)
    source_by_candidate = Counter(block.get("source") for item in candidates for block in item.image_blocks)
    summary_path = cache_dir / "summary.json"
    cache_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    return candidates, {
        "cache_dir": str(cache_dir),
        "cache_summary": cache_summary,
        "pdfs_seen": cache_summary.get("pdfs_seen", len(source_counts)),
        "pages_seen": cache_summary.get("pages_seen", len(candidates)),
        "candidate_pages": len(candidates),
        "pages_with_selected_image_block": sum(1 for item in candidates if item.target_source == "pdf_image_block"),
        "candidate_source_counts": dict(source_by_candidate),
        "scan_error_count": cache_summary.get("error_count", 0),
        "scan_errors": cache_summary.get("errors", [])[:30],
    }


def filter_figures(figures: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for fig in figures:
        if args.require_caption_text and not str(fig.get("caption_text") or "").strip():
            continue
        try:
            caption_score = float(fig.get("caption_score") or -999.0)
        except Exception:
            caption_score = -999.0
        if caption_score < args.min_caption_score:
            continue
        kept.append(fig)
    return kept


def choose_doc_splits_v1(
    candidates: list[v09.PageCandidate],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, str]:
    """Eval-first document split.

    v0.9 used a train-first split because the candidate pool was small. v1.0
    has many more candidate pages, so we reserve val/test first and train on
    the remaining PDFs. This gives a cleaner 1500/200/200 target when enough
    documents exist.
    """

    by_doc: dict[str, list[v09.PageCandidate]] = defaultdict(list)
    for item in candidates:
        by_doc[item.source_file].append(item)
    available = set(by_doc)
    split_docs: dict[str, str] = {}
    for split, target, max_docs in [
        ("val", args.val_target, int(getattr(args, "max_val_docs", 0) or 0)),
        ("test", args.test_target, int(getattr(args, "max_test_docs", 0) or 0)),
    ]:
        picked = pick_eval_docs_v1(by_doc, available, target, args.max_doc_pages_eval, rng, max_docs=max_docs)
        for doc in picked:
            split_docs[doc] = split
        available.difference_update(picked)
    for doc in sorted(available):
        split_docs[doc] = "train"
    return split_docs


def pick_eval_docs_v1(
    by_doc: dict[str, list[v09.PageCandidate]],
    available: set[str],
    target: int,
    per_doc_cap: int,
    rng: random.Random,
    max_docs: int = 0,
) -> list[str]:
    docs = list(available)
    rng.shuffle(docs)
    # Prefer documents that can contribute useful eval pages without consuming
    # the very largest train-rich PDFs first.
    if max_docs > 0:
        docs.sort(key=lambda doc: (min(len(by_doc[doc]), per_doc_cap), len(by_doc[doc]), doc), reverse=True)
    else:
        docs.sort(
            key=lambda doc: (
                min(abs(min(len(by_doc[doc]), per_doc_cap) - 10), 20),
                max(0, len(by_doc[doc]) - per_doc_cap),
                doc,
            )
        )
    picked: list[str] = []
    current = 0
    for doc in docs:
        if max_docs > 0 and len(picked) >= max_docs:
            break
        contribution = min(len(by_doc[doc]), per_doc_cap)
        if contribution <= 0:
            continue
        picked.append(doc)
        current += contribution
        if current >= target:
            break
    return picked


def select_split_candidates_v1(
    candidates: list[v09.PageCandidate],
    split_docs: dict[str, str],
    args: argparse.Namespace,
    rng: random.Random,
) -> list[v09.PageCandidate]:
    by_split_doc: dict[str, dict[str, list[v09.PageCandidate]]] = defaultdict(lambda: defaultdict(list))
    for item in candidates:
        by_split_doc[split_docs[item.source_file]][item.source_file].append(item)

    selected: list[v09.PageCandidate] = []
    selected.extend(v09.balanced_sample_by_doc(by_split_doc["val"], args.val_target, args.max_doc_pages_eval, rng))
    selected.extend(v09.balanced_sample_by_doc(by_split_doc["test"], args.test_target, args.max_doc_pages_eval, rng))
    train_target = max(args.train_min, min(args.train_target, args.train_max))
    train_rows = build_train_multitarget_candidates_v1(
        by_split_doc["train"],
        target=train_target,
        max_doc_pages=args.max_doc_pages_train,
        max_targets_per_page=args.max_train_targets_per_page,
        rng=rng,
    )
    if len(train_rows) < train_target and args.augment_train_to_target:
        train_rows = v09.augment_train_candidates(
            train_rows,
            target=train_target,
            rng=random.Random(args.seed + args.jitter_seed_offset),
        )
    selected.extend(train_rows[: args.train_max])
    selected.sort(key=lambda item: ({"train": 0, "val": 1, "test": 2}[split_docs[item.source_file]], item.source_file, item.page))
    return selected


def build_train_multitarget_candidates_v1(
    by_doc: dict[str, list[v09.PageCandidate]],
    *,
    target: int,
    max_doc_pages: int,
    max_targets_per_page: int,
    rng: random.Random,
) -> list[v09.PageCandidate]:
    by_doc_units: dict[str, list[v09.PageCandidate]] = {}
    for doc, rows in by_doc.items():
        ordered_pages = sorted(rows, key=lambda item: (-v09.page_target_score_from_candidate(item), item.page))
        units: list[v09.PageCandidate] = []
        for page_candidate in ordered_pages[: min(len(ordered_pages), max_doc_pages)]:
            blocks = sorted(
                list(page_candidate.image_blocks),
                key=lambda item: v09.page_target_score(item),
                reverse=True,
            )[: max(1, max_targets_per_page)]
            for variant, block in enumerate(blocks):
                source = str(block.get("source") or page_candidate.target_source or "layout_candidate")
                target_source = source if source.startswith("train_") else f"train_{source}"
                units.append(v09.retarget_candidate(page_candidate, block, variant, target_source))
        by_doc_units[doc] = units

    docs = sorted(by_doc_units)
    rng.shuffle(docs)
    selected: list[v09.PageCandidate] = []
    cursor = 0
    while len(selected) < target and docs:
        doc = docs[cursor % len(docs)]
        bucket = by_doc_units[doc]
        if bucket:
            selected.append(bucket.pop(0))
        docs = [item for item in docs if by_doc_units[item]]
        cursor += 1
    return selected


def normalize_block(fig: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    block = copy.deepcopy(fig)
    block["bbox"] = [int(v) for v in fig.get("bbox", [])]
    if not block.get("bbox_pt"):
        block["bbox_pt"] = pixel_bbox_to_points(block["bbox"], int(row["page_width"]), int(row["page_height"]), int(row.get("page_dpi") or 150))
    block["caption_bbox"] = block.get("caption_bbox") or None
    block["caption_text"] = str(block.get("caption_text") or "")
    block["caption_score"] = float(block.get("caption_score") or 0.0)
    block["area_ratio"] = float(block.get("area_ratio") or 0.0)
    return block


def pixel_bbox_to_points(bbox: list[int], page_width: int, page_height: int, dpi: int) -> list[float]:
    scale = dpi / 72.0
    return [float(v) / scale for v in bbox]


def normalize_v1_task(task: dict[str, Any]) -> None:
    task["dataset_version"] = "v1.0_layout_candidate_core5"
    task["tool_schema_version"] = "v1.0_inspect_crop_core5"
    task["runtime_mode"] = "v1_0_layout_candidate_fixedsplit_eval_pagecap_train_multitarget"
    task.setdefault("gold", {})["label_source"] = "v1_0_layout_candidate_pdftext_opencv_caption_heuristic"
    meta = task.setdefault("candidate_meta", {})
    meta["v1_0_candidate_cache"] = True
    for region in task.get("region_candidates") or []:
        if region.get("source") in {"opencv_visual_region", "pdf_image_block"}:
            region.setdefault("hint", "v1.0 layout candidate cache region")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    q = manifest["quality"]
    lines = [
        "# AgentBench v1.0 Layout Candidate 构建报告",
        "",
        f"生成时间：{manifest['created_at']} CST",
        "",
        "## 输出位置",
        "",
        "```text",
        manifest["output_dir"],
        "```",
        "",
        "## 输入候选缓存",
        "",
        "```text",
        manifest["candidate_cache_dir"],
        "```",
        "",
        "## 规模",
        "",
        f"- scan PDFs：{q['scan_summary'].get('pdfs_seen')}",
        f"- scan pages：{q['scan_summary'].get('pages_seen')}",
        f"- all_candidate_pages：{q['all_candidate_pages']}",
        f"- selected_tasks：{q['selected_tasks']}",
        f"- split_counts：`{json.dumps(q['split_counts'], ensure_ascii=False)}`",
        f"- doc_counts_by_split：`{json.dumps(q['doc_counts_by_split'], ensure_ascii=False)}`",
        f"- unique_pages：{q['unique_pages']}",
        f"- unique_sources：{q['unique_sources']}",
        f"- target_source_counts：`{json.dumps(q.get('target_source_counts', {}), ensure_ascii=False)}`",
        "",
        "## 硬质量检查",
        "",
        f"- eval_page_cap_violations：{q.get('eval_page_cap_violations')}",
        f"- train_page_reuse_count：{q.get('train_page_reuse_count')}",
        f"- doc_split_violations：`{json.dumps(q['doc_split_violations'], ensure_ascii=False)}`",
        f"- builder_error_count：{q['builder_error_count']}",
        "",
        "## 标注质量信号",
        "",
        f"- caption_text_rate：{q['caption_text_rate']:.4f}",
        f"- caption_bbox_rate：{q['caption_bbox_rate']:.4f}",
        f"- mean_non_abstain_core5：{q['mean_non_abstain_core5']:.2f}",
        f"- mean_region_count：{q['mean_region_count']:.2f}",
        f"- mean_candidate_evidence_count：{q['mean_candidate_evidence_count']:.2f}",
        "",
        "## SFT 轨迹",
        "",
        f"- sft_rows_total：{q['sft_rows_total']}",
        f"- sft_split_counts：`{json.dumps(q['sft_split_counts'], ensure_ascii=False)}`",
        f"- sft_action_counts：`{json.dumps(q['sft_action_counts'], ensure_ascii=False)}`",
        "",
        "## 说明",
        "",
        "- 这版是 silver 数据：目标框来自 v1.0 layout candidate cache，包括 PDF image block 和 OpenCV visual region。",
        "- 当前外部 OCR 后端未成功跑通；图注来自 PDF text layer。",
        "- val/test 是文档级 unseen PDF，且每页最多 1 条。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
