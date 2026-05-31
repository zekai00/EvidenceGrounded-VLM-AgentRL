#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from build_evidence_index_v0_3 import (
    clean_text,
    evidence_id,
    make_display_snippet,
    make_evidence_summary,
    normalize_authority_source,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge VLM OCR fallback pages into a v0.3 evidence index copy.")
    parser.add_argument("--source-index-dir", default="/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_20260530_2149")
    parser.add_argument(
        "--fallback-dir",
        default="/root/datasets/evidence_grounded_vlm_agentrl/low_text_vlm_fallback_v0_2_qwen36flash_smoke_20260530_2257",
    )
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--version", default="evidence_index_v0_3_1_low_text_vlm_smoke")
    parser.add_argument("--min-chars", type=int, default=40)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now().strftime("%Y%m%d_%H%M")
    source_dir = Path(args.source_index_dir)
    fallback_dir = Path(args.fallback_dir)
    output_dir = Path(args.output_root) / f"{args.version}_{now}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)

    shutil.copytree(source_dir, output_dir)

    authority_map = load_authority_map(output_dir / "authority_sources.jsonl")
    fallback_rows = read_jsonl(fallback_dir / "fallback_pages.jsonl")
    usable = [
        row
        for row in fallback_rows
        if row.get("ok") and row.get("is_readable") and len(clean_text(row.get("transcription") or "")) >= args.min_chars
    ]

    page_rows: list[dict[str, Any]] = []
    document_rows: list[dict[str, Any]] = []
    corpus_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in usable:
        authority = resolve_authority(row, authority_map)
        if not authority:
            skipped.append({"source_file": row.get("source_file"), "page": row.get("page"), "reason": "authority_not_found"})
            continue
        page_row, document_row, corpus_row = make_rows(row, authority, fallback_dir)
        page_rows.append(page_row)
        document_rows.append(document_row)
        corpus_rows.append(corpus_row)

    append_jsonl(output_dir / "page_spans.jsonl", page_rows)
    append_jsonl(output_dir / "document_spans.jsonl", document_rows)
    append_jsonl(output_dir / "corpus_chunks.jsonl", corpus_rows)

    summary = build_summary(args, source_dir, fallback_dir, output_dir, fallback_rows, usable, page_rows, document_rows, corpus_rows, skipped)
    write_json(output_dir / "manifest_v0_3_1_merge.json", summary)
    write_report(output_dir / "VLM低文本页合并报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def make_rows(row: dict[str, Any], authority: dict[str, Any], fallback_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    page = int(row.get("page") or 0)
    text = clean_text(row.get("transcription") or "")
    base_extra = {
        "raw_text": text,
        "display_snippet": make_display_snippet(text),
        "evidence_summary": make_evidence_summary(text),
        "source_quality": "vlm_ocr_fallback",
        "citation_level": "page_image_transcription",
        "fallback_provider": row.get("provider"),
        "fallback_model": row.get("model"),
        "fallback_image_mode": row.get("image_mode"),
        "fallback_confidence": row.get("confidence"),
        "fallback_page_type": row.get("page_type"),
        "fallback_notes": row.get("notes"),
        "fallback_image_path": row.get("image_path"),
        "fallback_dir": str(fallback_dir),
        "quality": {
            "parse_status": "vlm_ocr_fallback",
            "needs_ocr": False,
            "needs_vlm": False,
            "snippet_policy": "sentence_boundary_or_full_short_text",
            "silver_text": True,
            "requires_human_spot_check": True,
        },
    }
    common = {
        "source_id": authority["source_id"],
        "source_file": authority["filename"],
        "source_stem": Path(authority["filename"]).stem,
        "title": authority["title"],
        "author": authority["author"],
        "category": authority["category"],
        "source_type": authority["source_type"],
        "authority_level": authority["authority_level"],
        "authority_weight": authority["authority_weight"],
        "curation_axis": authority["curation_axis"],
        "topics": authority["topics"],
        "source_url": authority["source_url"],
        "landing_page": authority["landing_page"],
        "page": page,
        "page_start": page,
        "page_end": page,
        "bbox": None,
        "text": text,
        "raw_text": text,
        "clean_text": text,
        "display_snippet": base_extra["display_snippet"],
        "evidence_summary": base_extra["evidence_summary"],
        "source_quality": base_extra["source_quality"],
        "citation_level": base_extra["citation_level"],
        "quality": base_extra["quality"],
        "fallback_provider": base_extra["fallback_provider"],
        "fallback_model": base_extra["fallback_model"],
        "fallback_image_mode": base_extra["fallback_image_mode"],
        "fallback_confidence": base_extra["fallback_confidence"],
        "fallback_page_type": base_extra["fallback_page_type"],
        "fallback_notes": base_extra["fallback_notes"],
        "fallback_image_path": base_extra["fallback_image_path"],
        "fallback_dir": base_extra["fallback_dir"],
    }
    page_row = dict(common)
    page_row.update(
        {
            "evidence_id": evidence_id("vlm_page_v0_3_1", authority["source_id"], page, text),
            "index_name": "page_spans",
            "retrieval_scope_base": "document",
            "evidence_type": "vlm_ocr_page_transcription",
            "block_index": None,
        }
    )

    document_row = dict(common)
    document_row.update(
        {
            "evidence_id": evidence_id("vlm_doc_v0_3_1", authority["source_id"], page, text),
            "index_name": "document_spans",
            "retrieval_scope_base": "same_document",
            "evidence_type": "vlm_ocr_page_chunk",
            "page": None,
            "segmentation": "vlm_page_transcription",
            "segmentation_unit_count": 1,
        }
    )

    corpus_row = dict(document_row)
    corpus_row.update(
        {
            "evidence_id": evidence_id("vlm_corpus_v0_3_1", authority["source_id"], page, text),
            "index_name": "corpus_chunks",
            "retrieval_scope_base": "corpus",
        }
    )
    return page_row, document_row, corpus_row


def load_authority_map(path: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        norm = normalize_authority_source(row)
        mapping[f"id::{norm['source_id']}"] = norm
        mapping[f"file::{norm['filename']}"] = norm
    return mapping


def resolve_authority(row: dict[str, Any], authority_map: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    source_id = str(row.get("source_id") or "")
    source_file = str(row.get("source_file") or "")
    return authority_map.get(f"id::{source_id}") or authority_map.get(f"file::{source_file}")


def build_summary(
    args: argparse.Namespace,
    source_dir: Path,
    fallback_dir: Path,
    output_dir: Path,
    fallback_rows: list[dict[str, Any]],
    usable: list[dict[str, Any]],
    page_rows: list[dict[str, Any]],
    document_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "builder": "scripts/merge_low_text_fallback_into_index_v0_3_1.py",
        "source_index_dir": str(source_dir),
        "fallback_dir": str(fallback_dir),
        "output_dir": str(output_dir),
        "min_chars": args.min_chars,
        "fallback_rows_total": len(fallback_rows),
        "usable_fallback_rows": len(usable),
        "skipped_rows": len(skipped),
        "added_page_spans": len(page_rows),
        "added_document_spans": len(document_rows),
        "added_corpus_chunks": len(corpus_rows),
        "fallback_model_counts": dict(Counter(row.get("fallback_model") for row in page_rows)),
        "source_quality": "vlm_ocr_fallback",
        "citation_level": "page_image_transcription",
        "outputs": {
            "page_spans": str(output_dir / "page_spans.jsonl"),
            "document_spans": str(output_dir / "document_spans.jsonl"),
            "corpus_chunks": str(output_dir / "corpus_chunks.jsonl"),
            "manifest": str(output_dir / "manifest_v0_3_1_merge.json"),
            "report": str(output_dir / "VLM低文本页合并报告.md"),
        },
        "limitations": [
            "This smoke index only merges the pages processed by the fallback run, not all low_text_pages.",
            "VLM OCR fallback text is silver evidence and should be marked lower-trust than PDF text layer.",
            "Rows are page-level transcriptions; no bbox-level citation is available for these fallback rows.",
        ],
        "skipped_examples": skipped[:10],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v0.3.1 Low-Text VLM Fallback 合并报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 原始 index：`{summary['source_index_dir']}`",
        f"- fallback 目录：`{summary['fallback_dir']}`",
        f"- 输出 index：`{summary['output_dir']}`",
        "",
        "## 合并规模",
        "",
        f"- fallback_rows_total：{summary['fallback_rows_total']}",
        f"- usable_fallback_rows：{summary['usable_fallback_rows']}",
        f"- skipped_rows：{summary['skipped_rows']}",
        f"- added_page_spans：{summary['added_page_spans']}",
        f"- added_document_spans：{summary['added_document_spans']}",
        f"- added_corpus_chunks：{summary['added_corpus_chunks']}",
        f"- fallback_model_counts：`{summary['fallback_model_counts']}`",
        "",
        "## 证据等级",
        "",
        "- source_quality：`vlm_ocr_fallback`",
        "- citation_level：`page_image_transcription`",
        "- 含义：该证据来自 VLM 对整页扫描图的转写，可用于检索和弱引用；不提供 bbox 级别引用，也不等价于人工校勘文本。",
        "",
        "## 输出",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}：`{value}`")
    lines.extend(["", "## 限制", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
