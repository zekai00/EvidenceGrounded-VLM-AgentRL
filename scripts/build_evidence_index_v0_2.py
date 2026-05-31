#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz


LEGACY_FILES = ["chunks.jsonl", "documents.jsonl", "pages.jsonl", "images.jsonl", "source_aliases.json", "manifest.json"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v0.2 offline evidence index for EvidenceGrounded-VLM-AgentRL.")
    parser.add_argument("--authority-corpus-root", default="/root/datasets/chinese_landscape_authority_corpus")
    parser.add_argument("--legacy-evidence-store", default="/root/datasets/evidence_grounded_vlm_agentrl/evidence_store_legacy_milvus_20260530_1625")
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--version", default="evidence_index_v0_2")
    parser.add_argument("--chunk-chars", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=160)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--low-text-page-threshold", type=int, default=40)
    parser.add_argument("--limit-pdfs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = Path(args.output_root)
    output_dir = output_root / f"{args.version}_{now}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    authority_root = Path(args.authority_corpus_root)
    sources = load_authority_sources(authority_root / "metadata" / "sources.jsonl")
    if args.limit_pdfs:
        sources = sources[: args.limit_pdfs]

    rows_authority = []
    rows_page_spans = []
    rows_document_spans = []
    rows_corpus_chunks = []
    low_text_pages = []
    errors = []
    stats = Counter()

    for source in sources:
        authority = normalize_authority_source(source)
        rows_authority.append(authority)
        local_path = Path(str(source.get("local_path") or ""))
        if not local_path.exists() or local_path.suffix.lower() != ".pdf":
            continue
        try:
            page_texts = extract_pdf_page_spans(local_path, authority, args, rows_page_spans, low_text_pages)
            doc_chunks = build_document_chunks(page_texts, authority, args)
            rows_document_spans.extend(doc_chunks)
            for chunk in doc_chunks:
                corpus = dict(chunk)
                corpus["evidence_id"] = evidence_id("corpus", authority["source_id"], corpus.get("page_start"), corpus["text"])
                corpus["index_name"] = "corpus_chunks"
                corpus["retrieval_scope_base"] = "corpus"
                rows_corpus_chunks.append(corpus)
            stats["pdfs_parsed"] += 1
            stats["pdf_pages"] += len(page_texts)
        except Exception as exc:
            errors.append({"source_id": authority["source_id"], "path": str(local_path), "error": f"{type(exc).__name__}: {exc}"})

    legacy_map, legacy_rows = import_legacy_chunks(Path(args.legacy_evidence_store))
    rows_corpus_chunks.extend(legacy_rows)

    text_source_rows = import_text_sources(authority_root / "metadata" / "text_sources", sources, args)
    rows_document_spans.extend(text_source_rows["document_spans"])
    rows_corpus_chunks.extend(text_source_rows["corpus_chunks"])

    write_jsonl(output_dir / "authority_sources.jsonl", rows_authority)
    write_jsonl(output_dir / "page_spans.jsonl", rows_page_spans)
    write_jsonl(output_dir / "document_spans.jsonl", rows_document_spans)
    write_jsonl(output_dir / "corpus_chunks.jsonl", rows_corpus_chunks)
    write_json(output_dir / "legacy_chunk_map.json", legacy_map)
    write_jsonl(output_dir / "low_text_pages.jsonl", low_text_pages)
    write_jsonl(output_dir / "errors.jsonl", errors)

    summary = build_summary(args, output_dir, rows_authority, rows_page_spans, rows_document_spans, rows_corpus_chunks, low_text_pages, legacy_rows, text_source_rows, errors, stats)
    write_json(output_dir / "manifest.json", summary)
    write_report(output_dir / "构建报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def load_authority_sources(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_authority_source(source: dict[str, Any]) -> dict[str, Any]:
    local_path = str(source.get("local_path") or "")
    return {
        "source_id": str(source.get("id") or stable_hash(local_path or source.get("filename") or source.get("title") or "source")),
        "title": source.get("title") or "",
        "author": source.get("author") or "",
        "category": source.get("category") or "",
        "source_type": source.get("source_type") or "",
        "authority_level": source.get("authority_level") or "",
        "authority_weight": source.get("authority_weight"),
        "curation_axis": source.get("curation_axis") or "",
        "dynasties": source.get("dynasties") or [],
        "topics": source.get("topics") or [],
        "source_url": source.get("source_url") or "",
        "landing_page": source.get("landing_page") or "",
        "license_note": source.get("license_note") or "",
        "filename": source.get("filename") or Path(local_path).name,
        "local_path": local_path,
        "sha256": source.get("sha256") or "",
        "download_status": source.get("download_status") or "",
        "import_priority": source.get("import_priority") or "",
    }


def extract_pdf_page_spans(
    path: Path,
    authority: dict[str, Any],
    args: argparse.Namespace,
    rows_page_spans: list[dict[str, Any]],
    low_text_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    page_texts = []
    with fitz.open(path) as doc:
        for page_index, page in enumerate(doc):
            page_num = page_index + 1
            blocks = page.get_text("blocks")
            block_rows = []
            page_text_parts = []
            for block_index, block in enumerate(blocks):
                if len(block) < 5:
                    continue
                text = clean_text(str(block[4]))
                if len(text) < args.min_text_chars:
                    continue
                bbox = normalize_bbox([float(block[0]), float(block[1]), float(block[2]), float(block[3])], page.rect)
                row = make_evidence_row(
                    evidence_id=evidence_id("page", authority["source_id"], page_num, block_index, text),
                    index_name="page_spans",
                    retrieval_scope_base="document",
                    evidence_type="page_span",
                    authority=authority,
                    page=page_num,
                    page_start=page_num,
                    page_end=page_num,
                    text=text,
                    bbox=bbox,
                    source_quality="pdf_text_layer",
                    citation_level="page_span",
                    extra={"block_index": block_index},
                )
                block_rows.append(row)
                page_text_parts.append(text)
            rows_page_spans.extend(block_rows)
            page_text = clean_text("\n".join(page_text_parts))
            page_texts.append({"page": page_num, "text": page_text, "source_quality": "pdf_text_layer"})
            if len(page_text) < args.low_text_page_threshold:
                low_text_pages.append(
                    {
                        "source_id": authority["source_id"],
                        "source_file": authority["filename"],
                        "local_path": authority["local_path"],
                        "page": page_num,
                        "text_chars": len(page_text),
                        "reason": "low_pdf_text_layer_chars",
                        "recommended_fallback": "ocr_then_vlm_if_needed",
                    }
                )
    return page_texts


def build_document_chunks(page_texts: list[dict[str, Any]], authority: dict[str, Any], args: argparse.Namespace) -> list[dict[str, Any]]:
    chunks = []
    current_parts: list[str] = []
    current_start = None
    current_end = None
    for page in page_texts:
        text = page.get("text") or ""
        if not text:
            continue
        for piece in split_text(text, args.chunk_chars, args.chunk_overlap):
            if current_start is None:
                current_start = page["page"]
            current_end = page["page"]
            current_parts.append(piece)
            joined = clean_text("\n".join(current_parts))
            if len(joined) >= args.chunk_chars:
                chunks.append(make_document_chunk(authority, current_start, current_end, joined))
                overlap_text = joined[-args.chunk_overlap :] if args.chunk_overlap > 0 else ""
                current_parts = [overlap_text] if overlap_text else []
                current_start = current_end
    joined = clean_text("\n".join(current_parts))
    if joined:
        chunks.append(make_document_chunk(authority, current_start or 1, current_end or current_start or 1, joined))
    return chunks


def make_document_chunk(authority: dict[str, Any], page_start: int, page_end: int, text: str) -> dict[str, Any]:
    return make_evidence_row(
        evidence_id=evidence_id("doc", authority["source_id"], page_start, text),
        index_name="document_spans",
        retrieval_scope_base="same_document",
        evidence_type="document_chunk",
        authority=authority,
        page=None,
        page_start=page_start,
        page_end=page_end,
        text=text,
        bbox=None,
        source_quality="pdf_text_layer",
        citation_level="page_range_chunk",
        extra={},
    )


def import_legacy_chunks(legacy_root: Path) -> tuple[dict[str, str], list[dict[str, Any]]]:
    rows = []
    mapping = {}
    chunks_path = legacy_root / "chunks.jsonl"
    if not chunks_path.exists():
        return mapping, rows
    with chunks_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            old_id = str(item.get("chunk_id") or "")
            text = clean_text("\n".join(str(item.get(key) or "") for key in ["contextual_prefix", "raw_chunk_text"]))
            if not old_id or not text:
                continue
            ev_id = evidence_id("legacy", old_id, item.get("doc_id"), text)
            mapping[old_id] = ev_id
            source_file = str(item.get("source_file") or "")
            row = {
                "evidence_id": ev_id,
                "legacy_chunk_id": old_id,
                "legacy_milvus_id": item.get("legacy_milvus_id"),
                "index_name": "corpus_chunks",
                "retrieval_scope_base": "corpus",
                "evidence_type": "legacy_milvus_chunk",
                "doc_id": item.get("doc_id"),
                "source_id": "legacy_milvus",
                "source_file": source_file,
                "source_stem": Path(source_file).stem,
                "title": item.get("title") or "",
                "author": "",
                "category": "legacy_milvus",
                "source_type": "legacy_milvus_pdf_chunk",
                "authority_level": "legacy",
                "authority_weight": 0.6,
                "source_url": "",
                "landing_page": "",
                "page": None,
                "page_start": item.get("page_start"),
                "page_end": item.get("page_end"),
                "bbox": None,
                "text": text,
                "raw_chunk_text": item.get("raw_chunk_text") or "",
                "contextual_prefix": item.get("contextual_prefix") or "",
                "source_quality": "legacy_milvus",
                "citation_level": "chunk",
                "quality": item.get("quality") or {},
                "metadata": item.get("metadata") or {},
            }
            rows.append(row)
    return mapping, rows


def import_text_sources(text_sources_dir: Path, sources: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    by_stem = {Path(str(source.get("filename") or "")).stem: normalize_authority_source(source) for source in sources}
    by_id = {str(source.get("id") or ""): normalize_authority_source(source) for source in sources}
    document_rows = []
    corpus_rows = []
    if not text_sources_dir.exists():
        return {"document_spans": document_rows, "corpus_chunks": corpus_rows}
    for path in sorted(text_sources_dir.glob("*.txt")):
        text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
        if not text:
            continue
        source_key = path.stem.replace("_text_pdf", "")
        authority = by_id.get(source_key) or by_stem.get(source_key) or {
            "source_id": source_key,
            "title": path.stem,
            "author": "",
            "category": "metadata_text_sources",
            "source_type": "metadata_text_source",
            "authority_level": "",
            "authority_weight": None,
            "curation_axis": "",
            "dynasties": [],
            "topics": [],
            "source_url": "",
            "landing_page": "",
            "license_note": "",
            "filename": path.name,
            "local_path": str(path),
            "sha256": "",
            "download_status": "text_source",
            "import_priority": "",
        }
        for index, chunk in enumerate(split_text(text, args.chunk_chars, args.chunk_overlap)):
            row = make_evidence_row(
                evidence_id=evidence_id("txt", authority["source_id"], index, chunk),
                index_name="document_spans",
                retrieval_scope_base="same_document",
                evidence_type="metadata_text_chunk",
                authority=authority,
                page=None,
                page_start=None,
                page_end=None,
                text=chunk,
                bbox=None,
                source_quality="metadata_text_source",
                citation_level="chunk",
                extra={"text_source_path": str(path), "chunk_index": index},
            )
            document_rows.append(row)
            corpus = dict(row)
            corpus["evidence_id"] = evidence_id("corpus_txt", authority["source_id"], index, chunk)
            corpus["index_name"] = "corpus_chunks"
            corpus["retrieval_scope_base"] = "corpus"
            corpus_rows.append(corpus)
    return {"document_spans": document_rows, "corpus_chunks": corpus_rows}


def make_evidence_row(
    *,
    evidence_id: str,
    index_name: str,
    retrieval_scope_base: str,
    evidence_type: str,
    authority: dict[str, Any],
    page: int | None,
    page_start: int | None,
    page_end: int | None,
    text: str,
    bbox: list[int] | None,
    source_quality: str,
    citation_level: str,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "evidence_id": evidence_id,
        "index_name": index_name,
        "retrieval_scope_base": retrieval_scope_base,
        "evidence_type": evidence_type,
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
        "page_start": page_start,
        "page_end": page_end,
        "bbox": bbox,
        "text": text,
        "source_quality": source_quality,
        "citation_level": citation_level,
        "quality": {
            "parse_status": "parsed",
            "needs_ocr": False,
            "needs_vlm": False,
        },
    }
    row.update(extra)
    return row


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    authority_rows: list[dict[str, Any]],
    page_rows: list[dict[str, Any]],
    document_rows: list[dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    low_text_pages: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
    text_source_rows: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
    stats: Counter,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "builder": "scripts/build_evidence_index_v0_2.py",
        "output_dir": str(output_dir),
        "authority_corpus_root": args.authority_corpus_root,
        "legacy_evidence_store": args.legacy_evidence_store,
        "authority_sources": len(authority_rows),
        "pdfs_parsed": stats["pdfs_parsed"],
        "pdf_pages": stats["pdf_pages"],
        "page_spans": len(page_rows),
        "document_spans": len(document_rows),
        "corpus_chunks": len(corpus_rows),
        "legacy_chunks_imported": len(legacy_rows),
        "metadata_text_document_spans": len(text_source_rows["document_spans"]),
        "metadata_text_corpus_chunks": len(text_source_rows["corpus_chunks"]),
        "low_text_pages": len(low_text_pages),
        "errors": len(errors),
        "source_quality_counts": dict(Counter(row.get("source_quality") for row in page_rows + document_rows + corpus_rows)),
        "authority_level_counts": dict(Counter(row.get("authority_level") for row in corpus_rows)),
        "category_counts": dict(Counter(row.get("category") for row in corpus_rows)),
        "limitations": [
            "This index is offline JSONL, not a Milvus collection.",
            "PDF text layer is parsed first; low-text pages are recorded for OCR/VLM fallback but not OCRed in the default full build.",
            "Legacy Milvus chunks are imported as low-authority corpus evidence for backward compatibility.",
        ],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v0.2 Evidence Index 构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- authority corpus：`{summary['authority_corpus_root']}`",
        f"- legacy evidence store：`{summary['legacy_evidence_store']}`",
        "",
        "## 规模",
        "",
        f"- authority_sources：{summary['authority_sources']}",
        f"- pdfs_parsed：{summary['pdfs_parsed']}",
        f"- pdf_pages：{summary['pdf_pages']}",
        f"- page_spans：{summary['page_spans']}",
        f"- document_spans：{summary['document_spans']}",
        f"- corpus_chunks：{summary['corpus_chunks']}",
        f"- legacy_chunks_imported：{summary['legacy_chunks_imported']}",
        f"- low_text_pages：{summary['low_text_pages']}",
        f"- errors：{summary['errors']}",
        "",
        "## 分布",
        "",
        f"- source_quality_counts：`{summary['source_quality_counts']}`",
        f"- authority_level_counts：`{summary['authority_level_counts']}`",
        f"- category_counts：`{summary['category_counts']}`",
        "",
        "## 限制",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_text(text: str, chunk_chars: int, overlap: int) -> list[str]:
    text = clean_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def normalize_bbox(values: list[float], rect: fitz.Rect) -> list[int]:
    if rect.width <= 0 or rect.height <= 0:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = values
    return [
        int(round(x0 / rect.width * 1000)),
        int(round(y0 / rect.height * 1000)),
        int(round(x1 / rect.width * 1000)),
        int(round(y1 / rect.height * 1000)),
    ]


def clean_text(value: str) -> str:
    value = str(value).replace("\u0000", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def evidence_id(*parts: Any) -> str:
    return "ev_" + stable_hash("|".join(str(part) for part in parts if part is not None))[:20]


def stable_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
