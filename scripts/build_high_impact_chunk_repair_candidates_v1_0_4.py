#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INDEX = "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_llm_overlay_20260611_"
DEFAULT_ADJUDICATION = (
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "evidence_chunk_adjudication_llm_merged_v1_0_4_20260611_015644/adjudicated_samples.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为高影响 evidence chunk 生成 page_spans 回溯修复候选。")
    parser.add_argument("--index", required=True)
    parser.add_argument("--adjudication", default=DEFAULT_ADJUDICATION)
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--version", default="high_impact_chunk_repair_candidates_v1_0_4")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--page-window", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.version}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = Path(args.index)
    chunks = {str(row.get("evidence_id")): row for row in iter_jsonl(index_dir / "corpus_chunks.jsonl")}
    page_spans = load_page_spans(index_dir / "page_spans.jsonl")
    selected = select_high_impact_samples(Path(args.adjudication), chunks, args.limit)
    candidates = [make_candidate(sample, chunks, page_spans, args.page_window) for sample in selected]

    manifest = build_manifest(args, output_dir, candidates)
    write_jsonl(output_dir / "repair_candidates.jsonl", candidates)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "修复候选报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def select_high_impact_samples(path: Path, chunks: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in iter_jsonl(path):
        evidence_id = str(row.get("evidence_id") or "")
        chunk = chunks.get(evidence_id)
        if not chunk:
            continue
        adjudication = row.get("adjudication") or {}
        text = str(chunk.get("clean_text") or chunk.get("text") or "")
        score = 0
        if row.get("sample_bucket") == "balanced32_retrieve_hit":
            score += 5
        if row.get("adjudication_status") == "needs_llm_or_human_review":
            score += 4
        if (row.get("offline_vs_adjudicated") or {}).get("role_changed"):
            score += 3
        if text and not sentence_end_ok(text):
            score += 2
        if str(adjudication.get("evidence_role") or "") in {"ocr_noise", "front_matter", "bibliography", "toc"}:
            score += 1
        if score > 0:
            scored.append((score, row))
    scored.sort(key=lambda item: (-item[0], str(item[1].get("audit_id"))))
    return [row for _score, row in scored[:limit]]


def make_candidate(
    sample: dict[str, Any],
    chunks: dict[str, dict[str, Any]],
    page_spans: dict[tuple[str, int], list[dict[str, Any]]],
    page_window: int,
) -> dict[str, Any]:
    evidence_id = str(sample.get("evidence_id") or "")
    chunk = chunks[evidence_id]
    source_file = str(chunk.get("source_file") or "")
    page_start = to_int(chunk.get("page_start") if chunk.get("page_start") is not None else chunk.get("page"))
    page_end = to_int(chunk.get("page_end")) or page_start
    original_text = clean_text(str(chunk.get("clean_text") or chunk.get("text") or ""))
    repaired_parts: list[str] = []
    if page_start is not None and page_end is not None:
        for page in range(max(1, page_start - page_window), page_end + page_window + 1):
            repaired_parts.extend(str(row.get("text") or "") for row in page_spans.get((source_file, page), []))
    repaired_text = clean_text("\n".join(repaired_parts)) or original_text
    adjudication = sample.get("adjudication") or {}
    return {
        "audit_id": sample.get("audit_id"),
        "chunk_id": evidence_id,
        "evidence_id": evidence_id,
        "source_file": source_file,
        "page_start": page_start,
        "page_end": page_end,
        "sample_bucket": sample.get("sample_bucket"),
        "adjudication_status": sample.get("adjudication_status"),
        "adjudicated_evidence_role": adjudication.get("evidence_role"),
        "offline_role": (sample.get("offline_label") or {}).get("evidence_role_pred"),
        "role_changed": (sample.get("offline_vs_adjudicated") or {}).get("role_changed"),
        "repair_strategy": "page_spans_same_page_range" if repaired_text != original_text else "original_text_fallback",
        "original_len": len(original_text),
        "repaired_len": len(repaired_text),
        "original_sentence_end_ok": sentence_end_ok(original_text),
        "repaired_sentence_end_ok": sentence_end_ok(repaired_text),
        "original_text": original_text[:2400],
        "repaired_text": repaired_text[:3600],
    }


def load_page_spans(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int], list[dict[str, Any]]] = {}
    if not path.exists():
        return out
    for row in iter_jsonl(path):
        source_file = str(row.get("source_file") or "")
        page = to_int(row.get("page"))
        if not source_file or page is None:
            continue
        out.setdefault((source_file, page), []).append(row)
    for rows in out.values():
        rows.sort(key=lambda item: int(item.get("block_index") or 0))
    return out


def build_manifest(args: argparse.Namespace, output_dir: Path, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    role_counts = Counter(str(row.get("adjudicated_evidence_role")) for row in candidates)
    status_counts = Counter(str(row.get("adjudication_status")) for row in candidates)
    strategy_counts = Counter(str(row.get("repair_strategy")) for row in candidates)
    return {
        "created_at": now(),
        "index": args.index,
        "adjudication": args.adjudication,
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "page_window": args.page_window,
        "role_counts": dict(role_counts),
        "status_counts": dict(status_counts),
        "strategy_counts": dict(strategy_counts),
        "original_bad_sentence_end_count": sum(not row.get("original_sentence_end_ok") for row in candidates),
        "repaired_bad_sentence_end_count": sum(not row.get("repaired_sentence_end_ok") for row in candidates),
        "artifacts": {
            "repair_candidates": str(output_dir / "repair_candidates.jsonl"),
            "report": str(output_dir / "修复候选报告.md"),
        },
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 高影响 Chunk 修复候选报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- index：`{manifest['index']}`",
        f"- adjudication：`{manifest['adjudication']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        f"- 候选数：{manifest['candidate_count']}",
        f"- page_window：{manifest['page_window']}",
        f"- 原始句末不完整候选：{manifest['original_bad_sentence_end_count']}",
        f"- 修复后句末不完整候选：{manifest['repaired_bad_sentence_end_count']}",
        "",
        "## role 分布",
        "",
    ]
    append_counts(lines, manifest["role_counts"])
    lines.extend(["", "## status 分布", ""])
    append_counts(lines, manifest["status_counts"])
    lines.extend(["", "## repair strategy 分布", ""])
    append_counts(lines, manifest["strategy_counts"])
    lines.extend(["", "## 产物", ""])
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_counts(lines: list[str], counts: dict[str, int]) -> None:
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")


def sentence_end_ok(value: str) -> bool:
    return clean_text(value).endswith(("。", "！", "？", "；", ".", "!", "?", ";", "”", "’", "》", "】", "）", ")"))


def clean_text(value: str) -> str:
    value = re.sub(r"[ \t\r\f\v]+", " ", str(value or ""))
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    raise SystemExit(main())
