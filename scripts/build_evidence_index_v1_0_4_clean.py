#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from evidence_noise_rules import classify_evidence_row, evidence_text


DEFAULT_SOURCE_INDEX = "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建带噪声过滤标注的 v1.0.4 clean evidence index。")
    parser.add_argument("--source-index", default=DEFAULT_SOURCE_INDEX)
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--version", default="evidence_index_v1_0_4_clean")
    parser.add_argument("--keep-unusable", action="store_true", help="保留不可检索 chunk；默认过滤到 filtered_out_corpus_chunks.jsonl。")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-filtered-examples", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_index = Path(args.source_index)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.version}_{timestamp()}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_sidecar_files(source_index, output_dir)
    summary, filtered_examples = build_clean_corpus(source_index, output_dir, keep_unusable=args.keep_unusable, max_examples=args.max_filtered_examples)

    manifest = {
        "created_at": now(),
        "builder": "scripts/build_evidence_index_v1_0_4_clean.py",
        "source_index": str(source_index),
        "output_dir": str(output_dir),
        "keep_unusable": bool(args.keep_unusable),
        "summary": summary,
        "artifacts": {
            "corpus_chunks": str(output_dir / "corpus_chunks.jsonl"),
            "filtered_out_corpus_chunks": str(output_dir / "filtered_out_corpus_chunks.jsonl"),
            "filtered_examples": str(output_dir / "filtered_examples.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    write_jsonl(output_dir / "filtered_examples.jsonl", filtered_examples)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_clean_corpus(
    source_index: Path,
    output_dir: Path,
    *,
    keep_unusable: bool,
    max_examples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    input_path = source_index / "corpus_chunks.jsonl"
    output_path = output_dir / "corpus_chunks.jsonl"
    filtered_path = output_dir / "filtered_out_corpus_chunks.jsonl"

    total = 0
    kept = 0
    filtered = 0
    type_counts: Counter[str] = Counter()
    kept_type_counts: Counter[str] = Counter()
    filtered_type_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    filtered_examples: list[dict[str, Any]] = []

    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout, filtered_path.open("w", encoding="utf-8") as ffiltered:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            cls = classify_evidence_row(row)
            clean_type = cls["clean_evidence_type"]
            type_counts[clean_type] += 1
            for reason in cls["noise_reasons"]:
                reason_counts[str(reason)] += 1

            annotated = dict(row)
            annotated["raw_evidence_type"] = row.get("evidence_type")
            annotated["clean_evidence_type"] = clean_type
            annotated["noise_score"] = cls["noise_score"]
            annotated["noise_reasons"] = cls["noise_reasons"]
            annotated["usable_for_claim"] = cls["usable_for_claim"]
            annotated["usable_for_retrieval"] = cls["usable_for_retrieval"]

            if cls["usable_for_retrieval"] or keep_unusable:
                fout.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                kept += 1
                kept_type_counts[clean_type] += 1
            else:
                ffiltered.write(json.dumps(annotated, ensure_ascii=False) + "\n")
                filtered += 1
                filtered_type_counts[clean_type] += 1
                if len(filtered_examples) < max_examples:
                    filtered_examples.append(example_row(annotated))

    summary = {
        "source_corpus_chunks": total,
        "clean_corpus_chunks": kept,
        "filtered_out_chunks": filtered,
        "filtered_out_rate": round(filtered / max(1, total), 6),
        "type_counts": dict(type_counts),
        "kept_type_counts": dict(kept_type_counts),
        "filtered_type_counts": dict(filtered_type_counts),
        "noise_reason_counts": dict(reason_counts),
    }
    return summary, filtered_examples


def copy_sidecar_files(source_index: Path, output_dir: Path) -> None:
    skip = {"corpus_chunks.jsonl", "manifest.json", "构建报告.md"}
    for path in source_index.iterdir():
        if not path.is_file() or path.name in skip:
            continue
        shutil.copy2(path, output_dir / path.name)
    if (source_index / "manifest.json").exists():
        shutil.copy2(source_index / "manifest.json", output_dir / "source_manifest.json")
    if (source_index / "构建报告.md").exists():
        shutil.copy2(source_index / "构建报告.md", output_dir / "构建报告.v0_3_1.md")


def example_row(row: dict[str, Any]) -> dict[str, Any]:
    text = evidence_text(row).replace("\n", " ")
    return {
        "evidence_id": row.get("evidence_id"),
        "source_file": row.get("source_file"),
        "page_start": row.get("page_start") if row.get("page_start") is not None else row.get("page"),
        "page_end": row.get("page_end"),
        "clean_evidence_type": row.get("clean_evidence_type"),
        "noise_score": row.get("noise_score"),
        "noise_reasons": row.get("noise_reasons"),
        "snippet": text[:500],
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# v1.0.4 Clean Evidence Index 构建报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- 源索引：`{manifest['source_index']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        f"- 是否保留不可检索 chunk：{manifest['keep_unusable']}",
        "",
        "## 总体统计",
        "",
        f"- 源 corpus_chunks：{summary['source_corpus_chunks']}",
        f"- clean corpus_chunks：{summary['clean_corpus_chunks']}",
        f"- filtered out chunks：{summary['filtered_out_chunks']}",
        f"- filtered out rate：{summary['filtered_out_rate']}",
        "",
        "## 原始类型分布",
        "",
    ]
    for key, value in sorted(summary["type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 保留类型分布", ""])
    for key, value in sorted(summary["kept_type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 过滤类型分布", ""])
    for key, value in sorted(summary["filtered_type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 噪声原因分布", ""])
    for key, value in sorted(summary["noise_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 产物", ""])
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
