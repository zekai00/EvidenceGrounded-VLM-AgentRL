#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_INDEX = "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_clean_20260610_2046"
DEFAULT_ADJUDICATION = (
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "evidence_chunk_adjudication_llm_merged_v1_0_4_20260611_015644/adjudicated_samples.jsonl"
)

NO_RETRIEVAL_ROLES = {"toc", "bibliography", "front_matter", "back_matter", "ocr_noise"}
NO_CLAIM_ROLES = NO_RETRIEVAL_ROLES | {"low_value_background"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建带 LLM adjudication overlay 的 v1.0.4 evidence index。")
    parser.add_argument("--base-index", default=DEFAULT_BASE_INDEX)
    parser.add_argument("--adjudication", default=DEFAULT_ADJUDICATION)
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--version", default="evidence_index_v1_0_4_llm_overlay")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_index = Path(args.base_index)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.version}_{timestamp()}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    overlay = load_adjudication(Path(args.adjudication))
    copy_sidecar_files(base_index, output_dir)
    summary = build_overlay_corpus(base_index, output_dir, overlay)
    manifest = {
        "created_at": now(),
        "builder": "scripts/build_evidence_index_v1_0_4_llm_overlay.py",
        "base_index": str(base_index),
        "adjudication": args.adjudication,
        "output_dir": str(output_dir),
        "summary": summary,
        "artifacts": {
            "corpus_chunks": str(output_dir / "corpus_chunks.jsonl"),
            "adjudication_overlay": str(output_dir / "adjudication_overlay.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    write_jsonl(output_dir / "adjudication_overlay.jsonl", overlay.values())
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def load_adjudication(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        evidence_id = str(row.get("evidence_id") or "")
        adjudication = row.get("adjudication") or {}
        if not evidence_id or not isinstance(adjudication, dict):
            continue
        allowed = [str(item) for item in adjudication.get("claim_allowed_fields") or [] if str(item)]
        role = str(adjudication.get("evidence_role") or "")
        status = str(row.get("adjudication_status") or "")
        usable_for_claim = status == "accepted_auto" and role not in NO_CLAIM_ROLES and bool(allowed)
        usable_for_retrieval = role not in NO_RETRIEVAL_ROLES
        out[evidence_id] = {
            "evidence_id": evidence_id,
            "adjudication_status": status,
            "adjudicated_evidence_role": role,
            "adjudicated_claim_allowed_fields": allowed,
            "adjudication_confidence": adjudication.get("confidence"),
            "adjudication_needs_review": adjudication.get("needs_review"),
            "adjudication_provider": adjudication.get("provider"),
            "adjudication_label_source": adjudication.get("label_source"),
            "adjudication_model": adjudication.get("model"),
            "adjudication_rationale": adjudication.get("rationale"),
            "usable_for_claim_by_adjudication": usable_for_claim,
            "usable_for_retrieval_by_adjudication": usable_for_retrieval,
        }
    return out


def build_overlay_corpus(base_index: Path, output_dir: Path, overlay: dict[str, dict[str, Any]]) -> dict[str, Any]:
    input_path = base_index / "corpus_chunks.jsonl"
    output_path = output_dir / "corpus_chunks.jsonl"
    total = 0
    overlay_hits = 0
    role_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    retrieval_blocked = 0
    claim_blocked = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            evidence_id = str(row.get("evidence_id") or "")
            adjudication = overlay.get(evidence_id)
            if adjudication:
                overlay_hits += 1
                role_counts[str(adjudication.get("adjudicated_evidence_role") or "")] += 1
                status_counts[str(adjudication.get("adjudication_status") or "")] += 1
                row["base_usable_for_claim"] = row.get("usable_for_claim")
                row["base_usable_for_retrieval"] = row.get("usable_for_retrieval")
                row.update(adjudication)
                row["claim_allowed_fields"] = adjudication.get("adjudicated_claim_allowed_fields") or []
                if not adjudication.get("usable_for_claim_by_adjudication"):
                    row["usable_for_claim"] = False
                    claim_blocked += 1
                if not adjudication.get("usable_for_retrieval_by_adjudication"):
                    row["usable_for_retrieval"] = False
                    retrieval_blocked += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "base_corpus_chunks": total,
        "adjudication_rows": len(overlay),
        "overlay_hits": overlay_hits,
        "missing_from_base_index": len(overlay) - overlay_hits,
        "retrieval_blocked_by_adjudication": retrieval_blocked,
        "claim_blocked_by_adjudication": claim_blocked,
        "adjudication_status_counts": dict(status_counts),
        "adjudicated_role_counts": dict(role_counts),
    }


def copy_sidecar_files(base_index: Path, output_dir: Path) -> None:
    skip = {"corpus_chunks.jsonl", "manifest.json", "构建报告.md", "adjudication_overlay.jsonl"}
    for path in base_index.iterdir():
        if not path.is_file() or path.name in skip:
            continue
        shutil.copy2(path, output_dir / path.name)
    if (base_index / "manifest.json").exists():
        shutil.copy2(base_index / "manifest.json", output_dir / "base_manifest.json")
    if (base_index / "构建报告.md").exists():
        shutil.copy2(base_index / "构建报告.md", output_dir / "构建报告.base.md")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    summary = manifest["summary"]
    lines = [
        "# v1.0.4 LLM Overlay Evidence Index 构建报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- base index：`{manifest['base_index']}`",
        f"- adjudication：`{manifest['adjudication']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        "",
        "## 总体统计",
        "",
        f"- base corpus chunks：{summary['base_corpus_chunks']}",
        f"- adjudication rows：{summary['adjudication_rows']}",
        f"- overlay hits：{summary['overlay_hits']}",
        f"- missing from base index：{summary['missing_from_base_index']}",
        f"- retrieval blocked by adjudication：{summary['retrieval_blocked_by_adjudication']}",
        f"- claim blocked by adjudication：{summary['claim_blocked_by_adjudication']}",
        "",
        "## adjudication status",
        "",
    ]
    append_counts(lines, summary["adjudication_status_counts"])
    lines.extend(["", "## adjudicated role", ""])
    append_counts(lines, summary["adjudicated_role_counts"])
    lines.extend(["", "## 产物", ""])
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_counts(lines: list[str], counts: dict[str, int]) -> None:
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")


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
