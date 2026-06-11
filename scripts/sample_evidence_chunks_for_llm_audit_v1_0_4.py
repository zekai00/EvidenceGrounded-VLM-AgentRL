#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from source_registry_rules_v1_0_4 import chunk_offline_label, chunk_text


DEFAULT_INDEX = "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_clean_20260610_2046"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="为 v1.0.4 LLM/VLM evidence role 审计采样 chunk。")
    parser.add_argument("--evidence-index", default=DEFAULT_INDEX)
    parser.add_argument("--source-registry", required=True)
    parser.add_argument("--rollout-hits", nargs="*", default=[])
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--version", default="evidence_chunk_audit_seed_v1_0_4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.version}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(Path(args.source_registry))
    chunks = list(iter_jsonl(Path(args.evidence_index) / "corpus_chunks.jsonl"))
    chunks_by_id = {str(row.get("evidence_id")): row for row in chunks}
    high_impact_ids = load_rollout_evidence_ids(args.rollout_hits)
    samples = sample_chunks(chunks, chunks_by_id, registry, high_impact_ids, args.sample_size)

    write_jsonl(output_dir / "audit_samples.jsonl", samples)
    manifest = build_manifest(args, output_dir, samples)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "采样报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def sample_chunks(
    chunks: list[dict[str, Any]],
    chunks_by_id: dict[str, dict[str, Any]],
    registry: dict[str, dict[str, Any]],
    high_impact_ids: list[str],
    sample_size: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def add(row: dict[str, Any], bucket: str) -> None:
        eid = str(row.get("evidence_id"))
        if not eid or eid in selected_ids or len(selected) >= sample_size:
            return
        selected_ids.add(eid)
        source = registry.get(str(row.get("source_id") or "")) or registry.get(str(row.get("source_file") or ""))
        label = chunk_offline_label(row, source)
        selected.append(
            {
                "audit_id": f"audit_{len(selected):04d}",
                "sample_bucket": bucket,
                "evidence_id": row.get("evidence_id"),
                "source_id": row.get("source_id"),
                "source_file": row.get("source_file"),
                "page_start": row.get("page_start") if row.get("page_start") is not None else row.get("page"),
                "page_end": row.get("page_end"),
                "clean_evidence_type": row.get("clean_evidence_type"),
                "source_type": row.get("source_type"),
                "source_authority": (source or {}).get("source_authority"),
                "evidence_roles_source": (source or {}).get("evidence_roles"),
                "claim_allowed_fields_source": (source or {}).get("claim_allowed_fields"),
                "text": chunk_text(row)[:1800],
                "offline_label": label,
                "llm_prompt": build_llm_prompt(row, source, label),
                "audit_status": "pending_llm_or_human_review",
            }
        )

    high_impact_target = min(sample_size, max(40, sample_size // 3))
    high_impact_added = 0
    for eid in high_impact_ids:
        row = chunks_by_id.get(eid)
        if row:
            added_before = len(selected)
            add(row, "balanced32_retrieve_hit")
            if len(selected) > added_before:
                high_impact_added += 1
        if high_impact_added >= high_impact_target:
            break

    buckets = [
        ("legacy_project_pdf", lambda r, s: r.get("source_type") == "legacy_project_pdf"),
        ("museum_catalog", lambda r, s: r.get("source_type") in {"museum_catalog", "symposium_proceedings"}),
        ("object_metadata", lambda r, s: r.get("source_type") == "museum_collection_entry_text_pdf"),
        ("ancient_theory", lambda r, s: str(r.get("source_type") or "").startswith("ancient_theory")),
        ("palace_museum", lambda r, s: str(r.get("source_type") or "").startswith("palace_museum")),
        ("figure_reference_text", lambda r, s: r.get("clean_evidence_type") == "figure_reference_text"),
    ]
    per_bucket_target = max(12, sample_size // max(1, len(buckets) + 1))
    for bucket, pred in buckets:
        added_before = len(selected)
        for row in chunks:
            source = registry.get(str(row.get("source_id") or "")) or registry.get(str(row.get("source_file") or ""))
            if pred(row, source):
                add(row, bucket)
            if len(selected) - added_before >= per_bucket_target or len(selected) >= sample_size:
                break
        if len(selected) >= sample_size:
            break

    for row in chunks:
        add(row, "fill_remaining")
        if len(selected) >= sample_size:
            break
    return selected


def build_llm_prompt(row: dict[str, Any], source: dict[str, Any] | None, label: dict[str, Any]) -> str:
    return (
        "请判断下面 evidence chunk 的用途。只需要判断它能安全支持哪些 claim 字段，"
        "不要做完整美术史标注。\n"
        f"source_file: {row.get('source_file')}\n"
        f"source_type: {row.get('source_type')}\n"
        f"source_authority: {(source or {}).get('source_authority')}\n"
        f"source evidence_roles: {(source or {}).get('evidence_roles')}\n"
        f"offline_guess: {label.get('evidence_role_pred')}\n"
        "候选 evidence_role: object_metadata, caption_or_plate, style_analysis, historical_context, "
        "theory_primary_text, teaching_overview, low_value_background, toc, bibliography, "
        "front_matter, back_matter, ocr_noise。\n"
        "请输出 JSON，字段包括 evidence_role, claim_allowed_fields, confidence, needs_review, rationale。\n"
        "chunk_text:\n"
        f"{chunk_text(row)[:1800]}"
    )


def load_registry(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        if row.get("id"):
            out[str(row["id"])] = row
        if row.get("source_file"):
            out[str(row["source_file"])] = row
        if row.get("filename"):
            out[str(row["filename"])] = row
    return out


def load_rollout_evidence_ids(paths: list[str]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for row in iter_jsonl(p):
            for key in ("evidence_id",):
                eid = row.get(key)
                if eid and str(eid) not in seen:
                    seen.add(str(eid))
                    ids.append(str(eid))
            for result in row.get("results") or []:
                eid = result.get("evidence_id")
                if eid and str(eid) not in seen:
                    seen.add(str(eid))
                    ids.append(str(eid))
    return ids


def build_manifest(args: argparse.Namespace, output_dir: Path, samples: list[dict[str, Any]]) -> dict[str, Any]:
    bucket_counts = Counter(str(row.get("sample_bucket")) for row in samples)
    role_counts = Counter(str((row.get("offline_label") or {}).get("evidence_role_pred")) for row in samples)
    source_type_counts = Counter(str(row.get("source_type")) for row in samples)
    return {
        "created_at": now(),
        "evidence_index": args.evidence_index,
        "source_registry": args.source_registry,
        "rollout_hits": args.rollout_hits,
        "output_dir": str(output_dir),
        "sample_count": len(samples),
        "bucket_counts": dict(bucket_counts),
        "offline_role_counts": dict(role_counts),
        "source_type_counts": dict(source_type_counts),
        "artifacts": {
            "audit_samples": str(output_dir / "audit_samples.jsonl"),
            "report": str(output_dir / "采样报告.md"),
        },
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Evidence Chunk LLM/VLM 审计采样报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- evidence index：`{manifest['evidence_index']}`",
        f"- source registry：`{manifest['source_registry']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        f"- 样本数：{manifest['sample_count']}",
        "",
        "## 样本桶分布",
        "",
    ]
    for key, value in sorted(manifest["bucket_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 离线弱标签分布", ""])
    for key, value in sorted(manifest["offline_role_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## source_type 分布", ""])
    for key, value in sorted(manifest["source_type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 产物", ""])
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
