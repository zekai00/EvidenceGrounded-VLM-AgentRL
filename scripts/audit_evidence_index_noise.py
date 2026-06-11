#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from evidence_noise_rules import classify_evidence_row, evidence_text


DEFAULT_INDEX = "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 evidence index 中的目录、参考文献、OCR 等噪声。")
    parser.add_argument("--evidence-index", default=DEFAULT_INDEX)
    parser.add_argument("--rollouts", nargs="*", default=[])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-examples", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_dir = Path(args.evidence_index)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"evidence_index_noise_audit_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, by_id, index_summary, examples = audit_index(index_dir, args.max_examples)
    rollout_summary, rollout_hits = audit_rollouts(args.rollouts, by_id)

    manifest = {
        "created_at": now(),
        "evidence_index": str(index_dir),
        "output_dir": str(output_dir),
        "index_summary": index_summary,
        "rollout_summary": rollout_summary,
        "artifacts": {
            "noise_examples": str(output_dir / "noise_examples.jsonl"),
            "rollout_retrieve_hits": str(output_dir / "rollout_retrieve_hits.jsonl"),
            "report": str(output_dir / "噪声审计报告.md"),
        },
    }
    write_jsonl(output_dir / "noise_examples.jsonl", examples)
    write_jsonl(output_dir / "rollout_retrieve_hits.jsonl", rollout_hits)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "噪声审计报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def audit_index(index_dir: Path, max_examples: int) -> tuple[int, dict[str, dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    by_id: dict[str, dict[str, Any]] = {}
    type_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unusable_count = 0
    examples: list[dict[str, Any]] = []

    corpus_path = index_dir / "corpus_chunks.jsonl"
    rows = 0
    for row in iter_jsonl(corpus_path):
        rows += 1
        cls = classify_evidence_row(row)
        eid = str(row.get("evidence_id") or "")
        by_id[eid] = {"row": row, "classification": cls}
        clean_type = cls["clean_evidence_type"]
        type_counts[clean_type] += 1
        for reason in cls["noise_reasons"]:
            reason_counts[str(reason)] += 1
        source_counts[str(row.get("source_file") or "")][clean_type] += 1
        if not cls["usable_for_retrieval"]:
            unusable_count += 1
            if len(examples) < max_examples:
                examples.append(example_row(row, cls))

    noisy_sources = []
    for source_file, counts in source_counts.items():
        total = sum(counts.values())
        noisy = sum(v for k, v in counts.items() if k in noise_types())
        if noisy:
            noisy_sources.append(
                {
                    "source_file": source_file,
                    "total": total,
                    "noisy": noisy,
                    "noise_rate": round(noisy / max(1, total), 6),
                    "type_counts": dict(counts),
                }
            )
    noisy_sources.sort(key=lambda item: (item["noise_rate"], item["noisy"]), reverse=True)

    summary = {
        "corpus_chunks": rows,
        "usable_for_retrieval": rows - unusable_count,
        "unusable_for_retrieval": unusable_count,
        "unusable_rate": round(unusable_count / max(1, rows), 6),
        "type_counts": dict(type_counts),
        "noise_reason_counts": dict(reason_counts),
        "top_noisy_sources": noisy_sources[:20],
    }
    return rows, by_id, summary, examples


def audit_rollouts(paths: list[str], by_id: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not paths:
        return {"enabled": False}, []

    hit_rows: list[dict[str, Any]] = []
    rank_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    missing_ids = 0
    retrieve_calls = 0
    top1_noise = 0
    top5_noise_calls = 0

    for path in paths:
        for rollout in iter_jsonl(Path(path)):
            task_id = rollout.get("task_id")
            for call_idx, results in enumerate(iter_retrieve_results(rollout)):
                retrieve_calls += 1
                call_has_top5_noise = False
                for rank, result in enumerate(results[:5], start=1):
                    eid = str(result.get("evidence_id") or "")
                    item = by_id.get(eid)
                    if not item:
                        missing_ids += 1
                        clean_type = "missing_from_index"
                        cls = {}
                    else:
                        cls = item["classification"]
                        clean_type = cls["clean_evidence_type"]
                    type_counts[clean_type] += 1
                    if clean_type in noise_types():
                        rank_counts[f"rank{rank}"] += 1
                        call_has_top5_noise = True
                        if rank == 1:
                            top1_noise += 1
                    hit_rows.append(
                        {
                            "rollout_path": path,
                            "task_id": task_id,
                            "call_index": call_idx,
                            "rank": rank,
                            "evidence_id": eid,
                            "clean_evidence_type": clean_type,
                            "noise_score": cls.get("noise_score"),
                            "usable_for_retrieval": cls.get("usable_for_retrieval"),
                            "source_file": result.get("source_file"),
                            "page_start": result.get("page_start"),
                            "score": result.get("score"),
                            "display_snippet": result.get("display_snippet"),
                        }
                    )
                if call_has_top5_noise:
                    top5_noise_calls += 1

    summary = {
        "enabled": True,
        "rollout_paths": paths,
        "retrieve_calls": retrieve_calls,
        "top1_noise_calls": top1_noise,
        "top1_noise_rate": round(top1_noise / max(1, retrieve_calls), 6),
        "top5_noise_calls": top5_noise_calls,
        "top5_noise_rate": round(top5_noise_calls / max(1, retrieve_calls), 6),
        "missing_evidence_ids_in_top5": missing_ids,
        "top5_type_counts": dict(type_counts),
        "noise_rank_counts": dict(rank_counts),
    }
    return summary, hit_rows


def iter_retrieve_results(rollout: dict[str, Any]) -> Iterable[list[dict[str, Any]]]:
    for step in rollout.get("steps") or []:
        result = step.get("result")
        if isinstance(result, dict) and result.get("tool") == "retrieve_evidence":
            results = result.get("results") or []
            if isinstance(results, list):
                yield results


def example_row(row: dict[str, Any], cls: dict[str, Any]) -> dict[str, Any]:
    text = evidence_text(row).replace("\n", " ")
    return {
        "evidence_id": row.get("evidence_id"),
        "source_file": row.get("source_file"),
        "page_start": row.get("page_start") if row.get("page_start") is not None else row.get("page"),
        "page_end": row.get("page_end"),
        "clean_evidence_type": cls["clean_evidence_type"],
        "noise_score": cls["noise_score"],
        "noise_reasons": cls["noise_reasons"],
        "snippet": text[:500],
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    index_summary = manifest["index_summary"]
    rollout_summary = manifest["rollout_summary"]
    lines = [
        "# Evidence Index 噪声审计报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- evidence index：`{manifest['evidence_index']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        "",
        "## 索引总体统计",
        "",
        f"- corpus_chunks：{index_summary['corpus_chunks']}",
        f"- 可用于检索：{index_summary['usable_for_retrieval']}",
        f"- 不应进入检索：{index_summary['unusable_for_retrieval']}",
        f"- 噪声比例：{index_summary['unusable_rate']}",
        "",
        "## 类型分布",
        "",
    ]
    for key, value in sorted(index_summary["type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 噪声原因分布", ""])
    for key, value in sorted(index_summary["noise_reason_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")

    if rollout_summary.get("enabled"):
        lines.extend(
            [
                "",
                "## Rollout 检索污染",
                "",
                f"- retrieve 调用数：{rollout_summary['retrieve_calls']}",
                f"- top1 噪声调用数：{rollout_summary['top1_noise_calls']}",
                f"- top1 噪声率：{rollout_summary['top1_noise_rate']}",
                f"- top5 含噪声调用数：{rollout_summary['top5_noise_calls']}",
                f"- top5 含噪声率：{rollout_summary['top5_noise_rate']}",
                f"- top5 中索引缺失 evidence_id：{rollout_summary['missing_evidence_ids_in_top5']}",
                "",
                "### top5 类型分布",
                "",
            ]
        )
        for key, value in sorted(rollout_summary["top5_type_counts"].items(), key=lambda item: (-item[1], item[0])):
            lines.append(f"- {key}: {value}")

    lines.extend(["", "## 噪声最多的文档", ""])
    for item in index_summary["top_noisy_sources"][:10]:
        lines.append(f"- `{item['source_file']}`: noisy={item['noisy']}/{item['total']}，rate={item['noise_rate']}")
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


def noise_types() -> set[str]:
    return {"toc", "bibliography", "front_matter", "back_matter", "header_footer", "ocr_noise"}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    raise SystemExit(main())
