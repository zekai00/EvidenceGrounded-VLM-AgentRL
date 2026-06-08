#!/usr/bin/env python3
"""Audit region candidate oracle quality without loading a policy model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.data import read_jsonl  # noqa: E402
from evidence_agent_env.tools.region_proposal import propose_regions  # noqa: E402


def bbox_iou(a: Any, b: Any) -> float:
    try:
        ax1, ay1, ax2, ay2 = [float(x) for x in a]
        bx1, by1, bx2, by2 = [float(x) for x in b]
    except Exception:
        return 0.0
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument("--include-gold-regions", action="store_true")
    parser.add_argument("--bad-example-limit", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.tasks)
    if args.max_tasks > 0:
        rows = rows[: args.max_tasks]
    records = [audit_one(task, args.top_k, args.include_gold_regions) for task in rows]
    summary = summarize(records, args)
    (output_dir / "region_candidate_audit.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(output_dir / "region_candidate_audit.md", summary, records, args.bad_example_limit)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def audit_one(task: dict[str, Any], top_k: int, include_gold: bool) -> dict[str, Any]:
    gold_bbox = (task.get("gold") or {}).get("image_bbox")
    regions = propose_regions(task, top_k=top_k, include_gold=include_gold)
    scored: list[dict[str, Any]] = []
    for rank, region in enumerate(regions, start=1):
        iou = bbox_iou(region.get("bbox"), gold_bbox) if gold_bbox else 0.0
        scored.append(
            {
                "rank": rank,
                "region_id": region.get("region_id"),
                "bbox": region.get("bbox"),
                "iou": round(iou, 6),
                "source": region.get("source"),
                "type": region.get("type"),
                "score": region.get("score"),
                "has_caption_evidence_id": bool(region.get("caption_evidence_id")),
                "caption_evidence_id": region.get("caption_evidence_id"),
                "caption_hint": region.get("caption_hint") or region.get("nearby_text") or region.get("hint"),
            }
        )
    best = max(scored, key=lambda item: float(item["iou"]), default=None)
    top1 = scored[0] if scored else None
    return {
        "task_id": task.get("task_id"),
        "split": task.get("split"),
        "source_file": task.get("source_file"),
        "page": task.get("page"),
        "page_image": task.get("page_image"),
        "gold_bbox": gold_bbox,
        "candidate_count": len(scored),
        "oracle_iou": best.get("iou") if best else 0.0,
        "oracle_rank": best.get("rank") if best else None,
        "oracle_region_id": best.get("region_id") if best else None,
        "oracle_source": best.get("source") if best else None,
        "oracle_type": best.get("type") if best else None,
        "top1_iou": top1.get("iou") if top1 else 0.0,
        "top1_region_id": top1.get("region_id") if top1 else None,
        "top1_source": top1.get("source") if top1 else None,
        "top1_type": top1.get("type") if top1 else None,
        "caption_candidate_count": sum(int(item["has_caption_evidence_id"]) for item in scored),
        "regions": scored,
    }


def summarize(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    n = max(1, len(records))
    oracle_ious = [float(item.get("oracle_iou") or 0.0) for item in records]
    top1_ious = [float(item.get("top1_iou") or 0.0) for item in records]
    source_counts = Counter(str(item.get("oracle_source") or "unknown") for item in records)
    type_counts = Counter(str(item.get("oracle_type") or "unknown") for item in records)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tasks": args.tasks,
        "top_k": args.top_k,
        "tasks_used": len(records),
        "include_gold_regions": args.include_gold_regions,
        "oracle_hit_rate_iou50": sum(iou >= 0.5 for iou in oracle_ious) / n,
        "oracle_hit_rate_iou70": sum(iou >= 0.7 for iou in oracle_ious) / n,
        "oracle_mean_iou": sum(oracle_ious) / n,
        "top1_hit_rate_iou50": sum(iou >= 0.5 for iou in top1_ious) / n,
        "top1_mean_iou": sum(top1_ious) / n,
        "mean_candidate_count": sum(int(item.get("candidate_count") or 0) for item in records) / n,
        "mean_caption_candidate_count": sum(int(item.get("caption_candidate_count") or 0) for item in records) / n,
        "oracle_source_counts": dict(source_counts),
        "oracle_type_counts": dict(type_counts),
        "bad_iou50_count": sum(float(item.get("oracle_iou") or 0.0) < 0.5 for item in records),
    }


def write_markdown(path: Path, summary: dict[str, Any], records: list[dict[str, Any]], bad_limit: int) -> None:
    lines = [
        "# Region Candidate Audit",
        "",
        f"- created_at: {summary['created_at']}",
        f"- tasks: `{summary['tasks']}`",
        f"- tasks_used: {summary['tasks_used']}",
        f"- top_k: {summary['top_k']}",
        f"- include_gold_regions: {summary['include_gold_regions']}",
        "",
        "## Summary",
        "",
        f"- oracle_hit_rate_iou50: {summary['oracle_hit_rate_iou50']:.3f}",
        f"- oracle_hit_rate_iou70: {summary['oracle_hit_rate_iou70']:.3f}",
        f"- oracle_mean_iou: {summary['oracle_mean_iou']:.3f}",
        f"- top1_hit_rate_iou50: {summary['top1_hit_rate_iou50']:.3f}",
        f"- top1_mean_iou: {summary['top1_mean_iou']:.3f}",
        f"- mean_candidate_count: {summary['mean_candidate_count']:.2f}",
        f"- mean_caption_candidate_count: {summary['mean_caption_candidate_count']:.2f}",
        f"- bad_iou50_count: {summary['bad_iou50_count']}",
        f"- oracle_source_counts: `{json.dumps(summary['oracle_source_counts'], ensure_ascii=False)}`",
        f"- oracle_type_counts: `{json.dumps(summary['oracle_type_counts'], ensure_ascii=False)}`",
        "",
        "## Bad Examples",
        "",
    ]
    bad = sorted(records, key=lambda item: float(item.get("oracle_iou") or 0.0))[:bad_limit]
    for item in bad:
        lines.extend(
            [
                f"### {item.get('task_id')}",
                "",
                f"- source: {item.get('source_file')} p.{item.get('page')}",
                f"- page_image: `{item.get('page_image')}`",
                f"- gold_bbox: `{item.get('gold_bbox')}`",
                f"- oracle_iou: {item.get('oracle_iou')}; oracle_rank: {item.get('oracle_rank')}; oracle_region_id: {item.get('oracle_region_id')}; oracle_source: {item.get('oracle_source')}; oracle_type: {item.get('oracle_type')}",
                f"- top1_iou: {item.get('top1_iou')}; top1_region_id: {item.get('top1_region_id')}; top1_source: {item.get('top1_source')}; top1_type: {item.get('top1_type')}",
                "",
            ]
        )
        for region in (item.get("regions") or [])[:5]:
            lines.append(
                f"  - rank {region.get('rank')}: {region.get('region_id')} iou={region.get('iou')} type={region.get('type')} source={region.get('source')} bbox={region.get('bbox')}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
