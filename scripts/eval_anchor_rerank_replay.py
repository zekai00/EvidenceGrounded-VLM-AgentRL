#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from evidence_noise_rules import classify_evidence_row
from src.evidence_agent_env.data import EvidenceIndex, build_anchor_profile, compact_for_match, item_page_distance
from src.evidence_agent_env.verifier import gold_evidence_ids


DEFAULT_OLD_INDEX = "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
DEFAULT_CLEAN_INDEX = "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v1_0_4_clean_20260610_2046"
DEFAULT_TASKS = "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615/val_tasks.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对旧 rollout 中的 retrieve query 做 anchor rerank replay。")
    parser.add_argument("--old-index", default=DEFAULT_OLD_INDEX)
    parser.add_argument("--clean-index", default=DEFAULT_CLEAN_INDEX)
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"anchor_rerank_replay_v1_0_4_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(Path(args.tasks))
    runs = {
        "old_index_no_anchor": EvidenceIndex(args.old_index, anchor_rerank=False),
        "clean_index_no_anchor": EvidenceIndex(args.clean_index, anchor_rerank=False),
        "clean_index_anchor": EvidenceIndex(args.clean_index, anchor_rerank=True),
    }
    calls = list(iter_retrieve_calls(args.rollouts, tasks))

    summaries: dict[str, Any] = {}
    all_hits: list[dict[str, Any]] = []
    for name, index in runs.items():
        summary, hits = eval_run(name, index, calls, args.top_k)
        summaries[name] = summary
        all_hits.extend(hits)

    manifest = {
        "created_at": now(),
        "old_index": args.old_index,
        "clean_index": args.clean_index,
        "tasks": args.tasks,
        "rollouts": args.rollouts,
        "retrieve_calls": len(calls),
        "top_k": args.top_k,
        "summaries": summaries,
        "artifacts": {
            "hits": str(output_dir / "replay_hits.jsonl"),
            "report": str(output_dir / "评估报告.md"),
        },
    }
    write_jsonl(output_dir / "replay_hits.jsonl", all_hits)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "评估报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def eval_run(
    run_name: str,
    index: EvidenceIndex,
    calls: list[dict[str, Any]],
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    hit_rows: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    noise_types = {"toc", "bibliography", "front_matter", "back_matter", "header_footer", "ocr_noise"}
    retrieve_calls = 0
    top1_noise = 0
    top5_noise = 0
    top1_gold = 0
    top5_gold = 0
    top1_current_or_nearby = 0
    top5_current_or_nearby = 0
    top5_figure_label = 0
    top5_caption_term = 0

    for call in calls:
        retrieve_calls += 1
        task = call["task"]
        profile = build_anchor_profile(task)
        gold_ids = gold_evidence_ids(task)
        results = index.search(call["query"], call["scope"], task, top_k)
        call_top5_noise = False
        call_top5_gold = False
        call_top5_current_or_nearby = False
        call_top5_figure_label = False
        call_top5_caption_term = False

        for rank, result in enumerate(results, start=1):
            item = index.open(str(result.get("evidence_id"))) or {}
            clean_type = item.get("clean_evidence_type") or classify_evidence_row(item)["clean_evidence_type"]
            type_counts[str(clean_type)] += 1
            is_noise = clean_type in noise_types
            is_gold = str(result.get("evidence_id")) in gold_ids
            is_current_or_nearby = current_or_nearby(item, task)
            has_figure_label = contains_any(item, profile["figure_labels"])
            has_caption_term = contains_any(item, profile["caption_terms"])

            if rank == 1:
                top1_noise += int(is_noise)
                top1_gold += int(is_gold)
                top1_current_or_nearby += int(is_current_or_nearby)
            call_top5_noise = call_top5_noise or is_noise
            call_top5_gold = call_top5_gold or is_gold
            call_top5_current_or_nearby = call_top5_current_or_nearby or is_current_or_nearby
            call_top5_figure_label = call_top5_figure_label or has_figure_label
            call_top5_caption_term = call_top5_caption_term or has_caption_term
            hit_rows.append(
                {
                    "run": run_name,
                    "task_id": task.get("task_id"),
                    "query": call["query"],
                    "scope": call["scope"],
                    "rank": rank,
                    "evidence_id": result.get("evidence_id"),
                    "score": result.get("score"),
                    "clean_evidence_type": clean_type,
                    "is_noise": is_noise,
                    "is_gold": is_gold,
                    "is_current_or_nearby": is_current_or_nearby,
                    "has_figure_label": has_figure_label,
                    "has_caption_term": has_caption_term,
                    "source_file": result.get("source_file"),
                    "page_start": result.get("page_start"),
                    "display_snippet": result.get("display_snippet"),
                }
            )

        top5_noise += int(call_top5_noise)
        top5_gold += int(call_top5_gold)
        top5_current_or_nearby += int(call_top5_current_or_nearby)
        top5_figure_label += int(call_top5_figure_label)
        top5_caption_term += int(call_top5_caption_term)

    denom = max(1, retrieve_calls)
    summary = {
        "retrieve_calls": retrieve_calls,
        "top1_noise_rate": round(top1_noise / denom, 6),
        "top5_noise_rate": round(top5_noise / denom, 6),
        "top1_gold_hit_rate": round(top1_gold / denom, 6),
        "top5_gold_hit_rate": round(top5_gold / denom, 6),
        "top1_current_or_nearby_rate": round(top1_current_or_nearby / denom, 6),
        "top5_current_or_nearby_rate": round(top5_current_or_nearby / denom, 6),
        "top5_figure_label_hit_rate": round(top5_figure_label / denom, 6),
        "top5_caption_term_hit_rate": round(top5_caption_term / denom, 6),
        "topk_type_counts": dict(type_counts),
    }
    return summary, hit_rows


def iter_retrieve_calls(paths: list[str], tasks: dict[str, dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for path in paths:
        for rollout in iter_jsonl(Path(path)):
            task = tasks.get(str(rollout.get("task_id"))) or {
                "task_id": rollout.get("task_id"),
                "source_file": rollout.get("source_file"),
                "page": rollout.get("page"),
            }
            for step in rollout.get("steps") or []:
                action = step.get("parsed_action") or step.get("model_parsed_action") or {}
                if action.get("action") != "retrieve_evidence":
                    continue
                yield {
                    "task": task,
                    "query": str(action.get("query") or ""),
                    "scope": str(action.get("scope") or "same_document"),
                }


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        tasks[str(row.get("task_id"))] = row
    return tasks


def current_or_nearby(item: dict[str, Any], task: dict[str, Any]) -> bool:
    distance = item_page_distance(item, task.get("page"))
    return distance is not None and distance <= 1


def contains_any(item: dict[str, Any], terms: list[str]) -> bool:
    text = compact_for_match(" ".join(str(item.get(key, "")) for key in ("display_snippet", "evidence_summary", "clean_text", "text")))
    for term in terms:
        normalized = compact_for_match(term)
        if len(normalized) >= 4 and normalized in text:
            return True
    return False


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Anchor Rerank Replay 评估报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- old index：`{manifest['old_index']}`",
        f"- clean index：`{manifest['clean_index']}`",
        f"- tasks：`{manifest['tasks']}`",
        f"- retrieve calls：{manifest['retrieve_calls']}",
        "",
        "## 指标对比",
        "",
        "| run | top1_noise | top5_noise | top1_gold | top5_gold | top1_nearby | top5_nearby | figure_label | caption_term |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, summary in manifest["summaries"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    f"{summary['top1_noise_rate']:.6f}",
                    f"{summary['top5_noise_rate']:.6f}",
                    f"{summary['top1_gold_hit_rate']:.6f}",
                    f"{summary['top5_gold_hit_rate']:.6f}",
                    f"{summary['top1_current_or_nearby_rate']:.6f}",
                    f"{summary['top5_current_or_nearby_rate']:.6f}",
                    f"{summary['top5_figure_label_hit_rate']:.6f}",
                    f"{summary['top5_caption_term_hit_rate']:.6f}",
                ]
            )
            + " |"
        )
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
