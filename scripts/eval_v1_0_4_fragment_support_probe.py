#!/usr/bin/env python
"""Replay rollouts with v1.0.4 field-fragment support labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_TASKS = (
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "gold_eval_v1_0_4_caption_corrected_20260611_1830/val_gold_50.jsonl"
)
DEFAULT_ROLLOUTS = [
    "baseline_val50=outputs/v1_0_4_gold_eval_caption_corrected_val50_bf16_20260611_2055/rollouts.jsonl",
    "behavior_repair_C_val50=outputs/v1_0_4_behavior_repair_C_pos70_replay90_20step_val50_bf16_20260612_0240/rollouts.jsonl",
    "field_policy_probe_val16=outputs/v1_0_4_field_policy_prompt_reward_probe_val16_bf16_20260612_1146/rollouts.jsonl",
]
CORE_FIELDS = ["caption_text", "image_scope", "depicted_work_title", "displayed_region", "object_type"]
LOCAL_CAPTION_RISK_FIELDS = {"image_scope", "displayed_region", "object_type"}
POSITIVE_LABELS = {"support"}
WEAK_LABELS = {"weak_support"}
NEGATIVE_LABELS = {"no_support", "wrong_target", "contradict"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--rollout", action="append", default=[])
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or f"outputs/v1_0_4_fragment_support_replay_{datetime.now().strftime('%Y%m%d_%H%M')}")
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = {str(row.get("task_id")): row for row in read_jsonl(args.tasks)}
    label_rows = read_jsonl(args.labels)
    label_index = build_label_index(label_rows)
    rollout_specs = parse_rollout_specs(args.rollout or DEFAULT_ROLLOUTS)

    run_results = {}
    for name, path in rollout_specs.items():
        if not Path(path).exists():
            continue
        records = read_jsonl(path)
        run_results[name] = evaluate_run(name, records, tasks, label_index)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tasks": args.tasks,
        "labels": args.labels,
        "label_count": len(label_rows),
        "label_coverage": label_index["coverage_summary"],
        "runs": run_results,
    }
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "fragment_support_replay_report.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def build_label_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task_field_eid: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    by_task_field_positive: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        key = (str(row.get("task_id")), str(row.get("field")), str(row.get("evidence_id")))
        by_task_field_eid[key].append(row)
        if row.get("final_label") in POSITIVE_LABELS:
            by_task_field_positive[(key[0], key[1])].add(key[2])
    return {
        "by_task_field_eid": by_task_field_eid,
        "by_task_field_positive": by_task_field_positive,
        "coverage_summary": {
            "task_count": len({str(row.get("task_id")) for row in rows}),
            "field_counts": dict(Counter(str(row.get("field")) for row in rows)),
            "label_counts": dict(Counter(str(row.get("final_label")) for row in rows)),
        },
    }


def evaluate_run(
    name: str,
    records: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    label_index: dict[str, Any],
) -> dict[str, Any]:
    field_counts = {
        field: Counter(tp=0, fp=0, fn=0, predicted=0, gold_supported=0, supported=0) for field in CORE_FIELDS
    }
    evidence_counts = {
        field: Counter(cited=0, cited_labeled=0, cited_positive=0, positive_known=0, wrong_target=0, weak=0, negative=0)
        for field in CORE_FIELDS
    }
    task_counters = Counter()
    cited_label_counts = Counter()
    per_task_rows = []

    for record in records:
        task_id = str(record.get("task_id"))
        task = tasks.get(task_id)
        if not task:
            continue
        gold_by_field = {
            str(claim.get("field")): claim for claim in (task.get("gold") or {}).get("claims", []) if claim.get("field")
        }
        pred_by_field = {str(claim.get("field")): claim for claim in record.get("final_claims") or [] if claim.get("field")}
        retrieved_ids, opened_ids = trajectory_evidence_sets(record)
        task_has_external_gain = False
        task_has_external_harm = False
        task_has_wrong_target = False
        task_local_overgeneralization = False

        for field in CORE_FIELDS:
            gold = gold_by_field.get(field, {})
            pred = pred_by_field.get(field)
            positive_known = label_index["by_task_field_positive"].get((task_id, field), set())
            evidence_counts[field]["positive_known"] += len(positive_known)
            if gold and not gold.get("abstain"):
                field_counts[field]["gold_supported"] += 1
            if pred and not pred.get("abstain"):
                field_counts[field]["predicted"] += 1
                cited_ids = [str(eid) for eid in pred.get("evidence_ids") or []]
                citation_labels = [aggregate_label(label_index, task_id, field, eid) for eid in cited_ids]
                is_value_match = value_match(gold.get("value"), pred.get("value")) if gold else False
                is_supported = is_value_match and any(label["label"] in POSITIVE_LABELS for label in citation_labels)
                if is_supported:
                    field_counts[field]["tp"] += 1
                    field_counts[field]["supported"] += 1
                else:
                    field_counts[field]["fp"] += 1
                for eid, label in zip(cited_ids, citation_labels):
                    evidence_counts[field]["cited"] += 1
                    cited_label_counts[label["label"]] += 1
                    if label["known"]:
                        evidence_counts[field]["cited_labeled"] += 1
                    if label["label"] in POSITIVE_LABELS:
                        evidence_counts[field]["cited_positive"] += 1
                        if not eid.startswith("local_caption_"):
                            task_has_external_gain = True
                    elif label["label"] in WEAK_LABELS:
                        evidence_counts[field]["weak"] += 1
                    elif label["label"] in NEGATIVE_LABELS:
                        evidence_counts[field]["negative"] += 1
                        if not eid.startswith("local_caption_"):
                            task_has_external_harm = True
                    if label["label"] == "wrong_target":
                        evidence_counts[field]["wrong_target"] += 1
                        task_has_wrong_target = True
                    if (
                        eid.startswith("local_caption_")
                        and field in LOCAL_CAPTION_RISK_FIELDS
                        and label["label"] not in POSITIVE_LABELS
                    ):
                        task_local_overgeneralization = True
            if gold and not gold.get("abstain"):
                pred_supported = pred and not pred.get("abstain") and value_match(gold.get("value"), pred.get("value"))
                pred_labels = [
                    aggregate_label(label_index, task_id, field, str(eid))
                    for eid in ((pred or {}).get("evidence_ids") or [])
                ]
                if not (pred_supported and any(item["label"] in POSITIVE_LABELS for item in pred_labels)):
                    field_counts[field]["fn"] += 1

        if retrieved_ids:
            task_counters["retrieved_task"] += 1
        external_opened = opened_ids - local_caption_ids(opened_ids)
        if external_opened:
            task_counters["external_open_task"] += 1
            if not task_has_external_gain:
                task_counters["external_open_no_positive_citation_task"] += 1
        if retrieved_ids and not task_has_external_gain:
            task_counters["retrieve_no_external_positive_citation_task"] += 1
        if task_has_external_gain:
            task_counters["tool_gain_task"] += 1
        if task_has_external_harm:
            task_counters["tool_harm_task"] += 1
        if task_has_wrong_target:
            task_counters["wrong_target_task"] += 1
        if task_local_overgeneralization:
            task_counters["local_caption_overgeneralization_task"] += 1
        per_task_rows.append(
            {
                "task_id": task_id,
                "tool_gain": task_has_external_gain,
                "tool_harm": task_has_external_harm,
                "wrong_target": task_has_wrong_target,
                "local_caption_overgeneralization": task_local_overgeneralization,
            }
        )

    n = max(1, len(per_task_rows))
    field_metrics = {}
    for field in CORE_FIELDS:
        c = field_counts[field]
        e = evidence_counts[field]
        claim_p = c["tp"] / max(1, c["tp"] + c["fp"])
        claim_r = c["tp"] / max(1, c["tp"] + c["fn"])
        cite_p = e["cited_positive"] / max(1, e["cited"])
        cite_r = e["cited_positive"] / max(1, e["positive_known"])
        field_metrics[field] = {
            "claim_support_precision": round(claim_p, 6),
            "claim_support_recall": round(claim_r, 6),
            "claim_support_f1": round(f1(claim_p, claim_r), 6),
            "cited_evidence_precision": round(cite_p, 6),
            "cited_evidence_recall": round(cite_r, 6),
            "cited_evidence_f1": round(f1(cite_p, cite_r), 6),
            "wrong_target_citation_rate": round(e["wrong_target"] / max(1, e["cited"]), 6),
            "weak_citation_rate": round(e["weak"] / max(1, e["cited"]), 6),
            "negative_citation_rate": round(e["negative"] / max(1, e["cited"]), 6),
            "cited_label_coverage": round(e["cited_labeled"] / max(1, e["cited"]), 6),
            "counts": dict(c),
            "evidence_counts": dict(e),
        }
    micro = micro_metrics(field_counts, evidence_counts)
    micro.update(
        {
            "tool_gain_task_rate": round(task_counters["tool_gain_task"] / n, 6),
            "tool_harm_task_rate": round(task_counters["tool_harm_task"] / n, 6),
            "wrong_target_task_rate": round(task_counters["wrong_target_task"] / n, 6),
            "local_caption_overgeneralization_task_rate": round(
                task_counters["local_caption_overgeneralization_task"] / n, 6
            ),
            "retrieved_task_rate": round(task_counters["retrieved_task"] / n, 6),
            "external_open_task_rate": round(task_counters["external_open_task"] / n, 6),
            "external_open_no_positive_citation_task_rate": round(
                task_counters["external_open_no_positive_citation_task"] / max(1, task_counters["external_open_task"]),
                6,
            ),
            "retrieve_no_external_positive_citation_task_rate": round(
                task_counters["retrieve_no_external_positive_citation_task"] / max(1, task_counters["retrieved_task"]),
                6,
            ),
        }
    )
    return {
        "run_name": name,
        "task_count": len(per_task_rows),
        "micro": micro,
        "per_field": field_metrics,
        "cited_label_counts": dict(cited_label_counts),
        "task_counters": dict(task_counters),
    }


def aggregate_label(label_index: dict[str, Any], task_id: str, field: str, evidence_id: str) -> dict[str, Any]:
    rows = label_index["by_task_field_eid"].get((task_id, field, evidence_id), [])
    if not rows:
        return {"label": "unknown", "known": False, "support_score": 0.0}
    labels = [str(row.get("final_label")) for row in rows]
    if "support" in labels:
        label = "support"
    elif "wrong_target" in labels:
        label = "wrong_target"
    elif "contradict" in labels:
        label = "contradict"
    elif "weak_support" in labels:
        label = "weak_support"
    else:
        label = "no_support"
    score = max(float(row.get("support_score") or 0.0) for row in rows)
    return {"label": label, "known": True, "support_score": score}


def micro_metrics(field_counts: dict[str, Counter], evidence_counts: dict[str, Counter]) -> dict[str, Any]:
    tp = sum(c["tp"] for c in field_counts.values())
    fp = sum(c["fp"] for c in field_counts.values())
    fn = sum(c["fn"] for c in field_counts.values())
    claim_p = tp / max(1, tp + fp)
    claim_r = tp / max(1, tp + fn)
    cited = sum(e["cited"] for e in evidence_counts.values())
    cited_positive = sum(e["cited_positive"] for e in evidence_counts.values())
    positive_known = sum(e["positive_known"] for e in evidence_counts.values())
    wrong = sum(e["wrong_target"] for e in evidence_counts.values())
    weak = sum(e["weak"] for e in evidence_counts.values())
    negative = sum(e["negative"] for e in evidence_counts.values())
    labeled = sum(e["cited_labeled"] for e in evidence_counts.values())
    cite_p = cited_positive / max(1, cited)
    cite_r = cited_positive / max(1, positive_known)
    return {
        "claim_support_precision": round(claim_p, 6),
        "claim_support_recall": round(claim_r, 6),
        "claim_support_f1": round(f1(claim_p, claim_r), 6),
        "cited_evidence_precision": round(cite_p, 6),
        "cited_evidence_recall": round(cite_r, 6),
        "cited_evidence_f1": round(f1(cite_p, cite_r), 6),
        "wrong_target_citation_rate": round(wrong / max(1, cited), 6),
        "weak_citation_rate": round(weak / max(1, cited), 6),
        "negative_citation_rate": round(negative / max(1, cited), 6),
        "cited_label_coverage": round(labeled / max(1, cited), 6),
        "counts": {"tp": tp, "fp": fp, "fn": fn, "cited": cited, "cited_positive": cited_positive},
    }


def trajectory_evidence_sets(record: dict[str, Any]) -> tuple[set[str], set[str]]:
    retrieved, opened = set(), set()
    for step in record.get("steps") or []:
        action = step.get("parsed_action") or {}
        result = step.get("result") or {}
        if result.get("tool") == "retrieve_evidence":
            for item in result.get("results") or []:
                if item.get("evidence_id"):
                    retrieved.add(str(item.get("evidence_id")))
        if action.get("action") == "open_evidence" or result.get("tool") == "open_evidence":
            eid = action.get("evidence_id") or result.get("evidence_id")
            if eid and not result.get("error"):
                opened.add(str(eid))
    return retrieved, opened


def local_caption_ids(ids: set[str]) -> set[str]:
    return {eid for eid in ids if eid.startswith("local_caption_")}


def value_match(gold_value: Any, pred_value: Any) -> bool:
    g = normalize(gold_value)
    p = normalize(pred_value)
    if not g or not p:
        return False
    return g == p or g in p or p in g


def normalize(value: Any) -> str:
    import re

    return re.sub(r"[《》\s,，。.;；:：'\"“”‘’()（）\[\]【】/_\\|-]+", "", str(value or "").lower())


def f1(p: float, r: float) -> float:
    if p <= 0.0 or r <= 0.0:
        return 0.0
    return 2 * p * r / max(1e-12, p + r)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# v1.0.4 Fragment Support Probe 离线回放报告", ""]
    lines.extend(
        [
            "## 数据",
            "",
            f"- tasks: `{summary['tasks']}`",
            f"- labels: `{summary['labels']}`",
            f"- label_count: {summary['label_count']}",
            "",
            "## Runs",
            "",
        ]
    )
    for name, result in summary["runs"].items():
        micro = result["micro"]
        lines.extend(
            [
                f"### {name}",
                "",
                f"- task_count: {result['task_count']}",
                f"- claim_support_p/r/f1: {micro['claim_support_precision']:.3f}/{micro['claim_support_recall']:.3f}/{micro['claim_support_f1']:.3f}",
                f"- cited_evidence_p/r/f1: {micro['cited_evidence_precision']:.3f}/{micro['cited_evidence_recall']:.3f}/{micro['cited_evidence_f1']:.3f}",
                f"- wrong_target_citation_rate: {micro['wrong_target_citation_rate']:.3f}",
                f"- local_caption_overgeneralization_task_rate: {micro['local_caption_overgeneralization_task_rate']:.3f}",
                f"- tool_gain_task_rate: {micro['tool_gain_task_rate']:.3f}",
                f"- tool_harm_task_rate: {micro['tool_harm_task_rate']:.3f}",
                f"- external_open_no_positive_citation_task_rate: {micro['external_open_no_positive_citation_task_rate']:.3f}",
                f"- retrieve_no_external_positive_citation_task_rate: {micro['retrieve_no_external_positive_citation_task_rate']:.3f}",
                f"- cited_label_coverage: {micro['cited_label_coverage']:.3f}",
                "",
                "| Field | claim_f1 | cited_f1 | wrong_target | local/neg notes |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for field, metrics in result["per_field"].items():
            lines.append(
                f"| {field} | {metrics['claim_support_f1']:.3f} | {metrics['cited_evidence_f1']:.3f} | "
                f"{metrics['wrong_target_citation_rate']:.3f} | neg={metrics['negative_citation_rate']:.3f}, weak={metrics['weak_citation_rate']:.3f} |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_rollout_specs(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            out[Path(item).parent.name or Path(item).stem] = item
        else:
            name, path = item.split("=", 1)
            out[name.strip()] = path.strip()
    return out


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
