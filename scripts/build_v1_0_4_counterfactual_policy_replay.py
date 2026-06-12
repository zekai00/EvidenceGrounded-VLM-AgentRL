#!/usr/bin/env python3
"""Build small v1.0.4 counterfactual replay rows for field/evidence policy.

The rows target one failure mode only: local caption evidence is useful for
caption/title, but should not be generalized to fields such as displayed_region
or object_type unless the text explicitly supports that field.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_ROOT = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v1_0_3_no_select_sft_20260608_0615"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
RISK_FIELDS = {"image_scope", "displayed_region", "object_type"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-train", type=int, default=120)
    parser.add_argument("--max-val", type=int, default=40)
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    source_root = Path(args.source_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / default_output_name()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    (output_dir / "sft").mkdir(parents=True)

    tasks = load_tasks_by_split(source_root)
    sft_rows = load_sft_rows_by_split(source_root)
    rows_by_split = {
        split: build_split_rows(tasks.get(split, []), sft_rows.get(split, []), rng)
        for split in ["train", "val"]
    }
    train_rows = balanced_sample(rows_by_split.get("train", []), args.max_train, rng)
    val_rows = balanced_sample(rows_by_split.get("val", []), args.max_val, rng)

    write_jsonl(output_dir / "sft" / "train.jsonl", train_rows)
    write_jsonl(output_dir / "sft" / "val.jsonl", val_rows)
    write_jsonl(output_dir / "counterfactual_rows.jsonl", train_rows + val_rows)

    manifest = {
        "created_at": now(),
        "dataset_version": "v1.0.4_counterfactual_field_policy_replay",
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "policy": {
            "no_test_split_used": True,
            "goal": [
                "do_not_use_local_caption_as_generic_support_for_non_caption_fields",
                "after_retrieve_open_external_evidence_before_using_it_for_claims",
                "abstain_risk_fields_when_only_local_caption_is_available",
            ],
        },
        "candidate_counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "counts": {
            "train_rows": len(train_rows),
            "val_rows": len(val_rows),
        },
        "action_counts": {
            "train": dict(action_counter(train_rows)),
            "val": dict(action_counter(val_rows)),
        },
        "patch_type_counts": {
            "train": dict(patch_type_counter(train_rows)),
            "val": dict(patch_type_counter(val_rows)),
        },
        "field_counts": {
            "train": dict(field_counter(train_rows)),
            "val": dict(field_counter(val_rows)),
        },
        "artifacts": {
            "train": str(output_dir / "sft" / "train.jsonl"),
            "val": str(output_dir / "sft" / "val.jsonl"),
            "rows": str(output_dir / "counterfactual_rows.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_split_rows(
    tasks: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    tasks_by_id = {str(task.get("task_id")): task for task in tasks}
    rows: list[dict[str, Any]] = []
    for row in sft_rows:
        task = tasks_by_id.get(str(row.get("task_id")))
        if not task or not has_local_caption_caption_support(task):
            continue
        action = row.get("action") or {}
        name = action.get("action")
        if name == "write_claims_chunk":
            rows.extend(build_abstain_rows(row, task))
        if name == "write_claims_chunk" and retrieve_result(row):
            open_row = build_open_external_row(row, task)
            if open_row:
                rows.append(open_row)
    rng.shuffle(rows)
    return rows


def build_abstain_rows(row: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    action = row.get("action") or {}
    for claim in action.get("claims") or []:
        field = str(claim.get("field") or "")
        if field not in RISK_FIELDS:
            continue
        evidence_ids = [str(eid) for eid in claim.get("evidence_ids") or []]
        if evidence_ids and not all(eid.startswith("local_caption_") for eid in evidence_ids):
            continue
        out.append(make_abstain_row(row, task, field, "local_caption_only_risk_claim_replaced_by_abstain"))
    for abstain in action.get("abstains") or []:
        field = str(abstain.get("field") or "")
        if field in {"image_scope", "displayed_region"}:
            out.append(make_abstain_row(row, task, field, "local_caption_only_risk_abstain_replay"))
    return out


def make_abstain_row(row: dict[str, Any], task: dict[str, Any], field: str, patch_type: str) -> dict[str, Any]:
    cloned = patch_clone(row, patch_type=patch_type)
    cloned["action"] = {
        "action": "write_claims_chunk",
        "claims": [],
        "abstains": [{"field": field, "reason": local_caption_insufficient_reason(field)}],
    }
    cloned["available_actions"] = ["write_claims_chunk"]
    cloned["phase_name"] = "v1_0_4_counterfactual_field_policy_abstain"
    cloned["phase_hint"] = (
        "当前只有 local caption 或尚未打开能支持该字段的外部 evidence。"
        "local caption 不能泛化支持非 caption 字段；该字段必须 abstain，或先打开能支持该字段的外部 evidence。"
    )
    cloned["field_policy_hint"] = field_policy_hint(task)
    cloned["patch_meta"].update({"field": field, "local_caption_ids": local_caption_ids(task)})
    return cloned


def build_open_external_row(row: dict[str, Any], task: dict[str, Any]) -> dict[str, Any] | None:
    result = retrieve_result(row)
    if not result:
        return None
    local_ids = set(local_caption_ids(task))
    for item in result.get("results") or []:
        evidence_id = str(item.get("evidence_id") or "")
        if evidence_id and evidence_id not in local_ids:
            cloned = patch_clone(row, patch_type="retrieve_then_open_external_for_noncaption")
            cloned["action"] = {"action": "open_evidence", "evidence_id": evidence_id}
            cloned["available_actions"] = ["open_evidence"]
            cloned["phase_name"] = "v1_0_4_counterfactual_field_policy_open_external"
            cloned["phase_hint"] = (
                "retrieve_evidence 已返回候选外部 evidence。"
                "如果要用检索结果支持 image_scope/displayed_region/object_type，必须先 open_evidence 读取它，不能只 retrieve 就写 claim。"
            )
            cloned["field_policy_hint"] = field_policy_hint(task)
            cloned["patch_meta"].update({"external_evidence_id": evidence_id, "local_caption_ids": list(local_ids)})
            return cloned
    return None


def patch_clone(row: dict[str, Any], *, patch_type: str) -> dict[str, Any]:
    cloned = deepcopy(row)
    cloned["messages"] = []
    cloned["label_source"] = f"v1_0_4_counterfactual_field_policy_{patch_type}"
    cloned["tool_schema_version"] = "v1.0.4_counterfactual_field_policy_no_select"
    cloned["variant"] = "v1_0_4_counterfactual_field_policy"
    cloned["patch_meta"] = {
        "patch_type": patch_type,
        "source_split": row.get("split"),
        "source_task_id": row.get("task_id"),
        "source_action": (row.get("action") or {}).get("action"),
    }
    return cloned


def has_local_caption_caption_support(task: dict[str, Any]) -> bool:
    local_ids = set(local_caption_ids(task))
    if not local_ids:
        return False
    for claim in (task.get("gold") or {}).get("claims") or []:
        if claim.get("field") != "caption_text" or claim.get("abstain"):
            continue
        if local_ids & {str(eid) for eid in claim.get("evidence_ids") or []}:
            return True
    return False


def retrieve_result(row: dict[str, Any]) -> dict[str, Any] | None:
    for result in reversed(row.get("tool_results") or []):
        if isinstance(result, dict) and result.get("tool") == "retrieve_evidence" and result.get("results"):
            return result
    return None


def local_caption_ids(task: dict[str, Any]) -> list[str]:
    return [
        str(item.get("evidence_id"))
        for item in task.get("local_evidence") or []
        if str(item.get("evidence_id") or "").startswith("local_caption_")
    ]


def local_caption_insufficient_reason(field: str) -> str:
    return (
        f"当前仅有 local caption，未打开能够明确支持 {field} 的外部 evidence；"
        f"不能用 local caption 泛化支持 {field}。"
    )


def field_policy_hint(task: dict[str, Any]) -> str:
    return (
        "local_caption 可支持 caption_text 和图注中明确出现的题名/作者/朝代等信息；"
        "image_scope/displayed_region/object_type 必须由明确文本或已打开外部 evidence 支持，否则 abstain。"
    )


def balanced_sample(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        out = list(rows)
        rng.shuffle(out)
        return out
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str((row.get("patch_meta") or {}).get("patch_type") or "")].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    cursor = 0
    while len(selected) < limit and keys:
        key = keys[cursor % len(keys)]
        cursor += 1
        bucket = buckets.get(key) or []
        if bucket:
            selected.append(bucket.pop())
        if not bucket:
            buckets.pop(key, None)
            keys = sorted(buckets)
            cursor = 0
    rng.shuffle(selected)
    return selected


def load_tasks_by_split(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {split: list(iter_jsonl(source_root / f"{split}_tasks.jsonl")) for split in ["train", "val"]}


def load_sft_rows_by_split(source_root: Path) -> dict[str, list[dict[str, Any]]]:
    return {split: list(iter_jsonl(source_root / "sft" / f"{split}.jsonl")) for split in ["train", "val"]}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("action") or {}).get("action") or "") for row in rows)


def patch_type_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(str((row.get("patch_meta") or {}).get("patch_type") or "") for row in rows)


def field_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        action = row.get("action") or {}
        for item in action.get("claims") or []:
            if isinstance(item, dict) and item.get("field"):
                counter[str(item.get("field"))] += 1
        for item in action.get("abstains") or []:
            if isinstance(item, dict) and item.get("field"):
                counter[str(item.get("field"))] += 1
    return counter


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Counterfactual Field Policy Replay 构建报告",
        "",
        f"- created_at: {manifest['created_at']}",
        f"- dataset_version: {manifest['dataset_version']}",
        f"- source_root: `{manifest['source_root']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## 构建原则",
        "",
        "- 不使用 test split。",
        "- 数据只用于小规模 field/evidence policy probe 或后续谨慎 SFT，不作为当前 best。",
        "- 目标是抑制 local caption 对非 caption 字段的泛化引用。",
        "",
        "## 规模",
        "",
    ]
    for key, value in manifest["counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Patch 类型", ""])
    for split, counts in manifest["patch_type_counts"].items():
        lines.append(f"### {split}")
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## 字段分布", ""])
    for split, counts in manifest["field_counts"].items():
        lines.append(f"### {split}")
        for key, value in counts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_output_name() -> str:
    return f"agentbench_v1_0_4_counterfactual_field_policy_replay_{datetime.now().strftime('%Y%m%d_%H%M')}"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
