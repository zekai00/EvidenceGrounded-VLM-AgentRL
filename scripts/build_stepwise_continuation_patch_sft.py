#!/usr/bin/env python3
"""Build SFT rows that mimic online stepwise TOOL_STATE_UPDATE prompts."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


FIELD_REASON = {
    "object_type": "证据不足，无法可靠判断对象类型",
    "visual_elements": "证据不足，无法可靠判断视觉元素",
    "technique": "证据不足，无法可靠判断技法",
    "composition": "证据不足，无法可靠判断构图",
    "medium_dimensions": "图注未明确材质或尺寸",
    "collection": "图注未明确馆藏信息",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main-sft-dir",
        default="/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_chunked_claim_sft_20260604_1650/sft",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=606051)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_dir) if args.output_dir else Path(
        f"/root/datasets/evidence_grounded_vlm_agentrl/agentbench_v0_6_stepwise_continuation_patch_sft_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=False)
    rng = random.Random(args.seed)

    train_source = read_jsonl(Path(args.main_sft_dir) / "train.jsonl")
    val_source = read_jsonl(Path(args.main_sft_dir) / "val.jsonl")
    train_rows = build_split(train_source, rng, {"write8": 240, "write4": 240, "object_only": 240, "finish": 96})
    val_rows = build_split(val_source, rng, {"write8": 32, "write4": 32, "object_only": 32, "finish": 16})
    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "val.jsonl", val_rows)
    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "purpose": "Patch empty generations after online TOOL_STATE_UPDATE claim-continuation states.",
        "source": str(Path(args.main_sft_dir)),
        "seed": args.seed,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_action_distribution": dict(action_counter(train_rows)),
        "val_action_distribution": dict(action_counter(val_rows)),
        "train_kind_distribution": dict(Counter(row.get("stepwise_patch_kind") for row in train_rows)),
        "val_kind_distribution": dict(Counter(row.get("stepwise_patch_kind") for row in val_rows)),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), **manifest}, ensure_ascii=False, indent=2))
    return 0


def build_split(source_rows: list[dict[str, Any]], rng: random.Random, counts: dict[str, int]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        action_name = (row.get("action") or {}).get("action")
        remaining = (row.get("claim_state") or {}).get("remaining_fields") or []
        if action_name == "write_claims_chunk" and len(remaining) == 8:
            buckets["write8"].append(row)
        elif action_name == "write_claims_chunk" and len(remaining) == 4:
            buckets["write4"].append(row)
            fields = action_fields(row.get("action") or {})
            if "object_type" in fields:
                buckets["object_only"].append(row)
        elif action_name == "finish" and not remaining:
            buckets["finish"].append(row)
    output: list[dict[str, Any]] = []
    for kind, count in counts.items():
        bucket = list(buckets.get(kind, []))
        if not bucket:
            continue
        rng.shuffle(bucket)
        chosen = bucket[:count] if len(bucket) >= count else bucket + rng.choices(bucket, k=max(0, count - len(bucket)))
        for row in chosen:
            output.append(make_stepwise_row(row, kind))
    rng.shuffle(output)
    return output


def make_stepwise_row(row: dict[str, Any], kind: str) -> dict[str, Any]:
    action = json.loads(json.dumps(row.get("action") or {}, ensure_ascii=False))
    claim_state = json.loads(json.dumps(row.get("claim_state") or {}, ensure_ascii=False))
    if kind == "object_only":
        claim_state["remaining_fields"] = ["object_type"]
        claim_state["written_fields"] = [field for field in claim_state.get("written_fields", []) if field != "object_type"]
        claim_state["abstained_fields"] = [field for field in claim_state.get("abstained_fields", []) if field != "object_type"]
        action = {
            "action": "write_claims_chunk",
            "claims": [],
            "abstains": [{"field": "object_type", "reason": FIELD_REASON["object_type"]}],
        }
    elif kind == "finish":
        claim_state["remaining_fields"] = []
        action = {"action": "finish", "status": "done"}

    phase = "finish_ready" if kind == "finish" else "claim_continuation"
    available_actions = ["finish"] if kind == "finish" else ["abstain_claim", "write_claims_chunk"]
    state = {
        "step": row.get("step"),
        "last_action": simplify_action((row.get("history") or [{}])[-1]) if row.get("history") else None,
        "last_reward": 0.1 if kind == "finish" else 0.0,
        "available_actions": available_actions,
        "phase": phase,
        "last_tool_result": {"tool": "write_claims_chunk", "claim_state": compact_claim_state(claim_state)},
        "available_regions": [],
        "available_region_ids": [],
        "selected_evidence_ids": row.get("selected_evidence_ids") or [],
        "visible_evidence_ids": list((claim_state.get("evidence_ids") or row.get("selected_evidence_ids") or [])[:8]),
        "valid_crop_count": 1,
        "last_crop": {"available": True, "role": "last_crop"},
        "claim_state": compact_claim_state(claim_state),
    }
    text = build_prompt(state, phase, claim_state)
    content: list[dict[str, Any]] = []
    for image_path in row.get("images") or []:
        content.append({"type": "image", "image": image_path})
    content.append({"type": "text", "text": text})
    new_row = {
        "task_id": f"{row.get('task_id')}_stepwise_{kind}_{row.get('step')}",
        "source_task_id": row.get("task_id"),
        "split": row.get("split"),
        "step": row.get("step"),
        "tool_schema_version": "v0.6_stepwise_continuation_patch",
        "action": action,
        "history": row.get("history") or [],
        "tool_results": row.get("tool_results") or [],
        "draft_claims": row.get("draft_claims") or [],
        "selected_evidence_ids": row.get("selected_evidence_ids") or [],
        "images": row.get("images") or [],
        "prompt_text": text,
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))},
        ],
        "label_source": "stepwise_continuation_patch",
        "claim_state": claim_state,
        "stepwise_patch_kind": kind,
    }
    return new_row


def build_prompt(state: dict[str, Any], phase: str, claim_state: dict[str, Any]) -> str:
    remaining = claim_state.get("remaining_fields") or []
    if phase == "finish_ready":
        phase_rule = "当前所有 claim 字段均已写完或 abstain；available_actions 只有 finish，必须输出 {\"action\":\"finish\",\"status\":\"done\"}。"
    else:
        first_remaining = str(remaining[0]) if remaining else "字段名"
        phase_rule = (
            "当前处于 claim 写入阶段；除非 remaining_fields 为空，否则必须继续写完或 abstain 剩余字段，"
            "不要 retrieve/open/propose/crop，不要空输出。"
            f" 如果证据不足，输出例如 {{\"action\":\"write_claims_chunk\",\"claims\":[],\"abstains\":[{{\"field\":\"{first_remaining}\",\"reason\":\"证据不足，无法可靠判断该字段\"}}]}}。"
        )
    return (
        "你是 evidence-grounded figure understanding 的 VLM tool-call agent。"
        "现在处于在线 stepwise rollout，中间工具调用已经完成；只根据下面状态输出下一步工具 JSON。\n\n"
        "[TOOL_STATE_UPDATE]\n"
        "硬约束：下一步 action 必须属于 available_actions；只输出一个非空 JSON 对象；不要输出空字符串；done 不是 action，只有 finish 出现在 available_actions 时才能用 {\"action\":\"finish\",\"status\":\"done\"}。\n"
        f"阶段规则：{phase_rule}\n"
        + json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        + "\n继续执行，只输出一个 JSON 对象。"
    )


def compact_claim_state(claim_state: dict[str, Any]) -> dict[str, Any]:
    return {
        "written_fields": claim_state.get("written_fields") or [],
        "abstained_fields": claim_state.get("abstained_fields") or [],
        "remaining_fields": claim_state.get("remaining_fields") or [],
    }


def simplify_action(action: Any) -> Any:
    if not isinstance(action, dict):
        return action
    keep = {key: action.get(key) for key in ["action", "field", "region_id", "scope", "top_k", "status"] if key in action}
    if "evidence_ids" in action:
        keep["evidence_ids"] = action.get("evidence_ids")
    if "claims" in action:
        keep["claims"] = action.get("claims")
    if "abstains" in action:
        keep["abstains"] = action.get("abstains")
    return keep


def action_fields(action: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for claim in action.get("claims") or []:
        if isinstance(claim, dict) and claim.get("field"):
            fields.add(str(claim.get("field")))
    for abstain in action.get("abstains") or []:
        if isinstance(abstain, dict) and abstain.get("field"):
            fields.add(str(abstain.get("field")))
    return fields


def action_counter(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter((row.get("action") or {}).get("action", "") for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
