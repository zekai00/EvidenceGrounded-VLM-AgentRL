#!/usr/bin/env python3
"""Build a region-selection hard-negative SFT patch dataset.

The patch targets crop_region states. It keeps replay rows, adds a short phase
hint only to crop_region rows, and oversamples crop states where the correct
region is not the first proposed candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from evidence_agent_env.prompting import PromptConfig, build_messages_from_observation  # noqa: E402


DEFAULT_INPUT_DIR = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/"
    "agentbench_v0_6_chunked_claim_sft_20260604_1650/sft"
)

DEFAULT_PHASE_HINT = (
    "阶段提示：当前已经看到 propose_regions 返回的候选区域，并且通常已经选择了本页图注 evidence。"
    "下一步如果需要裁剪，请调用 crop_region。正确目标图像不一定是第一个候选；"
    "不要裁剪 text_or_caption_candidate、正文、页眉页脚或图注框本身；"
    "优先选择与任务目标、图注内容和页面中的山水画图像区域一致的 figure_candidate。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-crop-oversample", type=int, default=3)
    parser.add_argument("--train-hard-oversample", type=int, default=8)
    parser.add_argument("--eval-crop-oversample", type=int, default=1)
    parser.add_argument("--replay-non-crop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--drop-pre-crop-select",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="For crop_region rows, remove earlier select_evidence state so the row matches crop-only region_selection masks.",
    )
    parser.add_argument("--phase-hint", default=DEFAULT_PHASE_HINT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_at": now(),
        "dataset_version": "v0.6_region_selection_hardnegative_patch",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "target_action": "crop_region",
        "train_crop_oversample": args.train_crop_oversample,
        "train_hard_oversample": args.train_hard_oversample,
        "eval_crop_oversample": args.eval_crop_oversample,
        "replay_non_crop": args.replay_non_crop,
        "drop_pre_crop_select": args.drop_pre_crop_select,
        "phase_hint": args.phase_hint,
        "seed": args.seed,
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        src = args.input_dir / f"{split}.jsonl"
        rows = read_jsonl(src)
        patched, stats = patch_split(rows, split=split, args=args)
        random.shuffle(patched)
        out_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(out_path, patched)
        stats["file"] = str(out_path)
        manifest["splits"][split] = stats
        print(
            f"[{split}] {stats['source_rows']} -> {stats['rows_written']} rows; "
            f"crop={stats['patched_action_counts'].get('crop_region', 0)}; "
            f"hard={stats['hard_crop_rows_seen']}; target_not_top1={stats['target_not_top1_rows_seen']}"
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(args.output_dir / "构建报告.md", manifest)
    print(f"manifest -> {manifest_path}")
    return 0


def patch_split(rows: list[dict[str, Any]], *, split: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    patched: list[dict[str, Any]] = []
    copy_counter: Counter[int] = Counter()
    target_rank_counter: Counter[str] = Counter()
    target_type_counter: Counter[str] = Counter()
    target_source_counter: Counter[str] = Counter()
    target_not_top1_rows_seen = 0
    hard_crop_rows_seen = 0
    hinted_crop_rows_seen = 0
    missing_region_rows = 0

    for row in rows:
        action = action_name(row)
        is_crop = action == "crop_region"
        diagnostic = crop_diagnostic(row) if is_crop else {}
        is_target_not_top1 = is_crop and int(diagnostic.get("target_rank") or 0) > 1
        is_hard = bool(is_target_not_top1)
        if is_crop:
            rank_value = diagnostic.get("target_rank")
            target_rank_counter[str(rank_value if rank_value is not None else "missing")] += 1
            target_type_counter[str(diagnostic.get("target_type") or "missing")] += 1
            target_source_counter[str(diagnostic.get("target_source") or "missing")] += 1
            target_not_top1_rows_seen += int(is_target_not_top1)
            hard_crop_rows_seen += int(is_hard)
            missing_region_rows += int(diagnostic.get("target_rank") is None)

        if not is_crop and not args.replay_non_crop:
            continue
        if is_crop and is_hard and split == "train":
            copies = max(1, args.train_hard_oversample)
        elif is_crop:
            copies = max(1, args.train_crop_oversample if split == "train" else args.eval_crop_oversample)
        else:
            copies = 1
        copy_counter[copies] += 1

        for copy_index in range(copies):
            copied = copy.deepcopy(row)
            if is_crop:
                if args.drop_pre_crop_select:
                    drop_pre_crop_select_state(copied)
                    rebuild_crop_messages(copied)
                add_phase_hint(copied, args.phase_hint)
                hinted_crop_rows_seen += int(copy_index == 0)
            copied["patch_source"] = {
                "kind": "v0_6_region_selection_hardnegative",
                "split": split,
                "is_crop_region": is_crop,
                "is_hard_negative": bool(is_hard and split == "train"),
                "copy_index": copy_index,
                "copies": copies,
                "crop_diagnostic": diagnostic,
            }
            patched.append(copied)

    return patched, {
        "source_rows": len(rows),
        "rows_written": len(patched),
        "source_action_counts": dict(sorted(action_counts(rows).items())),
        "patched_action_counts": dict(sorted(action_counts(patched).items())),
        "crop_rows_seen": sum(1 for row in rows if action_name(row) == "crop_region"),
        "target_not_top1_rows_seen": target_not_top1_rows_seen,
        "hard_crop_rows_seen": hard_crop_rows_seen if split == "train" else 0,
        "missing_region_rows": missing_region_rows,
        "hinted_crop_rows_seen": hinted_crop_rows_seen,
        "target_rank_distribution": dict(sorted(target_rank_counter.items(), key=lambda item: item[0])),
        "target_type_distribution": dict(sorted(target_type_counter.items())),
        "target_source_distribution": dict(sorted(target_source_counter.items())),
        "copy_count_distribution": {str(k): v for k, v in sorted(copy_counter.items())},
    }


def drop_pre_crop_select_state(row: dict[str, Any]) -> None:
    """Align old crop rows with the newer crop-only region_selection protocol."""
    row["history"] = [
        item
        for item in (row.get("history") or [])
        if not (isinstance(item, dict) and item.get("action") == "select_evidence")
    ]
    row["tool_results"] = [
        item
        for item in (row.get("tool_results") or [])
        if not (isinstance(item, dict) and item.get("tool") == "select_evidence")
    ]
    row["selected_evidence_ids"] = []
    row["claim_state"] = row.get("claim_state") or {
        "written_fields": [],
        "abstained_fields": [],
        "remaining_fields": [
            "caption_text",
            "image_scope",
            "depicted_work_title",
            "displayed_region",
            "object_type",
            "artist",
            "dynasty",
            "visual_elements",
            "technique",
            "composition",
            "medium_dimensions",
            "collection",
        ],
    }


def rebuild_crop_messages(row: dict[str, Any]) -> None:
    images = []
    for index, image in enumerate(row.get("images") or []):
        if isinstance(image, dict):
            images.append(image)
        elif image:
            images.append({"path": str(image), "role": "page" if index == 0 else "image"})
    obs = {
        "task_id": row.get("task_id"),
        "source_file": row.get("source_file", ""),
        "page": row.get("page", ""),
        "step": row.get("step"),
        "images": images,
        "history": row.get("history") or [],
        "tool_results": row.get("tool_results") or [],
        "draft_claims": row.get("draft_claims") or [],
        "claim_state": row.get("claim_state") or {},
        "selected_evidence_ids": row.get("selected_evidence_ids") or [],
        "visible_evidence_ids": [],
        "available_actions": ["crop_region"],
        "tool_mask": {
            "enabled": True,
            "phase": "region_selection",
            "allowed_actions": ["crop_region"],
            "reason": "candidate regions are available; crop a target region before selecting evidence.",
            "step": row.get("step"),
        },
    }
    prompt_config = PromptConfig(
        max_history_actions=8,
        max_tool_results=6,
        max_evidence_per_result=3,
        snippet_chars=180,
        max_text_chars=24000,
        head_text_chars=5000,
        coordinate_info=True,
        tool_schema="chunked_claim",
        compact_claim_state=True,
        region_selection_hint=True,
        strict_claim_phase_hint=True,
    )
    row["messages"] = build_messages_from_observation(
        obs,
        prompt_config,
        include_assistant_action=row.get("action") if isinstance(row.get("action"), dict) else None,
    )
    row["prompt_text"] = first_user_text(row["messages"])


def first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text")
    return ""


def crop_diagnostic(row: dict[str, Any]) -> dict[str, Any]:
    action = row.get("action") or {}
    target_region_id = str(action.get("region_id") or "")
    regions = first_regions(row)
    target = None
    target_rank = None
    for rank, region in enumerate(regions, start=1):
        if str(region.get("region_id")) == target_region_id:
            target = region
            target_rank = rank
            break
    top1 = regions[0] if regions else {}
    return {
        "target_region_id": target_region_id,
        "target_rank": target_rank,
        "target_bbox": target.get("bbox") if target else None,
        "target_type": target.get("type") if target else None,
        "target_source": target.get("source") if target else None,
        "top1_region_id": top1.get("region_id"),
        "top1_type": top1.get("type"),
        "top1_source": top1.get("source"),
        "candidate_count": len(regions),
        "caption_candidate_count": sum(int(bool(region.get("caption_evidence_id"))) for region in regions),
    }


def first_regions(row: dict[str, Any]) -> list[dict[str, Any]]:
    for result in row.get("tool_results") or []:
        if isinstance(result, dict) and result.get("tool") == "propose_regions":
            return [item for item in (result.get("regions") or []) if isinstance(item, dict)]
    return []


def add_phase_hint(row: dict[str, Any], hint: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            if hint not in content:
                message["content"] = f"{hint}\n\n{content}"
            return
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", ""))
                    if hint not in text:
                        item["text"] = f"{hint}\n\n{text}"
                    return
            content.append({"type": "text", "text": hint})
            return


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row.get("task_id", "")), int(row.get("step", 0) or 0)


def action_name(row: dict[str, Any]) -> str:
    action = row.get("action")
    return str(action.get("action", "")) if isinstance(action, dict) else ""


def action_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(action_name(row) for row in rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v0.6 Region Selection Hard-Negative Patch SFT 构建报告",
        "",
        f"生成时间：{manifest['created_at']}",
        "",
        "## 1. 目标",
        "",
        "本数据集用于修复 v0.6 250-step adapter 在 `crop_region` 阶段选择错误 `region_id` 的问题。",
        "当前候选池 top-k oracle 召回已经足够，主要瓶颈是模型不会稳定从多个候选里选对目标图像区域。",
        "",
        "## 2. 数据位置",
        "",
        f"- 输入：`{manifest['input_dir']}`",
        f"- 输出：`{manifest['output_dir']}`",
        "",
        "## 3. 构建策略",
        "",
        "- 保留非 `crop_region` replay rows，避免 patch 只学局部动作后破坏整体工具 schema。",
        "- 对 `crop_region` rows 加入阶段提示。",
        "- 对训练集中正确 region 不是 top1 的 hard-negative 样本做更高过采样。",
        "- 不在提示中写入正确 `region_id`。",
        "",
        "## 4. Split 统计",
        "",
        "| split | source rows | rows written | crop rows | target not top1 | hard rows | patched crop rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, stats in manifest["splits"].items():
        crop_written = stats["patched_action_counts"].get("crop_region", 0)
        lines.append(
            f"| {split} | {stats['source_rows']} | {stats['rows_written']} | "
            f"{stats['crop_rows_seen']} | {stats['target_not_top1_rows_seen']} | "
            f"{stats['hard_crop_rows_seen']} | {crop_written} |"
        )
    lines.extend(
        [
            "",
            "## 5. Rank 分布",
            "",
        ]
    )
    for split, stats in manifest["splits"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                "```json",
                json.dumps(
                    {
                        "target_rank_distribution": stats["target_rank_distribution"],
                        "target_type_distribution": stats["target_type_distribution"],
                        "target_source_distribution": stats["target_source_distribution"],
                        "copy_count_distribution": stats["copy_count_distribution"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## 6. 阶段提示",
            "",
            "```text",
            manifest["phase_hint"],
            "```",
            "",
            "## 7. 注意事项",
            "",
            "- 这是 SFT patch 数据，不是 on-policy RL 数据。",
            "- 该 patch 只解决 region selection，不直接解决 evidence/claim。",
            "- 训练后必须同时评测 `crop_region` 和 `write_claims_chunk`，避免局部 patch 破坏 claim 写入。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
