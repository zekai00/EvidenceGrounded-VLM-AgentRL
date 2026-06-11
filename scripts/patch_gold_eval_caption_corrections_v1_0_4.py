#!/usr/bin/env python3
"""Apply a traceable caption correction patch to v1.0.4 GoldEval."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = "/root/datasets/evidence_grounded_vlm_agentrl/gold_eval_v1_0_4_20260611_1701"
DEFAULT_OUTPUT_ROOT = "/root/datasets/evidence_grounded_vlm_agentrl"


CORRECTIONS: dict[str, dict[str, Any]] = {
    # Human reviewed multi-caption splits.
    "egva_v0_9_fixed_001882": {
        "new_caption_text": "图4．3明·仇英《枫溪垂钓图》(局部)",
        "correction_type": "multi_caption_split",
        "correction_source": "human_review",
        "notes": "原 caption 同时包含图4.3和图4.4；目标对应图4.3《枫溪垂钓图》。",
    },
    "egva_v0_9_fixed_001734": {
        "new_caption_text": "〔图七〕 《重江叠嶂图》 卷本幅部分折痕",
        "correction_type": "multi_caption_split",
        "correction_source": "human_review_and_vlm_suggested",
        "notes": "原 caption 同时包含图七和图八；目标对应图七。",
    },
    "egva_v0_9_fixed_001774": {
        "new_caption_text": "图6: 关山行旅图",
        "correction_type": "multi_caption_split",
        "correction_source": "human_review_and_vlm_suggested",
        "notes": "目标在右侧，对应图6；图5《匡庐图》为相邻图。",
    },
    "egva_v0_9_fixed_001889": {
        "new_caption_text": "图5-10明·董其昌《秋兴八景图册》之八",
        "correction_type": "multi_caption_split",
        "correction_source": "human_review_and_vlm_suggested",
        "notes": "目标对应后一半图5-10；图5-9为相邻图。",
    },
    # Human/visual checked completion.
    "egva_v0_9_fixed_001643": {
        "new_caption_text": "图1-3 仇英《西园雅集图》 纸本 86.6×30厘米 台北故宫博物院",
        "correction_type": "caption_completion_and_ocr_fix",
        "correction_source": "human_review_visual",
        "notes": "overlay 可见完整图注；原 caption 截断且将“西园”误识别为“西元”。",
        "depicted_work_title": "西园雅集图",
    },
    # VLM suggested, low-risk completion of accepted truncated captions.
    "egva_v0_9_fixed_001522": {
        "new_caption_text": "Fig. 156. Ch'en Jung (active ca. 1235-62). Nine Dragons, dated 1244. Detail. Handscroll, ink and touch of red pigment on paper, 18 1/4 x 431 7/8 in. (46.3 x 1096.4 cm). Francis Gardner Curtis Fund, Museum of Fine Arts, Boston (17.1697)",
        "correction_type": "truncated_caption_completion",
        "correction_source": "vlm_suggested_human_review_reason_confirmed",
        "notes": "VLM 明确指出末尾截断并给出候选补全文本；人工确认应补全截断图注。",
    },
    "egva_v0_9_fixed_001541": {
        "new_caption_text": "Figure 39. Unidentified artist (14th century), Along the Riverbank at Dusk. Hanging scroll, ink and color on silk, 70 1/2 x 45 7/8 in. (179 x 116.5 cm), National Palace Museum, Taipei",
        "correction_type": "truncated_caption_completion",
        "correction_source": "vlm_suggested",
        "notes": "VLM 一审标记为 truncated，并给出补全文本；核心目标和图注匹配为 good。",
    },
    # Low-risk OCR normalization.
    "egva_v0_9_fixed_001579": {
        "new_caption_text": "Fig. 67 Farewell by a Stream on a Clear Day Chao Yuan, active ca. 1350-75 Hanging scroll, ink on paper The Metropolitan Museum of Art",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正 OCR 噪声和标点，不改变目标对应关系。",
    },
    "egva_v0_9_fixed_001552": {
        "new_caption_text": "Figure 81. Lü Ji (ca. 1430-ca. 1504), \"Autumn,\" from Birds and Flowers of the Four Seasons. Set of four hanging scrolls, ink and color on silk, each 69 1/4 x 39 1/2 in. (176 x 100.8 cm). Tokyo National Museum",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正图号、姓名、年代和尺寸 OCR 噪声。",
    },
    "egva_v0_9_fixed_001528": {
        "new_caption_text": "Figure 5. Attributed to Dong Yuan (active ca. 930s-60s), Wintry Groves and Layered Banks. Hanging scroll, ink and color on silk, 71 1/2 x 45 7/8 in. (181.5 x 116.5 cm). Kurokawa Institute of Ancient Cultures, Hyogo Prefecture, Japan",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正数字、标题和尺寸 OCR 噪声。",
    },
    "egva_v0_9_fixed_001887": {
        "new_caption_text": "图5-3明·董其昌《秋兴八景图册》之一 图5-4明·董其昌《秋兴八景图册》之二",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正“重其昌”为“董其昌”。",
    },
    "egva_v0_9_fixed_001793": {
        "new_caption_text": "图2-8 《匡庐图》 Figure 2-8 \"Kuanglu\"",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正英文引号和截断引号。",
    },
    "egva_v0_9_fixed_001789": {
        "new_caption_text": "图2-2 《潇湘图》（局部） Figure 2-2 \"Xiaoxiang map\" (local)",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正空格与英文引号。",
    },
    "egva_v0_9_fixed_001860": {
        "new_caption_text": "图2.2.8 戴进《聘贤图》",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正图号 OCR 空格。",
    },
    "egva_v0_9_fixed_001802": {
        "new_caption_text": "图4-4 《富春山居图》（局部） Figure 4-4 \"Fuchun Mountains\" (local)",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正英文引号和空格。",
    },
    "egva_v0_9_fixed_001846": {
        "new_caption_text": "图3-18 北宋范宽《溪山行旅图》绢本 台北故宫博物院藏 206.3cm*106.3cm",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正“台 北”断字和空格。",
    },
    "egva_v0_9_fixed_001868": {
        "new_caption_text": "图4.3.1 雪舟《四季山水图卷》",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正图号和“^水”为“山水”。",
    },
    "egva_v0_9_fixed_001893": {
        "new_caption_text": "图6-5《秋林赏菊图》110x69cm纸本设色",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正尺寸 OCR 空格。",
    },
    "egva_v0_9_fixed_001874": {
        "new_caption_text": "《坐看云起时》 180cm×145cm",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正尺寸 OCR 空格。",
    },
    "egva_v0_9_fixed_001898": {
        "new_caption_text": "图6-10《守望青山》138×69cm纸本设色",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正图号 6-lO 为 6-10。",
    },
    "egva_v0_9_fixed_001870": {
        "new_caption_text": "《古道通幽》 180cm×145cm",
        "correction_type": "ocr_normalization",
        "correction_source": "vlm_suggested",
        "notes": "修正尺寸 OCR 空格。",
    },
}


SKIPPED_CANDIDATES = [
    {
        "task_id": "egva_v0_9_fixed_001810",
        "reason": "VLM suggested text appends URL/date not present in current caption; require PDF/page verification before writing back.",
    },
    {
        "task_id": "egva_v0_9_fixed_001811",
        "reason": "VLM suggested text appends missing date; require PDF/page verification before writing back.",
    },
    {
        "task_id": "egva_v0_9_fixed_001814",
        "reason": "VLM suggested text appends URL/date; require PDF/page verification before writing back.",
    },
    {
        "task_id": "egva_v0_9_fixed_001819",
        "reason": "VLM suggested text appends image source after a truncated marker; leave for manual/page verification.",
    },
    {
        "task_id": "egva_v0_9_fixed_001794",
        "reason": "English translation is noisy and VLM suggestion does not materially improve it; leave unchanged.",
    },
    {
        "task_id": "egva_v0_9_fixed_001555",
        "reason": "In human review queue, not in GoldEval accepted split; do not add to GoldEval by caption patch.",
    },
    {
        "task_id": "egva_v0_9_fixed_001597",
        "reason": "In human review queue, not in GoldEval accepted split; do not add to GoldEval by caption patch.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch GoldEval caption_text with traceable corrections.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = resolve_output_dir(args)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    if output_dir.exists():
        raise RuntimeError(f"Output directory already exists: {output_dir}")
    shutil.copytree(input_dir, output_dir)

    all_corrections: list[dict[str, Any]] = []
    split_summaries = {}
    for split, filename in [("val", "val_gold_50.jsonl"), ("test", "test_gold_100.jsonl")]:
        rows = read_jsonl(output_dir / filename)
        patched_rows, corrections = patch_rows(rows)
        write_jsonl(output_dir / filename, patched_rows)
        all_corrections.extend({"split": split, **item} for item in corrections)
        split_summaries[split] = summarize_split(patched_rows, corrections)

    write_jsonl(output_dir / "caption_corrections.jsonl", all_corrections)
    write_jsonl(output_dir / "caption_correction_skipped_candidates.jsonl", SKIPPED_CANDIDATES)
    summary = build_summary(input_dir, output_dir, all_corrections, split_summaries)
    write_json(output_dir / "caption_correction_summary.json", summary)
    update_manifest(output_dir, input_dir, summary)
    write_report(output_dir / "CaptionCorrectionPatch报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return Path(args.output_root) / f"gold_eval_v1_0_4_caption_corrected_{stamp}"


def patch_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patched = []
    corrections = []
    for row in rows:
        row = copy.deepcopy(row)
        task_id = str(row.get("task_id") or "")
        correction = CORRECTIONS.get(task_id)
        if correction:
            old_caption = ((row.get("gold") or {}).get("caption_text") or "")
            apply_caption_correction(row, correction)
            corrections.append(
                {
                    "task_id": task_id,
                    "old_caption_text": old_caption,
                    "new_caption_text": correction["new_caption_text"],
                    "correction_type": correction["correction_type"],
                    "correction_source": correction["correction_source"],
                    "notes": correction["notes"],
                    "depicted_work_title": correction.get("depicted_work_title"),
                }
            )
        patched.append(row)
    return patched, corrections


def apply_caption_correction(row: dict[str, Any], correction: dict[str, Any]) -> None:
    gold = row.setdefault("gold", {})
    old_caption = str(gold.get("caption_text") or "")
    new_caption = correction["new_caption_text"]
    gold["caption_original_text"] = old_caption
    gold["caption_text"] = new_caption
    gold["caption_correction"] = {
        "applied": True,
        "version": "v1.0.4_caption_correction_patch",
        "old_caption_text": old_caption,
        "new_caption_text": new_caption,
        "correction_type": correction["correction_type"],
        "correction_source": correction["correction_source"],
        "notes": correction["notes"],
        "created_at": now_cst(),
    }
    row.setdefault("gold_eval", {})["caption_correction"] = gold["caption_correction"]
    row["gold_eval_status"] = "accepted_gold"

    for claim in gold.get("claims") or []:
        if claim.get("field") == "caption_text":
            claim["original_value"] = claim.get("value")
            claim["value"] = new_caption
            claim["caption_correction_applied"] = True
        if correction.get("depicted_work_title") and claim.get("field") == "depicted_work_title":
            claim["original_value"] = claim.get("value")
            claim["original_abstain"] = claim.get("abstain")
            claim["value"] = correction["depicted_work_title"]
            claim["abstain"] = False
            claim["evidence_ids"] = caption_evidence_ids(gold)
            claim["support_type"] = "page_caption_text"
            claim["caption_correction_applied"] = True

    if correction.get("depicted_work_title"):
        gold["depicted_work_title_original"] = gold.get("depicted_work_title")
        gold["depicted_work_title"] = correction["depicted_work_title"]

    for item in row.get("local_evidence") or []:
        if str(item.get("evidence_id", "")).startswith("local_caption_"):
            item["original_display_snippet"] = item.get("display_snippet")
            item["display_snippet"] = new_caption
            item["caption_correction_applied"] = True

    for candidate in row.get("region_candidates") or []:
        for key in ["caption_hint", "linked_caption_text", "nearby_text"]:
            if candidate.get(key) == old_caption:
                candidate[f"original_{key}"] = candidate.get(key)
                candidate[key] = new_caption

    if gold.get("evidence_query"):
        gold["evidence_query_original"] = gold.get("evidence_query")
        gold["evidence_query"] = str(gold["evidence_query"]).replace(old_caption, new_caption)


def caption_evidence_ids(gold: dict[str, Any]) -> list[str]:
    ids = [str(item) for item in gold.get("evidence_ids") or [] if str(item).startswith("local_caption_")]
    if ids:
        return ids[:1]
    for claim in gold.get("claims") or []:
        if claim.get("field") == "caption_text":
            return claim.get("evidence_ids") or []
    return []


def summarize_split(rows: list[dict[str, Any]], corrections: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "unique_task_ids": len({row.get("task_id") for row in rows}),
        "corrections": len(corrections),
        "correction_type_counts": dict(Counter(item["correction_type"] for item in corrections)),
        "correction_source_counts": dict(Counter(item["correction_source"] for item in corrections)),
        "gold_eval_status_counts": dict(Counter(row.get("gold_eval_status") for row in rows)),
        "caption_correction_rows": sum(1 for row in rows if ((row.get("gold") or {}).get("caption_correction") or {}).get("applied")),
    }


def build_summary(input_dir: Path, output_dir: Path, corrections: list[dict[str, Any]], split_summaries: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": now_cst(),
        "version": "v1.0.4_caption_correction_patch",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "corrections": len(corrections),
        "correction_type_counts": dict(Counter(item["correction_type"] for item in corrections)),
        "correction_source_counts": dict(Counter(item["correction_source"] for item in corrections)),
        "splits": split_summaries,
        "skipped_candidates": len(SKIPPED_CANDIDATES),
        "outputs": {
            "val_gold_50": str(output_dir / "val_gold_50.jsonl"),
            "test_gold_100": str(output_dir / "test_gold_100.jsonl"),
            "caption_corrections": str(output_dir / "caption_corrections.jsonl"),
            "skipped_candidates": str(output_dir / "caption_correction_skipped_candidates.jsonl"),
            "report": str(output_dir / "CaptionCorrectionPatch报告.md"),
        },
        "policy": [
            "Do not overwrite the original GoldEval directory.",
            "Only patch accepted GoldEval rows, not train or raw silver tasks.",
            "Keep original caption text and correction provenance in each patched row.",
            "Skip candidates that require unverifiable URL/date/source completion.",
        ],
    }


def update_manifest(output_dir: Path, input_dir: Path, summary: dict[str, Any]) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["caption_correction_patch"] = {
        "input_dir": str(input_dir),
        "summary": summary,
    }
    write_json(manifest_path, manifest)
    summary_path = output_dir / "summary.json"
    if summary_path.exists():
        base_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        base_summary["caption_correction_patch"] = summary
        write_json(summary_path, base_summary)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 GoldEval Caption Correction Patch 报告",
        "",
        f"生成时间：{summary['created_at']}",
        "",
        "## 输入与输出",
        "",
        f"- 输入 GoldEval：`{summary['input_dir']}`",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- 修正明细：`{summary['outputs']['caption_corrections']}`",
        f"- 跳过候选：`{summary['outputs']['skipped_candidates']}`",
        "",
        "## 修正规模",
        "",
        f"- 总修正数：{summary['corrections']}",
        f"- correction_type：`{json.dumps(summary['correction_type_counts'], ensure_ascii=False)}`",
        f"- correction_source：`{json.dumps(summary['correction_source_counts'], ensure_ascii=False)}`",
        f"- 跳过候选数：{summary['skipped_candidates']}",
        "",
    ]
    for split, item in summary["splits"].items():
        lines.extend(
            [
                f"### {split}",
                "",
                f"- 行数：{item['rows']}",
                f"- unique task_id：{item['unique_task_ids']}",
                f"- 修正数：{item['corrections']}",
                f"- 修正类型：`{json.dumps(item['correction_type_counts'], ensure_ascii=False)}`",
                f"- 已带 caption_correction 的行：{item['caption_correction_rows']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 原则",
            "",
            "- 不覆盖原始 GoldEval，而是生成 caption-corrected 新目录。",
            "- 只修 accepted GoldEval，不改 train 或原始 silver task。",
            "- 每条保留 `caption_original_text`、`caption_correction`、claim `original_value` 和 local evidence `original_display_snippet`。",
            "- 对需要 PDF 进一步确认的 URL、日期、来源补全先跳过。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def now_cst() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CST")


if __name__ == "__main__":
    raise SystemExit(main())
