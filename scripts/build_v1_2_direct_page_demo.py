#!/usr/bin/env python3
"""Build a small v1.2 direct page-image demo with remote VLM calls.

This script intentionally skips PDF scanning. It consumes an existing
page_records.jsonl directory whose page_image files already exist.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_gold_eval_v1_0_4 as gold_review  # noqa: E402
import build_v1_1_clean_evidence_fragment_probe as v11  # noqa: E402


DEFAULT_INPUT_DIR = Path("/root/datasets/evidence_grounded_vlm_agentrl/v1_2_remote_vlm_first_probe_select10_filtered_offline")
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")
MODEL = "qwen3.7-max-2026-06-08"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
META_FIELDS = [
    "creator_or_attribution",
    "creation_period_or_dynasty",
    "collection_institution",
    "dimensions",
    "medium_material",
]

PROMPT = """你是 EvidenceGrounded-VLM-AgentRL v1.2 的单页标注员。请看一整页 PDF 图，找出中国/东亚古典山水画或山水画局部目标，并输出 JSON。

关键规则：
1. bbox 统一使用 0-1000 归一化坐标 [x1,y1,x2,y2]，不是像素坐标。原点是页面左上角，x 按页面宽度归一化，y 按页面高度归一化。
2. target_bbox_norm1000 只框目标图像，不含图注、正文、页码、相邻图。
3. caption_bbox_norm1000 框完整对应图注；多行图注要全框。
4. 如果同一行里有多个图注，caption_bbox_norm1000 只框当前 detection 对应的那一段图注，不要框整行。
5. 如果目标图像没有可见图注，caption_text 写 null，caption_bbox_norm1000 写 null；不要为了凑字段去框正文。
6. caption_text 以页面视觉读字为准；如果 PDF text preview 读错，以图像为准。
7. 不要用模型内部知识补全页面没有出现的信息。
8. collection_institution 只能填明确的收藏机构/藏馆名，例如 National Palace Museum, Taipei、The Metropolitan Museum of Art、北京故宫博物院、台北故宫博物院。Gift/Bequest/Purchase/Accession number/编号/捐赠/购藏/入藏说明都不是 collection_institution；如果只有这些信息而没有机构名，collection_institution 必须 abstain=true。
9. Metadata5 没有页面可见证据就 abstain=true。
10. 多图页必须拆成多条 detection：每一张独立图、局部图、册页、对比图中的子图都要单独输出自己的 target_bbox_norm1000、caption_text、depicted_work_title、image_scope、object_type 和 metadata_fields。
11. 不允许把多个作品或多个编号子图用一个大 target_bbox_norm1000 框起来；如果一个图注写“由上至下/从左至右/1/2/3/4”，请按编号输出，每个编号一条 detection。
12. 一个编号项就是一条 detection：即使该编号项视觉上由左右两半、多块拼接、翻转示意、对照局部组成，也必须合成同一条 detection，target_bbox_norm1000 覆盖该编号项的整体视觉范围，不要拆成两条。
13. 子图共享同一条总图注时，caption_text 可以写“总图号 + 当前编号子图说明”，例如“〔图二十四〕1.《鹊华秋色图》卷左右两半位置对调示意图”；caption_bbox_norm1000 可以指向共享图注中覆盖该编号说明的文本区域，不能因为共享图注就合并 target。

只输出 JSON：
{
  "page_summary": "一句话",
  "detections": [
    {
      "target_bbox_norm1000": [0,0,0,0],
      "caption_bbox_norm1000": [0,0,0,0] 或 null,
      "caption_text": "完整 corrected caption 或 null",
      "depicted_work_title": "题名或空串",
      "image_scope": "full_work|partial_detail|album_leaf_or_section|multi_work_comparison|unclear",
      "object_type": "painting|painting_detail|diagram|text_page|photo|other|unclear",
      "object_domain": "landscape_painting|landscape_detail|classical_painting_unclear_landscape|non_landscape_artwork|text_only|other|unclear",
      "caption_target_match": "yes|no|uncertain",
      "metadata_fields": {
        "creator_or_attribution": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "creation_period_or_dynasty": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "collection_institution": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "dimensions": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""},
        "medium_material": {"value": "", "abstain": true, "source": "unsupported", "confidence": 0.0, "reason": ""}
      },
      "accept_for_probe": true,
      "needs_human_review": false,
      "confidence": 0.0,
      "reason": "简短说明"
    }
  ]
}

页面元数据：
"""

ORG_PATTERNS = [
    re.compile(r"(National Palace Museum(?:,?\s*Taipei)?)", re.I),
    re.compile(r"(The Metropolitan Museum of Art|Metropolitan Museum of Art)", re.I),
    re.compile(r"(Palace Museum(?:,?\s*Beijing)?)", re.I),
    re.compile(r"(Museum of Fine Arts,?\s*Boston|Freer Gallery of Art|Cleveland Museum of Art|Princeton University Art Museum)", re.I),
    re.compile(r"((?:北京|台北|臺北|南京|上海|辽宁|遼寧)?故宫博物院)"),
    re.compile(r"([\u4e00-\u9fff]{2,20}(?:博物院|博物馆|美术馆|藝術館|艺术馆))"),
]
BAD_COLLECTION_RE = re.compile(
    r"\b(Gift|Bequest|Purchase|Accession|Anonymous Loan|Lent by)\b|捐赠|購藏|购藏|入藏|编号|館藏號|藏品編號",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dotenv", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DASHSCOPE_API_KEY")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--disable-thinking", action="store_true", help="Pass Qwen3.x chat_template_kwargs enable_thinking=false.")
    parser.add_argument("--pages", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--image-max-side", type=int, default=0, help="0 sends the original page image; positive values downscale max side.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--bbox-snap", choices=["none"], default="none", help="Deprecated compatibility flag; bbox conversion is norm1000-only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    gold_review.load_dotenv(Path(args.dotenv))
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"v1_2_direct_10page_remote_vlm_demo_{stamp}"
    for sub in ["pages", "overlays", "crops", "captions", "json"]:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    records = []
    for row in v11.read_jsonl(Path(args.input_dir) / "page_records.jsonl"):
        if Path(row.get("page_image", "")).exists():
            records.append(row)
    records = records[: args.pages]

    rows: list[dict[str, Any] | None] = [None] * len(records)
    max_workers = max(1, min(args.concurrency, len(records)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(review_one, i, record, args, out_dir) for i, record in enumerate(records)]
        for future in as_completed(futures):
            index, row = future.result()
            rows[index] = row
            print(
                json.dumps(
                    {
                        "done": index + 1,
                        "source_file": row.get("source_file"),
                        "page_num": row.get("page_num"),
                        "detections": len(row.get("detections") or []),
                        "ok": row.get("ok"),
                        "elapsed_sec": row.get("elapsed_sec"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    final_rows = [row for row in rows if row is not None]
    write_assets(out_dir, records, final_rows)
    write_review_md(out_dir, args, final_rows)
    (out_dir / "json" / "page_level_vlm_annotations.json").write_text(
        json.dumps(final_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("OUTPUT_DIR", out_dir)
    print("MARKDOWN", out_dir / "v1.2直接单页远端VLM_10页样例.md")
    print("JSON", out_dir / "json" / "page_level_vlm_annotations.json")
    return 0


def review_one(index: int, record: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> tuple[int, dict[str, Any]]:
    page_image = Path(record["page_image"])
    with Image.open(page_image) as image:
        width, height = image.size
    image_url, sent_size = image_data_url(page_image, out_dir, args.image_max_side)
    meta = {
        "source_file": record.get("source_file"),
        "page_num": record.get("page_num"),
        "page_size_px": [width, height],
        "sent_image_size_px": list(sent_size),
        "pdf_text_preview": v11.truncate(record.get("raw_pdf_text") or record.get("pdf_text") or "", 900),
    }
    client = OpenAI(
        api_key=args.api_key or os.environ.get(args.api_key_env) or "EMPTY",
        base_url=args.base_url,
        timeout=args.timeout,
    )
    extra_body = {"chat_template_kwargs": {"enable_thinking": False}} if args.disable_thinking else None
    start = time.time()
    raw = ""
    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": PROMPT + json.dumps(meta, ensure_ascii=False, indent=2)},
                    ],
                }
            ],
            temperature=0,
            max_tokens=args.max_tokens,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            parsed = parse_json(raw)
            ok = True
            error = ""
        except Exception as exc:
            parsed = {"page_summary": "", "detections": []}
            ok = False
            error = repr(exc)
    except Exception as exc:
        parsed = {"page_summary": "", "detections": []}
        ok = False
        error = repr(exc)

    parsed_items = [item for item in (parsed.get("detections") if isinstance(parsed.get("detections"), list) else []) if isinstance(item, dict)]
    coordinate_mode = "norm1000"
    detections = []
    for det_index, item in enumerate(parsed_items):
        raw_target_bbox = normalize_bbox(item.get("target_bbox_norm1000"))
        raw_caption_bbox = normalize_bbox(item.get("caption_bbox_norm1000"))
        raw_caption_text = item.get("caption_text")
        caption_text = None if raw_caption_text is None else v11.normalize_space(raw_caption_text or "")
        target_resolution = resolve_norm1000_bbox(raw_target_bbox, width, height)
        caption_resolution = resolve_norm1000_bbox(raw_caption_bbox, width, height)
        metadata = item.get("metadata_fields") if isinstance(item.get("metadata_fields"), dict) else {}
        for field in META_FIELDS:
            metadata.setdefault(field, empty_field())
        metadata["collection_institution"] = strict_collection(metadata.get("collection_institution"))
        detections.append(
            {
                "detection_index": det_index,
                "raw_target_bbox_returned": raw_target_bbox,
                "raw_caption_bbox_returned": raw_caption_bbox,
                "raw_target_bbox_key": "target_bbox_norm1000",
                "raw_caption_bbox_key": "caption_bbox_norm1000",
                "target_bbox_px": target_resolution.get("bbox"),
                "caption_bbox_px": caption_resolution.get("bbox"),
                "target_bbox_norm1000": target_resolution.get("norm1000"),
                "caption_bbox_norm1000": caption_resolution.get("norm1000"),
                "target_bbox": target_resolution.get("bbox"),
                "caption_bbox": caption_resolution.get("bbox"),
                "target_bbox_coordinate_source": target_resolution.get("source"),
                "caption_bbox_coordinate_source": caption_resolution.get("source"),
                "page_coordinate_mode": coordinate_mode,
                "caption_text": caption_text,
                "corrected_caption_text": caption_text,
                "depicted_work_title": v11.normalize_space(item.get("depicted_work_title") or ""),
                "image_scope": v11.normalize_space(item.get("image_scope") or "unclear"),
                "object_type": v11.normalize_space(item.get("object_type") or "unclear"),
                "object_domain": v11.normalize_space(item.get("object_domain") or "unclear"),
                "caption_target_match": v11.normalize_space(item.get("caption_target_match") or "uncertain"),
                "metadata_fields": metadata,
                "accept_for_probe": bool(item.get("accept_for_probe")),
                "needs_human_review": bool(item.get("needs_human_review")),
                "confidence": v11.safe_float(item.get("confidence"), 0.0),
                "reason": str(item.get("reason") or ""),
            }
        )

    return index, {
        **{
            k: record.get(k)
            for k in [
                "page_id",
                "doc_id",
                "source_file",
                "source_path",
                "rel_path",
                "page_num",
                "page_image",
                "width",
                "height",
                "category",
            ]
        },
        "ok": ok,
        "error": error,
        "review_model": args.model,
        "elapsed_sec": round(time.time() - start, 2),
        "page_summary": str(parsed.get("page_summary") or ""),
        "detections": detections,
        "raw_response": raw,
        "sent_image_size_px": list(sent_size),
        "page_coordinate_mode": coordinate_mode,
    }


def image_data_url(path: Path, out_dir: Path, max_side: int) -> tuple[str, tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        if max_side > 0:
            image.thumbnail((max_side, max_side))
        sent_size = image.size
        tmp = out_dir / "json" / f"tmp_{v11.sha1_text(str(path) + str(max_side))[:12]}.jpg"
        image.save(tmp, quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(tmp.read_bytes()).decode("ascii"), sent_size


def parse_json(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        raise ValueError("no JSON object found")
    text = match.group(0)
    try:
        return json.loads(text)
    except Exception:
        repaired = repair_common_json_commas(text)
        return json.loads(repaired)


def repair_common_json_commas(text: str) -> str:
    # Remote VLM sometimes returns JSON-looking output missing a comma before
    # the next key in long objects. Keep this conservative and schema-key based.
    keys = [
        "accept_for_probe",
        "needs_human_review",
        "confidence",
        "reason",
        "metadata_fields",
        "caption_target_match",
        "object_domain",
        "object_type",
        "image_scope",
        "depicted_work_title",
        "caption_text",
        "caption_bbox_norm1000",
        "target_bbox_norm1000",
    ]
    key_alt = "|".join(re.escape(k) for k in keys)
    text = re.sub(rf'([}}\]"])\\s*\\n\\s*("({key_alt})"\\s*:)', r"\1,\n\2", text)
    text = re.sub(r'(\})\s*\n\s*(\{)', r"\1,\n\2", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def normalize_bbox(value: Any) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        out = [int(round(float(v))) for v in value]
    except Exception:
        return None
    out = [max(0, min(1000, v)) for v in out]
    if out[2] <= out[0] or out[3] <= out[1]:
        return None
    return out


def denorm_bbox(value: list[int], width: int, height: int) -> list[int]:
    return v11.clamp_bbox(
        [
            round(value[0] * width / 1000),
            round(value[1] * height / 1000),
            round(value[2] * width / 1000),
            round(value[3] * height / 1000),
        ],
        width,
        height,
    )


def resolve_norm1000_bbox(value: list[int] | None, width: int, height: int) -> dict[str, Any]:
    if not value:
        return {"bbox": None, "norm1000": None, "source": "missing"}
    bbox = denorm_bbox(value, width, height)
    if not v11.valid_bbox(bbox):
        return {"bbox": None, "norm1000": None, "source": "invalid_norm1000"}
    return {"bbox": bbox, "norm1000": value, "source": "norm1000"}


def empty_field() -> dict[str, Any]:
    return {"value": "", "abstain": True, "source": "unsupported", "confidence": 0.0, "reason": ""}


def strict_collection(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {**empty_field(), "reason": "collection_institution 未返回结构化字段。"}
    value = v11.normalize_space(entry.get("value") or "")
    if not value:
        return {**entry, "value": "", "abstain": True}
    if BAD_COLLECTION_RE.search(value):
        for pattern in ORG_PATTERNS:
            match = pattern.search(value)
            if match:
                kept = v11.normalize_space(match.group(1))
                return {
                    "value": kept,
                    "abstain": False,
                    "source": entry.get("source") or "caption",
                    "confidence": min(0.9, v11.safe_float(entry.get("confidence"), 0.8)),
                    "reason": "原返回包含入藏说明，已严格保留其中明确机构名。",
                }
        return {
            **empty_field(),
            "reason": "页面只提供 Gift/Bequest/Purchase/编号等入藏信息，没有明确收藏机构名。",
        }
    for pattern in ORG_PATTERNS:
        match = pattern.search(value)
        if match:
            return {**entry, "value": v11.normalize_space(match.group(1)), "abstain": False}
    return {**empty_field(), "reason": f"返回值“{value}”不是明确藏馆/收藏机构名。"}


def write_assets(out_dir: Path, records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    for index, (record, row) in enumerate(zip(records, rows), start=1):
        src = Path(record["page_image"])
        page_key = f"P{index:02d}_{v11.safe_stem(record['source_file'])[:40]}_p{int(record['page_num']):04d}"
        shutil.copy2(src, out_dir / "pages" / f"{page_key}.png")
        image = Image.open(src).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        for det_index, det in enumerate(row.get("detections") or [], start=1):
            target_bbox = det.get("target_bbox")
            caption_bbox = det.get("caption_bbox")
            if v11.valid_bbox(target_bbox):
                draw.rectangle(target_bbox, outline=(220, 0, 0), width=5)
                draw.text((target_bbox[0] + 4, max(0, target_bbox[1] - 18)), f"T{det_index}", fill=(220, 0, 0), font=font)
                Image.open(src).convert("RGB").crop(tuple(v11.clamp_bbox(target_bbox, image.width, image.height))).save(
                    out_dir / "crops" / f"{page_key}_T{det_index}.jpg", quality=92
                )
            if v11.valid_bbox(caption_bbox):
                draw.rectangle(caption_bbox, outline=(0, 185, 210), width=5)
                draw.text((caption_bbox[0] + 4, max(0, caption_bbox[1] - 18)), f"C{det_index}", fill=(0, 130, 150), font=font)
                Image.open(src).convert("RGB").crop(tuple(v11.clamp_bbox(caption_bbox, image.width, image.height))).save(
                    out_dir / "captions" / f"{page_key}_C{det_index}.jpg", quality=92
                )
        overlay = out_dir / "overlays" / f"{page_key}_overlay.jpg"
        image.save(overlay, quality=92)
        row["page_key"] = page_key
        row["page_copy"] = str(out_dir / "pages" / f"{page_key}.png")
        row["overlay_image"] = str(overlay)


def write_review_md(out_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v1.2 直接单页 VLM 10页样例",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 模型：`{args.model}`",
        f"- base_url：`{args.base_url}`",
        f"- disable_thinking：`{args.disable_thinking}`",
        f"- 并发：`{args.concurrency}`",
        "- 输入：已知 page image，未扫描 PDF",
        "- collection_institution 规则：只允许明确藏馆/收藏机构名；Gift/Bequest/Purchase/编号不算，只有这些则 abstain。",
        "",
    ]
    for index, row in enumerate(rows, start=1):
        overlay_rel = Path(row["overlay_image"]).relative_to(out_dir)
        lines.extend(
            [
                f"## P{index:02d} `{row.get('source_file')}` p{row.get('page_num')}",
                "",
                f"- ok：`{row.get('ok')}`；elapsed：`{row.get('elapsed_sec')}` sec；detections：{len(row.get('detections') or [])}",
                f"- summary：{html.escape(row.get('page_summary') or '')}",
                "",
                f"![overlay]({overlay_rel})",
                "",
            ]
        )
        if row.get("error"):
            lines.extend([f"- error：`{row.get('error')}`", ""])
        for det_index, det in enumerate(row.get("detections") or [], start=1):
            page_key = row["page_key"]
            target_rel = Path("crops") / f"{page_key}_T{det_index}.jpg"
            caption_rel = Path("captions") / f"{page_key}_C{det_index}.jpg"
            lines.extend([f"### P{index:02d}-T{det_index}", "", f"![target]({target_rel})", ""])
            if v11.valid_bbox(det.get("caption_bbox")):
                lines.extend([f"![caption]({caption_rel})", ""])
            else:
                lines.extend(["- caption_bbox：`missing`", ""])
            lines.extend(["#### Core4", "", "| field | value |", "|---|---|"])
            for field in ["caption_text", "depicted_work_title", "image_scope", "object_type", "object_domain", "caption_target_match", "confidence", "reason"]:
                value = "null" if det.get(field) is None else str(det.get(field, ""))
                lines.append(f"| `{field}` | {html.escape(value).replace('|', '/')} |")
            lines.extend(
                [
                    "",
                    "#### Metadata5",
                    "",
                    "| field | value | abstain | source | confidence | reason |",
                    "|---|---|---:|---|---:|---|",
                ]
            )
            metadata = det.get("metadata_fields") or {}
            for field in META_FIELDS:
                entry = metadata.get(field) or {}
                lines.append(
                    f"| `{field}` | {html.escape(str(entry.get('value', ''))).replace('|', '/')} | `{entry.get('abstain')}` | `{html.escape(str(entry.get('source', ''))).replace('|', '/')}` | `{entry.get('confidence', '')}` | {html.escape(str(entry.get('reason', '')).replace('|', '/'))} |"
                )
            corrected = "null" if det.get("corrected_caption_text") is None else html.escape(det.get("corrected_caption_text") or "")
            lines.extend(["", f"- corrected_caption_text：{corrected}", ""])
    lines.extend(["## 原始 JSON", "", f"- `{out_dir / 'json' / 'page_level_vlm_annotations.json'}`", ""])
    (out_dir / "v1.2直接单页远端VLM_10页样例.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
