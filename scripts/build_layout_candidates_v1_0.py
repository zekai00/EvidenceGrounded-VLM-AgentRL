#!/usr/bin/env python3
"""Build page-level layout/figure candidate cache for v1.0 AgentBench.

This stage is intentionally model-light: it combines PDF image blocks,
PDF text-layer captions, and OpenCV visual-region proposals. OCR/VLM
adjudication can consume this cache later without rescanning all PDFs.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_agentbench_v0_9_fixedsplit_train_multitarget import (  # noqa: E402
    bbox_iou,
    caption_like,
    caption_score,
    clean_text,
    load_source_meta,
    normalize_spaces,
    safe_name,
    scale_bbox_to_rendered,
)
from evidence_agent_env.data import write_jsonl  # noqa: E402


DEFAULT_RAW_PDFS = Path("/root/datasets/chinese_landscape_authority_corpus")
DEFAULT_EVIDENCE_INDEX = Path(
    "/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_1_low_text_vlm_full_20260531_0140"
)
DEFAULT_OUTPUT_ROOT = Path("/root/datasets/evidence_grounded_vlm_agentrl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-pdfs-dir", default=str(DEFAULT_RAW_PDFS))
    parser.add_argument("--evidence-index-dir", default=str(DEFAULT_EVIDENCE_INDEX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--page-dpi", type=int, default=150)
    parser.add_argument("--min-area-ratio", type=float, default=0.003)
    parser.add_argument("--max-area-ratio", type=float, default=0.86)
    parser.add_argument("--min-width-ratio", type=float, default=0.045)
    parser.add_argument("--min-height-ratio", type=float, default=0.035)
    parser.add_argument("--max-figure-candidates-per-page", type=int, default=16)
    parser.add_argument("--max-text-blocks-per-page", type=int, default=40)
    parser.add_argument("--overlay-limit", type=int, default=160)
    parser.add_argument("--smoke-max-pdfs", type=int, default=0)
    parser.add_argument("--smoke-max-pages", type=int, default=0)
    parser.add_argument("--skip-failed-partial-downloads", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(Path(args.output_root))
    if output_dir.exists():
        raise FileExistsError(f"output_dir already exists: {output_dir}")
    for child in ["overlays", "sample_pages"]:
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    source_meta = load_source_meta(Path(args.evidence_index_dir))
    pdfs = collect_pdfs(Path(args.raw_pdfs_dir), args)
    page_rows: list[dict[str, Any]] = []
    figure_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    pdfs_seen = 0
    pages_seen = 0
    pages_with_candidates = 0

    for pdf_index, pdf_path in enumerate(pdfs, 1):
        try:
            with fitz.open(pdf_path) as doc:
                pdfs_seen += 1
                if pdf_index % 10 == 0:
                    print(
                        f"[layout] pdf {pdf_index}/{len(pdfs)} pages={pages_seen} candidate_pages={pages_with_candidates} current={pdf_path.name}",
                        file=sys.stderr,
                        flush=True,
                    )
                if len(doc) <= 0:
                    errors.append({"source_path": str(pdf_path), "error": "empty_or_invalid_pdf"})
                    continue
                for page_index, page in enumerate(doc):
                    if args.smoke_max_pages and pages_seen >= args.smoke_max_pages:
                        break
                    pages_seen += 1
                    try:
                        row = build_page_row(pdf_path, page, page_index + 1, len(doc), source_meta.get(pdf_path.name, {}), args)
                    except Exception as exc:
                        errors.append(
                            {
                                "source_path": str(pdf_path),
                                "page": page_index + 1,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        continue
                    if not row["figure_candidates"]:
                        continue
                    pages_with_candidates += 1
                    page_rows.append(row)
                    for cand in row["figure_candidates"]:
                        item = {key: value for key, value in row.items() if key != "figure_candidates"}
                        item.update(cand)
                        figure_rows.append(item)
                    if len(page_rows) <= args.overlay_limit:
                        draw_overlay(row, output_dir / "overlays" / f"{safe_name(row['source_stem'])}_p{row['page']:04d}.jpg")
                if args.smoke_max_pages and pages_seen >= args.smoke_max_pages:
                    break
        except Exception as exc:
            errors.append({"source_path": str(pdf_path), "error": f"{type(exc).__name__}: {exc}"})

    write_jsonl(output_dir / "page_candidates.jsonl", page_rows)
    write_jsonl(output_dir / "figure_candidates.jsonl", figure_rows)
    summary = summarize(args, output_dir, pdfs, page_rows, figure_rows, errors, pdfs_seen, pages_seen, pages_with_candidates)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(build_manifest(args, output_dir, summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "候选缓存构建报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def default_output_dir(root: Path) -> Path:
    return root / f"layout_candidates_v1_0_{datetime.now().strftime('%Y%m%d_%H%M')}"


def collect_pdfs(root: Path, args: argparse.Namespace) -> list[Path]:
    pdfs = sorted(root.rglob("*.pdf"))
    if args.skip_failed_partial_downloads:
        pdfs = [path for path in pdfs if "failed_partial_downloads" not in path.parts]
    if args.smoke_max_pdfs:
        pdfs = pdfs[: args.smoke_max_pdfs]
    return pdfs


def build_page_row(
    pdf_path: Path,
    page: fitz.Page,
    page_num: int,
    page_count: int,
    source_meta: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rect = page.rect
    width = int(round(rect.width * args.page_dpi / 72.0))
    height = int(round(rect.height * args.page_dpi / 72.0))
    text_blocks = collect_text_blocks(page, rect, args.page_dpi, args.max_text_blocks_per_page)
    page_image = render_page_to_pil(page, args.page_dpi)
    pdf_candidates = pdf_image_candidates(pdf_path, page, rect, text_blocks, args)
    visual_candidates = opencv_visual_candidates(page_image, text_blocks, args)
    figure_candidates = merge_candidates(pdf_candidates + visual_candidates, width, height, text_blocks, args)
    page_id = f"{safe_name(pdf_path.stem)}_p{page_num:04d}"
    for idx, cand in enumerate(figure_candidates):
        cand["candidate_id"] = f"{page_id}_c{idx:03d}"
        cand["rank"] = idx + 1
    return {
        "page_id": page_id,
        "source_file": pdf_path.name,
        "source_stem": pdf_path.stem,
        "source_path": str(pdf_path),
        "page": page_num,
        "page_count": page_count,
        "page_width": width,
        "page_height": height,
        "source_meta": source_meta,
        "text_blocks": text_blocks,
        "figure_candidates": figure_candidates,
    }


def render_page_to_pil(page: fitz.Page, dpi: int) -> Image.Image:
    pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def collect_text_blocks(page: fitz.Page, rect: fitz.Rect, dpi: int, limit: int) -> list[dict[str, Any]]:
    scale = dpi / 72.0
    blocks: list[dict[str, Any]] = []
    try:
        raw_blocks = page.get_text("blocks")
    except Exception:
        raw_blocks = []
    for block_index, block in enumerate(raw_blocks):
        if len(block) < 5:
            continue
        text = clean_text(str(block[4]))
        if not text:
            continue
        bbox = [
            max(0, int(round(float(block[0]) * scale))),
            max(0, int(round(float(block[1]) * scale))),
            min(int(round(rect.width * scale)), int(round(float(block[2]) * scale))),
            min(int(round(rect.height * scale)), int(round(float(block[3]) * scale))),
        ]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        blocks.append(
            {
                "block_id": f"txt_{len(blocks):03d}",
                "source": "pdf_text_layer",
                "bbox": bbox,
                "text": normalize_spaces(text),
                "char_count": len(normalize_spaces(text)),
                "looks_like_caption": caption_like(normalize_spaces(text)[:120]),
            }
        )
    blocks.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return blocks[: max(1, int(limit))]


def pdf_image_candidates(
    pdf_path: Path,
    page: fitz.Page,
    rect: fitz.Rect,
    text_blocks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for block_index, bbox_pt in enumerate(iter_image_rects(page)):
        bbox = scale_bbox_to_rendered(tuple(float(v) for v in bbox_pt), rect, args.page_dpi)
        if not candidate_size_ok(bbox, int(rect.width * args.page_dpi / 72.0), int(rect.height * args.page_dpi / 72.0), args):
            continue
        caption = best_caption_for_bbox(bbox, text_blocks)
        candidates.append(
            {
                "bbox": bbox,
                "bbox_pt": [float(v) for v in bbox_pt],
                "source": "pdf_image_block",
                "type": "figure_candidate",
                "raw_score": 0.0,
                "caption_bbox": caption.get("bbox"),
                "caption_text": caption.get("text", ""),
                "caption_score": float(caption.get("score", -999.0)),
                "linked_text_block_id": caption.get("block_id"),
                "area_ratio": area_ratio(bbox, int(rect.width * args.page_dpi / 72.0), int(rect.height * args.page_dpi / 72.0)),
                "block_index": block_index,
            }
        )
    return candidates


def iter_image_rects(page: fitz.Page) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    seen_xrefs: set[int] = set()
    try:
        images = page.get_images(full=True)
    except Exception:
        images = []
    for image in images:
        try:
            xref = int(image[0])
        except Exception:
            continue
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)
        try:
            placements = page.get_image_rects(xref)
        except Exception:
            placements = []
        for item in placements:
            rects.append((float(item.x0), float(item.y0), float(item.x1), float(item.y1)))
    if rects:
        return rects
    try:
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") == 1:
                rects.append(tuple(float(v) for v in block.get("bbox", [0, 0, 0, 0])))
    except Exception:
        pass
    return rects


def opencv_visual_candidates(image: Image.Image, text_blocks: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    width, height = image.size
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    # Non-white/ink mask. This catches scanned-page figures when PDF image
    # blocks are missing, but we later downscore text-heavy boxes.
    _, mask = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []
    for contour_index, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        bbox = [int(x), int(y), int(x + w), int(y + h)]
        if not candidate_size_ok(bbox, width, height, args):
            continue
        text_overlap = text_overlap_ratio(bbox, text_blocks)
        if text_overlap > 0.55:
            continue
        caption = best_caption_for_bbox(bbox, text_blocks)
        candidates.append(
            {
                "bbox": bbox,
                "bbox_pt": None,
                "source": "opencv_visual_region",
                "type": "figure_candidate",
                "raw_score": round(1.0 - text_overlap, 6),
                "caption_bbox": caption.get("bbox"),
                "caption_text": caption.get("text", ""),
                "caption_score": float(caption.get("score", -999.0)),
                "linked_text_block_id": caption.get("block_id"),
                "area_ratio": area_ratio(bbox, width, height),
                "text_overlap_ratio": round(text_overlap, 6),
                "contour_index": contour_index,
            }
        )
    return candidates


def merge_candidates(
    candidates: list[dict[str, Any]],
    width: int,
    height: int,
    text_blocks: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for item in candidates:
        bbox = normalize_bbox(item.get("bbox"), width, height)
        if not bbox or not candidate_size_ok(bbox, width, height, args):
            continue
        item = dict(item)
        item["bbox"] = bbox
        item["area_ratio"] = round(area_ratio(bbox, width, height), 6)
        item["target_score"] = round(target_score(item), 6)
        scored.append(item)
    scored.sort(key=lambda item: item["target_score"], reverse=True)
    unique: list[dict[str, Any]] = []
    for item in scored:
        if any(bbox_iou(item["bbox"], other["bbox"]) >= 0.88 for other in unique):
            continue
        unique.append(item)
        if len(unique) >= args.max_figure_candidates_per_page:
            break
    return unique


def target_score(item: dict[str, Any]) -> float:
    source_bonus = 4.0 if item.get("source") == "pdf_image_block" else 1.2
    cap = float(item.get("caption_score") or -999.0)
    cap_bonus = max(-2.0, min(8.0, cap))
    area = float(item.get("area_ratio") or 0.0)
    size_bonus = 2.0 * min(1.0, area / 0.18)
    full_page_penalty = 3.0 if area > 0.68 else 0.0
    text_penalty = 2.0 * float(item.get("text_overlap_ratio") or 0.0)
    return source_bonus + cap_bonus + size_bonus - full_page_penalty - text_penalty


def best_caption_for_bbox(image_bbox: list[int], text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for block in text_blocks:
        score = caption_score(image_bbox, block["bbox"], block["text"])
        if best is None or score > best["score"]:
            best = {"bbox": block["bbox"], "text": block["text"], "score": score, "block_id": block.get("block_id")}
    if best is None or best["score"] < -8.0:
        return {"bbox": None, "text": "", "score": -999.0, "block_id": None}
    return best


def candidate_size_ok(bbox: list[int], width: int, height: int, args: argparse.Namespace) -> bool:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return False
    bw = x2 - x1
    bh = y2 - y1
    area = (bw * bh) / max(1, width * height)
    return (
        args.min_area_ratio <= area <= args.max_area_ratio
        and bw / max(1, width) >= args.min_width_ratio
        and bh / max(1, height) >= args.min_height_ratio
    )


def normalize_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except Exception:
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area_ratio(bbox: list[int], width: int, height: int) -> float:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1]) / max(1, width * height)


def text_overlap_ratio(bbox: list[int], text_blocks: list[dict[str, Any]]) -> float:
    area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    overlap = 0
    for block in text_blocks:
        tb = block.get("bbox") or []
        if len(tb) != 4:
            continue
        x1 = max(bbox[0], int(tb[0]))
        y1 = max(bbox[1], int(tb[1]))
        x2 = min(bbox[2], int(tb[2]))
        y2 = min(bbox[3], int(tb[3]))
        if x2 > x1 and y2 > y1:
            overlap += (x2 - x1) * (y2 - y1)
    return min(1.0, overlap / area)


def draw_overlay(row: dict[str, Any], out: Path) -> None:
    with fitz.open(row["source_path"]) as doc:
        image = render_page_to_pil(doc[int(row["page"]) - 1], int(round(row["page_width"] * 72 / doc[int(row["page"]) - 1].rect.width))).convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = ["red", "orange", "magenta", "lime", "cyan", "yellow"]
    for idx, cand in enumerate(row.get("figure_candidates") or []):
        color = colors[idx % len(colors)]
        draw.rectangle(cand["bbox"], outline=color, width=4 if idx == 0 else 2)
        draw.text((cand["bbox"][0], max(0, cand["bbox"][1] - 16)), f"{idx}:{cand['source']}:{cand['target_score']:.1f}", fill=color)
    for block in row.get("text_blocks") or []:
        if block.get("looks_like_caption"):
            draw.rectangle(block["bbox"], outline="blue", width=2)
    image.save(out, quality=90)


def summarize(
    args: argparse.Namespace,
    output_dir: Path,
    pdfs: list[Path],
    page_rows: list[dict[str, Any]],
    figure_rows: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    pdfs_seen: int,
    pages_seen: int,
    pages_with_candidates: int,
) -> dict[str, Any]:
    source_counts = Counter(row["source_file"] for row in page_rows)
    candidate_sources = Counter(row.get("source") for row in figure_rows)
    figure_counts = [len(row.get("figure_candidates") or []) for row in page_rows]
    caption_linked = [row for row in figure_rows if row.get("caption_text")]
    return {
        "created_at": now(),
        "output_dir": str(output_dir),
        "pdfs_total": len(pdfs),
        "pdfs_seen": pdfs_seen,
        "pages_seen": pages_seen,
        "pages_with_candidates": pages_with_candidates,
        "page_candidate_rows": len(page_rows),
        "figure_candidate_rows": len(figure_rows),
        "unique_sources": len(source_counts),
        "candidate_sources": dict(candidate_sources),
        "mean_candidates_per_candidate_page": sum(figure_counts) / max(1, len(figure_counts)),
        "max_candidates_per_page": max(figure_counts) if figure_counts else 0,
        "caption_link_rate": len(caption_linked) / max(1, len(figure_rows)),
        "top_sources": dict(source_counts.most_common(20)),
        "error_count": len(errors),
        "errors": errors[:50],
        "args": vars(args),
        "ocr_backend_status": {
            "paddleocr": "installed_but_current_cpu_runtime_failed_in_smoke",
            "easyocr": "installed_but_model_download_timed_out_in_smoke",
            "current_cache": "pdf_text_layer_plus_opencv_visual_regions_no_external_ocr",
        },
    }


def build_manifest(args: argparse.Namespace, output_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "created_at": summary["created_at"],
        "dataset_version": "layout_candidates_v1_0_pdftext_opencv",
        "builder": "scripts/build_layout_candidates_v1_0.py",
        "output_dir": str(output_dir),
        "raw_pdfs_dir": args.raw_pdfs_dir,
        "evidence_index_dir": args.evidence_index_dir,
        "summary": summary,
        "files": {
            "page_candidates": str(output_dir / "page_candidates.jsonl"),
            "figure_candidates": str(output_dir / "figure_candidates.jsonl"),
            "summary": str(output_dir / "summary.json"),
            "report": str(output_dir / "候选缓存构建报告.md"),
        },
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# v1.0 Layout Candidate Cache 构建报告",
        "",
        f"生成时间：{summary['created_at']} CST",
        "",
        "## 输出位置",
        "",
        "```text",
        summary["output_dir"],
        "```",
        "",
        "## 规模",
        "",
        f"- pdfs_seen：{summary['pdfs_seen']}",
        f"- pages_seen：{summary['pages_seen']}",
        f"- pages_with_candidates：{summary['pages_with_candidates']}",
        f"- page_candidate_rows：{summary['page_candidate_rows']}",
        f"- figure_candidate_rows：{summary['figure_candidate_rows']}",
        f"- unique_sources：{summary['unique_sources']}",
        f"- candidate_sources：`{json.dumps(summary['candidate_sources'], ensure_ascii=False)}`",
        f"- mean_candidates_per_candidate_page：{summary['mean_candidates_per_candidate_page']:.2f}",
        f"- max_candidates_per_page：{summary['max_candidates_per_page']}",
        f"- caption_link_rate：{summary['caption_link_rate']:.4f}",
        "",
        "## OCR 后端状态",
        "",
        "- 当前缓存实际使用：PDF text layer + OpenCV visual regions。",
        "- PaddleOCR 已安装，但当前 CPU runtime smoke 触发 PaddlePaddle oneDNN/PIR 兼容错误。",
        "- EasyOCR 已安装，但首次模型下载在当前网络下超时。",
        "- 因此本轮先完成可复现候选扩容；后续可在修好 OCR 环境后只补跑低文本页。",
        "",
        "## 说明",
        "",
        "- `pdf_image_block` 来自 PDF 内置图片对象。",
        "- `opencv_visual_region` 来自页面渲染图的非白色视觉连通区域，经过文本重叠过滤。",
        "- 该 cache 不是最终训练集；后续需要再经过 split、任务生成、oracle replay 和 VLM 抽检/裁决。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


if __name__ == "__main__":
    raise SystemExit(main())
