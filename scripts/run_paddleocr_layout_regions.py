#!/usr/bin/env python3
"""Run PaddleOCR layout parsing and normalize page regions for agent candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


FIGURE_LABELS = {
    "image",
    "figure",
    "figure_title",
    "chart",
    "seal",
    "table",
}
CAPTION_LABELS = {
    "text",
    "title",
    "figure_title",
    "figure_caption",
    "caption",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="PDF page image path.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--engine", choices=["layout", "paddleocr_vl"], default="layout")
    parser.add_argument("--vl-rec-model-dir", default="/root/models/PaddleOCR-VL-1.6")
    parser.add_argument("--layout-model-dir", default="")
    parser.add_argument("--pipeline-version", default="v1.6")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--device", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_results = run_engine(args)
    regions = normalize_results(raw_results, args.top_k)
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image": args.image,
        "engine": args.engine,
        "vl_rec_model_dir": args.vl_rec_model_dir if args.engine == "paddleocr_vl" else None,
        "layout_model_dir": args.layout_model_dir or None,
        "regions": regions,
        "raw_summary": summarize_raw(raw_results),
    }
    (output_dir / "paddleocr_layout_regions.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    draw_regions(Path(args.image), regions, output_dir / "paddleocr_layout_regions_overlay.jpg")
    print(json.dumps(payload["raw_summary"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({"regions": regions[: min(5, len(regions))]}, ensure_ascii=False, indent=2), flush=True)
    return 0


def run_engine(args: argparse.Namespace) -> list[Any]:
    if args.engine == "layout":
        from paddleocr import LayoutDetection

        kwargs: dict[str, Any] = {"threshold": args.threshold}
        if args.layout_model_dir:
            kwargs["model_dir"] = args.layout_model_dir
        if args.device:
            kwargs["device"] = args.device
        model = LayoutDetection(**kwargs)
        return list(model.predict(args.image))

    from paddleocr import PaddleOCRVL

    kwargs = {
        "pipeline_version": args.pipeline_version,
        "vl_rec_model_dir": args.vl_rec_model_dir,
        "use_layout_detection": True,
        "layout_threshold": args.threshold,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": False,
    }
    if args.layout_model_dir:
        kwargs["layout_detection_model_dir"] = args.layout_model_dir
    if args.device:
        kwargs["device"] = args.device
    model = PaddleOCRVL(**kwargs)
    return list(model.predict(args.image, use_layout_detection=True, max_new_tokens=256))


def summarize_raw(raw_results: list[Any]) -> dict[str, Any]:
    return {
        "result_count": len(raw_results),
        "result_types": [type(item).__name__ for item in raw_results[:20]],
        "keys": [sorted(to_plain_dict(item).keys()) for item in raw_results[:3]],
    }


def normalize_results(raw_results: list[Any], top_k: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for raw in raw_results:
        boxes.extend(extract_boxes(raw))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for item in boxes:
        bbox = item.get("bbox")
        if not valid_bbox(bbox):
            continue
        label = str(item.get("label") or item.get("type") or "unknown")
        score = safe_float(item.get("score"), 0.0)
        key = tuple(int(round(float(v))) for v in bbox) + (label,)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "region_id": f"pocr_{len(unique):03d}",
                "bbox": [int(round(float(v))) for v in bbox],
                "type": map_label_to_type(label),
                "source": "paddleocr_layout",
                "score": round(score, 6),
                "label": label,
                "hint": f"PaddleOCR layout label={label}",
                "text": item.get("text") or item.get("content") or "",
            }
        )
    unique.sort(key=lambda item: (region_priority(item), safe_float(item.get("score"), 0.0)), reverse=True)
    return unique[: max(1, int(top_k))]


def extract_boxes(raw: Any) -> list[dict[str, Any]]:
    data = to_plain_dict(raw)
    boxes: list[dict[str, Any]] = []
    boxes.extend(extract_box_list(data))
    for key in ["layout_det_res", "layout_parsing_result", "parsing_res_list", "res", "result"]:
        value = data.get(key)
        if value is not None:
            boxes.extend(extract_box_list(to_plain(value)))
    return boxes


def extract_box_list(value: Any) -> list[dict[str, Any]]:
    value = to_plain(value)
    if isinstance(value, list):
        boxes: list[dict[str, Any]] = []
        for item in value:
            boxes.extend(extract_box_list(item))
        return boxes
    if not isinstance(value, dict):
        return []

    bbox = first_present(value, ["bbox", "box", "coordinate", "poly", "points"])
    label = first_present(value, ["label", "type", "category", "block_label", "class_name"])
    score = first_present(value, ["score", "confidence", "prob"])
    text = first_present(value, ["text", "content", "rec_text", "markdown"])
    current = []
    if bbox is not None:
        current.append({"bbox": normalize_bbox(bbox), "label": label, "score": score, "text": text})
    for child_key in ["boxes", "dt_polys", "layout", "blocks", "items", "children", "res", "result"]:
        if child_key in value:
            current.extend(extract_box_list(value[child_key]))
    return current


def normalize_bbox(value: Any) -> list[float] | None:
    value = to_plain(value)
    if isinstance(value, list) and len(value) == 4 and all(isinstance(x, (int, float)) for x in value):
        x1, y1, x2, y2 = [float(x) for x in value]
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
        xs: list[float] = []
        ys: list[float] = []
        for point in value:
            if len(point) >= 2 and isinstance(point[0], (int, float)) and isinstance(point[1], (int, float)):
                xs.append(float(point[0]))
                ys.append(float(point[1]))
        if xs and ys:
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def valid_bbox(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return False
    return x2 > x1 and y2 > y1


def map_label_to_type(label: str) -> str:
    label_l = label.lower()
    if label_l in FIGURE_LABELS or "image" in label_l or "figure" in label_l:
        return "figure_candidate"
    if label_l in CAPTION_LABELS or "caption" in label_l:
        return "caption_or_text_candidate"
    return "layout_candidate"


def region_priority(item: dict[str, Any]) -> int:
    typ = str(item.get("type") or "")
    if typ == "figure_candidate":
        return 3
    if typ == "caption_or_text_candidate":
        return 2
    return 1


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def to_plain_dict(value: Any) -> dict[str, Any]:
    plain = to_plain(value)
    return plain if isinstance(plain, dict) else {}


def to_plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    if hasattr(value, "json"):
        try:
            return json.loads(value.json())
        except Exception:
            pass
    if hasattr(value, "to_json"):
        try:
            return json.loads(value.to_json())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return to_plain(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return to_plain(vars(value))
        except Exception:
            pass
    return str(value)


def draw_regions(image_path: Path, regions: list[dict[str, Any]], output_path: Path) -> None:
    with Image.open(image_path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    colors = {
        "figure_candidate": "red",
        "caption_or_text_candidate": "cyan",
        "layout_candidate": "yellow",
    }
    for idx, region in enumerate(regions):
        bbox = region.get("bbox")
        if not valid_bbox(bbox):
            continue
        color = colors.get(str(region.get("type")), "yellow")
        x1, y1, x2, y2 = [int(v) for v in bbox]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 4, y1 + 4), f"{idx}:{region.get('label')}", fill=color, font=font)
    canvas.save(output_path)


if __name__ == "__main__":
    raise SystemExit(main())
