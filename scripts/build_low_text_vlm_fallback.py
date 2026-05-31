#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image


SYSTEM_PROMPT = """你是严谨的文献页面转写助手。你会看到一页 PDF 渲染图。

任务：
1. 忠实转写页面中可读的文字，不要翻译，不要改写，不要补充页面里没有的内容。
2. 尽量保留标题、段落、图注、页眉页脚的相对顺序。
3. 如果是竖排古籍、影印本、扫描页，也尽量转写；无法辨认的字用 □。
4. 如果页面主要是图片、空白页、目录装饰页或文字很少，请说明 page_type，并把能读的文字写出来。
5. 必须只输出 JSON 对象，不要输出 Markdown。

JSON schema：
{
  "is_readable": true,
  "page_type": "text|scan_text|image_plate|mixed|blank|unknown",
  "transcription": "忠实转写文本",
  "text_language": "zh|en|mixed|unknown",
  "confidence": 0.0,
  "notes": "一句话说明可读性、版式或问题"
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DashScope VLM OCR fallback on low-text PDF pages.")
    parser.add_argument("--evidence-index-dir", default="/root/datasets/evidence_grounded_vlm_agentrl/evidence_index_v0_3_20260530_2149")
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="", help="Use a fixed output directory instead of output-root/version/timestamp.")
    parser.add_argument("--version", default="low_text_vlm_fallback_v0_1")
    parser.add_argument("--model", default="qwen3.7-max-2026-05-20")
    parser.add_argument(
        "--fallback-models",
        default="qwen3.7-max-2026-05-17,qwen3.7-max,qwen3.6-plus-2026-04-02,qwen3.6-flash-2026-04-16",
    )
    parser.add_argument("--dotenv", default="/root/Workspace/VLM/EviTool-VL/.env")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dpi", type=int, default=170)
    parser.add_argument("--image-max-side", type=int, default=1600)
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--dashscope-image-format", choices=["auto", "image", "image_url"], default="auto")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true", help="With --resume, retry rows whose previous result had ok=false.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(args.dotenv)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(f"DASHSCOPE_API_KEY is not set. Check {args.dotenv}")

    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = Path(args.output_root)
    output_dir = Path(args.output_dir) if args.output_dir else output_root / f"{args.version}_{now}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rendered_pages").mkdir(exist_ok=True)

    low_text_path = Path(args.evidence_index_dir) / "low_text_pages.jsonl"
    rows = read_jsonl(low_text_path)
    selected = rows[args.offset :]
    if args.limit:
        selected = selected[: args.limit]

    stream_path = output_dir / "fallback_pages.jsonl"
    completed = load_completed(stream_path) if args.resume else {}
    if args.retry_errors:
        completed_for_skip = {key: value for key, value in completed.items() if value.get("ok")}
        results = list(completed_for_skip.values())
    else:
        completed_for_skip = completed
        results = list(completed.values())
    pending = [row for row in selected if page_key(row) not in completed_for_skip]

    client = DashScopeClient(args, api_key)
    if args.workers <= 1:
        for row in pending:
            result = process_one(row, output_dir, client, args)
            append_jsonl(stream_path, result)
            results.append(result)
            if args.sleep:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, row, output_dir, client, args): row for row in pending}
            for future in as_completed(futures):
                result = future.result()
                append_jsonl(stream_path, result)
                results.append(result)

    summary = build_summary(args, output_dir, rows, selected, results)
    write_json(output_dir / "manifest.json", summary)
    write_report(output_dir / "VLM低文本页Fallback报告.md", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


class DashScopeClient:
    def __init__(self, args: argparse.Namespace, api_key: str):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=args.request_timeout,
            max_retries=0,
        )
        models = [args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()]
        self.models = dedupe(models)
        self.args = args

    def infer(self, image_path: Path, page_row: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        last_error: Exception | None = None
        for model in self.models:
            for mode in image_modes_for_model(model, self.args.dashscope_image_format):
                try:
                    print(
                        json.dumps(
                            {
                                "event": "vlm_request",
                                "model": model,
                                "mode": mode,
                                "source_file": page_row.get("source_file"),
                                "page": page_row.get("page"),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    content = build_content(image_path, page_row, self.args, mode)
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": content}],
                        temperature=self.args.temperature,
                        max_tokens=self.args.max_tokens,
                        response_format={"type": "json_object"},
                    )
                    raw = response.choices[0].message.content or ""
                    parsed = parse_json_object(raw)
                    return parsed, model, mode
                except Exception as exc:
                    last_error = exc
                    continue
        raise RuntimeError(f"all models failed: {last_error!r}")


def process_one(row: dict[str, Any], output_dir: Path, client: DashScopeClient, args: argparse.Namespace) -> dict[str, Any]:
    try:
        image_path = render_page(row, output_dir / "rendered_pages", args.dpi, args.image_max_side)
        parsed, model, mode = client.infer(image_path, row)
        transcription = clean_text(parsed.get("transcription") or "")
        result = {
            "source_id": row.get("source_id"),
            "source_file": row.get("source_file"),
            "local_path": row.get("local_path"),
            "page": row.get("page"),
            "text_chars_before": row.get("text_chars"),
            "image_path": str(image_path),
            "provider": "dashscope_openai_compatible",
            "model": model,
            "image_mode": mode,
            "ok": True,
            "is_readable": bool(parsed.get("is_readable", bool(transcription))),
            "page_type": str(parsed.get("page_type") or "unknown"),
            "transcription": transcription,
            "text_language": str(parsed.get("text_language") or "unknown"),
            "confidence": clamp_float(parsed.get("confidence"), default=0.0),
            "notes": str(parsed.get("notes") or ""),
            "text_chars_after": len(transcription),
            "raw_response": parsed,
        }
    except Exception as exc:
        result = {
            "source_id": row.get("source_id"),
            "source_file": row.get("source_file"),
            "local_path": row.get("local_path"),
            "page": row.get("page"),
            "text_chars_before": row.get("text_chars"),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    print(json.dumps({"source_file": row.get("source_file"), "page": row.get("page"), "ok": result.get("ok"), "chars": result.get("text_chars_after", 0)}, ensure_ascii=False), flush=True)
    return result


def render_page(row: dict[str, Any], out_dir: Path, dpi: int, max_side: int) -> Path:
    pdf_path = Path(str(row.get("local_path") or ""))
    page_num = int(row.get("page") or 1)
    safe = safe_name(f"{Path(str(row.get('source_file') or pdf_path.name)).stem}_p{page_num:04d}")
    out = out_dir / f"{safe}.jpg"
    if out.exists():
        return out
    with fitz.open(pdf_path) as doc:
        pix = doc[page_num - 1].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
        tmp = out.with_suffix(".png")
        pix.save(tmp)
    image = Image.open(tmp).convert("RGB")
    image.thumbnail((max_side, max_side))
    image.save(out, quality=90)
    tmp.unlink(missing_ok=True)
    return out


def build_content(image_path: Path, row: dict[str, Any], args: argparse.Namespace, mode: str) -> Any:
    prompt = (
        SYSTEM_PROMPT
        + "\n页面元数据：\n"
        + json.dumps(
            {
                "source_file": row.get("source_file"),
                "source_id": row.get("source_id"),
                "page": row.get("page"),
                "text_chars_before": row.get("text_chars"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    data_url = image_data_url(image_path)
    if mode == "image":
        return [{"type": "image", "image": data_url}, {"type": "text", "text": prompt}]
    return [{"type": "image_url", "image_url": {"url": data_url}}, {"type": "text", "text": prompt}]


def image_modes_for_model(model: str, requested: str = "auto") -> list[str]:
    if requested != "auto":
        return [requested]
    lower = model.lower()
    if "qwen3.7-max" in lower or "qwen-max" in lower:
        return ["image", "image_url"]
    return ["image_url", "image"]


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in response: {text[:300]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return parsed


def build_summary(args: argparse.Namespace, output_dir: Path, all_rows: list[dict[str, Any]], selected: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    ok_rows = [row for row in results if row.get("ok")]
    readable = [row for row in ok_rows if row.get("is_readable")]
    with_text = [row for row in ok_rows if int(row.get("text_chars_after") or 0) >= 40]
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "builder": "scripts/build_low_text_vlm_fallback.py",
        "output_dir": str(output_dir),
        "evidence_index_dir": args.evidence_index_dir,
        "model": args.model,
        "fallback_models": [item.strip() for item in args.fallback_models.split(",") if item.strip()],
        "low_text_pages_total": len(all_rows),
        "selected_pages": len(selected),
        "processed_pages": len(results),
        "ok_pages": len(ok_rows),
        "readable_pages": len(readable),
        "pages_with_40plus_chars": len(with_text),
        "avg_chars_after": round(sum(int(row.get("text_chars_after") or 0) for row in ok_rows) / max(1, len(ok_rows)), 2),
        "page_type_counts": dict(Counter(row.get("page_type", "error") for row in results)),
        "source_file_counts": dict(Counter(row.get("source_file", "") for row in results)),
        "outputs": {
            "fallback_pages": str(output_dir / "fallback_pages.jsonl"),
            "rendered_pages": str(output_dir / "rendered_pages"),
            "report": str(output_dir / "VLM低文本页Fallback报告.md"),
        },
        "limitations": [
            "This is a fallback OCR/VLM transcription run, not a full rebuilt evidence index.",
            "Transcriptions are model-generated and should be treated as silver text until spot-checked.",
            "The smoke run intentionally limits page count before spending quota on all low-text pages.",
        ],
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Low-Text Pages VLM Fallback 报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- evidence index：`{summary['evidence_index_dir']}`",
        f"- 主模型：`{summary['model']}`",
        "",
        "## 规模",
        "",
        f"- low_text_pages_total：{summary['low_text_pages_total']}",
        f"- selected_pages：{summary['selected_pages']}",
        f"- processed_pages：{summary['processed_pages']}",
        f"- ok_pages：{summary['ok_pages']}",
        f"- readable_pages：{summary['readable_pages']}",
        f"- pages_with_40plus_chars：{summary['pages_with_40plus_chars']}",
        f"- avg_chars_after：{summary['avg_chars_after']}",
        f"- page_type_counts：`{summary['page_type_counts']}`",
        "",
        "## 输出",
        "",
    ]
    for key, value in summary["outputs"].items():
        lines.append(f"- {key}：`{value}`")
    lines.extend(["", "## 限制", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def page_key(row: dict[str, Any]) -> str:
    return f"{row.get('source_id')}::{row.get('page')}"


def load_completed(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        rows[page_key(row)] = row
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_dotenv(path: str) -> None:
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def clean_text(value: str) -> str:
    value = str(value).replace("\u0000", " ")
    value = re.sub(r"[ \t\r\f\v]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def clamp_float(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def dedupe(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value)
    return value[:140] or "page"


if __name__ == "__main__":
    raise SystemExit(main())
