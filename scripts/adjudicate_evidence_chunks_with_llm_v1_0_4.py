#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from source_registry_rules_v1_0_4 import ANCHOR_FIELDS, CONTEXT_FIELDS, OBJECT_METADATA_FIELDS, VISUAL_FIELDS


ALL_CLAIM_FIELDS = sorted(set(OBJECT_METADATA_FIELDS + VISUAL_FIELDS + ANCHOR_FIELDS + CONTEXT_FIELDS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="合并 v1.0.4 evidence chunk 的离线/LLM/VLM/人工裁决标签。"
    )
    parser.add_argument("--samples", required=True, help="sample_evidence_chunks 脚本输出的 audit_samples.jsonl")
    parser.add_argument(
        "--provider",
        default="offline",
        choices=["offline", "dashscope", "llm_jsonl", "vlm_jsonl", "manual_jsonl"],
        help="offline 使用规则弱标签；dashscope 直接调用文本 LLM；*_jsonl 读取外部裁决结果并按 audit_id/evidence_id 合并。",
    )
    parser.add_argument("--labels", default="", help="当 provider 为 *_jsonl 时，外部裁决 JSONL 路径。")
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--version", default="evidence_chunk_adjudication_v1_0_4")
    parser.add_argument("--min-confidence-for-auto", type=float, default=0.75)
    parser.add_argument("--dotenv", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--model", default="qwen3.7-plus")
    parser.add_argument(
        "--fallback-models",
        default=(
            "qwen3.6-plus,glm-5.1,kimi-k2.6,qwen3.5-plus-2026-04-20,"
            "qwen3.6-27b,deepseek-v4-flash,deepseek-v4-pro,qwen3.7-max,"
            "qwen3.7-max-preview,qwen3.7-plus-2026-05-26,qwen3.7-max-2026-06-08"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.provider in {"llm_jsonl", "vlm_jsonl", "manual_jsonl"} and not args.labels:
        raise SystemExit(f"--provider {args.provider} 需要同时提供 --labels")

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.version}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = list(iter_jsonl(Path(args.samples)))
    if args.limit:
        samples = samples[: args.limit]
    external_labels: dict[str, dict[str, Any]] = {}
    direct_model_counts: Counter[str] = Counter()
    if args.provider == "dashscope":
        load_dotenv(Path(args.dotenv))
        client = DashScopeLLMClient(args)
        rows = []
        stream_path = output_dir / "dashscope_adjudicated_stream.jsonl"
        for index, sample in enumerate(samples):
            row = adjudicate_sample_with_dashscope(sample, client, args.min_confidence_for_auto)
            rows.append(row)
            direct_model_counts.update([str((row.get("adjudication") or {}).get("model") or "unknown")])
            append_jsonl(stream_path, row)
            print(
                json.dumps(
                    {
                        "index": index,
                        "audit_id": row.get("audit_id"),
                        "status": row.get("adjudication_status"),
                        "role": (row.get("adjudication") or {}).get("evidence_role"),
                        "model": (row.get("adjudication") or {}).get("model"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if args.sleep:
                time.sleep(args.sleep)
    else:
        external_labels = load_external_labels(Path(args.labels)) if args.labels else {}
        rows = [
            adjudicate_sample(sample, args.provider, external_labels, args.min_confidence_for_auto)
            for sample in samples
        ]
    review_queue = [row for row in rows if row.get("adjudication_status") != "accepted_auto"]

    write_jsonl(output_dir / "adjudicated_samples.jsonl", rows)
    write_jsonl(output_dir / "llm_or_human_review_queue.jsonl", review_queue)
    manifest = build_manifest(args, output_dir, rows, review_queue, external_labels, direct_model_counts)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "审计报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def adjudicate_sample(
    sample: dict[str, Any],
    provider: str,
    external_labels: dict[str, dict[str, Any]],
    min_confidence_for_auto: float,
) -> dict[str, Any]:
    external = {}
    if provider != "offline":
        external = external_labels.get(str(sample.get("audit_id") or "")) or external_labels.get(
            str(sample.get("evidence_id") or "")
        ) or {}

    if external:
        adjudication = normalize_external_label(external, provider)
        adjudication = apply_source_field_boundary(adjudication, sample)
        source = "external_label"
    else:
        adjudication = normalize_offline_label(sample.get("offline_label") or {})
        adjudication = apply_source_field_boundary(adjudication, sample)
        source = "offline_rules" if provider == "offline" else "offline_rules_fallback"

    confidence = float(adjudication.get("confidence") or 0.0)
    needs_review = bool(adjudication.get("needs_review"))
    status = "accepted_auto" if confidence >= min_confidence_for_auto and not needs_review else "needs_llm_or_human_review"

    row = dict(sample)
    row.update(
        {
            "adjudication": {
                **adjudication,
                "provider": provider,
                "label_source": source,
                "adjudicated_at": now(),
            },
            "adjudication_status": status,
            "offline_vs_adjudicated": compare_with_offline(sample.get("offline_label") or {}, adjudication),
        }
    )
    return row


def adjudicate_sample_with_dashscope(
    sample: dict[str, Any],
    client: "DashScopeLLMClient",
    min_confidence_for_auto: float,
) -> dict[str, Any]:
    try:
        label, model, raw_response = client.adjudicate(sample)
        adjudication = normalize_external_label(label, "dashscope")
        adjudication = apply_source_field_boundary(adjudication, sample)
        source = "dashscope_direct"
    except Exception as exc:
        adjudication = normalize_offline_label(sample.get("offline_label") or {})
        adjudication = apply_source_field_boundary(adjudication, sample)
        model = "dashscope_failed"
        raw_response = f"{type(exc).__name__}: {exc}"
        source = "offline_rules_after_dashscope_failure"
    confidence = float(adjudication.get("confidence") or 0.0)
    needs_review = bool(adjudication.get("needs_review"))
    status = "accepted_auto" if confidence >= min_confidence_for_auto and not needs_review else "needs_llm_or_human_review"
    row = dict(sample)
    row.update(
        {
            "adjudication": {
                **adjudication,
                "provider": "dashscope",
                "label_source": source,
                "model": model,
                "raw_response": raw_response,
                "adjudicated_at": now(),
            },
            "adjudication_status": status,
            "offline_vs_adjudicated": compare_with_offline(sample.get("offline_label") or {}, adjudication),
        }
    )
    return row


def normalize_offline_label(label: dict[str, Any]) -> dict[str, Any]:
    allowed = normalize_fields(label.get("claim_allowed_fields_pred") or [])
    disallowed = normalize_fields(label.get("claim_disallowed_fields_pred") or missing_fields(allowed))
    return {
        "evidence_role": str(label.get("evidence_role_pred") or "unknown"),
        "claim_allowed_fields": allowed,
        "claim_disallowed_fields": disallowed,
        "confidence": float(label.get("confidence") or 0.0),
        "needs_review": bool(label.get("needs_llm_review")),
        "rationale": str(label.get("rationale") or ""),
    }


def normalize_external_label(row: dict[str, Any], provider: str) -> dict[str, Any]:
    label = row.get("adjudication") if isinstance(row.get("adjudication"), dict) else row
    allowed = normalize_fields(label.get("claim_allowed_fields") or label.get("allowed_fields") or [])
    disallowed = normalize_fields(label.get("claim_disallowed_fields") or label.get("disallowed_fields") or [])
    if not disallowed:
        disallowed = missing_fields(allowed)
    role = str(label.get("evidence_role") or label.get("role") or "unknown")
    confidence = float(label.get("confidence") or 0.0)
    needs_review_raw = label.get("needs_review")
    needs_review = coerce_bool(needs_review_raw, confidence < 0.75)
    return {
        "evidence_role": role,
        "claim_allowed_fields": allowed,
        "claim_disallowed_fields": disallowed,
        "confidence": confidence,
        "needs_review": needs_review,
        "rationale": str(label.get("rationale") or label.get("reason") or f"{provider} 外部标签。"),
    }


def apply_source_field_boundary(adjudication: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    source_allowed = set(normalize_fields(sample.get("claim_allowed_fields_source") or []))
    if not source_allowed:
        return adjudication
    allowed = set(normalize_fields(adjudication.get("claim_allowed_fields") or []))
    bounded_allowed = allowed & source_allowed
    if bounded_allowed == allowed:
        return adjudication
    updated = dict(adjudication)
    updated["claim_allowed_fields"] = sorted(bounded_allowed)
    updated["claim_disallowed_fields"] = missing_fields(updated["claim_allowed_fields"])
    removed = sorted(allowed - bounded_allowed)
    rationale = str(updated.get("rationale") or "")
    updated["rationale"] = f"{rationale} 已按 source registry 移除越界字段：{removed}。"
    return updated


def coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "1", "y"}:
            return True
        if text in {"false", "no", "0", "n"}:
            return False
    return bool(value)


def compare_with_offline(offline_label: dict[str, Any], adjudication: dict[str, Any]) -> dict[str, Any]:
    offline_role = str(offline_label.get("evidence_role_pred") or "")
    adjudicated_role = str(adjudication.get("evidence_role") or "")
    offline_allowed = set(normalize_fields(offline_label.get("claim_allowed_fields_pred") or []))
    adjudicated_allowed = set(normalize_fields(adjudication.get("claim_allowed_fields") or []))
    return {
        "role_changed": bool(offline_role and adjudicated_role and offline_role != adjudicated_role),
        "allowed_fields_added": sorted(adjudicated_allowed - offline_allowed),
        "allowed_fields_removed": sorted(offline_allowed - adjudicated_allowed),
    }


def load_external_labels(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        keys = [row.get("audit_id"), row.get("evidence_id")]
        adjudication = row.get("adjudication") if isinstance(row.get("adjudication"), dict) else row
        for key in keys:
            if key:
                out[str(key)] = adjudication
    return out


class DashScopeLLMClient:
    def __init__(self, args: argparse.Namespace):
        from openai import OpenAI

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise RuntimeError(f"DASHSCOPE_API_KEY is not set. Check {args.dotenv}")
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            timeout=args.request_timeout,
        )
        self.model_order = dedupe_keep_order(
            [args.model] + [item.strip() for item in args.fallback_models.split(",") if item.strip()]
        )
        self.args = args

    def adjudicate(self, sample: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
        last_error: Exception | None = None
        for model in self.model_order:
            try:
                messages = build_dashscope_messages(sample)
                kwargs = {
                    "model": model,
                    "messages": messages,
                    "temperature": self.args.temperature,
                    "max_tokens": self.args.max_tokens,
                    "response_format": {"type": "json_object"},
                }
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""
                parsed = parse_json_object(content)
                return parsed, model, content
            except Exception as exc:
                last_error = exc
                continue
        raise RuntimeError(f"all DashScope LLM models failed: {last_error!r}")


def build_dashscope_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    offline = sample.get("offline_label") or {}
    system = (
        "你是 evidence chunk 审计器。任务是判断一段文本证据能安全支持哪些 claim 字段，"
        "不是回答艺术史知识，也不是扩写内容。必须保守：只有文本中明确出现或来源角色明确允许时才开放字段。"
    )
    user = f"""请只输出 JSON object，字段如下：
{{
  "evidence_role": "object_metadata|caption_or_plate|style_analysis|historical_context|theory_primary_text|teaching_overview|low_value_background|toc|bibliography|front_matter|back_matter|ocr_noise",
  "claim_allowed_fields": ["从候选字段中选择"],
  "confidence": 0.0,
  "needs_review": true,
  "rationale": "一句话中文理由"
}}

候选 claim 字段：
{ALL_CLAIM_FIELDS}

判定规则：
- bibliography、目录、索引、版权页、封面页、OCR 乱码、重复页眉页脚：claim_allowed_fields 必须为空。
- object_metadata 只用于明确给出作品名、作者、朝代/年代、馆藏、材质尺寸等元数据的文本。
- caption_or_plate 只用于明确图版、图号、caption、plate、figure anchor 或图像局部说明的文本。
- style_analysis/historical_context 只能支持风格、构图、技法、历史背景等背景型字段，不应支持 artist、collection、medium_dimensions 这类强元数据。
- theory_primary_text 只用于古代画论/原典，可支持 theory_concept、style_analysis、composition、technique。
- legacy/低质来源或含混摘要应标为 low_value_background 或 needs_review=true。

sample_bucket: {sample.get("sample_bucket")}
evidence_id: {sample.get("evidence_id")}
source_file: {sample.get("source_file")}
source_type: {sample.get("source_type")}
source_authority: {sample.get("source_authority")}
source_roles: {sample.get("evidence_roles_source")}
source_allowed_fields: {sample.get("claim_allowed_fields_source")}
clean_evidence_type: {sample.get("clean_evidence_type")}
offline_guess: {offline.get("evidence_role_pred")}
offline_allowed_fields: {offline.get("claim_allowed_fields_pred")}

chunk_text:
{str(sample.get("text") or "")[:2200]}
"""
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    if not isinstance(obj, dict):
        raise ValueError("LLM response is not a JSON object")
    return obj


def build_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, Any]],
    review_queue: list[dict[str, Any]],
    external_labels: dict[str, dict[str, Any]],
    direct_model_counts: Counter[str] | None = None,
) -> dict[str, Any]:
    role_counts = Counter(str((row.get("adjudication") or {}).get("evidence_role")) for row in rows)
    status_counts = Counter(str(row.get("adjudication_status")) for row in rows)
    source_type_counts = Counter(str(row.get("source_type")) for row in rows)
    source_authority_counts = Counter(str(row.get("source_authority")) for row in rows)
    allowed_field_counts: Counter[str] = Counter()
    confidence_buckets: Counter[str] = Counter()
    role_changed = 0
    for row in rows:
        allowed_field_counts.update((row.get("adjudication") or {}).get("claim_allowed_fields") or [])
        confidence_buckets.update([bucket_confidence(float((row.get("adjudication") or {}).get("confidence") or 0.0))])
        role_changed += int(bool((row.get("offline_vs_adjudicated") or {}).get("role_changed")))
    return {
        "created_at": now(),
        "samples": args.samples,
        "provider": args.provider,
        "labels": args.labels,
        "model": getattr(args, "model", ""),
        "fallback_models": getattr(args, "fallback_models", ""),
        "direct_model_counts": dict(direct_model_counts or {}),
        "external_label_key_count": len(external_labels),
        "output_dir": str(output_dir),
        "sample_count": len(rows),
        "accepted_auto_count": status_counts.get("accepted_auto", 0),
        "review_queue_count": len(review_queue),
        "role_changed_count": role_changed,
        "status_counts": dict(status_counts),
        "adjudicated_role_counts": dict(role_counts),
        "source_type_counts": dict(source_type_counts),
        "source_authority_counts": dict(source_authority_counts),
        "allowed_field_counts": dict(allowed_field_counts),
        "confidence_buckets": dict(confidence_buckets),
        "artifacts": {
            "adjudicated_samples": str(output_dir / "adjudicated_samples.jsonl"),
            "review_queue": str(output_dir / "llm_or_human_review_queue.jsonl"),
            "report": str(output_dir / "审计报告.md"),
        },
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Evidence Chunk 审计合并报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- 输入样本：`{manifest['samples']}`",
        f"- 裁决来源：{manifest['provider']}",
        f"- 首选模型：{manifest.get('model') or '无'}",
        f"- 外部标签路径：`{manifest['labels']}`" if manifest["labels"] else "- 外部标签路径：无，本次使用离线规则弱标签",
        f"- 输出目录：`{manifest['output_dir']}`",
        f"- 样本数：{manifest['sample_count']}",
        f"- 自动接受：{manifest['accepted_auto_count']}",
        f"- 待 LLM/VLM 或人工复核：{manifest['review_queue_count']}",
        f"- 相对离线弱标签 role 发生变化：{manifest['role_changed_count']}",
        "",
        "## adjudication_status 分布",
        "",
    ]
    append_count_section(lines, manifest["status_counts"])
    lines.extend(["", "## 裁决 evidence_role 分布", ""])
    append_count_section(lines, manifest["adjudicated_role_counts"])
    lines.extend(["", "## source_type 分布", ""])
    append_count_section(lines, manifest["source_type_counts"])
    lines.extend(["", "## source_authority 分布", ""])
    append_count_section(lines, manifest["source_authority_counts"])
    lines.extend(["", "## claim_allowed_fields 分布", ""])
    append_count_section(lines, manifest["allowed_field_counts"])
    lines.extend(["", "## confidence 分桶", ""])
    append_count_section(lines, manifest["confidence_buckets"])
    if manifest.get("direct_model_counts"):
        lines.extend(["", "## 直接调用模型分布", ""])
        append_count_section(lines, manifest["direct_model_counts"])
    lines.extend(
        [
            "",
            "## 使用说明",
            "",
            "- `adjudicated_samples.jsonl` 是当前可消费标签；`adjudication.label_source` 会标明来自离线规则还是外部裁决。",
            "- `llm_or_human_review_queue.jsonl` 是下一步最该交给 LLM/VLM 或人工看的样本，不需要全量逐条审。",
            "- 若接入外部 LLM/VLM，请输出含 `audit_id` 或 `evidence_id` 的 JSONL，再用 `--provider llm_jsonl` 或 `--provider vlm_jsonl --labels <path>` 合并。",
            "",
            "## 产物",
            "",
        ]
    )
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_count_section(lines: list[str], counts: dict[str, int]) -> None:
    if not counts:
        lines.append("- 无")
        return
    for key, value in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")


def bucket_confidence(value: float) -> str:
    if value >= 0.9:
        return "0.90-1.00"
    if value >= 0.75:
        return "0.75-0.89"
    if value >= 0.5:
        return "0.50-0.74"
    return "0.00-0.49"


def normalize_fields(fields: Iterable[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for field in fields:
        text = str(field or "").strip()
        if not text or text not in ALL_CLAIM_FIELDS or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return sorted(out)


def missing_fields(allowed: list[str]) -> list[str]:
    return sorted(set(ALL_CLAIM_FIELDS) - set(allowed))


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


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def dedupe_keep_order(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    raise SystemExit(main())
