#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from source_registry_rules_v1_0_4 import build_registry_row


DEFAULT_SOURCES = "/root/datasets/chinese_landscape_authority_corpus/metadata/sources.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 v1.0.4 source registry。")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--version", default="source_registry_v1_0_4")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{args.version}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = list(iter_jsonl(Path(args.sources)))
    rows = [build_registry_row(source) for source in sources]
    write_jsonl(output_dir / "source_registry.jsonl", rows)

    manifest = build_manifest(args, output_dir, rows)
    write_json(output_dir / "manifest.json", manifest)
    write_report(output_dir / "构建报告.md", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def build_manifest(args: argparse.Namespace, output_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_type_counts = Counter(str(row.get("source_type_normalized") or "") for row in rows)
    authority_counts = Counter(str(row.get("source_authority") or "") for row in rows)
    role_counts: Counter[str] = Counter()
    review_count = 0
    for row in rows:
        role_counts.update(str(role) for role in row.get("evidence_roles") or [])
        review_count += int(bool(row.get("needs_review")))
    return {
        "created_at": now(),
        "sources": args.sources,
        "output_dir": str(output_dir),
        "source_count": len(rows),
        "needs_review_count": review_count,
        "source_type_counts": dict(source_type_counts),
        "source_authority_counts": dict(authority_counts),
        "evidence_role_counts": dict(role_counts),
        "artifacts": {
            "source_registry": str(output_dir / "source_registry.jsonl"),
            "report": str(output_dir / "构建报告.md"),
        },
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# v1.0.4 Source Registry 构建报告",
        "",
        f"- 创建时间：{manifest['created_at']}",
        f"- 输入 sources：`{manifest['sources']}`",
        f"- 输出目录：`{manifest['output_dir']}`",
        f"- 来源记录数：{manifest['source_count']}",
        f"- 需要复核：{manifest['needs_review_count']}",
        "",
        "## source_authority 分布",
        "",
    ]
    for key, value in sorted(manifest["source_authority_counts"].items(), key=lambda item: (item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## source_type_normalized 分布", ""])
    for key, value in sorted(manifest["source_type_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## evidence_roles 分布", ""])
    for key, value in sorted(manifest["evidence_role_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## 产物", ""])
    for key, value in manifest["artifacts"].items():
        lines.append(f"- {key}: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")


if __name__ == "__main__":
    raise SystemExit(main())
