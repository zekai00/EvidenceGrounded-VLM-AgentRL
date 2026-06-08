"""Data loading and lightweight evidence retrieval."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize(text: str) -> list[str]:
    text = str(text or "").lower()
    latin = re.findall(r"[a-z0-9_]+", text)
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    return latin + cjk


class EvidenceIndex:
    """Small in-process retrieval over corpus_chunks.jsonl.

    This is intentionally simple. It is a deterministic local verifier/env
    retrieval backend, not the final production retriever.
    """

    def __init__(self, index_dir: str | Path, max_chunks: int = 0) -> None:
        self.index_dir = Path(index_dir)
        path = self.index_dir / "corpus_chunks.jsonl"
        self.chunks = read_jsonl(path)
        if max_chunks and max_chunks > 0:
            self.chunks = self.chunks[:max_chunks]
        self.by_id = {str(item.get("evidence_id")): item for item in self.chunks}
        self._token_cache: dict[str, Counter[str]] = {}

    def open(self, evidence_id: str) -> dict[str, Any] | None:
        return self.by_id.get(str(evidence_id))

    def search(self, query: str, scope: str, task: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        top_k = max(1, int(top_k))
        query_counts = Counter(tokenize(query))
        if not query_counts:
            query_counts = Counter(tokenize(task.get("source_file", "")))
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.chunks:
            if not self._in_scope(item, scope, task):
                continue
            score = self._score(query_counts, item, task)
            if score > 0:
                scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._public_result(item, score) for score, item in scored[:top_k]]

    def _in_scope(self, item: dict[str, Any], scope: str, task: dict[str, Any]) -> bool:
        if scope == "corpus":
            return True
        source_file = str(task.get("source_file", ""))
        if str(item.get("source_file", "")) != source_file:
            return False
        if scope == "same_document":
            return True
        page = task.get("page")
        try:
            page_num = int(page)
        except Exception:
            return True
        start = item.get("page_start") if item.get("page_start") is not None else item.get("page")
        end = item.get("page_end") if item.get("page_end") is not None else start
        try:
            start_i, end_i = int(start), int(end)
        except Exception:
            return scope != "current_page"
        if scope == "current_page":
            return start_i <= page_num <= end_i
        if scope == "nearby_pages":
            return start_i <= page_num + 1 and end_i >= page_num - 1
        return False

    def _score(self, query_counts: Counter[str], item: dict[str, Any], task: dict[str, Any]) -> float:
        eid = str(item.get("evidence_id"))
        if eid not in self._token_cache:
            text = " ".join(
                str(item.get(key, ""))
                for key in ["title", "source_file", "display_snippet", "evidence_summary", "text"]
            )
            self._token_cache[eid] = Counter(tokenize(text))
        doc_counts = self._token_cache[eid]
        if not doc_counts:
            return 0.0
        overlap = sum(min(count, doc_counts.get(tok, 0)) for tok, count in query_counts.items())
        norm = math.sqrt(sum(v * v for v in doc_counts.values()))
        score = overlap / max(1.0, norm)
        if str(item.get("source_file", "")) == str(task.get("source_file", "")):
            score += 0.2
        return score

    def _public_result(self, item: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "evidence_id": item.get("evidence_id"),
            "score": round(float(score), 6),
            "source_file": item.get("source_file"),
            "page_start": item.get("page_start") if item.get("page_start") is not None else item.get("page"),
            "page_end": item.get("page_end"),
            "authority_level": item.get("authority_level"),
            "citation_level": item.get("citation_level"),
            "source_quality": item.get("source_quality"),
            "display_snippet": item.get("display_snippet") or item.get("evidence_summary") or item.get("text", "")[:320],
        }
