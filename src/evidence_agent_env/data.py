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

    def __init__(self, index_dir: str | Path, max_chunks: int = 0, anchor_rerank: bool = False) -> None:
        self.index_dir = Path(index_dir)
        self.anchor_rerank = bool(anchor_rerank)
        path = self.index_dir / "corpus_chunks.jsonl"
        self.chunks = read_jsonl(path)
        if max_chunks and max_chunks > 0:
            self.chunks = self.chunks[:max_chunks]
        self.by_id = {str(item.get("evidence_id")): item for item in self.chunks}
        self._token_cache: dict[str, Counter[str]] = {}
        self._text_cache: dict[str, str] = {}
        self._anchor_cache: dict[str, dict[str, Any]] = {}

    def open(self, evidence_id: str) -> dict[str, Any] | None:
        return self.by_id.get(str(evidence_id))

    def search(self, query: str, scope: str, task: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        top_k = max(1, int(top_k))
        query_counts = Counter(tokenize(query))
        if not query_counts:
            query_counts = Counter(tokenize(task.get("source_file", "")))
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.chunks:
            if item.get("usable_for_retrieval") is False:
                continue
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
        try:
            noise_score = float(item.get("noise_score") or 0.0)
        except Exception:
            noise_score = 0.0
        if noise_score > 0:
            score *= max(0.0, 1.0 - min(0.95, noise_score))
        if self.anchor_rerank:
            score += self._anchor_bonus(item, task)
        return score

    def _anchor_bonus(self, item: dict[str, Any], task: dict[str, Any]) -> float:
        profile = self._anchor_profile(task)
        if not profile:
            return 0.0

        bonus = 0.0
        source_match = str(item.get("source_file", "")) == str(task.get("source_file", ""))
        page_distance = item_page_distance(item, task.get("page"))
        text = self._item_text(item)
        compact_text = compact_for_match(text)
        label_hits = sum(1 for label in profile["figure_labels"] if compact_for_match(label) in compact_text)
        term_hits = 0
        for term in profile["caption_terms"]:
            normalized = compact_for_match(term)
            if len(normalized) >= 4 and normalized in compact_text:
                term_hits += 1
        entity_hits = 0
        for term in profile["entity_terms"]:
            normalized = compact_for_match(term)
            if len(normalized) >= 2 and normalized in compact_text:
                entity_hits += 1

        has_anchor_text = bool(label_hits or term_hits or entity_hits)
        if source_match and page_distance is not None:
            if has_anchor_text:
                if page_distance == 0:
                    bonus += 0.08
                elif page_distance == 1:
                    bonus += 0.05
                elif page_distance <= 3:
                    bonus += 0.02
            elif page_distance == 0:
                bonus += 0.015
            elif page_distance == 1:
                bonus += 0.01

        if label_hits:
            bonus += min(0.30, 0.18 + 0.04 * (label_hits - 1))

        if term_hits:
            bonus += min(0.16, 0.04 * term_hits)

        if entity_hits:
            bonus += min(0.09, 0.03 * entity_hits)

        if item.get("clean_evidence_type") == "figure_reference_text":
            bonus += 0.02

        return bonus

    def _anchor_profile(self, task: dict[str, Any]) -> dict[str, Any]:
        cache_key = str(task.get("task_id") or id(task))
        if cache_key not in self._anchor_cache:
            self._anchor_cache[cache_key] = build_anchor_profile(task)
        return self._anchor_cache[cache_key]

    def _item_text(self, item: dict[str, Any]) -> str:
        eid = str(item.get("evidence_id"))
        if eid not in self._text_cache:
            self._text_cache[eid] = " ".join(
                str(item.get(key, ""))
                for key in ["title", "source_file", "display_snippet", "evidence_summary", "clean_text", "text"]
            )
        return self._text_cache[eid]

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
            "clean_evidence_type": item.get("clean_evidence_type"),
            "noise_score": item.get("noise_score"),
            "adjudicated_evidence_role": item.get("adjudicated_evidence_role"),
            "adjudication_status": item.get("adjudication_status"),
            "adjudicated_claim_allowed_fields": item.get("adjudicated_claim_allowed_fields"),
            "usable_for_claim_by_adjudication": item.get("usable_for_claim_by_adjudication"),
            "display_snippet": item.get("display_snippet") or item.get("evidence_summary") or item.get("text", "")[:320],
        }


FIGURE_LABEL_RE = re.compile(r"(?:图|圖|fig\.?|figure)\s*[A-Za-z]?\s*\d+(?:[.\-]\d+)*", re.IGNORECASE)
TITLE_RE = re.compile(r"《([^》]{2,80})》")
LATIN_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
CJK_TERM_RE = re.compile(r"[\u4e00-\u9fff]{2,24}")


def build_anchor_profile(task: dict[str, Any]) -> dict[str, Any]:
    captions: list[str] = []
    for item in task.get("local_evidence") or []:
        text = str(item.get("display_snippet") or item.get("text") or "")
        if text:
            captions.append(text)
    for region in task.get("region_candidates") or []:
        if region.get("target_region_rank") not in {None, 1, "1"} and region.get("type") == "figure_candidate":
            continue
        for key in ("caption_hint", "linked_caption_text", "nearby_text"):
            text = str(region.get(key) or "")
            if text:
                captions.append(text)

    deduped_captions = stable_unique(captions)[:12]
    figure_labels: list[str] = []
    caption_terms: list[str] = []
    entity_terms: list[str] = []
    for caption in deduped_captions:
        figure_labels.extend(FIGURE_LABEL_RE.findall(caption))
        entity_terms.extend(match.group(1).strip() for match in TITLE_RE.finditer(caption))
        caption_terms.extend(extract_caption_terms(caption))

    return {
        "captions": deduped_captions,
        "figure_labels": stable_unique(figure_labels)[:12],
        "caption_terms": stable_unique(caption_terms)[:32],
        "entity_terms": stable_unique(entity_terms)[:24],
    }


def extract_caption_terms(text: str) -> list[str]:
    cleaned = FIGURE_LABEL_RE.sub(" ", str(text))
    cleaned = re.sub(r"[\[\]（）()【】,:;，。；：/\\|]+", " ", cleaned)
    terms: list[str] = []
    terms.extend(match.group(1).strip() for match in TITLE_RE.finditer(text))
    terms.extend(term for term in CJK_TERM_RE.findall(cleaned) if not is_generic_caption_term(term))
    latin_words = [word.lower() for word in LATIN_WORD_RE.findall(cleaned)]
    latin_words = [word for word in latin_words if word not in GENERIC_LATIN_TERMS]
    for size in (3, 2):
        for idx in range(0, max(0, len(latin_words) - size + 1)):
            terms.append(" ".join(latin_words[idx : idx + size]))
    terms.extend(latin_words)
    return [term for term in terms if len(compact_for_match(term)) >= 2]


GENERIC_LATIN_TERMS = {
    "fig",
    "figure",
    "plate",
    "source",
    "from",
    "and",
    "the",
    "with",
    "landscape",
    "painting",
}


def is_generic_caption_term(term: str) -> bool:
    generic = {"来源", "自制", "图像", "模式", "山水画", "局部", "图片", "画面"}
    return term in generic or term.isdigit()


def item_page_distance(item: dict[str, Any], task_page: Any) -> int | None:
    try:
        page = int(task_page)
    except Exception:
        return None
    start = item.get("page_start") if item.get("page_start") is not None else item.get("page")
    end = item.get("page_end") if item.get("page_end") is not None else start
    try:
        start_i, end_i = int(start), int(end)
    except Exception:
        return None
    if start_i <= page <= end_i:
        return 0
    if page < start_i:
        return start_i - page
    return page - end_i


def compact_for_match(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = compact_for_match(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out
