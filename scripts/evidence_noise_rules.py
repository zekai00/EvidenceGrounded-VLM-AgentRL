#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Any


DOT_LEADER_RE = re.compile(r"(?:\.{5,}|…{2,}|·{4,}|_{5,})")
SECTION_PAGE_RE = re.compile(
    r"(?:^|\s|\n)(?:第[一二三四五六七八九十百]+章|\d+(?:\.\d+){1,4})"
    r".{0,80}?(?:\.{5,}|…{2,}|·{4,}|_{5,})\s*\d{1,4}(?:\s|$)"
)
REF_MARK_RE = re.compile(r"(?:^|\n|\s)\[\d{1,3}\]")
FIGURE_REF_RE = re.compile(r"(?:图|圖|Fig\.?|Figure)\s*[A-Za-z]?\s*\d+(?:[.\-]\d+)*", re.IGNORECASE)


def evidence_text(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("display_snippet", "evidence_summary", "clean_text", "text", "raw_text", "title"):
        value = row.get(key)
        if value:
            parts.append(str(value))
    return "\n".join(parts)


def classify_evidence_row(row: dict[str, Any]) -> dict[str, Any]:
    text = evidence_text(row)
    compact = re.sub(r"\s+", "", text)
    lower = text.lower()
    reasons: list[str] = []

    dot_lines = sum(1 for line in text.splitlines() if DOT_LEADER_RE.search(line))
    section_page_hits = len(SECTION_PAGE_RE.findall(text))
    ref_hits = len(REF_MARK_RE.findall(text))
    char_count = max(1, len(compact))
    cjk_latin_digit_count = len(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", compact))
    punctuation_count = len(re.findall(r"[^\u4e00-\u9fffA-Za-z0-9]", compact))

    source_quality = str(row.get("source_quality") or "")
    citation_level = str(row.get("citation_level") or "")
    raw_type = str(row.get("evidence_type") or "")
    page_start = _to_int(row.get("page_start") if row.get("page_start") is not None else row.get("page"))

    clean_type = "normal_body_text"
    noise_score = 0.0

    if "caption" in citation_level or "caption" in raw_type:
        clean_type = "caption"
        reasons.append("caption_metadata")
    elif is_toc_text(text, dot_lines, section_page_hits):
        clean_type = "toc"
        noise_score = 0.98
        reasons.extend(["dot_leader_or_section_page_pattern"])
        if "目录" in text or "目 录" in text or "contents" in lower:
            reasons.append("toc_keyword")
    elif is_bibliography_text(text, ref_hits):
        clean_type = "bibliography"
        noise_score = 0.92
        reasons.append("bibliography_pattern")
    elif is_front_matter_text(text, page_start):
        clean_type = "front_matter"
        noise_score = 0.78
        reasons.append("front_matter_keyword")
    elif is_back_matter_text(text):
        clean_type = "back_matter"
        noise_score = 0.84
        reasons.append("back_matter_keyword")
    elif is_header_footer_text(text):
        clean_type = "header_footer"
        noise_score = 0.76
        reasons.append("header_footer_pattern")
    elif is_ocr_noise_text(compact, cjk_latin_digit_count, punctuation_count, char_count, source_quality):
        clean_type = "ocr_noise"
        noise_score = 0.86
        reasons.append("low_text_quality_or_high_punctuation")
    elif FIGURE_REF_RE.search(text):
        clean_type = "figure_reference_text"
        noise_score = 0.02
        reasons.append("figure_reference_pattern")

    unusable_types = {"toc", "bibliography", "front_matter", "back_matter", "header_footer", "ocr_noise"}
    usable_for_claim = clean_type not in unusable_types
    usable_for_retrieval = clean_type not in unusable_types

    return {
        "clean_evidence_type": clean_type,
        "noise_score": round(float(noise_score), 4),
        "noise_reasons": sorted(set(reasons)),
        "usable_for_claim": usable_for_claim,
        "usable_for_retrieval": usable_for_retrieval,
        "dot_leader_line_count": dot_lines,
        "section_page_pattern_count": section_page_hits,
        "reference_marker_count": ref_hits,
    }


def is_toc_text(text: str, dot_lines: int, section_page_hits: int) -> bool:
    lower = text.lower()
    toc_keywords = ["目录", "目 录", "contents", "本章小结", "主要结论", "创新之处", "不足与展望"]
    if section_page_hits >= 2:
        return True
    if dot_lines >= 3 and any(keyword in text or keyword in lower for keyword in toc_keywords):
        return True
    if dot_lines >= 4 and len(text) < 2500:
        return True
    return False


def is_bibliography_text(text: str, ref_hits: int) -> bool:
    lower = text.lower()
    if "参考文献" in text or "bibliography" in lower or "references" in lower:
        return True
    if ref_hits >= 6 and ("出版社" in text or "journal" in lower or "doi" in lower):
        return True
    return False


def is_front_matter_text(text: str, page_start: int | None) -> bool:
    if page_start is not None and page_start > 12:
        return False
    keywords = ["摘要", "abstract", "关键词", "key words", "目录", "目 录"]
    return sum(1 for keyword in keywords if keyword in text.lower()) >= 2


def is_back_matter_text(text: str) -> bool:
    patterns = [
        "致谢",
        "作者在读期间",
        "科研成果简介",
        "攻读硕士",
        "攻读博士",
        "附录",
        "声明",
    ]
    return any(pattern in text for pattern in patterns)


def is_header_footer_text(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if len(compact) > 80:
        return False
    if compact.isdigit():
        return True
    return any(keyword in compact for keyword in ["硕士学位论文", "博士学位论文", "万方数据", "版权所有"])


def is_ocr_noise_text(
    compact: str,
    cjk_latin_digit_count: int,
    punctuation_count: int,
    char_count: int,
    source_quality: str,
) -> bool:
    if char_count < 20:
        return False
    signal_ratio = cjk_latin_digit_count / char_count
    punctuation_ratio = punctuation_count / char_count
    if "vlm_ocr" in source_quality and signal_ratio < 0.35:
        return True
    if punctuation_ratio > 0.65 and cjk_latin_digit_count < 40:
        return True
    return False


def _to_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None
