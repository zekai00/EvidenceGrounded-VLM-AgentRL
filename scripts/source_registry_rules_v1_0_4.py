#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path
from typing import Any


OBJECT_METADATA_FIELDS = [
    "depicted_work_title",
    "parent_work_title",
    "artist",
    "dynasty",
    "collection",
    "medium_dimensions",
]
VISUAL_FIELDS = [
    "image_scope",
    "displayed_region",
    "object_type",
    "visual_subject",
    "visual_elements",
    "composition",
    "technique",
]
ANCHOR_FIELDS = ["caption_text", "figure_label", "entity_status"]
CONTEXT_FIELDS = ["style_analysis", "historical_context", "theory_concept"]


def build_registry_row(source: dict[str, Any]) -> dict[str, Any]:
    source_type = str(source.get("source_type") or "")
    category = str(source.get("category") or "")
    authority = source_authority(source)
    roles = evidence_roles_for(source_type, category)
    row = dict(source)
    row.update(
        {
            "source_type_normalized": normalize_source_type(source_type),
            "source_authority": authority,
            "text_quality": text_quality(source),
            "source_periods": normalize_periods(source.get("dynasties") or []),
            "coverage_periods": coverage_periods(source),
            "school_lineage_tags": school_lineage_tags(source),
            "evidence_roles": roles,
            "claim_allowed_fields": claim_allowed_fields(roles, source_type),
            "claim_disallowed_fields": claim_disallowed_fields(roles, source_type),
            "retrieval_priority": retrieval_priority(authority, source_type, roles),
            "needs_review": needs_review(source, roles),
            "registry_version": "source_registry_v1_0_4",
            "registry_label_source": "rule_seed",
        }
    )
    return row


def source_authority(source: dict[str, Any]) -> str:
    source_type = str(source.get("source_type") or "")
    category = str(source.get("category") or "")
    current = str(source.get("authority_level") or "")
    if source_type == "museum_collection_entry_text_pdf":
        return "A+"
    if source_type in {"museum_catalog", "symposium_proceedings"}:
        return "A"
    if source_type in {"palace_museum_article", "palace_museum_article_merged_pdf"}:
        return "A-"
    if source_type.startswith("ancient_theory"):
        return "A-"
    if source_type == "legacy_project_pdf":
        return "B"
    if category in {"03_作品教学资料"}:
        return "C" if current == "C" else "A-"
    return current or "B"


def normalize_source_type(source_type: str) -> str:
    mapping = {
        "museum_collection_entry_text_pdf": "object_metadata_entry",
        "museum_catalog": "museum_catalog",
        "symposium_proceedings": "museum_scholarly_catalog",
        "palace_museum_article": "official_museum_article",
        "palace_museum_article_merged_pdf": "official_museum_article",
        "ancient_theory_text_pdf": "ancient_theory_primary_text",
        "ancient_theory_scan_pdf": "ancient_theory_primary_text",
        "legacy_project_pdf": "legacy_secondary_literature",
        "museum_education_resource": "museum_education_resource",
        "museum_exhibition_guide": "museum_exhibition_guide",
    }
    return mapping.get(source_type, source_type or "unknown_source_type")


def text_quality(source: dict[str, Any]) -> str:
    status = str(source.get("download_status") or "")
    source_type = str(source.get("source_type") or "")
    if status in {"failed", "missing"}:
        return "unavailable"
    if source_type in {"museum_collection_entry_text_pdf", "ancient_theory_text_pdf"}:
        return "high"
    if source_type in {"museum_catalog", "symposium_proceedings", "palace_museum_article", "palace_museum_article_merged_pdf"}:
        return "medium"
    if source_type == "legacy_project_pdf":
        return "needs_audit"
    if source_type.endswith("_scan_pdf"):
        return "medium"
    return "medium"


def evidence_roles_for(source_type: str, category: str) -> list[str]:
    if source_type == "museum_collection_entry_text_pdf":
        return ["object_metadata"]
    if source_type == "museum_catalog":
        return ["object_metadata", "caption_or_plate", "style_analysis", "historical_context"]
    if source_type == "symposium_proceedings":
        return ["style_analysis", "historical_context", "secondary_scholarship"]
    if source_type in {"palace_museum_article", "palace_museum_article_merged_pdf"}:
        return ["style_analysis", "historical_context", "object_metadata"]
    if source_type.startswith("ancient_theory"):
        return ["theory_primary_text", "style_analysis"]
    if source_type == "legacy_project_pdf":
        return ["secondary_scholarship", "low_trust_legacy"]
    if source_type in {"museum_education_resource", "museum_exhibition_guide"}:
        return ["teaching_overview", "historical_context"]
    if category == "08_优先补充文献":
        return ["secondary_scholarship"]
    return ["secondary_scholarship"]


def claim_allowed_fields(roles: list[str], source_type: str) -> list[str]:
    allowed: set[str] = set()
    if "object_metadata" in roles:
        allowed.update(OBJECT_METADATA_FIELDS)
        allowed.update(ANCHOR_FIELDS)
    if "caption_or_plate" in roles:
        allowed.update(ANCHOR_FIELDS)
        allowed.update(["image_scope", "object_type", "displayed_region"])
    if "style_analysis" in roles or "secondary_scholarship" in roles:
        allowed.update(["style_analysis", "historical_context", "composition", "technique", "visual_elements", "object_type"])
    if "theory_primary_text" in roles:
        allowed.update(["theory_concept", "style_analysis", "composition", "technique"])
    if "teaching_overview" in roles:
        allowed.update(["historical_context", "style_analysis", "object_type"])
    if "low_trust_legacy" in roles:
        allowed.difference_update({"collection", "medium_dimensions"})
    if source_type == "legacy_project_pdf":
        allowed.difference_update({"collection", "medium_dimensions"})
    return sorted(allowed)


def claim_disallowed_fields(roles: list[str], source_type: str = "") -> list[str]:
    all_fields = set(OBJECT_METADATA_FIELDS + VISUAL_FIELDS + ANCHOR_FIELDS + CONTEXT_FIELDS)
    allowed = set(claim_allowed_fields(roles, source_type))
    return sorted(all_fields - allowed)


def retrieval_priority(authority: str, source_type: str, roles: list[str]) -> float:
    base = {"A+": 1.2, "A": 1.0, "A-": 0.85, "B": 0.55, "C": 0.25}.get(authority, 0.5)
    if "object_metadata" in roles:
        base += 0.08
    if source_type == "legacy_project_pdf":
        base -= 0.12
    if "low_trust_legacy" in roles:
        base -= 0.10
    return round(max(0.05, min(1.3, base)), 3)


def needs_review(source: dict[str, Any], roles: list[str]) -> bool:
    source_type = str(source.get("source_type") or "")
    status = str(source.get("download_status") or "")
    if status in {"failed", "missing", "duplicate_skipped"}:
        return True
    if source_type == "legacy_project_pdf":
        return True
    if "low_trust_legacy" in roles:
        return True
    if not source.get("dynasties") and source_type not in {"museum_collection_entry_text_pdf"}:
        return True
    return False


def normalize_periods(dynasties: list[Any]) -> list[str]:
    out: list[str] = []
    for item in dynasties:
        text = str(item or "").strip()
        if not text:
            continue
        out.extend(PERIOD_MAP.get(text, [slug_period(text)]))
    return stable_unique(out) or ["unknown_period"]


PERIOD_MAP = {
    "魏晋南北朝": ["pre_tang"],
    "刘宋": ["pre_tang"],
    "南齐": ["pre_tang"],
    "隋": ["pre_tang"],
    "唐": ["tang_five_dynasties"],
    "五代": ["tang_five_dynasties"],
    "宋": ["northern_song", "southern_song"],
    "北宋": ["northern_song"],
    "南宋": ["southern_song"],
    "元": ["yuan"],
    "明": ["ming"],
    "清": ["qing"],
    "近现代": ["modern"],
}


def coverage_periods(source: dict[str, Any]) -> list[str]:
    periods = normalize_periods(source.get("dynasties") or [])
    if len(periods) > 2:
        return ["cross_period"] + periods
    topics = " ".join(str(t) for t in source.get("topics") or [])
    title = str(source.get("title") or "")
    if any(term in topics + title for term in ["通史", "演变", "发展", "历代", "唐宋元", "宋元", "元明", "明清"]):
        return stable_unique(["cross_period"] + periods)
    return periods


def school_lineage_tags(source: dict[str, Any]) -> list[str]:
    text = " ".join([str(source.get("title") or ""), str(source.get("author") or "")] + [str(t) for t in source.get("topics") or []])
    rules = [
        ("李郭", "li_guo_tradition"),
        ("董巨", "dong_ju_tradition"),
        ("马夏", "ma_xia_school"),
        ("元四家", "yuan_four_masters"),
        ("吴门", "wu_school"),
        ("浙派", "zhe_school"),
        ("松江", "songjiang_school"),
        ("南北宗", "southern_northern_school_theory"),
        ("南宗", "southern_school_theory"),
        ("北宗", "northern_school_theory"),
        ("四王", "four_wangs"),
        ("清初四王", "four_wangs"),
        ("四僧", "four_monks"),
        ("清初四僧", "four_monks"),
        ("扬州", "yangzhou_school"),
        ("正统派", "orthodox_school"),
        ("文人画", "literati_painting"),
        ("青绿", "blue_green_landscape"),
        ("巨碑", "monumental_landscape"),
        ("园林", "garden_landscape"),
        ("近现代", "modern_ink"),
    ]
    tags = [tag for pattern, tag in rules if pattern in text]
    return stable_unique(tags)


def chunk_offline_label(chunk: dict[str, Any], registry_row: dict[str, Any] | None = None) -> dict[str, Any]:
    text = chunk_text(chunk)
    clean_type = str(chunk.get("clean_evidence_type") or "")
    roles = set((registry_row or {}).get("evidence_roles") or [])
    source_type = str((registry_row or chunk).get("source_type") or "")
    label = {
        "audit_label_source": "offline_rules",
        "evidence_role_pred": role_for_chunk_text(text, clean_type, roles, source_type),
        "claim_allowed_fields_pred": [],
        "claim_disallowed_fields_pred": [],
        "confidence": 0.55,
        "needs_llm_review": True,
        "rationale": "",
    }
    role = label["evidence_role_pred"]
    source_allowed = set((registry_row or {}).get("claim_allowed_fields") or [])
    allowed: set[str] = set()
    if role == "object_metadata":
        allowed.update(OBJECT_METADATA_FIELDS + ANCHOR_FIELDS)
        label["confidence"] = 0.82
    elif role == "caption_or_plate":
        allowed.update(ANCHOR_FIELDS + ["image_scope", "object_type", "displayed_region"])
        label["confidence"] = 0.78
    elif role in {"style_analysis", "historical_context"}:
        allowed.update(["style_analysis", "historical_context", "composition", "technique", "visual_elements", "object_type"])
        label["confidence"] = 0.62
    elif role == "theory_primary_text":
        allowed.update(["theory_concept", "style_analysis", "composition", "technique"])
        label["confidence"] = 0.78
    elif role in {"toc", "bibliography", "front_matter", "back_matter", "ocr_noise", "low_value_background"}:
        allowed.clear()
        label["confidence"] = 0.78
    if source_allowed and allowed:
        allowed &= source_allowed
    label["claim_allowed_fields_pred"] = sorted(allowed)
    all_fields = set(OBJECT_METADATA_FIELDS + VISUAL_FIELDS + ANCHOR_FIELDS + CONTEXT_FIELDS)
    label["claim_disallowed_fields_pred"] = sorted(all_fields - allowed)
    label["needs_llm_review"] = (
        registry_row is None
        or label["confidence"] < 0.75
        or role in {"style_analysis", "historical_context", "low_value_background"}
    )
    label["rationale"] = offline_rationale(role, source_type, clean_type)
    return label


def role_for_chunk_text(text: str, clean_type: str, roles: set[str], source_type: str) -> str:
    lower = text.lower()
    if clean_type in {"toc", "bibliography", "front_matter", "back_matter", "ocr_noise"}:
        return clean_type
    if looks_like_bibliography_or_index(text, lower):
        return "bibliography"
    if looks_like_repeated_title_page(text, lower):
        return "front_matter"
    if "theory_primary_text" in roles or source_type.startswith("ancient_theory"):
        return "theory_primary_text"
    if re.search(r"(accession|object number|medium|dimensions|collection|馆藏|藏品|尺寸|材质|作者|年代)", lower):
        return "object_metadata"
    if re.search(r"(图\s*\d|圖\s*\d|fig\.|figure|plate|插图|图版|caption)", lower):
        return "caption_or_plate"
    if any(term in text for term in ["风格", "画派", "笔墨", "皴法", "构图", "设色", "南宗", "北宗", "文人画"]):
        return "style_analysis"
    if any(term in text for term in ["时期", "历史", "演变", "发展", "影响", "收藏", "传承"]):
        return "historical_context"
    if "low_trust_legacy" in roles:
        return "low_value_background"
    return "historical_context"


def looks_like_bibliography_or_index(text: str, lower: str) -> bool:
    if any(term in lower for term in ["bibliography", "references", "works cited", "glossary-index", "list of plates"]):
        return True
    year_hits = len(re.findall(r"\b(18|19|20)\d{2}\b", lower))
    citation_terms = sum(
        int(term in lower)
        for term in ["university press", "yale university", "journal", "vol.", " no.", " pp.", "edited by", "publisher"]
    )
    if year_hits >= 3 and citation_terms >= 1:
        return True
    see_terms = lower.count("see also") + lower.count("see under") + lower.count(" see ")
    page_ref_hits = len(re.findall(r"\b\d{1,3}[-,]\s*\d{1,3}\b", lower))
    if see_terms >= 2 and page_ref_hits >= 4:
        return True
    dense_page_refs = len(re.findall(r"\b\d{1,3}(?:-\d{1,3})?(?:,\s*\d{1,3}){2,}\b", lower))
    return dense_page_refs >= 4 and len(text) < 2500


def looks_like_repeated_title_page(text: str, lower: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 6:
        return False
    unique_ratio = len(set(lines)) / max(1, len(lines))
    title_terms = sum(int(term in lower) for term in ["copyright", "isbn", "printed in", "all rights reserved", "preface"])
    return unique_ratio <= 0.45 or title_terms >= 2


def offline_rationale(role: str, source_type: str, clean_type: str) -> str:
    if clean_type in {"toc", "bibliography", "front_matter", "back_matter", "ocr_noise"}:
        return f"clean_evidence_type={clean_type}，不应支持核心 claim。"
    if role == "theory_primary_text":
        return "来源为古代画论或理论文本，适合支持理论/技法/构图概念，不适合支持具体作品元数据。"
    if role == "object_metadata":
        return "文本或来源类型显示其可支持作品身份、作者、年代、馆藏等元数据。"
    if role == "caption_or_plate":
        return "文本包含图号、图版或 caption 线索，适合支持 figure anchor 与图注相关字段。"
    if role == "style_analysis":
        return "文本包含风格、画派、笔墨或构图分析线索，适合作为风格/技法背景证据。"
    if role == "low_value_background":
        return "来源属于需要审计的 legacy 文献，默认只作为弱背景。"
    return f"根据 source_type={source_type} 和文本线索给出弱标签，建议 LLM/人工复核。"


def chunk_text(chunk: dict[str, Any]) -> str:
    for key in ("clean_text", "text", "raw_text", "display_snippet", "evidence_summary"):
        value = str(chunk.get(key) or "").strip()
        if value:
            return value
    return ""


def slug_period(text: str) -> str:
    cleaned = re.sub(r"\W+", "_", text.lower()).strip("_")
    return cleaned or "unknown_period"


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def source_key_from_path(path: str) -> str:
    return Path(path).name
