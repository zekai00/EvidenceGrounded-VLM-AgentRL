#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import fitz
from PIL import Image, ImageDraw, ImageFont


CHINESE_DYNASTIES = [
    "先秦",
    "秦",
    "汉",
    "魏晋",
    "东晋",
    "南北朝",
    "隋",
    "唐",
    "五代",
    "北宋",
    "南宋",
    "宋",
    "元",
    "明",
    "清",
    "近现代",
    "现代",
    "当代",
]

TECHNIQUE_KEYWORDS = ["水墨", "设色", "青绿", "浅绛", "皴法", "泼墨", "泼彩", "工笔", "写意", "界画", "积墨", "焦墨"]
COMPOSITION_KEYWORDS = ["高远", "平远", "深远", "三远", "留白", "边角", "虚实", "散点透视", "全景", "布局", "空间", "构图"]
VISUAL_KEYWORDS = [
    "山",
    "水",
    "云",
    "雾",
    "树",
    "松",
    "石",
    "溪",
    "河",
    "江",
    "桥",
    "亭",
    "建筑",
    "舟",
    "船",
    "人物",
    "行旅",
    "渔",
    "瀑布",
    "园林",
    "梯田",
    "村落",
    "楼阁",
]

EVIDENCE_FILES = ["chunks.jsonl", "documents.jsonl", "pages.jsonl", "images.jsonl", "source_aliases.json", "manifest.json"]


@dataclass(frozen=True)
class Candidate:
    pdf_path: Path
    source_file: str
    source_stem: str
    page_num: int
    bbox_pt: tuple[float, float, float, float]
    image_bbox: list[int]
    area_ratio: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build claim-level evidence-seeking VLM AgentBench from raw PDFs.")
    parser.add_argument("--raw-pdfs-dir", default="/root/Workspace/ChineseLandscape/data/raw_pdfs")
    parser.add_argument("--evidence-store-root", default="/root/Workspace/ChineseLandscape/data/processed/documents")
    parser.add_argument("--output-root", default="/root/datasets/evidence_grounded_vlm_agentrl")
    parser.add_argument("--version", default="agentbench_v0_1_local_evidence")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--page-dpi", type=int, default=150)
    parser.add_argument("--crop-dpi", type=int, default=200)
    parser.add_argument("--min-area-ratio", type=float, default=0.008)
    parser.add_argument("--max-area-ratio", type=float, default=0.75)
    parser.add_argument("--min-width-ratio", type=float, default=0.07)
    parser.add_argument("--min-height-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260530)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = datetime.now().strftime("%Y%m%d_%H%M")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / f"{args.version}_{now}"
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in ["pages", "crops", "overlays", "review", "sft", "episodes"]:
        (output_dir / child).mkdir(exist_ok=True)

    evidence_snapshot = snapshot_evidence_store(Path(args.evidence_store_root), output_root, now)
    chunks = load_chunks(evidence_snapshot / "chunks.jsonl")
    chunks_by_source = index_chunks_by_source(chunks)

    all_candidates = collect_candidates(Path(args.raw_pdfs_dir), args)
    selected = select_balanced_candidates(all_candidates, args.limit)
    split_map = build_source_split(selected)

    rendered_pages: dict[tuple[str, int], Path] = {}
    tasks: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    sft_rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    errors: list[dict[str, Any]] = []

    for index, candidate in enumerate(selected):
        try:
            task = build_task(
                index=index,
                candidate=candidate,
                output_dir=output_dir,
                page_dpi=args.page_dpi,
                crop_dpi=args.crop_dpi,
                rendered_pages=rendered_pages,
                split=split_map[candidate.source_stem],
                chunks=chunks,
                chunks_by_source=chunks_by_source,
                evidence_snapshot=evidence_snapshot,
            )
            tasks.append(task)
            claim_rows.extend(task_to_claim_rows(task))
            evidence_rows.extend(task_to_evidence_rows(task))
            episode = build_oracle_episode(task)
            episodes.append(episode)
            for sample in episode_to_sft_samples(episode, task):
                sft_rows_by_split[task["split"]].append(sample)
        except Exception as exc:
            errors.append(
                {
                    "source_file": candidate.source_file,
                    "page": candidate.page_num,
                    "image_bbox": candidate.image_bbox,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_jsonl(output_dir / "tasks_all.jsonl", tasks)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / f"{split}_tasks.jsonl", [task for task in tasks if task["split"] == split])
    write_jsonl(output_dir / "claim_gold.jsonl", claim_rows)
    write_jsonl(output_dir / "evidence_links.jsonl", evidence_rows)
    write_jsonl(output_dir / "episodes" / "oracle_episodes.jsonl", episodes)
    for split in ["train", "val", "test"]:
        write_jsonl(output_dir / "sft" / f"{split}.jsonl", sft_rows_by_split.get(split, []))

    review_path = write_review_html(output_dir / "review" / "review.html", tasks[:200])
    summary = build_summary(args, output_dir, evidence_snapshot, all_candidates, selected, tasks, claim_rows, evidence_rows, sft_rows_by_split, errors)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "manifest.json").write_text(json.dumps(build_manifest(args, output_dir, evidence_snapshot, summary), ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output_dir / "构建报告.md", summary, review_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def snapshot_evidence_store(source_root: Path, output_root: Path, now: str) -> Path:
    target = output_root / f"evidence_store_legacy_milvus_{now}"
    target.mkdir(parents=True, exist_ok=True)
    for name in EVIDENCE_FILES:
        src = source_root / name
        if src.exists():
            shutil.copy2(src, target / name)
    manifest_path = target / "manifest.json"
    extra = {
        "snapshot_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "snapshot_note": "Copied from ChineseLandscape canonical evidence store for EvidenceGrounded-VLM-AgentRL v0.1 local evidence.",
        "source_type": "legacy_milvus_pdf",
        "authority_level": "local_research_pdf",
        "citation_level": "chunk",
        "page_level_citation_available": False,
    }
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        data["evidence_grounded_agentrl_snapshot"] = extra
    else:
        data = extra
    manifest_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            text = "\n".join(str(row.get(key) or "") for key in ["title", "contextual_prefix", "raw_chunk_text", "retrieval_text"])
            row["_norm_source_file"] = normalize_source(str(row.get("source_file") or ""))
            row["_search_text"] = text
            row["_search_norm"] = normalize_text(text)
            row["_tokens"] = set(tokenize(text))
            chunks.append(row)
    return chunks


def index_chunks_by_source(chunks: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        by_source[chunk["_norm_source_file"]].append(chunk)
    return by_source


def collect_candidates(raw_pdfs_dir: Path, args: argparse.Namespace) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen: set[tuple[str, int, tuple[int, int, int, int]]] = set()
    for pdf_path in sorted(raw_pdfs_dir.glob("*.pdf")):
        try:
            with fitz.open(pdf_path) as doc:
                for page_index, page in enumerate(doc):
                    rect = page.rect
                    for block in page.get_text("dict").get("blocks", []):
                        if block.get("type") != 1:
                            continue
                        x0, y0, x1, y1 = [float(v) for v in block.get("bbox", [0, 0, 0, 0])]
                        width = max(0.0, x1 - x0)
                        height = max(0.0, y1 - y0)
                        if rect.width <= 0 or rect.height <= 0:
                            continue
                        area_ratio = (width * height) / max(1.0, rect.width * rect.height)
                        if area_ratio < args.min_area_ratio or area_ratio > args.max_area_ratio:
                            continue
                        if width / rect.width < args.min_width_ratio or height / rect.height < args.min_height_ratio:
                            continue
                        norm = normalize_bbox([x0, y0, x1, y1], rect)
                        if bbox_area(norm) <= 0:
                            continue
                        coarse = tuple(round(v / 5) * 5 for v in norm)
                        key = (pdf_path.name, page_index + 1, coarse)
                        if key in seen:
                            continue
                        seen.add(key)
                        candidates.append(
                            Candidate(
                                pdf_path=pdf_path,
                                source_file=pdf_path.name,
                                source_stem=pdf_path.stem,
                                page_num=page_index + 1,
                                bbox_pt=(x0, y0, x1, y1),
                                image_bbox=norm,
                                area_ratio=round(area_ratio, 6),
                            )
                        )
        except Exception:
            continue
    return candidates


def select_balanced_candidates(candidates: list[Candidate], limit: int) -> list[Candidate]:
    by_source: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate.source_stem].append(candidate)
    for bucket in by_source.values():
        bucket.sort(key=lambda item: (item.page_num, -item.area_ratio))
    selected: list[Candidate] = []
    sources = sorted(by_source)
    cursor = 0
    while len(selected) < limit and sources:
        source = sources[cursor % len(sources)]
        bucket = by_source[source]
        if bucket:
            selected.append(bucket.pop(0))
        sources = [source for source in sources if by_source[source]]
        cursor += 1
    return selected


def build_source_split(candidates: list[Candidate]) -> dict[str, str]:
    counts = Counter(candidate.source_stem for candidate in candidates)
    total = sum(counts.values())
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
    current = Counter()
    split: dict[str, str] = {}
    order = {"train": 0, "val": 1, "test": 2}
    for source, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        target_split = min(
            targets,
            key=lambda name: (
                current[name] / max(1.0, targets[name]),
                current[name],
                order[name],
            ),
        )
        split[source] = target_split
        current[target_split] += count
    return split


def build_task(
    *,
    index: int,
    candidate: Candidate,
    output_dir: Path,
    page_dpi: int,
    crop_dpi: int,
    rendered_pages: dict[tuple[str, int], Path],
    split: str,
    chunks: list[dict[str, Any]],
    chunks_by_source: dict[str, list[dict[str, Any]]],
    evidence_snapshot: Path,
) -> dict[str, Any]:
    task_id = f"egva_v0_1_local_{index:06d}"
    page_image = render_page(candidate, output_dir / "pages", page_dpi, rendered_pages)
    crop_path = crop_region(candidate, output_dir / "crops", crop_dpi)
    caption = extract_caption(candidate)
    overlay_path = draw_overlay(page_image, candidate.image_bbox, caption.get("caption_bbox"), output_dir / "overlays" / f"{task_id}.jpg")
    fields = infer_fields(caption.get("caption_text", ""), candidate.source_stem)
    evidence_query = build_evidence_query(candidate, caption.get("caption_text", ""), fields)
    retrieved = search_evidence(evidence_query, candidate.source_file, chunks, chunks_by_source, top_k_source=8, top_k_global=5)
    claims, evidence_links = build_claims_and_evidence(fields, caption, candidate, retrieved)
    evidence_ids = sorted({eid for claim in claims for eid in claim.get("evidence_ids", [])})
    candidate_evidence_ids = [item["chunk_id"] for item in retrieved[:8]]
    return {
        "task_id": task_id,
        "split": split,
        "task_type": "claim_level_evidence_seeking",
        "source_type": "pdf_page",
        "source_file": candidate.source_file,
        "source_path": str(candidate.pdf_path),
        "source_stem": candidate.source_stem,
        "page": candidate.page_num,
        "page_image": str(page_image),
        "artwork_image": str(crop_path),
        "overlay_image": str(overlay_path),
        "goal": "Build claim-level grounded evidence for the Chinese landscape figure on this PDF page.",
        "gold": {
            "image_bbox": candidate.image_bbox,
            "caption_bbox": caption.get("caption_bbox"),
            "caption_text": caption.get("caption_text", ""),
            "claims": claims,
            "title": fields.get("title", ""),
            "artist": fields.get("artist", ""),
            "dynasty": fields.get("dynasty", ""),
            "visual_elements": fields.get("visual_elements", []),
            "technique": fields.get("technique", []),
            "composition": fields.get("composition", []),
            "evidence_chunk_ids": evidence_ids,
            "candidate_evidence_ids": candidate_evidence_ids,
            "evidence_query": evidence_query,
            "evidence_store": str(evidence_snapshot),
            "auto_label": True,
            "needs_review": True,
            "label_source": "from_scratch_pdf_blocks_with_legacy_milvus_evidence",
            "citation_level": "chunk",
        },
        "evidence_links": evidence_links,
        "candidate_meta": {
            "source": "pdf_image_block_from_raw_pdf",
            "area_ratio": candidate.area_ratio,
            "page_level_citation_available": False,
        },
    }


def render_page(candidate: Candidate, pages_dir: Path, dpi: int, cache: dict[tuple[str, int], Path]) -> Path:
    key = (candidate.source_file, candidate.page_num)
    if key in cache:
        return cache[key]
    out = pages_dir / f"{safe_name(candidate.source_stem)}_p{candidate.page_num:03d}.png"
    if not out.exists():
        with fitz.open(candidate.pdf_path) as doc:
            pix = doc[candidate.page_num - 1].get_pixmap(dpi=dpi, colorspace=fitz.csRGB)
            pix.save(out)
    cache[key] = out
    return out


def crop_region(candidate: Candidate, crops_dir: Path, dpi: int) -> Path:
    x0, y0, x1, y1 = candidate.image_bbox
    out = crops_dir / f"{safe_name(candidate.source_stem)}_p{candidate.page_num:03d}_{x0}_{y0}_{x1}_{y1}.jpg"
    if not out.exists():
        with fitz.open(candidate.pdf_path) as doc:
            pix = doc[candidate.page_num - 1].get_pixmap(clip=fitz.Rect(*candidate.bbox_pt), dpi=dpi, colorspace=fitz.csRGB)
            pix.save(out)
    return out


def extract_caption(candidate: Candidate) -> dict[str, Any]:
    with fitz.open(candidate.pdf_path) as doc:
        page = doc[candidate.page_num - 1]
        rect = page.rect
        image_bbox = candidate.image_bbox
        blocks = []
        for block in page.get_text("blocks"):
            if len(block) < 5:
                continue
            text = clean_text(str(block[4]))
            if not text:
                continue
            bbox = normalize_bbox([float(block[0]), float(block[1]), float(block[2]), float(block[3])], rect)
            score = caption_score(image_bbox, bbox, text)
            if score > -100:
                blocks.append({"bbox": bbox, "text": text, "score": score})
        blocks.sort(key=lambda item: item["score"], reverse=True)
        if not blocks:
            return {"caption_bbox": None, "caption_text": ""}
        selected = blocks[:1]
        first = selected[0]
        return {"caption_bbox": first["bbox"], "caption_text": first["text"], "caption_candidates": blocks[:5]}


def caption_score(image_bbox: list[int], text_bbox: list[int], text: str) -> float:
    ix0, iy0, ix1, iy1 = image_bbox
    tx0, ty0, tx1, ty1 = text_bbox
    horizontal_overlap = max(0, min(ix1, tx1) - max(ix0, tx0)) / max(1, min(ix1 - ix0, tx1 - tx0))
    gap_below = ty0 - iy1
    gap_above = iy0 - ty1
    near = -min(abs(gap_below), abs(gap_above)) / 50.0
    if 0 <= gap_below <= 120:
        near += 4.0
    if 0 <= gap_above <= 100:
        near += 2.0
    keyword = 0.0
    if re.search(r"(图\s*\d+|Figure|Fig\.?|《|作品|山水|画)", text, re.I):
        keyword += 4.0
    if len(text) > 260:
        keyword -= 2.0
    if horizontal_overlap < 0.05:
        keyword -= 2.0
    return near + keyword + horizontal_overlap


def infer_fields(caption: str, source_stem: str) -> dict[str, Any]:
    text = f"{caption} {source_stem}"
    title = ""
    match = re.search(r"《([^》]{1,32})》", text)
    if match:
        title = match.group(1).strip()
    artist = infer_artist(text)
    dynasty = next((item for item in CHINESE_DYNASTIES if item in text), "")
    technique = [item for item in TECHNIQUE_KEYWORDS if item in text]
    composition = [item for item in COMPOSITION_KEYWORDS if item in text]
    visual = [item for item in VISUAL_KEYWORDS if item in text]
    if "山水" in text and "山水" not in visual:
        visual.append("山水")
    return {
        "title": title,
        "artist": artist,
        "dynasty": dynasty,
        "visual_elements": dedupe_keep_order(visual),
        "technique": dedupe_keep_order(technique),
        "composition": dedupe_keep_order(composition),
    }


def infer_artist(text: str) -> str:
    patterns = [
        r"(?:画家|作者|作家|艺术家)([\u4e00-\u9fff]{2,4})",
        r"([\u4e00-\u9fff]{2,4})[的之]《",
        r"([\u4e00-\u9fff]{2,4})作品",
        r"([\u4e00-\u9fff]{2,4})所作",
    ]
    stop = {"山水画", "中国画", "传统山", "构图", "空间", "作品", "画面", "南宋画", "北宋画", "明代山", "清代山"}
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1).strip()
            if value and value not in stop:
                return value
    known = ["吴冠中", "顾恺之", "马远", "郭熙", "范宽", "李成", "董源", "巨然", "王维", "黄公望", "倪瓒", "王蒙", "沈周", "董其昌", "石涛", "龚贤"]
    for name in known:
        if name in text:
            return name
    return ""


def build_evidence_query(candidate: Candidate, caption: str, fields: dict[str, Any]) -> str:
    pieces = [candidate.source_stem, caption, fields.get("title", ""), fields.get("artist", ""), fields.get("dynasty", "")]
    for key in ["composition", "technique", "visual_elements"]:
        pieces.extend(fields.get(key) or [])
    return " ".join(item for item in dedupe_keep_order([str(piece).strip() for piece in pieces]) if item)[:500]


def search_evidence(
    query: str,
    source_file: str,
    chunks: list[dict[str, Any]],
    chunks_by_source: dict[str, list[dict[str, Any]]],
    *,
    top_k_source: int,
    top_k_global: int,
) -> list[dict[str, Any]]:
    source_norm = normalize_source(source_file)
    source_chunks = chunks_by_source.get(source_norm, [])
    source_scored = score_chunks(query, source_chunks, source_norm, source_boost=2.0)[:top_k_source]
    global_scored = score_chunks(query, chunks, source_norm, source_boost=0.5)[:top_k_global]
    merged = []
    seen = set()
    for item in source_scored + global_scored:
        if item["chunk_id"] in seen:
            continue
        seen.add(item["chunk_id"])
        merged.append(item)
    merged.sort(key=lambda item: item["score"], reverse=True)
    return merged


def score_chunks(query: str, chunks: list[dict[str, Any]], source_norm: str, source_boost: float) -> list[dict[str, Any]]:
    q_norm = normalize_text(query)
    q_tokens = set(tokenize(query))
    scored = []
    for chunk in chunks:
        token_overlap = len(q_tokens & chunk["_tokens"])
        if token_overlap == 0:
            phrase_bonus = 0
        else:
            phrase_bonus = sum(1 for term in important_terms(query) if term and term in chunk["_search_norm"])
        if token_overlap == 0 and phrase_bonus == 0:
            continue
        score = token_overlap + 3.0 * phrase_bonus
        if chunk["_norm_source_file"] == source_norm:
            score += source_boost
        raw = str(chunk.get("raw_chunk_text") or "")
        if len(raw) < 20:
            score -= 1.0
        scored.append(
            {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk.get("doc_id"),
                "source_file": chunk.get("source_file"),
                "title": chunk.get("title"),
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "score": round(float(score), 4),
                "raw_chunk_text": raw[:700],
                "contextual_prefix": str(chunk.get("contextual_prefix") or "")[:300],
                "quality": chunk.get("quality") or {},
                "_chunk": chunk,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def build_claims_and_evidence(
    fields: dict[str, Any],
    caption: dict[str, Any],
    candidate: Candidate,
    retrieved: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    claims = []
    evidence_links = []
    claim_specs: list[tuple[str, Any, str]] = [
        ("caption_text", caption.get("caption_text", ""), "text"),
        ("title", fields.get("title", ""), "text"),
        ("artist", fields.get("artist", ""), "text"),
        ("dynasty", fields.get("dynasty", ""), "text"),
        ("visual_elements", fields.get("visual_elements", []), "visual_text"),
        ("technique", fields.get("technique", []), "text"),
        ("composition", fields.get("composition", []), "text"),
    ]
    for field, value, support_type in claim_specs:
        if value in ("", [], None):
            claims.append(
                {
                    "claim_id": f"{field}",
                    "field": field,
                    "value": None,
                    "abstain": True,
                    "reason": "no reliable field value inferred from local PDF candidate",
                    "evidence_ids": [],
                    "support_type": support_type,
                }
            )
            continue
        support = label_support(field, value, retrieved)
        evidence_ids = [item["chunk_id"] for item in support if item["support_label"] == "supports"][:3]
        claim = {
            "claim_id": f"{field}",
            "field": field,
            "value": value,
            "abstain": False,
            "evidence_ids": evidence_ids,
            "candidate_evidence_ids": [item["chunk_id"] for item in support[:5]],
            "support_type": support_type,
            "evidence_status": "supports_found" if evidence_ids else "no_chunk_support_silver",
        }
        if field in {"visual_elements", "composition"}:
            claim["visual_bbox"] = candidate.image_bbox
        claims.append(claim)
        evidence_links.append(
            {
                "field": field,
                "value": value,
                "gold_evidence_ids": evidence_ids,
                "candidate_evidence_ids": [item["chunk_id"] for item in support[:5]],
                "support_labels": {item["chunk_id"]: item["support_label"] for item in support[:5]},
            }
        )
    return claims, evidence_links


def label_support(field: str, value: Any, retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = value if isinstance(value, list) else [value]
    values = [normalize_text(str(item)) for item in values if str(item).strip()]
    labeled = []
    for item in retrieved:
        text = normalize_text((item.get("raw_chunk_text") or "") + " " + (item.get("contextual_prefix") or ""))
        hit = sum(1 for val in values if val and val in text)
        if field == "caption_text":
            label = "background_only" if item["score"] > 0 else "irrelevant"
        elif hit > 0:
            label = "supports"
        elif item["score"] >= 5:
            label = "background_only"
        else:
            label = "irrelevant"
        copied = {key: value for key, value in item.items() if key != "_chunk"}
        copied["support_label"] = label
        labeled.append(copied)
    return labeled


def build_oracle_episode(task: dict[str, Any]) -> dict[str, Any]:
    gold = task["gold"]
    actions: list[dict[str, Any]] = [{"action": "crop_image", "bbox": gold["image_bbox"]}]
    if gold.get("caption_bbox"):
        actions.append({"action": "ocr_region", "bbox": gold["caption_bbox"]})
    actions.append({"action": "search_evidence", "query": gold.get("evidence_query", ""), "filters": {"source_file": task["source_file"]}})
    first_chunk = (gold.get("evidence_chunk_ids") or gold.get("candidate_evidence_ids") or [None])[0]
    if first_chunk:
        actions.append({"action": "open_chunk", "chunk_id": first_chunk})
    for claim in gold.get("claims") or []:
        if claim.get("abstain"):
            actions.append({"action": "abstain_claim", "field": claim["field"], "reason": claim.get("reason", "")})
        else:
            actions.append(
                {
                    "action": "write_claim",
                    "field": claim["field"],
                    "value": claim.get("value"),
                    "evidence_ids": claim.get("evidence_ids", []),
                    "visual_bbox": claim.get("visual_bbox"),
                    "confidence": 0.75 if claim.get("evidence_ids") else 0.45,
                }
            )
    card = {
        "image_bbox": gold.get("image_bbox"),
        "caption_bbox": gold.get("caption_bbox"),
        "claims": gold.get("claims") or [],
        "evidence_chunk_ids": gold.get("evidence_chunk_ids") or [],
    }
    actions.append({"action": "write_card", "card": card})
    actions.append({"action": "finish"})
    return {"task_id": task["task_id"], "split": task["split"], "actions": actions}


def episode_to_sft_samples(episode: dict[str, Any], task: dict[str, Any]) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    draft_claims: list[dict[str, Any]] = []
    rows = []
    for step, action in enumerate(episode["actions"]):
        prompt = build_prompt(task, step, history, tool_results, draft_claims)
        images = [task["page_image"]]
        if any(item.get("tool") == "crop_image" for item in tool_results):
            images.append(task["artwork_image"])
        rows.append(
            {
                "task_id": task["task_id"],
                "split": task["split"],
                "step": step,
                "messages": [
                    {"role": "user", "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}]},
                    {"role": "assistant", "content": json.dumps(action, ensure_ascii=False, separators=(",", ":"))},
                ],
                "action": action,
                "history": list(history),
                "tool_results": list(tool_results),
                "draft_claims": list(draft_claims),
                "images": images,
                "prompt_text": prompt,
            }
        )
        update_sft_state(task, action, history, tool_results, draft_claims)
    return rows


def build_prompt(task: dict[str, Any], step: int, history: list[dict[str, Any]], tool_results: list[dict[str, Any]], draft_claims: list[dict[str, Any]]) -> str:
    return (
        "你是证据约束的多模态主动取证 VLM agent。根据页面图像、工具返回和当前 claim 草稿，输出下一步 JSON action。\n"
        f"任务：{task.get('goal')}\n"
        f"task_id：{task.get('task_id')}\n"
        f"source_file：{task.get('source_file')}\n"
        f"step：{step}\n"
        "可用动作：crop_image(bbox), ocr_region(bbox), search_evidence(query, filters), open_chunk(chunk_id), "
        "write_claim(field, value, evidence_ids, visual_bbox, confidence), abstain_claim(field, reason), write_card(card), finish。\n"
        f"历史动作：{json.dumps(history[-6:], ensure_ascii=False)}\n"
        f"工具返回：{json.dumps(tool_results[-4:], ensure_ascii=False)}\n"
        f"当前 claims：{json.dumps(draft_claims, ensure_ascii=False)}\n"
        "只输出一个 JSON 对象；没有证据支持的字段应使用 abstain_claim。"
    )


def update_sft_state(task: dict[str, Any], action: dict[str, Any], history: list[dict[str, Any]], tool_results: list[dict[str, Any]], draft_claims: list[dict[str, Any]]) -> None:
    name = action.get("action")
    if name == "crop_image":
        tool_results.append({"tool": "crop_image", "bbox": action.get("bbox"), "crop_path": task.get("artwork_image")})
    elif name == "ocr_region":
        tool_results.append({"tool": "ocr_region", "bbox": action.get("bbox"), "text": task.get("gold", {}).get("caption_text", "")})
    elif name == "search_evidence":
        tool_results.append(
            {
                "tool": "search_evidence",
                "query": action.get("query"),
                "results": task.get("gold", {}).get("candidate_evidence_ids", [])[:5],
            }
        )
    elif name == "open_chunk":
        tool_results.append({"tool": "open_chunk", "chunk_id": action.get("chunk_id")})
    elif name == "write_claim":
        draft_claims.append({key: action.get(key) for key in ["field", "value", "evidence_ids", "visual_bbox", "confidence"] if key in action})
    elif name == "abstain_claim":
        draft_claims.append({"field": action.get("field"), "abstain": True, "reason": action.get("reason", "")})
    history.append(action)


def task_to_claim_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for claim in task.get("gold", {}).get("claims", []):
        rows.append(
            {
                "task_id": task["task_id"],
                "split": task["split"],
                "source_file": task["source_file"],
                "page": task["page"],
                **claim,
            }
        )
    return rows


def task_to_evidence_rows(task: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for link in task.get("evidence_links") or []:
        rows.append({"task_id": task["task_id"], "split": task["split"], "source_file": task["source_file"], **link})
    return rows


def write_review_html(path: Path, tasks: list[dict[str, Any]]) -> Path:
    parts = [
        "<html><head><meta charset='utf-8'><title>EvidenceGrounded AgentBench Review</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .task{border:1px solid #ccc;padding:16px;margin:16px 0;} img{max-width:560px;border:1px solid #ddd;} code{white-space:pre-wrap;display:block;background:#f7f7f7;padding:8px;}</style>",
        "</head><body><h1>EvidenceGrounded AgentBench Review</h1>",
    ]
    for task in tasks:
        gold = task.get("gold", {})
        parts.append("<div class='task'>")
        parts.append(f"<h2>{task['task_id']} [{task['split']}]</h2>")
        parts.append(f"<p>{task['source_file']} page {task['page']}</p>")
        parts.append(f"<img src='file://{task['overlay_image']}' />")
        parts.append("<code>" + html_escape(json.dumps(gold, ensure_ascii=False, indent=2)[:4000]) + "</code>")
        parts.append("</div>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def build_summary(
    args: argparse.Namespace,
    output_dir: Path,
    evidence_snapshot: Path,
    all_candidates: list[Candidate],
    selected: list[Candidate],
    tasks: list[dict[str, Any]],
    claim_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    sft_rows_by_split: dict[str, list[dict[str, Any]]],
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    split_counter = Counter(task["split"] for task in tasks)
    source_counter = Counter(task["source_file"] for task in tasks)
    claim_counter = Counter(row["field"] for row in claim_rows)
    non_abstain = [row for row in claim_rows if not row.get("abstain")]
    with_evidence = [row for row in non_abstain if row.get("evidence_ids")]
    support_status = Counter(row.get("evidence_status", "none") for row in non_abstain)
    return {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S CST"),
        "output_dir": str(output_dir),
        "dataset_name": "EvidenceGrounded-AgentBench",
        "version": args.version,
        "source": "raw_pdf_image_blocks_plus_legacy_milvus_evidence_store",
        "raw_pdfs": len(list(Path(args.raw_pdfs_dir).glob("*.pdf"))),
        "evidence_snapshot": str(evidence_snapshot),
        "all_pdf_image_candidates": len(all_candidates),
        "selected_candidates": len(selected),
        "tasks": len(tasks),
        "splits": dict(split_counter),
        "unique_sources": len(source_counter),
        "top_sources": dict(source_counter.most_common(12)),
        "claims": len(claim_rows),
        "claim_fields": dict(claim_counter),
        "non_abstain_claims": len(non_abstain),
        "claims_with_chunk_evidence": len(with_evidence),
        "claim_evidence_coverage": len(with_evidence) / max(1, len(non_abstain)),
        "support_status": dict(support_status),
        "evidence_link_rows": len(evidence_rows),
        "sft_rows": {split: len(rows) for split, rows in sft_rows_by_split.items()},
        "auto_label": True,
        "needs_review": True,
        "citation_level": "chunk",
        "limitations": [
            "Gold is silver auto-label and must be manually reviewed before final val/test claims.",
            "Evidence store is legacy Milvus migration; many chunks lack reliable page_start/page_end.",
            "Current evidence labels are retrieval/heuristic support labels, not final human verified labels.",
        ],
        "errors": errors[:30],
        "error_count": len(errors),
    }


def build_manifest(args: argparse.Namespace, output_dir: Path, evidence_snapshot: Path, summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "build_time": summary["created_at"],
        "builder": "scripts/build_local_evidence_agentbench.py",
        "args": vars(args),
        "outputs": {
            "tasks_all": str(output_dir / "tasks_all.jsonl"),
            "train_tasks": str(output_dir / "train_tasks.jsonl"),
            "val_tasks": str(output_dir / "val_tasks.jsonl"),
            "test_tasks": str(output_dir / "test_tasks.jsonl"),
            "claim_gold": str(output_dir / "claim_gold.jsonl"),
            "evidence_links": str(output_dir / "evidence_links.jsonl"),
            "sft": str(output_dir / "sft"),
            "episodes": str(output_dir / "episodes" / "oracle_episodes.jsonl"),
            "review_html": str(output_dir / "review" / "review.html"),
        },
        "evidence_snapshot": str(evidence_snapshot),
        "summary": summary,
    }


def write_report(path: Path, summary: dict[str, Any], review_path: Path) -> None:
    lines = [
        "# EvidenceGrounded-AgentBench v0.1-local-evidence 构建报告",
        "",
        f"- 生成时间：{summary['created_at']}",
        f"- 输出目录：`{summary['output_dir']}`",
        f"- evidence snapshot：`{summary['evidence_snapshot']}`",
        "",
        "## 数据规模",
        "",
        f"- raw PDFs：{summary['raw_pdfs']}",
        f"- PDF image candidates：{summary['all_pdf_image_candidates']}",
        f"- selected tasks：{summary['tasks']}",
        f"- splits：`{summary['splits']}`",
        f"- unique sources：{summary['unique_sources']}",
        "",
        "## Claim 与证据",
        "",
        f"- claims：{summary['claims']}",
        f"- non-abstain claims：{summary['non_abstain_claims']}",
        f"- claims_with_chunk_evidence：{summary['claims_with_chunk_evidence']}",
        f"- claim_evidence_coverage：{summary['claim_evidence_coverage']:.4f}",
        f"- support_status：`{summary['support_status']}`",
        "",
        "## SFT",
        "",
        f"- sft_rows：`{summary['sft_rows']}`",
        "",
        "## 人工审核",
        "",
        f"- review HTML：`{review_path}`",
        "",
        "## 限制",
        "",
    ]
    lines.extend(f"- {item}" for item in summary["limitations"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_bbox(values: list[float], rect: fitz.Rect) -> list[int]:
    x0, y0, x1, y1 = values
    return [
        int(round(x0 / rect.width * 1000)),
        int(round(y0 / rect.height * 1000)),
        int(round(x1 / rect.width * 1000)),
        int(round(y1 / rect.height * 1000)),
    ]


def bbox_area(bbox: list[int]) -> int:
    if not bbox or len(bbox) != 4:
        return 0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def draw_overlay(page_image: Path, image_bbox: list[int], caption_bbox: list[int] | None, out: Path) -> Path:
    image = Image.open(page_image).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw_norm_box(draw, image.size, image_bbox, "red", "image_bbox")
    if caption_bbox:
        draw_norm_box(draw, image.size, caption_bbox, "blue", "caption_bbox")
    image.save(out, quality=92)
    return out


def draw_norm_box(draw: ImageDraw.ImageDraw, size: tuple[int, int], bbox: list[int], color: str, label: str) -> None:
    width, height = size
    x0, y0, x1, y1 = bbox
    box = [x0 / 1000 * width, y0 / 1000 * height, x1 / 1000 * width, y1 / 1000 * height]
    draw.rectangle(box, outline=color, width=4)
    draw.text((box[0], max(0, box[1] - 16)), label, fill=color)


def normalize_source(value: str) -> str:
    return normalize_text(Path(value).name.replace(".pdf", ""))


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value).lower())


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: str) -> list[str]:
    norm = normalize_text(value)
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", norm)
    tokens = []
    for word in words:
        tokens.append(word)
        if re.fullmatch(r"[\u4e00-\u9fff]+", word) and len(word) > 2:
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
    return tokens


def important_terms(value: str) -> list[str]:
    terms = re.findall(r"《([^》]{1,32})》|([\u4e00-\u9fff]{2,8})", value)
    flat = []
    for a, b in terms:
        item = a or b
        if item and item not in {"中国", "山水", "山水画", "作品", "图像", "论文"}:
            flat.append(normalize_text(item))
    return dedupe_keep_order(flat)[:20]


def dedupe_keep_order(values: list[Any]) -> list[Any]:
    result = []
    seen = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", value)
    return value[:120] or "untitled"


def html_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
