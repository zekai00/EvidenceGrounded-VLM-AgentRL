"""Region proposal helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image


def propose_regions(task: dict[str, Any], top_k: int = 8, include_gold: bool = False) -> list[dict[str, Any]]:
    if task.get("region_candidates"):
        return public_regions(task["region_candidates"], top_k)

    page_image = Path(task["page_image"])
    with Image.open(page_image) as image:
        width, height = image.size
    regions: list[dict[str, Any]] = []
    if include_gold and task.get("gold", {}).get("image_bbox"):
        regions.append(
            {
                "region_id": "r_gold",
                "bbox": task["gold"]["image_bbox"],
                "type": "debug_gold",
                "reason": "gold region exposed only for smoke/debug",
            }
        )
    cols, rows = 3, 4
    for row in range(rows):
        for col in range(cols):
            x1 = int(width * col / cols)
            x2 = int(width * (col + 1) / cols)
            y1 = int(height * row / rows)
            y2 = int(height * (row + 1) / rows)
            regions.append(
                {
                    "region_id": f"r_grid_{row}_{col}",
                    "bbox": [x1, y1, x2, y2],
                    "type": "grid",
                    "reason": "regular page grid fallback",
                }
            )
    return regions[: max(1, int(top_k))]


def public_regions(regions: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Return model-visible region proposals without hidden labels."""

    hidden_keys = {
        "is_target",
        "target_iou",
        "gold_iou",
        "source_task_id",
        "source_gold_bbox",
        "debug_reason",
    }
    visible: list[dict[str, Any]] = []
    for item in regions[: max(1, int(top_k))]:
        visible.append({key: value for key, value in item.items() if key not in hidden_keys})
    return visible
