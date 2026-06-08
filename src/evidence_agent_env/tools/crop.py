"""Crop tools for rendered PDF page images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from ..actions import coerce_bbox


def image_size(path: str | Path) -> dict[str, int]:
    with Image.open(path) as image:
        width, height = image.size
    return {"width": width, "height": height}


def crop_image(page_image: str | Path, bbox: Any, output_path: str | Path) -> dict[str, Any]:
    box = coerce_bbox(bbox)
    if box is None:
        raise ValueError("invalid bbox")
    page_image = Path(page_image)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(page_image) as image:
        width, height = image.size
        x1, y1, x2, y2 = box
        clipped = [max(0, x1), max(0, y1), min(width, x2), min(height, y2)]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError("bbox outside image")
        crop = image.crop(tuple(clipped))
        crop.save(output_path)
    return {"bbox": clipped, "crop_path": str(output_path), "page_image": str(page_image)}
