#!/usr/bin/env python3
"""Pass 2: crop each shelving unit from a CCTV frame and get structured shelf metrics.

Shelf positions come from the camera's saved layout (see detect_shelves.py). If the
camera has no layout yet, this frame is used to detect one, which is then saved and
reused for every later frame from that camera.

Setup:
    pip install google-genai pydantic opencv-python
    export GEMINI_API_KEY=...

Usage:
    python shelf_metrics.py frame.jpg
    python shelf_metrics.py frame.jpg --model gemini-2.5-pro --out result.json --save-crops crops/
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from google.genai import errors
from pydantic import BaseModel, Field

from detect_shelves import (LAYOUTS_DIR, CameraLayout, Shelf, camera_id_for, detect_layout,
                            load_layout, save_layout, to_pixels)
from gemini_util import DEFAULT_MODEL, generate, make_client

CROP_PADDING = 0.04  # fraction of the unit's size added on each side
MIN_LONG_SIDE = 1024  # upscale small crops to at least this many pixels on the long side
MAX_UPSCALE = 4.0
HIDDEN_GREY = 128  # fill for parts of a crop covered by a nearer unit

PROMPT = """This image is a crop of one shelving unit ("{label}") from a retail CCTV frame,
enlarged for detail. {levels_text} Report metrics for this unit and for each level.
Ignore anything outside the unit that belongs to neighbouring fixtures.
Be conservative: if the image quality, angle, or occlusion (people, carts, glare)
prevents a reliable estimate, lower the confidence and say why in `notes`. Do not
invent products you cannot see. Percentages are 0-100."""


class LevelMetrics(BaseModel):
    level: int = Field(description="Level number as labelled on the image (L1 = 1 = top)")
    fill_percent: float = Field(description="How full this level is, 0-100")
    empty_gap_count: int = Field(description="Number of visible out-of-stock gaps")
    product_description: str = Field(description="What is stocked on this level, briefly")


class ShelfMetrics(BaseModel):
    shelf_visible: bool = Field(description="False if the unit is out of view or fully blocked")
    fill_percent: float = Field(description="Overall fullness of this unit, 0-100")
    out_of_stock_gap_count: int = Field(description="Total visible empty gaps")
    stock_status: Literal["well_stocked", "partially_stocked", "low_stock", "empty", "unknown"]
    estimated_facings: int = Field(description="Approximate number of product facings visible")
    distinct_products_estimate: int = Field(description="Approximate number of distinct SKUs")
    tidiness_score: int = Field(description="1 (messy, fallen/misplaced items) to 5 (neat, faced)")
    price_labels_visible: bool
    people_occluding_shelf: bool
    levels: list[LevelMetrics]
    confidence: float = Field(description="Overall confidence in these estimates, 0-1")
    notes: str = Field(description="Caveats, image quality issues, anything notable")


class ShelfResult(BaseModel):
    shelf_id: str
    label: str
    metrics: ShelfMetrics | None
    error: str | None = None


class FrameMetrics(BaseModel):
    frame: str
    camera_id: str
    layout: str
    model: str
    overall_fill_percent: float | None = Field(
        description="Area-weighted fill across visible units, None if none were readable")
    shelves: list[ShelfResult]


def crop_shelf(image: np.ndarray, shelf: Shelf, front_boxes: list[list[int]]) -> np.ndarray:
    """Crop the unit with padding, grey out nearer units in front of it, upscale it,
    and outline its levels as L1, L2, ..."""
    h, w = image.shape[:2]
    if front_boxes:
        image = image.copy()
        for front in front_boxes:
            fx0, fy0, fx1, fy1 = to_pixels(front, w, h)
            image[fy0:fy1, fx0:fx1] = HIDDEN_GREY
    x0, y0, x1, y1 = to_pixels(shelf.box_2d, w, h)
    pad_x, pad_y = round((x1 - x0) * CROP_PADDING), round((y1 - y0) * CROP_PADDING)
    cx0, cy0 = max(x0 - pad_x, 0), max(y0 - pad_y, 0)
    cx1, cy1 = min(x1 + pad_x, w), min(y1 + pad_y, h)
    crop = image[cy0:cy1, cx0:cx1]

    scale = min(max(MIN_LONG_SIDE / max(crop.shape[:2]), 1.0), MAX_UPSCALE)
    if scale > 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        crop = crop.copy()

    # Draw after upscaling so outlines stay thin relative to the products.
    for level in shelf.levels:
        lx0, ly0, lx1, ly1 = to_pixels(level.box_2d, w, h)
        p0 = (round((lx0 - cx0) * scale), round((ly0 - cy0) * scale))
        p1 = (round((lx1 - cx0) * scale), round((ly1 - cy0) * scale))
        cv2.rectangle(crop, p0, p1, (0, 255, 255), 2)
        cv2.putText(crop, f"L{level.level}", (p0[0] + 4, p0[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(crop, f"L{level.level}", (p0[0] + 4, p0[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return crop


def shelf_prompt(shelf: Shelf) -> str:
    hidden_text = (" Flat grey areas are other fixtures standing in front of this unit: ignore"
                   " them, and do not count them as empty shelf space." if shelf.hidden_by else "")
    if shelf.levels:
        labels = ", ".join(f"L{l.level}" for l in shelf.levels)
        levels_text = (f"Yellow outlines mark its {len(shelf.levels)} shelf levels, "
                       f"labelled {labels} from top to bottom; report one entry per label.")
    else:
        levels_text = "No levels are marked; split it into levels yourself, 1 = top."
    return PROMPT.format(label=shelf.label, levels_text=levels_text + hidden_text)


def analyse_shelf(client, model: str, image: np.ndarray, shelf: Shelf,
                  front_boxes: list[list[int]], crops_dir: Path | None,
                  frame_stem: str) -> ShelfResult:
    crop = crop_shelf(image, shelf, front_boxes)
    if crops_dir:
        cv2.imwrite(str(crops_dir / f"{frame_stem}_{shelf.shelf_id}.jpg"), crop)
    ok, jpeg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        return ShelfResult(shelf_id=shelf.shelf_id, label=shelf.label, metrics=None,
                           error="Could not encode crop")
    try:
        metrics = generate(client, model, jpeg.tobytes(), "image/jpeg",
                           shelf_prompt(shelf), ShelfMetrics)
    except (errors.APIError, RuntimeError) as e:
        print(f"{shelf.shelf_id}: {e}", file=sys.stderr)
        return ShelfResult(shelf_id=shelf.shelf_id, label=shelf.label, metrics=None, error=str(e))
    return ShelfResult(shelf_id=shelf.shelf_id, label=shelf.label, metrics=metrics)


def overall_fill(layout: CameraLayout, results: list[ShelfResult]) -> float | None:
    """Average unit fill, weighted by each unit's area in the frame."""
    boxes = {s.shelf_id: s.box_2d for s in layout.shelves}
    total = weight = 0.0
    for r in results:
        if r.metrics and r.metrics.shelf_visible:
            y0, x0, y1, x1 = boxes[r.shelf_id]
            area = (y1 - y0) * (x1 - x0)
            total += r.metrics.fill_percent * area
            weight += area
    return round(total / weight, 1) if weight else None


def analyse(image_path: str, model: str, camera_id: str, layouts_dir: Path,
            workers: int, crops_dir: Path | None) -> FrameMetrics:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Can't read {image_path}")
    client = make_client()

    layout = load_layout(camera_id, layouts_dir)
    if layout is None:
        print(f"No layout for camera {camera_id!r}; detecting shelves on this frame.",
              file=sys.stderr)
        layout = detect_layout(client, image_path, camera_id, model)
        print(f"Saved {len(layout.shelves)} units to {save_layout(layout, layouts_dir)}",
              file=sys.stderr)

    h, w = image.shape[:2]
    if (w, h) != (layout.frame_width, layout.frame_height):
        print(f"Warning: frame is {w}x{h} but the layout was made on "
              f"{layout.frame_width}x{layout.frame_height}; boxes may be off.", file=sys.stderr)

    if crops_dir:
        crops_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda s: analyse_shelf(client, model, image, s, layout.front_boxes(s), crops_dir, stem),
            layout.shelves))

    return FrameMetrics(
        frame=image_path, camera_id=camera_id, layout=str(layouts_dir / f"{camera_id}.json"),
        model=model, overall_fill_percent=overall_fill(layout, results), shelves=results,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="Path to a CCTV frame (jpg/png/webp)")
    parser.add_argument("--camera", help="Camera ID (default: video name from the frame filename)")
    parser.add_argument("--layouts", type=Path, default=LAYOUTS_DIR)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--workers", type=int, default=4, help="Parallel Gemini calls")
    parser.add_argument("--save-crops", type=Path, help="Write the crops sent to Gemini here")
    parser.add_argument("--out", help="Also write the JSON result to this file")
    args = parser.parse_args()

    metrics = analyse(args.image, args.model, args.camera or camera_id_for(args.image),
                      args.layouts, args.workers, args.save_crops)
    output = json.dumps(metrics.model_dump(), indent=2)
    print(output)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
