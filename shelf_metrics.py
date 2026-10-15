#!/usr/bin/env python3
"""Pass 2: straighten each shelving unit from a CCTV frame and get structured shelf metrics.

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

from detect_shelves import (LAYOUTS_DIR, CameraLayout, Polygon, Shelf, camera_id_for,
                            detect_layout, load_layout, save_layout, to_pixels)
from gemini_util import DEFAULT_MODEL, generate, make_client

CROP_PADDING = 0.04  # fraction of the unit's size added on each side
MIN_LONG_SIDE = 1024  # scale straightened crops to at least this many pixels on the long side
MAX_UPSCALE = 4.0
HIDDEN_GREY = 128  # fill for parts of a crop covered by a nearer unit or outside the frame

PROMPT = """This image is one shelving unit ("{label}") cut from a retail CCTV frame,
perspective-corrected so its front faces the viewer, and enlarged for detail; expect
some stretching and blur from that. {levels_text} Report metrics for this unit and for each level.
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


def straighten_shelf(image: np.ndarray, shelf: Shelf,
                     front_polygons: list[Polygon]) -> np.ndarray:
    """Warp the unit's 4-corner face to an upright rectangle, grey out nearer units in
    front of it, and outline its levels as L1, L2, ..."""
    h, w = image.shape[:2]
    if front_polygons:
        image = image.copy()
        for front in front_polygons:
            cv2.fillPoly(image, [to_pixels(front, w, h).round().astype(np.int32)],
                         (HIDDEN_GREY,) * 3)

    quad = to_pixels(shelf.polygon, w, h)
    centre = quad.mean(axis=0)
    quad = centre + (quad - centre) * (1 + 2 * CROP_PADDING)
    tl, tr, br, bl = quad
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    scale = min(max(MIN_LONG_SIDE / max(width, height), 1.0), MAX_UPSCALE)
    out_w, out_h = max(round(width * scale), 1), max(round(height * scale), 1)
    warp = cv2.getPerspectiveTransform(
        quad.astype(np.float32), np.float32([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]]))
    out = cv2.warpPerspective(image, warp, (out_w, out_h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(HIDDEN_GREY,) * 3)

    # Draw after warping so outlines stay thin relative to the products.
    for level in shelf.levels:
        pts = cv2.perspectiveTransform(to_pixels(level.polygon, w, h).reshape(-1, 1, 2), warp)
        pts = pts.reshape(-1, 2).round().astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
        org = (int(pts[0][0]) + 4, int(pts[0][1]) + 22)
        cv2.putText(out, f"L{level.level}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, f"L{level.level}", org, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return out


def shelf_prompt(shelf: Shelf) -> str:
    hidden_text = (" Flat grey areas are other fixtures standing in front of this unit, or"
                   " outside the camera's view: ignore them, and do not count them as empty"
                   " shelf space.")
    if shelf.levels:
        labels = ", ".join(f"L{l.level}" for l in shelf.levels)
        levels_text = (f"Yellow outlines mark its {len(shelf.levels)} shelf levels, "
                       f"labelled {labels} from top to bottom; report one entry per label.")
    else:
        levels_text = "No levels are marked; split it into levels yourself, 1 = top."
    return PROMPT.format(label=shelf.label, levels_text=levels_text + hidden_text)


def analyse_shelf(client, model: str, image: np.ndarray, shelf: Shelf,
                  front_polygons: list[Polygon], crops_dir: Path | None,
                  frame_stem: str) -> ShelfResult:
    crop = straighten_shelf(image, shelf, front_polygons)
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
    areas = {s.shelf_id: cv2.contourArea(to_pixels(s.polygon, 1000, 1000))
             for s in layout.shelves}
    total = weight = 0.0
    for r in results:
        if r.metrics and r.metrics.shelf_visible:
            area = areas[r.shelf_id]
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
            lambda s: analyse_shelf(client, model, image, s, layout.front_polygons(s), crops_dir, stem),
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
