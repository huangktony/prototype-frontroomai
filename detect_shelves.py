#!/usr/bin/env python3
"""Pass 1: find the shelving units and their levels in a camera view, once per camera.

The cameras are fixed, so the layout found on one reference frame is saved to
layouts/<camera_id>.json and reused by shelf_metrics.py for every later frame from
that camera. Each unit and shelf level is a 4-corner polygon of its front face, so
angled shelves can be straightened in pass 2. A layouts/<camera_id>.jpg overlay is
written alongside for checking the shapes by eye; edit the JSON by hand to fix them.

Usage:
    python detect_shelves.py test_frames/4WXZTF7EWL0_00010s.jpg
    python detect_shelves.py frame.jpg --camera aisle3 --force
"""
import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, Field

from gemini_util import DEFAULT_MODEL, generate, make_client, read_image

LAYOUTS_DIR = Path("layouts")
MIN_AREA = 200  # in 0-1000 units squared; smaller polygons are treated as noise
MIN_VISIBLE = 0.15  # drop units with less than this fraction left after nearer units
MIN_HIDDEN = 0.005  # a nearer unit covering less than this fraction doesn't count
GRID = 1000  # resolution of the normalised masks used to compare polygons

PROMPT = """You are mapping the shelving in a fixed CCTV camera view of a retail store.
This map will be reused for every later frame from this camera, so describe the
fixtures themselves, not the products or people currently in view.

Find every shelving unit (bay, gondola section, wall rack, fridge, display stand)
whose product-facing side is visible enough to judge how stocked it is. Skip units
that are mostly out of frame, seen only edge-on, or too far away to make out products.

For each unit give a polygon of exactly 4 corners tracing its product-facing front
face as it appears in perspective (a shelf seen at an angle is a slanted
quadrilateral, not an upright rectangle), and inside it one 4-corner polygon per
shelf level from top to bottom, following the slope of that shelf's edge. Give
corners as [y, x] normalised to 0-1000, in the order top-left, top-right,
bottom-right, bottom-left. If a fixture hides the middle of a unit and both ends
are visible, return each visible part as its own unit.

Also give each unit a depth rank: 1 for the units nearest the camera, then 2, 3, ...
for units further back. Units at about the same distance share a rank. Trace each
unit's whole face even where a nearer unit stands in front of it; the depth ranks
are used to work out what is hidden.
People and carts move, so ignore them: trace the shelf as if they were not there."""

Polygon = list[list[int]]  # 4 corners, each [y, x] normalised 0-1000: TL, TR, BR, BL
POLYGON_HELP = "4 corners [[y, x], ...] normalised 0-1000: top-left, top-right, bottom-right, bottom-left"


# --- What Gemini returns in pass 1 ---

class DetectedLevel(BaseModel):
    polygon: Polygon = Field(description=POLYGON_HELP)


class DetectedUnit(BaseModel):
    label: str = Field(description="Short name, e.g. 'left wall drinks fridge'")
    polygon: Polygon = Field(description=POLYGON_HELP)
    levels: list[DetectedLevel] = Field(description="Shelf levels, top to bottom")
    depth: int = Field(description="1 = nearest the camera; higher = further back")


class Detection(BaseModel):
    units: list[DetectedUnit]
    notes: str = Field(description="Anything that made the shelving hard to map")


# --- What we store per camera ---

class Level(BaseModel):
    level_id: str  # e.g. "S1-L2"
    level: int  # 1 = top
    polygon: Polygon


class Shelf(BaseModel):
    shelf_id: str  # e.g. "S1", numbered left to right
    label: str
    polygon: Polygon  # whole front face, including parts hidden by nearer units
    depth: int = 1  # 1 = nearest the camera
    hidden_by: list[str] = []  # nearer units standing in front of part of this one
    levels: list[Level]


class CameraLayout(BaseModel):
    camera_id: str
    reference_frame: str
    frame_width: int
    frame_height: int
    model: str
    created_at: str
    notes: str
    shelves: list[Shelf]

    def front_polygons(self, shelf: Shelf) -> list[Polygon]:
        """Polygons of the nearer units standing in front of part of `shelf`."""
        return [s.polygon for s in self.shelves if s.shelf_id in shelf.hidden_by]


def camera_id_for(frame_path: str) -> str:
    """extract_frames.py names frames <video>_<seconds>s.jpg; the video is the camera."""
    return Path(frame_path).stem.rsplit("_", 1)[0]


def layout_path(camera_id: str, layouts_dir: Path = LAYOUTS_DIR) -> Path:
    return layouts_dir / f"{camera_id}.json"


def order_corners(xy: np.ndarray) -> np.ndarray:
    """Sort 4 (x, y) points clockwise starting from the top-left one."""
    centre = xy.mean(axis=0)
    xy = xy[np.argsort(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0]))]
    return np.roll(xy, -int(np.argmin(xy.sum(axis=1))), axis=0)


def clean_polygon(polygon: Polygon) -> Polygon | None:
    """Clamp to 0-1000, force 4 corners in TL/TR/BR/BL order, and drop tiny shapes.

    Anything that isn't a convex quadrilateral is replaced by its tightest
    rotated rectangle, so pass 2 can always straighten it.
    """
    xy = np.array([[x, y] for p in polygon if len(p) == 2 for y, x in [p]], np.float32)
    if len(xy) < 3:
        return None
    xy = np.clip(xy, 0, 1000)
    quad = order_corners(xy) if len(xy) == 4 else None
    if quad is None or not cv2.isContourConvex(quad):
        quad = order_corners(np.clip(cv2.boxPoints(cv2.minAreaRect(xy)), 0, 1000))
    if cv2.contourArea(quad) < MIN_AREA:
        return None
    return [[round(float(y)), round(float(x))] for x, y in quad]


def to_pixels(polygon: Polygon, width: int, height: int) -> np.ndarray:
    """[[y, x], ...] in 0-1000 -> float32 array of (x, y) pixel coordinates."""
    return np.array([[x * width / 1000, y * height / 1000] for y, x in polygon], np.float32)


def polygon_mask(polygon: Polygon, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(mask, [to_pixels(polygon, width, height).round().astype(np.int32)], 255)
    return mask


def visible_mask(polygon: Polygon, front_polygons: list[Polygon],
                 width: int, height: int) -> np.ndarray:
    """Pixel mask of `polygon` minus the nearer polygons standing in front of it."""
    mask = polygon_mask(polygon, width, height)
    for front in front_polygons:
        mask[polygon_mask(front, width, height) > 0] = 0
    return mask


def build_layout(detection: Detection, camera_id: str, frame: str,
                 width: int, height: int, model: str) -> CameraLayout:
    units = []
    for unit in detection.units:
        levels = [p for p in (clean_polygon(l.polygon) for l in unit.levels) if p]
        levels.sort(key=lambda p: sum(y for y, _ in p))  # top to bottom
        polygon = clean_polygon(unit.polygon)
        if polygon is None and levels:
            # No usable outline: span from the top level's top edge to the bottom level's.
            polygon = clean_polygon([levels[0][0], levels[0][1], levels[-1][2], levels[-1][3]])
        if polygon is None:
            continue
        units.append((max(unit.depth, 1), unit.label, polygon, levels))

    # Work out what nearer units hide, on a normalised grid so it doesn't depend on frame size.
    masks = [polygon_mask(u[2], GRID, GRID) for u in units]
    kept = []
    for i, (depth, label, polygon, levels) in enumerate(units):
        area = np.count_nonzero(masks[i])
        hidden = np.zeros_like(masks[i])
        fronts = []
        for j, other in enumerate(units):
            if other[0] < depth:
                overlap = cv2.bitwise_and(masks[i], masks[j])
                if np.count_nonzero(overlap) > MIN_HIDDEN * area:
                    fronts.append(j)
                    hidden |= overlap
        if area == 0 or 1 - np.count_nonzero(hidden) / area < MIN_VISIBLE:
            continue  # (almost) entirely behind nearer units
        kept.append((i, depth, label, polygon, levels, fronts))

    # Left to right by the leftmost corner, then top to bottom.
    kept.sort(key=lambda u: (min(x for _, x in u[3]), min(y for y, _ in u[3])))
    ids = {u[0]: f"S{n}" for n, u in enumerate(kept, 1)}
    shelves = []
    for i, depth, label, polygon, levels, fronts in kept:
        shelf_id = ids[i]
        shelves.append(Shelf(
            shelf_id=shelf_id, label=label, polygon=polygon, depth=depth,
            hidden_by=[ids[j] for j in fronts if j in ids],
            levels=[Level(level_id=f"{shelf_id}-L{n}", level=n, polygon=p)
                    for n, p in enumerate(levels, 1)],
        ))

    return CameraLayout(
        camera_id=camera_id, reference_frame=frame, frame_width=width, frame_height=height,
        model=model, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=detection.notes, shelves=shelves,
    )


def draw_overlay(image: np.ndarray, layout: CameraLayout) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for shelf in layout.shelves:
        pts = to_pixels(shelf.polygon, w, h).round().astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(out, f"{shelf.shelf_id} {shelf.label}", (int(pts[0][0]) + 3, max(int(pts[0][1]) - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        for level in shelf.levels:
            lpts = to_pixels(level.polygon, w, h).round().astype(np.int32)
            cv2.polylines(out, [lpts], True, (0, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(out, f"L{level.level}", (int(lpts[0][0]) + 3, int(lpts[0][1]) + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
    return out


def detect_layout(client, image_path: str, camera_id: str, model: str) -> CameraLayout:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Can't read {image_path}")
    data, mime = read_image(image_path)
    detection = generate(client, model, data, mime, PROMPT, Detection)
    h, w = image.shape[:2]
    return build_layout(detection, camera_id, image_path, w, h, model)


def save_layout(layout: CameraLayout, layouts_dir: Path = LAYOUTS_DIR) -> Path:
    layouts_dir.mkdir(parents=True, exist_ok=True)
    path = layout_path(layout.camera_id, layouts_dir)
    path.write_text(layout.model_dump_json(indent=2) + "\n")
    image = cv2.imread(layout.reference_frame)
    if image is not None:
        cv2.imwrite(str(path.with_suffix(".jpg")), draw_overlay(image, layout))
    return path


def load_layout(camera_id: str, layouts_dir: Path = LAYOUTS_DIR) -> CameraLayout | None:
    path = layout_path(camera_id, layouts_dir)
    if not path.exists():
        return None
    return CameraLayout.model_validate_json(path.read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="Reference frame from the camera (jpg/png/webp)")
    parser.add_argument("--camera", help="Camera ID (default: video name from the frame filename)")
    parser.add_argument("--layouts", type=Path, default=LAYOUTS_DIR)
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--force", action="store_true", help="Replace an existing layout")
    args = parser.parse_args()

    camera_id = args.camera or camera_id_for(args.image)
    path = layout_path(camera_id, args.layouts)
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to re-detect.", file=sys.stderr)
        return 1

    layout = detect_layout(make_client(), args.image, camera_id, args.model)
    path = save_layout(layout, args.layouts)
    levels = sum(len(s.levels) for s in layout.shelves)
    print(f"Found {len(layout.shelves)} shelving units, {levels} levels -> {path} "
          f"(overlay: {path.with_suffix('.jpg')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
