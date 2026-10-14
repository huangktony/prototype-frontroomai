#!/usr/bin/env python3
"""Pass 1: find the shelving units and their levels in a camera view, once per camera.

The cameras are fixed, so the layout found on one reference frame is saved to
layouts/<camera_id>.json and reused by shelf_metrics.py for every later frame from
that camera. A layouts/<camera_id>.jpg overlay is written alongside for checking
the boxes by eye; edit the JSON by hand to fix them.

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
MIN_BOX_SIZE = 10  # in 0-1000 units; smaller boxes are treated as noise
TRIM_TOLERANCE = 30  # a nearer unit within this of a full side counts as covering it

PROMPT = """You are mapping the shelving in a fixed CCTV camera view of a retail store.
This map will be reused for every later frame from this camera, so describe the
fixtures themselves, not the products or people currently in view.

Find every shelving unit (bay, gondola section, wall rack, fridge, display stand)
whose product-facing side is visible enough to judge how stocked it is. Skip units
that are mostly out of frame, seen only edge-on, or too far away to make out products.
For each unit give its bounding box and, inside it, one bounding box per shelf level
from top to bottom. Boxes are [ymin, xmin, ymax, xmax] normalised to 0-1000.

Box only the part of each unit that is actually visible. Where another shelf, fridge,
counter or wall stands in front of a unit, end its box (and its level boxes) at the
edge of that fixture rather than extending it behind, so boxes of different units
should overlap as little as possible and their sizes will differ. If a fixture hides
the middle of a unit and both ends are visible, return each visible part as its own
unit. Levels hidden entirely behind another fixture are left out.
Also give each unit a depth rank: 1 for the units nearest the camera, then 2, 3, ...
for units further back. Units at about the same distance share a rank.
People and carts move, so ignore them: box the shelf as if they were not there."""


# --- What Gemini returns in pass 1 ---

class DetectedLevel(BaseModel):
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax], normalised 0-1000")


class DetectedUnit(BaseModel):
    label: str = Field(description="Short name, e.g. 'left wall drinks fridge'")
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax], normalised 0-1000")
    levels: list[DetectedLevel] = Field(description="Shelf levels, top to bottom")
    depth: int = Field(description="1 = nearest the camera; higher = further back")


class Detection(BaseModel):
    units: list[DetectedUnit]
    notes: str = Field(description="Anything that made the shelving hard to map")


# --- What we store per camera ---

class Level(BaseModel):
    level_id: str  # e.g. "S1-L2"
    level: int  # 1 = top
    box_2d: list[int]  # [ymin, xmin, ymax, xmax], 0-1000


class Shelf(BaseModel):
    shelf_id: str  # e.g. "S1", numbered left to right
    label: str
    box_2d: list[int]  # visible extent, already cut back where a nearer unit covers a side
    depth: int = 1  # 1 = nearest the camera
    hidden_by: list[str] = []  # nearer units that still overlap this box
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

    def front_boxes(self, shelf: Shelf) -> list[list[int]]:
        """Boxes of the nearer units standing in front of part of `shelf`."""
        return [s.box_2d for s in self.shelves if s.shelf_id in shelf.hidden_by]


def camera_id_for(frame_path: str) -> str:
    """extract_frames.py names frames <video>_<seconds>s.jpg; the video is the camera."""
    return Path(frame_path).stem.rsplit("_", 1)[0]


def layout_path(camera_id: str, layouts_dir: Path = LAYOUTS_DIR) -> Path:
    return layouts_dir / f"{camera_id}.json"


def clean_box(box: list[int]) -> list[int] | None:
    """Clamp to 0-1000, fix swapped corners, and drop malformed or tiny boxes."""
    if len(box) != 4:
        return None
    y0, x0, y1, x1 = (min(max(int(v), 0), 1000) for v in box)
    y0, y1 = sorted((y0, y1))
    x0, x1 = sorted((x0, x1))
    if y1 - y0 < MIN_BOX_SIZE or x1 - x0 < MIN_BOX_SIZE:
        return None
    return [y0, x0, y1, x1]


def intersect(a: list[int], b: list[int]) -> list[int] | None:
    box = [max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])]
    return box if box[2] - box[0] >= MIN_BOX_SIZE and box[3] - box[1] >= MIN_BOX_SIZE else None


def trim_behind(box: list[int], front: list[int]) -> list[int] | None:
    """Cut `box` back to the edge of a nearer `front` box that covers one whole side of it.

    Returns None if nothing usable is left. Partial overlaps (front covers a corner or
    the middle) can't be cut from a rectangle, so they are left for hidden_by.
    """
    y0, x0, y1, x1 = box
    fy0, fx0, fy1, fx1 = front
    if intersect(box, front) is None:
        return box
    if fy0 <= y0 + TRIM_TOLERANCE and fy1 >= y1 - TRIM_TOLERANCE:  # covers full height
        if fx0 <= x0 + TRIM_TOLERANCE:
            x0 = max(x0, fx1)
        elif fx1 >= x1 - TRIM_TOLERANCE:
            x1 = min(x1, fx0)
    elif fx0 <= x0 + TRIM_TOLERANCE and fx1 >= x1 - TRIM_TOLERANCE:  # covers full width
        if fy0 <= y0 + TRIM_TOLERANCE:
            y0 = max(y0, fy1)
        elif fy1 >= y1 - TRIM_TOLERANCE:
            y1 = min(y1, fy0)
    if y1 - y0 < MIN_BOX_SIZE or x1 - x0 < MIN_BOX_SIZE:
        return None
    return [y0, x0, y1, x1]


def build_layout(detection: Detection, camera_id: str, frame: str,
                 width: int, height: int, model: str) -> CameraLayout:
    units = []
    for unit in detection.units:
        box = clean_box(unit.box_2d)
        level_boxes = sorted(filter(None, (clean_box(l.box_2d) for l in unit.levels)))
        if box is None and not level_boxes:
            continue
        # Grow the unit box to contain its levels, so the pass 2 crop includes them all.
        boxes = level_boxes + ([box] if box else [])
        box = [min(b[0] for b in boxes), min(b[1] for b in boxes),
               max(b[2] for b in boxes), max(b[3] for b in boxes)]
        units.append((max(unit.depth, 1), unit.label, box, level_boxes))

    # Nearest first, so each unit is cut back against nearer units that are already final.
    units.sort(key=lambda u: u[0])
    kept = []
    for depth, label, box, level_boxes in units:
        for front_depth, _, front, _ in kept:
            if front_depth < depth and box:
                box = trim_behind(box, front)
        if box is None:
            continue  # entirely behind nearer units
        level_boxes = [b for b in (intersect(l, box) for l in level_boxes) if b]
        kept.append((depth, label, box, level_boxes))

    kept.sort(key=lambda u: (u[2][1], u[2][0]))  # left to right, then top to bottom
    shelves = []
    for i, (depth, label, box, level_boxes) in enumerate(kept, 1):
        levels = [Level(level_id=f"S{i}-L{j}", level=j, box_2d=b)
                  for j, b in enumerate(level_boxes, 1)]
        shelves.append(Shelf(shelf_id=f"S{i}", label=label, box_2d=box, depth=depth,
                             levels=levels))
    for shelf in shelves:
        shelf.hidden_by = [f.shelf_id for f in shelves
                           if f.depth < shelf.depth and intersect(f.box_2d, shelf.box_2d)]

    return CameraLayout(
        camera_id=camera_id, reference_frame=frame, frame_width=width, frame_height=height,
        model=model, created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=detection.notes, shelves=shelves,
    )


def to_pixels(box: list[int], width: int, height: int) -> tuple[int, int, int, int]:
    """[ymin, xmin, ymax, xmax] in 0-1000 -> (x0, y0, x1, y1) in pixels."""
    y0, x0, y1, x1 = box
    return (round(x0 * width / 1000), round(y0 * height / 1000),
            round(x1 * width / 1000), round(y1 * height / 1000))


def visible_mask(box: list[int], front_boxes: list[list[int]],
                 width: int, height: int) -> np.ndarray:
    """Pixel mask of `box` minus the nearer boxes standing in front of it."""
    mask = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = to_pixels(box, width, height)
    mask[y0:y1, x0:x1] = 255
    for front in front_boxes:
        fx0, fy0, fx1, fy1 = to_pixels(front, width, height)
        mask[fy0:fy1, fx0:fx1] = 0
    return mask


def draw_overlay(image: np.ndarray, layout: CameraLayout) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    for shelf in layout.shelves:
        x0, y0, x1, y1 = to_pixels(shelf.box_2d, w, h)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(out, f"{shelf.shelf_id} {shelf.label}", (x0 + 3, max(y0 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        for level in shelf.levels:
            lx0, ly0, lx1, ly1 = to_pixels(level.box_2d, w, h)
            cv2.rectangle(out, (lx0, ly0), (lx1, ly1), (0, 200, 255), 1)
            cv2.putText(out, f"L{level.level}", (lx0 + 3, ly0 + 14),
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
