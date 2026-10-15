#!/usr/bin/env python3
"""Render a camera's saved shelf layout as transparent yellow highlights, for slides.

Usage:
    python present_layout.py test_frames/4WXZTF7EWL0_00016s.jpg
    python present_layout.py frame.jpg --camera aisle3 --out slide.jpg --no-levels
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from detect_shelves import LAYOUTS_DIR, CameraLayout, camera_id_for, load_layout, visible_mask

YELLOW = (0, 215, 255)  # BGR
FILL_ALPHA = 0.30


def outline(out: np.ndarray, mask: np.ndarray, thickness: int) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, YELLOW, thickness, cv2.LINE_AA)


def draw_highlight(image: np.ndarray, layout: CameraLayout, levels: bool = True) -> np.ndarray:
    """Highlight only the visible part of each unit, so units behind others get cut away."""
    h, w = image.shape[:2]
    masks = {s.shelf_id: visible_mask(s.polygon, layout.front_polygons(s), w, h)
             for s in layout.shelves}
    # Fill through one mask so overlapping units don't stack into darker patches.
    union = np.zeros((h, w), np.uint8)
    for mask in masks.values():
        union |= mask
    tint = image.copy()
    tint[union > 0] = YELLOW
    out = cv2.addWeighted(tint, FILL_ALPHA, image, 1 - FILL_ALPHA, 0)

    for shelf in layout.shelves:
        if levels:
            for level in shelf.levels:
                outline(out, visible_mask(level.polygon, layout.front_polygons(shelf), w, h), 1)
        outline(out, masks[shelf.shelf_id], 2)

    # Labels last so no outline crosses them.
    for shelf in layout.shelves:
        x0, y0, _, _ = cv2.boundingRect(masks[shelf.shelf_id])
        text = shelf.shelf_id if not levels else f"{shelf.shelf_id} ({len(shelf.levels)} levels)"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ty = y0 + th + 8 if y0 < th + 8 else y0  # keep the tag inside the frame
        cv2.rectangle(out, (x0, ty - th - 8), (x0 + tw + 10, ty), YELLOW, -1)
        cv2.putText(out, text, (x0 + 5, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="Frame to draw on (ideally one with no one blocking shelves)")
    parser.add_argument("--camera", help="Camera ID (default: video name from the frame filename)")
    parser.add_argument("--layouts", type=Path, default=LAYOUTS_DIR)
    parser.add_argument("--no-levels", action="store_true", help="Only highlight whole units")
    parser.add_argument("--out", type=Path, help="Default: presentation/<frame>_shelves.jpg")
    args = parser.parse_args()

    camera_id = args.camera or camera_id_for(args.image)
    layout = load_layout(camera_id, args.layouts)
    if layout is None:
        print(f"No layout for camera {camera_id!r}; run detect_shelves.py first.", file=sys.stderr)
        return 1
    image = cv2.imread(args.image)
    if image is None:
        print(f"Can't read {args.image}", file=sys.stderr)
        return 1

    out = args.out or Path("presentation") / f"{Path(args.image).stem}_shelves.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), draw_highlight(image, layout, levels=not args.no_levels),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
