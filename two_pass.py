#!/usr/bin/env python3
"""Two-pass shelf analysis: find shelf sections, drop weak ones, score each crop.

Pass 1 asks Gemini for shelf sections (label + box + confidence) on the full frame.
Weak sections are dropped (low confidence, bad geometry, duplicates). Pass 2 crops
each kept section out of the original frame and scores it on its own.

Usage:
    python two_pass.py frame.jpg
    python two_pass.py frame.jpg --min-confidence 0.7 --save-crops

Writes <out-dir>/<name>.json and <out-dir>/<name>_overlay.jpg (kept boxes in green,
dropped in red) so you can check pass 1 by eye before trusting pass 2.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import cv2
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

from shelf_metrics import DEFAULT_MODEL, generate_structured

PASS1_PROMPT = """You are looking at a single frame from a CCTV camera in a retail store.
List every distinct shelf section you can see: each gondola, wall shelf bay, endcap
display, or cooler. One entry per physical unit, not per shelf tier. Give each a short
label and a tight bounding box as [ymin, xmin, ymax, xmax], each normalized to 0-1000.
`confidence` (0-1) is how sure you are that the box covers one real, clearly visible
shelf section with products on it. Use a low value for anything mostly hidden, cut off
by the frame edge, or too blurry to judge. Do not include floors, people, or checkouts."""

PASS2_PROMPT = """This is a crop of one shelf section from a retail CCTV frame.
Estimate its stock state from what is visible. Do not guess about parts you cannot see.
`visible_fraction` (0-1) is how much of the section is actually visible and unobstructed
(people, carts, glare, or the crop edge reduce it). Percentages are 0-100."""


class Section(BaseModel):
    label: str
    kind: Literal["gondola", "wall_shelf", "cooler", "endcap", "other"]
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax], each 0-1000")
    confidence: float = Field(description="0-1, how sure this is a real, visible shelf section")


class Sections(BaseModel):
    sections: list[Section]


class RegionMetrics(BaseModel):
    fill_percent: float = Field(description="How full the section is, 0-100")
    empty_gap_count: int = Field(description="Visible out-of-stock gaps")
    stock_status: Literal["well_stocked", "partially_stocked", "low_stock", "empty", "unknown"]
    estimated_facings: int
    tidiness_score: int = Field(description="1 (messy) to 5 (neat, faced)")
    people_occluding_shelf: bool
    visible_fraction: float = Field(description="0-1, how much of the section is visible")
    notes: str


def jpeg_part(img) -> types.Part:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Could not encode image as JPEG")
    return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")


def to_pixels(box: list[int], w: int, h: int):
    """Normalized [ymin, xmin, ymax, xmax] (0-1000) -> clamped pixel (x1, y1, x2, y2), or None."""
    if len(box) != 4:
        return None
    ymin, xmin, ymax, xmax = (min(max(v, 0), 1000) for v in box)
    x1, x2 = round(xmin / 1000 * w), round(xmax / 1000 * w)
    y1, y2 = round(ymin / 1000 * h), round(ymax / 1000 * h)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def filter_sections(sections: list[Section], w: int, h: int, min_conf: float):
    """Drop invalid, low-confidence, odd-shaped, or duplicate sections. Returns (kept, dropped)."""
    kept, dropped = [], []
    # Highest confidence first, so duplicates lose to the better box.
    for s in sorted(sections, key=lambda s: s.confidence, reverse=True):
        info = {"label": s.label, "kind": s.kind, "confidence": s.confidence}
        box = to_pixels(s.box_2d, w, h)
        if box is None:
            dropped.append({**info, "box_px": None, "reason": "invalid box"})
            continue
        info["box_px"] = list(box)
        area = (box[2] - box[0]) * (box[3] - box[1]) / (w * h)
        aspect = (box[2] - box[0]) / (box[3] - box[1])
        if s.confidence < min_conf:
            reason = f"confidence {s.confidence:.2f} < {min_conf}"
        elif area < 0.01 or area > 0.6:
            reason = f"area {area:.1%} outside 1%-60%"
        elif aspect > 8 or aspect < 1 / 8:
            reason = f"aspect ratio {aspect:.1f} too extreme"
        elif any(iou(box, k["box_px"]) > 0.5 for k in kept):
            reason = "duplicate of a higher-confidence section"
        else:
            kept.append(info)
            continue
        dropped.append({**info, "reason": reason})
    kept.sort(key=lambda k: (k["box_px"][0], k["box_px"][1]))  # left to right, then top down
    for i, k in enumerate(kept, 1):
        k["id"] = f"s{i}"
    return kept, dropped


def crop_with_padding(img, box, pad: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = round((x2 - x1) * pad), round((y2 - y1) * pad)
    return img[max(y1 - py, 0): min(y2 + py, h), max(x1 - px, 0): min(x2 + px, w)]


def draw_overlay(img, kept, dropped):
    out = img.copy()
    for d in dropped:
        if d["box_px"]:
            x1, y1, x2, y2 = d["box_px"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, "dropped", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for k in kept:
        x1, y1, x2, y2 = k["box_px"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(out, f"{k['id']} {k['confidence']:.2f}", (x1 + 4, y1 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", type=Path, help="Path to a CCTV frame")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--out-dir", type=Path, default=Path("results/two_pass"))
    # Pass-1 confidence is self-reported by the model, so treat the threshold as a knob to tune.
    parser.add_argument("--min-confidence", type=float, default=0.6,
                        help="Drop pass-1 sections below this confidence")
    parser.add_argument("--min-visible", type=float, default=0.5,
                        help="Exclude regions from the totals if less of them is visible")
    parser.add_argument("--pad", type=float, default=0.05, help="Crop padding as a fraction of box size")
    parser.add_argument("--save-crops", action="store_true", help="Save each crop next to the output")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        return 1
    img = cv2.imread(str(args.image))
    if img is None:
        print(f"Can't read {args.image} as an image", file=sys.stderr)
        return 1
    h, w = img.shape[:2]
    client = genai.Client(api_key=api_key)

    # Pass 1: find sections on the full frame.
    found = generate_structured(client, args.model, [jpeg_part(img), PASS1_PROMPT], Sections)
    kept, dropped = filter_sections(found.sections, w, h, args.min_confidence)
    print(f"Pass 1: {len(found.sections)} sections found, {len(kept)} kept, {len(dropped)} dropped",
          file=sys.stderr)
    for d in dropped:
        print(f"  dropped {d['label']!r}: {d['reason']}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    cv2.imwrite(str(args.out_dir / f"{stem}_overlay.jpg"), draw_overlay(img, kept, dropped))

    # Pass 2: score each kept section on its own crop.
    crops_dir = args.out_dir / f"{stem}_crops"
    if args.save_crops and kept:
        crops_dir.mkdir(exist_ok=True)
    regions = []
    for k in kept:
        crop = crop_with_padding(img, k["box_px"], args.pad)
        if args.save_crops:
            cv2.imwrite(str(crops_dir / f"{k['id']}.jpg"), crop)
        try:
            metrics = generate_structured(client, args.model, [jpeg_part(crop), PASS2_PROMPT], RegionMetrics)
        except errors.ServerError as e:  # retries exhausted; keep the rest of the run
            regions.append({**k, "metrics": None, "error": f"{e.code} {e.status}"})
            print(f"  {k['id']} {k['label']!r}: FAILED ({e.code}), skipping", file=sys.stderr)
            continue
        regions.append({**k, "metrics": metrics.model_dump()})
        print(f"  {k['id']} {k['label']!r}: fill {metrics.fill_percent:.0f}%, "
              f"gaps {metrics.empty_gap_count}, visible {metrics.visible_fraction:.0%}", file=sys.stderr)

    # Totals only use regions that are visible enough to trust, weighted by box area.
    scored = [r for r in regions if r["metrics"]]
    usable = [r for r in scored if r["metrics"]["visible_fraction"] >= args.min_visible]
    weights = [(r["box_px"][2] - r["box_px"][0]) * (r["box_px"][3] - r["box_px"][1]) for r in usable]
    aggregate = {
        "regions_scored": len(scored),
        "regions_failed": len(regions) - len(scored),
        "regions_used": len(usable),
        "weighted_fill_percent": (
            round(sum(r["metrics"]["fill_percent"] * wt for r, wt in zip(usable, weights)) / sum(weights), 1)
            if usable else None
        ),
        "total_gaps": sum(r["metrics"]["empty_gap_count"] for r in usable),
    }

    result = {
        "image": args.image.name, "image_size": [w, h], "model": args.model,
        "settings": {"min_confidence": args.min_confidence, "min_visible": args.min_visible, "pad": args.pad},
        "aggregate": aggregate, "regions": regions, "dropped": dropped,
    }
    output = json.dumps(result, indent=2)
    (args.out_dir / f"{stem}.json").write_text(output + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
