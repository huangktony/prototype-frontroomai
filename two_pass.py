#!/usr/bin/env python3
"""Two-pass shelf analysis with saved per-camera regions.

Pass 1 asks Gemini for shelf sections (label + box + confidence) on the full frame.
Weak sections are dropped (low confidence, bad geometry, duplicates). Pass 2 crops
each kept section out of the original frame and scores it on its own.

A fixed camera sees the same shelves every time, so pass 1 only needs to run once:

    # 1. Find regions on a few frames, keep the ones most frames agree on, save them
    python two_pass.py detect --camera store1 frame_a.jpg frame_b.jpg frame_c.jpg
    #    -> partitions/store1.json + partitions/store1_overlay.jpg  (open the overlay)
    # 2. After checking the overlay (edit the JSON if a box is wrong), approve it
    python two_pass.py approve --camera store1
    # 3. Score any frame from that camera; pass 1 is skipped, region IDs stay the same
    python two_pass.py score --camera store1 frame.jpg --save-crops

    # Or run both passes on one frame with nothing saved
    python two_pass.py run frame.jpg

Scoring writes <out-dir>/<name>.json and <out-dir>/<name>_overlay.jpg.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
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


# ---------- geometry ----------

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


def to_normalized(box_px, w: int, h: int) -> list[int]:
    """Pixel (x1, y1, x2, y2) -> normalized [ymin, xmin, ymax, xmax] (0-1000)."""
    x1, y1, x2, y2 = box_px
    return [round(y1 / h * 1000), round(x1 / w * 1000), round(y2 / h * 1000), round(x2 / w * 1000)]


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
    return number_left_to_right(kept), dropped


def number_left_to_right(kept: list[dict]) -> list[dict]:
    kept.sort(key=lambda k: (k["box_px"][0], k["box_px"][1]))
    for i, k in enumerate(kept, 1):
        k["id"] = f"s{i}"
    return kept


def crop_with_padding(img, box, pad: float):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    px, py = round((x2 - x1) * pad), round((y2 - y1) * pad)
    return img[max(y1 - py, 0): min(y2 + py, h), max(x1 - px, 0): min(x2 + px, w)]


def draw_overlay(img, kept, dropped=()):
    out = img.copy()
    for d in dropped:
        if d["box_px"]:
            x1, y1, x2, y2 = d["box_px"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, "dropped", (x1 + 4, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for k in kept:
        x1, y1, x2, y2 = k["box_px"]
        text = k["id"] + (f" {k['confidence']:.2f}" if k.get("confidence") is not None else "")
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 3)
        cv2.putText(out, text, (x1 + 4, y1 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return out


# ---------- pass 1 ----------

def detect_sections(client, model: str, img, min_conf: float):
    """Run pass 1 on one frame. Returns (kept, dropped)."""
    h, w = img.shape[:2]
    found = generate_structured(client, model, [jpeg_part(img), PASS1_PROMPT], Sections)
    kept, dropped = filter_sections(found.sections, w, h, min_conf)
    print(f"Pass 1: {len(found.sections)} sections found, {len(kept)} kept, {len(dropped)} dropped",
          file=sys.stderr)
    for d in dropped:
        print(f"  dropped {d['label']!r}: {d['reason']}", file=sys.stderr)
    return kept, dropped


def consensus(per_frame: list[list[dict]], min_agree: float) -> list[dict]:
    """Merge pass-1 results from several frames of one camera.

    Boxes from different frames are grouped by overlap (IoU > 0.5, at most one box per
    frame per group). A group survives if enough frames contain it, and its box is the
    per-coordinate median. This drops one-off boxes caused by a person or a bad frame.
    Each frame's dicts must carry `norm`, a (x1, y1, x2, y2) box in 0-1000 coordinates.
    """
    groups: list[list[tuple[int, dict]]] = []
    for fi, kept in enumerate(per_frame):
        for s in kept:
            for g in groups:
                if all(fj != fi for fj, _ in g) and iou(s["norm"], g[0][1]["norm"]) > 0.5:
                    g.append((fi, s))
                    break
            else:
                groups.append([(fi, s)])

    need = max(1, math.ceil(len(per_frame) * min_agree))
    regions = []
    for g in groups:
        if len(g) < need:
            continue
        members = [s for _, s in g]
        best = max(members, key=lambda s: s["confidence"])
        box = [round(statistics.median(m["norm"][i] for m in members)) for i in range(4)]
        regions.append({
            "label": best["label"],
            "kind": Counter(m["kind"] for m in members).most_common(1)[0][0],
            "seen_in": f"{len(g)}/{len(per_frame)}",
            "box_px": box,  # normalized x1, y1, x2, y2; converted when saved
        })
    return number_left_to_right(regions)


# ---------- saved regions ----------

def partition_path(args) -> Path:
    if not re.fullmatch(r"[\w.-]+", args.camera):
        sys.exit(f"Camera name {args.camera!r} should only contain letters, numbers, '.', '_' or '-'")
    return args.partitions_dir / f"{args.camera}.json"


def load_partition(args) -> dict:
    path = partition_path(args)
    if not path.exists():
        sys.exit(f"No saved regions at {path}. Run `detect --camera {args.camera} <frames>` first.")
    return json.loads(path.read_text())


# ---------- pass 2 + output ----------

def score_regions(client, args, img, stem: str, kept: list[dict]) -> list[dict]:
    crops_dir = args.out_dir / f"{stem}_crops"
    if args.save_crops and kept:
        crops_dir.mkdir(parents=True, exist_ok=True)
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
    return regions


def aggregate_regions(regions: list[dict], min_visible: float) -> dict:
    """Totals only use regions visible enough to trust, weighted by box area."""
    scored = [r for r in regions if r["metrics"]]
    usable = [r for r in scored if r["metrics"]["visible_fraction"] >= min_visible]
    weights = [(r["box_px"][2] - r["box_px"][0]) * (r["box_px"][3] - r["box_px"][1]) for r in usable]
    return {
        "regions_scored": len(scored),
        "regions_failed": len(regions) - len(scored),
        "regions_used": len(usable),
        "weighted_fill_percent": (
            round(sum(r["metrics"]["fill_percent"] * wt for r, wt in zip(usable, weights)) / sum(weights), 1)
            if usable else None
        ),
        "total_gaps": sum(r["metrics"]["empty_gap_count"] for r in usable),
    }


def score_and_write(client, args, img, image: Path, kept, dropped, extra: dict) -> None:
    h, w = img.shape[:2]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out_dir / f"{image.stem}_overlay.jpg"), draw_overlay(img, kept, dropped))
    regions = score_regions(client, args, img, image.stem, kept)
    result = {
        "image": image.name, "image_size": [w, h], "model": args.model, **extra,
        "settings": {"min_visible": args.min_visible, "pad": args.pad},
        "aggregate": aggregate_regions(regions, args.min_visible), "regions": regions, "dropped": dropped,
    }
    output = json.dumps(result, indent=2)
    (args.out_dir / f"{image.stem}.json").write_text(output + "\n")
    print(output)


# ---------- commands ----------

def make_client(args):
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY first.")
    return genai.Client(api_key=api_key)


def read_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        sys.exit(f"Can't read {path} as an image")
    return img


def cmd_run(args) -> int:
    client = make_client(args)
    img = read_image(args.image)
    kept, dropped = detect_sections(client, args.model, img, args.min_confidence)
    score_and_write(client, args, img, args.image, kept, dropped,
                    {"settings_pass1": {"min_confidence": args.min_confidence}})
    return 0


def cmd_detect(args) -> int:
    path = partition_path(args)
    if path.exists() and not args.force:
        sys.exit(f"{path} already exists. Pass --force to overwrite (this resets approval).")
    client = make_client(args)

    per_frame, first_img = [], None
    for image in args.images:
        img = read_image(image)
        first_img = img if first_img is None else first_img
        h, w = img.shape[:2]
        print(f"{image.name}:", file=sys.stderr)
        kept, _ = detect_sections(client, args.model, img, args.min_confidence)
        for k in kept:
            x1, y1, x2, y2 = k["box_px"]
            k["norm"] = (x1 / w * 1000, y1 / h * 1000, x2 / w * 1000, y2 / h * 1000)
        per_frame.append(kept)

    merged = consensus(per_frame, args.min_agree)
    h, w = first_img.shape[:2]
    regions = []
    for r in merged:
        box_2d = to_normalized([round(v / 1000 * d) for v, d in zip(r["box_px"], (w, h, w, h))], w, h)
        regions.append({"id": r["id"], "label": r["label"], "kind": r["kind"],
                        "seen_in": r["seen_in"], "box_2d": box_2d})

    partition = {
        "camera": args.camera,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved": False,
        "source_frames": [p.name for p in args.images],
        "frame_size": [w, h],
        "regions": regions,
    }
    args.partitions_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partition, indent=2) + "\n")
    overlay_boxes = [{"id": r["id"], "confidence": None, "box_px": list(to_pixels(r["box_2d"], w, h))}
                     for r in regions]
    overlay = args.partitions_dir / f"{args.camera}_overlay.jpg"
    cv2.imwrite(str(overlay), draw_overlay(first_img, overlay_boxes))
    print(f"Saved {len(regions)} regions to {path}\nCheck {overlay}, fix any boxes in the JSON, "
          f"then run: python two_pass.py approve --camera {args.camera}")
    return 0


def cmd_approve(args) -> int:
    path = partition_path(args)
    partition = load_partition(args)
    partition["approved"] = True
    path.write_text(json.dumps(partition, indent=2) + "\n")
    print(f"Approved {len(partition['regions'])} regions for camera {args.camera!r}")
    return 0


def cmd_score(args) -> int:
    partition = load_partition(args)
    if not partition["approved"] and not args.allow_unapproved:
        sys.exit(f"Regions for {args.camera!r} aren't approved. Check the overlay, then run "
                 f"`approve --camera {args.camera}` (or pass --allow-unapproved).")
    client = make_client(args)
    img = read_image(args.image)
    h, w = img.shape[:2]
    sw, sh = partition["frame_size"]
    if abs(w / h - sw / sh) > 0.05:
        print(f"Warning: frame is {w}x{h} but regions were saved on {sw}x{sh}; "
              f"boxes may not line up", file=sys.stderr)

    kept = []
    for r in partition["regions"]:
        box = to_pixels(r["box_2d"], w, h)
        if box:
            kept.append({"id": r["id"], "label": r["label"], "kind": r["kind"],
                         "confidence": None, "box_px": list(box)})
    print(f"Using {len(kept)} saved regions for {args.camera!r} (pass 1 skipped)", file=sys.stderr)
    score_and_write(client, args, img, args.image, kept, [],
                    {"camera": args.camera, "regions_created": partition["created"]})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))

    scoring = argparse.ArgumentParser(add_help=False)
    scoring.add_argument("--out-dir", type=Path, default=Path("results/two_pass"))
    scoring.add_argument("--min-visible", type=float, default=0.5,
                         help="Exclude regions from the totals if less of them is visible")
    scoring.add_argument("--pad", type=float, default=0.05, help="Crop padding as a fraction of box size")
    scoring.add_argument("--save-crops", action="store_true", help="Save each crop next to the output")

    # Pass-1 confidence is self-reported by the model, so treat the threshold as a knob to tune.
    conf = argparse.ArgumentParser(add_help=False)
    conf.add_argument("--min-confidence", type=float, default=0.6,
                      help="Drop pass-1 sections below this confidence")

    cam = argparse.ArgumentParser(add_help=False)
    cam.add_argument("--camera", required=True, help="Camera name; regions are saved under this name")
    cam.add_argument("--partitions-dir", type=Path, default=Path("partitions"))

    p = sub.add_parser("run", parents=[common, scoring, conf], help="Both passes on one frame, nothing saved")
    p.add_argument("image", type=Path)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("detect", parents=[common, conf, cam], help="Find and save regions for a camera")
    p.add_argument("images", type=Path, nargs="+", help="A few frames from this camera")
    p.add_argument("--min-agree", type=float, default=0.5,
                   help="Keep a region only if this fraction of the frames contain it")
    p.add_argument("--force", action="store_true", help="Overwrite existing saved regions")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("approve", parents=[cam], help="Mark a camera's saved regions as checked")
    p.set_defaults(func=cmd_approve)

    p = sub.add_parser("score", parents=[common, scoring, cam], help="Score a frame using saved regions")
    p.add_argument("image", type=Path)
    p.add_argument("--allow-unapproved", action="store_true")
    p.set_defaults(func=cmd_score)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
