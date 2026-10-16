#!/usr/bin/env python3
"""Two-pass shelf analysis with saved per-camera regions.

Pass 1 asks Gemini for shelf sections as 4-corner polygons (with confidence and depth) on
the full frame, plus a 4-corner polygon for each shelf row inside each section. Weak
sections are dropped (low confidence, bad geometry, duplicates). Pass 2 straightens each
kept section into an upright crop with its rows outlined as L1, L2, ..., greys out nearer
units standing in front of it, and scores the section and each row on its own. Polygon handling is adapted from Parth's
parthswork/polygons branch (see polygons.py).

A fixed camera sees the same shelves every time, so pass 1 only needs to run once:

    # 1. Find regions on a few frames, keep the ones most frames agree on, save them
    python two_pass.py detect --camera store1 frame_a.jpg frame_b.jpg frame_c.jpg
    #    -> partitions/store1.json + partitions/store1_overlay.jpg  (open the overlay)
    # 2. After checking the overlay (edit the JSON if a polygon is wrong), approve it
    python two_pass.py approve --camera store1
    # 3. Score any frame from that camera; pass 1 is skipped, region IDs stay the same
    python two_pass.py score --camera store1 frame.jpg --save-crops

    # Or run both passes on one frame with nothing saved
    python two_pass.py run frame.jpg

    # Or a whole video: find shelves on its first frame (once per camera), then score a
    # frame every --every seconds against them
    python two_pass.py video videos/4WXZTF7EWL0.mp4 --every 2

Scoring writes <out-dir>/<name>.json and <out-dir>/<name>_overlay.jpg.
"""
import argparse
import csv
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

from polygons import (Polygon, clean_levels, clean_polygon, link_and_number, polygon_area,
                      polygon_iou, straighten, to_pixels)
from shelf_metrics import DEFAULT_MODEL, QuotaExhausted, generate_structured

POLYGON_HELP = ("4 corners [[y, x], ...] normalised 0-1000, in the order top-left, "
                "top-right, bottom-right, bottom-left")

PASS1_PROMPT = """You are looking at a single frame from a fixed CCTV camera in a retail store.
List every distinct shelf section you can see: each gondola, wall shelf bay, endcap
display, or cooler. One entry per physical unit, not per shelf tier. Skip units that are
mostly out of frame, seen only edge-on, or too far away to make out products.

For each, give a short label and a polygon of exactly 4 corners tracing its
product-facing front face as it appears in perspective (a shelf seen at an angle is a
slanted quadrilateral, not an upright rectangle). Give corners as [y, x] normalised to
0-1000, in the order top-left, top-right, bottom-right, bottom-left.

Inside each section, also give one 4-corner polygon per shelf row (tier), from the top
row to the bottom, each following the slope of that shelf's front edge, in the same
corner order. Leave out rows you cannot see at all. Give each row a `confidence` (0-1)
for how sure you are that it is a real shelf row whose products you can make out.

Give each a depth rank: 1 for the units nearest the camera, then 2, 3, ... for units
further back; units at about the same distance share a rank. Trace each unit's whole
face even where a nearer unit stands in front of it, because depth is used to work out
what is hidden. People and carts move, so ignore them and trace the shelf as if they
were not there.

`confidence` (0-1) is how sure you are that the polygon covers one real, clearly visible
shelf section with products on it. Use a low value for anything mostly hidden, cut off by
the frame edge, or too blurry to judge. Do not include floors, people, or checkouts."""

PASS2_PROMPT = """This image is one shelf section cut from a retail CCTV frame,
perspective-corrected so its front faces the viewer and enlarged for detail; expect some
stretching and blur from that. Flat grey areas are other fixtures standing in front of
it, or outside the camera's view: ignore them and do not count them as empty shelf space.
Estimate its stock state from what is visible. Do not guess about parts you cannot see.
`visible_fraction` (0-1) is how much of the section is actually visible and unobstructed
(grey areas, people, carts and glare all reduce it). Percentages are 0-100."""


class SectionLevel(BaseModel):
    polygon: Polygon = Field(description=POLYGON_HELP)
    confidence: float = Field(description="0-1, how sure this is a real, visible shelf row")


class Section(BaseModel):
    label: str
    kind: Literal["gondola", "wall_shelf", "cooler", "endcap", "other"]
    polygon: Polygon = Field(description=POLYGON_HELP)
    depth: int = Field(description="1 = nearest the camera; higher = further back")
    confidence: float = Field(description="0-1, how sure this is a real, visible shelf section")
    levels: list[SectionLevel] = Field(description="Shelf rows inside this section, top to bottom")


class Sections(BaseModel):
    sections: list[Section]


class StockMetrics(BaseModel):
    """Scored for the whole section, and again for each shelf row in it."""
    fill_percent: float = Field(description="How full it is, 0-100")
    empty_gap_count: int = Field(description="Visible out-of-stock gaps")
    stock_status: Literal["well_stocked", "partially_stocked", "low_stock", "empty", "unknown"]
    estimated_facings: int
    tidiness_score: int = Field(description="1 (messy) to 5 (neat, faced)")
    people_occluding_shelf: bool
    visible_fraction: float = Field(description="0-1, how much of it is visible")
    notes: str


class LevelMetrics(StockMetrics):
    level: int = Field(description="Row number as labelled on the image (L1 = 1 = top)")


class RegionMetrics(StockMetrics):
    levels: list[LevelMetrics] = Field(description="One entry per labelled shelf row")


# ---------- pass 1 ----------

def jpeg_part(img) -> types.Part:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError("Could not encode image as JPEG")
    return types.Part.from_bytes(data=buf.tobytes(), mime_type="image/jpeg")


def filter_sections(sections: list[Section], min_conf: float, min_row_conf: float):
    """Drop invalid, low-confidence, odd-shaped, or duplicate sections, and low-confidence
    shelf rows inside kept ones. Returns (kept, dropped)."""
    kept, dropped = [], []
    # Highest confidence first, so duplicates lose to the better polygon.
    for s in sorted(sections, key=lambda s: s.confidence, reverse=True):
        info = {"label": s.label, "kind": s.kind, "confidence": s.confidence, "depth": max(s.depth, 1)}
        polygon = clean_polygon(s.polygon)
        if polygon is None:
            dropped.append({**info, "polygon": None, "reason": "invalid polygon"})
            continue
        info["polygon"] = polygon
        info["levels"] = [{"level": n, "polygon": p} for n, p in
                          enumerate(clean_levels([l.polygon for l in s.levels
                                                   if l.confidence >= min_row_conf], polygon), 1)]
        info["rows_dropped"] = len(s.levels) - len(info["levels"])
        area = polygon_area(polygon)
        xs, ys = [x for _, x in polygon], [y for y, _ in polygon]
        aspect = (max(xs) - min(xs)) / max(max(ys) - min(ys), 1)
        if s.confidence < min_conf:
            reason = f"confidence {s.confidence:.2f} < {min_conf}"
        elif area < 0.01 or area > 0.6:
            reason = f"area {area:.1%} outside 1%-60%"
        elif aspect > 8 or aspect < 1 / 8:
            reason = f"aspect ratio {aspect:.1f} too extreme"
        elif any(polygon_iou(polygon, k["polygon"]) > 0.5 for k in kept):
            reason = "duplicate of a higher-confidence section"
        else:
            kept.append(info)
            continue
        dropped.append({**info, "reason": reason})
    return kept, dropped


def detect_sections(client, model: str, img, min_conf: float, min_row_conf: float):
    """Run pass 1 on one frame. Returns (kept, dropped); kept isn't numbered yet."""
    found = generate_structured(client, model, [jpeg_part(img), PASS1_PROMPT], Sections)
    kept, dropped = filter_sections(found.sections, min_conf, min_row_conf)
    rows = sum(len(k["levels"]) for k in kept)
    rows_dropped = sum(k.pop("rows_dropped") for k in kept)
    print(f"Pass 1: {len(found.sections)} sections found, {len(kept)} kept, {len(dropped)} dropped; "
          f"{rows} shelf rows kept, {rows_dropped} dropped (low confidence or invalid)",
          file=sys.stderr)
    for d in dropped:
        print(f"  dropped {d['label']!r}: {d['reason']}", file=sys.stderr)
    return kept, dropped


def consensus(per_frame: list[list[dict]], min_agree: float) -> list[dict]:
    """Merge pass-1 results from several frames of one camera.

    Polygons from different frames are grouped by overlap (IoU > 0.5, at most one per
    frame per group). A group survives if enough frames contain it, and its polygon is the
    per-corner median. This drops one-off shapes caused by a person or a bad frame.
    """
    groups: list[list[tuple[int, dict]]] = []
    for fi, kept in enumerate(per_frame):
        for s in kept:
            for g in groups:
                if all(fj != fi for fj, _ in g) and polygon_iou(s["polygon"], g[0][1]["polygon"]) > 0.5:
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
        median = [[round(statistics.median(m["polygon"][c][axis] for m in members)) for axis in (0, 1)]
                  for c in range(4)]
        polygon = clean_polygon(median)
        if polygon is None:
            continue
        # Rows: use the frames that agree most often on the row count, median each corner.
        row_count = Counter(len(m["levels"]) for m in members).most_common(1)[0][0]
        with_rows = [m for m in members if len(m["levels"]) == row_count]
        level_polygons = [clean_polygon([[round(statistics.median(m["levels"][n]["polygon"][c][axis]
                                                                 for m in with_rows))
                                          for axis in (0, 1)] for c in range(4)])
                          for n in range(row_count)]
        levels = clean_levels([p for p in level_polygons if p], polygon)
        best = max(members, key=lambda s: s["confidence"])
        regions.append({
            "label": best["label"],
            "kind": Counter(m["kind"] for m in members).most_common(1)[0][0],
            "depth": Counter(m["depth"] for m in members).most_common(1)[0][0],
            "seen_in": f"{len(g)}/{len(per_frame)}",
            "polygon": polygon,
            "levels": [{"level": n, "polygon": p} for n, p in enumerate(levels, 1)],
        })
    return regions


# ---------- saved regions ----------

def partition_path(args) -> Path:
    if not re.fullmatch(r"[\w.-]+", args.camera):
        sys.exit(f"Camera name {args.camera!r} should only contain letters, numbers, '.', '_' or '-'")
    return args.partitions_dir / f"{args.camera}.json"


def load_partition(args) -> dict:
    path = partition_path(args)
    if not path.exists():
        sys.exit(f"No saved regions at {path}. Run `detect --camera {args.camera} <frames>` first.")
    partition = json.loads(path.read_text())
    if any("polygon" not in r for r in partition["regions"]):
        sys.exit(f"{path} uses the old box format. Re-run `detect --camera {args.camera} ... --force`.")
    return partition


# ---------- drawing ----------

def draw_overlay(img, kept, dropped=()):
    out = img.copy()
    h, w = img.shape[:2]
    for d in dropped:
        if d["polygon"]:
            pts = to_pixels(d["polygon"], w, h).round().astype("int32")
            cv2.polylines(out, [pts], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(out, "dropped", (int(pts[0][0]) + 4, int(pts[0][1]) + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    for k in kept:
        for level in k.get("levels", []):
            lpts = to_pixels(level["polygon"], w, h).round().astype("int32")
            cv2.polylines(out, [lpts], True, (0, 215, 255), 1, cv2.LINE_AA)
        pts = to_pixels(k["polygon"], w, h).round().astype("int32")
        text = k["id"] + (f" {k['confidence']:.2f}" if k.get("confidence") is not None else "")
        cv2.polylines(out, [pts], True, (0, 200, 0), 3, cv2.LINE_AA)
        cv2.putText(out, text, (int(pts[0][0]) + 4, int(pts[0][1]) + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
    return out


# ---------- pass 2 + output ----------

def pass2_prompt(row_count: int) -> str:
    if row_count:
        labels = ", ".join(f"L{n}" for n in range(1, row_count + 1))
        rows = (f"Yellow outlines mark its {row_count} shelf rows, labelled {labels} from top to "
                f"bottom; give one `levels` entry per label.")
    else:
        rows = "No shelf rows are marked; split it into rows yourself in `levels`, 1 = top."
    return PASS2_PROMPT + "\n" + rows


def score_regions(client, args, img, stem: str, kept: list[dict]) -> list[dict]:
    crops_dir = args.out_dir / f"{stem}_crops"
    if args.save_crops and kept:
        crops_dir.mkdir(parents=True, exist_ok=True)
    polygons = {k["id"]: k["polygon"] for k in kept}
    regions = []
    quota_error = None
    for k in kept:
        if quota_error:  # every later call would fail too; record it and keep what we have
            regions.append({**k, "metrics": None, "error": "not scored: quota exhausted"})
            continue
        fronts = [polygons[i] for i in k["hidden_by"] if i in polygons]
        levels = [l["polygon"] for l in k.get("levels", [])]
        crop = straighten(img, k["polygon"], fronts, args.pad, levels)
        if args.save_crops:
            cv2.imwrite(str(crops_dir / f"{k['id']}.jpg"), crop)
        try:
            metrics = generate_structured(client, args.model, [jpeg_part(crop), pass2_prompt(len(levels))],
                                          RegionMetrics)
        except errors.ServerError as e:  # retries exhausted; keep the rest of the run
            regions.append({**k, "metrics": None, "error": f"{e.code} {e.status}"})
            print(f"  {k['id']} {k['label']!r}: FAILED ({e.code}), skipping", file=sys.stderr)
            continue
        except QuotaExhausted as e:
            quota_error = e
            regions.append({**k, "metrics": None, "error": "not scored: quota exhausted"})
            print(f"  {k['id']} {k['label']!r}: {e}", file=sys.stderr)
            continue
        regions.append({**k, "metrics": metrics.model_dump()})
        rows = " ".join(f"L{l.level}={l.fill_percent:.0f}%" for l in metrics.levels)
        print(f"  {k['id']} {k['label']!r}: fill {metrics.fill_percent:.0f}%, "
              f"gaps {metrics.empty_gap_count}, visible {metrics.visible_fraction:.0%}  {rows}",
              file=sys.stderr)
    return regions


def aggregate_regions(regions: list[dict], min_visible: float) -> dict:
    """Totals only use regions visible enough to trust, weighted by polygon area."""
    scored = [r for r in regions if r["metrics"]]
    usable = [r for r in scored if r["metrics"]["visible_fraction"] >= min_visible]
    weights = [polygon_area(r["polygon"]) for r in usable]
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


def score_and_write(client, args, img, image: Path, kept, dropped, extra: dict,
                    print_json: bool = True) -> dict:
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
    if print_json:
        print(output)
    return result


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
    kept, dropped = detect_sections(client, args.model, img, args.min_confidence,
                                    args.min_row_confidence)
    score_and_write(client, args, img, args.image, link_and_number(kept), dropped,
                    {"settings_pass1": {"min_confidence": args.min_confidence,
                                        "min_row_confidence": args.min_row_confidence}})
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
        print(f"{image.name}:", file=sys.stderr)
        kept, _ = detect_sections(client, args.model, img, args.min_confidence,
                                  args.min_row_confidence)
        per_frame.append(kept)

    regions = link_and_number(consensus(per_frame, args.min_agree))
    save_partition(args, regions, [p.name for p in args.images], first_img)
    print(f"Check the overlay, fix any polygons in the JSON, then run: "
          f"python two_pass.py approve --camera {args.camera}")
    return 0


def save_partition(args, regions: list[dict], source_frames: list[str], first_img) -> None:
    path = partition_path(args)
    h, w = first_img.shape[:2]
    partition = {
        "camera": args.camera,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "approved": False,
        "source_frames": source_frames,
        "frame_size": [w, h],
        "regions": [{"id": r["id"], "label": r["label"], "kind": r["kind"], "seen_in": r["seen_in"],
                     "depth": r["depth"], "hidden_by": r["hidden_by"], "polygon": r["polygon"],
                     "levels": r["levels"]}
                    for r in regions],
    }
    args.partitions_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(partition, indent=2) + "\n")
    overlay = args.partitions_dir / f"{args.camera}_overlay.jpg"
    cv2.imwrite(str(overlay), draw_overlay(first_img, regions))
    rows = sum(len(r["levels"]) for r in regions)
    print(f"Saved {len(regions)} regions ({rows} shelf rows) to {path}; overlay: {overlay}")


def cmd_approve(args) -> int:
    path = partition_path(args)
    partition = load_partition(args)
    partition["approved"] = True
    path.write_text(json.dumps(partition, indent=2) + "\n")
    print(f"Approved {len(partition['regions'])} regions for camera {args.camera!r}")
    return 0


def saved_regions(partition: dict) -> list[dict]:
    return [{"id": r["id"], "label": r["label"], "kind": r["kind"], "confidence": None,
             "depth": r["depth"], "hidden_by": r["hidden_by"], "polygon": r["polygon"],
             "levels": r.get("levels", [])}
            for r in partition["regions"]]


def video_frames(path: Path, every: float, start: float, end: float | None):
    """Yield (seconds, frame) every `every` seconds, like extract_frames.py."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Can't open {path} as a video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(end, duration) if end else duration
    t = start
    try:
        while t < end:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                break
            yield t, frame
            t += every
    finally:
        cap.release()


def cmd_video(args) -> int:
    args.camera = args.camera or args.video.stem
    path = partition_path(args)
    client = make_client(args)
    frames = video_frames(args.video, args.every, args.start, args.end)
    first = next(frames, None)
    if first is None:
        sys.exit(f"No frames read from {args.video}")

    # Pass 1 only once per camera: on the video's first frame, unless regions are saved.
    if path.exists() and not args.force:
        partition = load_partition(args)
        print(f"Using {len(partition['regions'])} saved regions for {args.camera!r} "
              f"(pass 1 skipped)", file=sys.stderr)
    else:
        print(f"Finding shelves on the first frame ({first[0]:.0f}s):", file=sys.stderr)
        kept, _ = detect_sections(client, args.model, first[1], args.min_confidence,
                                  args.min_row_confidence)
        # One frame, so consensus keeps every shape as is; it just adds the same fields as `detect`.
        regions = link_and_number(consensus([kept], 1.0))
        save_partition(args, regions, [f"{args.video.name}@{first[0]:.0f}s"], first[1])
        partition = load_partition(args)
    if not partition["approved"] and not args.allow_unapproved:
        sys.exit(f"Regions for {args.camera!r} aren't approved. Check "
                 f"{args.partitions_dir / (args.camera + '_overlay.jpg')}, then run "
                 f"`approve --camera {args.camera}` and this command again "
                 f"(or pass --allow-unapproved).")
    kept = saved_regions(partition)

    rows = []
    for t, img in [first, *frames]:
        name = Path(f"{args.video.stem}_{int(t):05d}s.jpg")
        print(f"{name.stem}:", file=sys.stderr)
        result = score_and_write(client, args, img, name, kept, [],
                                 {"camera": args.camera, "video": args.video.name, "seconds": t,
                                  "regions_created": partition["created"]}, print_json=False)
        agg = result["aggregate"]
        rows.append({"frame": name.stem, "seconds": round(t, 1), **agg})
        if any(r.get("error") == "not scored: quota exhausted" for r in result["regions"]):
            print("Stopping: Gemini quota exhausted. Frames scored so far are saved.", file=sys.stderr)
            break

    summary = args.out_dir / f"{args.video.stem}_summary.csv"
    with open(summary, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scored {len(rows)} frames; per-frame JSON in {args.out_dir}/, summary: {summary}")
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
              f"polygons may not line up", file=sys.stderr)

    kept = saved_regions(partition)
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
    scoring.add_argument("--pad", type=float, default=0.05, help="Crop padding as a fraction of size")
    scoring.add_argument("--save-crops", action="store_true", help="Save each crop next to the output")

    # Pass-1 confidence is self-reported by the model, so treat the threshold as a knob to tune.
    conf = argparse.ArgumentParser(add_help=False)
    conf.add_argument("--min-confidence", type=float, default=0.6,
                      help="Drop pass-1 sections below this confidence")
    # Rows are only considered inside sections that already passed --min-confidence, and are
    # smaller and harder to make out, so they get a slightly lower bar.
    conf.add_argument("--min-row-confidence", type=float, default=0.5,
                      help="Drop shelf rows below this confidence")

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

    p = sub.add_parser("video", parents=[common, scoring, conf],
                       help="Find shelves on a video's first frame, then score frames through it")
    p.add_argument("video", type=Path)
    p.add_argument("--camera", help="Camera name (default: the video's file name)")
    p.add_argument("--partitions-dir", type=Path, default=Path("partitions"))
    p.add_argument("--every", type=float, default=2, help="Seconds between scored frames")
    p.add_argument("--start", type=float, default=0, help="Start time in seconds")
    p.add_argument("--end", type=float, help="End time in seconds (default: end of video)")
    p.add_argument("--force", action="store_true", help="Re-find shelves even if regions are saved")
    p.add_argument("--allow-unapproved", action="store_true",
                   help="Score straight away without approving newly found regions")
    p.set_defaults(func=cmd_video)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except QuotaExhausted as e:  # e.g. pass 1 or `detect` hit the limit before anything was saved
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
