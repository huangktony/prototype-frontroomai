#!/usr/bin/env python3
"""Run shelf_metrics.analyse over a folder of frames and write one CSV row per run.

Usage:
    python run_batch.py test_frames
    python run_batch.py test_frames --limit 10 --repeats 3 --out results/summary.csv

--repeats runs each frame several times so you can see how much the numbers vary.
"""
import argparse
import csv
import os
import statistics
import sys
from pathlib import Path

from shelf_metrics import DEFAULT_MODEL, analyse

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FIELDS = [
    "overall_fill_percent", "out_of_stock_gap_count", "stock_status",
    "estimated_facings", "distinct_products_estimate", "tidiness_score",
    "price_labels_visible", "people_occluding_shelf", "confidence",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="Folder of frames")
    parser.add_argument("--out", type=Path, default=Path("results/summary.csv"))
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--limit", type=int, help="Only process the first N frames")
    parser.add_argument("--repeats", type=int, default=1, help="Runs per frame")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        return 1

    frames = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        print(f"No images found in {args.folder}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gaps_by_frame: dict[str, list[int]] = {}
    failures = 0
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "run", *FIELDS, "notes", "error"])
        writer.writeheader()
        for i, frame in enumerate(frames, 1):
            for run in range(1, args.repeats + 1):
                row = {"image": frame.name, "run": run}
                try:
                    metrics = analyse(str(frame), args.model, api_key).model_dump()
                    row.update({k: metrics[k] for k in FIELDS})
                    row["notes"] = metrics["notes"]
                    gaps_by_frame.setdefault(frame.name, []).append(metrics["out_of_stock_gap_count"])
                except Exception as e:  # keep going; record the failure in the CSV
                    row["error"] = f"{type(e).__name__}: {e}"[:300]
                    failures += 1
                writer.writerow(row)
                f.flush()
                print(f"[{i}/{len(frames)}] {frame.name} run {run}: "
                      f"{'ERROR' if 'error' in row else 'ok'}", file=sys.stderr)

    print(f"Wrote {args.out} ({len(frames) * args.repeats} runs, {failures} failed)")
    if args.repeats > 1:
        spreads = [statistics.pstdev(g) for g in gaps_by_frame.values() if len(g) > 1]
        if spreads:
            print(f"Gap-count std dev across repeats, averaged over frames: "
                  f"{statistics.mean(spreads):.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
