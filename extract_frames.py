#!/usr/bin/env python3
"""Extract frames from a local video file for testing shelf_metrics.py.

Setup:
    pip install opencv-python

Usage:
    python extract_frames.py clip.mp4 --every 10
    python extract_frames.py clip.mp4 --every 5 --start 30 --end 120 --out test_frames
"""
import argparse
import sys
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path, help="Path to a local video file")
    parser.add_argument("--every", type=float, default=10, help="Seconds between frames")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--end", type=float, help="End time in seconds (default: end of video)")
    parser.add_argument("--out", type=Path, default=Path("test_frames"))
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"Can't open {args.video}", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    end = min(args.end, duration) if args.end else duration

    args.out.mkdir(parents=True, exist_ok=True)
    saved = 0
    t = args.start
    while t < end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(args.out / f"{args.video.stem}_{int(t):05d}s.jpg"), frame)
        saved += 1
        t += args.every
    cap.release()

    print(f"Wrote {saved} frames to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
