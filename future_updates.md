# Planned Updates: Shelf Fullness Pipeline

## Current State

Two phases, both plain Gemini prompts:

1. Segment shelves into polygons
2. Crop each polygon and judge its fullness

Problems: polygons are loose and vary between runs, and fullness scores are inconsistent.

## Update 1: Add SAM 2 Between the Two Gemini Phases

Status: [ ] not started

- Phase 1 Gemini returns bounding boxes only (not polygons).
- SAM 2 takes each box and produces a pixel-accurate mask.
- Convert each mask to a polygon, then pass it to Phase 2 as before.
- Gemini stays in both phases. SAM 2 is only the refinement layer between them.

## Update 2: Median Polygons Across Multiple Frames

Status: [ ] not started

- Segmentation runs once per camera at setup, not per frame.
- Run the Gemini + SAM 2 step on many frames (e.g. 50 to 200, different times of day, no people in the aisle).
- Take the median (or mean) polygon per shelf across frames.
- Reject polygon sets that overlap, have inconsistent shelf heights, or are unstable across frames. Retry, and flag cameras that keep failing.
- Cache the final polygons per camera. Re-run setup if a camera is bumped or the layout changes.

## Update 3: Learn Each Shelf's "Full" Baseline

Status: [ ] not started

- Collect crops per shelf over days or weeks. Restocks mean every shelf eventually reaches full.
- Pick the fullest crops (e.g. top percentile) and store a representative one as that shelf's baseline.
- Phase 2 Gemini compares the current crop against the baseline instead of judging fullness in the abstract.
- Refresh baselines periodically as products or layouts change.
- Goal: no manual setup per store, so it scales to stores that are not identical.

## Open Questions

- How many frames are needed for a stable median polygon?
- Pick the baseline by top percentile of scores, or by detecting restock events?
- Does SAM 2 segment shelves cleanly, or do we need edge snapping?
