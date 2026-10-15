# prototype-frontroomai

## The idea

Stores already have CCTV cameras pointed at their shelves. This prototype tests whether a vision-language model (Gemini) can look at a single CCTV frame and turn it into **shelf metrics**, with no special hardware or per-shelf sensors:

- how full the shelf is (overall and per row)
- how many out-of-stock gaps are visible
- rough facings and distinct-product counts
- tidiness, price labels visible, whether a person is blocking the shelf
- a confidence score and notes when the image is too blurry or occluded to trust

If the numbers hold up on real CCTV-quality frames, this could feed restock alerts or shelf-compliance reporting. Right now we're only answering: **how accurate and usable are Gemini's estimates on CCTV footage?**

## How it works

```
CCTV video -> extract_frames.py -> frames (.jpg)
                                     |
            pass 1 (once per camera) detect_shelves.py -> layouts/<camera>.json
                                     |
            pass 2 (every frame)     shelf_metrics.py: straighten each unit -> Gemini -> JSON metrics
```

- [extract_frames.py](extract_frames.py) pulls a frame every N seconds from a local video file.
- [detect_shelves.py](detect_shelves.py) (pass 1) asks Gemini for every shelving unit and its levels as 4-corner polygons of their front faces, plus a depth rank (nearest the camera = 1). Where a nearer unit stands in front of part of another, that unit is recorded in `hidden_by`. The result is saved to `layouts/<camera>.json` (schema: `CameraLayout`), plus a `layouts/<camera>.jpg` overlay to check by eye. The cameras are fixed, so this runs once per camera. The camera ID is the video name in the frame filename. If a shape is wrong, edit the JSON by hand (corners are `[y, x]`, 0-1000, clockwise from top-left). Re-detect with `--force`.
- [shelf_metrics.py](shelf_metrics.py) (pass 2) loads the camera's layout, perspective-warps each unit's polygon to an upright rectangle (so angled shelves are seen front-on), greys out nearer units in front of it, outlines its levels as L1, L2, ..., and sends one Gemini call per unit (in parallel). Output is per-unit, per-level metrics (`ShelfMetrics`) plus an area-weighted overall fill. If the camera has no layout yet, it runs pass 1 on that frame first.
- [present_layout.py](present_layout.py) draws the visible part of each saved polygon as transparent yellow highlights on a frame, for slides (`presentation/`).

## Setup

```
python3 -m venv .venv && source .venv/bin/activate
pip install google-genai pydantic opencv-python yt-dlp
export GEMINI_API_KEY=...   # get one at https://aistudio.google.com/apikey
```

## Get test footage

We use two short public YouTube clips of small stores filmed by overhead CCTV cameras:

| Clip | Link |
| --- | --- |
| 4WXZTF7EWL0 (40s) | https://www.youtube.com/watch?v=4WXZTF7EWL0 |
| -1bRhYjw1qE (60s) | https://www.youtube.com/watch?v=-1bRhYjw1qE |

```
mkdir -p videos
yt-dlp -f "136/bestvideo[ext=mp4][height<=720]" -o "videos/%(id)s.%(ext)s" \
  "https://www.youtube.com/watch?v=4WXZTF7EWL0" "https://www.youtube.com/watch?v=-1bRhYjw1qE"

for v in videos/*.mp4; do python extract_frames.py "$v" --every 2 --out test_frames; done
```

This gives about 50 frames. `--every` sets seconds between frames; `--start` and `--end` limit the window. The video-only download needs no ffmpeg. If yt-dlp complains about a JavaScript runtime, add `--js-runtimes node`.

The videos belong to their uploaders and show real people. They're for private prototyping only: don't commit or redistribute the videos or frames (they're git-ignored).

## Run

Map the shelves once per camera, on a frame with nobody blocking them, and check the overlay:
```
python detect_shelves.py test_frames/4WXZTF7EWL0_00016s.jpg
python detect_shelves.py test_frames/-1bRhYjw1qE_00012s.jpg
```

One frame (add `--save-crops crops/` to see exactly what Gemini was sent):
```
python shelf_metrics.py test_frames/4WXZTF7EWL0_00010s.jpg --out results/4WXZTF7EWL0_00010s.json
```

Slide image of the detected regions (`--no-levels` for whole units only):
```
python present_layout.py test_frames/4WXZTF7EWL0_00016s.jpg
```

All frames:
```
mkdir -p results
for f in test_frames/*.jpg; do python shelf_metrics.py "$f" --out "results/$(basename "$f" .jpg).json"; done
```

Use `--model` or `GEMINI_MODEL` to change the Gemini model (default `gemini-3.8-flash`; older models like `gemini-2.5-flash` are no longer available to new accounts).

## Caveats

These are model estimates from a single frame, not measurements. Spot-check them against a few frames you've counted by hand before trusting any number. Many frames show people blocking shelves or shelves at steep angles, which is the point of testing on CCTV.
