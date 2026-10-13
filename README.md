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
CCTV video  ->  extract_frames.py  ->  frames (.jpg)  ->  shelf_metrics.py  ->  Gemini  ->  JSON metrics
```

- [extract_frames.py](extract_frames.py) pulls a frame every N seconds from a local video file.
- [shelf_metrics.py](shelf_metrics.py) sends one frame to Gemini and prints structured JSON (the schema is the `ShelfMetrics` class in the file; edit it to change the metrics).
- [run_batch.py](run_batch.py) runs `shelf_metrics.py` over a whole folder of frames and writes one CSV for comparing them.

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

One frame:
```
python shelf_metrics.py test_frames/4WXZTF7EWL0_00010s.jpg --out results/4WXZTF7EWL0_00010s.json
```

All frames, one CSV row per frame ([run_batch.py](run_batch.py)):
```
python run_batch.py test_frames --limit 10
```
Start with `--limit 10`, since every frame is one API call. The CSV lands in `results/summary.csv` (change with `--out`) and includes the metrics, Gemini's notes, and an `error` column for frames that failed. Failed frames don't stop the run. `--repeats 3` runs each frame several times and prints how much the gap count varies between runs, which is a quick consistency check. To compare against reality, hand-count gaps on 10-15 frames and add a `manual_gaps` column next to Gemini's `out_of_stock_gap_count`.

Or loop over single frames, saving one JSON file per frame:
```
mkdir -p results
for f in test_frames/*.jpg; do python shelf_metrics.py "$f" --out "results/$(basename "$f" .jpg).json"; done
```

Use `--model` or `GEMINI_MODEL` to change the Gemini model (default `gemini-3.8-flash`; older models like `gemini-2.5-flash` are no longer available to new accounts).

## Caveats

These are model estimates from a single frame, not measurements. Spot-check them against a few frames you've counted by hand before trusting any number. Many frames show people blocking shelves or shelves at steep angles, which is the point of testing on CCTV.
