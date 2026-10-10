#!/usr/bin/env python3
"""Send a CCTV frame of a store shelf to Gemini and get structured shelf metrics.

Setup:
    pip install google-genai pydantic
    export GEMINI_API_KEY=...

Usage:
    python shelf_metrics.py frame.jpg
    python shelf_metrics.py frame.jpg --model gemini-2.5-pro --out result.json
"""
import argparse
import json
import mimetypes
import os
import sys
import time
from typing import Literal

from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_ATTEMPTS = 5

PROMPT = """You are analysing a single frame from a CCTV camera in a retail store.
The frame shows one or more product shelves. Estimate retail shelf metrics from
what is visible. Be conservative: if the image quality, angle, or occlusion
(people, carts, glare) prevents a reliable estimate, lower the confidence and
say why in `notes`. Do not invent products you cannot see. Percentages are 0-100.
Split the shelving into rows from top to bottom (row 1 = top shelf)."""


class ShelfRow(BaseModel):
    row: int = Field(description="1 = top shelf, counting down")
    fill_percent: float = Field(description="How full the shelf row is, 0-100")
    empty_gap_count: int = Field(description="Number of visible out-of-stock gaps")
    product_description: str = Field(description="What is stocked on this row, briefly")


class ShelfMetrics(BaseModel):
    shelf_visible: bool = Field(description="Whether a shelf is clearly visible at all")
    overall_fill_percent: float = Field(description="Overall shelf fullness, 0-100")
    out_of_stock_gap_count: int = Field(description="Total visible empty gaps")
    stock_status: Literal["well_stocked", "partially_stocked", "low_stock", "empty", "unknown"]
    estimated_facings: int = Field(description="Approximate number of product facings visible")
    distinct_products_estimate: int = Field(description="Approximate number of distinct SKUs")
    tidiness_score: int = Field(description="1 (messy, fallen/misplaced items) to 5 (neat, faced)")
    price_labels_visible: bool
    people_occluding_shelf: bool
    rows: list[ShelfRow]
    confidence: float = Field(description="Overall confidence in these estimates, 0-1")
    notes: str = Field(description="Caveats, image quality issues, anything notable")


def analyse(image_path: str, model: str, api_key: str) -> ShelfMetrics:
    mime, _ = mimetypes.guess_type(image_path)
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"{image_path} does not look like an image file")
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    client = genai.Client(api_key=api_key)
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime), PROMPT],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ShelfMetrics,
                    temperature=0.0,
                ),
            )
            break
        except errors.ServerError as e:  # 5xx, e.g. 503 "high demand"
            if attempt == MAX_ATTEMPTS:
                raise
            wait = 5 * 2 ** (attempt - 1)
            print(f"Gemini server error ({e.code}); retrying in {wait}s "
                  f"({attempt}/{MAX_ATTEMPTS})", file=sys.stderr)
            time.sleep(wait)
    if response.parsed is None:
        raise RuntimeError(f"Gemini returned no parseable output: {response.text!r}")
    return response.parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="Path to a CCTV frame (jpg/png/webp)")
    parser.add_argument("--model", default=os.environ.get("GEMINI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--out", help="Also write the JSON result to this file")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY first.", file=sys.stderr)
        return 1

    metrics = analyse(args.image, args.model, api_key)
    output = json.dumps(metrics.model_dump(), indent=2)
    print(output)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
