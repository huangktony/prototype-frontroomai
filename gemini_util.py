"""Shared helpers for calling Gemini with a structured (pydantic) response."""
import mimetypes
import os
import re
import sys
import time

from google import genai
from google.genai import errors, types
from pydantic import BaseModel

DEFAULT_MODEL = "gemini-3.8-flash"
MAX_ATTEMPTS = 5
MAX_RATE_LIMIT_WAIT = 120  # seconds; longer 429 waits fail instead of hanging


def make_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY first.")
    return genai.Client(api_key=api_key)


def read_image(path: str) -> tuple[bytes, str]:
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        raise ValueError(f"{path} does not look like an image file")
    with open(path, "rb") as f:
        return f.read(), mime


def generate[T: BaseModel](client: genai.Client, model: str, image: bytes, mime: str,
                           prompt: str, schema: type[T]) -> T:
    """Send one image + prompt and parse the reply into `schema`, retrying 5xx errors."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=image, mime_type=mime), prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.0,
                ),
            )
            break
        except (errors.ServerError, errors.ClientError) as e:
            # Retry 5xx (e.g. 503 "high demand") and 429 rate limits; other 4xx are our fault.
            if attempt == MAX_ATTEMPTS or (isinstance(e, errors.ClientError) and e.code != 429):
                raise
            wait = 5 * 2 ** (attempt - 1)
            if e.code == 429:  # the free tier is a few requests per minute; wait as told
                match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                wait = int(match.group(1)) + 1 if match else 60
                if wait > MAX_RATE_LIMIT_WAIT:  # a daily quota, not a per-minute one
                    raise RuntimeError(f"Gemini quota exhausted; it resets in about "
                                       f"{wait // 60} minutes. Use a paid key or wait.") from e
            print(f"Gemini error {e.code}; retrying in {wait:.0f}s "
                  f"({attempt}/{MAX_ATTEMPTS})", file=sys.stderr)
            time.sleep(wait)
    if response.parsed is None:
        raise RuntimeError(f"Gemini returned no parseable output: {response.text!r}")
    return response.parsed
