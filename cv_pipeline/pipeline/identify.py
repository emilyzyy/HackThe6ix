"""Identify LEGO crops via the free Brickognize API.

- Responses cached in .cache/ keyed by SHA1 of the crop's JPEG bytes.
- Global rate limit (~1 request/sec) shared across a small worker pool.
- 3 attempts with exponential backoff on 429/5xx/network errors.
- Degrades to "unknown brick" instead of raising when the API is down.

Independently runnable: python -m pipeline.identify crop.jpg
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

API_URL = "https://api.brickognize.com/predict/"
MIN_SCORE = 0.7
ATTEMPTS = 3
BACKOFF_BASE = 1.0  # seconds: 1, 2, 4


@dataclass(frozen=True)
class Identification:
    part_id: str | None
    name: str
    category: str | None
    score: float
    # Top raw API candidates ({part_id, name, score}), regardless of
    # MIN_SCORE — downstream advisories (stud check) re-rank within these.
    candidates: tuple = ()


UNKNOWN = Identification(None, "unknown brick", None, 0.0)

_session: requests.Session | None = None
_session_lock = threading.Lock()

_rate_lock = threading.Lock()
_last_request = 0.0


def _get_session() -> requests.Session:
    global _session
    with _session_lock:
        if _session is None:
            _session = requests.Session()
        return _session


def _throttle(min_interval: float) -> None:
    """Block until at least min_interval has passed since the last request."""
    global _last_request
    with _rate_lock:
        wait = _last_request + min_interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()


def _parse(payload: dict) -> Identification:
    items = payload.get("items") or []
    if not items:
        return UNKNOWN
    candidates = tuple(
        {"part_id": item["id"], "name": item["name"],
         "score": float(item.get("score", 0.0)),
         "img_url": item.get("img_url")}
        for item in items[:5]
    )
    top = items[0]
    score = float(top.get("score", 0.0))
    if score < MIN_SCORE:
        return Identification(None, "unknown brick", None, score,
                              candidates=candidates)
    return Identification(top["id"], top["name"], top.get("category"), score,
                          candidates=candidates)


def _post_with_retries(
    jpeg: bytes, session: requests.Session, min_interval: float
) -> dict | None:
    for attempt in range(ATTEMPTS):
        try:
            _throttle(min_interval)
            resp = session.post(
                API_URL,
                files={"query_image": ("crop.jpg", jpeg, "image/jpeg")},
                timeout=30,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"retryable status {resp.status_code}")
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt < ATTEMPTS - 1:
                time.sleep(BACKOFF_BASE * 2**attempt)
    return None


def identify_crop(
    crop_bgr: np.ndarray,
    *,
    session: requests.Session | None = None,
    cache_dir: Path = Path(".cache"),
    min_interval: float = 1.0,
) -> Identification:
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        return UNKNOWN
    jpeg = buf.tobytes()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{hashlib.sha1(jpeg).hexdigest()}.json"
    if cache_file.exists():
        return _parse(json.loads(cache_file.read_text()))

    payload = _post_with_retries(jpeg, session or _get_session(), min_interval)
    if payload is None:  # API unreachable after retries: degrade, don't cache
        return UNKNOWN
    cache_file.write_text(json.dumps(payload))
    return _parse(payload)


def identify_all(
    crops: list[np.ndarray],
    *,
    workers: int = 4,
    min_interval: float = 1.0,
    cache_dir: Path = Path(".cache"),
) -> list[Identification]:
    """Identify crops concurrently (order preserved). The pool overlaps HTTP
    latency while _throttle keeps the global request rate at ~1/min_interval."""
    if not crops:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda c: identify_crop(
                    c, cache_dir=cache_dir, min_interval=min_interval
                ),
                crops,
            )
        )


if __name__ == "__main__":
    import sys

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"could not read image: {sys.argv[1]}")
    result = identify_crop(img)
    print(result)
