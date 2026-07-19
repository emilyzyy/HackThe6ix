"""Detect individual LEGO pieces in a photo.

Two detectors:
- "yolo": ultralytics model from models/lego_yolo.pt (preferred when present)
- "cv":   classical fallback for plain backgrounds. Foreground = pixels that
          differ from a local background estimate. Color-consistent interior
          seeds split nearby pieces with marker-controlled watershed.
"auto" picks yolo when the weights file exists, else cv.

Independently runnable: python -m pipeline.detect photo.jpg [--detector cv]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.instance_split import split_component

YOLO_WEIGHTS = Path("models/lego_yolo.pt")
_PROCESS_MAX_DIM = 1600  # downscale before mask analysis for speed
_MIN_AREA_FRAC = 0.001
_BG_DIFF_THRESH = 22

Box = tuple[int, int, int, int]


@dataclass
class Detections:
    boxes: list[Box] = field(default_factory=list)  # individual pieces
    clusters: list[Box] = field(default_factory=list)  # legacy API compatibility


def detect(img: np.ndarray, method: str = "auto") -> Detections:
    if method == "auto":
        method = "yolo" if YOLO_WEIGHTS.exists() else "cv"
    if method == "yolo":
        result = _detect_yolo(img)
    elif method == "cv":
        result = _detect_cv(img)
    else:
        raise ValueError(f"unknown detector: {method}")
    return result


def detect_pieces(img: np.ndarray, method: str = "auto") -> list[Box]:
    return detect(img, method).boxes


def _background_kernel_size(height: int, width: int) -> int:
    """Odd median-blur kernel kept large enough for cropped long bricks."""
    return max(41, (max(height, width) // 20) | 1)


def _detect_cv(img: np.ndarray) -> Detections:
    scale = min(1.0, _PROCESS_MAX_DIM / max(img.shape[:2]))
    small = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1.0 else img
    h, w = small.shape[:2]
    img_area = h * w

    # Primary background estimate: median blur with a kernel much larger
    # than piece details — robust on real photos (noise/shading make pieces
    # differ from it). Median blur is edge-preserving though, so perfectly
    # flat renders can vanish from the diff; if no boxes emerge, retry with
    # a per-tile median background surface, which handles the flat case.
    ksize = _background_kernel_size(h, w)
    smallf = small.astype(np.float32)

    def mask_from(diff: np.ndarray) -> np.ndarray:
        mask = (diff > _BG_DIFF_THRESH).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        return mask

    def boxes_from(mask: np.ndarray) -> list[Box]:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        found: list[Box] = []
        for component_id in range(1, count):
            x, y, bw, bh, _ = stats[component_id]
            if bw * bh < _MIN_AREA_FRAC * img_area:
                continue
            # Preserve border-touching LEGO plates. Only reject a thin strip
            # that hugs almost an entire image edge (a crop/table boundary).
            vertical_strip = bh >= 0.9 * h and bw <= 0.08 * w
            horizontal_strip = bw >= 0.9 * w and bh <= 0.08 * h
            touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
            shallow_edge_sliver = (
                touches_border
                and min(bw, bh) <= 0.02 * max(h, w)
                and max(bw, bh) >= 3 * min(bw, bh)
            )
            if vertical_strip or horizontal_strip or shallow_edge_sliver:
                continue
            component = labels == component_id
            found.extend(split_component(small, component,
                                         _MIN_AREA_FRAC * img_area))
        return found

    bg_blur = cv2.medianBlur(small, ksize)
    mask = mask_from(np.linalg.norm(smallf - bg_blur.astype(np.float32), axis=2))
    raw = boxes_from(mask)
    if not raw:
        tiles_y, tiles_x = 12, 16
        th, tw = h // tiles_y, w // tiles_x
        grid = np.median(
            small[: th * tiles_y, : tw * tiles_x]
            .reshape(tiles_y, th, tiles_x, tw, 3)
            .transpose(0, 2, 1, 3, 4)
            .reshape(tiles_y, tiles_x, -1, 3),
            axis=2,
        ).astype(np.float32)
        bg_tile = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = mask_from(np.linalg.norm(smallf - bg_tile, axis=2))
        raw = boxes_from(mask)

    result = Detections(boxes=raw)

    def unscale(b: Box) -> Box:
        return tuple(int(v / scale) for v in b)  # type: ignore[return-value]

    result.boxes = [unscale(b) for b in result.boxes]
    result.clusters = [unscale(b) for b in result.clusters]
    return result


def _detect_yolo(img: np.ndarray) -> Detections:
    from ultralytics import YOLO  # lazy: heavy import

    model = YOLO(str(YOLO_WEIGHTS))
    results = model.predict(img, conf=0.25, device="cpu", verbose=False)
    boxes: list[Box] = []
    for xyxy in results[0].boxes.xyxy.tolist():
        x0, y0, x1, y1 = (int(v) for v in xyxy)
        boxes.append((x0, y0, x1, y1))
    return Detections(boxes=boxes)


def crop_box(img: np.ndarray, box: Box, pad: float = 0.10) -> np.ndarray:
    x0, y0, x1, y1 = box
    px = int((x1 - x0) * pad)
    py = int((y1 - y0) * pad)
    h, w = img.shape[:2]
    return img[
        max(0, y0 - py) : min(h, y1 + py),
        max(0, x0 - px) : min(w, x1 + px),
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("--detector", default="auto", choices=["auto", "yolo", "cv"])
    args = parser.parse_args()

    img = cv2.imread(args.photo)
    if img is None:
        raise SystemExit(f"could not read image: {args.photo}")
    dets = detect(img, method=args.detector)
    print(f"{len(dets.boxes)} pieces, {len(dets.clusters)} clusters")
    debug = img.copy()
    for i, (x0, y0, x1, y1) in enumerate(dets.boxes):
        cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 255, 0), 6)
        cv2.putText(debug, str(i), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 0), 4)
    cv2.imwrite("detect_debug.jpg", debug)
    print("wrote detect_debug.jpg")
