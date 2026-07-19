"""Full-photo scan: detect -> identify -> color -> inventory.

Shared by cli.py and api.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import cv2

from pipeline.color import dominant_rgb, match_color, normalize_illumination
from pipeline.detect import Box, crop_box, detect
from pipeline.identify import identify_all
from pipeline.inventory import build_inventory


@dataclass
class ScanResult:
    inventory: list[dict] = field(default_factory=list)
    boxes: list[Box] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    clusters: list[Box] = field(default_factory=list)


def scan_image(
    img: np.ndarray,
    detector: str = "auto",
    cache_dir: Path = Path(".cache"),
) -> ScanResult:
    dets = detect(img, method=detector)
    crops = [crop_box(img, b) for b in dets.boxes]
    idents = identify_all(crops, cache_dir=cache_dir)
    # median of a downscaled frame ~= table color; a neutral table doubles
    # as a white reference that cancels dim light and color casts
    thumb = cv2.resize(img, (200, 200))
    b, g, r = np.median(thumb.reshape(-1, 3), axis=0).astype(int)
    bg_rgb = (int(r), int(g), int(b))
    colors = [
        match_color(normalize_illumination(dominant_rgb(c), bg_rgb)) for c in crops
    ]
    labels = [
        f"{ident.name} | {color.name} | {ident.score:.2f}"
        for ident, color in zip(idents, colors)
    ]
    return ScanResult(
        inventory=build_inventory(list(zip(idents, colors))),
        boxes=dets.boxes,
        labels=labels,
        clusters=dets.clusters,
    )
