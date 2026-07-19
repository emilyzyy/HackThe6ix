"""Group identified pieces into an inventory; merge multi-photo scans;
draw the debug overlay.
"""
from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from pipeline.color import LegoColor
from pipeline.detect import Box
from pipeline.identify import Identification


def build_inventory(items: list[tuple[Identification, LegoColor]]) -> list[dict]:
    counts: Counter = Counter()
    meta: dict[tuple, tuple[str, str]] = {}
    for ident, color in items:
        key = (ident.part_id, color.id, color.name)
        counts[key] += 1
        meta[key] = (ident.name, color.name)
    inventory = [
        {
            "part_id": part_id,
            "name": meta[(part_id, color_id, color_name)][0],
            "color": color_name,
            "color_id": color_id,
            "quantity": qty,
        }
        for (part_id, color_id, color_name), qty in counts.items()
    ]
    inventory.sort(key=lambda e: (-e["quantity"], e["name"], e["color"]))
    return inventory


def merge_inventories(*inventories: list[dict]) -> list[dict]:
    counts: Counter = Counter()
    meta: dict[tuple, tuple[str, str]] = {}
    for inv in inventories:
        for e in inv:
            key = (e["part_id"], e["color_id"], e["color"])
            counts[key] += e["quantity"]
            meta[key] = (e["name"], e["color"])
    merged = [
        {
            "part_id": part_id,
            "name": meta[(part_id, color_id, color_name)][0],
            "color": color_name,
            "color_id": color_id,
            "quantity": qty,
        }
        for (part_id, color_id, color_name), qty in counts.items()
    ]
    merged.sort(key=lambda e: (-e["quantity"], e["name"], e["color"]))
    return merged


def draw_debug(
    img: np.ndarray,
    boxes: list[Box],
    labels: list[str],
    clusters: list[Box] | None = None,
) -> np.ndarray:
    out = img.copy()
    thick = max(2, img.shape[1] // 800)
    font_scale = max(0.5, img.shape[1] / 2000)
    for (x0, y0, x1, y1), label in zip(boxes, labels):
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), thick)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick
        )
        ty = y0 - 8 if y0 - th - 12 > 0 else y1 + th + 8
        cv2.rectangle(out, (x0, ty - th - 4), (x0 + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (x0 + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 255, 0), thick)
    return out
