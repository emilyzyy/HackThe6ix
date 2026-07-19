"""Conservatively split one foreground component into LEGO instances.

Foreground geometry can join nearby pieces through shadows or morphology.
Credible, color-consistent interior regions provide watershed markers; when
the evidence is ambiguous, the component deliberately remains one piece.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.segmentation import watershed

from pipeline.color import PALETTE

Box = tuple[int, int, int, int]

_MAX_SEED_DELTA_E = 18.0
_MIN_SEED_AREA = 20
_MIN_SEED_FRAC = 0.035
_MIN_SEED_BG_DISTANCE = 20.0
_PALETTE_RGB = np.array([color.rgb for color in PALETTE], dtype=np.float32) / 255.0
_PALETTE_LAB = rgb2lab(_PALETTE_RGB.reshape(-1, 1, 3)).reshape(-1, 3)

# Nearby palette shades frequently occur on one physical piece because of
# highlights and shadows. They may share a marker only when their spatial
# bounding boxes overlap; disconnected same-color pieces remain independent.
_FAMILY_BY_NAME = {
    "red": "red", "dark red": "red", "pink": "red", "magenta": "red",
    "orange": "orange", "brown": "orange", "dark brown": "orange",
    "tan": "orange",
    "yellow": "yellow", "lime": "yellow",
    "green": "green", "dark green": "green",
    "teal": "blue", "cyan": "blue", "light blue": "blue",
    "blue": "blue", "dark blue": "blue", "purple": "blue",
    "white": "light-neutral", "beige": "light-neutral",
    "light gray": "light-neutral",
    "gray": "dark-neutral", "dark gray": "dark-neutral",
    "black": "dark-neutral",
}


@dataclass
class _Seed:
    mask: np.ndarray
    bbox: Box
    family: str


def _bbox(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _boxes_overlap(a: Box, b: Box) -> bool:
    return (min(a[2], b[2]) > max(a[0], b[0])
            and min(a[3], b[3]) > max(a[1], b[1]))


def _normalized_rgb(image_bgr: np.ndarray, component: np.ndarray) -> np.ndarray:
    rgb = image_bgr[..., ::-1].astype(np.float32)
    outside = rgb[~component]
    if len(outside):
        bg = np.median(outside, axis=0)
        if float(bg.max() - bg.min()) <= 30 and float(bg.max()) >= 100:
            rgb *= np.minimum(2.0, 245.0 / np.maximum(bg, 1.0))
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _candidate_seeds(image_bgr: np.ndarray, component: np.ndarray) -> list[_Seed]:
    interior = cv2.erode(
        component.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    component_area = int(component.sum())
    min_seed_area = max(_MIN_SEED_AREA, int(round(_MIN_SEED_FRAC * component_area)))
    if int(interior.sum()) < min_seed_area:
        return []

    outside = image_bgr[~component]
    background_bgr = (np.median(outside, axis=0) if len(outside)
                      else np.zeros(3, dtype=np.float32))
    rgb = _normalized_rgb(image_bgr, component)
    lab = rgb2lab(rgb.astype(np.float32) / 255.0)
    pixels = lab[interior]
    distances = np.stack(
        [deltaE_ciede2000(pixels, ref[None, :]) for ref in _PALETTE_LAB], axis=1
    )
    nearest = distances.argmin(axis=1)
    confident = distances[np.arange(len(pixels)), nearest] <= _MAX_SEED_DELTA_E

    labels = np.full(component.shape, -1, dtype=np.int16)
    ys, xs = np.nonzero(interior)
    labels[ys[confident], xs[confident]] = nearest[confident]

    seeds: list[_Seed] = []
    for palette_index, color in enumerate(PALETTE):
        color_mask = (labels == palette_index).astype(np.uint8)
        count, components = cv2.connectedComponents(color_mask)
        for component_id in range(1, count):
            candidate = components == component_id
            if int(candidate.sum()) < min_seed_area:
                continue
            candidate_bgr = np.median(image_bgr[candidate], axis=0)
            if np.linalg.norm(candidate_bgr - background_bgr) < _MIN_SEED_BG_DISTANCE:
                continue
            seeds.append(_Seed(
                candidate, _bbox(candidate),
                _FAMILY_BY_NAME.get(color.name, color.name),
            ))
    return seeds


def _merge_spatial_variants(seeds: list[_Seed]) -> list[np.ndarray]:
    parents = list(range(len(seeds)))

    def find(i: int) -> int:
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parents[rj] = ri

    for i, first in enumerate(seeds):
        for j in range(i + 1, len(seeds)):
            second = seeds[j]
            ix0 = max(first.bbox[0], second.bbox[0])
            iy0 = max(first.bbox[1], second.bbox[1])
            ix1 = min(first.bbox[2], second.bbox[2])
            iy1 = min(first.bbox[3], second.bbox[3])
            intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
            first_area = ((first.bbox[2] - first.bbox[0])
                          * (first.bbox[3] - first.bbox[1]))
            second_area = ((second.bbox[2] - second.bbox[0])
                           * (second.bbox[3] - second.bbox[1]))
            overlap = intersection / max(1, min(first_area, second_area))
            first_aspect = max(
                first.bbox[2] - first.bbox[0], first.bbox[3] - first.bbox[1]
            ) / max(1, min(
                first.bbox[2] - first.bbox[0], first.bbox[3] - first.bbox[1]
            ))
            second_aspect = max(
                second.bbox[2] - second.bbox[0], second.bbox[3] - second.bbox[1]
            ) / max(1, min(
                second.bbox[2] - second.bbox[0], second.bbox[3] - second.bbox[1]
            ))
            adjacent = bool(np.any(
                cv2.dilate(first.mask.astype(np.uint8),
                           np.ones((3, 3), np.uint8)).astype(bool)
                & second.mask
            ))
            same_shaded_region = adjacent and (
                overlap >= 0.50
                or (overlap >= 0.05 and max(first_aspect, second_aspect) >= 3.0)
            )
            if ((first.family == second.family
                 and _boxes_overlap(first.bbox, second.bbox))
                    or same_shaded_region):
                union(i, j)

    merged: dict[int, np.ndarray] = {}
    for i, seed in enumerate(seeds):
        root = find(i)
        if root not in merged:
            merged[root] = np.zeros(seed.mask.shape, dtype=bool)
        merged[root] |= seed.mask
    return list(merged.values())


def split_component(
    image_bgr: np.ndarray, component_mask: np.ndarray, min_area: float
) -> list[Box]:
    """Return boxes for credible instances within one connected component."""
    component = component_mask.astype(bool)
    if not component.any():
        return []
    outside = image_bgr[~component]
    background_bgr = (np.median(outside, axis=0) if len(outside)
                      else np.zeros(3, dtype=np.float32))
    contrast = np.linalg.norm(
        image_bgr.astype(np.float32) - background_bgr.astype(np.float32), axis=2
    ) >= _MIN_SEED_BG_DISTANCE
    supported_component = component & contrast
    fallback_mask = (supported_component if int(supported_component.sum()) >= min_area
                     else component)
    fallback = [_bbox(fallback_mask)]
    markers_masks = _merge_spatial_variants(_candidate_seeds(image_bgr, component))
    if len(markers_masks) < 2:
        return fallback

    markers = np.zeros(component.shape, dtype=np.int32)
    for marker_id, marker_mask in enumerate(markers_masks, start=1):
        markers[marker_mask] = marker_id
    distance = cv2.distanceTransform(component.astype(np.uint8), cv2.DIST_L2, 5)
    labels = watershed(-distance, markers, mask=component)

    boxes: list[Box] = []
    for marker_id in range(1, len(markers_masks) + 1):
        region = labels == marker_id
        supported_region = region & contrast
        # A low-contrast morphology bridge can make one watershed region wrap
        # around another piece. Once that bridge is removed, keep only the
        # supported island containing this marker.
        _, islands = cv2.connectedComponents(supported_region.astype(np.uint8))
        marker_islands = np.unique(islands[markers_masks[marker_id - 1]])
        marker_islands = marker_islands[marker_islands != 0]
        anchored = (np.isin(islands, marker_islands)
                    if len(marker_islands) else supported_region)
        box_region = (anchored if int(anchored.sum()) >= min_area else region)
        if int(box_region.sum()) >= min_area:
            boxes.append(_bbox(box_region))
    return boxes if len(boxes) >= 2 else fallback
