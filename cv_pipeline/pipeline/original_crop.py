"""Map fused table-canvas boxes into ORIGINAL full-res frame pixels.

Rectified views are used for localization and fusion only; identification
crops are cut from the unwarped original frames so raised pieces keep their
true proportions (the rectifying homography is exact only for the z=0 plane —
anything above it gets stretched, which corrupted Brickognize IDs).
"""
from __future__ import annotations

import cv2
import numpy as np

Box = tuple[int, int, int, int]


def masked_instance_crop(
    original: np.ndarray,
    mask_original: np.ndarray,
    pad_fraction: float = 0.20,
) -> tuple[Box, np.ndarray]:
    """Isolate one true original-frame instance mask with neutral context."""
    mask = np.asarray(mask_original, dtype=bool)
    if mask.shape != original.shape[:2]:
        raise ValueError("instance mask must match the original image")
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("instance mask is empty")
    piece_x0, piece_y0 = int(xs.min()), int(ys.min())
    piece_x1, piece_y1 = int(xs.max()) + 1, int(ys.max()) + 1
    pad_x = max(2, round((piece_x1 - piece_x0) * pad_fraction))
    pad_y = max(2, round((piece_y1 - piece_y0) * pad_fraction))
    height, width = mask.shape
    box = (
        max(0, piece_x0 - pad_x), max(0, piece_y0 - pad_y),
        min(width, piece_x1 + pad_x), min(height, piece_y1 + pad_y),
    )
    x0, y0, x1, y1 = box
    crop = original[y0:y1, x0:x1].copy()
    local_mask = mask[y0:y1, x0:x1].astype(np.uint8)
    # Keep a one-pixel safety rim so antialiased piece boundaries are not cut.
    keep = cv2.dilate(local_mask, np.ones((3, 3), np.uint8)).astype(bool)
    outside = ~keep
    if outside.any():
        background = np.median(crop[outside], axis=0).astype(crop.dtype)
        crop[outside] = background
    return box, crop


def _mapped_quad(box_canvas: Box, homography: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = box_canvas
    corners = np.array(
        [[x0, y0, 1.0], [x1, y0, 1.0], [x1, y1, 1.0], [x0, y1, 1.0]]
    )
    mapped = corners @ np.asarray(homography, dtype=float).T
    return mapped[:, :2] / mapped[:, 2:3]


def masked_crop(
    original: np.ndarray,
    box_canvas: Box,
    homography: np.ndarray,
    rgb_size: tuple[int, int],
    raise_uv_per_cm: tuple[float, float] = (0.0, 0.0),
    pad_frac: float = 0.2,
    raise_cm: float = 1.5,
) -> tuple[Box, np.ndarray]:
    """Crop the padded AABB but blank everything outside the piece's own
    projected quad (+pad, +raise extension) to the local background color.

    The AABB of a diagonal piece contains its neighbors; Brickognize then
    sometimes identifies a neighbor or a composite instead of the target.
    """
    box = original_box(box_canvas, homography, rgb_size, raise_uv_per_cm,
                       pad_frac=pad_frac, raise_cm=raise_cm)
    x0, y0, x1, y1 = box
    crop = original[y0:y1, x0:x1].copy()
    if crop.size == 0:
        return box, crop

    quad = _mapped_quad(box_canvas, homography) - [x0, y0]
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * (1.0 + 2.0 * pad_frac)
    shift = np.array(raise_uv_per_cm) * raise_cm
    hull = cv2.convexHull(
        np.vstack([expanded, expanded + shift]).astype(np.int32)
    )
    keep = np.zeros(crop.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(keep, hull, 255)
    outside = keep == 0
    if outside.any() and (~outside).any():
        crop[outside] = np.median(crop[outside], axis=0).astype(crop.dtype)
    return box, crop


def original_box_and_fit(
    box_canvas: Box,
    homography: np.ndarray,
    rgb_size: tuple[int, int],
    raise_uv_per_cm: tuple[float, float] = (0.0, 0.0),
    pad_frac: float = 0.2,
    raise_cm: float = 1.5,
) -> tuple[Box, bool]:
    """Like original_box, but also reports whether the padded box fits the
    original image without clamping — a clamped crop means the piece may run
    off the physical camera frame and that view is a poor ID source."""
    box = original_box(box_canvas, homography, rgb_size, raise_uv_per_cm,
                       pad_frac=pad_frac, raise_cm=raise_cm, _clamp=False)
    height, width = rgb_size
    fits = box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height
    clamped = (max(0, box[0]), max(0, box[1]),
               min(width, box[2]), min(height, box[3]))
    return clamped, fits


def original_box(
    box_canvas: Box,
    homography: np.ndarray,
    rgb_size: tuple[int, int],
    raise_uv_per_cm: tuple[float, float] = (0.0, 0.0),
    pad_frac: float = 0.2,
    raise_cm: float = 1.5,
    _clamp: bool = True,
) -> Box:
    """AABB of the canvas box mapped into the original frame, padded.

    `homography` maps canvas px -> original px (the view's rectification
    matrix, manifest v2). Symmetric `pad_frac` covers localization slack;
    the extra one-sided extension follows `raise_uv_per_cm * raise_cm`,
    the projected displacement of a piece top ~raise_cm above the plane —
    the plane-projected box under-covers piece tops in oblique views.
    """
    x0, y0, x1, y1 = box_canvas
    corners = np.array(
        [[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]]
    )
    mapped = corners @ np.asarray(homography, dtype=float).T
    uv = mapped[:, :2] / mapped[:, 2:3]
    mx0, my0 = uv.min(axis=0)
    mx1, my1 = uv.max(axis=0)

    pad_u = (mx1 - mx0) * pad_frac
    pad_v = (my1 - my0) * pad_frac
    mx0, mx1 = mx0 - pad_u, mx1 + pad_u
    my0, my1 = my0 - pad_v, my1 + pad_v

    du = raise_uv_per_cm[0] * raise_cm
    dv = raise_uv_per_cm[1] * raise_cm
    if du >= 0:
        mx1 += du
    else:
        mx0 += du
    if dv >= 0:
        my1 += dv
    else:
        my0 += dv

    height, width = rgb_size
    if not _clamp:
        return (int(np.floor(mx0)), int(np.floor(my0)),
                int(np.ceil(mx1)), int(np.ceil(my1)))
    return (
        int(max(0, np.floor(mx0))),
        int(max(0, np.floor(my0))),
        int(min(width, np.ceil(mx1))),
        int(min(height, np.ceil(my1))),
    )
