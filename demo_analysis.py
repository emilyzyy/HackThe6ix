"""Responsive analysis interstitial for the LEGO demo handoff."""
from __future__ import annotations

import math

import cv2
import numpy as np


YELLOW = (0, 213, 255)


def _fit(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    size = (
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def _masked_frame(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    out = image.copy()
    if mask is None:
        return out
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        mask = cv2.resize(
            mask.astype(np.uint8), image.shape[1::-1],
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    overlay = out.copy()
    overlay[mask] = YELLOW
    return cv2.addWeighted(overlay, 0.28, out, 0.72, 0)


def _place_card(
    canvas: np.ndarray,
    image: np.ndarray,
    center: tuple[int, int],
    angle: float,
    card_wh: tuple[int, int],
) -> None:
    card_w, card_h = card_wh
    photo = _fit(image, card_w - 16, card_h - 16)
    card = np.full((card_h, card_w, 3), 242, dtype=np.uint8)
    x = (card_w - photo.shape[1]) // 2
    y = (card_h - photo.shape[0]) // 2
    card[y:y + photo.shape[0], x:x + photo.shape[1]] = photo
    matrix = cv2.getRotationMatrix2D((card_w / 2, card_h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        card, matrix, (card_w, card_h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    valid = cv2.warpAffine(
        np.full((card_h, card_w), 255, dtype=np.uint8),
        matrix, (card_w, card_h), flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    x0 = center[0] - card_w // 2
    y0 = center[1] - card_h // 2
    x1, y1 = x0 + card_w, y0 + card_h
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas.shape[1], x1), min(canvas.shape[0], y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return
    rx0, ry0 = cx0 - x0, cy0 - y0
    rx1, ry1 = rx0 + cx1 - cx0, ry0 + cy1 - cy0
    region = canvas[cy0:cy1, cx0:cx1]
    alpha = (valid[ry0:ry1, rx0:rx1].astype(np.float32) / 255.0)[..., None]
    canvas[cy0:cy1, cx0:cx1] = np.clip(
        region * (1.0 - alpha) + rotated[ry0:ry1, rx0:rx1] * alpha,
        0, 255,
    ).astype(np.uint8)


def render_analysis(
    frames: list[np.ndarray],
    stage: str,
    now_s: float,
    size_wh: tuple[int, int],
    masks: list[np.ndarray] | None = None,
) -> np.ndarray:
    """Render a time-based photo stack and yellow analysis indicator."""
    width, height = map(int, size_wh)
    if width <= 0 or height <= 0:
        raise ValueError("analysis frame dimensions must be positive")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(30, 8, height, dtype=np.uint8)[:, None]
    canvas[..., 0] = gradient
    canvas[..., 1] = np.clip(gradient + 10, 0, 255)
    canvas[..., 2] = np.clip(gradient + 18, 0, 255)

    available = [np.asarray(frame) for frame in frames if frame is not None]
    if not available:
        available = [np.full((180, 240, 3), 32, dtype=np.uint8)]
    masks = list(masks or [])
    card_wh = (max(220, int(width * 0.42)), max(150, int(height * 0.48)))
    center = (width // 2, int(height * 0.56))
    phases = ((-5.0, -20, 12), (4.0, 22, 4), (-1.0, 0, -8))
    for index, (angle, dx, dy) in enumerate(phases):
        frame_index = (index + 1) % len(available)
        frame = available[frame_index]
        mask = masks[frame_index] if frame_index < len(masks) else None
        bob = int(round(3.0 * math.sin(now_s * 1.8 + index)))
        _place_card(
            canvas,
            _masked_frame(frame, mask),
            (center[0] + dx, center[1] + dy + bob),
            angle + 0.8 * math.sin(now_s + index),
            card_wh,
        )

    indicator = (width // 2, max(76, int(height * 0.17)))
    pulse = 0.5 + 0.5 * math.sin(now_s * 4.2)
    glow = canvas.copy()
    cv2.circle(glow, indicator, int(22 + pulse * 5), YELLOW, -1, cv2.LINE_AA)
    canvas = cv2.addWeighted(glow, 0.55, canvas, 0.45, 0)
    for offset, alpha in ((0.0, 0.75), (0.45, 0.45)):
        phase = (now_s * 0.65 + offset) % 1.0
        radius = int(30 + phase * 28)
        color = tuple(int(channel * (1.0 - 0.45 * phase)) for channel in YELLOW)
        cv2.circle(canvas, indicator, radius, color, 2, cv2.LINE_AA)
    for index in range(3):
        angle = now_s * 2.2 + index * (2.0 * math.pi / 3.0)
        point = (
            int(round(indicator[0] + math.cos(angle) * 47)),
            int(round(indicator[1] + math.sin(angle) * 47)),
        )
        cv2.circle(canvas, point, 4, YELLOW, -1, cv2.LINE_AA)

    title = "Analyzing your LEGO..."
    title_scale = max(0.72, min(1.2, width / 1100.0))
    title_size = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 3
    )[0]
    cv2.putText(
        canvas, title, ((width - title_size[0]) // 2, indicator[1] + 92),
        cv2.FONT_HERSHEY_SIMPLEX, title_scale, (245, 245, 245), 3,
        cv2.LINE_AA,
    )
    stage_scale = max(0.5, min(0.82, width / 1500.0))
    stage_size = cv2.getTextSize(
        stage, cv2.FONT_HERSHEY_SIMPLEX, stage_scale, 2
    )[0]
    cv2.putText(
        canvas, stage, ((width - stage_size[0]) // 2, indicator[1] + 126),
        cv2.FONT_HERSHEY_SIMPLEX, stage_scale, YELLOW, 2, cv2.LINE_AA,
    )
    return canvas
