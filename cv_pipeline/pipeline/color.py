"""Match a crop's dominant color to a curated base-color palette (CIEDE2000).

Matching runs against PALETTE, a hand-picked set of ~24 well-separated base
colors, not the full Rebrickable catalog: the catalog's near-duplicate shades
(several grays, bright vs regular green, Pearl/Trans finish variants) only add
noise that lighting variation would push pieces across. Each palette entry
carries the nearest representative Rebrickable color id so downstream part
lookups still work.

Independently runnable: python -m pipeline.color crop.jpg
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.color import deltaE_ciede2000, rgb2lab


@dataclass(frozen=True)
class LegoColor:
    id: int  # representative Rebrickable color id
    name: str
    rgb: tuple[int, int, int]


def _hx(hexstr: str) -> tuple[int, int, int]:
    return tuple(int(hexstr[i : i + 2], 16) for i in (0, 2, 4))


# Hand-picked hexes, chosen for CIEDE2000 separation (see
# test_palette_is_well_separated). Beige is lighter/creamier; tan is
# warmer/darker.
PALETTE: list[LegoColor] = [
    LegoColor(4, "red", _hx("C91A09")),
    LegoColor(320, "dark red", _hx("720E0F")),
    LegoColor(25, "orange", _hx("FE8A18")),
    LegoColor(14, "yellow", _hx("F2CD37")),
    LegoColor(19, "beige", _hx("E4CD9E")),
    LegoColor(84, "tan", _hx("AA7D55")),
    LegoColor(6, "brown", _hx("583927")),
    LegoColor(308, "dark brown", _hx("352100")),
    LegoColor(27, "lime", _hx("BBE90B")),
    LegoColor(2, "green", _hx("237841")),
    LegoColor(288, "dark green", _hx("184632")),
    LegoColor(3, "teal", _hx("008F9B")),
    LegoColor(322, "cyan", _hx("36AEBF")),
    LegoColor(73, "light blue", _hx("5A93DB")),
    LegoColor(23, "blue", _hx("0055BF")),
    LegoColor(272, "dark blue", _hx("0A3463")),
    LegoColor(85, "purple", _hx("3F3691")),
    LegoColor(29, "pink", _hx("E4ADC8")),
    LegoColor(26, "magenta", _hx("923978")),
    LegoColor(15, "white", _hx("FFFFFF")),
    LegoColor(71, "light gray", _hx("A0A5A9")),
    LegoColor(72, "gray", _hx("6C6E68")),
    LegoColor(8, "dark gray", _hx("3E4044")),
    LegoColor(0, "black", _hx("05131D")),
]

# Emily's physical demo inventory intentionally uses broad color buckets.
# Keep PALETTE above for inspectable per-view evidence and association hints;
# final inventory decisions are restricted to this separate profile.
INVENTORY_COLOR_NAMES: tuple[str, ...] = (
    "red", "orange", "yellow", "beige", "brown", "green",
    "dark green", "blue", "white", "light gray", "dark gray", "black",
)
INVENTORY_PALETTE: list[LegoColor] = [
    color for name in INVENTORY_COLOR_NAMES
    for color in PALETTE
    if color.name == name
]

NEUTRAL_COLOR_NAMES = frozenset(
    {"white", "light gray", "gray", "dark gray", "black"}
)
CHROMA_GATE = 18.0
STRONG_CHROMA = 50.0


def load_colors(csv_path: Path = Path("data/colors.csv")) -> list[LegoColor]:
    """Load the full Rebrickable catalog (solid opaque colors only).

    Kept as a reference loader; pipeline matching uses PALETTE instead.
    """
    colors = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            cid = int(row["id"])
            if cid == -1:  # "[Unknown]" placeholder row
                continue
            if row.get("is_trans", "f").lower() in ("t", "true"):
                continue
            h = row["rgb"].lstrip("#")
            colors.append(LegoColor(cid, row["name"], _hx(h)))
    return colors


def dominant_rgb(crop_bgr: np.ndarray) -> tuple[int, int, int]:
    """Median color of the central ~5% of pixels, excluding glare/shadow."""
    h, w = crop_bgr.shape[:2]
    side = max(2, int((0.05**0.5) * min(h, w)))
    cy, cx = h // 2, w // 2
    center = crop_bgr[
        max(0, cy - side // 2) : cy + side // 2 + 1,
        max(0, cx - side // 2) : cx + side // 2 + 1,
    ]
    px = center.reshape(-1, 3).astype(np.int32)
    lum = px.mean(axis=1)
    keep = px[(lum > 25) & (lum < 235)]
    if len(keep) == 0:  # uniformly dark/bright crop: use everything
        keep = px
    b, g, r = np.median(keep, axis=0).astype(int)
    return int(r), int(g), int(b)


def dominant_rgb_masked(
    image_bgr: np.ndarray, mask: np.ndarray
) -> tuple[int, int, int]:
    """Robust piece color from only one segmentation instance's pixels."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError("color mask must match image dimensions")
    if not mask.any():
        raise ValueError("color mask is empty")
    interior = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    if interior.sum() < max(9, 0.25 * mask.sum()):
        interior = mask
    pixels = image_bgr[interior].reshape(-1, 3).astype(np.int32)
    luminance = pixels.mean(axis=1)
    usable = pixels[(luminance > 25) & (luminance < 245)]
    if len(usable) < 9:
        usable = pixels
    if len(usable) >= 20:
        usable_luminance = usable.mean(axis=1)
        low, high = np.quantile(usable_luminance, [0.10, 0.90])
        central = usable[
            (usable_luminance >= low) & (usable_luminance <= high)
        ]
        if len(central):
            usable = central
    b, g, r = np.median(usable, axis=0).astype(int)
    return int(r), int(g), int(b)


def normalize_illumination(
    rgb: tuple[int, int, int],
    bg_rgb: tuple[int, int, int],
    neutral_tolerance: int = 30,
) -> tuple[int, int, int]:
    """White-patch correction: if the background is near-neutral (a white or
    gray table), scale channels so the background maps to white, cancelling
    dim light and color casts. Colored backgrounds are no reference — the
    input is returned unchanged.
    """
    if max(bg_rgb) - min(bg_rgb) > neutral_tolerance or max(bg_rgb) < 100:
        return rgb
    scaled = tuple(
        min(255, int(round(c * min(2.0, 245 / max(1, b)))))
        for c, b in zip(rgb, bg_rgb)
    )
    return scaled  # type: ignore[return-value]


def _lab_chroma(rgb: tuple[int, int, int]) -> float:
    lab = _rgb_lab(rgb)
    return float(np.hypot(lab[1], lab[2]))


def _lab_hue(lab: np.ndarray) -> float:
    return float(math.degrees(math.atan2(float(lab[2]), float(lab[1]))) % 360)


def _hue_distance(first: float, second: float) -> float:
    difference = abs(first - second)
    return min(difference, 360.0 - difference)


def choose_illuminant_correction(
    background_rgbs: list[tuple[int, int, int]],
) -> dict:
    """Select one correction for a session from table reference samples.

    A gray-world/von-Kries correction is only defensible when the putative
    neutral surface is bright, only mildly chromatic, and consistent across
    views.  Otherwise the surface may be colored or unstable and raw samples
    are safer.  Unlike the legacy white-patch operation, gray-world maps the
    reference channels to their mean and therefore does not force exposure to
    white.
    """
    if not background_rgbs:
        return {
            "method": "raw_fallback",
            "reason": "no_neutral_reference",
            "reference_count": 0,
        }
    values = np.asarray(background_rgbs, dtype=float)
    labs = np.asarray([_rgb_lab(tuple(map(int, rgb))) for rgb in values])
    chromas = np.hypot(labs[:, 1], labs[:, 2])
    means = values.mean(axis=1)
    channel_ratios = values.max(axis=1) / np.maximum(1.0, values.min(axis=1))
    normalized = values / np.maximum(1.0, values.sum(axis=1, keepdims=True))
    consistency_spread = float(
        np.max(np.linalg.norm(normalized - np.median(normalized, axis=0), axis=1))
    )
    reliable = bool(
        np.median(means) >= 100.0
        and np.median(chromas) <= CHROMA_GATE
        and np.max(channel_ratios) <= 1.40
        and consistency_spread <= 0.06
    )
    raw_score = float(np.median(chromas))
    gray_world_score = 0.0
    if not reliable:
        method = "raw_fallback"
        reason = "neutral_reference_unreliable"
    elif raw_score <= 2.0:
        method = "raw_fallback"
        reason = "reference_already_neutral"
    else:
        method = "gray_world"
        reason = "gray_world_best_neutrality"
    return {
        "method": method,
        "reason": reason,
        "reference_count": int(len(values)),
        "raw_neutrality_score": raw_score,
        "gray_world_neutrality_score": gray_world_score,
        "reference_chroma_median": float(np.median(chromas)),
        "reference_consistency_spread": consistency_spread,
        "reference_reliable": reliable,
    }


def apply_illuminant_correction(
    rgb: tuple[int, int, int],
    bg_rgb: tuple[int, int, int],
    plan: dict,
) -> tuple[tuple[int, int, int], dict]:
    """Apply a session plan, rejecting per-object chroma/hue damage."""
    details = {
        "method": plan.get("method", "raw_fallback"),
        "applied": False,
        "rejection_reason": None,
        "channel_gains": [1.0, 1.0, 1.0],
    }
    if plan.get("method") != "gray_world":
        details["rejection_reason"] = plan.get("reason", "raw_fallback")
        return rgb, details

    background = np.asarray(bg_rgb, dtype=float)
    target = float(background.mean())
    gains = np.clip(target / np.maximum(1.0, background), 0.75, 1.35)
    candidate = tuple(
        int(value)
        for value in np.clip(np.rint(np.asarray(rgb, dtype=float) * gains), 0, 255)
    )
    raw_lab, corrected_lab = _rgb_lab(rgb), _rgb_lab(candidate)
    raw_chroma = float(np.hypot(raw_lab[1], raw_lab[2]))
    corrected_chroma = float(np.hypot(corrected_lab[1], corrected_lab[2]))
    hue_shift = _hue_distance(_lab_hue(raw_lab), _lab_hue(corrected_lab))
    rejection_reason = None
    if raw_chroma >= STRONG_CHROMA:
        rejection_reason = "strong_chroma_stable"
    elif raw_chroma <= CHROMA_GATE and corrected_chroma > raw_chroma + 4.0:
        rejection_reason = "low_chroma_amplification"
    elif raw_chroma > CHROMA_GATE and hue_shift > 12.0:
        rejection_reason = "excessive_hue_shift"
    details.update({
        "channel_gains": [float(value) for value in gains],
        "raw_chroma": raw_chroma,
        "corrected_chroma": corrected_chroma,
        "hue_shift_deg": hue_shift,
        "rejection_reason": rejection_reason,
        "applied": rejection_reason is None,
    })
    return (candidate if rejection_reason is None else rgb), details


_lab_cache: dict[int, np.ndarray] = {}


def _rgb_lab(rgb: tuple[int, int, int]) -> np.ndarray:
    return rgb2lab(
        np.array(rgb, dtype=float).reshape(1, 1, 3) / 255.0
    ).reshape(3)


def palette_matches(
    rgb: tuple[int, int, int],
    colors: list[LegoColor] | None = None,
    limit: int = 3,
) -> list[dict]:
    """Rank palette candidates while preserving inspectable delta-E values."""
    if colors is None:
        colors = PALETTE
    key = id(colors)
    if key not in _lab_cache:
        arr = np.array([color.rgb for color in colors], dtype=float) / 255.0
        _lab_cache[key] = rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
    distances = deltaE_ciede2000(_rgb_lab(rgb)[None, :], _lab_cache[key])
    order = np.argsort(distances)[:max(0, int(limit))]
    return [
        {
            "color_id": colors[int(index)].id,
            "name": colors[int(index)].name,
            "rgb": list(colors[int(index)].rgb),
            "delta_e": float(distances[int(index)]),
        }
        for index in order
    ]


def rank_color_candidates(
    rgb: tuple[int, int, int],
    colors: list[LegoColor] | None = None,
    limit: int = 3,
) -> tuple[list[dict], dict]:
    """Rank candidates with a chroma gate and hue-first chromatic score."""
    if colors is None:
        colors = PALETTE
    lab = _rgb_lab(rgb)
    chroma = float(np.hypot(lab[1], lab[2]))
    neutral = chroma <= CHROMA_GATE
    candidates = [
        color for color in colors
        if (color.name in NEUTRAL_COLOR_NAMES) == neutral
    ]
    if not candidates:
        candidates = list(colors)
    candidate_labs = np.asarray([_rgb_lab(color.rgb) for color in candidates])
    delta_es = deltaE_ciede2000(lab[None, :], candidate_labs)
    scores = []
    target_hue = _lab_hue(lab)
    for index, (color, candidate_lab) in enumerate(zip(candidates, candidate_labs)):
        if neutral:
            score = float(delta_es[index])
            hue_delta = None
        else:
            candidate_chroma = float(np.hypot(candidate_lab[1], candidate_lab[2]))
            hue_delta = _hue_distance(target_hue, _lab_hue(candidate_lab))
            # Hue leads; lightness supports but cannot turn a dim blue into
            # dark blue solely because the exposure is low.
            score = (
                hue_delta
                + 0.10 * abs(float(lab[0] - candidate_lab[0]))
                + 0.20 * abs(chroma - candidate_chroma)
            )
        scores.append((score, index, hue_delta))
    ranked = []
    for score, index, hue_delta in sorted(scores)[:max(0, int(limit))]:
        color = candidates[index]
        row = {
            "color_id": color.id,
            "name": color.name,
            "rgb": list(color.rgb),
            "delta_e": float(delta_es[index]),
            "decision_score": float(score),
        }
        if hue_delta is not None:
            row["hue_delta_deg"] = float(hue_delta)
        ranked.append(row)
    return ranked, {
        "gate": "neutral" if neutral else "chromatic",
        "chroma": chroma,
        "chroma_gate": CHROMA_GATE,
        "chromatic_score": "hue+0.10*lightness+0.20*chroma",
    }


def _masked_interior_pixels(
    image_bgr: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image_bgr.shape[:2]:
        raise ValueError("color mask must match image dimensions")
    if not mask.any():
        raise ValueError("color mask is empty")
    interior = cv2.erode(
        mask.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    if interior.sum() < max(9, 0.25 * mask.sum()):
        interior = mask
    return image_bgr[interior].reshape(-1, 3)


def color_evidence(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    bg_rgb: tuple[int, int, int],
    sampled_rgb: tuple[int, int, int] | None = None,
    neutral_tolerance: int = 30,
    view_tilt_deg: float | None = None,
    segmentation_confidence: float | None = None,
    boundary_zone: bool = False,
    correction_plan: dict | None = None,
    view_quality: float | None = None,
) -> dict:
    """Describe the sample, correction, and gated palette decision."""
    raw_rgb = (
        dominant_rgb_masked(image_bgr, mask)
        if sampled_rgb is None else sampled_rgb
    )
    plan = correction_plan or choose_illuminant_correction([bg_rgb])
    corrected_rgb, correction = apply_illuminant_correction(
        raw_rgb, bg_rgb, plan
    )
    matches, decision = rank_color_candidates(corrected_rgb, limit=3)
    pixels = _masked_interior_pixels(image_bgr, mask)
    gray = cv2.cvtColor(
        pixels.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2GRAY
    ).reshape(-1).astype(float)
    p10, median, p90 = np.quantile(gray, [0.10, 0.50, 0.90])
    low_fraction = float(np.mean(gray <= 25))
    high_fraction = float(np.mean(gray >= 245))
    spread = float(p90 - p10)
    background_neutral = bool(plan.get("reference_reliable", False))
    flags = []
    if raw_rgb != corrected_rgb:
        flags.append("gray_world_applied")
    if correction.get("rejection_reason"):
        flags.append(f"correction_rejected:{correction['rejection_reason']}")
    if not background_neutral:
        flags.append("background_not_neutral")
    if low_fraction >= 0.10 and spread >= 40.0:
        flags.append("possible_shadow")
    if high_fraction >= 0.02 and spread >= 40.0:
        flags.append("possible_specular")
    if view_tilt_deg is not None and view_tilt_deg >= 30.0:
        flags.append("oblique_view")
    if segmentation_confidence is not None and segmentation_confidence < 0.70:
        flags.append("low_segmentation_confidence")
    if boundary_zone:
        flags.append("boundary_view")
    return {
        "raw_rgb": list(raw_rgb),
        "corrected_rgb": list(corrected_rgb),
        "decision_rgb": list(corrected_rgb),
        "lab": [float(value) for value in _rgb_lab(corrected_rgb)],
        "top_palette_matches": matches,
        "delta_e_margin": (
            float(matches[1]["delta_e"] - matches[0]["delta_e"])
            if len(matches) >= 2 else None
        ),
        "decision_margin": (
            float(matches[1]["decision_score"] - matches[0]["decision_score"])
            if len(matches) >= 2 else None
        ),
        "decision": decision,
        "correction": {"session_plan": plan, **correction},
        "luminance": {
            "pixel_count": int(gray.size),
            "p10": float(p10),
            "median": float(median),
            "p90": float(p90),
            "p90_minus_p10": spread,
            "shadow_clip_fraction": low_fraction,
            "highlight_clip_fraction": high_fraction,
        },
        "background": {
            "rgb": list(bg_rgb),
            "neutral_tolerance": int(neutral_tolerance),
            "is_neutral": background_neutral,
        },
        "view_quality": {
            "tilt_deg": view_tilt_deg,
            "segmentation_confidence": segmentation_confidence,
            "boundary_zone": bool(boundary_zone),
            "view_quality": view_quality,
        },
        "flags": flags,
    }


def match_color(
    rgb: tuple[int, int, int], colors: list[LegoColor] | None = None
) -> LegoColor:
    if colors is None:
        colors = PALETTE
    key = id(colors)
    if key not in _lab_cache:
        arr = np.array([c.rgb for c in colors], dtype=float) / 255.0
        _lab_cache[key] = rgb2lab(arr.reshape(-1, 1, 3)).reshape(-1, 3)
    target = _rgb_lab(rgb)
    dists = deltaE_ciede2000(target[None, :], _lab_cache[key])
    return colors[int(np.argmin(dists))]


if __name__ == "__main__":
    import sys

    import cv2

    img = cv2.imread(sys.argv[1])
    if img is None:
        raise SystemExit(f"could not read image: {sys.argv[1]}")
    rgb = dominant_rgb(img)
    color = match_color(rgb)
    print(f"dominant rgb={rgb} -> {color.name} (color_id={color.id})")
