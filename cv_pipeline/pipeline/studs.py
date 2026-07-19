"""Reliability-gated stud evidence for Brickognize candidates.

Dark pieces use separately calibrated luminance variants because the original
bright-piece Hough threshold can turn a visible 2 x 12 grid into a count of
four.  Every count carries visibility and reliability; weak evidence is kept
for inspection but cannot change an identification.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import cv2
import numpy as np

from pipeline.identify import Identification

STUD_DIAMETER_M = 0.0048
MIN_RELIABLE_COUNT = 4
CONTRADICTION_FACTOR = 3.0
HIGH_CONFIDENCE = 0.90
MIN_CANDIDATE_SCORE = 0.50
DARK_LUMA_THRESHOLD = 50.0
TRANSITION_LUMA_MAX = 80.0

_NAME_DIMS = re.compile(r"(\d+)\s*x\s*(\d+)")


@dataclass(frozen=True)
class StudVariant:
    name: str
    count: int
    centers_xy: tuple[tuple[float, float], ...]
    gamma: float | None
    hough_param2: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StudEvidence:
    count: int
    reliability: str
    visibility: str
    is_dark: bool
    median_luma: float | None
    variants: tuple[StudVariant, ...]
    reason: str

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "reliability": self.reliability,
            "visibility": self.visibility,
            "is_dark": self.is_dark,
            "median_luma": self.median_luma,
            "variants": [variant.to_dict() for variant in self.variants],
            "reason": self.reason,
        }


def expected_studs(name: str) -> int | None:
    """Stud count implied by a part name like ``Brick 1 x 8``."""
    match = _NAME_DIMS.search(name or "")
    if not match:
        return None
    return int(match.group(1)) * int(match.group(2))


def stud_radius_px(canvas_box, box_original, px_per_m) -> float:
    """Expected stud radius in original-frame pixels for this crop."""
    cw = max(1, canvas_box[2] - canvas_box[0])
    ch = max(1, canvas_box[3] - canvas_box[1])
    ow = max(1, box_original[2] - box_original[0])
    oh = max(1, box_original[3] - box_original[1])
    scale = (ow / cw + oh / ch) / 2.0
    return (STUD_DIAMETER_M / 2.0) * px_per_m * scale


def _local_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if mask is None:
        return np.ones(shape, dtype=bool)
    if mask.shape[:2] != shape:
        mask = cv2.resize(
            mask.astype(np.uint8), (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0


def _hough_centers(
    enhanced: np.ndarray,
    radius_px: float,
    param2: float,
    mask: np.ndarray,
) -> tuple[tuple[float, float], ...]:
    best: tuple[tuple[float, float], ...] = ()
    height, width = enhanced.shape
    for factor in (0.7, 1.0):
        radius = radius_px * factor
        if radius < 3:
            continue
        circles = cv2.HoughCircles(
            enhanced,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=radius * 1.6,
            param1=110,
            param2=param2,
            minRadius=max(2, int(radius * 0.65)),
            maxRadius=max(3, int(radius * 1.35)),
        )
        if circles is None:
            continue
        centers = []
        for x, y, _ in circles[0]:
            ix = min(width - 1, max(0, int(round(float(x)))))
            iy = min(height - 1, max(0, int(round(float(y)))))
            if mask[iy, ix]:
                centers.append((float(x), float(y)))
        if len(centers) > len(best):
            best = tuple(centers)
    return best


def _enhance(gray: np.ndarray, gamma: float | None) -> np.ndarray:
    if gamma is not None:
        normalized = gray.astype(np.float32) / 255.0
        gray = np.rint(np.power(normalized, gamma) * 255.0).astype(np.uint8)
    enhanced = cv2.createCLAHE(
        clipLimit=3.0, tileGridSize=(8, 8)
    ).apply(gray)
    return cv2.medianBlur(enhanced, 3)


def _grid_plausible(
    centers: tuple[tuple[float, float], ...], radius_px: float
) -> bool:
    if len(centers) < MIN_RELIABLE_COUNT:
        return False
    points = np.asarray(centers, dtype=np.float32)
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    distances[distances == 0] = np.inf
    nearest = distances.min(axis=1)
    plausible = (nearest >= radius_px * 1.5) & (nearest <= radius_px * 5.0)
    return float(plausible.mean()) >= 0.60


def analyze_studs(
    crop_bgr: np.ndarray,
    radius_px: float,
    mask: np.ndarray | None = None,
) -> StudEvidence:
    """Detect a stud grid and report whether the evidence can affect an ID."""
    if crop_bgr is None or crop_bgr.size == 0 or radius_px < 3:
        return StudEvidence(
            0, "unavailable", "not_visible", False, None, (),
            "empty crop or stud radius below three pixels",
        )
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    piece_mask = _local_mask(mask, gray.shape)
    if not np.any(piece_mask):
        return StudEvidence(
            0, "unavailable", "not_visible", False, None, (),
            "instance mask has no pixels",
        )
    median_luma = float(np.median(gray[piece_mask]))
    is_dark = median_luma < DARK_LUMA_THRESHOLD
    if is_dark:
        specifications = (
            ("clahe", None, 16.0),
            ("gamma_clahe", 0.5, 16.0),
        )
    elif median_luma <= TRANSITION_LUMA_MAX:
        specifications = (
            ("clahe", None, 18.0),
            ("gamma_clahe", 0.5, 16.0),
            ("gamma_soft_clahe", 0.75, 18.0),
        )
    else:
        specifications = (
            ("clahe", None, 22.0),
            ("gamma_soft_clahe", 0.75, 20.0),
        )
    variants = []
    for name, gamma, param2 in specifications:
        centers = _hough_centers(
            _enhance(gray, gamma), radius_px, param2, piece_mask
        )
        variants.append(StudVariant(
            name=name,
            count=len(centers),
            centers_xy=centers,
            gamma=gamma,
            hough_param2=param2,
        ))
    variants_tuple = tuple(variants)
    positive_counts = [
        variant.count for variant in variants if variant.count > 0
    ]
    count = (
        int(np.rint(np.median(positive_counts))) if positive_counts else 0
    )
    plausible_variants = [
        variant for variant in variants
        if _grid_plausible(variant.centers_xy, radius_px)
    ]
    if count < MIN_RELIABLE_COUNT:
        return StudEvidence(
            count, "low", "not_visible" if count == 0 else "partial",
            is_dark, median_luma, variants_tuple,
            "fewer than four plausible circles",
        )

    spread = (
        max(positive_counts) - min(positive_counts)
        if len(positive_counts) > 1 else 0
    )
    if len(plausible_variants) >= 2:
        if spread <= max(3.0, 0.25 * count):
            reliability = "high"
            reason = "preprocessing variants agree on a plausible grid"
        elif spread <= max(6.0, 0.50 * count):
            reliability = "medium"
            reason = "plausible grid with moderate variant spread"
        else:
            reliability = "low"
            reason = "preprocessing variants disagree"
    elif plausible_variants:
        reliability = "medium"
        reason = "only one preprocessing variant found a plausible grid"
    else:
        reliability = "low"
        reason = "detected circles do not form a plausible stud grid"
    visibility = "stud_face" if reliability in {"high", "medium"} else "partial"
    return StudEvidence(
        count, reliability, visibility, is_dark, median_luma,
        variants_tuple, reason,
    )


def count_studs(crop_bgr: np.ndarray, radius_px: float) -> int:
    """Backward-compatible bare count for callers that do not have a mask."""
    return analyze_studs(crop_bgr, radius_px).count


def aggregate_stud_evidence(
    evidences: list[StudEvidence],
) -> StudEvidence:
    """Combine distinct-view observations without promoting one view to high."""
    if not evidences:
        return StudEvidence(
            0, "unavailable", "not_visible", False, None, (),
            "no stud observations were available",
        )
    usable = [
        evidence for evidence in evidences
        if evidence.reliability in {"high", "medium"}
        and evidence.visibility == "stud_face"
    ]
    source = usable or evidences
    counts = [evidence.count for evidence in source]
    count = int(np.rint(np.median(counts))) if counts else 0
    lumas = [
        evidence.median_luma for evidence in evidences
        if evidence.median_luma is not None
    ]
    median_luma = float(np.median(lumas)) if lumas else None
    variants = tuple(
        variant for evidence in evidences for variant in evidence.variants
    )
    if not usable:
        reliability = "low"
        visibility = (
            "partial" if any(evidence.count for evidence in evidences)
            else "not_visible"
        )
        reason = "no view supplied a reliable visible stud face"
    elif len(usable) == 1:
        reliability = "medium"
        visibility = "stud_face"
        reason = "only one view supplied usable stud evidence"
    else:
        spread = max(counts) - min(counts)
        if spread <= max(3.0, 0.25 * count):
            reliability = "high"
            visibility = "stud_face"
            reason = "distinct views agree on the stud count"
        elif spread <= max(6.0, 0.50 * count):
            reliability = "medium"
            visibility = "stud_face"
            reason = "distinct views have moderate stud-count spread"
        else:
            reliability = "low"
            visibility = "partial"
            reason = "distinct views disagree on the stud count"
    return StudEvidence(
        count=count,
        reliability=reliability,
        visibility=visibility,
        is_dark=any(evidence.is_dark for evidence in evidences),
        median_luma=median_luma,
        variants=variants,
        reason=reason,
    )


def _contradicts(count: int, expected: int | None) -> bool:
    if expected is None:
        return False
    return (
        count >= CONTRADICTION_FACTOR * expected
        or count <= expected / CONTRADICTION_FACTOR
    )


def _from_candidate(candidate: dict, base: Identification) -> Identification:
    return Identification(
        candidate["part_id"], candidate["name"], None,
        float(candidate["score"]), candidates=base.candidates,
    )


def _coerce_evidence(value: StudEvidence | int) -> StudEvidence:
    if isinstance(value, StudEvidence):
        return value
    count = int(value)
    return StudEvidence(
        count=count,
        reliability="high" if count >= MIN_RELIABLE_COUNT else "low",
        visibility="stud_face" if count >= MIN_RELIABLE_COUNT else "partial",
        is_dark=False,
        median_luma=None,
        variants=(),
        reason="legacy bare count",
    )


def advise(
    ident: Identification,
    stud_count: StudEvidence | int,
) -> tuple[Identification, str]:
    """Apply structured stud evidence as a reliability-gated advisory."""
    evidence = _coerce_evidence(stud_count)
    if evidence.reliability not in {"high", "medium"}:
        action = (
            "unreliable" if isinstance(stud_count, StudEvidence)
            else "skipped"
        )
        return ident, action
    if ident.score >= HIGH_CONFIDENCE:
        return ident, "high-confidence"

    candidates = [
        candidate for candidate in ident.candidates
        if float(candidate["score"]) >= MIN_CANDIDATE_SCORE
    ]
    if ident.part_id is None:
        for candidate in candidates:
            expected = expected_studs(candidate["name"])
            if expected is None:
                continue
            if not _contradicts(evidence.count, expected):
                return _from_candidate(candidate, ident), "rescued"
        return ident, "no-rescue"

    if not _contradicts(evidence.count, expected_studs(ident.name)):
        return ident, "compatible"
    if evidence.reliability != "high":
        return ident, "ambiguous"
    for candidate in candidates:
        expected = expected_studs(candidate["name"])
        if expected is not None and not _contradicts(evidence.count, expected):
            return _from_candidate(candidate, ident), "corrected"
    demoted = Identification(
        None, "unknown brick", None, ident.score, candidates=ident.candidates
    )
    return demoted, "demoted"
