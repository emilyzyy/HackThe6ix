"""Metric silhouette evidence from unclipped table-projected polygons.

These rectangles are view-dependent projections, not literal contact
footprints.  The module therefore preserves every view and reports a
reliability level instead of asserting nominal LEGO dimensions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SilhouetteMeasurement:
    frame_id: int
    short_mm: float | None
    long_mm: float | None
    usable: bool
    reason: str
    confidence: float | None
    boundary_zone: bool
    topdownness: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MetricSilhouette:
    short_mm: float | None
    long_mm: float | None
    reliability: str
    measurements: tuple[SilhouetteMeasurement, ...]
    used_frame_ids: tuple[int, ...]
    rejected_frame_ids: tuple[int, ...]
    spread_mm: tuple[float, float] | None
    reason: str

    def to_dict(self) -> dict:
        return {
            "short_mm": self.short_mm,
            "long_mm": self.long_mm,
            "reliability": self.reliability,
            "measurements": [item.to_dict() for item in self.measurements],
            "used_frame_ids": list(self.used_frame_ids),
            "rejected_frame_ids": list(self.rejected_frame_ids),
            "spread_mm": (
                None if self.spread_mm is None else list(self.spread_mm)
            ),
            "reason": self.reason,
        }


def _unusable(observation, reason: str, view=None) -> SilhouetteMeasurement:
    gate = getattr(observation, "gate_provenance", None) or {}
    return SilhouetteMeasurement(
        frame_id=int(observation.frame_id),
        short_mm=None,
        long_mm=None,
        usable=False,
        reason=reason,
        confidence=getattr(observation, "confidence", None),
        boundary_zone=bool(
            gate.get("boundary_zone", not getattr(observation, "complete", True))
        ),
        topdownness=getattr(view, "topdownness", None),
    )


def measure_observation(
    observation,
    px_per_m: float,
    view=None,
) -> SilhouetteMeasurement:
    """Fit a metric rectangle to unclipped segmentation polygons only."""
    if px_per_m <= 0:
        return _unusable(observation, "invalid manifest scale", view)
    polygons = getattr(observation, "polygons_canvas", ()) or ()
    if not polygons:
        return _unusable(observation, "no segmentation polygons", view)
    arrays = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points):
            arrays.append(points)
    if not arrays:
        return _unusable(observation, "segmentation polygons have no finite points", view)
    points = np.concatenate(arrays, axis=0)
    if len(np.unique(points, axis=0)) < 3:
        return _unusable(observation, "segmentation polygons are degenerate", view)
    (_, _), (width_px, height_px), _ = cv2.minAreaRect(points)
    if width_px <= 0 or height_px <= 0:
        return _unusable(observation, "minimum-area rectangle is degenerate", view)
    short_px, long_px = sorted((float(width_px), float(height_px)))
    millimetres_per_pixel = 1000.0 / float(px_per_m)
    gate = getattr(observation, "gate_provenance", None) or {}
    return SilhouetteMeasurement(
        frame_id=int(observation.frame_id),
        short_mm=short_px * millimetres_per_pixel,
        long_mm=long_px * millimetres_per_pixel,
        usable=True,
        reason="measured from unclipped segmentation polygons",
        confidence=getattr(observation, "confidence", None),
        boundary_zone=bool(
            gate.get("boundary_zone", not getattr(observation, "complete", True))
        ),
        topdownness=getattr(view, "topdownness", None),
    )


def _values(items: list[SilhouetteMeasurement]) -> np.ndarray:
    return np.asarray(
        [(item.short_mm, item.long_mm) for item in items], dtype=np.float64
    )


def _within(values: np.ndarray, absolute: float, fraction: float) -> bool:
    medians = np.median(values, axis=0)
    ranges = np.ptp(values, axis=0)
    limits = np.maximum(absolute, fraction * medians)
    return bool(np.all(ranges <= limits))


def _reject_isolated_outlier(
    usable: list[SilhouetteMeasurement],
) -> tuple[list[SilhouetteMeasurement], list[SilhouetteMeasurement]]:
    if len(usable) < 3:
        return usable, []
    candidates = []
    for index, item in enumerate(usable):
        remainder = usable[:index] + usable[index + 1:]
        values = _values(remainder)
        if not _within(values, absolute=4.0, fraction=0.15):
            continue
        median = np.median(values, axis=0)
        outlier = np.asarray((item.short_mm, item.long_mm), dtype=np.float64)
        limits = np.maximum(4.0, 0.15 * median)
        normalized = np.max(np.abs(outlier - median) / limits)
        if normalized > 1.0:
            candidates.append((float(normalized), index))
    if not candidates:
        return usable, []
    _, rejected_index = max(candidates)
    return (
        usable[:rejected_index] + usable[rejected_index + 1:],
        [usable[rejected_index]],
    )


def aggregate_measurements(
    measurements: list[SilhouetteMeasurement],
) -> MetricSilhouette:
    ordered = tuple(sorted(measurements, key=lambda item: item.frame_id))
    usable = [item for item in ordered if item.usable]
    if not usable:
        return MetricSilhouette(
            None, None, "unavailable", ordered, (), (), None,
            "no usable segmentation-polygon measurements",
        )
    used, rejected = _reject_isolated_outlier(usable)
    values = _values(used)
    medians = np.median(values, axis=0)
    spread = np.ptp(values, axis=0) if len(used) > 1 else np.zeros(2)
    if len(used) == 1:
        reliability = "low"
        reason = "only one usable view"
    elif _within(values, absolute=4.0, fraction=0.15):
        reliability = "high"
        reason = "at least two independent views agree"
    elif _within(values, absolute=8.0, fraction=0.30):
        reliability = "medium"
        reason = "independent views have moderate spread"
    else:
        reliability = "low"
        reason = "independent views have high spread"
    if rejected:
        reason += "; one isolated distorted view rejected"
    return MetricSilhouette(
        short_mm=float(medians[0]),
        long_mm=float(medians[1]),
        reliability=reliability,
        measurements=ordered,
        used_frame_ids=tuple(item.frame_id for item in used),
        rejected_frame_ids=tuple(item.frame_id for item in rejected),
        spread_mm=(float(spread[0]), float(spread[1])),
        reason=reason,
    )
