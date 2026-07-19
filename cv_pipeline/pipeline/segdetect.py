"""YOLO instance masks on original RGB frames, gated by physical workspace."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np


Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class ROI:
    x0: int
    y0: int
    x1: int
    y1: int

    def as_tuple(self) -> Box:
        return self.x0, self.y0, self.x1, self.y1


@dataclass(frozen=True)
class ProjectedWorkspace:
    roi: ROI
    workspace_mask: np.ndarray = field(repr=False, compare=False)
    contours_original: tuple[np.ndarray, ...] = field(
        repr=False, compare=False
    )


@dataclass(frozen=True)
class GateConfig:
    inference_confidence: float = 0.15
    min_confidence: float = 0.20
    min_workspace_overlap: float = 0.50
    strong_workspace_overlap: float = 0.80
    boundary_confidence: float = 0.35
    boundary_margin_px: int = 5
    roi_pad_fraction: float = 0.04


@dataclass(frozen=True)
class GateEvidence:
    accepted: bool
    reason: str
    confidence: float
    workspace_overlap: float
    outside_fraction: float
    centroid_inside: bool
    boundary_zone: bool


@dataclass(frozen=True)
class RawSegmentation:
    mask: np.ndarray
    confidence: float
    source_instance_id: str


@dataclass(frozen=True)
class MappedInstance:
    box_original: Box
    box_canvas: Box
    mask_canvas: np.ndarray = field(repr=False, compare=False)
    polygons_canvas: tuple[np.ndarray, ...] = field(
        repr=False, compare=False
    )


@dataclass
class SegmentationDetection:
    frame_id: int
    view_index: int
    source_instance_id: str
    confidence: float
    box_original: Box
    box_canvas: Box
    mask_original: np.ndarray = field(repr=False, compare=False)
    mask_canvas: np.ndarray = field(repr=False, compare=False)
    polygons_canvas: tuple[np.ndarray, ...] = field(
        repr=False, compare=False
    )
    gate: GateEvidence
    inference_roi: Box
    workspace_contours_original: tuple[np.ndarray, ...] = field(
        repr=False, compare=False
    )
    image_original: np.ndarray = field(repr=False, compare=False)

    @property
    def accepted(self) -> bool:
        return self.gate.accepted


def _box_iou(first: Box, second: Box) -> float:
    ix0, iy0 = max(first[0], second[0]), max(first[1], second[1])
    ix1, iy1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _near_identical_instance_masks(
    first: SegmentationDetection, second: SegmentationDetection
) -> bool:
    # box-IoU >= 0.60 is a NECESSARY condition for the original duplicate test
    # (it is one term of the final AND below), and it is cheap (integers only).
    # Checking it first only short-circuits pairs the original logic would also
    # reject, so the mask decision is byte-identical to before — but the
    # expensive full-frame boolean mask AND/sum is skipped for the ~all
    # non-overlapping pairs, which is what made suppression O(n^2)-slow.
    if _box_iou(first.box_original, second.box_original) < 0.60:
        return False
    first_mask = np.asarray(first.mask_original, dtype=bool)
    second_mask = np.asarray(second.mask_original, dtype=bool)
    if first_mask.shape != second_mask.shape:
        return False
    first_area, second_area = int(first_mask.sum()), int(second_mask.sum())
    if not first_area or not second_area:
        return False
    intersection = int((first_mask & second_mask).sum())
    union = first_area + second_area - intersection
    mask_iou = intersection / union if union else 0.0
    containment = intersection / min(first_area, second_area)
    area_ratio = min(first_area, second_area) / max(first_area, second_area)
    return bool(
        mask_iou >= 0.65
        and containment >= 0.85
        and area_ratio >= 0.70
        and _box_iou(first.box_original, second.box_original) >= 0.60
    )


def suppress_same_view_duplicates(
    detections: list[SegmentationDetection],
) -> list[SegmentationDetection]:
    """Reject only near-identical accepted masks from one source view.

    Every detection remains in the returned audit list.  The lower-confidence
    duplicate receives a provenance gate instead of disappearing, while
    detections from different views and overlapping-but-disjoint masks are
    never compared as duplicates.
    """
    by_view: dict[int, list[int]] = {}
    for index, detection in enumerate(detections):
        if detection.accepted:
            by_view.setdefault(detection.view_index, []).append(index)
    for indices in by_view.values():
        kept: list[SegmentationDetection] = []
        ordered = sorted(
            indices,
            key=lambda index: (
                -detections[index].confidence,
                detections[index].source_instance_id,
            ),
        )
        for index in ordered:
            detection = detections[index]
            if any(
                _near_identical_instance_masks(detection, accepted)
                for accepted in kept
            ):
                detection.gate = replace(
                    detection.gate,
                    accepted=False,
                    reason="duplicate_suppressed",
                )
            else:
                kept.append(detection)
    return detections


def _perspective_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    return cv2.perspectiveTransform(
        points, np.asarray(homography, dtype=np.float64)
    )[0]


def _table_to_canvas(
    contour: list[list[float]], origin_xy, px_per_m: float
) -> np.ndarray:
    points = np.asarray(contour, dtype=np.float32)
    points[:, 0] = (points[:, 0] - float(origin_xy[0])) * px_per_m
    points[:, 1] = (points[:, 1] - float(origin_xy[1])) * px_per_m
    return points


def projected_workspace_roi(
    contours_table: list[list[list[float]]],
    origin_xy,
    px_per_m: float,
    homography: np.ndarray,
    image_wh: tuple[int, int],
    pad_fraction: float = 0.04,
) -> ProjectedWorkspace:
    """Project the hard table-space workspace to an original RGB frame."""
    width, height = image_wh
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not contours_table:
        raise ValueError("physical workspace has no hard contours")
    contours_original = tuple(
        _perspective_points(
            _table_to_canvas(contour, origin_xy, px_per_m), homography
        )
        for contour in contours_table
        if len(contour) >= 3
    )
    if not contours_original:
        raise ValueError("physical workspace has no valid polygon")
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(
        mask,
        [np.rint(contour).astype(np.int32) for contour in contours_original],
        1,
    )
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("projected physical workspace is outside this frame")
    pad = round(max(width, height) * max(0.0, pad_fraction))
    roi = ROI(
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(width, int(xs.max()) + 1 + pad),
        min(height, int(ys.max()) + 1 + pad),
    )
    return ProjectedWorkspace(roi, mask.astype(bool), contours_original)


def _box_from_mask(mask: np.ndarray) -> Box:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("instance mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def map_instance(
    mask_original: np.ndarray,
    canvas_to_original: np.ndarray,
    canvas_wh: tuple[int, int],
) -> MappedInstance:
    """Map one original-pixel mask into the common table canvas."""
    mask_original = np.asarray(mask_original, dtype=bool)
    box_original = _box_from_mask(mask_original)
    x0, y0, x1, y1 = box_original
    corners = np.asarray(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32
    )
    original_to_canvas = np.linalg.inv(
        np.asarray(canvas_to_original, dtype=np.float64)
    )
    mapped_corners = _perspective_points(corners, original_to_canvas)
    minima = np.floor(mapped_corners.min(axis=0)).astype(int)
    maxima = np.ceil(mapped_corners.max(axis=0)).astype(int)
    box_canvas = (
        int(minima[0]), int(minima[1]), int(maxima[0]), int(maxima[1])
    )
    contours, _ = cv2.findContours(
        mask_original.astype(np.uint8), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons_canvas = tuple(
        _perspective_points(contour.reshape(-1, 2), original_to_canvas)
        for contour in contours
        if len(contour) >= 3
    )
    mask_canvas = cv2.warpPerspective(
        mask_original.astype(np.uint8), original_to_canvas,
        canvas_wh, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    return MappedInstance(
        box_original, box_canvas, mask_canvas, polygons_canvas
    )


def gate_instance(
    mask: np.ndarray,
    workspace_mask: np.ndarray,
    confidence: float,
    config: GateConfig = GateConfig(),
) -> GateEvidence:
    """Apply conservative, inspectable validity rules to one model mask."""
    mask = np.asarray(mask, dtype=bool)
    workspace = np.asarray(workspace_mask, dtype=bool)
    if mask.shape != workspace.shape:
        raise ValueError("instance/workspace mask shape mismatch")
    area = int(mask.sum())
    if not area:
        return GateEvidence(
            False, "empty_mask", confidence, 0.0, 1.0, False, False
        )
    inside = int((mask & workspace).sum())
    overlap = inside / area
    outside = 1.0 - overlap
    ys, xs = np.nonzero(mask)
    cx = int(round(float(xs.mean())))
    cy = int(round(float(ys.mean())))
    centroid_inside = (
        0 <= cy < workspace.shape[0]
        and 0 <= cx < workspace.shape[1]
        and bool(workspace[cy, cx])
    )
    margin = max(1, int(config.boundary_margin_px))
    eroded = cv2.erode(
        workspace.astype(np.uint8), np.ones((margin, margin), np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)
    boundary_zone = bool(
        outside > 0.001
        or np.any(mask & workspace & ~eroded)
        or np.any(mask[0]) or np.any(mask[-1])
        or np.any(mask[:, 0]) or np.any(mask[:, -1])
    )
    accepted, reason = True, "accepted"
    if confidence < config.min_confidence:
        accepted, reason = False, "low_confidence"
    elif overlap < config.min_workspace_overlap:
        accepted, reason = False, "workspace_overlap"
    elif not centroid_inside and overlap < config.strong_workspace_overlap:
        accepted, reason = False, "centroid_outside"
    elif (
        boundary_zone
        and confidence < config.boundary_confidence
        and overlap < config.strong_workspace_overlap
    ):
        accepted, reason = False, "weak_boundary_evidence"
    elif boundary_zone:
        reason = "accepted_boundary"
    return GateEvidence(
        accepted, reason, float(confidence), float(overlap), float(outside),
        centroid_inside, boundary_zone,
    )


class YoloSegModel:
    """Small adapter isolating the Ultralytics runtime dependency."""

    def __init__(
        self,
        weights: str | Path = "models/lego_seg.pt",
        device: str = "mps",
        confidence: float = 0.15,
        imgsz: int = 640,
        retina_masks: bool = True,
        max_det: int = 300,
    ):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(f"missing segmentation weights: {weights}")
        self.model = YOLO(str(weights))
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.retina_masks = bool(retina_masks)
        self.max_det = int(max_det)

    def predict(self, image: np.ndarray) -> list[RawSegmentation]:
        result = self.model.predict(
            source=image, device=self.device, conf=self.confidence,
            imgsz=self.imgsz, retina_masks=self.retina_masks,
            max_det=self.max_det,
            verbose=False,
        )[0]
        if result.masks is None:
            return []
        masks = result.masks.data.detach().cpu().numpy() > 0.5
        if masks.shape[1:] != image.shape[:2]:
            masks = np.stack([
                cv2.resize(
                    mask.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                for mask in masks
            ])
        confidences = (
            result.boxes.conf.detach().cpu().numpy().tolist()
            if result.boxes is not None and result.boxes.conf is not None
            else [1.0] * len(masks)
        )
        return [
            RawSegmentation(mask, float(confidence), f"yolo-{index:03d}")
            for index, (mask, confidence) in enumerate(zip(masks, confidences))
        ]


def detect_original_instances(
    manifest: dict,
    views: list[Any],
    model: Any,
    gate_config: GateConfig = GateConfig(),
    max_detections_per_view: int | None = None,
) -> list[SegmentationDetection]:
    """Detect every selected original view and preserve gate provenance.

    ``max_detections_per_view`` bounds the expensive per-instance geometry
    (full-frame gate erosion + canvas warp) to the N highest-confidence raw
    masks per view. The demo showcase only needs a few confident pieces, so
    processing all ~90 masks is wasted work; this is opt-in and unused by the
    full inventory pipeline (default None keeps every detection).
    """
    if manifest.get("version") != 3:
        raise ValueError(
            "segmentation requires manifest v3 physical workspace geometry"
        )
    geometry = manifest.get("workspace_geometry", {})
    contours_table = geometry.get("hard_contours_table_xy")
    if (
        geometry.get("hard_source") != "depth_observed_plane_component"
        or not contours_table
    ):
        raise ValueError("manifest v3 has no detector-independent hard workspace")
    session_dir = Path(manifest["session_dir"])
    canvas_wh = tuple(int(value) for value in manifest["size_wh"])
    detections = []
    for view in views:
        if view.homography is None or view.rgb is None:
            raise ValueError(
                f"view {view.frame_id} lacks original-frame bridge geometry"
            )
        original = cv2.imread(str(session_dir / view.rgb))
        if original is None:
            raise ValueError(f"missing original frame: {session_dir / view.rgb}")
        projection = projected_workspace_roi(
            contours_table,
            manifest["origin_xy"],
            float(manifest["px_per_m"]),
            view.homography,
            (original.shape[1], original.shape[0]),
            gate_config.roi_pad_fraction,
        )
        x0, y0, x1, y1 = projection.roi.as_tuple()
        crop = original[y0:y1, x0:x1]
        raws = model.predict(crop)
        if max_detections_per_view is not None:
            raws = sorted(
                raws, key=lambda r: r.confidence, reverse=True
            )[:max(0, int(max_detections_per_view))]
        for raw in raws:
            crop_mask = np.asarray(raw.mask, dtype=bool)
            if crop_mask.shape != crop.shape[:2]:
                crop_mask = cv2.resize(
                    crop_mask.astype(np.uint8),
                    (crop.shape[1], crop.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if not crop_mask.any():
                continue
            mask_original = np.zeros(original.shape[:2], dtype=bool)
            mask_original[y0:y1, x0:x1] = crop_mask
            gate = gate_instance(
                mask_original, projection.workspace_mask,
                raw.confidence, gate_config,
            )
            mapped = map_instance(
                mask_original, view.homography, canvas_wh
            )
            detections.append(SegmentationDetection(
                frame_id=view.frame_id,
                view_index=view.view_index,
                source_instance_id=raw.source_instance_id,
                confidence=raw.confidence,
                box_original=mapped.box_original,
                box_canvas=mapped.box_canvas,
                mask_original=mask_original,
                mask_canvas=mapped.mask_canvas,
                polygons_canvas=mapped.polygons_canvas,
                gate=gate,
                inference_roi=projection.roi.as_tuple(),
                workspace_contours_original=projection.contours_original,
                image_original=original,
            ))
    return suppress_same_view_duplicates(detections)
