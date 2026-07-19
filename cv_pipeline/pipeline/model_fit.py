"""Diagnostic calibrated silhouette fitting for regular LEGO parts."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


REGULAR_NAME = re.compile(r"^(Brick|Plate) (\d+) x (\d+)$")
STUD_PITCH_M = 0.008
BRICK_HEIGHT_M = 0.0096
PLATE_HEIGHT_M = 0.0032


@dataclass(frozen=True)
class FrameCalibration:
    frame_id: int
    intrinsics: tuple[float, float, float, float]
    # Original RGB dimensions in (width, height) order.
    rgb_size: tuple[int, int]
    camera_to_world: np.ndarray
    world_to_table: np.ndarray
    view_tilt_deg: float | None = None


@dataclass(frozen=True)
class RegularModel:
    part_id: str
    name: str
    family: str
    studs: tuple[int, int]
    dimensions_xyz: tuple[float, float, float]

    @classmethod
    def from_identification(cls, part_id, name):
        if part_id is None or not isinstance(name, str):
            return None
        match = REGULAR_NAME.fullmatch(name)
        if match is None:
            return None
        family, first, second = match.groups()
        studs = (int(first), int(second))
        height = BRICK_HEIGHT_M if family == "Brick" else PLATE_HEIGHT_M
        return cls(
            part_id=str(part_id),
            name=name,
            family=family,
            studs=studs,
            dimensions_xyz=(
                studs[0] * STUD_PITCH_M,
                studs[1] * STUD_PITCH_M,
                height,
            ),
        )


@dataclass(frozen=True)
class RestingOrientation:
    label: str
    dimensions_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ViewFitLoss:
    iou: float
    boundary_error: float
    under_coverage: float
    spill: float
    total: float


@dataclass(frozen=True)
class FitObservation:
    frame_id: int
    mask_original: np.ndarray
    calibration: FrameCalibration
    confidence: float = 1.0
    boundary_zone: bool = False
    sharpness: float = 1.0


@dataclass(frozen=True)
class CandidateFit:
    model: RegularModel
    total: float
    orientation: RestingOrientation | None
    center_xy: tuple[float, float] | None
    yaw_deg: float | None
    view_losses: tuple[dict, ...]
    trimmed_frame_id: int | None
    reliability: str
    reason: str


@dataclass(frozen=True)
class ModelFitEvidence:
    candidates: tuple[CandidateFit, ...]
    best_part_id: str | None
    runner_up_margin: float | None
    reliability: str
    reason: str


def resting_orientations(model: RegularModel):
    """Enumerate physical vertical axes, deduplicating x/y yaw symmetry."""
    dimensions = model.dimensions_xyz
    orientations = []
    seen = set()
    axis_names = ("stud_width_up", "stud_length_up", "element_height_up")
    for vertical_axis in range(3):
        horizontal = sorted(
            dimensions[index] for index in range(3) if index != vertical_axis
        )
        oriented = (horizontal[0], horizontal[1], dimensions[vertical_axis])
        key = tuple(round(value, 9) for value in oriented)
        if key in seen:
            continue
        seen.add(key)
        orientations.append(RestingOrientation(
            label=axis_names[vertical_axis], dimensions_xyz=oriented
        ))
    return tuple(orientations)


def cuboid_vertices(center_xy, yaw_deg, dimensions_xyz):
    """Return eight table-frame vertices for a cuboid resting at z=0."""
    width, length, height = (float(value) for value in dimensions_xyz)
    local = np.array([
        [x, y, z]
        for z in (0.0, height)
        for y in (-length / 2.0, length / 2.0)
        for x in (-width / 2.0, width / 2.0)
    ])
    angle = np.deg2rad(float(yaw_deg))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    local[:, :2] = local[:, :2] @ rotation.T
    local[:, :2] += np.asarray(center_xy, dtype=np.float64)
    return local


def render_cuboid(mask_shape, calibration, vertices):
    """Rasterize the convex projected silhouette in original RGB space."""
    height, width = (int(value) for value in mask_shape[:2])
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels, valid = project_table_points(vertices, calibration)
    if len(pixels) < 4 or not np.all(valid):
        return mask.astype(bool)
    hull = cv2.convexHull(np.rint(pixels).astype(np.int32))
    if hull is not None and len(hull) >= 3:
        cv2.fillConvexPoly(mask, hull, 1)
    return mask.astype(bool)


def _mask_boundary(mask):
    binary = np.asarray(mask, dtype=np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8))
    return (binary > 0) & (eroded == 0)


def _symmetric_boundary_error(observed, predicted):
    observed_boundary = _mask_boundary(observed)
    predicted_boundary = _mask_boundary(predicted)
    if not np.any(observed_boundary) and not np.any(predicted_boundary):
        return 0.0
    if not np.any(observed_boundary) or not np.any(predicted_boundary):
        return 1.0
    distance_to_observed = cv2.distanceTransform(
        (~observed_boundary).astype(np.uint8), cv2.DIST_L2, 5
    )
    distance_to_predicted = cv2.distanceTransform(
        (~predicted_boundary).astype(np.uint8), cv2.DIST_L2, 5
    )
    symmetric_pixels = 0.5 * (
        float(distance_to_observed[predicted_boundary].mean())
        + float(distance_to_predicted[observed_boundary].mean())
    )
    diagonal = float(np.hypot(*observed.shape[:2]))
    return symmetric_pixels / max(diagonal, 1.0)


def score_silhouettes(observed, predicted):
    """Return transparent mask-fit terms and a diagnostic-only total."""
    observed = np.asarray(observed, dtype=bool)
    predicted = np.asarray(predicted, dtype=bool)
    if observed.shape != predicted.shape or observed.ndim != 2:
        raise ValueError("silhouette masks must have the same two-dimensional shape")
    full_diagonal = float(np.hypot(*observed.shape))
    union_mask = (observed | predicted).astype(np.uint8)
    if np.any(union_mask):
        x, y, width, height = cv2.boundingRect(union_mask)
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1 = min(observed.shape[1], x + width + 2)
        y1 = min(observed.shape[0], y + height + 2)
        observed = observed[y0:y1, x0:x1]
        predicted = predicted[y0:y1, x0:x1]
    intersection = int(np.count_nonzero(observed & predicted))
    union = int(np.count_nonzero(observed | predicted))
    observed_area = int(np.count_nonzero(observed))
    predicted_area = int(np.count_nonzero(predicted))
    iou = intersection / union if union else 1.0
    under = (
        int(np.count_nonzero(observed & ~predicted)) / observed_area
        if observed_area else 0.0
    )
    spill = (
        int(np.count_nonzero(predicted & ~observed)) / predicted_area
        if predicted_area else 0.0
    )
    boundary = _symmetric_boundary_error(observed, predicted)
    cropped_diagonal = float(np.hypot(*observed.shape))
    if cropped_diagonal:
        boundary *= cropped_diagonal / max(full_diagonal, 1.0)
    total = (
        0.35 * (1.0 - iou)
        + 0.20 * boundary
        + 0.25 * under
        + 0.20 * spill
    )
    return ViewFitLoss(
        iou=float(iou),
        boundary_error=float(boundary),
        under_coverage=float(under),
        spill=float(spill),
        total=float(total),
    )


def _observation_weight(observation, max_area):
    area = int(np.count_nonzero(observation.mask_original))
    area_weight = np.sqrt(area / max(1, max_area))
    boundary_weight = 0.5 if observation.boundary_zone else 1.0
    return max(0.0, float(observation.confidence)) * boundary_weight * area_weight


def _evaluate_hypothesis(
    orientation, center_xy, yaw_deg, observations, weights, allow_trim=True,
):
    vertices = cuboid_vertices(center_xy, yaw_deg, orientation.dimensions_xyz)
    losses = []
    for observation in observations:
        predicted = render_cuboid(
            observation.mask_original.shape,
            observation.calibration,
            vertices,
        )
        losses.append(score_silhouettes(observation.mask_original, predicted))
    included = list(range(len(observations)))
    trimmed = None
    if allow_trim and len(observations) >= 4:
        oblique = [
            index for index, observation in enumerate(observations)
            if (
                observation.calibration.view_tilt_deg is not None
                and observation.calibration.view_tilt_deg >= 30.0
            )
        ]
        only_oblique = oblique[0] if len(oblique) == 1 else None
        eligible = [
            index for index in included if index != only_oblique
        ]
        if eligible:
            worst = max(eligible, key=lambda index: losses[index].total)
            peers = [losses[index].total for index in eligible if index != worst]
            peer_median = float(np.median(peers)) if peers else 0.0
            if (
                losses[worst].total > 0.05
                and losses[worst].total > 1.5 * max(peer_median, 1e-9)
            ):
                included.remove(worst)
                trimmed = worst
    weight_sum = sum(weights[index] for index in included)
    if weight_sum <= 0:
        total = float("inf")
    else:
        total = sum(
            weights[index] * losses[index].total for index in included
        ) / weight_sum
    return float(total), losses, trimmed


def _fit_reliability(observations):
    if not observations:
        return "unavailable", "no original-mask observations are available"
    oblique = [
        observation for observation in observations
        if (
            observation.calibration.view_tilt_deg is not None
            and observation.calibration.view_tilt_deg >= 30.0
        )
    ]
    if not oblique:
        return "low", "no usable oblique view is available"
    if len(observations) < 2:
        return "low", "fewer than two calibrated views are available"
    return "diagnostic", "calibrated multi-view diagnostic only"


def _bounded_fit_observation(observation):
    """Crop fitting work around the observed piece and adjust intrinsics."""
    mask = np.asarray(observation.mask_original, dtype=bool)
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return observation
    object_width = int(xs.max() - xs.min() + 1)
    object_height = int(ys.max() - ys.min() + 1)
    margin = max(32, int(np.ceil(1.5 * max(object_width, object_height))))
    x0 = max(0, int(xs.min()) - margin)
    y0 = max(0, int(ys.min()) - margin)
    x1 = min(mask.shape[1], int(xs.max()) + margin + 1)
    y1 = min(mask.shape[0], int(ys.max()) + margin + 1)
    if (x0, y0, x1, y1) == (0, 0, mask.shape[1], mask.shape[0]):
        return observation
    fx, fy, cx, cy = observation.calibration.intrinsics
    calibration = FrameCalibration(
        frame_id=observation.calibration.frame_id,
        intrinsics=(fx, fy, cx - x0, cy - y0),
        rgb_size=(x1 - x0, y1 - y0),
        camera_to_world=observation.calibration.camera_to_world,
        world_to_table=observation.calibration.world_to_table,
        view_tilt_deg=observation.calibration.view_tilt_deg,
    )
    return FitObservation(
        frame_id=observation.frame_id,
        mask_original=mask[y0:y1, x0:x1],
        calibration=calibration,
        confidence=observation.confidence,
        boundary_zone=observation.boundary_zone,
        sharpness=observation.sharpness,
    )


def fit_regular_candidate(
    model, observations, seed_xy, yaw_values=None,
):
    """Fit one regular cuboid deterministically without making an ID decision."""
    observations = tuple(
        _bounded_fit_observation(observation)
        for observation in observations
        if observation.mask_original is not None
    )
    reliability, reason = _fit_reliability(observations)
    if not observations:
        return CandidateFit(
            model=model,
            total=float("inf"),
            orientation=None,
            center_xy=None,
            yaw_deg=None,
            view_losses=(),
            trimmed_frame_id=None,
            reliability=reliability,
            reason=reason,
        )
    max_area = max(
        int(np.count_nonzero(observation.mask_original))
        for observation in observations
    )
    weights = tuple(
        _observation_weight(observation, max_area)
        for observation in observations
    )
    coarse_offsets = np.arange(-0.012, 0.0121, 0.004)
    yaws = (
        tuple(float(value) for value in yaw_values)
        if yaw_values is not None
        else tuple(float(value) for value in range(0, 180, 15))
    )
    coarse = []
    for orientation in resting_orientations(model):
        for dx in coarse_offsets:
            for dy in coarse_offsets:
                center = (float(seed_xy[0] + dx), float(seed_xy[1] + dy))
                for yaw in yaws:
                    total, _, _ = _evaluate_hypothesis(
                        orientation, center, yaw, observations, weights
                    )
                    coarse.append((total, orientation, center, yaw))
    coarse.sort(key=lambda value: (
        value[0], value[1].label, value[2], value[3]
    ))
    seeds = coarse[:8]
    refined = {}
    refine_offsets = np.arange(-0.004, 0.0041, 0.001)
    yaw_offsets = np.arange(-7.5, 7.51, 2.5)
    for _, orientation, center, yaw in seeds:
        for dx in refine_offsets:
            for dy in refine_offsets:
                refined_center = (
                    float(center[0] + dx), float(center[1] + dy)
                )
                for yaw_offset in yaw_offsets:
                    refined_yaw = float((yaw + yaw_offset) % 180.0)
                    key = (
                        orientation.label,
                        round(refined_center[0], 6),
                        round(refined_center[1], 6),
                        round(refined_yaw, 6),
                    )
                    if key in refined:
                        continue
                    total, losses, trimmed = _evaluate_hypothesis(
                        orientation,
                        refined_center,
                        refined_yaw,
                        observations,
                        weights,
                    )
                    refined[key] = (
                        total, orientation, refined_center, refined_yaw,
                        losses, trimmed,
                    )
    best = min(refined.values(), key=lambda value: (
        value[0], value[1].label, value[2], value[3]
    ))
    total, orientation, center, yaw, losses, trimmed = best
    view_losses = tuple({
        "frame_id": observation.frame_id,
        "weight": float(weights[index]),
        "trimmed": index == trimmed,
        "iou": loss.iou,
        "boundary_error": loss.boundary_error,
        "under_coverage": loss.under_coverage,
        "spill": loss.spill,
        "total": loss.total,
    } for index, (observation, loss) in enumerate(zip(observations, losses)))
    return CandidateFit(
        model=model,
        total=total,
        orientation=orientation,
        center_xy=center,
        yaw_deg=yaw,
        view_losses=view_losses,
        trimmed_frame_id=(
            None if trimmed is None else observations[trimmed].frame_id
        ),
        reliability=reliability,
        reason=reason,
    )


def compare_regular_candidates(
    models, observations, seed_xy, yaw_values=None,
):
    """Rank candidate cuboids while retaining diagnostic-only reliability."""
    fits = tuple(sorted(
        (
            fit_regular_candidate(
                model, observations, seed_xy, yaw_values=yaw_values
            )
            for model in models
        ),
        key=lambda fit: (fit.total, fit.model.part_id),
    ))
    if not fits:
        return ModelFitEvidence(
            candidates=(),
            best_part_id=None,
            runner_up_margin=None,
            reliability="unavailable",
            reason="no regular candidate models are available",
        )
    margin = fits[1].total - fits[0].total if len(fits) > 1 else None
    return ModelFitEvidence(
        candidates=fits,
        best_part_id=fits[0].model.part_id,
        runner_up_margin=(None if margin is None else float(margin)),
        reliability=fits[0].reliability,
        reason=fits[0].reason,
    )


def _view_value(view, name):
    if isinstance(view, dict):
        return view.get(name)
    return getattr(view, name, None)


def _derive_view_tilt_deg(camera_to_world, world_to_table):
    camera_forward_world = (
        np.asarray(camera_to_world, dtype=np.float64)[:3, :3]
        @ np.array([0.0, 0.0, -1.0])
    )
    viewing_ray_table = (
        np.asarray(world_to_table, dtype=np.float64)[:3, :3]
        @ camera_forward_world
    )
    magnitude = float(np.linalg.norm(viewing_ray_table))
    if magnitude <= 0 or not np.isfinite(magnitude):
        return None
    viewing_ray_table /= magnitude
    cosine = float(np.clip(-viewing_ray_table[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def load_frame_calibrations(session_dir, views) -> dict[int, FrameCalibration]:
    """Load selected ARKit frame calibration in explicit matrix conventions."""
    session_dir = Path(session_dir)
    table_path = session_dir / "table_frame.json"
    frames_path = session_dir / "frames.jsonl"
    if not table_path.exists() or not frames_path.exists():
        return {}
    world_to_table = np.asarray(
        json.loads(table_path.read_text())["world_to_table"], dtype=np.float64
    )
    selected = {
        int(_view_value(view, "frame_id")): view for view in views
    }
    records = {}
    for line in frames_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        frame_id = int(record["frame_id"])
        if frame_id in selected:
            records[frame_id] = record
    calibrations = {}
    for frame_id, view in selected.items():
        record = records.get(frame_id)
        if record is None:
            continue
        intrinsics = record["intrinsics"]
        # Stored RGB size follows NumPy (height, width) order.
        height, width = (int(value) for value in record["rgb_size"])
        camera_to_world = np.asarray(record["pose_mat"], dtype=np.float64)
        stored_tilt = _view_value(view, "view_tilt_deg")
        calibrations[frame_id] = FrameCalibration(
            frame_id=frame_id,
            intrinsics=(
                float(intrinsics["fx"]),
                float(intrinsics["fy"]),
                float(intrinsics["cx"]),
                float(intrinsics["cy"]),
            ),
            rgb_size=(width, height),
            camera_to_world=camera_to_world,
            world_to_table=world_to_table,
            view_tilt_deg=(
                _derive_view_tilt_deg(camera_to_world, world_to_table)
                if stored_tilt is None
                else float(stored_tilt)
            ),
        )
    return calibrations


def project_table_points(points_table_xyz, calibration: FrameCalibration):
    """Project table-frame metres into original RGB pixels."""
    points = np.asarray(points_table_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("table points must have shape (N, 3)")
    homogeneous = np.column_stack([points, np.ones(len(points))])
    table_to_world = np.linalg.inv(calibration.world_to_table)
    world_to_camera = np.linalg.inv(calibration.camera_to_world)
    camera = (world_to_camera @ (table_to_world @ homogeneous.T)).T
    depth = -camera[:, 2]
    fx, fy, cx, cy = calibration.intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * camera[:, 0] / depth + cx
        v = fy * -camera[:, 1] / depth + cy
    pixels = np.column_stack([u, v])
    width, height = calibration.rgb_size
    valid = (
        (depth > 0)
        & np.isfinite(pixels).all(axis=1)
        & (u >= -0.5)
        & (u < width - 0.5)
        & (v >= -0.5)
        & (v < height - 0.5)
    )
    return pixels, valid
