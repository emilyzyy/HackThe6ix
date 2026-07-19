"""Fuse LEGO observations from clean rectified views on one table canvas."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pipeline.color import (
    INVENTORY_COLOR_NAMES,
    INVENTORY_PALETTE,
    PALETTE,
    LegoColor,
    choose_illuminant_correction,
    color_evidence,
    dominant_rgb,
    dominant_rgb_masked,
    match_color,
    normalize_illumination,
    rank_color_candidates,
)
from pipeline.detect import Box, crop_box, detect
from pipeline.depth_height import measure_instance_height
from pipeline.dimensions import parse_regular_dimensions, resolve_dimensions
from pipeline.evidence import (
    aggregate_appearance_support,
    arbitrate_identification,
)
from pipeline.footprint import aggregate_measurements, measure_observation
from pipeline.identify import MIN_SCORE, UNKNOWN, identify_all, identify_crop
from pipeline.inventory import build_inventory
from pipeline.model_fit import (
    FitObservation,
    RegularModel,
    compare_regular_candidates,
    cuboid_vertices,
    load_frame_calibrations,
    render_cuboid,
)
from pipeline.studs import (
    advise,
    aggregate_stud_evidence,
    analyze_studs,
    stud_radius_px,
)

SECOND_CHANCE_MAX_ALTERNATES = 3
SEG_SINGLETON_MIN_CONFIDENCE = 0.50
SEG_ID_REVIEW_SCORE = 0.85
CONFIRMATION_REASSIGN_MAX_SCORE = 0.15
# Brickognize accepts a modest amount of concurrency; the old 1 req/s serial
# pacing made a large pile take minutes. Bounded concurrency preserves results
# (same crops, same cache, same responses) while overlapping network latency;
# retry/backoff already handles any 429 from the free API.
DEFAULT_IDENTIFY_WORKERS = 10
DEFAULT_IDENTIFY_MIN_INTERVAL = 0.0


_COLOR_FAMILY = {
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
class Observation:
    frame_id: int
    view_index: int
    box: Box
    color: LegoColor
    complete: bool
    boundary_clearance: float
    quality: float
    image: np.ndarray | None = field(default=None, repr=False, compare=False)
    valid_mask: np.ndarray | None = field(default=None, repr=False, compare=False)
    detector: str = "cv"
    confidence: float | None = None
    source_instance_id: str | None = None
    box_original: Box | None = None
    mask_original: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    mask_canvas: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    polygons_canvas: tuple[np.ndarray, ...] = field(
        default=(), repr=False, compare=False
    )
    original_image: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )
    gate_provenance: dict | None = field(
        default=None, repr=False, compare=False
    )
    color_evidence: dict | None = field(
        default=None, repr=False, compare=False
    )
    identification_focus: float | None = None


@dataclass
class FusedInstance:
    observations: list[Observation]
    best: Observation


@dataclass
class RectifiedView:
    frame_id: int
    view_index: int
    image: np.ndarray
    valid_mask: np.ndarray
    quality: float
    # Bridge geometry (manifest v2, optional): canvas px -> original px.
    homography: np.ndarray | None = None
    rgb: str | None = None
    rgb_size: tuple[int, int] | None = None
    raise_uv_per_cm: tuple[float, float] = (0.0, 0.0)
    topdownness: float | None = None
    sharpness_score: float | None = None
    camera_position_table_xyz: tuple[float, float, float] | None = None
    camera_bearing_deg: float | None = None
    viewing_ray_table_xyz: tuple[float, float, float] | None = None
    view_tilt_deg: float | None = None
    selection_role: str | None = None


@dataclass
class MultiViewScanResult:
    inventory: list[dict]
    instances: list[FusedInstance]
    labels: list[str]
    mosaic: np.ndarray
    manifest: dict
    id_crops: list[dict] = field(default_factory=list)
    crops: list[np.ndarray] = field(default_factory=list)
    views: list["RectifiedView"] = field(default_factory=list)
    segmentation_detections: list = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)


def box_is_complete(box: Box, valid_mask: np.ndarray, margin: int = 8) -> bool:
    x0, y0, x1, y1 = box
    h, w = valid_mask.shape[:2]
    ax0, ay0 = x0 - margin, y0 - margin
    ax1, ay1 = x1 + margin, y1 + margin
    if ax0 < 0 or ay0 < 0 or ax1 > w or ay1 > h:
        return False
    return bool(np.all(valid_mask[ay0:ay1, ax0:ax1] > 0))


def _boundary_clearance(box: Box, valid_mask: np.ndarray) -> float:
    distance = cv2.distanceTransform(
        (valid_mask > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    x0, y0, x1, y1 = box
    region = distance[max(0, y0):y1, max(0, x0):x1]
    return float(region.min()) if region.size else 0.0


def _box_area(box: Box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _box_iou(first: Box, second: Box) -> float:
    ix0, iy0 = max(first[0], second[0]), max(first[1], second[1])
    ix1, iy1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    union = _box_area(first) + _box_area(second) - intersection
    return intersection / union if union else 0.0


def _compatibility(
    first: Observation,
    second: Observation,
    min_area_ratio: float | None = None,
) -> float | None:
    if first.view_index == second.view_index:
        return None
    first_family = _COLOR_FAMILY.get(first.color.name, first.color.name)
    second_family = _COLOR_FAMILY.get(second.color.name, second.color.name)
    areas = (_box_area(first.box), _box_area(second.box))
    required_area_ratio = (
        min_area_ratio
        if min_area_ratio is not None
        else (0.40 if first.complete and second.complete else 0.20)
    )
    if min(areas) / max(1, max(areas)) < required_area_ratio:
        return None
    c1 = np.array(((first.box[0] + first.box[2]) / 2,
                   (first.box[1] + first.box[3]) / 2))
    c2 = np.array(((second.box[0] + second.box[2]) / 2,
                   (second.box[1] + second.box[3]) / 2))
    diagonal1 = np.hypot(first.box[2] - first.box[0],
                         first.box[3] - first.box[1])
    diagonal2 = np.hypot(second.box[2] - second.box[0],
                         second.box[3] - second.box[1])
    normalized_distance = float(np.linalg.norm(c1 - c2)) / max(
        1.0, (diagonal1 + diagonal2) / 2
    )
    iou = _box_iou(first.box, second.box)
    if iou < 0.05 and normalized_distance > 0.45:
        return None
    same_family = first_family == second_family
    warm_shift = (
        first_family in {"red", "orange", "yellow"}
        and second_family in {"red", "orange", "yellow"}
        and (iou >= 0.08 or normalized_distance <= 0.35)
    )
    neutral_partial_shift = (
        first_family in {"light-neutral", "dark-neutral"}
        and second_family in {"light-neutral", "dark-neutral"}
        and not first.complete and not second.complete
        and (iou >= 0.08 or normalized_distance <= 0.35)
    )
    # Color is supporting evidence, not identity.  The same neutral plate can
    # shift blue/gray under changing white balance; near-identical table-space
    # geometry is stronger than that cross-family color disagreement.
    strong_geometric_match = iou >= 0.60 and normalized_distance <= 0.15
    if not (
        same_family or warm_shift or neutral_partial_shift
        or strong_geometric_match
    ):
        return None
    return normalized_distance - 0.5 * iou


def _best_observation(observations: list[Observation]) -> Observation:
    return max(
        observations,
        key=lambda observation: (
            observation.complete,
            observation.boundary_clearance,
            observation.quality,
            _box_area(observation.box),
            -observation.frame_id,
        ),
    )


def _legacy_consensus_color(instance: FusedInstance) -> LegoColor:
    votes: dict[str, int] = {}
    for observation in instance.observations:
        name = observation.color.name
        votes[name] = votes.get(name, 0) + 1
    best_name = instance.best.color.name
    winner = max(votes, key=lambda name: (votes[name], name == best_name))
    reference = next(color for color in PALETTE if color.name == winner)
    if reference.name in INVENTORY_COLOR_NAMES:
        return reference
    return match_color(reference.rgb, INVENTORY_PALETTE)


def _ambiguous_color(first: str, second: str) -> LegoColor:
    names = sorted({first, second})
    palette = {color.name: color for color in PALETTE}
    rgb = tuple(
        int(round((palette[names[0]].rgb[index] + palette[names[1]].rgb[index]) / 2))
        for index in range(3)
    )
    return LegoColor(-1, f"ambiguous({names[0]}/{names[1]})", rgb)


def _color_observation_weight(
    observation: Observation, max_quality: float
) -> tuple[float, list[str]]:
    evidence = observation.color_evidence or {}
    flags = set(evidence.get("flags", []))
    quality = max(0.0, float(observation.quality))
    weight = 0.5 + 0.5 * quality / max(1e-9, max_quality)
    reasons = []
    if "possible_shadow" in flags or "possible_specular" in flags:
        weight *= 0.55
        reasons.append("lighting_artifact")
    if "boundary_view" in flags:
        weight *= 0.65
        reasons.append("boundary")
    view_quality = evidence.get("view_quality", {})
    tilt = view_quality.get("tilt_deg")
    if tilt is not None and float(tilt) >= 45.0:
        weight *= 0.35
        reasons.append("strong_oblique")
    elif tilt is not None and float(tilt) >= 30.0:
        weight *= 0.60
        reasons.append("oblique")
    confidence = view_quality.get("segmentation_confidence")
    if confidence is not None:
        weight *= max(0.35, min(1.0, float(confidence)))
    return float(weight), reasons


def _weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values)
    threshold = sum(weight for _, weight in ordered) / 2
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return value
    return ordered[-1][0]


def consensus_color_details(instance: FusedInstance) -> tuple[LegoColor, dict]:
    """Aggregate numeric color evidence with view-quality weights."""
    usable = [
        observation for observation in instance.observations
        if (observation.color_evidence or {}).get("decision_rgb") is not None
        and (observation.color_evidence or {}).get("decision", {}).get("gate")
    ]
    if not usable:
        color = _legacy_consensus_color(instance)
        return color, {
            "strategy": "legacy_unweighted_name_vote",
            "predicted_color": color.name,
        }
    max_quality = max(1e-9, *(max(0.0, float(obs.quality)) for obs in usable))
    weighted = []
    gate_weights = {"neutral": 0.0, "chromatic": 0.0}
    for observation in usable:
        weight, reasons = _color_observation_weight(observation, max_quality)
        evidence = observation.color_evidence or {}
        gate = evidence["decision"]["gate"]
        gate_weights[gate] += weight
        weighted.append((observation, weight, reasons, gate))
    winning_gate = max(
        gate_weights,
        key=lambda gate: (gate_weights[gate], gate == (instance.best.color_evidence or {}).get("decision", {}).get("gate")),
    )
    selected = [row for row in weighted if row[3] == winning_gate]

    relative_luminances = []
    if winning_gate == "neutral":
        for observation, weight, _, _ in selected:
            evidence = observation.color_evidence or {}
            raw_lab = np.asarray(evidence.get("raw_rgb"), dtype=float)
            bg_lab = np.asarray(evidence.get("background", {}).get("rgb"), dtype=float)
            if raw_lab.size == 3 and bg_lab.size == 3:
                from pipeline.color import _rgb_lab
                ratio = float(_rgb_lab(tuple(map(int, raw_lab)))[0]) / max(
                    1e-9, float(_rgb_lab(tuple(map(int, bg_lab)))[0])
                )
                relative_luminances.append((ratio, weight))
        relative_luminance = (
            _weighted_median(relative_luminances) if relative_luminances else None
        )
        if relative_luminance is not None and relative_luminance >= 1.05:
            color = next(color for color in PALETTE if color.name == "white")
            return color, {
                "strategy": "quality_weighted_gated_scores",
                "predicted_color": color.name,
                "gate": winning_gate,
                "gate_weights": gate_weights,
                "relative_luminance": relative_luminance,
                "ambiguity_reason": None,
            }
        if relative_luminance is not None and relative_luminance < 0.25:
            color = next(color for color in PALETTE if color.name == "black")
            return color, {
                "strategy": "quality_weighted_gated_scores",
                "predicted_color": color.name,
                "gate": winning_gate,
                "gate_weights": gate_weights,
                "relative_luminance": relative_luminance,
                "ambiguity_reason": None,
            }
        if relative_luminance is not None and relative_luminance < 0.42:
            color = _ambiguous_color("black", "dark gray")
            return color, {
                "strategy": "quality_weighted_gated_scores",
                "predicted_color": color.name,
                "gate": winning_gate,
                "gate_weights": gate_weights,
                "relative_luminance": relative_luminance,
                "ambiguity_reason": "dark_neutral_exposure_overlap",
            }

    totals: dict[str, float] = {}
    weights_by_name: dict[str, float] = {}
    for observation, weight, _, _ in selected:
        rgb = tuple(map(int, (observation.color_evidence or {})["decision_rgb"]))
        matches, _ = rank_color_candidates(
            rgb, colors=INVENTORY_PALETTE, limit=len(INVENTORY_PALETTE)
        )
        for match in matches:
            name = match["name"]
            totals[name] = totals.get(name, 0.0) + weight * float(match["decision_score"])
            weights_by_name[name] = weights_by_name.get(name, 0.0) + weight
    ranking = sorted(
        (totals[name] / weights_by_name[name], name) for name in totals
    )
    best_score, best_name = ranking[0]
    second_score, second_name = ranking[1] if len(ranking) > 1 else (float("inf"), best_name)
    margin = float(second_score - best_score)
    ambiguity_reason = None
    shade_pairs = {
        frozenset({"green", "dark green"}),
        frozenset({"blue", "dark blue"}),
        frozenset({"red", "dark red"}),
        frozenset({"brown", "dark brown"}),
    }
    if margin < 4.0:
        ambiguity_reason = "small_aggregate_margin"
    elif frozenset({best_name, second_name}) in shade_pairs and best_name.startswith("dark "):
        ambiguity_reason = "shade_exposure_overlap"
    if ambiguity_reason:
        color = _ambiguous_color(best_name, second_name)
    else:
        color = next(
            color for color in INVENTORY_PALETTE if color.name == best_name
        )
    return color, {
        "strategy": "quality_weighted_gated_scores",
        "predicted_color": color.name,
        "gate": winning_gate,
        "gate_weights": gate_weights,
        "aggregate_top_matches": [
            {"name": name, "score": float(score)}
            for score, name in ranking[:3]
        ],
        "aggregate_margin": margin,
        "ambiguity_reason": ambiguity_reason,
    }


def consensus_color(instance: FusedInstance) -> LegoColor:
    return consensus_color_details(instance)[0]


def color_provenance(instance: FusedInstance) -> dict:
    """Serialize the exact per-view evidence behind the current color vote."""
    votes: dict[str, int] = {}
    rows = []
    max_quality = max(
        1e-9, *(max(0.0, float(obs.quality)) for obs in instance.observations)
    )
    for observation in instance.observations:
        votes[observation.color.name] = votes.get(observation.color.name, 0) + 1
        weight, downweight_reasons = _color_observation_weight(
            observation, max_quality
        )
        row = {
            "frame_id": observation.frame_id,
            "predicted_color": observation.color.name,
            "aggregation_weight": weight,
            "downweight_reasons": downweight_reasons,
            **(observation.color_evidence or {}),
        }
        rows.append(row)
    _, aggregation = consensus_color_details(instance)
    return {
        **aggregation,
        "output_palette": list(INVENTORY_COLOR_NAMES),
        "vote_counts": votes,
        "observations": rows,
    }


def fuse_observations(observations: list[Observation]) -> list[FusedInstance]:
    """Associate compatible views deterministically without same-view merges."""
    if not observations:
        return []
    parents = list(range(len(observations)))
    frames = [{observation.view_index} for observation in observations]

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edges = []
    for first_index, first in enumerate(observations):
        for second_index in range(first_index + 1, len(observations)):
            score = _compatibility(first, observations[second_index])
            if score is not None:
                edges.append((score, first.frame_id,
                              observations[second_index].frame_id,
                              first_index, second_index))
    for _, _, _, first_index, second_index in sorted(edges):
        first_root, second_root = find(first_index), find(second_index)
        if first_root == second_root or frames[first_root] & frames[second_root]:
            continue
        parents[second_root] = first_root
        frames[first_root] |= frames[second_root]

    groups: dict[int, list[Observation]] = {}
    for index, observation in enumerate(observations):
        groups.setdefault(find(index), []).append(observation)
    instances = [
        FusedInstance(group, _best_observation(group))
        for group in groups.values()
    ]
    return sorted(
        instances,
        key=lambda instance: (
            instance.best.box[1], instance.best.box[0], instance.best.frame_id
        ),
    )


def reassign_confirmation_only_instances(
    instances: list[FusedInstance], views: list[RectifiedView]
) -> list[FusedInstance]:
    """Repair cross-matches caused solely by oblique confirmation views.

    Table-plane homographies can shift raised bricks in oblique views. Two
    unrelated confirmation detections may therefore overlap more strongly
    with each other than with their coverage-view anchors. Reassign such an
    orphan only when every observation has a defensible, unambiguous anchor;
    otherwise preserve the original evidence unchanged.
    """
    roles = {view.view_index: view.selection_role for view in views}
    if not any(role == "confirmation" for role in roles.values()):
        return instances

    working = [
        FusedInstance(list(instance.observations), instance.best)
        for instance in instances
    ]
    confirmation_only = [
        instance for instance in working
        if len(instance.observations) >= 2
        and all(
            roles.get(observation.view_index) == "confirmation"
            for observation in instance.observations
        )
    ]

    for orphan in confirmation_only:
        anchors = [
            instance for instance in working
            if instance is not orphan
            and any(
                roles.get(observation.view_index) == "coverage"
                for observation in instance.observations
            )
        ]
        assignments: list[tuple[Observation, FusedInstance]] = []
        for observation in orphan.observations:
            candidates: list[tuple[float, FusedInstance]] = []
            observation_family = _COLOR_FAMILY.get(
                observation.color.name, observation.color.name
            )
            for anchor in anchors:
                if any(
                    existing.view_index == observation.view_index
                    for existing in anchor.observations
                ):
                    continue
                scores = [
                    score
                    for existing in anchor.observations
                    if _COLOR_FAMILY.get(
                        existing.color.name, existing.color.name
                    ) == observation_family
                    if (
                        score := _compatibility(
                            observation, existing, min_area_ratio=0.20
                        )
                    ) is not None
                ]
                if scores:
                    candidates.append((min(scores), anchor))
            candidates.sort(key=lambda candidate: candidate[0])
            if not candidates:
                assignments = []
                break
            if candidates[0][0] > CONFIRMATION_REASSIGN_MAX_SCORE:
                assignments = []
                break
            # A very small score gap means geometry does not identify a
            # unique physical anchor. Keep the orphan intact in that case.
            if (
                len(candidates) > 1
                and candidates[1][0] - candidates[0][0] < 0.03
            ):
                assignments = []
                break
            assignments.append((observation, candidates[0][1]))

        if (
            len(assignments) != len(orphan.observations)
            or len({id(anchor) for _, anchor in assignments})
            != len(assignments)
        ):
            continue
        for observation, anchor in assignments:
            anchor.observations.append(observation)
        working.remove(orphan)

    return sorted(
        working,
        key=lambda instance: (
            instance.best.box[1], instance.best.box[0], instance.best.frame_id
        ),
    )


def filter_unconfirmed_instances(
    instances: list[FusedInstance], views: list[RectifiedView]
) -> list[FusedInstance]:
    """Drop one-view detections contradicted by another usable observation.

    A singleton is retained when no other selected frame covers its table
    location; that is a genuinely single-view/partial piece. Only two or more
    clean views that cover the center without a corresponding object can
    reject it as a per-view shadow/stud artifact.
    """
    kept = []
    for instance in instances:
        if len(views) <= 1:
            kept.append(instance)
            continue
        if len(instance.observations) > 1:
            segmentation = all(
                observation.detector == "seg"
                for observation in instance.observations
            )
            boundary_only = all(
                not observation.complete
                for observation in instance.observations
            )
            if (
                segmentation
                and len(views) >= 3
                and boundary_only
                and len(instance.observations) <= len(views) // 2
            ):
                continue
            kept.append(instance)
            continue
        observation = instance.best
        # With three or more independent segmentation views, a mask seen only
        # once at a workspace/frame boundary is weak evidence, not a countable
        # piece.  The same applies to a very weak complete singleton.  Strong,
        # fully-contained singletons remain valid when other views genuinely
        # do not cover their location.  Keep the classical detector's legacy
        # behavior unchanged.
        if (
            len(views) >= 3
            and observation.detector == "seg"
            and (
                not observation.complete
                or (
                    observation.confidence is not None
                    and observation.confidence < SEG_SINGLETON_MIN_CONFIDENCE
                )
            )
        ):
            continue
        cx = int(round((observation.box[0] + observation.box[2]) / 2))
        cy = int(round((observation.box[1] + observation.box[3]) / 2))
        contradicting_views = sum(
            view.view_index != observation.view_index
            and 0 <= cy < view.valid_mask.shape[0]
            and 0 <= cx < view.valid_mask.shape[1]
            and view.valid_mask[cy, cx] > 0
            for view in views
        )
        if contradicting_views < 2:
            kept.append(instance)
    return kept


def _background_rgb(image: np.ndarray) -> tuple[int, int, int]:
    thumb = cv2.resize(image, (200, 200))
    b, g, r = np.median(thumb.reshape(-1, 3), axis=0).astype(int)
    return int(r), int(g), int(b)


def load_views(manifest_path) -> tuple[dict, list[RectifiedView]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") not in (1, 2, 3):
        raise ValueError(f"unsupported multiview manifest version: {manifest.get('version')}")
    session_dir = Path(manifest["session_dir"])
    views = []
    for view_index, view in enumerate(manifest["views"]):
        image = cv2.imread(str(session_dir / view["image"]))
        valid_mask = cv2.imread(
            str(session_dir / view["mask"]), cv2.IMREAD_GRAYSCALE
        )
        if image is None or valid_mask is None:
            raise ValueError(f"missing multiview image or mask for frame {view['frame_id']}")
        homography = view.get("homography")
        camera_position = view.get("camera_position_table_xyz")
        viewing_ray = view.get("viewing_ray_table_xyz")
        views.append(RectifiedView(
            frame_id=int(view["frame_id"]),
            view_index=view_index,
            image=image,
            valid_mask=valid_mask,
            quality=float(view["quality"]),
            homography=None if homography is None else np.array(homography),
            rgb=view.get("rgb"),
            rgb_size=tuple(view["rgb_size"]) if view.get("rgb_size") else None,
            raise_uv_per_cm=tuple(view.get("raise_uv_per_cm", (0.0, 0.0))),
            topdownness=(
                None if view.get("topdownness") is None
                else float(view["topdownness"])
            ),
            sharpness_score=(
                None if view.get("sharpness") is None
                else float(view["sharpness"])
            ),
            camera_position_table_xyz=(
                None if camera_position is None
                else tuple(float(value) for value in camera_position)
            ),
            camera_bearing_deg=(
                None if view.get("camera_bearing_deg") is None
                else float(view["camera_bearing_deg"])
            ),
            viewing_ray_table_xyz=(
                None if viewing_ray is None
                else tuple(float(value) for value in viewing_ray)
            ),
            view_tilt_deg=(
                None if view.get("view_tilt_deg") is None
                else float(view["view_tilt_deg"])
            ),
            selection_role=view.get("selection_role"),
        ))
    return manifest, views


def observations_from_views(
    views: list[RectifiedView], detector: str = "cv"
) -> list[Observation]:
    observations: list[Observation] = []
    backgrounds = {
        view.view_index: _background_rgb(view.image) for view in views
    }
    correction_plan = choose_illuminant_correction(list(backgrounds.values()))
    for view in views:
        image, valid_mask = view.image, view.valid_mask
        background_rgb = backgrounds[view.view_index]
        detections = detect(image, method=detector)
        for box in detections.boxes:
            crop = crop_box(image, box)
            raw_rgb = dominant_rgb(crop)
            crop_mask = np.ones(crop.shape[:2], dtype=bool)
            evidence = color_evidence(
                crop,
                crop_mask,
                bg_rgb=background_rgb,
                sampled_rgb=raw_rgb,
                neutral_tolerance=45,
                view_tilt_deg=view.view_tilt_deg,
                boundary_zone=not box_is_complete(box, valid_mask),
                correction_plan=correction_plan,
                view_quality=view.quality,
            )
            # Preserve the historical association hint: final color is decided
            # from the new evidence only after geometry-based fusion.
            color = match_color(
                normalize_illumination(
                    raw_rgb, background_rgb, neutral_tolerance=45
                )
            )
            observations.append(Observation(
                frame_id=view.frame_id,
                view_index=view.view_index,
                box=box,
                color=color,
                complete=box_is_complete(box, valid_mask),
                boundary_clearance=_boundary_clearance(box, valid_mask),
                quality=view.quality,
                image=image,
                valid_mask=valid_mask,
                color_evidence=evidence,
            ))
    return observations


def observations_from_segmentations(
    detections, views: list[RectifiedView]
) -> list[Observation]:
    """Convert accepted original-frame masks into table-canvas observations."""
    by_index = {view.view_index: view for view in views}
    observations = []
    backgrounds = {}
    for detection in detections:
        if detection.accepted and detection.view_index not in backgrounds:
            backgrounds[detection.view_index] = _background_rgb(
                detection.image_original
            )
    correction_plan = choose_illuminant_correction(list(backgrounds.values()))
    for detection in detections:
        if not detection.accepted:
            continue
        view = by_index[detection.view_index]
        original = detection.image_original
        background_rgb = backgrounds[detection.view_index]
        raw_rgb = dominant_rgb_masked(original, detection.mask_original)
        gate = detection.gate
        evidence = color_evidence(
            original,
            detection.mask_original,
            bg_rgb=background_rgb,
            sampled_rgb=raw_rgb,
            neutral_tolerance=45,
            view_tilt_deg=view.view_tilt_deg,
            segmentation_confidence=detection.confidence,
            boundary_zone=gate.boundary_zone,
            correction_plan=correction_plan,
            view_quality=view.quality,
        )
        # Preserve the historical association hint: final color is decided
        # from the new evidence only after geometry-based fusion.
        color = match_color(
            normalize_illumination(
                raw_rgb, background_rgb, neutral_tolerance=45
            )
        )
        observations.append(Observation(
            frame_id=detection.frame_id,
            view_index=detection.view_index,
            box=detection.box_canvas,
            color=color,
            complete=not gate.boundary_zone,
            boundary_clearance=(
                0.0 if gate.boundary_zone
                else _boundary_clearance(detection.box_canvas, view.valid_mask)
            ),
            quality=view.quality,
            image=view.image,
            valid_mask=view.valid_mask,
            detector="seg",
            confidence=detection.confidence,
            source_instance_id=detection.source_instance_id,
            box_original=detection.box_original,
            mask_original=detection.mask_original,
            mask_canvas=detection.mask_canvas,
            polygons_canvas=detection.polygons_canvas,
            original_image=original,
            gate_provenance={
                "accepted": gate.accepted,
                "reason": gate.reason,
                "workspace_overlap": gate.workspace_overlap,
                "outside_fraction": gate.outside_fraction,
                "centroid_inside": gate.centroid_inside,
                "boundary_zone": gate.boundary_zone,
            },
            color_evidence=evidence,
            identification_focus=_masked_focus_score(
                original, detection.mask_original
            ),
        ))
    return observations


def load_observations(manifest_path, detector: str = "cv") -> list[Observation]:
    _, views = load_views(manifest_path)
    return observations_from_views(views, detector=detector)


def _normalized_view(view: RectifiedView, target_bgr: np.ndarray) -> np.ndarray:
    valid_pixels = view.image[view.valid_mask > 0]
    background = (np.median(valid_pixels, axis=0) if len(valid_pixels)
                  else np.ones(3) * 128)
    gain = np.clip(target_bgr / np.maximum(background, 1), 0.75, 1.35)
    return np.clip(view.image.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _expanded_box(box: Box, pad: int, width: int, height: int) -> Box:
    return (max(0, box[0] - pad), max(0, box[1] - pad),
            min(width, box[2] + pad), min(height, box[3] + pad))


def compose_object_aware_mosaic(
    views: list[RectifiedView], instances: list[FusedInstance]
) -> np.ndarray:
    """Keep one clean base view and supplement only what it cannot see."""
    if not views:
        raise ValueError("cannot compose a mosaic without rectified views")
    height, width = views[0].image.shape[:2]
    medians = [
        np.median(view.image[view.valid_mask > 0], axis=0)
        for view in views if np.any(view.valid_mask > 0)
    ]
    target_bgr = np.median(np.stack(medians), axis=0)
    normalized = {
        view.view_index: _normalized_view(view, target_bgr) for view in views
    }
    base = max(
        views,
        key=lambda view: (np.count_nonzero(view.valid_mask), view.quality),
    )
    accum = np.zeros((height, width, 3), dtype=np.float64)
    weight_sum = np.zeros((height, width), dtype=np.float64)
    for view in views:
        if view.view_index == base.view_index:
            continue
        distance = cv2.distanceTransform(
            (view.valid_mask > 0).astype(np.uint8), cv2.DIST_L2, 5
        )
        weight = np.clip(distance / 20.0, 0.0, 1.0)
        accum += normalized[view.view_index].astype(np.float64) * weight[..., None]
        weight_sum += weight
    supplement = np.empty((height, width, 3), dtype=np.uint8)
    supplemented = weight_sum > 0
    supplement[supplemented] = np.clip(
        accum[supplemented] / weight_sum[supplemented, None], 0, 255
    ).astype(np.uint8)
    supplement[~supplemented] = target_bgr.astype(np.uint8)
    base_distance = cv2.distanceTransform(
        (base.valid_mask > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    base_alpha = np.clip(base_distance / 12.0, 0.0, 1.0)[..., None]
    mosaic = np.clip(
        normalized[base.view_index].astype(np.float32) * base_alpha
        + supplement.astype(np.float32) * (1.0 - base_alpha),
        0,
        255,
    ).astype(np.uint8)

    # A complete base-view observation needs no patch at all. For an object
    # visible only partially (or not at all) in the base, copy the best source
    # using a content-shaped alpha mask so rectangular crop edges stay hidden.
    for instance in sorted(instances, key=lambda item: _box_area(item.best.box),
                           reverse=True):
        base_observation = next(
            (observation for observation in instance.observations
             if observation.view_index == base.view_index),
            None,
        )
        if base_observation is not None and base_observation.complete:
            continue
        observation = instance.best
        if observation.view_index == base.view_index:
            continue
        source = normalized[observation.view_index]
        x0, y0, x1, y1 = _expanded_box(observation.box, 6, width, height)
        patch_height, patch_width = y1 - y0, x1 - x0
        if patch_height <= 0 or patch_width <= 0:
            continue
        destination = mosaic[y0:y1, x0:x1].astype(np.float32)
        source_patch = source[y0:y1, x0:x1].astype(np.float32)
        core = np.zeros((patch_height, patch_width), dtype=np.uint8)
        bx0 = max(0, observation.box[0] - x0)
        by0 = max(0, observation.box[1] - y0)
        bx1 = min(patch_width, observation.box[2] - x0)
        by1 = min(patch_height, observation.box[3] - y0)
        core[by0:by1, bx0:bx1] = 1
        ring_pixels = source_patch[core == 0]
        local_background = (
            np.median(ring_pixels, axis=0)
            if len(ring_pixels) else target_bgr
        )
        difference = np.linalg.norm(source_patch - local_background, axis=2)
        foreground = ((difference > 22) & (core > 0)).astype(np.uint8) * 255
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        foreground = cv2.morphologyEx(
            foreground, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
        )
        alpha = cv2.GaussianBlur(foreground, (9, 9), 0).astype(np.float32)
        alpha = (alpha / 255.0)[..., None]
        mosaic[y0:y1, x0:x1] = np.clip(
            source_patch * alpha + destination * (1.0 - alpha), 0, 255
        ).astype(np.uint8)
    return mosaic


def _masked_focus_score(image: np.ndarray, mask: np.ndarray) -> float:
    """Piece-interior focus without rewarding a sharp segmentation edge."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != image.shape[:2]:
        raise ValueError("focus mask must match image dimensions")
    if not mask.any():
        return 0.0
    interior = cv2.erode(
        mask.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    if interior.sum() < max(9, 0.25 * mask.sum()):
        interior = mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    response = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    values = response[interior]
    return float(np.quantile(values, 0.80)) if values.size else 0.0


def _identification_quality_details(
    observation: Observation,
    view: RectifiedView,
    box: list[int] | tuple[int, int, int, int],
) -> dict:
    focus = observation.identification_focus
    focus_source = "piece_mask"
    if focus is None:
        focus = max(0.0, float(view.quality))
        focus_source = "whole_view_fallback"
    if observation.mask_original is not None:
        piece_pixels = int(np.asarray(observation.mask_original, dtype=bool).sum())
    else:
        piece_pixels = _box_area(tuple(map(int, box)))
    size_weight = min(1.0, np.sqrt(max(0, piece_pixels)) / 64.0)
    tilt = view.view_tilt_deg
    pose_weight = (
        1.0 if tilt is None
        else max(0.35, float(np.cos(np.deg2rad(float(tilt)))))
    )
    confidence_weight = (
        1.0 if observation.confidence is None
        else max(0.35, min(1.0, float(observation.confidence)))
    )
    score = (
        float(np.log1p(max(0.0, float(focus))))
        * float(size_weight)
        * pose_weight
        * confidence_weight
    )
    return {
        "score": score,
        "focus": float(focus),
        "focus_source": focus_source,
        "piece_pixels": piece_pixels,
        "size_weight": float(size_weight),
        "pose_weight": pose_weight,
        "segmentation_confidence_weight": confidence_weight,
    }


def _ranked_id_observations(
    instance: FusedInstance, views: list[RectifiedView]
) -> list[tuple[Observation, list[int]]]:
    """Observations usable for ORIGINAL-frame ID crops, best first.

    Ranking follows the Bridge spec: least obliquity and sharpness (the
    view's quality) first, canvas completeness after — a piece clipped by
    the workspace canvas is usually whole in the original frame. A view
    whose mapped crop runs off the original image is ranked below all
    fitting ones.
    """
    from pipeline.original_crop import original_box_and_fit

    ranked = []
    for observation in instance.observations:
        view = views[observation.view_index]
        if view.homography is None or view.rgb is None or view.rgb_size is None:
            continue
        if observation.mask_original is not None:
            ys, xs = np.nonzero(observation.mask_original)
            if not len(xs):
                continue
            x0, y0 = int(xs.min()), int(ys.min())
            x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
            pad_x = max(2, round((x1 - x0) * 0.20))
            pad_y = max(2, round((y1 - y0) * 0.20))
            height, width = observation.mask_original.shape
            fits = (
                x0 - pad_x >= 0 and y0 - pad_y >= 0
                and x1 + pad_x <= width and y1 + pad_y <= height
            )
            box = (
                max(0, x0 - pad_x), max(0, y0 - pad_y),
                min(width, x1 + pad_x), min(height, y1 + pad_y),
            )
        else:
            box, fits = original_box_and_fit(
                observation.box, view.homography, view.rgb_size,
                view.raise_uv_per_cm,
            )
        identification_quality = _identification_quality_details(
            observation, view, box
        )
        ranked.append((
            (fits, identification_quality["score"], view.quality,
             observation.complete,
             observation.boundary_clearance, -observation.frame_id),
            observation, list(box),
        ))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [(observation, box) for _, observation, box in ranked]


def _identification_observation(
    instance: FusedInstance, views: list[RectifiedView]
) -> tuple[Observation, list[int] | None]:
    ranked = _ranked_id_observations(instance, views)
    if not ranked:
        return instance.best, None
    return ranked[0]


def _identification_crops(
    instances: list[FusedInstance], views: list[RectifiedView],
    session_dir: Path,
) -> tuple[list[np.ndarray], list[dict]]:
    """Cut each instance's ID crop from its chosen view's ORIGINAL frame.

    Rectification is exact only for the table plane; raised pieces get
    stretched in rectified views, which corrupts Brickognize IDs. Falls back
    to the rectified crop for v1 manifests without bridge geometry.
    """
    originals: dict[int, np.ndarray] = {}
    crops, provenance = [], []
    for instance in instances:
        observation, box = _identification_observation(instance, views)
        view = views[observation.view_index]
        if box is None:
            crops.append(crop_box(observation.image, observation.box))
            provenance.append({
                "frame_id": observation.frame_id, "rgb": None,
                "box_original": None, "source": "rectified-fallback",
                "identification_focus": observation.identification_focus,
            })
            continue
        if observation.original_image is not None:
            original = observation.original_image
        elif view.frame_id not in originals:
            original = cv2.imread(str(session_dir / view.rgb))
            if original is None:
                raise ValueError(f"missing original frame {view.rgb}")
            originals[view.frame_id] = original
        else:
            original = originals[view.frame_id]
        if observation.mask_original is not None:
            from pipeline.original_crop import masked_instance_crop

            box, crop = masked_instance_crop(
                original, observation.mask_original
            )
            source = "yolo-instance-mask"
        else:
            from pipeline.original_crop import masked_crop

            box, crop = masked_crop(
                original, observation.box, view.homography,
                original.shape[:2], view.raise_uv_per_cm,
            )
            source = "original-masked"
        crops.append(crop)
        info = {
            "frame_id": observation.frame_id, "rgb": view.rgb,
            "box_original": list(box), "source": source,
            "identification_focus": observation.identification_focus,
            "identification_quality": _identification_quality_details(
                observation, view, box
            ),
        }
        if observation.mask_original is not None:
            info.update({
                "source_instance_id": observation.source_instance_id,
                "segmentation_confidence": observation.confidence,
                "gate": observation.gate_provenance,
            })
        provenance.append(info)
    return crops, provenance


def _isolated_context_crop(
    instance, observation, instances, original, box
):
    """Return real padded context only when no other instance enters it."""
    x0, y0, x1, y1 = box
    for other in instances:
        if other is instance:
            continue
        for candidate in other.observations:
            if (
                candidate.view_index == observation.view_index
                and candidate.mask_original is not None
                and np.any(candidate.mask_original[y0:y1, x0:x1])
            ):
                return None
    return original[y0:y1, x0:x1].copy()


def _regular_family(name):
    parsed = parse_regular_dimensions(name)
    return None if parsed is None else (parsed.kind, parsed.studs)


def _needs_family_review(identification, detector, metric_flag):
    """Regular segmented parts need semantic evidence from another view."""
    return detector == "seg" and (
        identification.score < SEG_ID_REVIEW_SCORE
        or _regular_family(identification.name) is not None
        or metric_flag == "possible_non_canonical_pose"
    )


def _identity_entries_by_frame(entries):
    per_frame = {}
    for entry in entries:
        current = per_frame.get(entry["frame_id"])
        if current is None or float(entry["score"]) > float(current["score"]):
            per_frame[entry["frame_id"]] = entry
    return list(per_frame.values())


def _select_identity_consensus(entries):
    """Select one top identity per frame, then prefer repeated agreement."""
    if not entries:
        raise ValueError("identity consensus requires at least one entry")
    per_frame = _identity_entries_by_frame(entries)
    accepted = [
        entry for entry in per_frame
        if entry.get("part_id") is not None
        and float(entry.get("score", 0.0)) >= MIN_SCORE
    ]
    support = {}
    for entry in accepted:
        part_id = entry["part_id"]
        support[part_id] = support.get(part_id, 0) + 1
    if len(support) == 1:
        return max(accepted, key=lambda entry: float(entry["score"])), False, support
    if support:
        maximum = max(support.values())
        winners = [part_id for part_id, count in support.items()
                   if count == maximum]
        if maximum >= 2 and len(winners) == 1:
            winner = winners[0]
            selected = max(
                (entry for entry in accepted
                 if entry["part_id"] == winner),
                key=lambda entry: float(entry["score"]),
            )
            return selected, False, support
    primary = entries[0]
    if primary.get("part_id") is None or float(primary.get("score", 0.0)) < MIN_SCORE:
        selected = max(
            accepted, key=lambda entry: float(entry["score"]),
            default=primary,
        )
    else:
        selected = primary
    return selected, len(support) > 1, support


def _identity_entry(frame_id, source, identification):
    return {
        "frame_id": int(frame_id),
        "source": source,
        "part_id": identification.part_id,
        "name": identification.name,
        "score": float(identification.score),
        "candidates": [dict(candidate)
                       for candidate in identification.candidates],
    }


def _second_chance(
    instances, views, session_dir, identifications, crops, id_crops, cache_dir,
    metric_flags=None,
):
    """Weak chosen-crop IDs get one more try from alternate observations.

    Evidence (234326 orange beam): the top-ranked view's crop identified as
    unknown 0.59 while the next view's crop identified as Brick 1x8 0.84.
    """
    from pipeline.original_crop import masked_crop, masked_instance_crop

    originals: dict[int, np.ndarray] = {}
    for index, instance in enumerate(instances):
        primary = identifications[index]
        metric_flag = (
            metric_flags[index] if metric_flags is not None else None
        )
        segmentation_review = _needs_family_review(
            primary, instance.best.detector, metric_flag
        )
        weak_review = primary.score < MIN_SCORE
        info = id_crops[index]
        flags = info.setdefault("flags", [])
        entries = [_identity_entry(
            info.get("frame_id", instance.best.frame_id),
            info.get("source", "primary"),
            primary,
        )]
        internal_entries = [{
            "entry": entries[0],
            "identification": primary,
            "crop": crops[index],
            "provenance": None,
            "identification_quality": float(
                info.get("identification_quality", {}).get("score", 0.0)
            ),
        }]
        if not weak_review and not segmentation_review:
            info["brickognize_views"] = entries
            info["identity_support"] = (
                {primary.part_id: 1} if primary.part_id is not None else {}
            )
            info["appearance_support"] = aggregate_appearance_support(entries)
            continue
        if segmentation_review and _regular_family(primary.name) is not None:
            flags.append("family_reviewed")
        ranked = _ranked_id_observations(instance, views)
        # Evaluate ALL alternates and keep the highest-scoring acceptance:
        # a canvas-clipped alternate can produce a confident-but-wrong ID
        # (a cut-off beam reads as a wedge), so first-accept is unsafe.
        candidates = (
            ranked[:SECOND_CHANCE_MAX_ALTERNATES]
            if segmentation_review
            else ranked[1:1 + SECOND_CHANCE_MAX_ALTERNATES]
        )
        for observation, _ in candidates:
            view = views[observation.view_index]
            if observation.original_image is not None:
                original = observation.original_image
            elif view.frame_id not in originals:
                image = cv2.imread(str(session_dir / view.rgb))
                if image is None:
                    continue
                originals[view.frame_id] = image
                original = image
            else:
                original = originals[view.frame_id]
            if observation.mask_original is not None:
                box, crop = masked_instance_crop(
                    original, observation.mask_original
                )
                variants = [(crop, "yolo-instance-mask-second-chance")]
                context = _isolated_context_crop(
                    instance, observation, instances, original, box
                )
                if context is not None:
                    variants.append((
                        context, "yolo-instance-context-second-chance"
                    ))
            else:
                box, crop = masked_crop(
                    original, observation.box, view.homography,
                    original.shape[:2], view.raise_uv_per_cm,
                )
                variants = [(crop, "original-masked-second-chance")]
            for variant, source in variants:
                candidate = identify_crop(variant, cache_dir=cache_dir)
                entry = _identity_entry(
                    observation.frame_id, source, candidate
                )
                entries.append(entry)
                update = {
                    "frame_id": observation.frame_id, "rgb": view.rgb,
                    "box_original": list(box),
                    "source": source,
                    "identification_focus": observation.identification_focus,
                    "identification_quality": _identification_quality_details(
                        observation, view, box
                    ),
                }
                if observation.mask_original is not None:
                    update.update({
                        "source_instance_id": observation.source_instance_id,
                        "segmentation_confidence": observation.confidence,
                        "gate": observation.gate_provenance,
                    })
                internal_entries.append({
                    "entry": entry,
                    "identification": candidate,
                    "crop": variant,
                    "provenance": update,
                    "identification_quality": float(
                        update["identification_quality"]["score"]
                    ),
                })
        selected, disagreement, support = _select_identity_consensus(entries)
        per_frame = _identity_entries_by_frame(entries)
        info["brickognize_views"] = per_frame
        info["identity_support"] = support
        info["appearance_support"] = aggregate_appearance_support(per_frame)
        if disagreement:
            flags.append("identity_view_disagreement")
        if selected.get("part_id") is None:
            chosen = max(
                (
                    record for record in internal_entries
                    if record["entry"].get("part_id") is None
                ),
                key=lambda record: (
                    float(record["entry"].get("score", 0.0)),
                    float(record.get("identification_quality", 0.0)),
                ),
            )
            matches = [chosen]
            if chosen is not internal_entries[0]:
                flags.append("best_weak_view_selected")
        else:
            matches = [
                record for record in internal_entries
                if record["entry"]["frame_id"] == selected["frame_id"]
                and record["entry"]["part_id"] == selected["part_id"]
                and record["entry"]["name"] == selected["name"]
                and record["entry"]["score"] == selected["score"]
            ]
        if matches:
            chosen = matches[-1]
            identifications[index] = chosen["identification"]
            crops[index] = chosen["crop"]
            if chosen["provenance"] is not None:
                info.update(chosen["provenance"])
    return identifications, crops, id_crops


def _apply_stud_advisory(
    instances, views, identifications, crops, id_crops, px_per_m,
    session_dir,
):
    from pipeline.original_crop import masked_crop, masked_instance_crop

    originals = {}
    for index, instance in enumerate(instances):
        info = id_crops[index]
        ident = identifications[index]
        if info.get("box_original") is None:
            info["stud"] = {
                "count": None,
                "reliability": "unavailable",
                "visibility": "not_visible",
                "action": "no-geometry",
                "reason": "identification crop has no original-frame geometry",
            }
            continue
        evidences = []
        view_evidence = []
        seen_frames = set()
        for observation, _ in _ranked_id_observations(instance, views):
            if observation.frame_id in seen_frames:
                continue
            seen_frames.add(observation.frame_id)
            view = views[observation.view_index]
            if observation.original_image is not None:
                original = observation.original_image
            elif view.frame_id in originals:
                original = originals[view.frame_id]
            else:
                original = cv2.imread(str(session_dir / view.rgb))
                if original is None:
                    continue
                originals[view.frame_id] = original
            if observation.mask_original is not None:
                box, stud_crop = masked_instance_crop(
                    original, observation.mask_original
                )
                x0, y0, x1, y1 = box
                local_mask = observation.mask_original[y0:y1, x0:x1]
            else:
                box, stud_crop = masked_crop(
                    original, observation.box, view.homography,
                    original.shape[:2], view.raise_uv_per_cm,
                )
                local_mask = None
            radius = stud_radius_px(observation.box, box, px_per_m)
            evidence = analyze_studs(
                stud_crop, radius, mask=local_mask
            )
            evidences.append(evidence)
            serialized = evidence.to_dict()
            serialized.update({
                "frame_id": observation.frame_id,
                "box_original": list(box),
                "radius_px": round(radius, 1),
            })
            view_evidence.append(serialized)
            if len(evidences) >= SECOND_CHANCE_MAX_ALTERNATES:
                break
        if not evidences:
            matches = [
                observation for observation in instance.observations
                if observation.frame_id == info.get("frame_id")
            ]
            observation = (
                matches[0] if matches
                else _identification_observation(instance, views)[0]
            )
            radius = stud_radius_px(
                observation.box, info["box_original"], px_per_m
            )
            local_mask = None
            if observation.mask_original is not None:
                x0, y0, x1, y1 = info["box_original"]
                local_mask = observation.mask_original[y0:y1, x0:x1]
            evidence = analyze_studs(
                crops[index], radius, mask=local_mask
            )
            evidences.append(evidence)
            serialized = evidence.to_dict()
            serialized.update({
                "frame_id": observation.frame_id,
                "box_original": list(info["box_original"]),
                "radius_px": round(radius, 1),
            })
            view_evidence.append(serialized)
        consensus = aggregate_stud_evidence(evidences)
        proposal, proposal_action = advise(ident, consensus)
        dimension_action = info.get("metric_silhouette", {}).get(
            "action", "unavailable"
        )
        outcome = arbitrate_identification(
            current=ident,
            dimension_action=dimension_action,
            dimension_evidence=info.get("metric_silhouette", {}),
            stud_proposal=proposal,
            stud_action=proposal_action,
            identity_support=info.get("identity_support", {}),
            appearance_evidence=info.get("appearance_support", {}),
        )
        info["stud"] = consensus.to_dict()
        info["stud"].update({
            "views": view_evidence,
            "proposal_action": proposal_action,
            "proposal": {
                "part_id": proposal.part_id,
                "name": proposal.name,
                "score": proposal.score,
            },
            "action": outcome.action,
            "arbitration": outcome.to_dict(),
        })
        if outcome.identification is not ident:
            info["stud"]["was"] = f"{ident.name} ({ident.score:.2f})"
            identifications[index] = outcome.identification
    return identifications


def _attach_metric_silhouettes(instances, views, id_crops, px_per_m):
    """Serialize polygon measurements without changing identification."""
    by_index = {view.view_index: view for view in views}
    aggregates = []
    for instance, info in zip(instances, id_crops):
        measurements = [
            measure_observation(
                observation,
                px_per_m,
                view=by_index.get(observation.view_index),
            )
            for observation in instance.observations
        ]
        aggregate = aggregate_measurements(measurements)
        aggregates.append(aggregate)
        info["metric_silhouette"] = {
            "action": "shadow_only",
            "evidence": aggregate.to_dict(),
        }
    return aggregates


def _attach_depth_heights(instances, id_crops, manifest):
    """Serialize confidence-filtered height without using it for identity."""
    for instance, info in zip(instances, id_crops):
        info["depth_height"] = measure_instance_height(
            instance, manifest
        ).to_dict()


def _regular_appearance_models(info):
    support = info.get("appearance_support", {}).get("candidate_support", {})
    models = []
    seen = set()
    for candidate in support.values():
        if not (
            candidate.get("accepted_frame_ids")
            or candidate.get("weak_frame_ids")
        ):
            continue
        model = RegularModel.from_identification(
            candidate.get("part_id"), candidate.get("name")
        )
        if model is None or model.part_id in seen:
            continue
        seen.add(model.part_id)
        models.append(model)
    return tuple(models)


def _serialize_candidate_fit(fit):
    orientation = None
    if fit.orientation is not None:
        orientation = {
            "label": fit.orientation.label,
            "dimensions_xyz_m": list(fit.orientation.dimensions_xyz),
        }
    return {
        "part_id": fit.model.part_id,
        "name": fit.model.name,
        "total": fit.total,
        "orientation": orientation,
        "center_xy_m": (
            None if fit.center_xy is None else list(fit.center_xy)
        ),
        "yaw_deg": fit.yaw_deg,
        "view_losses": list(fit.view_losses),
        "trimmed_frame_id": fit.trimmed_frame_id,
        "reliability": fit.reliability,
        "reason": fit.reason,
    }


def _write_model_fit_overlays(
    analysis_dir, instance_index, observations, evidence,
):
    overlay_dir = Path(analysis_dir) / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    observations_by_frame = {
        observation.frame_id: observation for observation in observations
    }
    written = []
    for fit in evidence.candidates:
        if (
            fit.orientation is None
            or fit.center_xy is None
            or fit.yaw_deg is None
        ):
            continue
        vertices = cuboid_vertices(
            fit.center_xy, fit.yaw_deg, fit.orientation.dimensions_xyz
        )
        for view_loss in fit.view_losses:
            observation = observations_by_frame.get(view_loss["frame_id"])
            if observation is None:
                continue
            predicted = render_cuboid(
                observation.mask_original.shape,
                observation.calibration,
                vertices,
            )
            observed = np.asarray(observation.mask_original, dtype=bool)
            overlay = np.zeros((*observed.shape, 3), dtype=np.uint8)
            overlay[observed & ~predicted] = (0, 210, 0)
            overlay[predicted & ~observed] = (0, 0, 230)
            overlay[observed & predicted] = (0, 220, 220)
            path = overlay_dir / (
                f"instance_{instance_index:03d}_{fit.model.part_id}_"
                f"frame_{observation.frame_id:05d}.png"
            )
            cv2.imwrite(str(path), overlay)
            written.append(str(path.resolve()))
    return written


def _attach_model_fit_diagnostics(
    instances, views, id_crops, manifest, analysis_dir,
):
    """Attach opt-in cuboid diagnostics after identity arbitration."""
    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    calibrations = load_frame_calibrations(
        Path(manifest["session_dir"]), views
    )
    origin_x, origin_y = (
        float(value) for value in manifest["origin_xy"]
    )
    px_per_m = float(manifest["px_per_m"])
    summaries = []
    by_view_index = {view.view_index: view for view in views}
    for index, (instance, info) in enumerate(zip(instances, id_crops)):
        models = _regular_appearance_models(info)
        if len(models) < 2:
            serialized = {
                "reliability": "unavailable",
                "reason": "fewer than two regular appearance candidate models",
                "best_part_id": None,
                "runner_up_margin": None,
                "candidates": [],
                "overlay_paths": [],
                "review_recommended": False,
            }
            info["model_fit"] = serialized
            summaries.append(serialized)
            continue
        fit_observations = []
        for observation in instance.observations:
            calibration = calibrations.get(observation.frame_id)
            if calibration is None or observation.mask_original is None:
                continue
            view = by_view_index.get(observation.view_index)
            gate = observation.gate_provenance or {}
            fit_observations.append(FitObservation(
                frame_id=observation.frame_id,
                mask_original=observation.mask_original,
                calibration=calibration,
                confidence=(
                    1.0 if observation.confidence is None
                    else float(observation.confidence)
                ),
                boundary_zone=bool(gate.get("boundary_zone", False)),
                sharpness=(
                    1.0
                    if view is None or view.sharpness_score is None
                    else float(view.sharpness_score)
                ),
            ))
        if not fit_observations:
            serialized = {
                "reliability": "unavailable",
                "reason": "no calibrated original-mask observations",
                "best_part_id": None,
                "runner_up_margin": None,
                "candidates": [],
                "overlay_paths": [],
                "review_recommended": False,
            }
            info["model_fit"] = serialized
            summaries.append(serialized)
            continue
        x0, y0, x1, y1 = instance.best.box
        seed_xy = (
            origin_x + (x0 + x1) / (2.0 * px_per_m),
            origin_y + (y0 + y1) / (2.0 * px_per_m),
        )
        evidence = compare_regular_candidates(
            models, fit_observations, seed_xy
        )
        serialized = {
            "reliability": evidence.reliability,
            "reason": evidence.reason,
            "best_part_id": evidence.best_part_id,
            "runner_up_margin": evidence.runner_up_margin,
            "candidates": [
                _serialize_candidate_fit(fit) for fit in evidence.candidates
            ],
        }
        final_part_id = info.get("brickognize", {}).get("part_id")
        review_recommended = bool(
            evidence.reliability == "diagnostic"
            and evidence.best_part_id is not None
            and final_part_id != evidence.best_part_id
        )
        serialized["review_recommended"] = review_recommended
        if review_recommended:
            flags = info.setdefault("flags", [])
            if "diagnostic_model_disagreement" not in flags:
                flags.append("diagnostic_model_disagreement")
        serialized["overlay_paths"] = _write_model_fit_overlays(
            analysis_dir, index, fit_observations, evidence
        )
        info["model_fit"] = serialized
        summaries.append(serialized)
    (analysis_dir / "model_fit.json").write_text(
        json.dumps({"instances": summaries}, indent=2) + "\n"
    )


def _apply_dimension_consistency(
    identifications, metric_silhouettes, id_crops, manifest
):
    selection = manifest.get("selection") or {}
    angle_diverse = selection.get("angle_diverse")
    for index, (identification, metric, info) in enumerate(zip(
        identifications, metric_silhouettes, id_crops
    )):
        decision = resolve_dimensions(identification, metric)
        info["metric_silhouette"]["action"] = decision.action
        info["metric_silhouette"]["decision"] = decision.to_dict()
        flags = info.setdefault("flags", [])
        if decision.flag is not None:
            if decision.flag not in flags:
                flags.append(decision.flag)
        if angle_diverse is False:
            if "insufficient_view_diversity" not in flags:
                flags.append("insufficient_view_diversity")
        elif angle_diverse is None:
            if "view_diversity_unknown" not in flags:
                flags.append("view_diversity_unknown")
        identifications[index] = decision.identification
    return identifications


def scan_manifest(
    manifest_path, detector: str = "cv", cache_dir: Path = Path(".cache"),
    seg_model=None, gate_config=None, analysis_dir=None,
    review_draft: bool = False, identify_workers: int = 4,
    identify_min_interval: float = 1.0,
    view_limit: int | None = None,
    identify_limit: int | None = None,
    max_detections_per_view: int | None = None,
) -> MultiViewScanResult:
    started = time.perf_counter()
    manifest, views = load_views(manifest_path)
    if view_limit is not None:
        views = views[:max(1, int(view_limit))]
    loaded = time.perf_counter()
    segmentation_detections = []
    if detector == "seg":
        from pipeline.segdetect import (
            GateConfig,
            YoloSegModel,
            detect_original_instances,
        )

        gate_config = gate_config or GateConfig()
        seg_model = seg_model or YoloSegModel(
            confidence=gate_config.inference_confidence
        )
        segmentation_detections = detect_original_instances(
            manifest, views, seg_model, gate_config,
            max_detections_per_view=max_detections_per_view,
        )
        observations = observations_from_segmentations(
            segmentation_detections, views
        )
    else:
        observations = observations_from_views(views, detector=detector)
    detected = time.perf_counter()
    instances = reassign_confirmation_only_instances(
        fuse_observations(observations), views
    )
    instances = filter_unconfirmed_instances(instances, views)
    fused = time.perf_counter()
    session_dir = Path(manifest["session_dir"])
    crops, id_crops = _identification_crops(instances, views, session_dir)
    identify_options = {
        "cache_dir": cache_dir,
        # Bounded concurrency by default; review drafts may raise the ceiling.
        "workers": identify_workers if review_draft else DEFAULT_IDENTIFY_WORKERS,
        "min_interval": (
            identify_min_interval if review_draft
            else DEFAULT_IDENTIFY_MIN_INTERVAL
        ),
    }
    if identify_limit is None:
        identification_indices = list(range(len(crops)))
    else:
        from pipeline.fast_showcase import rank_showcase_indices

        identification_indices = rank_showcase_indices(
            instances, limit=max(0, int(identify_limit))
        )
    selected_identifications = identify_all(
        [crops[index] for index in identification_indices],
        **identify_options,
    )
    identifications = [UNKNOWN for _ in crops]
    for index, identification in zip(
        identification_indices, selected_identifications
    ):
        identifications[index] = identification
    for crop_info, identification in zip(id_crops, identifications):
        crop_info["brickognize_primary"] = {
            "part_id": identification.part_id,
            "name": identification.name,
            "score": identification.score,
            "candidates": [dict(candidate)
                           for candidate in identification.candidates],
        }
    if review_draft:
        for crop_info, identification in zip(id_crops, identifications):
            primary = crop_info["brickognize_primary"]
            crop_info["brickognize_raw"] = dict(primary)
            crop_info["brickognize_views"] = [{
                "frame_id": crop_info.get("frame_id"),
                "source": crop_info.get("source", "primary"),
                **dict(primary),
            }]
            crop_info["identity_support"] = (
                {identification.part_id: 1}
                if identification.part_id is not None else {}
            )
            crop_info["appearance_support"] = aggregate_appearance_support(
                crop_info["brickognize_views"]
            )
        identified = time.perf_counter()
        evidenced = identified
    else:
        metric_silhouettes = _attach_metric_silhouettes(
            instances, views, id_crops, manifest["px_per_m"]
        )
        _attach_depth_heights(instances, id_crops, manifest)
        metric_flags = [
            resolve_dimensions(identification, metric).flag
            for identification, metric in zip(
                identifications, metric_silhouettes
            )
        ]
        identifications, crops, id_crops = _second_chance(
            instances, views, session_dir, identifications, crops, id_crops,
            cache_dir, metric_flags=metric_flags,
        )
        for crop_info, identification in zip(id_crops, identifications):
            crop_info["brickognize_raw"] = {
                "part_id": identification.part_id,
                "name": identification.name,
                "score": identification.score,
                "candidates": [dict(candidate)
                               for candidate in identification.candidates],
            }
        identified = time.perf_counter()
        identifications = _apply_dimension_consistency(
            identifications, metric_silhouettes, id_crops, manifest
        )
        identifications = _apply_stud_advisory(
            instances, views, identifications, crops, id_crops,
            manifest["px_per_m"], session_dir,
        )
        evidenced = time.perf_counter()
    for crop_info, identification in zip(id_crops, identifications):
        crop_info["brickognize"] = {
            "part_id": identification.part_id,
            "name": identification.name,
            "score": identification.score,
        }
    if analysis_dir is not None:
        _attach_model_fit_diagnostics(
            instances, views, id_crops, manifest, analysis_dir
        )
    colors = [consensus_color(instance) for instance in instances]
    labels = [
        f"{identification.name} | {color.name} | {identification.score:.2f}"
        for identification, color in zip(identifications, colors)
    ]
    inventory = build_inventory(list(zip(identifications, colors)))
    mosaic = compose_object_aware_mosaic(views, instances)
    completed = time.perf_counter()
    return MultiViewScanResult(
        inventory=inventory,
        instances=instances,
        labels=labels,
        mosaic=mosaic,
        manifest=manifest,
        id_crops=id_crops,
        crops=crops,
        views=views,
        segmentation_detections=segmentation_detections,
        timings_s={
            "load": loaded - started,
            "detect": detected - loaded,
            "fuse": fused - detected,
            "identify": identified - fused,
            "evidence": evidenced - identified,
            "color_inventory": completed - evidenced,
            "total": completed - started,
        },
    )
