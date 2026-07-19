"""Confidence-filtered table-relative height evidence in shadow mode."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthHeightEvidence:
    action: str
    reliability: str
    frame_ids: tuple[int, ...]
    usable_pixels: int
    confidence_counts: dict[str, int]
    median_mm: float | None
    p10_mm: float | None
    p90_mm: float | None
    spread_mm: float | None
    reason: str
    observations: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _unavailable(frame_ids, reason, observations=()):
    return DepthHeightEvidence(
        action="shadow_only",
        reliability="unavailable",
        frame_ids=tuple(int(value) for value in frame_ids),
        usable_pixels=0,
        confidence_counts={},
        median_mm=None,
        p10_mm=None,
        p90_mm=None,
        spread_mm=None,
        reason=reason,
        observations=tuple(observations),
    )


def measure_aligned_depth(
    *, depth, confidence, mask_rgb, intrinsics, rgb_size, pose_mat,
    world_to_table, frame_id,
) -> DepthHeightEvidence:
    """Measure one aligned RGB-mask/depth observation without classifying it."""
    if confidence is None:
        return _unavailable(
            (frame_id,), "the historical frame has no confidence sidecar"
        )
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2 or not depth.size:
        return _unavailable((frame_id,), "the depth frame is empty")
    height, width = depth.shape
    confidence = np.asarray(confidence)
    if confidence.shape != depth.shape:
        confidence = cv2.resize(
            confidence, (width, height), interpolation=cv2.INTER_NEAREST
        )
    mask = np.asarray(mask_rgb, dtype=np.uint8)
    if mask.shape != depth.shape:
        mask = cv2.resize(
            mask, (width, height), interpolation=cv2.INTER_NEAREST
        )
    mask = mask > 0
    valid = (
        mask & np.isfinite(depth) & (depth > 0)
        & np.isfinite(confidence) & (confidence > 0)
    )
    if not np.any(valid):
        return _unavailable(
            (frame_id,), "no masked depth pixels have positive confidence"
        )

    rgb_height, rgb_width = int(rgb_size[0]), int(rgb_size[1])
    scale_x = width / max(1, rgb_width)
    scale_y = height / max(1, rgb_height)
    fx = float(intrinsics["fx"]) * scale_x
    fy = float(intrinsics["fy"]) * scale_y
    cx = float(intrinsics["cx"]) * scale_x
    cy = float(intrinsics["cy"]) * scale_y
    v, u = np.nonzero(valid)
    distance = depth[v, u]
    x = (u - cx) * distance / fx
    y = (v - cy) * distance / fy
    camera_points = np.column_stack([
        x, -y, -distance, np.ones_like(distance)
    ])
    world_points = (
        np.asarray(pose_mat, dtype=float) @ camera_points.T
    ).T
    table_points = (
        np.asarray(world_to_table, dtype=float) @ world_points.T
    ).T
    heights_mm = table_points[:, 2] * 1000.0
    levels, counts = np.unique(confidence[v, u], return_counts=True)
    confidence_counts = {
        str(int(level) if float(level).is_integer() else float(level)): int(count)
        for level, count in zip(levels, counts)
    }
    p10, median, p90 = np.percentile(heights_mm, [10, 50, 90])
    return DepthHeightEvidence(
        action="shadow_only",
        reliability="low",
        frame_ids=(int(frame_id),),
        usable_pixels=int(len(heights_mm)),
        confidence_counts=confidence_counts,
        median_mm=float(median),
        p10_mm=float(p10),
        p90_mm=float(p90),
        spread_mm=float(p90 - p10),
        reason="unvalidated confidence-filtered table-relative height",
    )


def _frame_records(session_dir: Path) -> dict[int, dict]:
    index = session_dir / "frames.jsonl"
    if not index.exists():
        return {}
    return {
        int(record["frame_id"]): record
        for line in index.read_text().splitlines()
        if line.strip()
        for record in (json.loads(line),)
    }


def measure_instance_height(instance, manifest) -> DepthHeightEvidence:
    """Load available selected-frame depth and aggregate shadow diagnostics."""
    session_dir = Path(manifest["session_dir"])
    table_path = session_dir / "table_frame.json"
    records = _frame_records(session_dir)
    frame_ids = tuple(
        observation.frame_id for observation in instance.observations
    )
    if not table_path.exists() or not records:
        return _unavailable(
            frame_ids, "session depth index or table frame is unavailable"
        )
    world_to_table = np.asarray(
        json.loads(table_path.read_text())["world_to_table"], dtype=float
    )
    observations = []
    for observation in instance.observations:
        if observation.mask_original is None:
            continue
        record = records.get(observation.frame_id)
        if record is None:
            continue
        depth_path = session_dir / record["depth"]
        confidence_rel = record.get("confidence")
        confidence_path = (
            session_dir / confidence_rel if confidence_rel else None
        )
        confidence = (
            np.load(confidence_path)
            if confidence_path is not None and confidence_path.exists()
            else None
        )
        evidence = measure_aligned_depth(
            depth=np.load(depth_path),
            confidence=confidence,
            mask_rgb=observation.mask_original,
            intrinsics=record["intrinsics"],
            rgb_size=record["rgb_size"],
            pose_mat=np.asarray(record["pose_mat"], dtype=float),
            world_to_table=world_to_table,
            frame_id=observation.frame_id,
        )
        observations.append(evidence)
    if not observations:
        return _unavailable(
            frame_ids, "no original-frame instance mask has aligned depth"
        )
    usable = [
        evidence for evidence in observations
        if evidence.median_mm is not None
    ]
    serialized = tuple(evidence.to_dict() for evidence in observations)
    if not usable:
        return _unavailable(
            frame_ids,
            "no observation has confidence-filtered depth pixels",
            serialized,
        )
    confidence_counts = {}
    for evidence in usable:
        for level, count in evidence.confidence_counts.items():
            confidence_counts[level] = confidence_counts.get(level, 0) + count
    medians = np.asarray([evidence.median_mm for evidence in usable])
    p10_values = np.asarray([evidence.p10_mm for evidence in usable])
    p90_values = np.asarray([evidence.p90_mm for evidence in usable])
    p10 = float(np.median(p10_values))
    p90 = float(np.median(p90_values))
    return DepthHeightEvidence(
        action="shadow_only",
        reliability="low",
        frame_ids=tuple(
            frame_id for evidence in usable for frame_id in evidence.frame_ids
        ),
        usable_pixels=sum(evidence.usable_pixels for evidence in usable),
        confidence_counts=confidence_counts,
        median_mm=float(np.median(medians)),
        p10_mm=p10,
        p90_mm=p90,
        spread_mm=p90 - p10,
        reason="unvalidated multi-view height remains shadow-only",
        observations=serialized,
    )
