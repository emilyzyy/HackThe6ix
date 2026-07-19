"""Versioned scene-state artifact for the live scan viewer and showcase.

A fused scan is a post-capture batch (see PROGRESS.md Phase-11 timing: fusion is
instant but view-selection/YOLO/Brickognize all run after capture ends), so a
UI cannot get fused pieces incrementally *during* a scan. What it CAN do is
replay the recorded frames in capture order and project each already-fused
piece into the current frame using that frame's pose + intrinsics, lighting a
piece the moment it first becomes visible. This module exports exactly the
data that replay needs, using only signals the pipeline genuinely produces.

Deliberately honest about missing signals:
- A fused instance has no persistent cross-scan identity; ``instance_id`` is the
  deterministic sorted index within this scan's output, stable across the
  artifacts of one scan but re-derived if the session is re-scanned.
- ``confidence`` is the YOLO box confidence and is only real for the ``seg``
  detector; the classical ``cv`` detector genuinely has none, so it is ``null``
  rather than a fabricated proxy.
- ``complete`` is the real box-inside-workspace-margin signal, not a learned
  mask-completeness score (no such score exists in this pipeline).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCENE_STATE_VERSION = 1


def canvas_to_table_xy(points, origin_xy, px_per_m):
    """Canvas pixels -> table-frame metres (the export canvas is a linear,
    axis-aligned rasterisation of the table plane)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    table = np.empty_like(pts)
    table[:, 0] = origin_xy[0] + pts[:, 0] / px_per_m
    table[:, 1] = origin_xy[1] + pts[:, 1] / px_per_m
    return table


def _box_table(box_canvas, origin_xy, px_per_m):
    if box_canvas is None:
        return None, None
    x0, y0, x1, y1 = box_canvas
    corners = canvas_to_table_xy(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], origin_xy, px_per_m
    )
    table_box = [
        float(corners[:, 0].min()), float(corners[:, 1].min()),
        float(corners[:, 0].max()), float(corners[:, 1].max()),
    ]
    centroid = [
        float((table_box[0] + table_box[2]) / 2),
        float((table_box[1] + table_box[3]) / 2),
    ]
    return table_box, centroid


def _best_polygon_table(instance, origin_xy, px_per_m):
    polygons = getattr(instance.best, "polygons_canvas", ()) or ()
    if not polygons:
        return None
    largest = max(polygons, key=lambda p: len(np.asarray(p).reshape(-1, 2)))
    table = canvas_to_table_xy(largest, origin_xy, px_per_m)
    return [[float(x), float(y)] for x, y in table]


def _frame_timestamps(session_dir: Path) -> dict[int, float]:
    index = session_dir / "frames.jsonl"
    if not index.exists():
        return {}
    timestamps: dict[int, float] = {}
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        timestamps[int(record["frame_id"])] = float(record.get("timestamp", 0.0))
    return timestamps


def _world_to_table(session_dir: Path):
    table_path = session_dir / "table_frame.json"
    if not table_path.exists():
        return None
    try:
        payload = json.loads(table_path.read_text())
    except (ValueError, OSError):
        return None
    matrix = payload.get("world_to_table")
    return matrix if matrix is not None else None


def _observation_row(observation, timestamps):
    box_original = (
        list(observation.box_original)
        if observation.box_original is not None else None
    )
    centroid = None
    if box_original is not None:
        centroid = [
            float((box_original[0] + box_original[2]) / 2),
            float((box_original[1] + box_original[3]) / 2),
        ]
    return {
        "frame_id": int(observation.frame_id),
        "timestamp": timestamps.get(int(observation.frame_id)),
        "view_index": int(observation.view_index),
        "detector": observation.detector,
        "box_original": box_original,
        "centroid_original": centroid,
        # Real YOLO box confidence for seg; genuinely absent for cv.
        "confidence": (
            None if observation.confidence is None
            else float(observation.confidence)
        ),
        "complete": bool(observation.complete),
        "boundary_clearance": float(observation.boundary_clearance),
        "identification_focus": (
            None if observation.identification_focus is None
            else float(observation.identification_focus)
        ),
    }


def build_scene_state(
    result, session_dir, *, detector: str, predicted_colors: list[str],
) -> dict:
    """Assemble the versioned scene-state dict from a scan result."""
    session_dir = Path(session_dir)
    manifest = result.manifest
    origin_xy = tuple(manifest["origin_xy"])
    px_per_m = float(manifest["px_per_m"])
    timestamps = _frame_timestamps(session_dir)
    geometry = manifest.get("workspace_geometry", {})
    selection = manifest.get("selection", {})

    instances = []
    for instance_id, instance in enumerate(result.instances):
        id_crop = (
            result.id_crops[instance_id]
            if instance_id < len(result.id_crops) else {}
        )
        brickognize = id_crop.get("brickognize", {})
        table_box, table_centroid = _box_table(
            instance.best.box, origin_xy, px_per_m
        )
        color = (
            predicted_colors[instance_id]
            if instance_id < len(predicted_colors) else None
        )
        instances.append({
            "instance_id": instance_id,
            "part_id": brickognize.get("part_id"),
            "part_name": brickognize.get("name"),
            "id_score": brickognize.get("score"),
            # Colour is exported for display but is known-shaky; the showcase
            # confirmation asks about part identity, not colour.
            "color": color,
            "canvas_box": list(instance.best.box),
            "table_box_xy": table_box,
            "table_centroid_xy": table_centroid,
            "table_polygon_xy": _best_polygon_table(
                instance, origin_xy, px_per_m
            ),
            "best_frame_id": int(instance.best.frame_id),
            "flags": list(id_crop.get("flags", [])),
            "observations": [
                _observation_row(observation, timestamps)
                for observation in instance.observations
            ],
        })

    return {
        "version": SCENE_STATE_VERSION,
        "session_id": session_dir.name,
        "session_dir": str(session_dir),
        "detector": detector,
        "table": {
            "world_to_table": _world_to_table(session_dir),
            "origin_xy": [float(origin_xy[0]), float(origin_xy[1])],
            "px_per_m": px_per_m,
            "canvas_size_wh": list(manifest["size_wh"]),
        },
        "coverage": {
            # Spatial: fraction of the physical workspace observed by the
            # selected views. Never a fraction of pieces identified.
            "spatial_fraction": manifest.get("selected_coverage"),
            "union_fraction": manifest.get("union_coverage"),
            "coverage_complete": selection.get("coverage_complete"),
            "workspace_bounds_table_xy": geometry.get("hard_bounds_table_xy"),
            "workspace_contours_table_xy": geometry.get(
                "hard_contours_table_xy", []
            ),
        },
        "timings_s": dict(result.timings_s),
        "instance_count": len(instances),
        "instances": instances,
    }


def write_scene_state(
    result, session_dir, output_dir, *, detector, predicted_colors,
) -> Path:
    state = build_scene_state(
        result, session_dir, detector=detector,
        predicted_colors=predicted_colors,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "scene_state.json"
    path.write_text(json.dumps(state, indent=2) + "\n")
    return path
