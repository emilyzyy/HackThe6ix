"""Export a small coverage-complete set of clean rectified session views.

Every exported view shares one table-coordinate canvas and carries a valid
pixel mask. Inventory detection can therefore run before any images are
blended, then associate observations by their common pixel/table positions.

Usage: python multiview.py sessions/<timestamp>
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from plane import compute_table_frame, load_table_frame
from session_io import SessionReader
from topdown import (
    _load_or_build_grid,
    _fill_invalid_with_median,
    _rectify,
    _topdownness,
    plane_homography,
    sharpness,
)
from transforms import intrinsics_to_K, project_points

SELECTION_PX_PER_M = 250.0


@dataclass
class ViewCandidate:
    frame_id: int
    quality: float
    image: np.ndarray | None
    valid: np.ndarray
    topdownness: float | None = None
    sharpness_score: float | None = None
    camera_position_table_xyz: tuple[float, float, float] | None = None
    camera_bearing_deg: float | None = None


def circular_separation_deg(first: float, second: float) -> float:
    """Smallest unsigned separation between two circular bearings."""
    difference = abs((float(first) - float(second)) % 360.0)
    return min(difference, 360.0 - difference)


def selection_geometry(
    selected: list[ViewCandidate], minimum_diversity_deg: float = 30.0,
) -> dict:
    pairs = []
    for first_index, first in enumerate(selected):
        if first.camera_bearing_deg is None:
            continue
        for second in selected[first_index + 1:]:
            if second.camera_bearing_deg is None:
                continue
            pairs.append({
                "frame_ids": [first.frame_id, second.frame_id],
                "separation_deg": circular_separation_deg(
                    first.camera_bearing_deg, second.camera_bearing_deg
                ),
            })
    minimum = (
        min(pair["separation_deg"] for pair in pairs) if pairs else None
    )
    return {
        "pairwise_bearing_separation": pairs,
        "minimum_bearing_separation_deg": minimum,
        "requested_minimum_deg": float(minimum_diversity_deg),
        "angle_diverse": bool(
            minimum is not None and minimum >= minimum_diversity_deg
        ),
    }


def select_covering_views(
    candidates: list[ViewCandidate], max_frames: int = 5,
    min_gain: float = 0.002, min_frames: int = 1,
) -> list[ViewCandidate]:
    """Cover the union, then retain angle-diverse confirmation views.

    Coverage alone can legitimately stop after one frame.  Instance inventory
    still needs independent views so a one-frame background false positive can
    be contradicted without raising a global confidence threshold that would
    also delete real edge pieces.
    """
    if not candidates or max_frames <= 0:
        return []
    min_frames = max(1, min(int(min_frames), max_frames, len(candidates)))
    union = np.logical_or.reduce([candidate.valid for candidate in candidates])
    union_pixels = int(union.sum())
    if union_pixels == 0:
        return []

    selected: list[ViewCandidate] = []
    covered = np.zeros_like(union)
    remaining = list(candidates)
    while remaining and len(selected) < max_frames:
        ranked = []
        for candidate in remaining:
            new_pixels = int((candidate.valid & ~covered).sum())
            ranked.append((new_pixels, candidate.quality,
                           -candidate.frame_id, candidate))
        new_pixels, _, _, chosen = max(ranked, key=lambda item: item[:3])
        if selected and new_pixels / union_pixels < min_gain:
            if len(selected) >= min_frames:
                break
            # Once meaningful coverage gain is exhausted, choose a clean view
            # from a different camera bearing. Restrict this to the better half
            # of the quality distribution; time separation remains the
            # tiebreaker and the fallback when bearing is unavailable.
            quality_floor = float(np.median([
                candidate.quality for candidate in candidates
            ]))
            eligible = [
                candidate for candidate in remaining
                if candidate.quality >= quality_floor
            ] or remaining
            frame_span = max(
                1,
                max(candidate.frame_id for candidate in candidates)
                - min(candidate.frame_id for candidate in candidates),
            )
            quality_scale = max(
                1e-9, max(max(0.0, candidate.quality)
                          for candidate in eligible)
            )
            bearing_eligible = [
                candidate for candidate in eligible
                if candidate.camera_bearing_deg is not None
            ]
            selected_bearings = [
                item.camera_bearing_deg for item in selected
                if item.camera_bearing_deg is not None
            ]
            if bearing_eligible and selected_bearings:
                chosen = max(
                    bearing_eligible,
                    key=lambda candidate: (
                        min(circular_separation_deg(
                            candidate.camera_bearing_deg, bearing
                        ) for bearing in selected_bearings),
                        min(abs(candidate.frame_id - item.frame_id)
                            for item in selected) / frame_span,
                        candidate.quality,
                        -candidate.frame_id,
                    ),
                )
            else:
                chosen = max(
                    eligible,
                    key=lambda candidate: (
                        min(abs(candidate.frame_id - item.frame_id)
                            for item in selected) / frame_span
                        * (0.5 + 0.5 * max(0.0, candidate.quality)
                           / quality_scale),
                        candidate.quality,
                        -candidate.frame_id,
                    ),
                )
        selected.append(chosen)
        remaining.remove(chosen)
        covered |= chosen.valid
        if (
            len(selected) >= min_frames
            and int((covered & union).sum()) / union_pixels >= 0.995
        ):
            break
    return selected


def _canvas_geometry(session_dir: Path, px_per_m: float):
    try:
        table_frame = load_table_frame(session_dir)
    except FileNotFoundError:
        table_frame = compute_table_frame(session_dir)
        table_frame["world_to_table"] = np.array(table_frame["world_to_table"])

    (x0, x1), (y0, y1) = table_frame["extent_xy"]
    grid = _load_or_build_grid(session_dir)
    workspace = grid.admissible_workspace_bounds_xy(2)
    if workspace is not None:
        x0 = max(x0, workspace[0][0])
        x1 = min(x1, workspace[0][1])
        y0 = max(y0, workspace[1][0])
        y1 = min(y1, workspace[1][1])
    origin_xy = (x0, y0)
    size_wh = (
        int(np.ceil((x1 - x0) * px_per_m)),
        int(np.ceil((y1 - y0) * px_per_m)),
    )
    return table_frame, grid, origin_xy, size_wh


def _workspace_mask(contours_xy, origin_xy, px_per_m, size_wh):
    """Rasterize table-coordinate admissible contours on a canvas."""
    mask = np.zeros((size_wh[1], size_wh[0]), dtype=np.uint8)
    origin = np.asarray(origin_xy, dtype=float)
    polygons = []
    for contour in contours_xy:
        points = np.rint(
            (np.asarray(contour, dtype=float) - origin) * px_per_m
        ).astype(np.int32)
        if len(points) >= 3:
            polygons.append(points)
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask.astype(bool)


def _erode_valid(valid, edge_margin_px):
    if edge_margin_px <= 0:
        return valid.astype(bool)
    kernel_size = 2 * int(edge_margin_px) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(
        valid.astype(np.uint8), kernel,
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    ).astype(bool)


def _raise_uv_per_cm(record, table_frame, origin_xy, px_per_m, size_wh):
    """Image displacement (px) of a point 1 cm above the plane, at canvas
    center — the direction identification crops must extend so raised piece
    tops are not clipped in oblique views."""
    W2T = table_frame["world_to_table"]
    T2W = np.linalg.inv(W2T)
    cx = origin_xy[0] + (size_wh[0] / 2) / px_per_m
    cy = origin_xy[1] + (size_wh[1] / 2) / px_per_m
    p0 = (T2W @ [cx, cy, 0.0, 1.0])[:3]
    p1 = (T2W @ [cx, cy, 0.01, 1.0])[:3]
    K = intrinsics_to_K(**record.intrinsics)
    uv, valid = project_points(np.stack([p0, p1]), K, record.pose_mat,
                               (10**9, 10**9))
    if not valid.all():
        return [0.0, 0.0]
    return [float(uv[1][0] - uv[0][0]), float(uv[1][1] - uv[0][1])]


def _camera_geometry(record, table_frame, origin_xy, px_per_m, size_wh):
    camera_world = np.append(record.pose_mat[:3, 3], 1.0)
    camera_table = (
        table_frame["world_to_table"] @ camera_world
    )[:3]
    center_x = origin_xy[0] + size_wh[0] / (2.0 * px_per_m)
    center_y = origin_xy[1] + size_wh[1] / (2.0 * px_per_m)
    bearing = float(np.degrees(np.arctan2(
        camera_table[1] - center_y,
        camera_table[0] - center_x,
    )) % 360.0)
    return tuple(float(value) for value in camera_table), bearing


def export_multiview(
    session_dir, px_per_mm: float = 2.0, max_frames: int = 5,
    edge_margin_px: int = 8, min_frames: int = 3,
) -> tuple[Path, dict]:
    session_dir = Path(session_dir)
    frames = SessionReader(session_dir).frames()
    if not frames:
        raise ValueError(f"no frames in {session_dir}")

    px_per_m = px_per_mm * 1000.0
    table_frame, grid, origin_xy, size_wh = _canvas_geometry(
        session_dir, px_per_m
    )
    contours_xy = grid.admissible_workspace_contours_xy()
    selection_px_per_m = min(px_per_m, SELECTION_PX_PER_M)
    width_m = size_wh[0] / px_per_m
    height_m = size_wh[1] / px_per_m
    selection_size_wh = (
        max(1, int(np.ceil(width_m * selection_px_per_m))),
        max(1, int(np.ceil(height_m * selection_px_per_m))),
    )
    selection_workspace = _workspace_mask(
        contours_xy, origin_xy, selection_px_per_m, selection_size_wh
    )
    selection_edge_margin = int(round(
        max(0, edge_margin_px) * selection_px_per_m / px_per_m
    ))
    candidates: list[ViewCandidate] = []
    for record in frames:
        _, valid = _rectify(
            record, table_frame, origin_xy, selection_px_per_m,
            selection_size_wh,
        )
        valid = _erode_valid(valid, selection_edge_margin)
        valid &= selection_workspace
        topdownness = _topdownness(record, table_frame)
        sharpness_score = sharpness(record.load_rgb())
        quality = topdownness ** 2 * sharpness_score
        camera_position, camera_bearing = _camera_geometry(
            record, table_frame, origin_xy, px_per_m, size_wh
        )
        candidates.append(ViewCandidate(
            record.frame_id,
            quality,
            None,
            valid,
            topdownness=topdownness,
            sharpness_score=sharpness_score,
            camera_position_table_xyz=camera_position,
            camera_bearing_deg=camera_bearing,
        ))

    union = np.logical_or.reduce([candidate.valid for candidate in candidates])
    confirmation_views = min(max_frames, max(1, min_frames))
    selected = select_covering_views(
        candidates, max_frames=max_frames, min_frames=confirmation_views
    )
    selected_union = (
        np.logical_or.reduce([candidate.valid for candidate in selected])
        if selected else np.zeros_like(union)
    )

    records = {record.frame_id: record for record in frames}
    full_workspace = _workspace_mask(
        contours_xy, origin_xy, px_per_m, size_wh
    )
    output_dir = session_dir / "multiview"
    output_dir.mkdir(exist_ok=True)
    view_entries = []
    for candidate in selected:
        record = records[candidate.frame_id]
        image, valid = _rectify(
            record, table_frame, origin_xy, px_per_m, size_wh
        )
        valid = _erode_valid(valid, edge_margin_px)
        valid &= full_workspace
        image = _fill_invalid_with_median(image, valid)
        candidate.image = image
        stem = f"view_{candidate.frame_id:05d}"
        image_path = output_dir / f"{stem}.jpg"
        mask_path = output_dir / f"{stem}.mask.png"
        cv2.imwrite(str(image_path), image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_path), valid.astype(np.uint8) * 255)
        homography = plane_homography(
            intrinsics_to_K(**record.intrinsics), record.pose_mat,
            table_frame["world_to_table"], px_per_m, origin_xy,
        )
        view_entries.append({
            "frame_id": candidate.frame_id,
            "quality": candidate.quality,
            "topdownness": candidate.topdownness,
            "sharpness": candidate.sharpness_score,
            "camera_position_table_xyz": list(
                candidate.camera_position_table_xyz
            ) if candidate.camera_position_table_xyz is not None else None,
            "camera_bearing_deg": candidate.camera_bearing_deg,
            "image": str(image_path.relative_to(session_dir)),
            "mask": str(mask_path.relative_to(session_dir)),
            "size_wh": list(size_wh),
            # Bridge geometry (manifest v2): canvas px -> original frame px,
            # plus where the original pixels live and how a raised point
            # displaces in that frame (identification crops must not clip
            # piece tops in oblique views).
            "homography": homography.tolist(),
            "rgb": record.rgb,
            "rgb_size": list(record.rgb_size),
            "raise_uv_per_cm": _raise_uv_per_cm(
                record, table_frame, origin_xy, px_per_m, size_wh
            ),
        })

    workspace_pixels = max(1, int(selection_workspace.sum()))
    union_pixels = int(union.sum())
    manifest = {
        "version": 3,
        "session_dir": str(session_dir.resolve()),
        "px_per_m": px_per_m,
        "origin_xy": list(origin_xy),
        "size_wh": list(size_wh),
        "edge_margin_px": edge_margin_px,
        "union_coverage": float(union_pixels / workspace_pixels),
        "selected_coverage": float(
            selected_union.sum() / workspace_pixels
        ),
        "selection": {
            "max_frames": max_frames,
            "min_confirmation_views": confirmation_views,
            "strategy": "coverage_then_angle_diverse_quality",
            "coverage_px_per_m": selection_px_per_m,
            "selected_union_fraction": float(
                (selected_union & union).sum() / max(1, union_pixels)
            ),
            **selection_geometry(selected),
        },
        # The hard polygon is intentionally broader than the dense workspace
        # used for the legacy rectified canvas.  It comes only from repeated
        # table-plane depth observations, never from detector outputs.
        "workspace_geometry": {
            "hard_source": "depth_observed_plane_component",
            "hard_contours_table_xy": (
                contours_xy
            ),
            "hard_bounds_table_xy": (
                grid.admissible_workspace_bounds_xy(2)
            ),
            "dense_bounds_table_xy": grid.workspace_bounds_xy(2),
            "table_extent_xy": table_frame["extent_xy"],
        },
        "views": view_entries,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path, manifest


def main():
    parser = argparse.ArgumentParser(
        description="Export coverage-aware clean views for multi-view detection."
    )
    parser.add_argument("session")
    parser.add_argument("--px-per-mm", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=5)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--edge-margin", type=int, default=8)
    args = parser.parse_args()
    path, manifest = export_multiview(
        args.session, px_per_mm=args.px_per_mm,
        max_frames=args.max_frames, edge_margin_px=args.edge_margin,
        min_frames=args.min_frames,
    )
    print(f"wrote {path}")
    print(
        f"selected {len(manifest['views'])} views; usable RGB coverage "
        f"{100 * manifest['selected_coverage']:.1f}% / "
        f"{100 * manifest['union_coverage']:.1f}% union"
    )


if __name__ == "__main__":
    main()
