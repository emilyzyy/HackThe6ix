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
    WORKSPACE_CROP_PAD_X_M,
    WORKSPACE_CROP_PAD_Y_M,
    _load_or_build_grid,
    _fill_invalid_with_median,
    _rectify,
    _topdownness,
    sharpness,
)


@dataclass
class ViewCandidate:
    frame_id: int
    quality: float
    image: np.ndarray
    valid: np.ndarray


def select_covering_views(
    candidates: list[ViewCandidate], max_frames: int = 5,
    min_gain: float = 0.002,
) -> list[ViewCandidate]:
    """Greedily cover the usable union, preferring quality on coverage ties."""
    if not candidates or max_frames <= 0:
        return []
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
            break
        selected.append(chosen)
        remaining.remove(chosen)
        covered |= chosen.valid
        if int((covered & union).sum()) / union_pixels >= 0.995:
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
    workspace = grid.workspace_bounds_xy(2)
    if workspace is not None:
        x0 = max(x0, workspace[0][0] - WORKSPACE_CROP_PAD_X_M)
        x1 = min(x1, workspace[0][1] + WORKSPACE_CROP_PAD_X_M)
        y0 = max(y0, workspace[1][0] - WORKSPACE_CROP_PAD_Y_M)
        y1 = min(y1, workspace[1][1] + WORKSPACE_CROP_PAD_Y_M)
    origin_xy = (x0, y0)
    size_wh = (
        int(np.ceil((x1 - x0) * px_per_m)),
        int(np.ceil((y1 - y0) * px_per_m)),
    )
    return table_frame, origin_xy, size_wh


def export_multiview(
    session_dir, px_per_mm: float = 2.0, max_frames: int = 5,
    edge_margin_px: int = 8,
) -> tuple[Path, dict]:
    session_dir = Path(session_dir)
    frames = SessionReader(session_dir).frames()
    if not frames:
        raise ValueError(f"no frames in {session_dir}")

    px_per_m = px_per_mm * 1000.0
    table_frame, origin_xy, size_wh = _canvas_geometry(session_dir, px_per_m)
    kernel_size = 2 * max(0, edge_margin_px) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    candidates: list[ViewCandidate] = []
    for record in frames:
        image, valid = _rectify(
            record, table_frame, origin_xy, px_per_m, size_wh
        )
        if edge_margin_px:
            valid = cv2.erode(valid.astype(np.uint8), kernel).astype(bool)
        image = _fill_invalid_with_median(image, valid)
        quality = _topdownness(record, table_frame) ** 2 * sharpness(
            record.load_rgb()
        )
        candidates.append(ViewCandidate(record.frame_id, quality, image, valid))

    union = np.logical_or.reduce([candidate.valid for candidate in candidates])
    selected = select_covering_views(candidates, max_frames=max_frames)
    selected_union = (
        np.logical_or.reduce([candidate.valid for candidate in selected])
        if selected else np.zeros_like(union)
    )

    output_dir = session_dir / "multiview"
    output_dir.mkdir(exist_ok=True)
    view_entries = []
    for candidate in selected:
        stem = f"view_{candidate.frame_id:05d}"
        image_path = output_dir / f"{stem}.jpg"
        mask_path = output_dir / f"{stem}.mask.png"
        cv2.imwrite(str(image_path), candidate.image,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(mask_path), candidate.valid.astype(np.uint8) * 255)
        view_entries.append({
            "frame_id": candidate.frame_id,
            "quality": candidate.quality,
            "image": str(image_path.relative_to(session_dir)),
            "mask": str(mask_path.relative_to(session_dir)),
            "size_wh": list(size_wh),
        })

    canvas_pixels = union.size
    manifest = {
        "version": 1,
        "session_dir": str(session_dir.resolve()),
        "px_per_m": px_per_m,
        "origin_xy": list(origin_xy),
        "size_wh": list(size_wh),
        "edge_margin_px": edge_margin_px,
        "union_coverage": float(union.sum() / canvas_pixels),
        "selected_coverage": float(selected_union.sum() / canvas_pixels),
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
    parser.add_argument("--edge-margin", type=int, default=8)
    args = parser.parse_args()
    path, manifest = export_multiview(
        args.session, px_per_mm=args.px_per_mm,
        max_frames=args.max_frames, edge_margin_px=args.edge_margin,
    )
    print(f"wrote {path}")
    print(
        f"selected {len(manifest['views'])} views; usable RGB coverage "
        f"{100 * manifest['selected_coverage']:.1f}% / "
        f"{100 * manifest['union_coverage']:.1f}% union"
    )


if __name__ == "__main__":
    main()
