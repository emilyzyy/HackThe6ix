"""Phase 4: synthesize a high-res orthographic top-down image of the table.

Default (best-frame) mode rectifies the single most top-down, least blurry
frame via a plane homography. Optional --mode ortho builds a diagnostic
composite where every output pixel takes its color from the best observation
across all frames.

Output feeds lego-cv:  sessions/<ts>/topdown.jpg  (+ topdown.json metadata)

Usage: python topdown.py sessions/<ts> [--mode ortho|best-frame]
                                       [--px-per-mm 2] [--out topdown.jpg]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from coverage import CoverageGrid, replay_session
from plane import compute_table_frame, load_table_frame
from session_io import SessionReader
from transforms import intrinsics_to_K

WORKSPACE_CROP_PAD_X_M = 0.025
WORKSPACE_CROP_PAD_Y_M = 0.005


def sharpness(img_bgr):
    """Focus measure: variance of the Laplacian of the grayscale image."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def plane_homography(K, cam_to_world, world_to_table, px_per_m, origin_xy):
    """3x3 H mapping top-down pixel coords -> source image pixels.

    Top-down pixel (u, v) sits at table point
    (origin_xy[0] + u/px_per_m, origin_xy[1] + v/px_per_m, 0).
    """
    # Table pixel -> homogeneous world point, as a 4x3 affine map.
    T2W = np.linalg.inv(world_to_table)
    A = np.zeros((4, 3))
    A[:3, 0] = T2W[:3, 0] / px_per_m
    A[:3, 1] = T2W[:3, 1] / px_per_m
    A[:3, 2] = (T2W @ [origin_xy[0], origin_xy[1], 0.0, 1.0])[:3]
    A[3, 2] = 1.0
    # World -> camera, then ARKit camera -> pinhole (flip y, z; see transforms).
    W2C = np.linalg.inv(cam_to_world)
    F = np.diag([1.0, -1.0, -1.0])
    return K @ F @ W2C[:3, :] @ A


def _frame_geometry(rec, tf, origin_xy, px_per_m, out_wh):
    """Per-output-pixel geometric score terms for one frame, in table space."""
    W2T = tf["world_to_table"]
    cam_t = (W2T @ np.append(rec.pose_mat[:3, 3], 1.0))[:3]
    w, h = out_wh
    us = origin_xy[0] + (np.arange(w) + 0.5) / px_per_m
    vs = origin_xy[1] + (np.arange(h) + 0.5) / px_per_m
    dx = us[None, :] - cam_t[0]
    dy = vs[:, None] - cam_t[1]
    dz = -cam_t[2]  # table points at z=0
    dist2 = dx * dx + dy * dy + dz * dz
    cos_inc = np.abs(dz) / np.sqrt(dist2)
    return cos_inc, dist2


def _rectify(rec, tf, origin_xy, px_per_m, out_wh):
    """Warp one frame's RGB onto the top-down grid; returns (image, valid)."""
    K = intrinsics_to_K(**rec.intrinsics)
    H = plane_homography(K, rec.pose_mat, tf["world_to_table"],
                         px_per_m, origin_xy)
    img = rec.load_rgb()
    warped = cv2.warpPerspective(
        img, H.astype(np.float64), out_wh,
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ones = np.ones(img.shape[:2], dtype=np.uint8)
    valid = cv2.warpPerspective(
        ones, H.astype(np.float64), out_wh,
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return warped, valid


def _fill_invalid_with_median(image, valid):
    """Fill pixels outside a rectified frame with its median valid color."""
    if valid.all() or not valid.any():
        return image.copy()
    filled = image.copy()
    fill_color = np.median(image[valid], axis=0).astype(image.dtype)
    filled[~valid] = fill_color
    return filled


def _topdownness(rec, tf):
    """cos of the angle between the camera's forward axis and straight down."""
    fwd_world = rec.pose_mat[:3, :3] @ [0, 0, -1]
    n = tf["world_to_table"][2, :3]  # table up-normal in world coords (row z)
    return float(np.clip(-(fwd_world @ n), 0.0, 1.0))


def _load_or_build_grid(session_dir, cell_m=0.005):
    cov = Path(session_dir) / "coverage.npz"
    if cov.exists():
        return CoverageGrid.load(cov)
    grid, _ = replay_session(session_dir, cell_m=cell_m)
    return grid


def export_topdown(session_dir, mode="best-frame", px_per_mm=2.0,
                   out_name="topdown.jpg", workspace=True, min_seen=2):
    session_dir = Path(session_dir)
    try:
        tf = load_table_frame(session_dir)
    except FileNotFoundError:
        tf = compute_table_frame(session_dir)
        tf["world_to_table"] = np.array(tf["world_to_table"])

    px_per_m = px_per_mm * 1000.0
    (x0, x1), (y0, y1) = tf["extent_xy"]

    # Bound the output to the observed workspace so background beyond the
    # swept table (desk edge, laptop, floor) never reaches the detector.
    if workspace:
        grid = _load_or_build_grid(session_dir)
        wb = grid.workspace_bounds_xy(min_seen)
        if wb is None:
            workspace = False
        else:
            x0 = max(x0, wb[0][0] - WORKSPACE_CROP_PAD_X_M)
            x1 = min(x1, wb[0][1] + WORKSPACE_CROP_PAD_X_M)
            y0 = max(y0, wb[1][0] - WORKSPACE_CROP_PAD_Y_M)
            y1 = min(y1, wb[1][1] + WORKSPACE_CROP_PAD_Y_M)
    origin_xy = (x0, y0)
    out_wh = (int(np.ceil((x1 - x0) * px_per_m)),
              int(np.ceil((y1 - y0) * px_per_m)))
    frames = SessionReader(session_dir).frames()
    if not frames:
        raise ValueError(f"no frames in {session_dir}")

    best_frame_id = None
    if mode == "best-frame":
        scores = [( _topdownness(rec, tf) ** 2 * sharpness(rec.load_rgb()),
                   rec) for rec in frames]
        score, rec = max(scores, key=lambda s: s[0])
        best_frame_id = rec.frame_id
        out_img, valid = _rectify(rec, tf, origin_xy, px_per_m, out_wh)
        out_img = _fill_invalid_with_median(out_img, valid)
    elif mode == "ortho":
        out_img = np.zeros((out_wh[1], out_wh[0], 3), dtype=np.uint8)
        best_score = np.zeros((out_wh[1], out_wh[0]), dtype=np.float32)
        for rec in frames:
            warped, valid = _rectify(rec, tf, origin_xy, px_per_m, out_wh)
            cos_inc, dist2 = _frame_geometry(rec, tf, origin_xy, px_per_m,
                                             out_wh)
            score = (sharpness(rec.load_rgb()) * cos_inc ** 3
                     / dist2).astype(np.float32)
            score[~valid] = 0.0
            better = score > best_score
            out_img[better] = warped[better]
            best_score = np.maximum(best_score, score)
    else:
        raise ValueError(f"unknown mode {mode!r}")

    out_path = session_dir / out_name
    cv2.imwrite(str(out_path), out_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Pixel (u, v) maps to table point origin_xy + (u, v) / px_per_m.
    meta = {"mode": mode, "px_per_m": px_per_m, "origin_xy": list(origin_xy),
            "size_wh": list(out_wh), "best_frame_id": best_frame_id,
            "workspace": bool(workspace)}
    with open(session_dir / "topdown.json", "w") as f:
        json.dump(meta, f, indent=1)
    return out_path, meta


def main():
    ap = argparse.ArgumentParser(description="Export a top-down table image.")
    ap.add_argument("session")
    ap.add_argument("--mode", choices=["ortho", "best-frame"],
                    default="best-frame")
    ap.add_argument("--px-per-mm", type=float, default=2.0)
    ap.add_argument("--out", default="topdown.jpg")
    ap.add_argument("--no-workspace", action="store_true",
                    help="skip cropping to the bounded workspace")
    args = ap.parse_args()
    path, meta = export_topdown(args.session, mode=args.mode,
                                px_per_mm=args.px_per_mm, out_name=args.out,
                                workspace=not args.no_workspace)
    w, h = meta["size_wh"]
    extra = (f" (best frame {meta['best_frame_id']})"
             if meta["best_frame_id"] is not None else "")
    print(f"wrote {path} — {w}x{h} px, {args.px_per_mm:g} px/mm, "
          f"{meta['mode']} mode{extra}")


if __name__ == "__main__":
    main()
