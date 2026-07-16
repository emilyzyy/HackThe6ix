"""Synthetic session generator: the offline test bed for Phases 2-4.

Renders a checkerboard-textured table plane at world z=0 (Z-up world) viewed
by a camera orbiting above it, and writes a real session (JPEG/npy +
frames.jsonl) through SessionWriter — so downstream modules exercise the exact
on-disk schema. Depth is the exact ray-plane z-depth, so unprojection lands on
the plane to machine precision.

Also runnable:  python -m tests.synthetic --out sessions/synthetic
"""

import argparse

import cv2
import numpy as np

from session_io import SessionWriter
from transforms import intrinsics_to_K, scale_intrinsics, mat_to_quat

CHECKER_M = 0.02  # checker square size, meters


def look_at_pose(pos, target, up=(0.0, 1.0, 0.0)):
    """4x4 camera-to-world with the camera at pos looking at target (-Z fwd)."""
    pos = np.asarray(pos, dtype=float)
    fwd = np.asarray(target, dtype=float) - pos
    fwd /= np.linalg.norm(fwd)
    up = np.asarray(up, dtype=float)
    if abs(fwd @ up) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, fwd)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = right, true_up, -fwd
    T[:3, 3] = pos
    return T


def sample_texture(xy):
    """(N,2) table-plane coords -> (N,3) uint8 RGB checkerboard + gradient."""
    xy = np.asarray(xy)
    checker = (np.floor(xy[:, 0] / CHECKER_M) + np.floor(xy[:, 1] / CHECKER_M)) % 2
    r = np.where(checker > 0, 220, 50)
    b = np.where(checker > 0, 200, 160)
    g = np.clip(128 + 600 * xy[:, 0], 0, 255)  # gradient breaks the symmetry
    return np.column_stack([r, g, b]).astype(np.uint8)


def _ray_plane_zdepth(K, cam_to_world, hw):
    """Per-pixel z-depth to the z=0 plane and the world hit points."""
    h, w = hw
    v, u = np.mgrid[0:h, 0:w]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    # ARKit camera ray directions, parameterized by z-depth.
    dirs_cam = np.stack([(u - cx) / fx, -(v - cy) / fy,
                         -np.ones_like(u, dtype=float)], axis=-1)
    R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
    dirs_world = dirs_cam @ R.T
    denom = dirs_world[..., 2]
    z = -t[2] / denom  # solves (t + z*dir).z == 0
    hits = t + z[..., None] * dirs_world
    return z.astype(np.float32), hits


def make_synthetic_session(out_dir, n_frames=12, height=0.40, radius=0.15,
                           rgb_hw=(480, 640), depth_hw=(192, 256), seed=0):
    rng = np.random.default_rng(seed)
    fx = rgb_hw[1] / (2 * np.tan(np.deg2rad(30)))  # ~60 deg horizontal FOV
    coeffs = {"fx": fx, "fy": fx, "cx": rgb_hw[1] / 2, "cy": rgb_hw[0] / 2}
    K_rgb = intrinsics_to_K(**coeffs)
    K_depth = scale_intrinsics(K_rgb, (rgb_hw[1], rgb_hw[0]),
                               (depth_hw[1], depth_hw[0]))

    writer = SessionWriter(out_dir)
    for i in range(n_frames):
        theta = 2 * np.pi * i / n_frames
        pos = [radius * np.cos(theta), radius * np.sin(theta),
               height + rng.uniform(-0.02, 0.02)]
        target = rng.uniform(-0.02, 0.02, size=3) * [1, 1, 0]
        T = look_at_pose(pos, target)

        depth, _ = _ray_plane_zdepth(K_depth, T, depth_hw)
        _, hits = _ray_plane_zdepth(K_rgb, T, rgb_hw)
        rgb = sample_texture(hits[..., :2].reshape(-1, 2)).reshape(*rgb_hw, 3)

        qx, qy, qz, qw = mat_to_quat(T[:3, :3])
        pose = {"qx": qx, "qy": qy, "qz": qz, "qw": qw,
                "tx": T[0, 3], "ty": T[1, 3], "tz": T[2, 3]}
        writer.add_frame(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth, coeffs,
                         pose, device_type=1, timestamp=i / 6.0)
    writer.close()
    return writer.session_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate a synthetic session.")
    ap.add_argument("--out", default="sessions/synthetic")
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()
    path = make_synthetic_session(args.out, n_frames=args.frames)
    print("synthetic session written to", path)
