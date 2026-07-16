"""Pure pose / camera-projection math shared by all lego-capture modules.

Conventions (record3d / ARKit):
- Camera-to-world poses. Camera space: +X right, +Y up, camera looks down -Z.
- Image pixel v grows downward, so pixel->camera unprojection flips Y and Z.
- Intrinsics dicts use fx, fy, cx, cy (record3d's tx/ty coeffs are renamed to
  cx/cy at the capture boundary in capture.py, nowhere else).
"""

import numpy as np


def quat_to_mat(qx, qy, qz, qw):
    """3x3 rotation matrix from a quaternion (normalized internally)."""
    n = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


def pose_to_mat(qx, qy, qz, qw, tx, ty, tz):
    """4x4 camera-to-world transform from quaternion + translation."""
    T = np.eye(4)
    T[:3, :3] = quat_to_mat(qx, qy, qz, qw)
    T[:3, 3] = [tx, ty, tz]
    return T


def intrinsics_to_K(fx, fy, cx, cy):
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


def scale_intrinsics(K, from_wh, to_wh):
    """Rescale a pinhole K given for one image size to another size."""
    sx = to_wh[0] / from_wh[0]
    sy = to_wh[1] / from_wh[1]
    S = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]])
    return S @ K


def unproject_depth(depth, K, cam_to_world, stride=1, max_depth=np.inf):
    """Depth map -> (N,3) world points. Skips nonpositive / too-deep pixels."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[::stride, ::stride].astype(np.float64)
    u, v, d = u.ravel(), v.ravel(), d.ravel()
    keep = (d > 0) & (d <= max_depth)
    u, v, d = u[keep], v[keep], d[keep]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    x = (u - cx) * d / fx
    y = (v - cy) * d / fy
    # Pinhole (image-down y, camera-forward z) -> ARKit camera (+Y up, -Z forward).
    pts_cam = np.column_stack([x, -y, -d])
    R, t = cam_to_world[:3, :3], cam_to_world[:3, 3]
    return pts_cam @ R.T + t


def project_points(points_world, K, cam_to_world, wh):
    """World points -> (N,2) pixels and (N,) validity (in front and in image)."""
    world_to_cam = np.linalg.inv(cam_to_world)
    pts = np.asarray(points_world, dtype=np.float64)
    pts_cam = pts @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    # ARKit camera -> pinhole coords (see unproject_depth).
    x, y, d = pts_cam[:, 0], -pts_cam[:, 1], -pts_cam[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * x / d + K[0, 2]
        v = K[1, 1] * y / d + K[1, 2]
    pix = np.column_stack([u, v])
    # Pixel centers are at integer coords, so the image spans [-0.5, size-0.5).
    valid = (d > 0) & (u > -0.5) & (u < wh[0] - 0.5) & (v > -0.5) & (v < wh[1] - 0.5)
    return pix, valid
