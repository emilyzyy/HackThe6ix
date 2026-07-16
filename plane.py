"""Phase 2: detect the table plane and establish the table-local frame.

Plane convention: unit normal n and offset d with n·p + d = 0 (open3d's),
n flipped to point up toward the cameras. The table frame maps world points
so the table lies at z=0 with the origin at the observed centroid.

Usage: python plane.py sessions/<ts> [--stride 4] [--frames 12]
Writes sessions/<ts>/table_frame.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from session_io import SessionReader
from transforms import intrinsics_to_K, scale_intrinsics, unproject_depth


def fit_table_plane(reader, stride=4, n_frames=12, distance_threshold=0.008,
                    voxel=0.005, max_depth=1.5):
    """Pooled RANSAC plane fit over evenly spaced frames.

    Returns (normal, d, inlier_points, stats): unit normal pointing toward the
    cameras (up), n·p + d = 0.
    """
    import open3d as o3d  # local import: heavy, and only this step needs it

    frames = reader.frames()
    if not frames:
        raise ValueError(f"no frames in {reader.session_dir}")
    picks = frames[:: max(1, len(frames) // n_frames)][:n_frames]

    clouds = []
    for rec in picks:
        K_rgb = intrinsics_to_K(**rec.intrinsics)
        K_d = scale_intrinsics(K_rgb, (rec.rgb_size[1], rec.rgb_size[0]),
                               (rec.depth_size[1], rec.depth_size[0]))
        clouds.append(unproject_depth(rec.load_depth(), K_d, rec.pose_mat,
                                      stride=stride, max_depth=max_depth))
    pts = np.concatenate(clouds)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd = pcd.voxel_down_sample(voxel)
    (a, b, c, d), inlier_idx = pcd.segment_plane(
        distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000)
    normal = np.array([a, b, c])
    nrm = np.linalg.norm(normal)
    normal, d = normal / nrm, d / nrm

    # Point the normal at the cameras (i.e. up from the table).
    mean_cam = np.mean([rec.pose_mat[:3, 3] for rec in picks], axis=0)
    if normal @ mean_cam + d < 0:
        normal, d = -normal, -d

    down = np.asarray(pcd.points)
    inliers = down[inlier_idx]
    stats = {"points": len(down), "inliers": len(inlier_idx),
             "inlier_fraction": len(inlier_idx) / len(down),
             "frames_used": len(picks)}
    return normal, float(d), inliers, stats


def table_frame_from_plane(normal, d, points):
    """world->table 4x4: table plane becomes z=0, origin at points' centroid."""
    n = np.asarray(normal, dtype=float)
    nrm = np.linalg.norm(n)
    n, d = n / nrm, d / nrm
    centroid = np.asarray(points, dtype=float).mean(axis=0)
    origin = centroid - (n @ centroid + d) * n

    ref = np.array([1.0, 0.0, 0.0])
    if abs(n @ ref) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    x = ref - (ref @ n) * n
    x /= np.linalg.norm(x)
    y = np.cross(n, x)

    R = np.vstack([x, y, n])  # rows: table axes in world coords
    W2T = np.eye(4)
    W2T[:3, :3] = R
    W2T[:3, 3] = -R @ origin
    return W2T


def observed_extent(world_to_table, points, pad=0.02):
    """[[xmin, xmax], [ymin, ymax]] of points in table coords, padded."""
    h = np.column_stack([points, np.ones(len(points))])
    xy = (world_to_table @ h.T).T[:, :2]
    lo = np.percentile(xy, 1, axis=0) - pad
    hi = np.percentile(xy, 99, axis=0) + pad
    return [[float(lo[0]), float(hi[0])], [float(lo[1]), float(hi[1])]]


def compute_table_frame(session_dir, **fit_kwargs):
    """Fit plane + build frame + write table_frame.json. Returns the dict."""
    reader = SessionReader(session_dir)
    normal, d, inliers, stats = fit_table_plane(reader, **fit_kwargs)
    W2T = table_frame_from_plane(normal, d, inliers)
    out = {
        "world_to_table": W2T.tolist(),
        "plane": {"normal": normal.tolist(), "d": d},
        "extent_xy": observed_extent(W2T, inliers),
        "fit_stats": stats,
    }
    with open(Path(session_dir) / "table_frame.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


def load_table_frame(session_dir):
    with open(Path(session_dir) / "table_frame.json") as f:
        d = json.load(f)
    d["world_to_table"] = np.array(d["world_to_table"])
    return d


def main():
    ap = argparse.ArgumentParser(description="Fit the table plane of a session.")
    ap.add_argument("session")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()
    out = compute_table_frame(args.session, stride=args.stride,
                              n_frames=args.frames)
    s = out["fit_stats"]
    ext = out["extent_xy"]
    print(f"plane normal {np.round(out['plane']['normal'], 4).tolist()} "
          f"d={out['plane']['d']:.4f}")
    print(f"inliers {s['inliers']}/{s['points']} "
          f"({100 * s['inlier_fraction']:.0f}%) from {s['frames_used']} frames")
    print(f"extent x [{ext[0][0]:.3f}, {ext[0][1]:.3f}] "
          f"y [{ext[1][0]:.3f}, {ext[1][1]:.3f}] m")
    print("wrote", Path(args.session) / "table_frame.json")


if __name__ == "__main__":
    main()
