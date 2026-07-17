"""Phase 2: 2D coverage grid over the table plane.

Cells live in table-frame XY (see plane.py). Per cell we track how many
retained frames observed it (seen_count) and from which directions
(angle_bins: bitmask of 8 azimuth octants of the point->camera direction).
"Coverage complete" is measured against the convex hull of everything
observed so far, not the full grid bounds.

Usage: python coverage.py sessions/<ts> [--cell 0.003] [--min-seen 2]
"""

import argparse

import cv2
import numpy as np

from transforms import unproject_depth

N_ANGLE_BINS = 8


class CoverageGrid:
    def __init__(self, bounds_xy, cell_m=0.003):
        self.bounds = [list(map(float, b)) for b in bounds_xy]
        self.cell_m = float(cell_m)
        (x0, x1), (y0, y1) = self.bounds
        self.width = max(1, int(np.ceil((x1 - x0) / cell_m)))
        self.height = max(1, int(np.ceil((y1 - y0) / cell_m)))
        self.seen_count = np.zeros((self.height, self.width), dtype=np.uint16)
        self.angle_bins = np.zeros((self.height, self.width), dtype=np.uint8)

    def _cells_for_points(self, xy):
        (x0, _), (y0, _) = self.bounds
        ix = np.floor((xy[:, 0] - x0) / self.cell_m).astype(int)
        iy = np.floor((xy[:, 1] - y0) / self.cell_m).astype(int)
        ok = (ix >= 0) & (ix < self.width) & (iy >= 0) & (iy < self.height)
        return ix[ok], iy[ok], ok

    def mark_frame(self, depth, K_depth, cam_to_world, world_to_table,
                   z_band=0.02, stride=1, max_depth=1.5):
        """Unproject one depth frame onto the table and mark observed cells."""
        pts = unproject_depth(np.nan_to_num(depth), K_depth, cam_to_world,
                              stride=stride, max_depth=max_depth)
        if len(pts) == 0:
            return 0
        h = np.column_stack([pts, np.ones(len(pts))])
        pts_t = (world_to_table @ h.T).T[:, :3]
        band = np.abs(pts_t[:, 2]) < z_band
        xy = pts_t[band, :2]
        if len(xy) == 0:
            return 0
        ix, iy, ok = self._cells_for_points(xy)
        if len(ix) == 0:
            return 0

        # Count each cell once per frame regardless of point density.
        flat = np.unique(iy * self.width + ix)
        uy, ux = flat // self.width, flat % self.width
        self.seen_count[uy, ux] += 1

        # Azimuth of the point->camera direction, in table XY.
        cam_t = (world_to_table @ np.append(cam_to_world[:3, 3], 1.0))[:3]
        cell_xy = np.column_stack([(ux + 0.5) * self.cell_m + self.bounds[0][0],
                                   (uy + 0.5) * self.cell_m + self.bounds[1][0]])
        az = np.arctan2(cam_t[1] - cell_xy[:, 1], cam_t[0] - cell_xy[:, 0])
        bins = (np.floor((az + np.pi) / (2 * np.pi / N_ANGLE_BINS))
                .astype(int) % N_ANGLE_BINS)
        np.bitwise_or.at(self.angle_bins, (uy, ux),
                         (1 << bins).astype(np.uint8))
        return len(flat)

    def observed_mask(self, min_seen=1):
        return self.seen_count >= min_seen

    def hull_mask(self, min_seen=1):
        """Cells inside the convex hull of everything observed so far."""
        obs = self.observed_mask(min_seen)
        if not obs.any():
            return np.zeros_like(obs)
        ys, xs = np.nonzero(obs)
        hull = cv2.convexHull(np.column_stack([xs, ys]).astype(np.int32))
        mask = np.zeros(obs.shape, dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)
        return mask.astype(bool)

    def hull_area_m2(self, min_seen=1):
        return float(self.hull_mask(min_seen).sum()) * self.cell_m ** 2

    def hull_coverage_fraction(self, min_seen=1):
        hull = self.hull_mask(min_seen)
        if not hull.any():
            return 0.0
        return float(self.observed_mask(min_seen)[hull].mean())

    def save(self, path):
        np.savez_compressed(path, seen_count=self.seen_count,
                            angle_bins=self.angle_bins, cell_m=self.cell_m,
                            bounds=np.array(self.bounds))

    @classmethod
    def load(cls, path):
        z = np.load(path)
        grid = cls(z["bounds"].tolist(), cell_m=float(z["cell_m"]))
        grid.seen_count = z["seen_count"]
        grid.angle_bins = z["angle_bins"]
        return grid


def replay_session(session_dir, cell_m=0.003, stride=2):
    """Build a coverage grid by replaying a recorded session."""
    from plane import load_table_frame, compute_table_frame
    from session_io import SessionReader
    from transforms import intrinsics_to_K, scale_intrinsics

    try:
        tf = load_table_frame(session_dir)
    except FileNotFoundError:
        tf = compute_table_frame(session_dir)
        tf["world_to_table"] = np.array(tf["world_to_table"])
    grid = CoverageGrid(tf["extent_xy"], cell_m=cell_m)
    for rec in SessionReader(session_dir).frames():
        K_rgb = intrinsics_to_K(**rec.intrinsics)
        K_d = scale_intrinsics(K_rgb, (rec.rgb_size[1], rec.rgb_size[0]),
                               (rec.depth_size[1], rec.depth_size[0]))
        grid.mark_frame(rec.load_depth(), K_d, rec.pose_mat,
                        tf["world_to_table"], stride=stride)
    return grid, tf


def main():
    ap = argparse.ArgumentParser(description="Replay a session into a coverage grid.")
    ap.add_argument("session")
    ap.add_argument("--cell", type=float, default=0.003, help="cell size in m")
    ap.add_argument("--min-seen", type=int, default=2)
    args = ap.parse_args()

    grid, _ = replay_session(args.session, cell_m=args.cell)
    out = f"{args.session}/coverage.npz"
    grid.save(out)
    frac = grid.hull_coverage_fraction(min_seen=args.min_seen)
    print(f"grid {grid.width}x{grid.height} cells ({args.cell * 1000:.0f} mm)")
    print(f"observed cells: {int(grid.observed_mask(args.min_seen).sum())}")
    print(f"hull coverage: {100 * frac:.1f}% (min_seen={args.min_seen})")
    print("wrote", out)


if __name__ == "__main__":
    main()
