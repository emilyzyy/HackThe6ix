"""Phase 3: live/replay coverage display.

Aerial-map view of the table grid: unseen cells gray, seen cells filled with
RGB projected onto the plane (latest observation wins — detail comes from the
high-res RGB, not depth). Shows % bounded-workspace coverage and a
COVERAGE COMPLETE banner.

Replay:  python viewer.py sessions/<ts> [--fps 2] [--cell 0.005]
Live:    python capture.py --live-view   (uses LiveCoverageView below)
"""

import argparse
import time

import cv2
import numpy as np

from coverage import CoverageGrid
from transforms import project_points

UNSEEN_GRAY = 128
MIN_WINDOW_H = 480
# "Complete" is only meaningful once a real workspace has been swept.
MIN_COMPLETE_FRAMES = 20
MIN_WORKSPACE_AREA_M2 = 0.04  # 20 cm x 20 cm


def completion_allowed(grid, frames_processed, min_seen=2):
    return (frames_processed >= MIN_COMPLETE_FRAMES
            and grid.workspace_area_m2(min_seen) >= MIN_WORKSPACE_AREA_M2)


class CoverageView:
    """Keeps an RGB canvas aligned to a CoverageGrid (one pixel per cell)."""

    def __init__(self, grid, world_to_table):
        self.grid = grid
        self.world_to_table = world_to_table
        self.table_to_world = np.linalg.inv(world_to_table)
        self.canvas = np.full((grid.height, grid.width, 3), UNSEEN_GRAY,
                              dtype=np.uint8)
        # Precomputed world positions of all cell centers (on the plane).
        (x0, _), (y0, _) = grid.bounds
        ys, xs = np.mgrid[0:grid.height, 0:grid.width]
        cx = (xs.ravel() + 0.5) * grid.cell_m + x0
        cy = (ys.ravel() + 0.5) * grid.cell_m + y0
        pts_t = np.column_stack([cx, cy, np.zeros_like(cx),
                                 np.ones_like(cx)])
        self._centers_world = (self.table_to_world @ pts_t.T).T[:, :3]
        self._cell_ix = np.column_stack([ys.ravel(), xs.ravel()])

    def update(self, rgb_bgr, K_rgb, cam_to_world, rgb_wh):
        """Paint cells visible in this frame with its projected RGB."""
        observed = self.grid.observed_mask().ravel()
        if not observed.any():
            return
        pix, valid = project_points(self._centers_world[observed], K_rgb,
                                    cam_to_world, rgb_wh)
        sel = np.nonzero(observed)[0][valid]
        if len(sel) == 0:
            return
        uv = np.rint(pix[valid]).astype(int)
        u = np.clip(uv[:, 0], 0, rgb_wh[0] - 1)
        v = np.clip(uv[:, 1], 0, rgb_wh[1] - 1)
        iy, ix = self._cell_ix[sel, 0], self._cell_ix[sel, 1]
        self.canvas[iy, ix] = rgb_bgr[v, u]

    def render(self, min_seen=2, complete_threshold=0.95, show_complete=True):
        """BGR frame for display: canvas + coverage overlay (table y up).

        show_complete=False suppresses the COMPLETE banner — callers gate it
        on minimum frames/workspace area, since a barely-swept region
        trivially covers itself.
        """
        img = self.canvas.copy()
        img[~self.grid.observed_mask()] = UNSEEN_GRAY
        img = cv2.flip(img, 0)  # table +y up on screen
        if img.shape[0] < MIN_WINDOW_H:
            s = int(np.ceil(MIN_WINDOW_H / img.shape[0]))
            img = cv2.resize(img, None, fx=s, fy=s,
                             interpolation=cv2.INTER_NEAREST)
        frac = self.grid.workspace_coverage_fraction(min_seen)
        label = f"workspace coverage {100 * frac:.0f}%  (cells seen>={min_seen})"
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 1, cv2.LINE_AA)
        if show_complete and frac >= complete_threshold:
            cv2.putText(img, "COVERAGE COMPLETE", (10, 64),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2,
                        cv2.LINE_AA)
        return img


class LiveCoverageView:
    """Hooks a CaptureController: fits the plane once enough frames exist,
    then renders coverage on the main thread at <=2 Hz. Never blocks capture.
    """

    def __init__(self, controller, min_frames_for_plane=20, cell_m=0.005,
                 complete_threshold=0.95):
        self.controller = controller
        self.min_frames = min_frames_for_plane
        self.cell_m = cell_m
        self.threshold = complete_threshold
        self._pending = []
        self._grid = None
        self._view = None
        self._tf = None
        self._announced = False
        controller.add_frame_listener(self._on_frame)

    def _on_frame(self, rec, snap):
        # Writer thread: just queue lightweight refs; all math on main thread.
        self._pending.append((rec, snap))

    def _try_init(self):
        from plane import compute_table_frame
        if self.controller.writer.frame_count < self.min_frames:
            return False
        tf = compute_table_frame(self.controller.session_dir)
        tf["world_to_table"] = np.array(tf["world_to_table"])
        self._tf = tf
        self._grid = CoverageGrid(tf["extent_xy"], cell_m=self.cell_m)
        self._view = CoverageView(self._grid, tf["world_to_table"])
        return True

    def _process_pending(self):
        from transforms import intrinsics_to_K, scale_intrinsics
        pending, self._pending = self._pending, []
        for rec, snap in pending:
            K_rgb = intrinsics_to_K(**rec.intrinsics)
            K_d = scale_intrinsics(K_rgb, (rec.rgb_size[1], rec.rgb_size[0]),
                                   (rec.depth_size[1], rec.depth_size[0]))
            self._grid.mark_frame(snap["depth"], K_d, rec.pose_mat,
                                  self._tf["world_to_table"], stride=2)
            self._view.update(snap["rgb"][:, :, ::-1], K_rgb, rec.pose_mat,
                              (rec.rgb_size[1], rec.rgb_size[0]))

    def run_until(self, stopped_event):
        """Main-thread render loop; returns on q keypress or stream stop."""
        print("Live view: press q in the window to stop capture.")
        while not stopped_event.is_set():
            if self._grid is None:
                if not self._try_init():
                    if cv2.waitKey(250) & 0xFF == ord("q"):
                        return
                    continue
                print(f"\ntable plane locked "
                      f"({self._tf['fit_stats']['inliers']} inliers)")
            self._process_pending()
            allowed = completion_allowed(
                self._grid, self.controller.writer.frame_count)
            img = self._view.render(complete_threshold=self.threshold,
                                    show_complete=allowed)
            frac = self._grid.workspace_coverage_fraction(2)
            if allowed and frac >= self.threshold and not self._announced:
                self._announced = True
                print("\ncoverage complete")
            cv2.imshow("coverage", img)
            if cv2.waitKey(500) & 0xFF == ord("q"):
                return


def main():
    ap = argparse.ArgumentParser(description="Replay a session's coverage.")
    ap.add_argument("session")
    ap.add_argument("--fps", type=float, default=2, help="replay rate")
    ap.add_argument("--cell", type=float, default=0.005, help="cell size, m")
    ap.add_argument("--threshold", type=float, default=0.95)
    args = ap.parse_args()

    from plane import compute_table_frame, load_table_frame
    from session_io import SessionReader
    from transforms import intrinsics_to_K, scale_intrinsics

    try:
        tf = load_table_frame(args.session)
    except FileNotFoundError:
        tf = compute_table_frame(args.session)
        tf["world_to_table"] = np.array(tf["world_to_table"])

    grid = CoverageGrid(tf["extent_xy"], cell_m=args.cell)
    view = CoverageView(grid, tf["world_to_table"])
    announced = False
    for n, rec in enumerate(SessionReader(args.session).frames(), start=1):
        K_rgb = intrinsics_to_K(**rec.intrinsics)
        K_d = scale_intrinsics(K_rgb, (rec.rgb_size[1], rec.rgb_size[0]),
                               (rec.depth_size[1], rec.depth_size[0]))
        grid.mark_frame(rec.load_depth(), K_d, rec.pose_mat,
                        tf["world_to_table"])
        view.update(rec.load_rgb(), K_rgb, rec.pose_mat,
                    (rec.rgb_size[1], rec.rgb_size[0]))
        allowed = completion_allowed(grid, n)
        img = view.render(complete_threshold=args.threshold,
                          show_complete=allowed)
        frac = grid.workspace_coverage_fraction(2)
        if allowed and frac >= args.threshold and not announced:
            announced = True
            print("coverage complete")
        cv2.imshow("coverage replay", img)
        if cv2.waitKey(int(1000 / args.fps)) & 0xFF == ord("q"):
            break
    print(f"final workspace coverage: {100 * grid.workspace_coverage_fraction(2):.1f}%")
    cv2.imshow("coverage replay", view.render())
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
