import numpy as np
import pytest

from coverage import CoverageGrid
from plane import compute_table_frame
from session_io import SessionReader
from transforms import intrinsics_to_K, scale_intrinsics
from viewer import CoverageView, UNSEEN_GRAY
from tests.synthetic import make_synthetic_session, sample_texture


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    out = tmp_path_factory.mktemp("viewer") / "session"
    make_synthetic_session(out, n_frames=12, seed=5)
    tf = compute_table_frame(out)
    tf["world_to_table"] = np.array(tf["world_to_table"])
    return out, tf


def _play_all(session_dir, tf, cell_m=0.01):
    grid = CoverageGrid(tf["extent_xy"], cell_m=cell_m)
    view = CoverageView(grid, tf["world_to_table"])
    for rec in SessionReader(session_dir).frames():
        K_rgb = intrinsics_to_K(**rec.intrinsics)
        K_d = scale_intrinsics(K_rgb, (rec.rgb_size[1], rec.rgb_size[0]),
                               (rec.depth_size[1], rec.depth_size[0]))
        grid.mark_frame(rec.load_depth(), K_d, rec.pose_mat,
                        tf["world_to_table"])
        view.update(rec.load_rgb(), K_rgb, rec.pose_mat,
                    (rec.rgb_size[1], rec.rgb_size[0]))
    return grid, view


def test_observed_cells_get_projected_rgb(session):
    session_dir, tf = session
    grid, view = _play_all(session_dir, tf)
    obs = grid.observed_mask()
    assert obs.any()
    # Ground truth: texture green channel encodes table x (G = 128 + 600x).
    ys, xs = np.nonzero(obs)
    (x0, _), (y0, _) = grid.bounds
    cell_x = (xs + 0.5) * grid.cell_m + x0
    cell_y = (ys + 0.5) * grid.cell_m + y0
    truth = sample_texture(np.column_stack([cell_x, cell_y]))
    canvas_rgb = view.canvas[ys, xs][:, ::-1]  # canvas is BGR
    err = np.abs(canvas_rgb.astype(int) - truth.astype(int))
    # JPEG + nearest-neighbor sampling on 2 cm checkers: allow some edge error.
    assert np.median(err[:, 1]) < 12   # smooth gradient channel: tight
    assert (err.max(axis=1) < 60).mean() > 0.8  # checker channels: majority


def test_unseen_cells_stay_gray(session):
    session_dir, tf = session
    grid, view = _play_all(session_dir, tf)
    unseen = ~grid.observed_mask()
    assert unseen.any()
    assert (view.canvas[unseen] == UNSEEN_GRAY).all()


def test_render_smoke_headless(session):
    session_dir, tf = session
    grid, view = _play_all(session_dir, tf)
    img = view.render(min_seen=1, complete_threshold=0.95)
    assert img.ndim == 3 and img.shape[2] == 3 and img.dtype == np.uint8
    assert img.shape[0] >= 300  # upscaled to a usable window size


def test_render_hides_banner_when_not_allowed(session):
    session_dir, tf = session
    grid, view = _play_all(session_dir, tf)
    complete = view.render(min_seen=1, complete_threshold=0.0,
                           show_complete=True)
    hidden = view.render(min_seen=1, complete_threshold=0.0,
                         show_complete=False)
    assert (complete != hidden).any()  # banner drawn only when allowed
