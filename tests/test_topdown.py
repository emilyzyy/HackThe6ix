import cv2
import numpy as np
import pytest

from plane import compute_table_frame
from session_io import SessionReader
from topdown import plane_homography, sharpness, export_topdown
from transforms import intrinsics_to_K, project_points
from tests.synthetic import make_synthetic_session, sample_texture


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    out = tmp_path_factory.mktemp("topdown") / "session"
    make_synthetic_session(out, n_frames=16, seed=11)
    compute_table_frame(out)
    return out


def test_plane_homography_matches_project_points(session):
    rec = SessionReader(session).frames()[3]
    K = intrinsics_to_K(**rec.intrinsics)
    from plane import load_table_frame
    tf = load_table_frame(session)
    W2T = tf["world_to_table"]
    px_per_m = 2000.0
    origin_xy = (-0.1, -0.1)
    H = plane_homography(K, rec.pose_mat, W2T, px_per_m, origin_xy)

    T2W = np.linalg.inv(W2T)
    for out_uv in [(0, 0), (100, 40), (37, 211), (400, 400)]:
        x = origin_xy[0] + out_uv[0] / px_per_m
        y = origin_xy[1] + out_uv[1] / px_per_m
        world = (T2W @ [x, y, 0.0, 1.0])[:3]
        pix, _ = project_points(world[None], K, rec.pose_mat, (10**9, 10**9))
        got = H @ [out_uv[0], out_uv[1], 1.0]
        got = got[:2] / got[2]
        np.testing.assert_allclose(got, pix[0], atol=1e-6)


def test_sharpness_prefers_sharp_images():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 5)
    assert sharpness(sharp) > sharpness(blurred) * 5


def _truth_correlation(out_path, meta, session_dir):
    """Correlate exported image against the ground-truth texture.

    The synthetic texture is defined in WORLD xy; export pixels are in TABLE
    xy (origin at the observed centroid), so map table -> world first.
    """
    from plane import load_table_frame
    T2W = np.linalg.inv(load_table_frame(session_dir)["world_to_table"])
    img = cv2.imread(str(out_path))
    h, w = img.shape[:2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    x = meta["origin_xy"][0] + (us.ravel() + 0.5) / meta["px_per_m"]
    y = meta["origin_xy"][1] + (vs.ravel() + 0.5) / meta["px_per_m"]
    table_pts = np.column_stack([x, y, np.zeros_like(x), np.ones_like(x)])
    world_xy = (T2W @ table_pts.T).T[:, :2]
    truth = sample_texture(world_xy)  # RGB
    got = img.reshape(-1, 3)[:, ::-1].astype(float)  # BGR -> RGB
    filled = got.sum(axis=1) > 0
    corr = np.corrcoef(got[filled, 1], truth[filled, 1].astype(float))[0, 1]
    checker_truth = truth[filled, 0] > 128
    checker_got = got[filled, 0] > 128
    return corr, (checker_truth == checker_got).mean(), filled.mean()


def test_ortho_export_reproduces_texture(session):
    out, meta = export_topdown(session, mode="ortho", px_per_mm=1.0)
    assert out.exists()
    corr, checker_acc, filled = _truth_correlation(out, meta, session)
    assert corr > 0.9          # smooth gradient channel
    assert checker_acc > 0.8   # checker channel
    assert filled > 0.5


def test_best_frame_export(session):
    out, meta = export_topdown(session, mode="best-frame", px_per_mm=1.0)
    assert out.exists()
    assert meta["mode"] == "best-frame"
    assert meta["best_frame_id"] is not None
    corr, checker_acc, _ = _truth_correlation(out, meta, session)
    assert corr > 0.85
    assert checker_acc > 0.75
