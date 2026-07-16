import numpy as np
import pytest

from plane import fit_table_plane, table_frame_from_plane
from session_io import SessionReader
from transforms import unproject_depth, scale_intrinsics, intrinsics_to_K
from tests.synthetic import make_synthetic_session, look_at_pose


@pytest.fixture(scope="module")
def synthetic_session(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth") / "session"
    return make_synthetic_session(out, n_frames=12, seed=3)


def test_look_at_pose_looks_at_target():
    T = look_at_pose(pos=[0.1, 0.2, 0.5], target=[0.0, 0.0, 0.0])
    fwd = T[:3, :3] @ [0, 0, -1]  # camera looks down its -Z
    expected = np.array([-0.1, -0.2, -0.5])
    np.testing.assert_allclose(fwd, expected / np.linalg.norm(expected), atol=1e-9)
    # Rotation is orthonormal.
    np.testing.assert_allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-9)


def test_synthetic_depth_is_self_consistent(synthetic_session):
    """Unprojected synthetic depth must land exactly on the ground-truth plane."""
    rec = SessionReader(synthetic_session).frames()[0]
    K_rgb = intrinsics_to_K(**rec.intrinsics)
    K_d = scale_intrinsics(K_rgb, (rec.rgb_size[1], rec.rgb_size[0]),
                           (rec.depth_size[1], rec.depth_size[0]))
    pts = unproject_depth(rec.load_depth(), K_d, rec.pose_mat)
    assert np.abs(pts[:, 2]).max() < 1e-6  # table at world z=0


def test_table_frame_flat_plane():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    W2T = table_frame_from_plane(normal=[0, 0, 1], d=0.0, points=pts)
    h = np.column_stack([pts, np.ones(3)])
    out = (W2T @ h.T).T[:, :3]
    np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-12)


def test_table_frame_tilted_plane():
    # Plane x + z = 1 (normal (1,0,1)/sqrt2, point (1,0,0)).
    n = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    d = -float(n @ [1.0, 0.0, 0.0])  # open3d convention n·p + d = 0
    rng = np.random.default_rng(0)
    uv = rng.normal(size=(50, 2))
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(n, e1)
    pts = np.array([1.0, 0.0, 0.0]) + uv[:, :1] * e1 + uv[:, 1:] * e2
    W2T = table_frame_from_plane(n, d, pts)
    h = np.column_stack([pts, np.ones(len(pts))])
    out = (W2T @ h.T).T
    np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-9)
    # Origin is the centroid: mean xy is ~0.
    np.testing.assert_allclose(out[:, :2].mean(axis=0), 0.0, atol=1e-9)


def test_fit_table_plane_on_synthetic(synthetic_session):
    reader = SessionReader(synthetic_session)
    normal, d, inliers, stats = fit_table_plane(reader)
    # Ground truth: z=0 plane, normal up.
    assert np.dot(normal, [0, 0, 1]) > np.cos(np.deg2rad(0.5))
    # Plane offset: distance of origin from plane is |d| (unit normal).
    assert abs(d) < 0.005
    assert stats["inlier_fraction"] > 0.9
    W2T = table_frame_from_plane(normal, d, inliers)
    h = np.column_stack([inliers, np.ones(len(inliers))])
    z = (W2T @ h.T).T[:, 2]
    assert np.abs(z).max() < 0.01
