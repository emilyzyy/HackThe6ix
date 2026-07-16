import numpy as np
import pytest

from transforms import (
    quat_to_mat,
    pose_to_mat,
    intrinsics_to_K,
    scale_intrinsics,
    unproject_depth,
    project_points,
)


def test_identity_quaternion():
    R = quat_to_mat(0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-12)


def test_quaternion_90deg_about_z():
    # 90° about +Z: (x,y,z,w) = (0, 0, sin45, cos45); maps +X to +Y.
    s = np.sqrt(0.5)
    R = quat_to_mat(0.0, 0.0, s, s)
    np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_quaternion_is_normalized_before_use():
    s = np.sqrt(0.5)
    R1 = quat_to_mat(0.0, 0.0, s, s)
    R2 = quat_to_mat(0.0, 0.0, 2 * s, 2 * s)  # same rotation, unnormalized
    np.testing.assert_allclose(R1, R2, atol=1e-12)


def test_pose_to_mat_layout():
    T = pose_to_mat(0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3)
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(T[3], [0, 0, 0, 1])


def test_intrinsics_to_K():
    K = intrinsics_to_K(700.0, 710.0, 360.0, 480.0)
    np.testing.assert_allclose(
        K, [[700.0, 0, 360.0], [0, 710.0, 480.0], [0, 0, 1]]
    )


def test_scale_intrinsics_half_resolution():
    K = intrinsics_to_K(700.0, 710.0, 360.0, 480.0)
    K2 = scale_intrinsics(K, from_wh=(720, 960), to_wh=(360, 480))
    np.testing.assert_allclose(K2[0, 0], 350.0)
    np.testing.assert_allclose(K2[1, 1], 355.0)
    np.testing.assert_allclose(K2[0, 2], 180.0)
    np.testing.assert_allclose(K2[1, 2], 240.0)
    np.testing.assert_allclose(K2[2], [0, 0, 1])


def _flat_depth_setup():
    """Camera at world origin, identity orientation, flat 1 m depth plane."""
    h, w = 6, 8
    depth = np.full((h, w), 1.0, dtype=np.float32)
    K = intrinsics_to_K(100.0, 100.0, w / 2, h / 2)
    cam_to_world = np.eye(4)
    return depth, K, cam_to_world, (w, h)


def test_unproject_depth_center_pixel_on_optical_axis():
    depth, K, T, _ = _flat_depth_setup()
    pts = unproject_depth(depth, K, T)
    assert pts.shape == (depth.size, 3)
    # Principal point at pixel (4, 3): camera looks down -Z (ARKit), so the
    # point at depth 1 in front of the camera is at world z = -1.
    idx = 3 * 8 + 4
    np.testing.assert_allclose(pts[idx], [0, 0, -1.0], atol=1e-9)


def test_unproject_depth_image_down_is_world_negative_y():
    # ARKit camera: +Y is up in camera space, image v grows downward.
    depth, K, T, _ = _flat_depth_setup()
    pts = unproject_depth(depth, K, T)
    top = pts[0 * 8 + 4]     # v=0 (top of image)
    bottom = pts[5 * 8 + 4]  # v=5 (bottom)
    assert top[1] > 0 > bottom[1]


def test_unproject_respects_stride_and_max_depth():
    depth, K, T, _ = _flat_depth_setup()
    depth[0, 0] = 5.0
    pts = unproject_depth(depth, K, T, stride=2, max_depth=2.0)
    assert pts.shape[0] == (3 * 4) - 1  # strided grid minus the too-deep pixel


def test_unproject_skips_nonpositive_depth():
    depth, K, T, _ = _flat_depth_setup()
    depth[2, 2] = 0.0
    pts = unproject_depth(depth, K, T)
    assert pts.shape[0] == depth.size - 1


def test_project_unproject_round_trip():
    depth, K, T, wh = _flat_depth_setup()
    # A camera moved and rotated in the world: round trip must still hold.
    s = np.sqrt(0.5)
    T = pose_to_mat(0.0, s, 0.0, s, 0.5, -0.2, 1.0)
    pts = unproject_depth(depth, K, T)
    pix, valid = project_points(pts, K, T, wh)
    assert valid.all()
    u = np.tile(np.arange(8), 6).astype(float)
    v = np.repeat(np.arange(6), 8).astype(float)
    np.testing.assert_allclose(pix[:, 0], u, atol=1e-6)
    np.testing.assert_allclose(pix[:, 1], v, atol=1e-6)


def test_project_points_behind_camera_invalid():
    K = intrinsics_to_K(100.0, 100.0, 4.0, 3.0)
    T = np.eye(4)  # camera at origin looking down -Z
    pts = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, +1.0]])  # in front / behind
    pix, valid = project_points(pts, K, T, (8, 6))
    assert valid.tolist() == [True, False]


def test_project_points_outside_image_invalid():
    K = intrinsics_to_K(100.0, 100.0, 4.0, 3.0)
    T = np.eye(4)
    pts = np.array([[10.0, 0.0, -1.0]])  # projects far off-image
    _, valid = project_points(pts, K, T, (8, 6))
    assert not valid[0]


def test_mat_to_quat_round_trip():
    from transforms import mat_to_quat
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        R = quat_to_mat(*q)
        q2 = mat_to_quat(R)
        R2 = quat_to_mat(*q2)
        np.testing.assert_allclose(R2, R, atol=1e-9)
