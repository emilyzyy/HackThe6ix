import json

import cv2
import numpy as np
import pytest

from multiview import ViewCandidate, export_multiview, select_covering_views
from plane import compute_table_frame
from tests.synthetic import make_synthetic_session


def _candidate(frame_id, quality, valid):
    image = np.full((*valid.shape, 3), frame_id, dtype=np.uint8)
    return ViewCandidate(frame_id, quality, image, valid)


def test_selection_reaches_union_and_prefers_sharp_coverage_tie():
    left = np.zeros((2, 4), dtype=bool)
    left[:, :2] = True
    right = np.zeros((2, 4), dtype=bool)
    right[:, 2:] = True
    candidates = [
        _candidate(1, 1.0, left),
        _candidate(2, 2.0, left),
        _candidate(3, 1.0, right),
    ]

    selected = select_covering_views(candidates, max_frames=3)

    assert [view.frame_id for view in selected] == [2, 3]


@pytest.fixture
def session(tmp_path):
    out = tmp_path / "session"
    make_synthetic_session(out, n_frames=16, seed=21)
    compute_table_frame(out)
    return out


def test_export_writes_versioned_common_canvas_manifest(session):
    path, manifest = export_multiview(
        session, px_per_mm=1.0, max_frames=3, edge_margin_px=2
    )

    assert path.exists()
    assert json.loads(path.read_text()) == manifest
    assert manifest["version"] == 2
    assert 1 <= len(manifest["views"]) <= 3
    assert 0 < manifest["selected_coverage"] <= manifest["union_coverage"] <= 1
    assert all((session / view["image"]).exists() for view in manifest["views"])
    assert all((session / view["mask"]).exists() for view in manifest["views"])
    assert len({tuple(view["size_wh"]) for view in manifest["views"]}) == 1


def test_export_fills_pixels_outside_each_valid_footprint(session):
    _, manifest = export_multiview(
        session, px_per_mm=1.0, max_frames=2, edge_margin_px=2
    )
    view = manifest["views"][0]
    image = cv2.imread(str(session / view["image"]))
    valid = cv2.imread(str(session / view["mask"]), cv2.IMREAD_GRAYSCALE) > 0

    assert (~valid).any()
    assert float(image[~valid].mean()) > 40.0


def test_workspace_pad_is_symmetric():
    from topdown import WORKSPACE_CROP_PAD_M
    assert WORKSPACE_CROP_PAD_M == 0.025


def test_manifest_v2_carries_bridge_geometry(tmp_path):
    import json
    import numpy as np
    from multiview import export_multiview
    from plane import compute_table_frame
    from session_io import SessionReader
    from tests.synthetic import make_synthetic_session
    from topdown import plane_homography
    from transforms import intrinsics_to_K, project_points

    session = make_synthetic_session(tmp_path / "session", n_frames=10, seed=2)
    compute_table_frame(session)
    manifest_path, manifest = export_multiview(session, max_frames=3)
    assert manifest["version"] == 2

    frames = {rec.frame_id: rec for rec in SessionReader(session).frames()}
    from plane import load_table_frame
    tf = load_table_frame(session)
    W2T = tf["world_to_table"]
    T2W = np.linalg.inv(W2T)
    px_per_m = manifest["px_per_m"]
    origin_xy = manifest["origin_xy"]

    for view in manifest["views"]:
        rec = frames[view["frame_id"]]
        # Homography is exactly the one used to rectify this frame.
        K = intrinsics_to_K(**rec.intrinsics)
        expected_H = plane_homography(K, rec.pose_mat, W2T, px_per_m,
                                      tuple(origin_xy))
        np.testing.assert_allclose(np.array(view["homography"]), expected_H,
                                   rtol=1e-9)
        # Original frame reference is valid.
        assert (session / view["rgb"]).exists()
        assert view["rgb_size"] == rec.rgb_size
        # raise_uv_per_cm equals the actual projected displacement of a point
        # raised 1 cm above the plane at canvas center (guards the
        # frame-association trap: wrong frame's pose would not match).
        w, h = manifest["size_wh"]
        cx = origin_xy[0] + (w / 2) / px_per_m
        cy = origin_xy[1] + (h / 2) / px_per_m
        p0 = (T2W @ [cx, cy, 0.0, 1.0])[:3]
        p1 = (T2W @ [cx, cy, 0.01, 1.0])[:3]
        uv, valid = project_points(np.stack([p0, p1]), K, rec.pose_mat,
                                   (10**9, 10**9))
        assert valid.all()
        np.testing.assert_allclose(np.array(view["raise_uv_per_cm"]),
                                   uv[1] - uv[0], atol=1e-6)
