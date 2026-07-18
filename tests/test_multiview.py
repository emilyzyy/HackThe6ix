import json

import cv2
import numpy as np
import pytest

import multiview
from multiview import ViewCandidate, export_multiview, select_covering_views
from plane import compute_table_frame
from tests.synthetic import make_synthetic_session


def _candidate(
    frame_id, quality, valid, *, bearing=None, ray=None,
    sharpness_score=None
):
    image = np.full((*valid.shape, 3), frame_id, dtype=np.uint8)
    kwargs = {}
    if ray is not None:
        kwargs["viewing_ray_table_xyz"] = ray
    return ViewCandidate(
        frame_id,
        quality,
        image,
        valid,
        camera_bearing_deg=bearing,
        sharpness_score=sharpness_score,
        **kwargs,
    )


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


def test_selection_adds_time_diverse_confirmation_views_after_full_coverage():
    full = np.ones((3, 4), dtype=bool)
    candidates = [
        _candidate(0, 1.0, full),
        _candidate(10, 1.0, full),
        _candidate(20, 10.0, full),
        _candidate(30, 1.0, full),
        _candidate(40, 1.0, full),
    ]

    selected = select_covering_views(
        candidates, max_frames=5, min_frames=3
    )

    assert len(selected) == 3
    assert selected[0].frame_id == 20
    assert max(view.frame_id for view in selected) - min(
        view.frame_id for view in selected
    ) == 40


def test_confirmation_selection_falls_back_to_time_when_rays_unavailable():
    full = np.ones((3, 4), dtype=bool)
    candidates = [
        _candidate(0, 10.0, full, bearing=5.0, sharpness_score=10.0),
        _candidate(10, 8.0, full, bearing=12.0, sharpness_score=8.0),
        _candidate(20, 8.0, full, bearing=100.0, sharpness_score=8.0),
    ]

    selected = select_covering_views(
        candidates, max_frames=2, min_frames=2
    )

    assert [item.frame_id for item in selected] == [0, 20]


def test_circular_bearing_separation_wraps_at_360_degrees():
    from multiview import circular_separation_deg

    assert circular_separation_deg(355.0, 5.0) == pytest.approx(10.0)
    assert circular_separation_deg(10.0, 190.0) == pytest.approx(180.0)


def test_view_ray_separation_detects_obliquity_not_opposite_bearing():
    from multiview import view_ray_separation_deg

    near_top_a = (0.01, 0.0, -0.99995)
    near_top_b = (-0.01, 0.0, -0.99995)
    oblique = (0.70, 0.0, -0.714)

    assert view_ray_separation_deg(near_top_a, near_top_b) < 2.0
    assert view_ray_separation_deg(near_top_a, oblique) > 40.0


def test_confirmation_selection_maximizes_minimum_ray_separation():
    full = np.ones((3, 4), dtype=bool)
    candidates = [
        _candidate(0, 10.0, full, ray=(0.0, 0.0, -1.0)),
        _candidate(10, 8.0, full, ray=(0.1, 0.0, -0.995)),
        _candidate(20, 8.0, full, ray=(0.7, 0.0, -0.714)),
    ]

    selected = select_covering_views(
        candidates, max_frames=2, min_frames=2
    )

    assert [item.frame_id for item in selected] == [0, 20]


def test_selection_reserves_last_two_slots_for_oblique_confirmation():
    def mask(stop):
        value = np.zeros((1, 100), dtype=bool)
        value[:, :stop] = True
        return value

    candidates = [
        _candidate(1, 10.0, mask(70), ray=(0.0, 0.0, -1.0)),
        _candidate(2, 9.0, np.roll(mask(20), 70, axis=1),
                   ray=(0.05, 0.0, -0.999)),
        _candidate(3, 8.0, np.roll(mask(7), 90, axis=1),
                   ray=(-0.05, 0.0, -0.999)),
        _candidate(4, 7.0, np.roll(mask(1), 97, axis=1),
                   ray=(0.1, 0.0, -0.995)),
        _candidate(5, 7.0, np.roll(mask(1), 98, axis=1),
                   ray=(0.2, 0.0, -0.98)),
        _candidate(6, 7.0, np.roll(mask(1), 99, axis=1),
                   ray=(0.3, 0.0, -0.954)),
        _candidate(20, 8.0, mask(70), ray=(0.7, 0.0, -0.714)),
        _candidate(30, 8.0, mask(70), ray=(-0.7, 0.0, -0.714)),
    ]

    selected = select_covering_views(
        candidates, max_frames=5, min_frames=3
    )

    assert [item.frame_id for item in selected[:3]] == [1, 2, 3]
    assert {item.frame_id for item in selected[3:]} == {20, 30}
    assert all(item.selection_role == "confirmation"
               for item in selected[3:])


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
    assert manifest["version"] == 3
    assert len(manifest["views"]) == 3
    assert manifest["selection"]["min_confirmation_views"] == 3
    assert 0 < manifest["selected_coverage"] <= manifest["union_coverage"] <= 1
    assert all((session / view["image"]).exists() for view in manifest["views"])
    assert all((session / view["mask"]).exists() for view in manifest["views"])
    assert len({tuple(view["size_wh"]) for view in manifest["views"]}) == 1
    assert manifest["selection"]["strategy"] == (
        "coverage_then_view_ray_diverse_quality"
    )
    assert "pairwise_view_ray_separation" in manifest["selection"]
    assert all(view["topdownness"] is not None for view in manifest["views"])
    assert all(view["sharpness"] is not None for view in manifest["views"])
    assert all(
        len(view["camera_position_table_xyz"]) == 3
        for view in manifest["views"]
    )
    assert all(
        0.0 <= view["camera_bearing_deg"] < 360.0
        for view in manifest["views"]
    )
    assert all(
        len(view["viewing_ray_table_xyz"]) == 3
        for view in manifest["views"]
    )
    assert all(0.0 <= view["view_tilt_deg"] < 90.0
               for view in manifest["views"])
    assert all(view["selection_role"] in {"coverage", "confirmation"}
               for view in manifest["views"])
    assert "coverage_complete" in manifest["selection"]
    assert "coverage_insufficient_reason" in manifest["selection"]


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


def test_manifest_v3_carries_physical_and_dense_workspace(session):
    _, manifest = export_multiview(session, max_frames=2)

    assert manifest["version"] == 3
    geometry = manifest["workspace_geometry"]
    assert geometry["hard_source"] == "depth_observed_plane_component"
    assert geometry["hard_contours_table_xy"]
    assert geometry["dense_bounds_table_xy"] is not None
    assert geometry["table_extent_xy"]


def test_export_canvas_uses_admissible_bounds(session):
    from multiview import _canvas_geometry

    table, grid, origin, size = _canvas_geometry(session, 1000.0)
    hard = grid.admissible_workspace_bounds_xy()
    expected = (
        max(table["extent_xy"][0][0], hard[0][0]),
        min(table["extent_xy"][0][1], hard[0][1]),
        max(table["extent_xy"][1][0], hard[1][0]),
        min(table["extent_xy"][1][1], hard[1][1]),
    )

    assert origin == pytest.approx((expected[0], expected[2]))
    assert size == (
        int(np.ceil((expected[1] - expected[0]) * 1000.0)),
        int(np.ceil((expected[3] - expected[2]) * 1000.0)),
    )


def test_export_rectifies_candidates_coarsely_and_selected_views_fully(
    session, monkeypatch
):
    calls = []
    real_rectify = multiview._rectify

    def recording_rectify(record, table, origin, scale, size):
        calls.append((record.frame_id, scale, size))
        return real_rectify(record, table, origin, scale, size)

    monkeypatch.setattr(multiview, "_rectify", recording_rectify)

    _, manifest = export_multiview(
        session, px_per_mm=1.0, max_frames=3
    )

    full = [call for call in calls if call[1] == 1000.0]
    coarse = [call for call in calls if call[1] < 1000.0]
    assert len(full) == len(manifest["views"])
    assert len(coarse) == 16


def test_manifest_v3_carries_bridge_geometry(tmp_path):
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
    assert manifest["version"] == 3

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
