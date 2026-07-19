"""Runner integration: detect -> back-project -> track -> render plumbing."""
from __future__ import annotations

import sys
import threading
import json
from types import SimpleNamespace

import numpy as np
import pytest

import live_scan_viewer
from coverage import CoverageGrid
from live_glow import project_polygon, table_to_image_homography
from live_scan_viewer import LiveGlowSession
from tests.synthetic import look_at_pose
from transforms import intrinsics_to_K


def _frame(frame_id, pose, K):
    return {
        "frame_id": frame_id,
        "image": np.full((1080, 1440, 3), 40, dtype=np.uint8),
        "pose_mat": pose,
        "K": K,
        "timestamp": float(frame_id),
    }


def _pose_frame(
    frame_id=0, *, translation=(0.0, 0.0, 0.0), yaw_deg=0.0
):
    angle = np.deg2rad(yaw_deg)
    cosine, sine = np.cos(angle), np.sin(angle)
    pose = np.eye(4)
    pose[:3, :3] = np.array([
        [cosine, -sine, 0.0],
        [sine, cosine, 0.0],
        [0.0, 0.0, 1.0],
    ])
    pose[:3, 3] = translation
    return {"frame_id": frame_id, "pose_mat": pose}


def test_latest_frame_scheduler_runs_first_frame_and_enforces_interval():
    scheduler = live_scan_viewer.LatestFrameScheduler()

    assert scheduler.should_detect(_pose_frame(0), now=0.0)
    assert not scheduler.should_detect(
        _pose_frame(1, translation=(0.10, 0.0, 0.0)), now=0.49
    )
    assert scheduler.should_detect(
        _pose_frame(2, translation=(0.10, 0.0, 0.0)), now=0.50
    )


def test_latest_frame_scheduler_continues_when_pose_is_stationary():
    scheduler = live_scan_viewer.LatestFrameScheduler()
    assert scheduler.should_detect(_pose_frame(0), now=0.0)

    assert scheduler.should_detect(_pose_frame(1), now=0.50)


def test_latest_frame_scheduler_rejects_duplicate_frame_after_interval():
    scheduler = live_scan_viewer.LatestFrameScheduler()
    frame = _pose_frame(7)
    assert scheduler.should_detect(frame, now=0.0)

    assert not scheduler.should_detect(frame, now=1.0)


def test_live_diagnostics_counts_phone_frames_and_completed_passes():
    diagnostics = live_scan_viewer.LiveDiagnostics()

    diagnostics.note_phone_frame()
    diagnostics.note_phone_frame()
    diagnostics.note_segmentation_pass()

    assert diagnostics.snapshot() == {
        "phone_frames": 2,
        "segmentation_passes": 1,
    }


def test_pass_diagnostics_prints_and_appends_jsonl(tmp_path, capsys):
    path = tmp_path / "passes.jsonl"
    diagnostics = live_scan_viewer.PassDiagnostics(path)
    event = {
        "frame_id": 42,
        "inference_latency_ms": 318.5,
        "raw_mask_count": 12,
        "visible_table_count": 0,
        "visible_screen_count": 3,
        "fallback_reason": "no_visible_table_output",
        "fallback_flash_count": 3,
    }

    diagnostics.record(event)

    assert json.loads(path.read_text().strip()) == event
    output = capsys.readouterr().out
    assert "seg f42" in output
    assert "raw=12" in output
    assert "flash=3" in output


def test_rgb_hud_includes_workspace_progress_and_live_diagnostics(monkeypatch):
    labels = []
    text_styles = []
    real_put_text = live_scan_viewer.cv2.putText

    def recording_put_text(image, text, *args, **kwargs):
        labels.append(text)
        text_styles.append((text, args[2], args[4]))
        return real_put_text(image, text, *args, **kwargs)

    monkeypatch.setattr(live_scan_viewer.cv2, "putText", recording_put_text)

    live_scan_viewer._overlay_hud(
        np.zeros((120, 320, 3), dtype=np.uint8),
        piece_count=12,
        coverage_fraction=0.846,
        phone_frames=40,
        segmentation_passes=7,
    )

    assert "You've scanned 85% of the workspace" in labels
    assert any("phone frames 40" in label for label in labels)
    assert any("segmentation passes 7" in label for label in labels)
    workspace_style = next(
        style for style in text_styles if "workspace" in style[0]
    )
    assert workspace_style[1] >= 1.0
    assert workspace_style[2] >= 3


def test_workspace_crop_blacks_background_and_reports_image_offset():
    image = np.full((100, 120, 3), 90, dtype=np.uint8)
    contour = np.array([[20, 10], [80, 10], [80, 70], [20, 70]], float)

    crop, offset, workspace = live_scan_viewer._workspace_crop(
        image, [contour], pad_fraction=0.02
    )

    assert offset == (18, 8)
    assert workspace.shape == image.shape[:2]
    assert tuple(crop[0, 0]) == (0, 0, 0)
    assert tuple(crop[20, 20]) == (90, 90, 90)


def test_workspace_overlap_rasterizes_only_polygon_region(monkeypatch):
    workspace = np.ones((1920, 1440), dtype=bool)
    polygon = np.array([[700, 900], [740, 900], [740, 940], [700, 940]])
    raster_shapes = []
    real_fill_poly = live_scan_viewer.cv2.fillPoly

    def recording_fill_poly(image, *args, **kwargs):
        raster_shapes.append(image.shape)
        return real_fill_poly(image, *args, **kwargs)

    monkeypatch.setattr(live_scan_viewer.cv2, "fillPoly", recording_fill_poly)

    overlap = live_scan_viewer._workspace_overlap(polygon, workspace)

    assert overlap == 1.0
    assert max(height * width for height, width in raster_shapes) < \
        workspace.size // 100


def _square_at(cx, cy, half=4.0):
    return np.array([
        [cx - half, cy - half], [cx + half, cy - half],
        [cx + half, cy + half], [cx - half, cy + half],
    ])


def _depth_frame(frame_id=0, *, depth_m=0.50, fx=900.0, fy=900.0):
    K = intrinsics_to_K(fx, fy, 720.0, 540.0)
    pose = look_at_pose([0.0, 0.0, depth_m], [0.0, 0.0, 0.0])
    frame = _frame(frame_id, pose, K)
    frame["depth"] = np.full((192, 256), depth_m, dtype=np.float32)
    return frame


def test_screen_piece_gate_accepts_depth_supported_lego_size():
    frame = _depth_frame(depth_m=0.50, fx=900.0, fy=900.0)

    evidence = live_scan_viewer.screen_piece_evidence(
        _square_at(720, 540, half=35), 0.90, frame
    )

    assert evidence.accepted
    assert evidence.reason == "accepted"


def test_screen_piece_gate_rejects_oversized_mask():
    frame = _depth_frame(depth_m=0.70, fx=900.0, fy=900.0)

    evidence = live_scan_viewer.screen_piece_evidence(
        _square_at(720, 540, half=220), 0.95, frame
    )

    assert not evidence.accepted
    assert evidence.reason == "oversized"


def test_screen_piece_gate_rejects_missing_depth_support():
    frame = _depth_frame(depth_m=0.50, fx=900.0, fy=900.0)
    frame["depth"][:] = 0.0

    evidence = live_scan_viewer.screen_piece_evidence(
        _square_at(720, 540, half=35), 0.90, frame
    )

    assert not evidence.accepted
    assert evidence.reason == "depth_unsupported"


def test_screen_scheduler_spaces_and_cools_regions():
    scheduler = live_scan_viewer.ScreenFlashScheduler(
        max_count=4,
        min_spacing_fraction=0.08,
        cooldown_s=1.5,
    )
    candidates = [
        (_square_at(100 + 180 * index, 200, half=20), 0.95 - 0.01 * index)
        for index in range(6)
    ]

    first = scheduler.select(candidates, (1080, 1440, 3), now=0.0)
    scheduler.arm(first, (1080, 1440, 3), now=0.0)
    second = scheduler.select(candidates, (1080, 1440, 3), now=0.5)

    assert len(first) == 4
    assert not {
        scheduler.cell(polygon, (1080, 1440, 3)) for polygon, _ in first
    } & {
        scheduler.cell(polygon, (1080, 1440, 3)) for polygon, _ in second
    }


def test_screen_scheduler_fail_opens_when_every_region_is_cooling():
    scheduler = live_scan_viewer.ScreenFlashScheduler(max_count=4)
    candidates = [(_square_at(720, 540, half=20), 0.95)]
    first = scheduler.select(candidates, (1080, 1440, 3), now=0.0)
    scheduler.arm(first, (1080, 1440, 3), now=0.0)

    selected = scheduler.select(
        candidates,
        (1080, 1440, 3),
        now=0.5,
        guarantee_one=True,
    )

    assert len(selected) == 1


def test_spaced_mask_selection_spreads_flashes_across_current_view():
    detections = [
        (_square_at(20, 20), 0.99),
        (_square_at(28, 20), 0.98),
        (_square_at(180, 20), 0.90),
        (_square_at(20, 180), 0.89),
        (_square_at(180, 180), 0.88),
    ]

    selected = live_scan_viewer._select_spaced_masks(
        detections,
        image_shape=(200, 200, 3),
        max_count=4,
        min_spacing_fraction=0.30,
    )

    centroids = [polygon.mean(axis=0) for polygon, _ in selected]
    assert len(centroids) == 4
    assert all(
        np.linalg.norm(left - right) >= 0.30 * np.hypot(200, 200)
        for index, left in enumerate(centroids)
        for right in centroids[index + 1:]
    )


def test_neon_scanner_sweep_moves_down_then_back_up():
    frame = np.zeros((120, 200, 3), dtype=np.uint8)
    period = 2.0

    at_top = live_scan_viewer._render_scanner_sweep(
        frame, now=0.0, period_s=period
    )
    at_bottom = live_scan_viewer._render_scanner_sweep(
        frame, now=period / 2.0, period_s=period
    )
    back_at_top = live_scan_viewer._render_scanner_sweep(
        frame, now=period, period_s=period
    )

    top_row = int(np.argmax(at_top.max(axis=2).mean(axis=1)))
    bottom_row = int(np.argmax(at_bottom.max(axis=2).mean(axis=1)))
    return_row = int(np.argmax(back_at_top.max(axis=2).mean(axis=1)))
    assert top_row < frame.shape[0] // 4
    assert bottom_row > 3 * frame.shape[0] // 4
    assert abs(return_row - top_row) <= 1
    assert int(at_top.max()) >= 240


def test_coverage_presentation_eases_from_95_to_100():
    display = live_scan_viewer.CoveragePresentation(
        rate_per_s=0.04, completion_s=4.0
    )

    assert display.update(0.0, elapsed_s=0.0) == 0.0
    assert 0.40 <= display.update(0.0, elapsed_s=12.0) <= 0.60
    assert display.update(0.0, elapsed_s=23.75) == pytest.approx(0.95)
    assert display.update(0.0, elapsed_s=25.75) == pytest.approx(0.975)
    assert display.update(0.0, elapsed_s=27.75) == pytest.approx(1.0)
    assert display.complete


def test_coverage_presentation_starts_completion_when_lidar_reaches_95():
    display = live_scan_viewer.CoveragePresentation(
        rate_per_s=0.04, completion_s=4.0
    )

    assert display.update(0.95, elapsed_s=3.0) == pytest.approx(0.95)
    assert display.update(0.95, elapsed_s=5.0) == pytest.approx(0.975)


def test_live_yolo_detector_requests_coarse_masks(monkeypatch, tmp_path):
    captured = {}

    class _FakeYoloSegModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def predict(self, image):
            return []

    monkeypatch.setitem(
        sys.modules,
        "pipeline.segdetect",
        SimpleNamespace(YoloSegModel=_FakeYoloSegModel),
    )

    live_scan_viewer.YoloMaskDetector(
        weights=tmp_path / "lego.pt", lego_cv_dir=tmp_path
    )

    assert captured["retina_masks"] is False
    assert captured["max_det"] == 96
    assert captured["imgsz"] == 640


def test_live_yolo_detector_adds_lego_cv_virtualenv_runtime(
    monkeypatch, tmp_path
):
    runtime = (
        tmp_path
        / ".venv"
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    runtime.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setitem(
        sys.modules,
        "pipeline.segdetect",
        SimpleNamespace(
            YoloSegModel=lambda **kwargs: SimpleNamespace(predict=lambda image: [])
        ),
    )

    live_scan_viewer.YoloMaskDetector(
        weights=tmp_path / "lego.pt", lego_cv_dir=tmp_path
    )

    assert sys.path[0] == str(tmp_path)
    assert sys.path[1] == str(runtime)


def test_load_live_workspace_replays_depth_geometry(monkeypatch):
    contours = [[[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]]]
    world_to_table = np.diag([1.0, 1.0, 1.0, 1.0])

    class _Grid:
        def admissible_workspace_contours_xy(self, **kwargs):
            assert kwargs == {"min_seen": 2, "close_cells": 5}
            return contours

    monkeypatch.setattr(
        live_scan_viewer,
        "replay_session",
        lambda session_dir, cell_m, stride: (
            _Grid(), {"world_to_table": world_to_table}
        ),
        raising=False,
    )

    loaded_world, loaded_contours, loaded_grid = \
        live_scan_viewer._load_live_workspace(
        "session"
    )

    np.testing.assert_array_equal(loaded_world, world_to_table)
    assert loaded_contours == contours
    assert isinstance(loaded_grid, _Grid)


def test_live_coverage_fraction_uses_fixed_reference_workspace():
    reference = CoverageGrid([[0.0, 0.04], [0.0, 0.02]], cell_m=0.01)
    reference.seen_count[:, :] = 2
    tracker = live_scan_viewer.LiveCoverageTracker(
        reference, np.eye(4), min_seen=2
    )
    tracker.grid.seen_count[:, :2] = 2

    assert tracker.fraction() == 0.5


def test_live_coverage_updates_latest_depth_at_bounded_cadence(monkeypatch):
    reference = CoverageGrid([[-0.1, 0.1], [-0.1, 0.1]], cell_m=0.01)
    reference.seen_count[:, :] = 2
    tracker = live_scan_viewer.LiveCoverageTracker(
        reference, np.eye(4), max_update_hz=4.0
    )
    calls = []
    monkeypatch.setattr(
        tracker.grid,
        "mark_frame",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    frame = {
        "image": np.zeros((100, 200, 3), dtype=np.uint8),
        "depth": np.ones((50, 100), dtype=np.float32),
        "K": np.eye(3),
        "pose_mat": np.eye(4),
    }

    assert tracker.update(frame, now=0.0)
    assert not tracker.update(frame, now=0.10)
    assert tracker.update(frame, now=0.25)
    assert len(calls) == 2


def test_live_record3d_callback_snapshots_depth_with_rgb():
    rgb = np.full((4, 6, 3), 20, dtype=np.uint8)
    depth = np.full((2, 3), 0.5, dtype=np.float32)

    class _Session:
        def get_rgb_frame(self):
            return rgb

        def get_depth_frame(self):
            return depth

        def get_camera_pose(self):
            return SimpleNamespace(
                qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                tx=0.0, ty=0.0, tz=0.5,
            )

        def get_intrinsic_mat(self):
            return SimpleNamespace(fx=100.0, fy=100.0, tx=3.0, ty=2.0)

    source = live_scan_viewer.LiveRecord3DSource.__new__(
        live_scan_viewer.LiveRecord3DSource
    )
    source.session = _Session()
    source._latest = None
    source._lock = threading.Lock()
    source._event = threading.Event()

    source._on_new_frame()
    depth[:] = 0.0

    assert source._event.is_set()
    np.testing.assert_array_equal(
        source._latest["depth"], np.full((2, 3), 0.5, dtype=np.float32)
    )


def test_live_main_passes_lidar_workspace_to_viewer(monkeypatch):
    contours = [[[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]]]
    reference_grid = object()
    captured = {}
    monkeypatch.setattr(
        live_scan_viewer, "YoloMaskDetector", lambda *args, **kwargs: "detector"
    )
    monkeypatch.setattr(
        live_scan_viewer,
        "_load_live_workspace",
        lambda session_dir: (np.eye(4), contours, reference_grid),
    )
    monkeypatch.setattr(
        live_scan_viewer,
        "run_live",
        lambda world, detector, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["live_scan_viewer.py", "--live", "--world-to-table", "anchor"],
    )

    live_scan_viewer.main()

    assert captured["workspace_contours_table"] == contours
    assert captured["reference_grid"] is reference_grid


def test_recorded_replay_passes_lidar_workspace_to_session(monkeypatch):
    contours = [[[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1]]]
    captured = {}

    class _EmptySource:
        def __init__(self, *args, **kwargs):
            self.world_to_table = np.eye(4)

        def frames(self):
            return []

    class _CapturingSession:
        def __init__(self, detector, world_to_table, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(live_scan_viewer, "RecordedFrameSource", _EmptySource)
    monkeypatch.setattr(live_scan_viewer, "LiveGlowSession", _CapturingSession)
    monkeypatch.setattr(
        live_scan_viewer,
        "_load_live_workspace",
        lambda session_dir: (np.eye(4), contours, object()),
    )

    live_scan_viewer.run_recorded(
        "session", lambda image: [], headless=True
    )

    assert captured["workspace_contours_table"] == contours


def test_loose_piece_gate_accepts_long_plate_and_rejects_large_chunk():
    plate_2x16 = np.array([
        [-0.064, -0.008], [0.064, -0.008],
        [0.064, 0.008], [-0.064, 0.008],
    ])
    assembled_chunk = np.array([
        [-0.10, -0.05], [0.10, -0.05],
        [0.10, 0.05], [-0.10, 0.05],
    ])

    assert live_scan_viewer._is_loose_piece(plate_2x16)
    assert not live_scan_viewer._is_loose_piece(assembled_chunk)


def _workspace_session(detector, workspace_table, *, height=0.6):
    K = intrinsics_to_K(900.0, 900.0, 720.0, 540.0)
    pose = look_at_pose([0.0, 0.0, height], [0.0, 0.0, 0.0])
    session = LiveGlowSession(
        detector,
        np.eye(4),
        workspace_contours_table=[workspace_table],
        min_confidence=0.6,
        activation_stagger_s=0.0,
    )
    return session, _frame(0, pose, K), table_to_image_homography(
        K, pose, np.eye(4)
    )


def _local_polygon_detector(full_polygon, workspace_table, homography):
    image = np.zeros((1080, 1440, 3), dtype=np.uint8)
    _, (x0, y0), _ = live_scan_viewer._workspace_crop(
        image, [project_polygon(workspace_table, homography)]
    )
    return _FixedImageDetector(
        np.asarray(full_polygon) - np.array([x0, y0]), confidence=0.9
    )


def test_live_session_restores_crop_offset_and_tracks_valid_piece():
    workspace = np.array([
        [-0.30, -0.30], [0.30, -0.30],
        [0.30, 0.30], [-0.30, 0.30],
    ])
    piece = np.array([
        [-0.02, -0.01], [0.02, -0.01],
        [0.02, 0.01], [-0.02, 0.01],
    ])
    _, frame, homography = _workspace_session(lambda image: [], workspace)
    detector = _local_polygon_detector(
        project_polygon(piece, homography), workspace, homography
    )
    session, frame, _ = _workspace_session(detector, workspace)

    session.detect(frame, now=0.0)

    assert len(session.tracker.tracks) == 1
    np.testing.assert_allclose(
        session.tracker.tracks[0].table_centroid, piece.mean(axis=0), atol=1e-6
    )


def test_live_session_keeps_animating_when_workspace_projection_disappears(
    monkeypatch,
):
    workspace = np.array([
        [-0.30, -0.30], [0.30, -0.30],
        [0.30, 0.30], [-0.30, 0.30],
    ])
    calls = []

    def detector(image):
        calls.append(image.shape)
        height, width = image.shape[:2]
        return [(_square_at(width / 2, height / 2, half=20), 0.90)]

    session, frame, _ = _workspace_session(detector, workspace)
    frame["depth"] = np.full((192, 256), 0.60, dtype=np.float32)
    monkeypatch.setattr(
        live_scan_viewer,
        "_workspace_crop",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("workspace left frame")
        ),
    )

    session.detect(frame, now=1.0)
    rendered, count = session.render(frame, now=1.05)

    assert calls == [frame["image"].shape]
    assert len(session.screen_flashes) == 1
    assert count == 1
    center_y, center_x = frame["image"].shape[0] // 2, \
        frame["image"].shape[1] // 2
    assert rendered[center_y, center_x, 2] > frame["image"][
        center_y, center_x, 2
    ]


def test_live_session_rejects_polygon_outside_lidar_workspace():
    workspace = np.array([
        [-0.12, -0.12], [0.12, -0.12],
        [0.12, 0.12], [-0.12, 0.12],
    ])
    outside = np.array([
        [0.12, -0.01], [0.15, -0.01],
        [0.15, 0.01], [0.12, 0.01],
    ])
    _, frame, homography = _workspace_session(lambda image: [], workspace)
    detector = _local_polygon_detector(
        project_polygon(outside, homography), workspace, homography
    )
    session, frame, _ = _workspace_session(detector, workspace)

    session.detect(frame, now=0.0)

    assert session.tracker.tracks == []


def test_live_session_rejects_oversized_inside_workspace_polygon():
    workspace = np.array([
        [-0.30, -0.30], [0.30, -0.30],
        [0.30, 0.30], [-0.30, 0.30],
    ])
    large = np.array([
        [-0.10, -0.05], [0.10, -0.05],
        [0.10, 0.05], [-0.10, 0.05],
    ])
    _, frame, homography = _workspace_session(lambda image: [], workspace)
    detector = _local_polygon_detector(
        project_polygon(large, homography), workspace, homography
    )
    session, frame, _ = _workspace_session(detector, workspace)

    session.detect(frame, now=0.0)

    assert session.tracker.tracks == []


class _FixedImageDetector:
    """Returns the same image-space mask every call (a stationary camera sees
    a piece in a fixed pixel region)."""

    def __init__(self, polygon_px, confidence=0.9):
        self.polygon_px = np.asarray(polygon_px, dtype=np.float64)
        self.confidence = confidence

    def __call__(self, image_bgr):
        return [(self.polygon_px, self.confidence)]


class _RelativeCenterDetector:
    """Returns a centred mask in the coordinate system of each input image."""

    def __init__(self, confidence=0.9, half=20):
        self.confidence = confidence
        self.half = half
        self.calls = []

    def __call__(self, image_bgr):
        self.calls.append(image_bgr.shape)
        height, width = image_bgr.shape[:2]
        return [(
            _square_at(width / 2, height / 2, half=self.half),
            self.confidence,
        )]


def test_raw_masks_rejected_by_table_still_emit_depth_safe_fallback(
    monkeypatch,
):
    workspace = np.array([
        [-0.12, -0.12], [0.12, -0.12],
        [0.12, 0.12], [-0.12, 0.12],
    ])
    detector = _RelativeCenterDetector(confidence=0.90, half=20)
    session, frame, _ = _workspace_session(detector, workspace)
    frame["depth"] = np.full((192, 256), 0.55, dtype=np.float32)
    monkeypatch.setattr(
        live_scan_viewer,
        "_workspace_overlap",
        lambda *args, **kwargs: 0.0,
    )

    stats = session.detect(frame, now=0.0)
    _, visible = session.render(frame, now=0.05)

    assert stats["raw_mask_count"] > 0
    assert stats["workspace_rejected_count"] > 0
    assert stats["fallback_flash_count"] >= 1
    assert stats["fallback_reason"] == "no_visible_table_output"
    assert visible >= 1
    assert len(detector.calls) == 2


def test_live_render_count_excludes_faded_tracks():
    K = intrinsics_to_K(900.0, 900.0, 720.0, 540.0)
    pose = look_at_pose([0.0, 0.0, 0.4], [0.0, 0.0, 0.0])
    polygon = np.array([
        [-0.02, -0.02], [0.02, -0.02],
        [0.02, 0.02], [-0.02, 0.02],
    ])
    detector = _FixedImageDetector(project_polygon(
        polygon, table_to_image_homography(K, pose, np.eye(4))
    ))
    session = LiveGlowSession(
        detector,
        np.eye(4),
        min_confidence=0.6,
        activation_stagger_s=0.0,
    )
    frame = _frame(0, pose, K)

    session.detect(frame, now=0.0)
    _, count = session.render(frame, now=live_scan_viewer.FLASH_END_S)

    assert count == 0


def test_detect_activates_at_inference_completion_not_submission(monkeypatch):
    workspace = np.array([
        [-0.12, -0.12], [0.12, -0.12],
        [0.12, 0.12], [-0.12, 0.12],
    ])
    session = LiveGlowSession(
        _FixedImageDetector(_square_at(720, 540, half=20)),
        np.eye(4),
        workspace_contours_table=[workspace],
        min_confidence=0.6,
        activation_stagger_s=0.0,
        clock=lambda: 0.40,
    )
    frame = _depth_frame()
    monkeypatch.setattr(
        live_scan_viewer,
        "_workspace_crop",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("workspace left frame")
        ),
    )

    stats = session.detect(frame, now=0.0)

    assert stats["inference_latency_ms"] == 400.0
    assert session.screen_flashes[0].activation_time == 0.40


def test_detect_backproject_track_render_single_activation():
    K = intrinsics_to_K(900.0, 900.0, 720.0, 540.0)
    pose = look_at_pose([0.0, 0.0, 0.4], [0.0, 0.0, 0.0])
    W2T = np.eye(4)
    # A fixed image mask around image centre.
    table_poly = np.array([[-0.02, -0.02], [0.02, -0.02],
                           [0.02, 0.02], [-0.02, 0.02]])
    img_poly = project_polygon(table_poly, table_to_image_homography(K, pose, W2T))
    detector = _FixedImageDetector(img_poly)

    session = LiveGlowSession(detector, W2T, min_confidence=0.6,
                              activation_stagger_s=0.0)

    # Stream several stationary frames; detection every frame.
    activation_time = None
    for fid in range(5):
        frame = _frame(fid, pose, K)
        session.detect(frame, now=fid * 0.1)
        rendered, count = session.render(frame, now=fid * 0.1 + 0.01)
        assert count == 1  # exactly one discovered piece, persists
        track = session.tracker.tracks[0]
        activation_time = activation_time or track.activation_time
        assert track.activation_time == activation_time  # activated once

    # The rendered glow tinted the piece region.
    frame = _frame(99, pose, K)
    rendered, _ = session.render(frame, now=0.2)
    cx, cy = np.rint(img_poly.mean(0)).astype(int)
    assert rendered[cy, cx][2] > 40 + 30  # shifted toward yellow glow hue
    assert tuple(rendered[3, 3]) == (40, 40, 40)  # corner untouched


def test_live_session_defaults_to_spaced_discoveries_and_scan_memory():
    session = LiveGlowSession(lambda image: [], np.eye(4))

    assert session.tracker.max_new_tracks_per_update == 8
    assert session.tracker.new_track_spacing_m == 0.07
    assert session.tracker.track_ttl_s == 30.0


def test_two_pieces_activate_progressively_not_at_once():
    K = intrinsics_to_K(900.0, 900.0, 720.0, 540.0)
    pose = look_at_pose([0.0, 0.0, 0.5], [0.0, 0.0, 0.0])
    W2T = np.eye(4)
    H = table_to_image_homography(K, pose, W2T)
    polys = [
        np.array([[-0.06, -0.01], [-0.03, -0.01], [-0.03, 0.01], [-0.06, 0.01]]),
        np.array([[0.03, -0.01], [0.06, -0.01], [0.06, 0.01], [0.03, 0.01]]),
    ]
    img_polys = [project_polygon(p, H) for p in polys]

    class _TwoDetector:
        def __call__(self, image_bgr):
            return [(img_polys[0], 0.9), (img_polys[1], 0.9)]

    session = LiveGlowSession(_TwoDetector(), W2T, min_confidence=0.6,
                              activation_stagger_s=0.5)
    session.detect(_frame(0, pose, K), now=0.0)
    # Both seen in one inference update, but only one lights immediately.
    _, count_now = session.render(_frame(0, pose, K), now=0.01)
    assert count_now == 1
    _, count_later = session.render(_frame(1, pose, K), now=0.6)
    # The first flash has ended when the second staggered flash starts. The
    # truthful render count is one visible glow, even though both tracks have
    # now activated.
    assert count_later == 1
    assert len(session.tracker.active_tracks(now=0.6)) == 2
