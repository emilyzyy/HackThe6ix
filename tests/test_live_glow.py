"""Live-glow engine: table-anchored tracks that stay attached across camera
motion, activate once, and animate independently of the detection cadence."""
from __future__ import annotations

import numpy as np

from live_glow import (
    FLASH_END_S,
    GLOW_HUE_BGR,
    GlowTracker,
    back_project_polygon,
    glow_envelope,
    project_polygon,
    table_to_image_homography,
)
from tests.synthetic import look_at_pose
from transforms import intrinsics_to_K


def _setup(height=0.4, look_from=(0.0, 0.0), look_at=(0.0, 0.0)):
    """Camera above a z=0 table (identity world<->table)."""
    K = intrinsics_to_K(900.0, 900.0, 720.0, 540.0)
    pose = look_at_pose([look_from[0], look_from[1], height],
                        [look_at[0], look_at[1], 0.0])
    world_to_table = np.eye(4)
    return K, pose, world_to_table


def test_round_trip_image_table_image_on_plane():
    K, pose, W2T = _setup()
    H = table_to_image_homography(K, pose, W2T)
    # A square LEGO footprint in table metres (~2x4 stud plate).
    table_poly = np.array([[-0.008, -0.016], [0.008, -0.016],
                           [0.008, 0.016], [-0.008, 0.016]])
    img_poly = project_polygon(table_poly, H)
    back = back_project_polygon(img_poly, H)
    np.testing.assert_allclose(back, table_poly, atol=1e-6)


def test_mask_stays_attached_as_camera_moves():
    # Detect a piece from one pose, back-project to table, then project the
    # SAME table polygon into a different camera pose. It must land where the
    # piece physically is in the new view (i.e. re-projecting the table anchor
    # equals directly projecting the physical footprint from the new pose).
    K, pose_a, W2T = _setup(look_from=(0.0, 0.0))
    table_poly = np.array([[0.02, 0.01], [0.05, 0.01],
                           [0.05, 0.03], [0.02, 0.03]])
    H_a = table_to_image_homography(K, pose_a, W2T)
    detected_img = project_polygon(table_poly, H_a)
    anchor_table = back_project_polygon(detected_img, H_a)

    # Camera pans (translates straight down over a shifted spot), so the piece
    # shifts within the frame rather than staying on the optical axis.
    _, pose_b, _ = _setup(look_from=(0.06, -0.04), look_at=(0.06, -0.04))
    H_b = table_to_image_homography(K, pose_b, W2T)
    from_anchor = project_polygon(anchor_table, H_b)
    from_physical = project_polygon(table_poly, H_b)
    # The anchored mask tracks the physical piece to sub-pixel accuracy.
    np.testing.assert_allclose(from_anchor, from_physical, atol=1e-3)
    # And it actually moved in image space between the two views.
    assert np.linalg.norm(
        project_polygon(anchor_table, H_a).mean(0) - from_anchor.mean(0)
    ) > 20.0


def test_glow_envelope_flashes_then_turns_off():
    at_start = glow_envelope(age_s=0.0)
    fading = glow_envelope(age_s=0.35)
    off = glow_envelope(age_s=FLASH_END_S)

    assert at_start["phase"] == "flash"
    assert fading["phase"] == "fade"
    assert off["phase"] == "off"
    assert at_start["fill_alpha"] > fading["fill_alpha"] > 0.0
    assert at_start["edge_alpha"] > fading["edge_alpha"] > 0.0
    assert off["fill_alpha"] == 0.0
    assert off["edge_alpha"] == 0.0


def test_live_glow_uses_short_lego_yellow_flash():
    assert FLASH_END_S == 0.48
    assert GLOW_HUE_BGR == (0, 213, 255)


def _det(cx, cy, *, conf=0.9, half=0.01):
    poly = np.array([[cx - half, cy - half], [cx + half, cy - half],
                     [cx + half, cy + half], [cx - half, cy + half]])
    return {"table_polygon": poly,
            "table_centroid": np.array([cx, cy]), "confidence": conf}


def test_track_activates_once_and_persists():
    tracker = GlowTracker(min_confidence=0.6, activation_stagger_s=0.0)
    # First sighting creates + activates one track.
    tracker.update([_det(0.02, 0.03)], now=0.0, camera_xy=(0.0, 0.0))
    active = tracker.active_tracks(now=0.05)
    assert len(active) == 1
    track_id = active[0].track_id
    born = active[0].activation_time

    # Re-detected slightly moved in later frames: same track, NOT re-activated.
    tracker.update([_det(0.021, 0.031)], now=0.3, camera_xy=(0.01, 0.0))
    tracker.update([_det(0.022, 0.032)], now=0.6, camera_xy=(0.02, 0.0))
    active = tracker.active_tracks(now=0.65)
    assert len(active) == 1
    assert active[0].track_id == track_id
    assert active[0].activation_time == born  # activation happened once


def test_low_confidence_detection_ignored():
    tracker = GlowTracker(min_confidence=0.6, activation_stagger_s=0.0)
    tracker.update([_det(0.0, 0.0, conf=0.3)], now=0.0, camera_xy=(0.0, 0.0))
    assert tracker.active_tracks(now=0.1) == []


def test_nearby_detections_cannot_collapse_into_one_track_per_update():
    tracker = GlowTracker(min_confidence=0.6, activation_stagger_s=0.0)

    tracker.update(
        [_det(0.000, 0.0, half=0.005), _det(0.015, 0.0, half=0.005)],
        now=0.0,
        camera_xy=(0.0, 0.0),
    )

    assert len(tracker.tracks) == 2
    assert {round(track.table_centroid[0], 3) for track in tracker.tracks} == {
        0.000, 0.015,
    }


def test_new_track_creation_is_capped_per_update():
    tracker = GlowTracker(
        min_confidence=0.6,
        activation_stagger_s=0.0,
        max_new_tracks_per_update=3,
    )

    tracker.update(
        [_det(index * 0.05, 0.0) for index in range(6)],
        now=0.0,
        camera_xy=(0.0, 0.0),
    )

    assert len(tracker.tracks) == 3


def test_new_track_budget_prefers_high_confidence_discoveries():
    tracker = GlowTracker(
        min_confidence=0.6,
        activation_stagger_s=0.0,
        max_new_tracks_per_update=1,
    )

    tracker.update(
        [_det(0.0, 0.0, conf=0.70), _det(0.10, 0.0, conf=0.95)],
        now=0.0,
        camera_xy=(0.0, 0.0),
    )

    assert len(tracker.tracks) == 1
    assert tracker.tracks[0].confidence == 0.95


def test_new_tracks_are_spatially_separated_within_one_update():
    tracker = GlowTracker(
        min_confidence=0.6,
        activation_stagger_s=0.0,
        max_new_tracks_per_update=10,
        new_track_spacing_m=0.04,
    )

    tracker.update(
        [_det(x, 0.0, half=0.005) for x in (0.00, 0.02, 0.05, 0.07)],
        now=0.0,
        camera_xy=(0.0, 0.0),
    )

    assert [round(track.table_centroid[0], 2) for track in tracker.tracks] == [
        0.00, 0.05,
    ]


def test_existing_track_refresh_does_not_consume_new_track_budget():
    tracker = GlowTracker(
        min_confidence=0.6,
        activation_stagger_s=0.0,
        max_new_tracks_per_update=1,
        new_track_spacing_m=0.04,
    )
    tracker.update([_det(0.0, 0.0)], now=0.0, camera_xy=(0.0, 0.0))

    tracker.update(
        [_det(0.001, 0.0), _det(0.10, 0.0), _det(0.20, 0.0)],
        now=1.0,
        camera_xy=(0.0, 0.0),
    )

    assert len(tracker.tracks) == 2
    np.testing.assert_allclose(
        tracker.tracks[0].table_centroid, np.array([0.001, 0.0])
    )


def test_progressive_activation_staggers_new_tracks():
    tracker = GlowTracker(min_confidence=0.6, activation_stagger_s=0.5)
    # Three pieces seen in ONE inference update must not all activate at once.
    tracker.update(
        [_det(0.00, 0.0), _det(0.05, 0.0), _det(0.10, 0.0)],
        now=0.0, camera_xy=(0.0, 0.0),
    )
    assert len(tracker.active_tracks(now=0.01)) == 1
    assert len(tracker.active_tracks(now=0.55)) == 2
    assert len(tracker.active_tracks(now=1.05)) == 3


def test_default_activation_pace_keeps_up_with_crowded_live_scan():
    tracker = GlowTracker(min_confidence=0.6)
    tracker.update(
        [_det(0.00, 0.0), _det(0.05, 0.0)],
        now=0.0,
        camera_xy=(0.0, 0.0),
    )

    assert len(tracker.active_tracks(now=0.01)) == 1
    assert len(tracker.active_tracks(now=0.10)) == 2


def test_progressive_order_follows_camera_travel_direction():
    # Camera travels in +x; pieces should light up in +x order (the scan pulse).
    tracker = GlowTracker(min_confidence=0.6, activation_stagger_s=0.5)
    tracker.update([_det(0.10, 0.0)], now=-1.0, camera_xy=(-0.02, 0.0))
    tracker.update([_det(0.00, 0.0)], now=-1.0, camera_xy=(-0.01, 0.0))
    # establish +x travel history, then a batch of three at once
    tracker = GlowTracker(min_confidence=0.6, activation_stagger_s=0.5)
    tracker.set_camera_travel((1.0, 0.0))
    tracker.update(
        [_det(0.10, 0.0), _det(0.00, 0.0), _det(0.05, 0.0)],
        now=0.0, camera_xy=(0.0, 0.0),
    )
    first = tracker.active_tracks(now=0.01)
    assert len(first) == 1
    # The first to light is the one furthest back along travel (smallest x).
    assert first[0].table_centroid[0] == 0.0


def test_render_glow_tints_inside_and_leaves_outside():
    from live_glow import Track, render_glow, GLOW_HUE_BGR
    K, pose, W2T = _setup()
    H = table_to_image_homography(K, pose, W2T)
    frame = np.full((1080, 1440, 3), 30, dtype=np.uint8)
    poly = np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]])
    track = Track(track_id=0, table_polygon=poly,
                  table_centroid=np.array([0.0, 0.0]), confidence=0.9,
                  created_at=0.0, last_seen=0.0, activation_time=0.0)
    out = render_glow(frame, [track], H, now=0.05)
    img_poly = np.rint(project_polygon(poly, H)).astype(int)
    cx, cy = img_poly.mean(0).astype(int)
    # Centre pixel shifted toward LEGO yellow (red channel up from 30).
    assert out[cy, cx][2] > frame[cy, cx][2] + 40
    # A far corner is untouched.
    assert tuple(out[5, 5]) == (30, 30, 30)


def test_render_glow_fresh_brighter_than_faded_and_then_invisible():
    from live_glow import Track, render_glow
    K, pose, W2T = _setup()
    H = table_to_image_homography(K, pose, W2T)
    frame = np.full((1080, 1440, 3), 30, dtype=np.uint8)
    poly = np.array([[-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02]])

    def _center_val(age):
        tr = Track(0, poly, np.array([0.0, 0.0]), 0.9, 0.0, 0.0,
                   activation_time=0.0)
        out = render_glow(frame, [tr], H, now=age)
        ip = np.rint(project_polygon(poly, H)).astype(int)
        cx, cy = ip.mean(0).astype(int)
        return int(out[cy, cx][2])

    assert _center_val(0.0) > _center_val(0.35) > _center_val(0.60)
    assert _center_val(0.60) == 30


def test_render_glow_does_no_blur_work_after_flash_ends(monkeypatch):
    import cv2
    from live_glow import Track, render_glow

    K, pose, W2T = _setup()
    H = table_to_image_homography(K, pose, W2T)
    frame = np.full((1080, 1440, 3), 30, dtype=np.uint8)
    poly = np.array([
        [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02],
    ])
    track = Track(
        0, poly, np.array([0.0, 0.0]), 0.9, 0.0, 0.0,
        activation_time=0.0,
    )

    def unexpected_blur(*args, **kwargs):
        raise AssertionError("invisible tracks must not be blurred")

    monkeypatch.setattr(cv2, "GaussianBlur", unexpected_blur)

    out = render_glow(frame, [track], H, now=0.60)

    np.testing.assert_array_equal(out, frame)


def test_render_glow_limits_blur_work_to_piece_region(monkeypatch):
    """A small piece must not trigger a full-frame blur for every track."""
    import cv2
    from live_glow import Track, render_glow

    K, pose, W2T = _setup()
    H = table_to_image_homography(K, pose, W2T)
    frame = np.full((1080, 1440, 3), 30, dtype=np.uint8)
    poly = np.array([
        [-0.02, -0.02], [0.02, -0.02], [0.02, 0.02], [-0.02, 0.02],
    ])
    track = Track(
        0, poly, np.array([0.0, 0.0]), 0.9, 0.0, 0.0,
        activation_time=0.0,
    )
    blur_shapes = []
    real_blur = cv2.GaussianBlur

    def recording_blur(image, *args, **kwargs):
        blur_shapes.append(image.shape)
        return real_blur(image, *args, **kwargs)

    monkeypatch.setattr(cv2, "GaussianBlur", recording_blur)
    render_glow(frame, [track], H, now=0.05)

    assert blur_shapes
    assert max(height * width for height, width in blur_shapes) < frame.size // 12
