"""Live-glow engine: table-anchored tracks that stay attached across camera
motion, activate once, and animate independently of the detection cadence."""
from __future__ import annotations

import numpy as np

from live_glow import (
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


def test_glow_envelope_pulses_then_settles():
    # At activation: a bright pulse. Later: a steady, subtler highlight.
    at_start = glow_envelope(age_s=0.0)
    mid_pulse = glow_envelope(age_s=0.15)
    settled = glow_envelope(age_s=2.0)
    assert at_start["phase"] == "pulse"
    assert settled["phase"] == "steady"
    # Peak pulse fill is clearly brighter than the settled state.
    assert max(at_start["fill_alpha"], mid_pulse["fill_alpha"]) > \
        settled["fill_alpha"] + 0.15
    # Steady state is a real, persistent (non-zero) highlight.
    assert 0.0 < settled["fill_alpha"] < max(
        at_start["fill_alpha"], mid_pulse["fill_alpha"]
    )
    # Edge tracing is brightest during the discovery pulse.
    assert at_start["edge_alpha"] >= settled["edge_alpha"]


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
    # Centre pixel shifted toward the glow hue (blue channel up from 30).
    assert out[cy, cx][0] > frame[cy, cx][0] + 40
    # A far corner is untouched.
    assert tuple(out[5, 5]) == (30, 30, 30)


def test_render_glow_fresh_brighter_than_settled():
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
        return int(out[cy, cx][0])

    assert _center_val(0.0) > _center_val(2.0)
