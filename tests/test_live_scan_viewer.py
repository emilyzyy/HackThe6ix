"""Runner integration: detect -> back-project -> track -> render plumbing."""
from __future__ import annotations

import numpy as np

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


class _FixedImageDetector:
    """Returns the same image-space mask every call (a stationary camera sees
    a piece in a fixed pixel region)."""

    def __init__(self, polygon_px, confidence=0.9):
        self.polygon_px = np.asarray(polygon_px, dtype=np.float64)
        self.confidence = confidence

    def __call__(self, image_bgr):
        return [(self.polygon_px, self.confidence)]


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
    assert rendered[cy, cx][0] > 40 + 30  # shifted toward glow hue
    assert tuple(rendered[3, 3]) == (40, 40, 40)  # corner untouched


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
    assert count_later == 2
