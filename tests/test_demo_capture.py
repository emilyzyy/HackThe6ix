import numpy as np

from demo_capture import LatestPreviewFrame, snapshot_to_live_frame
from tests.test_capture import _snapshot


def test_snapshot_to_live_frame_preserves_frame_geometry():
    frame = snapshot_to_live_frame(_snapshot(3, 4.0), frame_id=9)

    assert frame["frame_id"] == 9
    assert frame["image"].shape == (24, 32, 3)
    assert frame["depth"].shape == (6, 8)
    assert frame["K"][0, 0] == 700.0
    assert frame["pose_mat"][0, 3] == 3.0


def test_latest_preview_frame_numbers_and_replaces_frames():
    latest = LatestPreviewFrame()

    latest.offer(_snapshot(1, 1.0))
    first = latest.get()
    latest.offer(_snapshot(2, 2.0))
    second = latest.get()

    assert first["frame_id"] == 0
    assert second["frame_id"] == 1
    assert int(np.median(second["image"])) == 2
