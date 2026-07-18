import numpy as np
import pytest

from capture import FrameThrottler, CaptureController
from session_io import SessionReader


def test_throttler_first_frame_retained():
    t = FrameThrottler(target_fps=6)
    assert t.should_retain(100.0)


def test_throttler_caps_rate():
    t = FrameThrottler(target_fps=6)
    # 30 fps input for 2 seconds.
    times = [i / 30 for i in range(60)]
    kept = [ts for ts in times if t.should_retain(ts)]
    # Never exceeds target rate, but stays close to it.
    assert 10 <= len(kept) <= 12
    diffs = np.diff(kept)
    assert (diffs >= 1 / 6 - 1e-9).all()


def test_throttler_passes_slow_input_through():
    t = FrameThrottler(target_fps=10)
    times = [i / 3 for i in range(9)]  # 3 fps input
    assert all(t.should_retain(ts) for ts in times)


def _snapshot(i, t):
    """Fake stream snapshot with the frame index baked into pose AND pixels."""
    return {
        "rgb": np.full((24, 32, 3), i, dtype=np.uint8),  # RGB order, flat
        "depth": np.full((6, 8), 0.5, dtype=np.float32),
        "confidence": np.full((6, 8), 2, dtype=np.uint8),
        "coeffs": {"fx": 700.0, "fy": 700.0, "cx": 16.0, "cy": 12.0},
        "pose": {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
                 "tx": float(i), "ty": 0.0, "tz": 0.0},
        "device_type": 1,
        "timestamp": t,
    }


def test_controller_pose_image_sync_and_throttle(tmp_path):
    # Large queue: tests submit instantly, unlike a real 30 fps stream.
    ctl = CaptureController(tmp_path / "sess", target_fps=6, queue_size=64)
    # 60 frames at a simulated 30 fps, called like the stream callback would.
    for i in range(60):
        t = i / 30
        if ctl.offer(t):
            ctl.submit(_snapshot(i, t))
    summary = ctl.stop()

    frames = SessionReader(tmp_path / "sess").frames()
    assert 10 <= len(frames) <= 12
    assert summary["frames"] == len(frames)
    for rec in frames:
        img = rec.load_rgb()
        baked = int(round(float(np.median(img))))
        # The pose stored with this frame must describe this exact frame.
        assert baked == int(rec.pose_qt["tx"])
        np.testing.assert_array_equal(
            rec.load_confidence(), np.full((6, 8), 2, dtype=np.uint8)
        )
    # frame_ids are sequential regardless of source frame index.
    assert [f.frame_id for f in frames] == list(range(len(frames)))


def test_controller_stats(tmp_path):
    ctl = CaptureController(tmp_path / "sess", target_fps=30, queue_size=64)
    for i in range(10):
        t = i / 30
        if ctl.offer(t):
            ctl.submit(_snapshot(i, t))
    ctl.stop()
    stats = ctl.stats()
    assert stats["frames"] == 10
    assert stats["bytes"] > 0
    assert stats["dropped"] == 0


def test_controller_full_queue_drops_instead_of_blocking(tmp_path):
    ctl = CaptureController(tmp_path / "sess", target_fps=1000, queue_size=2)
    for i in range(50):
        if ctl.offer(i / 1000):
            ctl.submit(_snapshot(i, i / 1000))
    summary = ctl.stop()
    stats = ctl.stats()
    assert stats["dropped"] > 0
    assert summary["frames"] + stats["dropped"] == 50


def test_controller_stop_is_idempotent(tmp_path):
    ctl = CaptureController(tmp_path / "sess", target_fps=6)
    s1 = ctl.stop()
    s2 = ctl.stop()
    assert s1["frames"] == s2["frames"] == 0


def test_no_frames_warning_fires_only_when_stalled():
    from capture import no_frames_warning
    assert no_frames_warning(2.0, 0) is None          # too early to tell
    assert no_frames_warning(30.0, 12) is None        # frames flowing
    warning = no_frames_warning(6.0, 0)
    assert warning is not None
    assert "another" in warning and "Record3D" in warning


def test_copy_confidence_frame_handles_absent_empty_and_populated_getters():
    from capture import _copy_confidence_frame

    class Missing:
        pass

    class Empty:
        def get_confidence_frame(self):
            return np.array([], dtype=np.uint8)

    class NoneFrame:
        def get_confidence_frame(self):
            return None

    class Malformed:
        def get_confidence_frame(self):
            return np.array([object()], dtype=object)

    class Unsupported:
        def get_confidence_frame(self):
            raise RuntimeError("confidence is unavailable on this device")

    class Populated:
        def get_confidence_frame(self):
            return np.full((2, 3), 2, dtype=np.uint8)

    assert _copy_confidence_frame(Missing()) is None
    assert _copy_confidence_frame(Empty()) is None
    assert _copy_confidence_frame(NoneFrame()) is None
    assert _copy_confidence_frame(Malformed()) is None
    assert _copy_confidence_frame(Unsupported()) is None
    np.testing.assert_array_equal(
        _copy_confidence_frame(Populated()),
        np.full((2, 3), 2, dtype=np.uint8),
    )
