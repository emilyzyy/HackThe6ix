import json

import numpy as np
import pytest

from session_io import SessionWriter, SessionReader, format_summary
from transforms import pose_to_mat


def _fake_frame(i):
    rgb = np.full((48, 64, 3), i * 10 % 255, dtype=np.uint8)
    depth = np.full((12, 16), 0.5 + i * 0.1, dtype=np.float32)
    coeffs = {"fx": 700.0, "fy": 710.0, "cx": 32.0, "cy": 24.0}
    pose = {"qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0,
            "tx": float(i), "ty": 0.2, "tz": 0.3}
    return rgb, depth, coeffs, pose


def test_writer_reader_round_trip(tmp_path):
    w = SessionWriter(tmp_path / "sess")
    for i in range(3):
        rgb, depth, coeffs, pose = _fake_frame(i)
        rec = w.add_frame(rgb, depth, coeffs, pose, device_type=1,
                          timestamp=100.0 + i * 0.2)
        assert rec.frame_id == i
    summary = w.close()

    r = SessionReader(w.session_dir)
    frames = r.frames()
    assert len(frames) == 3
    for i, rec in enumerate(frames):
        rgb, depth, coeffs, pose = _fake_frame(i)
        assert rec.frame_id == i
        assert rec.timestamp == pytest.approx(100.0 + i * 0.2)
        assert rec.intrinsics == coeffs
        assert rec.pose_qt == pose
        assert rec.device_type == 1
        np.testing.assert_allclose(rec.pose_mat, pose_to_mat(**pose))
        np.testing.assert_array_equal(rec.load_depth(), depth)
        loaded_rgb = rec.load_rgb()
        assert loaded_rgb.shape == rgb.shape
        # JPEG is lossy; a flat image should survive nearly exactly.
        assert np.abs(loaded_rgb.astype(int) - rgb.astype(int)).max() <= 3
        assert rec.rgb_size == [rgb.shape[0], rgb.shape[1]]
        assert rec.depth_size == [depth.shape[0], depth.shape[1]]

    assert summary["frames"] == 3
    assert summary["duration_s"] == pytest.approx(0.4)
    assert summary["bytes"] > 0


def test_jsonl_is_line_valid_json(tmp_path):
    w = SessionWriter(tmp_path / "sess")
    rgb, depth, coeffs, pose = _fake_frame(0)
    w.add_frame(rgb, depth, coeffs, pose, device_type=1, timestamp=1.0)
    w.close()
    lines = (w.session_dir / "frames.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["frame_id"] == 0
    assert len(rec["pose_mat"]) == 4 and len(rec["pose_mat"][0]) == 4
    # Paths in the index are relative to the session dir.
    assert not rec["rgb"].startswith("/")
    assert (w.session_dir / rec["rgb"]).exists()
    assert (w.session_dir / rec["depth"]).exists()


def test_empty_session_summary(tmp_path):
    w = SessionWriter(tmp_path / "sess")
    summary = w.close()
    assert summary["frames"] == 0
    assert summary["duration_s"] == 0.0


def test_format_summary_mentions_key_numbers():
    text = format_summary({"frames": 42, "duration_s": 7.0, "bytes": 3 * 1024 * 1024})
    assert "42" in text and "7.0" in text and "3.0 MB" in text
