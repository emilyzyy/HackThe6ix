"""Session persistence: frames/ + frames.jsonl index, one line per frame.

Schema per line (paths relative to the session dir; pose_mat is camera-to-world):
  {"frame_id", "timestamp", "rgb", "depth", "rgb_size", "depth_size",
   "intrinsics": {fx, fy, cx, cy}, "pose_qt": {qx..qw, tx..tz},
   "pose_mat": 4x4 nested lists, "device_type", "confidence"?}
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from transforms import pose_to_mat

JPEG_QUALITY = 95


@dataclass
class FrameRecord:
    session_dir: Path
    frame_id: int
    timestamp: float
    rgb: str
    depth: str
    rgb_size: list
    depth_size: list
    intrinsics: dict
    pose_qt: dict
    pose_mat: np.ndarray
    device_type: int
    confidence: str | None = None

    def load_rgb(self):
        """BGR uint8 image (as saved; capture stores BGR-order JPEGs)."""
        img = cv2.imread(str(self.session_dir / self.rgb))
        if img is None:
            raise FileNotFoundError(self.session_dir / self.rgb)
        return img

    def load_depth(self):
        return np.load(self.session_dir / self.depth)

    def load_confidence(self):
        if self.confidence is None:
            return None
        return np.load(self.session_dir / self.confidence)

    @classmethod
    def from_json(cls, session_dir, line):
        d = json.loads(line)
        d.setdefault("confidence", None)
        d["pose_mat"] = np.array(d["pose_mat"])
        return cls(session_dir=Path(session_dir), **d)


class SessionWriter:
    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        (self.session_dir / "frames").mkdir(parents=True, exist_ok=True)
        self._index = open(self.session_dir / "frames.jsonl", "w")
        self._next_id = 0
        self._first_ts = None
        self._last_ts = None
        self._bytes = 0

    def add_frame(
        self, rgb_bgr, depth, coeffs, pose, device_type, timestamp,
        confidence=None,
    ):
        fid = self._next_id
        self._next_id += 1
        rgb_rel = f"frames/{fid:05d}.rgb.jpg"
        depth_rel = f"frames/{fid:05d}.depth.npy"
        confidence_rel = None
        cv2.imwrite(str(self.session_dir / rgb_rel), rgb_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        np.save(self.session_dir / depth_rel, depth)
        if confidence is not None and np.asarray(confidence).size:
            confidence_rel = f"frames/{fid:05d}.confidence.npy"
            np.save(self.session_dir / confidence_rel, confidence)

        rec = FrameRecord(
            session_dir=self.session_dir,
            frame_id=fid,
            timestamp=float(timestamp),
            rgb=rgb_rel,
            depth=depth_rel,
            rgb_size=[int(rgb_bgr.shape[0]), int(rgb_bgr.shape[1])],
            depth_size=[int(depth.shape[0]), int(depth.shape[1])],
            intrinsics={k: float(v) for k, v in coeffs.items()},
            pose_qt={k: float(v) for k, v in pose.items()},
            pose_mat=pose_to_mat(**pose),
            device_type=int(device_type),
            confidence=confidence_rel,
        )
        line = json.dumps({
            "frame_id": rec.frame_id,
            "timestamp": rec.timestamp,
            "rgb": rec.rgb,
            "depth": rec.depth,
            "rgb_size": rec.rgb_size,
            "depth_size": rec.depth_size,
            "intrinsics": rec.intrinsics,
            "pose_qt": rec.pose_qt,
            "pose_mat": rec.pose_mat.tolist(),
            "device_type": rec.device_type,
            "confidence": rec.confidence,
        })
        self._index.write(line + "\n")
        self._index.flush()

        self._bytes += (self.session_dir / rgb_rel).stat().st_size
        self._bytes += (self.session_dir / depth_rel).stat().st_size
        if confidence_rel is not None:
            self._bytes += (
                self.session_dir / confidence_rel
            ).stat().st_size
        if self._first_ts is None:
            self._first_ts = rec.timestamp
        self._last_ts = rec.timestamp
        return rec

    @property
    def frame_count(self):
        return self._next_id

    @property
    def bytes_written(self):
        return self._bytes

    def close(self):
        if not self._index.closed:
            self._index.close()
        duration = 0.0
        if self._first_ts is not None and self._next_id > 1:
            duration = self._last_ts - self._first_ts
        return {"frames": self._next_id, "duration_s": duration, "bytes": self._bytes}


class SessionReader:
    def __init__(self, session_dir):
        self.session_dir = Path(session_dir)
        index = self.session_dir / "frames.jsonl"
        if not index.exists():
            raise FileNotFoundError(f"no frames.jsonl in {self.session_dir}")

    def frames(self):
        lines = (self.session_dir / "frames.jsonl").read_text().splitlines()
        return [FrameRecord.from_json(self.session_dir, ln) for ln in lines if ln.strip()]


def format_summary(summary):
    mb = summary["bytes"] / (1024 * 1024)
    return (f"{summary['frames']} frames, {summary['duration_s']:.1f} s, "
            f"{mb:.1f} MB")
