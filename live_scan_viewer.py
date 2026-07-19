"""Live scan viewer (P3): full-screen phone feed with segmentation masks that
glow the moment a piece is discovered and stay attached as the camera pans.

Data flow (identical for live phone and recorded replay):

    frame + pose  ->  async sampled YOLO  ->  back-project mask to table plane
                  ->  GlowTracker (one table-space track per piece)
    every UI tick ->  project active tracks into the current frame -> glow

The UI never blocks on YOLO: a worker thread runs detection on the latest
frame only (stale frames dropped) and updates the shared tracker; the main
loop renders the glow every frame from the projected tracks. YOLO is injected
as a detector callable, so the engine has no hard dependency on the model.

Usage:
  # Recorded real scan, headless -> writes an mp4 preview (no phone needed):
  python live_scan_viewer.py sessions/<ts> --headless --out preview.mp4
  # Live phone (Record3D over USB), on-screen window:
  python live_scan_viewer.py --live --world-to-table sessions/<ts>
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from live_glow import GlowTracker, back_project_polygon, render_glow, \
    table_to_image_homography
from plane import compute_table_frame, load_table_frame
from session_io import SessionReader
from transforms import intrinsics_to_K

MIN_GLOW_CONFIDENCE = 0.55
MIN_MASK_AREA_PX = 400


# --------------------------------------------------------------------------
# Detectors (injected). The engine only needs: image -> [(polygon_px, conf)].
# --------------------------------------------------------------------------
class YoloMaskDetector:
    """Wrap lego-cv's YoloSegModel to return image-space mask polygons."""

    def __init__(self, weights, lego_cv_dir, device="mps", confidence=0.25):
        import sys
        sys.path.insert(0, str(lego_cv_dir))
        from pipeline.segdetect import YoloSegModel

        self.model = YoloSegModel(weights=weights, device=device,
                                  confidence=confidence)

    def __call__(self, image_bgr):
        results = []
        for raw in self.model.predict(image_bgr):
            mask = np.asarray(raw.mask, dtype=np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(contour) < MIN_MASK_AREA_PX:
                continue
            results.append((contour.reshape(-1, 2).astype(np.float64),
                            float(raw.confidence)))
        return results


# --------------------------------------------------------------------------
# Frame sources. Each yields dicts: frame_id, image (bgr), pose_mat, K, ts.
# --------------------------------------------------------------------------
class RecordedFrameSource:
    """Replay a recorded session's frames in capture order at real cadence."""

    def __init__(self, session_dir, realtime=True, speed=1.0):
        self.session_dir = Path(session_dir)
        self.records = SessionReader(self.session_dir).frames()
        self.realtime = realtime
        self.speed = speed
        self.world_to_table = _load_world_to_table(self.session_dir)

    def frames(self):
        start_wall = time.monotonic()
        start_ts = self.records[0].timestamp if self.records else 0.0
        for record in self.records:
            if self.realtime:
                target = (record.timestamp - start_ts) / max(1e-6, self.speed)
                sleep = target - (time.monotonic() - start_wall)
                if sleep > 0:
                    time.sleep(sleep)
            yield {
                "frame_id": record.frame_id,
                "image": record.load_rgb(),
                "pose_mat": record.pose_mat,
                "K": intrinsics_to_K(**record.intrinsics),
                "timestamp": record.timestamp,
            }


class LiveRecord3DSource:
    """Live phone frames via Record3D. Requires a world_to_table (a prior
    session's plane fit) so masks can be anchored while scanning."""

    def __init__(self, world_to_table, dev_idx=0):
        from record3d import Record3DStream

        self.world_to_table = np.asarray(world_to_table, dtype=float)
        self._latest = None
        self._lock = threading.Lock()
        self._event = threading.Event()
        self.stopped = threading.Event()
        devs = Record3DStream.get_connected_devices()
        if len(devs) <= dev_idx:
            raise RuntimeError(
                f"{len(devs)} Record3D device(s); cannot use index {dev_idx}"
            )
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = lambda: self.stopped.set()
        self.session.connect(devs[dev_idx])

    def _on_new_frame(self):
        rgb = np.asarray(self.session.get_rgb_frame())
        pose = self.session.get_camera_pose()
        coeffs = self.session.get_intrinsic_mat()
        import transforms
        pose_mat = transforms.pose_to_mat(
            pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz
        )
        frame = {
            "frame_id": int(time.monotonic() * 1000),
            "image": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            "pose_mat": pose_mat,
            "K": intrinsics_to_K(fx=coeffs.fx, fy=coeffs.fy,
                                 cx=coeffs.tx, cy=coeffs.ty),
            "timestamp": time.monotonic(),
        }
        with self._lock:
            self._latest = frame
        self._event.set()

    def frames(self):
        while not self.stopped.is_set():
            self._event.wait(0.1)
            self._event.clear()
            with self._lock:
                frame = self._latest
            if frame is not None:
                yield frame


def _load_world_to_table(session_dir):
    try:
        table = load_table_frame(session_dir)
    except FileNotFoundError:
        table = compute_table_frame(session_dir)
        table["world_to_table"] = np.asarray(table["world_to_table"])
    return np.asarray(table["world_to_table"], dtype=float)


# --------------------------------------------------------------------------
# The live session: shared tracker updated async, rendered every UI frame.
# --------------------------------------------------------------------------
class LiveGlowSession:
    def __init__(self, detector, world_to_table, *,
                 min_confidence=MIN_GLOW_CONFIDENCE, **tracker_kwargs):
        self.detector = detector
        self.world_to_table = np.asarray(world_to_table, dtype=float)
        self.tracker = GlowTracker(min_confidence=min_confidence,
                                   **tracker_kwargs)
        self._lock = threading.Lock()

    def detect(self, frame, now):
        """Run YOLO on one frame and fold masks into table-space tracks."""
        homography = table_to_image_homography(
            frame["K"], frame["pose_mat"], self.world_to_table
        )
        camera_table = (
            self.world_to_table @ np.append(frame["pose_mat"][:3, 3], 1.0)
        )[:2]
        detections = []
        for polygon_px, confidence in self.detector(frame["image"]):
            table_poly = back_project_polygon(polygon_px, homography)
            detections.append({
                "table_polygon": table_poly,
                "table_centroid": table_poly.mean(axis=0),
                "confidence": confidence,
            })
        with self._lock:
            self.tracker.update(detections, now, camera_xy=tuple(camera_table))

    def render(self, frame, now):
        homography = table_to_image_homography(
            frame["K"], frame["pose_mat"], self.world_to_table
        )
        with self._lock:
            active = list(self.tracker.active_tracks(now))
        out = render_glow(frame["image"], active, homography, now)
        return out, len(active)


def _overlay_hud(image, piece_count, extra=""):
    banner = image.copy()
    cv2.rectangle(banner, (0, 0), (image.shape[1], 54), (0, 0, 0), -1)
    image = cv2.addWeighted(banner, 0.45, image, 0.55, 0)
    cv2.putText(image, f"pieces discovered: {piece_count}   {extra}",
                (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 230, 120), 2,
                cv2.LINE_AA)
    return image


def run_recorded(session_dir, detector, *, headless, out_path=None,
                 detect_every=4, speed=4.0, tracker_kwargs=None):
    """Drive the live pipeline over a recorded scan. Detection runs on sampled
    frames (async in the live path); rendering runs every frame."""
    source = RecordedFrameSource(session_dir, realtime=not headless,
                                 speed=speed)
    session = LiveGlowSession(detector, source.world_to_table,
                              **(tracker_kwargs or {}))
    writer = None
    wall0 = time.monotonic()

    # Async detection worker fed from the latest frame (drops stale frames).
    # Used for the on-screen path (rendering must never wait on YOLO). The
    # headless preview instead samples detection deterministically inline, so
    # the written frames reproduce the same detect/render interleaving without
    # a second redundant YOLO pass.
    latest = {"frame": None}
    lock = threading.Lock()
    stop = threading.Event()

    def worker():
        seen = None
        while not stop.is_set():
            with lock:
                frame = latest["frame"]
            if frame is None or frame["frame_id"] == seen:
                time.sleep(0.005)
                continue
            seen = frame["frame_id"]
            session.detect(frame, time.monotonic() - wall0)

    if not headless:
        threading.Thread(target=worker, daemon=True).start()

    max_pieces = 0
    for index, frame in enumerate(source.frames()):
        if headless and index % detect_every == 0:
            session.detect(frame, time.monotonic() - wall0)
        with lock:
            latest["frame"] = frame
        now = time.monotonic() - wall0
        rendered, count = session.render(frame, now)
        max_pieces = max(max_pieces, count)
        rendered = _overlay_hud(rendered, count,
                                extra=f"f{frame['frame_id']}")
        if headless:
            if writer is None and out_path is not None:
                h, w = rendered.shape[:2]
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h)
                )
            if writer is not None:
                writer.write(rendered)
        else:
            cv2.imshow("live scan", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    stop.set()
    if writer is not None:
        writer.release()
    if not headless:
        cv2.destroyAllWindows()
    return max_pieces


def run_live(world_to_table, detector, *, detect_every_s=0.0,
             tracker_kwargs=None):
    source = LiveRecord3DSource(world_to_table)
    session = LiveGlowSession(detector, world_to_table,
                              **(tracker_kwargs or {}))
    wall0 = time.monotonic()
    latest = {"frame": None}
    lock = threading.Lock()
    stop = threading.Event()

    def worker():
        seen = None
        while not stop.is_set():
            with lock:
                frame = latest["frame"]
            if frame is None or frame["frame_id"] == seen:
                time.sleep(0.005)
                continue
            seen = frame["frame_id"]
            session.detect(frame, time.monotonic() - wall0)

    threading.Thread(target=worker, daemon=True).start()
    print("Live scan: press q to stop.")
    try:
        for frame in source.frames():
            with lock:
                latest["frame"] = frame
            now = time.monotonic() - wall0
            rendered, count = session.render(frame, now)
            rendered = _overlay_hud(rendered, count)
            cv2.imshow("live scan", rendered)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        stop.set()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?",
                        help="recorded session dir (replay/headless)")
    parser.add_argument("--live", action="store_true",
                        help="use the connected Record3D phone")
    parser.add_argument("--world-to-table",
                        help="session dir whose table_frame.json to reuse for "
                             "live anchoring")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--out", type=Path, help="mp4 preview path (headless)")
    parser.add_argument("--lego-cv", type=Path,
                        default=Path("/Users/emily/lego-cv"))
    parser.add_argument("--weights", type=Path,
                        default=Path("/Users/emily/lego-cv/models/lego_seg.pt"))
    parser.add_argument("--detect-every", type=int, default=4)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument("--stagger", type=float, default=0.35)
    args = parser.parse_args()

    detector = YoloMaskDetector(args.weights, args.lego_cv)
    tracker_kwargs = {"activation_stagger_s": args.stagger}

    if args.live:
        world = _load_world_to_table(args.world_to_table or args.session)
        run_live(world, detector, tracker_kwargs=tracker_kwargs)
    else:
        run_recorded(args.session, detector, headless=args.headless,
                     out_path=args.out, detect_every=args.detect_every,
                     speed=args.speed, tracker_kwargs=tracker_kwargs)


if __name__ == "__main__":
    main()
