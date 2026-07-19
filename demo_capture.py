"""Shared frame bridge for recording and the full-rate live demo preview."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np

from transforms import intrinsics_to_K, pose_to_mat


def snapshot_to_live_frame(snapshot: dict, frame_id: int) -> dict:
    """Convert one synchronized CaptureController snapshot for live rendering."""
    pose = snapshot["pose"]
    return {
        "frame_id": int(frame_id),
        "image": cv2.cvtColor(np.asarray(snapshot["rgb"]), cv2.COLOR_RGB2BGR),
        "depth": np.asarray(snapshot["depth"]),
        "pose_mat": pose_to_mat(
            pose["qx"], pose["qy"], pose["qz"], pose["qw"],
            pose["tx"], pose["ty"], pose["tz"],
        ),
        "K": intrinsics_to_K(**snapshot["coeffs"]),
        "timestamp": float(snapshot["timestamp"]),
    }


class LatestPreviewFrame:
    """Thread-safe latest-only buffer fed by the Record3D callback."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._next_id = 0

    def offer(self, snapshot: dict) -> None:
        with self._lock:
            frame_id = self._next_id
            self._next_id += 1
            self._frame = snapshot_to_live_frame(snapshot, frame_id)

    def get(self) -> dict | None:
        with self._lock:
            return self._frame


def run_demo_capture(
    anchor_dir: Path,
    session_dir: Path,
    detector,
    *,
    target_fps: float = 6.0,
) -> Path:
    """Record and render one auto-completing scan through one phone client."""
    from capture import CaptureController, Record3DSource
    from live_scan_viewer import (
        CoveragePresentation,
        LatestFrameScheduler,
        LiveCoverageTracker,
        LiveDiagnostics,
        LiveGlowSession,
        PassDiagnostics,
        _load_live_workspace,
        _overlay_hud,
        _render_scanner_sweep,
    )

    session_dir = Path(session_dir)
    world, contours, reference_grid = _load_live_workspace(anchor_dir)
    controller = CaptureController(session_dir, target_fps=target_fps)
    latest = LatestPreviewFrame()
    source = Record3DSource(controller, preview_listener=latest.offer)
    wall0 = time.monotonic()
    glow = LiveGlowSession(
        detector,
        world,
        workspace_contours_table=contours,
        clock=lambda: time.monotonic() - wall0,
    )
    coverage = LiveCoverageTracker(reference_grid, world)
    presentation = CoveragePresentation()
    diagnostics = LiveDiagnostics()
    pass_diagnostics = PassDiagnostics(session_dir / "live-diagnostics.jsonl")
    scheduler = LatestFrameScheduler()
    stop = threading.Event()

    def detector_worker():
        seen = None
        while not stop.is_set():
            frame = latest.get()
            if frame is None or frame["frame_id"] == seen:
                time.sleep(0.005)
                continue
            seen = frame["frame_id"]
            now = time.monotonic() - wall0
            if scheduler.should_detect(frame, now):
                pass_diagnostics.record(glow.detect(frame, now))
                diagnostics.note_segmentation_pass()

    def coverage_worker():
        seen = None
        while not stop.is_set():
            frame = latest.get()
            if frame is None or frame["frame_id"] == seen:
                time.sleep(0.005)
                continue
            seen = frame["frame_id"]
            coverage.update(frame, time.monotonic() - wall0)

    threading.Thread(target=detector_worker, daemon=True).start()
    threading.Thread(target=coverage_worker, daemon=True).start()
    print(f"Recording demo session to {session_dir}")
    last_frame_id = None
    try:
        while not source.stopped.is_set():
            if source.preview_error is not None:
                raise RuntimeError("live preview callback failed") from source.preview_error
            frame = latest.get()
            if frame is None or frame["frame_id"] == last_frame_id:
                time.sleep(0.003)
                continue
            last_frame_id = frame["frame_id"]
            diagnostics.note_phone_frame()
            now = time.monotonic() - wall0
            rendered, count = glow.render(frame, now)
            rendered = _render_scanner_sweep(rendered, now)
            state = diagnostics.snapshot()
            fraction = presentation.update(coverage.fraction(), now)
            rendered = _overlay_hud(
                rendered,
                count,
                coverage_fraction=fraction,
                phone_frames=state["phone_frames"],
                segmentation_passes=state["segmentation_passes"],
            )
            cv2.imshow("LEGO Scanner", rendered)
            key = cv2.waitKey(1) & 0xFF
            if presentation.complete:
                cv2.waitKey(350)
                break
            if key == ord("q"):
                break
    finally:
        stop.set()
        try:
            source.disconnect()
        finally:
            controller.stop()
        cv2.destroyAllWindows()
    return session_dir
