"""Phase 1: record a session from the record3d USB stream.

Frame retention path (the sync-critical part):
  Record3DSource.on_new_frame (stream callback thread)
    -> controller.offer(t)           throttle decision, no copies yet
    -> snapshot ALL buffers          rgb/depth/pose/intrinsics at one instant
    -> controller.submit(snapshot)   bounded queue
    -> writer thread                 JPEG/npy encode + frames.jsonl append

Usage: python capture.py [--fps 6] [--out sessions] [--dev 0] [--live-view]
Press Enter to stop. Status line every 2 s; no per-frame printing.
"""

import argparse
import queue
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from session_io import SessionWriter, format_summary

DEVICE_TYPE_TRUEDEPTH = 0


class FrameThrottler:
    """Rate-limit retention to target_fps based on frame timestamps."""

    def __init__(self, target_fps):
        self.interval = 1.0 / target_fps
        self._last = None

    def should_retain(self, t):
        if self._last is None or t - self._last >= self.interval - 1e-9:
            self._last = t
            return True
        return False


class CaptureController:
    """Stream-agnostic capture core: throttle -> queue -> writer thread."""

    def __init__(self, session_dir, target_fps=6, queue_size=8):
        self.writer = SessionWriter(session_dir)
        self.session_dir = self.writer.session_dir
        self._throttler = FrameThrottler(target_fps)
        self._queue = queue.Queue(maxsize=queue_size)
        self._dropped = 0
        self._summary = None
        self._frame_listeners = []
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    def add_frame_listener(self, fn):
        """fn(FrameRecord, snapshot) called on the writer thread per retained frame."""
        self._frame_listeners.append(fn)

    def offer(self, t):
        """Throttle gate. Call before copying any buffers."""
        return self._throttler.should_retain(t)

    def submit(self, snapshot):
        """Queue an already-copied snapshot dict. Never blocks the stream thread."""
        try:
            self._queue.put_nowait(snapshot)
        except queue.Full:
            self._dropped += 1

    def _write_loop(self):
        while True:
            snap = self._queue.get()
            if snap is None:
                return
            bgr = cv2.cvtColor(snap["rgb"], cv2.COLOR_RGB2BGR)
            rec = self.writer.add_frame(
                bgr, snap["depth"], snap["coeffs"], snap["pose"],
                device_type=snap["device_type"], timestamp=snap["timestamp"])
            for fn in self._frame_listeners:
                fn(rec, snap)

    def stats(self):
        return {"frames": self.writer.frame_count,
                "bytes": self.writer.bytes_written,
                "dropped": self._dropped}

    def stop(self):
        if self._summary is None:
            self._queue.put(None)
            self._writer_thread.join(timeout=30)
            self._summary = self.writer.close()
        return self._summary


class Record3DSource:
    """Owns the Record3DStream and feeds snapshots to a CaptureController."""

    def __init__(self, controller, dev_idx=0):
        from record3d import Record3DStream  # imported here: tests run without it

        self.controller = controller
        self.stopped = threading.Event()
        devs = Record3DStream.get_connected_devices()
        if len(devs) <= dev_idx:
            raise RuntimeError(
                f"{len(devs)} device(s) found; cannot use index {dev_idx}. "
                "Is the iPhone connected via USB with Record3D streaming?")
        self.session = Record3DStream()
        self.session.on_new_frame = self._on_new_frame
        self.session.on_stream_stopped = self._on_stream_stopped
        self.session.connect(devs[dev_idx])

    def _on_new_frame(self):
        t = time.monotonic()
        if not self.controller.offer(t):
            return
        # Snapshot everything at one instant, on this thread, before the
        # stream overwrites its buffers — pose and pixels must match exactly.
        rgb = np.asarray(self.session.get_rgb_frame()).copy()
        depth = np.asarray(self.session.get_depth_frame()).copy()
        pose = self.session.get_camera_pose()
        coeffs = self.session.get_intrinsic_mat()
        device_type = self.session.get_device_type()
        if device_type == DEVICE_TYPE_TRUEDEPTH:
            rgb = cv2.flip(rgb, 1)
            depth = cv2.flip(depth, 1)
        self.controller.submit({
            "rgb": rgb,
            "depth": depth,
            "coeffs": {"fx": coeffs.fx, "fy": coeffs.fy,
                       "cx": coeffs.tx, "cy": coeffs.ty},
            "pose": {"qx": pose.qx, "qy": pose.qy, "qz": pose.qz, "qw": pose.qw,
                     "tx": pose.tx, "ty": pose.ty, "tz": pose.tz},
            "device_type": device_type,
            "timestamp": t,
        })

    def _on_stream_stopped(self):
        self.stopped.set()

    def disconnect(self):
        self.session.disconnect()


NO_FRAMES_WARN_AFTER_S = 5.0


def no_frames_warning(elapsed_s, frame_count, warn_after_s=NO_FRAMES_WARN_AFTER_S):
    """Connected-but-silent is a device-side state, not a code path we can fix.

    The Record3D app streams to exactly ONE client: a second client connects
    fine but never receives a frame. Same symptom when the app isn't actively
    streaming (backgrounded, screen locked, or wedged after a client vanished).
    """
    if elapsed_s < warn_after_s or frame_count > 0:
        return None
    return (
        f"\nWARNING: connected but no frames after {elapsed_s:.0f}s.\n"
        "  - Is another client already connected? (demo-main.py or a stuck\n"
        "    capture.py: check `ps aux | grep -E 'demo-main|capture.py'`)\n"
        "  - Is the Record3D app in the foreground with USB streaming active\n"
        "    and the screen unlocked? Toggle streaming off/on if it is.")


def _status_loop(controller, stop_event, start_time):
    warned = False
    while not stop_event.wait(2.0):
        s = controller.stats()
        elapsed = time.monotonic() - start_time
        if not warned:
            warning = no_frames_warning(elapsed, s["frames"])
            if warning:
                warned = True
                print(warning)
        fps = s["frames"] / elapsed if elapsed > 0 else 0.0
        mb = s["bytes"] / (1024 * 1024)
        line = (f"\r[REC {int(elapsed) // 60:02d}:{int(elapsed) % 60:02d}] "
                f"frames={s['frames']} ({fps:.1f} fps) disk={mb:.0f} MB")
        if s["dropped"]:
            line += f" dropped={s['dropped']}"
        print(line, end="", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Record a record3d capture session.")
    ap.add_argument("--fps", type=float, default=6, help="retention rate (default 6)")
    ap.add_argument("--out", default="sessions", help="sessions root dir")
    ap.add_argument("--dev", type=int, default=0, help="device index")
    ap.add_argument("--live-view", action="store_true",
                    help="show live coverage window (Phase 3)")
    args = ap.parse_args()

    session_dir = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    controller = CaptureController(session_dir, target_fps=args.fps)

    live_view = None
    if args.live_view:
        from viewer import LiveCoverageView
        live_view = LiveCoverageView(controller)

    print("Connecting to device...")
    source = Record3DSource(controller, dev_idx=args.dev)
    print(f"Recording to {controller.session_dir} — press Enter to stop.")

    start = time.monotonic()
    status_stop = threading.Event()
    status = threading.Thread(target=_status_loop,
                              args=(controller, status_stop, start), daemon=True)
    status.start()

    try:
        if live_view is not None:
            live_view.run_until(source.stopped)  # renders on main thread; press q to stop
        else:
            _wait_enter_or_stream_stop(source.stopped)
    except KeyboardInterrupt:
        pass

    status_stop.set()
    source.disconnect()
    summary = controller.stop()
    print("\nSession saved:", controller.session_dir)
    print(format_summary(summary))
    if controller.stats()["dropped"]:
        print(f"warning: {controller.stats()['dropped']} frames dropped (writer fell behind)")


def _wait_enter_or_stream_stop(stopped_event):
    waiter = threading.Thread(target=input, daemon=True)
    waiter.start()
    while waiter.is_alive() and not stopped_event.is_set():
        waiter.join(timeout=0.25)
    if stopped_event.is_set():
        print("\nStream stopped by device.")


if __name__ == "__main__":
    main()
