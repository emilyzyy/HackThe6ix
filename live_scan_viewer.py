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
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from coverage import CoverageGrid, replay_session
from live_glow import DEFAULT_ACTIVATION_STAGGER_S, FLASH_END_S, \
    GLOW_HUE_BGR, GlowTracker, \
    back_project_polygon, glow_envelope, project_polygon, render_glow, \
    table_to_image_homography
from plane import compute_table_frame, load_table_frame
from session_io import SessionReader
from transforms import intrinsics_to_K, scale_intrinsics

MIN_GLOW_CONFIDENCE = 0.55
MIN_MASK_AREA_PX = 400
WORKSPACE_ROI_PAD_FRACTION = 0.02
MIN_WORKSPACE_OVERLAP = 0.80
MAX_PIECE_DIMENSION_M = 0.16
MAX_PIECE_FOOTPRINT_AREA_M2 = 0.008
DETECTION_INTERVAL_S = 0.50
LIVE_MAX_NEW_TRACKS_PER_PASS = 8
LIVE_NEW_TRACK_SPACING_M = 0.07
LIVE_MASK_SPACING_FRACTION = 0.07
LIVE_TRACK_MEMORY_S = 30.0
LIVE_MAX_DETECTIONS = 96
SCREEN_FLASHES_PER_PASS = 4
SCREEN_FLASH_SPACING_FRACTION = 0.08
SCREEN_FLASH_COOLDOWN_S = 1.5
SCREEN_FLASH_MAX_RESULT_AGE_S = 0.75
SCANNER_SWEEP_PERIOD_S = 2.4
DEMO_COVERAGE_RATE_PER_S = 0.04
DEMO_COVERAGE_COMPLETION_S = 4.0


@dataclass
class ScreenFlash:
    """Short animation-only mask used when the stale workspace leaves view."""

    table_polygon: np.ndarray  # Pixel polygon; rendered through identity H.
    activation_time: float


@dataclass(frozen=True)
class ScreenPieceEvidence:
    """LiDAR-backed plausibility result for an animation-only RGB mask."""

    accepted: bool
    reason: str
    depth_support: float = 0.0
    median_depth_m: float | None = None
    width_m: float | None = None
    length_m: float | None = None
    area_m2: float | None = None


def screen_piece_evidence(
    polygon_px,
    confidence,
    frame,
    *,
    min_confidence=MIN_GLOW_CONFIDENCE,
    min_depth_samples=6,
    min_depth_support=0.25,
    min_depth_m=0.08,
    max_depth_m=1.5,
):
    """Reject full-screen animation masks without live LEGO-scale depth."""
    if float(confidence) < float(min_confidence):
        return ScreenPieceEvidence(False, "low_confidence")
    depth = frame.get("depth")
    if depth is None:
        return ScreenPieceEvidence(False, "depth_unsupported")
    depth = np.asarray(depth, dtype=np.float32)
    polygon = np.asarray(polygon_px, dtype=np.float32).reshape(-1, 2)
    image = np.asarray(frame["image"])
    if (
        len(polygon) < 3
        or depth.ndim != 2
        or not np.isfinite(polygon).all()
    ):
        return ScreenPieceEvidence(False, "depth_unsupported")

    scale = np.array(
        [depth.shape[1] / image.shape[1], depth.shape[0] / image.shape[0]],
        dtype=np.float32,
    )
    polygon_depth = np.rint(polygon * scale).astype(np.int32)
    x0 = max(0, int(polygon_depth[:, 0].min()))
    y0 = max(0, int(polygon_depth[:, 1].min()))
    x1 = min(depth.shape[1], int(polygon_depth[:, 0].max()) + 1)
    y1 = min(depth.shape[0], int(polygon_depth[:, 1].max()) + 1)
    if x0 >= x1 or y0 >= y1:
        return ScreenPieceEvidence(False, "depth_unsupported")
    local_polygon = polygon_depth - np.array([x0, y0], dtype=np.int32)
    region = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(region, [local_polygon], 1)
    polygon_area = int(region.sum())
    if polygon_area == 0:
        return ScreenPieceEvidence(False, "depth_unsupported")
    samples = depth[y0:y1, x0:x1]
    valid = (
        region.astype(bool)
        & np.isfinite(samples)
        & (samples >= float(min_depth_m))
        & (samples <= float(max_depth_m))
    )
    valid_count = int(valid.sum())
    support = valid_count / polygon_area
    if valid_count < int(min_depth_samples) or support < float(min_depth_support):
        return ScreenPieceEvidence(
            False, "depth_unsupported", depth_support=float(support)
        )

    median_depth_m = float(np.median(samples[valid]))
    pixel_width, pixel_length = cv2.minAreaRect(polygon)[1]
    K = np.asarray(frame["K"], dtype=float)
    focal_px = max(1e-6, float(K[0, 0] + K[1, 1]) / 2.0)
    width_m = float(pixel_width) * median_depth_m / focal_px
    length_m = float(pixel_length) * median_depth_m / focal_px
    area_m2 = width_m * length_m
    accepted = bool(
        max(width_m, length_m) <= MAX_PIECE_DIMENSION_M
        and area_m2 <= MAX_PIECE_FOOTPRINT_AREA_M2
    )
    return ScreenPieceEvidence(
        accepted,
        "accepted" if accepted else "oversized",
        depth_support=float(support),
        median_depth_m=median_depth_m,
        width_m=width_m,
        length_m=length_m,
        area_m2=area_m2,
    )


def _visible_glow_tracks(tracks, homography, image_shape, now):
    """Return only tracks that currently draw at least one on-screen pixel."""
    height, width = image_shape[:2]
    visible = []
    for track in tracks:
        if track.activation_time is None:
            continue
        envelope = glow_envelope(float(now) - track.activation_time)
        if (
            envelope["fill_alpha"] <= 0.0
            and envelope["edge_alpha"] <= 0.0
        ):
            continue
        polygon = project_polygon(track.table_polygon, homography)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            continue
        if (
            polygon[:, 0].max() < 0
            or polygon[:, 0].min() >= width
            or polygon[:, 1].max() < 0
            or polygon[:, 1].min() >= height
        ):
            continue
        visible.append(track)
    return visible


class LatestFrameScheduler:
    """Permit the newest unique frame at a fixed wall-clock cadence."""

    def __init__(self, min_interval_s=DETECTION_INTERVAL_S):
        self.min_interval_s = float(min_interval_s)
        self._last_time = None
        self._last_frame_id = None

    def should_detect(self, frame, now):
        frame_id = frame["frame_id"]
        if frame_id == self._last_frame_id:
            return False
        if (
            self._last_time is not None
            and float(now) - self._last_time < self.min_interval_s
        ):
            return False
        self._last_frame_id = frame_id
        self._last_time = float(now)
        return True


class LiveDiagnostics:
    """Thread-safe counters exposed in the live RGB HUD."""

    def __init__(self):
        self._lock = threading.Lock()
        self._phone_frames = 0
        self._segmentation_passes = 0

    def note_phone_frame(self):
        with self._lock:
            self._phone_frames += 1

    def note_segmentation_pass(self):
        with self._lock:
            self._segmentation_passes += 1

    def snapshot(self):
        with self._lock:
            return {
                "phone_frames": self._phone_frames,
                "segmentation_passes": self._segmentation_passes,
            }


class PassDiagnostics:
    """Print and optionally persist one auditable event per completed pass."""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, event):
        line = json.dumps(event, sort_keys=True)
        summary = (
            f"seg f{event.get('frame_id')} "
            f"{event.get('inference_latency_ms', 0):.0f}ms "
            f"raw={event.get('raw_mask_count', 0)} "
            f"table={event.get('visible_table_count', 0)} "
            f"screen={event.get('visible_screen_count', 0)} "
            f"fallback={event.get('fallback_reason', 'not_needed')} "
            f"flash={event.get('fallback_flash_count', 0)}"
        )
        with self._lock:
            print(summary, flush=True)
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(line + "\n")


class LiveCoverageTracker:
    """Live depth coverage measured against a fixed anchor workspace."""

    def __init__(
        self,
        reference_grid,
        world_to_table,
        *,
        max_update_hz=4.0,
        min_seen=2,
    ):
        self.grid = CoverageGrid(
            reference_grid.bounds, cell_m=reference_grid.cell_m
        )
        self.world_to_table = np.asarray(world_to_table, dtype=float)
        self.reference_mask = reference_grid.admissible_workspace_mask(
            min_seen=2, close_cells=5
        )
        if not self.reference_mask.any():
            raise ValueError("anchor session has no reference coverage mask")
        self.min_seen = int(min_seen)
        self.min_interval_s = 1.0 / max(1e-6, float(max_update_hz))
        self._last_update = None
        self._lock = threading.Lock()

    def update(self, frame, now):
        if "depth" not in frame:
            return False
        if (
            self._last_update is not None
            and float(now) - self._last_update < self.min_interval_s
        ):
            return False
        image = frame["image"]
        depth = np.asarray(frame["depth"])
        K_depth = scale_intrinsics(
            frame["K"],
            (image.shape[1], image.shape[0]),
            (depth.shape[1], depth.shape[0]),
        )
        with self._lock:
            self.grid.mark_frame(
                depth,
                K_depth,
                frame["pose_mat"],
                self.world_to_table,
                stride=2,
            )
        self._last_update = float(now)
        return True

    def fraction(self):
        with self._lock:
            observed = self.grid.observed_mask(self.min_seen)
            return float(observed[self.reference_mask].mean())


def _workspace_crop(
    image_bgr, contours_px, *, pad_fraction=WORKSPACE_ROI_PAD_FRACTION
):
    """Black-mask and crop an RGB frame to the projected LiDAR workspace."""
    height, width = image_bgr.shape[:2]
    contours = [
        np.rint(np.asarray(contour, dtype=float)).astype(np.int32)
        for contour in contours_px
        if len(contour) >= 3
    ]
    workspace = np.zeros((height, width), dtype=np.uint8)
    if contours:
        cv2.fillPoly(workspace, contours, 1)
    ys, xs = np.nonzero(workspace)
    if not len(xs):
        raise ValueError("projected LiDAR workspace is outside this frame")
    pad = round(max(width, height) * max(0.0, float(pad_fraction)))
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(width, int(xs.max()) + 1 + pad)
    y1 = min(height, int(ys.max()) + 1 + pad)
    crop = image_bgr[y0:y1, x0:x1].copy()
    crop_workspace = workspace[y0:y1, x0:x1].astype(bool)
    crop[~crop_workspace] = 0
    return crop, (x0, y0), workspace.astype(bool)


def _is_loose_piece(
    table_polygon,
    *,
    max_dimension_m=MAX_PIECE_DIMENSION_M,
    max_area_m2=MAX_PIECE_FOOTPRINT_AREA_M2,
):
    """Reject footprints too large to plausibly be one loose demo piece."""
    polygon = np.asarray(table_polygon, dtype=np.float32).reshape(-1, 2)
    if len(polygon) < 3 or not np.isfinite(polygon).all():
        return False
    width_m, length_m = cv2.minAreaRect(polygon)[1]
    return bool(
        max(width_m, length_m) <= float(max_dimension_m)
        and width_m * length_m <= float(max_area_m2)
    )


def _workspace_overlap(polygon_px, workspace_mask):
    """Fraction of an image polygon supported by the LiDAR workspace."""
    polygon = np.rint(np.asarray(polygon_px, dtype=float)).astype(np.int32)
    if len(polygon) < 3:
        return 0.0
    height, width = workspace_mask.shape
    x0 = max(0, int(polygon[:, 0].min()))
    y0 = max(0, int(polygon[:, 1].min()))
    x1 = min(width, int(polygon[:, 0].max()) + 1)
    y1 = min(height, int(polygon[:, 1].max()) + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    local_polygon = polygon - np.array([x0, y0], dtype=np.int32)
    instance = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(instance, [local_polygon], 1)
    area = int(instance.sum())
    if not area:
        return 0.0
    inside = int((
        instance.astype(bool) & workspace_mask[y0:y1, x0:x1]
    ).sum())
    return inside / area


def _select_spaced_masks(
    detections,
    image_shape,
    *,
    max_count=LIVE_MAX_NEW_TRACKS_PER_PASS,
    min_spacing_fraction=LIVE_MASK_SPACING_FRACTION,
):
    """Choose a small far-apart subset so a crowded pile flashes broadly."""
    candidates = [
        (np.asarray(polygon, dtype=float), float(confidence))
        for polygon, confidence in detections
        if len(polygon) >= 3
    ]
    if not candidates or max_count <= 0:
        return []

    candidates.sort(key=lambda item: item[1], reverse=True)
    diagonal = float(np.hypot(image_shape[0], image_shape[1]))
    min_spacing = max(0.0, float(min_spacing_fraction)) * diagonal
    selected = [candidates.pop(0)]
    selected_centroids = [selected[0][0].mean(axis=0)]

    while candidates and len(selected) < int(max_count):
        ranked = []
        for index, candidate in enumerate(candidates):
            centroid = candidate[0].mean(axis=0)
            nearest = min(
                float(np.linalg.norm(centroid - accepted))
                for accepted in selected_centroids
            )
            if nearest >= min_spacing:
                ranked.append((nearest, candidate[1], index, centroid))
        if not ranked:
            break
        _, _, chosen_index, centroid = max(
            ranked, key=lambda item: (item[0], item[1])
        )
        chosen = candidates.pop(chosen_index)
        selected.append(chosen)
        selected_centroids.append(centroid)
    return selected


class ScreenFlashScheduler:
    """Select a few far-apart masks while rotating recently used regions."""

    def __init__(
        self,
        max_count=SCREEN_FLASHES_PER_PASS,
        min_spacing_fraction=SCREEN_FLASH_SPACING_FRACTION,
        cooldown_s=SCREEN_FLASH_COOLDOWN_S,
    ):
        self.max_count = int(max_count)
        self.min_spacing_fraction = float(min_spacing_fraction)
        self.cooldown_s = float(cooldown_s)
        self._armed_at = {}

    @staticmethod
    def cell(polygon, image_shape):
        height, width = image_shape[:2]
        centroid = np.asarray(polygon, dtype=float).mean(axis=0)
        column = int(np.clip(np.floor(8.0 * centroid[0] / max(1, width)), 0, 7))
        row = int(np.clip(np.floor(6.0 * centroid[1] / max(1, height)), 0, 5))
        return row, column

    def select(self, candidates, image_shape, now, *, guarantee_one=False):
        now = float(now)
        self._armed_at = {
            cell: armed_at
            for cell, armed_at in self._armed_at.items()
            if now - armed_at < self.cooldown_s
        }
        ready = [
            (polygon, confidence)
            for polygon, confidence in candidates
            if self.cell(polygon, image_shape) not in self._armed_at
        ]
        selected = _select_spaced_masks(
            ready,
            image_shape,
            max_count=self.max_count,
            min_spacing_fraction=self.min_spacing_fraction,
        )
        if not selected and guarantee_one and candidates:
            selected = [max(candidates, key=lambda item: float(item[1]))]
        return selected

    def arm(self, selected, image_shape, now):
        for polygon, _ in selected:
            self._armed_at[self.cell(polygon, image_shape)] = float(now)


def _render_scanner_sweep(
    image,
    now,
    *,
    period_s=SCANNER_SWEEP_PERIOD_S,
    hue_bgr=GLOW_HUE_BGR,
):
    """Overlay a cheap cyan neon scan line moving down and back up."""
    out = image.copy()
    height, width = out.shape[:2]
    if not height or not width:
        return out
    phase = (float(now) % max(1e-6, float(period_s))) / max(
        1e-6, float(period_s)
    )
    travel = 2.0 * phase if phase <= 0.5 else 2.0 * (1.0 - phase)
    margin = max(1, int(round(height * 0.05)))
    center_y = int(round(margin + travel * max(0, height - 1 - 2 * margin)))
    radius = max(6, int(round(height * 0.025)))
    y0, y1 = max(0, center_y - radius), min(height, center_y + radius + 1)
    rows = np.arange(y0, y1, dtype=np.float32)
    distance = np.abs(rows - center_y) / max(1.0, float(radius))
    alpha = (0.42 * np.maximum(0.0, 1.0 - distance) ** 2)[:, None, None]
    roi = out[y0:y1].astype(np.float32)
    hue = np.asarray(hue_bgr, dtype=np.float32)[None, None, :]
    out[y0:y1] = np.clip(roi * (1.0 - alpha) + hue * alpha, 0, 255)
    cv2.line(
        out,
        (0, center_y),
        (width - 1, center_y),
        tuple(int(channel) for channel in hue_bgr),
        max(2, int(round(height * 0.0025))),
        cv2.LINE_AA,
    )
    return out


class CoveragePresentation:
    """Truthful workspace coverage plus a deterministic completion flourish."""

    def __init__(
        self,
        rate_per_s=DEMO_COVERAGE_RATE_PER_S,
        completion_s=DEMO_COVERAGE_COMPLETION_S,
    ):
        self.rate_per_s = float(rate_per_s)
        self.completion_s = max(1e-6, float(completion_s))
        self._ramp_started_at = None
        self.complete = False

    def update(self, measured_fraction, elapsed_s):
        measured = min(0.95, float(np.clip(measured_fraction, 0.0, 1.0)))
        animated = min(
            0.95, max(0.0, float(elapsed_s)) * self.rate_per_s
        )
        base = max(measured, animated)
        if base < 0.95:
            return base
        if self._ramp_started_at is None:
            self._ramp_started_at = float(elapsed_s)
        progress = min(
            1.0,
            max(0.0, float(elapsed_s) - self._ramp_started_at)
            / self.completion_s,
        )
        self.complete = progress >= 1.0
        return 0.95 + 0.05 * progress


# --------------------------------------------------------------------------
# Detectors (injected). The engine only needs: image -> [(polygon_px, conf)].
# --------------------------------------------------------------------------
def _add_lego_cv_runtime(lego_cv_dir):
    """Expose lego-cv's model dependencies to the capture Python process."""
    root = Path(lego_cv_dir)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        root / ".venv" / "lib" / version / "site-packages",
        root / ".venv" / "Lib" / "site-packages",
    ]
    runtime = next((path for path in candidates if path.is_dir()), None)
    if runtime is not None and str(runtime) not in sys.path:
        # Keep the lego-cv source tree first, then its matching third-party
        # runtime, while leaving lego-capture's Record3D packages available.
        sys.path.insert(1, str(runtime))
    return runtime


class YoloMaskDetector:
    """Wrap lego-cv's YoloSegModel to return image-space mask polygons."""

    def __init__(self, weights, lego_cv_dir, device="mps", confidence=0.25):
        sys.path.insert(0, str(lego_cv_dir))
        _add_lego_cv_runtime(lego_cv_dir)
        from pipeline.segdetect import YoloSegModel

        self.model = YoloSegModel(weights=weights, device=device,
                                  confidence=confidence, imgsz=640,
                                  retina_masks=False,
                                  max_det=LIVE_MAX_DETECTIONS)

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
                "depth": record.load_depth(),
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
        depth = np.asarray(self.session.get_depth_frame()).copy()
        pose = self.session.get_camera_pose()
        coeffs = self.session.get_intrinsic_mat()
        now = time.monotonic()
        import transforms
        pose_mat = transforms.pose_to_mat(
            pose.qx, pose.qy, pose.qz, pose.qw, pose.tx, pose.ty, pose.tz
        )
        frame = {
            "frame_id": int(now * 1_000_000),
            "image": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            "depth": depth,
            "pose_mat": pose_mat,
            "K": intrinsics_to_K(fx=coeffs.fx, fy=coeffs.fy,
                                 cx=coeffs.tx, cy=coeffs.ty),
            "timestamp": now,
        }
        with self._lock:
            self._latest = frame
        self._event.set()

    def frames(self):
        while not self.stopped.is_set():
            if not self._event.wait(0.1):
                continue
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


def _load_live_workspace(session_dir):
    """Replay recorded LiDAR into the hard table contour used by live YOLO."""
    grid, table = replay_session(session_dir, cell_m=0.003, stride=2)
    contours = grid.admissible_workspace_contours_xy(
        min_seen=2, close_cells=5
    )
    if not contours:
        raise ValueError("anchor session has no LiDAR-observed workspace")
    return np.asarray(table["world_to_table"], dtype=float), contours, grid


# --------------------------------------------------------------------------
# The live session: shared tracker updated async, rendered every UI frame.
# --------------------------------------------------------------------------
class LiveGlowSession:
    def __init__(self, detector, world_to_table, *,
                 min_confidence=MIN_GLOW_CONFIDENCE,
                 workspace_contours_table=None,
                 min_workspace_overlap=MIN_WORKSPACE_OVERLAP,
                 clock=None,
                 **tracker_kwargs):
        self.detector = detector
        self.world_to_table = np.asarray(world_to_table, dtype=float)
        self.workspace_contours_table = tuple(
            np.asarray(contour, dtype=float)
            for contour in (workspace_contours_table or [])
        )
        self.min_workspace_overlap = float(min_workspace_overlap)
        self.clock = clock
        tracker_kwargs.setdefault(
            "max_new_tracks_per_update", LIVE_MAX_NEW_TRACKS_PER_PASS
        )
        tracker_kwargs.setdefault(
            "new_track_spacing_m", LIVE_NEW_TRACK_SPACING_M
        )
        tracker_kwargs.setdefault("track_ttl_s", LIVE_TRACK_MEMORY_S)
        self.tracker = GlowTracker(min_confidence=min_confidence,
                                   **tracker_kwargs)
        self.screen_flashes = []
        self.screen_scheduler = ScreenFlashScheduler()
        self._lock = threading.Lock()

    def detect(self, frame, now):
        """Run one latest-frame pass and guarantee depth-safe visible output."""
        submitted_at = float(now)
        homography = table_to_image_homography(
            frame["K"], frame["pose_mat"], self.world_to_table
        )
        camera_table = (
            self.world_to_table @ np.append(frame["pose_mat"][:3, 3], 1.0)
        )[:2]
        detector_image = frame["image"]
        offset = np.zeros(2, dtype=float)
        workspace_mask = None
        table_path_available = True
        first_call_full_frame = not self.workspace_contours_table
        workspace_outside_view = False
        if self.workspace_contours_table:
            contours_px = [
                project_polygon(contour, homography)
                for contour in self.workspace_contours_table
            ]
            try:
                detector_image, crop_offset, workspace_mask = _workspace_crop(
                    frame["image"], contours_px
                )
            except ValueError:
                # The workspace comes from an older AR session and can leave
                # the current phone view mid-pan. Keep the demo animation alive
                # with current full-RGB masks instead of silently skipping YOLO.
                detector_image = frame["image"]
                workspace_outside_view = True
                table_path_available = False
                first_call_full_frame = True
            else:
                offset = np.asarray(crop_offset, dtype=float)
        raw_first = list(self.detector(detector_image))
        first_completed_at = (
            float(self.clock()) if self.clock else submitted_at
        )
        selected_first = _select_spaced_masks(raw_first, detector_image.shape)
        detections = []
        workspace_rejected_count = 0
        size_rejected_count = 0
        for polygon_px, confidence in selected_first:
            if not table_path_available:
                continue
            polygon_px = np.asarray(polygon_px, dtype=float) + offset
            if (
                workspace_mask is not None
                and _workspace_overlap(polygon_px, workspace_mask)
                < self.min_workspace_overlap
            ):
                workspace_rejected_count += 1
                continue
            table_poly = back_project_polygon(polygon_px, homography)
            if not _is_loose_piece(table_poly):
                size_rejected_count += 1
                continue
            detections.append({
                "table_polygon": table_poly,
                "table_centroid": table_poly.mean(axis=0),
                "confidence": confidence,
            })

        with self._lock:
            self.screen_flashes = [
                flash for flash in self.screen_flashes
                if first_completed_at - flash.activation_time < FLASH_END_S
            ]
            track_ids_before = {
                track.track_id for track in self.tracker.tracks
            }
            self.tracker.update(
                detections,
                first_completed_at,
                camera_xy=tuple(camera_table),
            )
            # Start newly discovered glows at inference completion. Waiting for
            # the next UI render would incorrectly make their lifetime depend
            # on whichever frame happened to render first.
            active = list(self.tracker.active_tracks(first_completed_at))
            visible_table = _visible_glow_tracks(
                active,
                homography,
                frame["image"].shape,
                first_completed_at,
            )
            visible_screen = _visible_glow_tracks(
                self.screen_flashes,
                np.eye(3),
                frame["image"].shape,
                first_completed_at,
            )
            new_track_count = len({
                track.track_id for track in self.tracker.tracks
            } - track_ids_before)

        completed_at = first_completed_at
        depth_rejected_count = 0
        fallback_size_rejected_count = 0
        fallback_flash_count = 0
        fallback_full_frame_call_count = 0
        fallback_reason = "not_needed"
        if not visible_table and not visible_screen:
            fallback_reason = (
                "workspace_outside_view"
                if workspace_outside_view
                else "no_visible_table_output"
            )
            if first_call_full_frame:
                fallback_raw = raw_first
            else:
                fallback_raw = list(self.detector(frame["image"]))
                fallback_full_frame_call_count = 1
                completed_at = (
                    float(self.clock()) if self.clock else submitted_at
                )

            result_age_s = max(0.0, completed_at - submitted_at)
            if result_age_s <= SCREEN_FLASH_MAX_RESULT_AGE_S:
                plausible = []
                for polygon_px, confidence in fallback_raw:
                    polygon_px = np.asarray(polygon_px, dtype=float)
                    evidence = screen_piece_evidence(
                        polygon_px,
                        confidence,
                        frame,
                        min_confidence=self.tracker.min_confidence,
                    )
                    if evidence.accepted:
                        plausible.append((polygon_px, float(confidence)))
                    elif evidence.reason == "oversized":
                        fallback_size_rejected_count += 1
                    else:
                        depth_rejected_count += 1
                selected_fallback = self.screen_scheduler.select(
                    plausible,
                    frame["image"].shape,
                    completed_at,
                    guarantee_one=True,
                )
                self.screen_scheduler.arm(
                    selected_fallback,
                    frame["image"].shape,
                    completed_at,
                )
                with self._lock:
                    self.screen_flashes = [
                        flash for flash in self.screen_flashes
                        if completed_at - flash.activation_time < FLASH_END_S
                    ]
                    self.screen_flashes.extend(
                        ScreenFlash(polygon, completed_at)
                        for polygon, _ in selected_fallback
                    )
                    fallback_flash_count = len(selected_fallback)
                    visible_screen = _visible_glow_tracks(
                        self.screen_flashes,
                        np.eye(3),
                        frame["image"].shape,
                        completed_at,
                    )
            else:
                fallback_reason = "stale_result"

        return {
            "frame_id": frame["frame_id"],
            "submitted_at": submitted_at,
            "completed_at": completed_at,
            "inference_latency_ms": round(
                max(0.0, completed_at - submitted_at) * 1000.0, 3
            ),
            "result_age_ms": round(
                max(0.0, completed_at - submitted_at) * 1000.0, 3
            ),
            "raw_mask_count": len(raw_first),
            "selected_mask_count": len(selected_first),
            "workspace_rejected_count": workspace_rejected_count,
            "size_rejected_count": size_rejected_count,
            "accepted_table_count": len(detections),
            "new_track_count": new_track_count,
            "visible_table_count": len(visible_table),
            "visible_screen_count": len(visible_screen),
            "depth_rejected_count": depth_rejected_count,
            "fallback_size_rejected_count": fallback_size_rejected_count,
            "fallback_reason": fallback_reason,
            "fallback_flash_count": fallback_flash_count,
            "fallback_full_frame_call_count": fallback_full_frame_call_count,
        }

    def render(self, frame, now):
        homography = table_to_image_homography(
            frame["K"], frame["pose_mat"], self.world_to_table
        )
        with self._lock:
            active = list(self.tracker.active_tracks(now))
            self.screen_flashes = [
                flash for flash in self.screen_flashes
                if now - flash.activation_time < FLASH_END_S
            ]
            screen_flashes = list(self.screen_flashes)
        visible_table = _visible_glow_tracks(
            active, homography, frame["image"].shape, now
        )
        visible_screen = _visible_glow_tracks(
            screen_flashes, np.eye(3), frame["image"].shape, now
        )
        out = render_glow(frame["image"], visible_table, homography, now)
        out = render_glow(out, visible_screen, np.eye(3), now)
        return out, len(visible_table) + len(visible_screen)


def _overlay_hud(
    image,
    piece_count,
    extra="",
    *,
    coverage_fraction=None,
    phone_frames=None,
    segmentation_passes=None,
):
    banner_height = 88 if coverage_fraction is not None else 54
    banner = image.copy()
    cv2.rectangle(
        banner, (0, 0), (image.shape[1], banner_height), (0, 0, 0), -1
    )
    image = cv2.addWeighted(banner, 0.45, image, 0.55, 0)
    status = f"pieces discovered: {piece_count}"
    if phone_frames is not None:
        status += f"   phone frames {phone_frames}"
    if segmentation_passes is not None:
        status += f"   segmentation passes {segmentation_passes}"
    if extra:
        status += f"   {extra}"
    cv2.putText(image, status,
                (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 230, 120), 2,
                cv2.LINE_AA)
    if coverage_fraction is not None:
        coverage = int(round(100.0 * float(coverage_fraction)))
        cv2.putText(
            image,
            f"You've scanned {coverage}% of the workspace",
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.05,
            GLOW_HUE_BGR,
            3,
            cv2.LINE_AA,
        )
    return image


def run_recorded(session_dir, detector, *, headless, out_path=None,
                 detect_every=4, speed=4.0, tracker_kwargs=None,
                 diagnostics_path=None):
    """Drive the live pipeline over a recorded scan. Detection runs on sampled
    frames (async in the live path); rendering runs every frame."""
    source = RecordedFrameSource(session_dir, realtime=not headless,
                                 speed=speed)
    world_to_table, workspace_contours, _ = _load_live_workspace(session_dir)
    wall0 = time.monotonic()
    session = LiveGlowSession(detector, world_to_table,
                              workspace_contours_table=workspace_contours,
                              clock=lambda: time.monotonic() - wall0,
                              **(tracker_kwargs or {}))
    pass_diagnostics = PassDiagnostics(diagnostics_path)
    writer = None

    # Async detection worker fed from the latest frame (drops stale frames).
    # Used for the on-screen path (rendering must never wait on YOLO). The
    # headless preview instead samples detection deterministically inline, so
    # the written frames reproduce the same detect/render interleaving without
    # a second redundant YOLO pass.
    latest = {"frame": None}
    lock = threading.Lock()
    stop = threading.Event()
    scheduler = LatestFrameScheduler()

    def worker():
        seen = None
        while not stop.is_set():
            with lock:
                frame = latest["frame"]
            if frame is None or frame["frame_id"] == seen:
                time.sleep(0.005)
                continue
            seen = frame["frame_id"]
            now = time.monotonic() - wall0
            if scheduler.should_detect(frame, now):
                pass_diagnostics.record(session.detect(frame, now))

    if not headless:
        threading.Thread(target=worker, daemon=True).start()

    max_pieces = 0
    for index, frame in enumerate(source.frames()):
        if headless and index % detect_every == 0:
            pass_diagnostics.record(
                session.detect(frame, time.monotonic() - wall0)
            )
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


def run_live(world_to_table, detector, *, workspace_contours_table,
             reference_grid,
             detect_every_s=DETECTION_INTERVAL_S, tracker_kwargs=None,
             diagnostics_path=None):
    source = LiveRecord3DSource(world_to_table)
    wall0 = time.monotonic()
    session = LiveGlowSession(detector, world_to_table,
                              workspace_contours_table=workspace_contours_table,
                              clock=lambda: time.monotonic() - wall0,
                              **(tracker_kwargs or {}))
    pass_diagnostics = PassDiagnostics(diagnostics_path)
    latest = {"frame": None}
    lock = threading.Lock()
    stop = threading.Event()
    scheduler = LatestFrameScheduler(min_interval_s=detect_every_s)
    diagnostics = LiveDiagnostics()
    coverage = LiveCoverageTracker(reference_grid, world_to_table)
    coverage_presentation = CoveragePresentation()

    def worker():
        seen = None
        while not stop.is_set():
            with lock:
                frame = latest["frame"]
            if frame is None or frame["frame_id"] == seen:
                time.sleep(0.005)
                continue
            seen = frame["frame_id"]
            now = time.monotonic() - wall0
            if scheduler.should_detect(frame, now):
                pass_diagnostics.record(session.detect(frame, now))
                diagnostics.note_segmentation_pass()

    def coverage_worker():
        seen = None
        while not stop.is_set():
            with lock:
                frame = latest["frame"]
            if frame is None or frame["frame_id"] == seen:
                time.sleep(0.005)
                continue
            seen = frame["frame_id"]
            coverage.update(frame, time.monotonic() - wall0)

    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=coverage_worker, daemon=True).start()
    print("Live scan: press q to stop.")
    try:
        for frame in source.frames():
            diagnostics.note_phone_frame()
            with lock:
                latest["frame"] = frame
            now = time.monotonic() - wall0
            rendered, count = session.render(frame, now)
            rendered = _render_scanner_sweep(rendered, now)
            state = diagnostics.snapshot()
            rendered = _overlay_hud(
                rendered,
                count,
                coverage_fraction=coverage_presentation.update(
                    coverage.fraction(), now
                ),
                phone_frames=state["phone_frames"],
                segmentation_passes=state["segmentation_passes"],
            )
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
    parser.add_argument(
        "--diagnostics",
        type=Path,
        help="optional per-segmentation-pass JSONL log",
    )
    parser.add_argument("--lego-cv", type=Path,
                        default=Path("/Users/emily/lego-cv"))
    parser.add_argument("--weights", type=Path,
                        default=Path("/Users/emily/lego-cv/models/lego_seg.pt"))
    parser.add_argument("--detect-every", type=int, default=4)
    parser.add_argument("--speed", type=float, default=4.0)
    parser.add_argument(
        "--stagger", type=float, default=DEFAULT_ACTIVATION_STAGGER_S
    )
    args = parser.parse_args()

    detector = YoloMaskDetector(args.weights, args.lego_cv)
    tracker_kwargs = {"activation_stagger_s": args.stagger}

    if args.live:
        world, workspace_contours, reference_grid = _load_live_workspace(
            args.world_to_table or args.session
        )
        run_live(world, detector,
                 workspace_contours_table=workspace_contours,
                 reference_grid=reference_grid,
                 diagnostics_path=args.diagnostics,
                 tracker_kwargs=tracker_kwargs)
    else:
        run_recorded(args.session, detector, headless=args.headless,
                     out_path=args.out, detect_every=args.detect_every,
                     speed=args.speed, tracker_kwargs=tracker_kwargs,
                     diagnostics_path=args.diagnostics)


if __name__ == "__main__":
    main()
