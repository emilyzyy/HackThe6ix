"""Live-glow engine (P3): table-anchored piece tracks for the live scan viewer.

The live path is: live frame + pose -> async sampled YOLO -> a temporary
table-space track per piece -> project that track into every subsequent live
frame -> glow. This module is the pure core (no camera, no window, no YOLO):

- image<->table projection for on-plane points, reusing the exact plane
  homography the batch pipeline already uses (topdown.plane_homography);
- GlowTracker: associate async detections to persistent table-space tracks,
  activate each track exactly once, stagger activations along the camera's
  travel direction (the scanning "pulse"), and expose them for rendering;
- glow_envelope: a short wall-clock discovery flash that fades completely,
  sampled every UI frame independently of the (slower, sampled) detection
  cadence.

Anchoring assumes pieces rest on the table plane (z=0) — the same assumption
the whole pipeline makes; a piece's 2D mask is back-projected onto that plane.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from topdown import plane_homography

# Glow timing (wall-clock seconds): a strong single discovery flash followed
# by a quick fade. Track identity remains alive after the pixels disappear, so
# repeated detections do not restart the animation.
FLASH_DURATION_S = 0.20
FLASH_END_S = 0.48
FLASH_PEAK_ALPHA = 0.75
FLASH_PEAK_EDGE_ALPHA = 1.0
DEFAULT_ACTIVATION_STAGGER_S = 0.08


def table_to_image_homography(K, cam_to_world, world_to_table):
    """3x3 mapping table-plane metres (x, y, 1) -> image pixels (on z=0).

    Reuses the batch pipeline's plane homography with unit scale and zero
    origin so the "canvas" axes are literally table metres.
    """
    return plane_homography(
        np.asarray(K, dtype=float),
        np.asarray(cam_to_world, dtype=float),
        np.asarray(world_to_table, dtype=float),
        px_per_m=1.0,
        origin_xy=(0.0, 0.0),
    )


def project_polygon(table_polygon, homography):
    """Table-metre polygon -> image-pixel polygon."""
    pts = np.asarray(table_polygon, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(homography)).reshape(-1, 2)


def back_project_polygon(image_polygon, homography):
    """Image-pixel polygon (assumed on the plane) -> table-metre polygon."""
    inv = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    pts = np.asarray(image_polygon, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, inv).reshape(-1, 2)


def glow_envelope(age_s: float) -> dict:
    """Glow strength as a function of seconds since a track activated."""
    if age_s < 0.0:
        age_s = 0.0
    if age_s < FLASH_DURATION_S:
        progress = age_s / FLASH_DURATION_S
        intensity = 1.0 - 0.2 * progress
        phase = "flash"
    elif age_s < FLASH_END_S:
        progress = (age_s - FLASH_DURATION_S) / (
            FLASH_END_S - FLASH_DURATION_S
        )
        intensity = 0.8 * (1.0 - progress) ** 2
        phase = "fade"
    else:
        intensity = 0.0
        phase = "off"
    fill = FLASH_PEAK_ALPHA * intensity
    edge = FLASH_PEAK_EDGE_ALPHA * intensity
    return {"fill_alpha": float(fill), "edge_alpha": float(edge), "phase": phase}


# A single consistent scan/glow hue (BGR). Not the piece's real colour — the
# live effect only communicates "a physical piece is understood here".
GLOW_HUE_BGR = (0, 213, 255)  # LEGO yellow (#FFD500)


def render_glow(frame_bgr, tracks, homography, now, *, hue_bgr=GLOW_HUE_BGR):
    """Draw each active track's table polygon onto the frame as a translucent
    glowing mask with a traced boundary. Pure: returns a new image.

    ``tracks`` are activated Tracks; each is projected through ``homography``
    (table metres -> this frame's pixels) so the mask stays attached to the
    physical piece as the camera moves.
    """
    out = frame_bgr.copy()
    height, width = out.shape[:2]
    for track in tracks:
        if track.activation_time is None:
            continue
        env = glow_envelope(now - track.activation_time)
        if env["fill_alpha"] <= 0.0 and env["edge_alpha"] <= 0.0:
            continue
        polygon = np.rint(
            project_polygon(track.table_polygon, homography)
        ).astype(np.int32)
        # Skip polygons entirely outside this view.
        if (polygon[:, 0].max() < 0 or polygon[:, 0].min() >= width
                or polygon[:, 1].max() < 0 or polygon[:, 1].min() >= height):
            continue
        sigma = max(3.0, min(width, height) * 0.01)
        thickness = max(2, int(min(width, height) * 0.004))
        # Glow composition used to allocate, blur, and blend a full-resolution
        # frame once per track. A busy pile therefore blocked the UI for several
        # seconds even though inference itself was asynchronous. Restrict the
        # work to the projected piece plus a four-sigma halo margin.
        pad = int(np.ceil(4.0 * sigma)) + thickness + 2
        x0 = max(0, int(polygon[:, 0].min()) - pad)
        y0 = max(0, int(polygon[:, 1].min()) - pad)
        x1 = min(width, int(polygon[:, 0].max()) + pad + 1)
        y1 = min(height, int(polygon[:, 1].max()) + pad + 1)
        local_polygon = polygon - np.array([x0, y0], dtype=np.int32)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        cv2.fillPoly(mask, [local_polygon], 255)
        # Soft outer halo so the piece reads as "glowing", not just tinted.
        halo = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma)
        region = mask.astype(bool)
        fill = env["fill_alpha"]
        hue = np.array(hue_bgr, dtype=np.float32)
        roi = out[y0:y1, x0:x1].astype(np.float32)
        roi[region] = roi[region] * (1.0 - fill) + hue * fill
        halo_alpha = (halo.astype(np.float32) / 255.0)[..., None] * (
            0.35 * env["edge_alpha"]
        )
        halo_alpha[region] = 0.0  # halo only outside the solid fill
        roi = (roi * (1.0 - halo_alpha) + hue * halo_alpha).astype(np.uint8)
        edge = tuple(int(min(255, c * 1.0 + 60)) for c in hue_bgr)
        traced = roi.copy()
        cv2.polylines(
            traced, [local_polygon], True, edge, thickness, cv2.LINE_AA
        )
        out[y0:y1, x0:x1] = cv2.addWeighted(
            traced, env["edge_alpha"], roi, 1.0 - env["edge_alpha"], 0.0
        )
    return out


@dataclass
class Track:
    track_id: int
    table_polygon: np.ndarray  # metres, (N,2)
    table_centroid: np.ndarray  # metres, (2,)
    confidence: float
    created_at: float
    last_seen: float
    activation_time: float | None = None  # wall-clock when the glow started
    _pending_since: float = 0.0

    @property
    def activated(self) -> bool:
        return self.activation_time is not None


@dataclass
class GlowTracker:
    """Associate async detections to persistent table-space tracks."""

    min_confidence: float = 0.60
    # Two detections within this table distance are the same physical piece.
    match_distance_m: float = 0.03
    # Space out new activations so pieces light progressively, not all at once.
    activation_stagger_s: float = DEFAULT_ACTIVATION_STAGGER_S
    # Drop a track that has not been re-seen for this long (unbounded panning).
    track_ttl_s: float = 4.0
    # Live pacing can bound and spatially distribute only newly created tracks;
    # existing tracks remain eligible for refresh regardless of this budget.
    max_new_tracks_per_update: int | None = None
    new_track_spacing_m: float = 0.0

    tracks: list[Track] = field(default_factory=list)
    _ids: itertools.count = field(default_factory=lambda: itertools.count())
    _pending: list[int] = field(default_factory=list)  # track ids awaiting glow
    _last_activation_at: float = -1e9
    _camera_history: list[tuple[float, np.ndarray]] = field(default_factory=list)
    _forced_travel: np.ndarray | None = None

    def set_camera_travel(self, direction) -> None:
        """Override the estimated camera travel direction (mainly for tests)."""
        vector = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(vector))
        self._forced_travel = vector / norm if norm else None

    def _travel_direction(self) -> np.ndarray | None:
        if self._forced_travel is not None:
            return self._forced_travel
        if len(self._camera_history) < 2:
            return None
        delta = self._camera_history[-1][1] - self._camera_history[0][1]
        norm = float(np.linalg.norm(delta))
        return delta / norm if norm > 1e-6 else None

    def update(self, detections, now: float, camera_xy=None) -> None:
        """Fold one async inference result into the table-space tracks."""
        if camera_xy is not None:
            self._camera_history.append((now, np.asarray(camera_xy, dtype=float)))
            self._camera_history = [
                item for item in self._camera_history if now - item[0] <= 1.0
            ] or self._camera_history[-2:]

        matched_track_ids = set()
        new_centroids = []
        new_track_count = 0
        ordered = sorted(
            detections,
            key=lambda detection: float(detection.get("confidence", 0.0)),
            reverse=True,
        )
        for detection in ordered:
            if float(detection.get("confidence", 0.0)) < self.min_confidence:
                continue
            centroid = np.asarray(detection["table_centroid"], dtype=float)
            match = self._nearest_track(centroid, matched_track_ids)
            if match is not None:
                # Keep the mask attached to the freshest observed shape.
                match.table_polygon = np.asarray(
                    detection["table_polygon"], dtype=float
                )
                match.table_centroid = centroid
                match.confidence = float(detection["confidence"])
                match.last_seen = now
                matched_track_ids.add(match.track_id)
            else:
                if (
                    self.max_new_tracks_per_update is not None
                    and new_track_count >= self.max_new_tracks_per_update
                ):
                    continue
                if self.new_track_spacing_m > 0.0 and any(
                    np.linalg.norm(centroid - accepted)
                    < self.new_track_spacing_m
                    for accepted in new_centroids
                ):
                    continue
                track = Track(
                    track_id=next(self._ids),
                    table_polygon=np.asarray(
                        detection["table_polygon"], dtype=float
                    ),
                    table_centroid=centroid,
                    confidence=float(detection["confidence"]),
                    created_at=now,
                    last_seen=now,
                    _pending_since=now,
                )
                self.tracks.append(track)
                self._pending.append(track.track_id)
                matched_track_ids.add(track.track_id)
                new_centroids.append(centroid)
                new_track_count += 1

        self._expire(now)

    def _nearest_track(self, centroid, excluded_ids=()) -> Track | None:
        best, best_distance = None, self.match_distance_m
        for track in self.tracks:
            if track.track_id in excluded_ids:
                continue
            distance = float(np.linalg.norm(track.table_centroid - centroid))
            if distance <= best_distance:
                best, best_distance = track, distance
        return best

    def _expire(self, now: float) -> None:
        keep = []
        for track in self.tracks:
            if now - track.last_seen <= self.track_ttl_s:
                keep.append(track)
            elif track.track_id in self._pending:
                self._pending.remove(track.track_id)
        self.tracks = keep

    def _release_pending(self, now: float) -> None:
        """Activate queued tracks one at a time, ordered along camera travel."""
        if not self._pending:
            return
        travel = self._travel_direction()

        def order_key(track_id: int):
            track = self._by_id(track_id)
            if travel is not None:
                # Furthest back along travel lights first, so the pulse sweeps
                # forward in the pan direction.
                return (float(track.table_centroid @ travel), track._pending_since)
            return (track._pending_since, track_id)

        while self._pending:
            if now - self._last_activation_at < self.activation_stagger_s:
                break
            self._pending.sort(key=order_key)
            track = self._by_id(self._pending.pop(0))
            track.activation_time = now
            self._last_activation_at = now

    def _by_id(self, track_id: int) -> Track:
        return next(track for track in self.tracks if track.track_id == track_id)

    def active_tracks(self, now: float) -> list[Track]:
        """Tracks whose glow has started, releasing staggered activations."""
        self._release_pending(now)
        return [track for track in self.tracks if track.activated]
