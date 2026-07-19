# Motion-Paced Live Glow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the live RGB viewer responsive and visually appealing over an approximately 800-piece table by segmenting only after meaningful camera motion, spacing inference passes, and admitting a small spatially diverse set of new glows per pass.

**Architecture:** A camera-pose scheduler permits the first inference immediately, then requires both a 0.75-second interval and either 1.5 cm translation or 5 degrees rotation. YOLO still receives the newest camera frame, but its live candidate cap drops to 96. The tracker refreshes existing tracks while admitting no more than 12 new tracks per update, with 4 cm separation among new discoveries in that update; every accepted polygon remains table-anchored and renders on every RGB frame independently of inference.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Ultralytics YOLO, pytest, Record3D.

## Global Constraints

- Do not assume or encode any camera sweep direction or table-zone order.
- Keep RGB acquisition and glow reprojection unthrottled.
- Minimum time between inference starts is 0.75 seconds.
- Meaningful motion is at least 0.015 m translation or 5 degrees rotation since the previous accepted inference frame.
- Admit at most 12 new tracks per inference update and keep their centroids at least 0.04 m apart within that update.
- Retain tracks for 30 seconds so normal backtracking does not immediately rediscover and reflash the same pieces.
- Live YOLO uses `retina_masks=False`, `imgsz=640`, and `max_det=96`; batch defaults remain unchanged.
- Preserve the LiDAR workspace and loose-piece gates from the previous implementation.
- Do not commit, push, merge, reset, or discard existing changes.

---

### Task 1: Motion-triggered inference scheduler

**Files:**
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Test: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`

**Interfaces:**
- Produces: `MotionDetectionScheduler(min_interval_s=0.75, min_translation_m=0.015, min_rotation_deg=5.0)`.
- Produces: `should_detect(frame: dict, now: float) -> bool`.

- [x] Add tests proving the first frame runs, early frames are suppressed, stationary frames remain suppressed after the interval, and translation or rotation in either direction releases the newest frame.
- [x] Run the focused tests and confirm they fail because the scheduler does not exist.
- [x] Implement pose-delta scheduling without using a sweep direction.
- [x] Wire the scheduler into both asynchronous on-screen workers while leaving headless deterministic replay unchanged.
- [x] Run the focused viewer tests.

### Task 2: Spatially paced new discoveries

**Files:**
- Modify: `/Users/emily/lego-capture/live_glow.py`
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Test: `/Users/emily/lego-capture/tests/test_live_glow.py`
- Test: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`

**Interfaces:**
- Extends: `GlowTracker(max_new_tracks_per_update: int | None = None, new_track_spacing_m: float = 0.0)`.
- Live defaults: `max_new_tracks_per_update=12`, `new_track_spacing_m=0.04`, and `track_ttl_s=30.0`.

- [x] Add tracker tests proving the per-update cap and spacing apply only to new tracks while existing tracks can still refresh.
- [x] Run the focused tests and confirm they fail against unrestricted track creation.
- [x] Implement the cap and greedy table-distance spacing after confidence and one-to-one matching checks.
- [x] Configure live sessions with the approved pacing and memory defaults.
- [x] Run the focused tracker and viewer tests.

### Task 3: Bounded live model output and verification

**Files:**
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Test: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`

**Interfaces:**
- Live detector configuration: `YoloSegModel(..., imgsz=640, retina_masks=False, max_det=96)`.

- [x] Change the existing live-detector regression expectation from 300 candidates to 96 and confirm it fails.
- [x] Set the live-only model cap to 96 without changing `lego-cv` batch defaults.
- [x] Run both complete test suites.
- [x] Replay `/Users/emily/lego-capture/sessions/20260719-032336` and benchmark final live detection timing and admitted-track pacing.
- [x] Audit planned diffs and leave all changes unstaged and uncommitted.
