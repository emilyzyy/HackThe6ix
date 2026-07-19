# Time-Paced Live Segmentation and Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guarantee continued segmentation while the phone pans, preserve a responsive RGB/glow view, and restore meaningful live LiDAR workspace progress.

**Architecture:** Replace pose-dependent inference scheduling with a latest-frame, wall-clock scheduler that permits one pass every 0.50 seconds and drops stale frames. Snapshot aligned depth with every live RGB callback, then feed the newest depth frame to an independent lightweight coverage worker. Coverage is measured against the anchor session's fixed LiDAR workspace mask and displayed with frame/pass diagnostics in the RGB HUD.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Ultralytics YOLO, Record3D, pytest.

## Global Constraints

- Inference must not depend on camera direction, translation, or rotation.
- The first received frame is eligible immediately; later passes require 0.50 seconds.
- Only the newest unprocessed phone frame may be inferred; stale frames are dropped.
- RGB rendering remains on the main thread and is not throttled by inference or coverage.
- LiDAR coverage runs concurrently and latest-only at no more than 4 Hz.
- Coverage denominator is the fixed admissible LiDAR workspace from the anchor session.
- HUD copy is `You've scanned N% of the workspace` and includes phone-frame plus completed-segmentation counters.
- Existing LiDAR workspace masking, loose-piece sizing, live `max_det=96`, and glow behavior remain intact.
- Do not commit, push, merge, reset, or discard existing changes.

---

### Task 1: Guaranteed latest-frame inference cadence

**Files:**
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Test: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`

**Interfaces:**
- Replaces: `MotionDetectionScheduler` with `LatestFrameScheduler(min_interval_s=0.50)`.
- Produces: `should_detect(frame: dict, now: float) -> bool` based only on frame identity and wall time.

- [x] Replace pose-based tests with tests proving first-frame acceptance, time throttling, stationary-frame continuation, and duplicate-frame suppression.
- [x] Run the focused tests and confirm they fail against motion-dependent behavior.
- [x] Implement the time-only scheduler and wire both on-screen detection workers.
- [x] Add thread-safe live diagnostics for received frame count and completed segmentation pass count.
- [x] Run focused viewer tests.

### Task 2: Concurrent fixed-workspace LiDAR coverage

**Files:**
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Test: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`

**Interfaces:**
- Extends live frames with: `depth: np.ndarray`.
- Produces: `LiveCoverageTracker(reference_grid, world_to_table, max_update_hz=4.0)`.
- Produces: `update(frame: dict, now: float) -> bool` and `fraction() -> float`.

- [x] Add tests proving progress uses the fixed reference mask, increases with valid depth, and respects its update cadence.
- [x] Run the focused tests and confirm the tracker is missing.
- [x] Snapshot Record3D depth alongside RGB/pose/intrinsics.
- [x] Implement a same-bounds live grid whose observed cells are divided by the anchor reference mask.
- [x] Run a latest-only coverage worker in parallel with inference.
- [x] Run focused viewer tests.

### Task 3: RGB HUD and end-to-end verification

**Files:**
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Test: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`

**Interfaces:**
- Extends: `_overlay_hud(image, piece_count, extra="", coverage_fraction=None, phone_frames=None, segmentation_passes=None)`.

- [x] Add image-level tests proving the coverage copy and diagnostics alter the HUD only when supplied.
- [x] Implement the combined RGB overlay with `You've scanned N% of the workspace`.
- [x] Run both complete repository test suites.
- [x] Replay `/Users/emily/lego-capture/sessions/20260719-032336`, simulate scheduling/coverage, and inspect the rendered preview.
- [x] Audit diffs and leave everything unstaged and uncommitted.
