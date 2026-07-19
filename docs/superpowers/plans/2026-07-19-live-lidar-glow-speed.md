# Live LiDAR-Gated Glow Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the phone RGB glow viewer update faster, reject detections outside the LiDAR-observed table and detections too large to be loose LEGO pieces, and replace the persistent glow with a short bright flash.

**Architecture:** Reuse the capture session's depth-derived admissible workspace contour and project it into every RGB frame. Run the live-only segmenter on a black-masked crop of that projection, then validate every returned polygon by workspace overlap and table-metric footprint before tracking it. Keep batch segmentation at retina-mask quality while allowing the live adapter to request coarse masks.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Ultralytics YOLO, pytest, Record3D.

## Global Constraints

- Preserve batch inference defaults: `retina_masks=True`, `max_det=300`, and `imgsz=640`.
- Live inference uses `retina_masks=False`, `max_det=300`, and `imgsz=640`.
- Accept a live polygon only when at least 80% of its rasterized area overlaps the projected LiDAR workspace.
- Loose-piece footprint limits are 0.16 m maximum side and 0.008 m² maximum oriented bounding-box area.
- Glow uses one bright discovery flash and is fully transparent after 0.60 seconds; track identity still persists internally.
- Do not commit, push, merge, reset, or discard existing changes.

---

### Task 1: Configurable live-only mask quality

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/segdetect.py`
- Test: `/Users/emily/lego-cv/tests/test_segdetect.py`

**Interfaces:**
- Produces: `YoloSegModel(..., retina_masks: bool = True, max_det: int = 300)`.
- Preserves: `predict(image) -> list[RawSegmentation]` and all batch defaults.

- [x] Add a fake-Ultralytics regression test proving constructor values reach `model.predict` and masks are resized to the source image.
- [x] Run the test and confirm it fails because `YoloSegModel` does not accept the new options.
- [x] Store `retina_masks` and `max_det` in `YoloSegModel` and forward them from `predict`.
- [x] Run the focused segmentation test and the full `lego-cv` test suite.

### Task 2: Projected LiDAR workspace and loose-piece gate

**Files:**
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Test: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`

**Interfaces:**
- Produces: `_load_live_workspace(session_dir) -> tuple[np.ndarray, list[list[list[float]]]]`.
- Produces: `_workspace_crop(image, contours_px) -> tuple[np.ndarray, tuple[int, int], np.ndarray]`.
- Produces: `_workspace_overlap(polygon_px, workspace_mask) -> float`.
- Produces: `_is_loose_piece(table_polygon, max_dimension_m=0.16, max_area_m2=0.008) -> bool`.
- Extends: `LiveGlowSession(..., workspace_contours_table=None, min_workspace_overlap=0.80)`.

- [x] Add focused tests for crop offset restoration, outside-workspace rejection, oversized rejection, and valid loose-piece acceptance.
- [x] Run the focused tests and confirm the new behavior fails.
- [x] Build contours by replaying the anchor session's recorded LiDAR depth, project and black-mask the RGB inference crop, restore image coordinates, and apply overlap plus metric-footprint gates before tracking.
- [x] Wire both recorded replay and `--live` startup to pass their depth-derived contours, while retaining optional contours for existing injected-unit tests.
- [x] Instantiate the live detector with coarse masks while leaving its 640 input and 300-candidate cap unchanged.
- [x] Run the focused viewer tests.

### Task 3: Short discovery flash and end-to-end verification

**Files:**
- Modify: `/Users/emily/lego-capture/live_glow.py`
- Test: `/Users/emily/lego-capture/tests/test_live_glow.py`

**Interfaces:**
- Preserves: `glow_envelope(age_s) -> dict`.
- Changes phases to: `flash`, `fade`, and `off` with zero fill/edge alpha at and after 0.60 seconds.

- [x] Replace the persistent-glow expectations with tests for a bright onset, decay, and complete transparency after 0.60 seconds.
- [x] Run the focused test and confirm it fails against the persistent steady highlight.
- [x] Implement a one-cycle flash that reaches zero alpha by 0.60 seconds without changing tracker lifetime.
- [x] Run all capture tests and all CV tests.
- [x] Replay `/Users/emily/lego-capture/sessions/20260719-032336` headlessly and benchmark full-frame retina masks against LiDAR-cropped coarse masks.
- [x] Inspect the final diff to confirm only the planned files and local plan changed; do not stage or commit them.
