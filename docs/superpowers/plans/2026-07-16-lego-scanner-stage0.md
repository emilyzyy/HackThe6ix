# LEGO Scanner Stage 0 — Capture Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record synchronized RGB/depth/pose sessions from a LiDAR iPhone over record3d USB, track table-plane coverage live, and export a high-res orthographic top-down image for the existing lego-cv detector.

**Architecture:** A capture recorder snapshots RGB+depth+intrinsics+pose atomically inside the record3d frame callback and persists them (JPEG + npy + frames.jsonl) at ~5–10 fps. Downstream phases (plane fit, coverage grid, viewer, top-down export) all operate on recorded sessions, so everything after Phase 1 is testable offline; the viewer additionally hooks the live queue. A synthetic-session generator (known plane, known poses, rendered checkerboard) is the offline test bed for Phases 2–4.

**Tech Stack:** Python 3.11, record3d 1.4.1, numpy 2.4.6, opencv-python 5.0, open3d (Phase 2+), pytest.

## Global Constraints

- CPU-only, Apple Silicon laptop — no CUDA, no GPU-only dependencies.
- Depth is ~256×192 and only resolves the table, never LEGO pieces; RGB is the sole high-res analysis channel.
- Do not modify `lego-cv/`; Stage 0 output contract is `sessions/<ts>/topdown.jpg`.
- Each module independently runnable (`python <module>.py --help` works).
- Pose/image metadata must be snapshotted atomically per frame — a mismatch silently corrupts everything downstream.
- Console output during capture: one status line every ~2 s (carriage-return style), never per-frame prints.
- Unit tests for all pure logic (quaternion→matrix, plane math, grid marking) with mocked stream data.

## File Structure

```
lego-capture/
  transforms.py     # pure math: quat→R, pose→4x4, intrinsics scaling, project/unproject
  session_io.py     # SessionWriter / SessionReader, frames.jsonl schema, session summary
  capture.py        # record3d glue, FrameThrottler, CLI, status line
  plane.py          # Open3D table-plane fit → table_frame.json (world→table 4x4)
  coverage.py       # CoverageGrid: seen counts, azimuth-bin view diversity, % coverage
  viewer.py         # live/replay coverage window (gray unseen, RGB-projected seen cells)
  topdown.py        # orthographic export (per-cell best obs) + --mode best-frame
  tests/
    synthetic.py    # synthetic session generator (known plane/poses, checkerboard table)
    test_transforms.py test_session_io.py test_capture.py
    test_plane.py test_coverage.py test_topdown.py
  sessions/<YYYYMMDD-HHMMSS>/
    frames/00042.rgb.jpg 00042.depth.npy
    frames.jsonl  table_frame.json  coverage.npz  topdown.jpg
```

### frames.jsonl schema (one line per retained frame)

```json
{"frame_id": 42, "timestamp": 1234.567,
 "rgb": "frames/00042.rgb.jpg", "depth": "frames/00042.depth.npy",
 "rgb_size": [1440, 1920], "depth_size": [192, 256],
 "intrinsics": {"fx": 715.3, "fy": 715.3, "cx": 360.1, "cy": 480.2},
 "pose_qt": {"qx": 0, "qy": 0, "qz": 0, "qw": 1, "tx": 0, "ty": 0, "tz": 0},
 "pose_mat": [[...], [...], [...], [0,0,0,1]],
 "device_type": 1}
```

`pose_mat` is camera-to-world (ARKit convention: X right, Y up, Z toward viewer;
camera looks down −Z). `intrinsics` are the raw record3d coeffs; record3d reports
them for the RGB image — `transforms.scale_intrinsics` rescales them to the depth
resolution (verify the assumption on the first real capture by checking that
unprojected depth forms a plane; the synthetic fixture encodes the same convention).

---

### Task 1: Scaffold + transforms.py

**Files:** Create `transforms.py`, `tests/test_transforms.py`, `pytest.ini`; `git init`; install pytest.

**Produces (later tasks consume):**
- `quat_to_mat(qx, qy, qz, qw) -> np.ndarray (3,3)`
- `pose_to_mat(qx, qy, qz, qw, tx, ty, tz) -> np.ndarray (4,4)` camera-to-world
- `intrinsics_to_K(fx, fy, cx, cy) -> np.ndarray (3,3)`
- `scale_intrinsics(K, from_wh, to_wh) -> np.ndarray (3,3)`
- `unproject_depth(depth, K, cam_to_world, stride=1, max_depth=…) -> (N,3) world points`
  (pinhole: x=(u−cx)·d/fx, y=(v−cy)·d/fy, z=d in camera coords, then ARKit flip
  [x, −y, −z] before applying cam_to_world — record3d/ARKit camera looks down −Z
  with +Y up, image v grows downward)
- `project_points(points_world, K, cam_to_world, wh) -> (N,2) pixels + (N,) valid mask`

Tests: identity/90° quaternions, unit norm handling, round-trip project(unproject(depth))≈pixel grid, intrinsics scaling halves fx when image halves, pose_to_mat matches demo-main helper.

- [ ] Write failing tests → run → implement → pass → commit.

### Task 2: session_io.py

**Files:** Create `session_io.py`, `tests/test_session_io.py`.

**Produces:**
- `FrameRecord` dataclass mirroring the jsonl schema (with `.pose_mat` np array, `.load_rgb()`, `.load_depth()`)
- `SessionWriter(root_dir)`: `.add_frame(rgb_bgr, depth, coeffs_dict, pose_dict, device_type, timestamp) -> FrameRecord` (writes JPEG quality 95 + .npy + appends jsonl line, flushes per line), `.close() -> summary dict {frames, duration_s, bytes}`
- `SessionReader(session_dir)`: `.frames() -> list[FrameRecord]`, `.session_dir`
- `format_summary(summary) -> str`

Tests: tmpdir round-trip (write 3 synthetic frames, read back, arrays equal, pose_mat == pose_to_mat(pose_qt)), jsonl is valid line-json, summary counts/duration correct.

- [ ] Write failing tests → implement → pass → commit.

### Task 3: capture.py

**Files:** Create `capture.py`, `tests/test_capture.py`.

**Consumes:** `SessionWriter`, `pose_to_mat`.

**Produces:**
- `FrameThrottler(target_fps)`: `.should_retain(t) -> bool` (pure, monotonic-time based)
- `CaptureController(session_factory, target_fps)` — record3d-agnostic core:
  `.on_new_frame(snapshot_dict)` called with an already-copied
  `{rgb, depth, coeffs, pose, device_type, timestamp}`; internally throttles,
  queues, and a writer loop persists. `.stop() -> summary`.
- `Record3DSource`: owns `Record3DStream`; its `on_new_frame` callback snapshots
  **all** buffers immediately (rgb.copy(), depth.copy(), pose, intrinsics — one
  point in time, callback thread) then hands the dict to the controller. TrueDepth
  flip handled here as in demo-main.
- CLI: `python capture.py [--fps 6] [--out sessions] [--dev 0] [--live-view]` —
  connect, print "recording… press Enter to stop", status line every 2 s
  (`\r[REC 00:42] frames=252 (6.0 fps) disk=310 MB`), Enter → stop → final summary.

Tests (no device): throttler retains ~N of M synthetic timestamps at target fps and never exceeds it; controller fed 60 fake snapshots at 30 fps with **frame index baked into both the pose tx and a corner pixel of the fake RGB** → after stop, every jsonl record's pose tx matches the pixel value in its own JPEG (the sync guarantee), retained count ≈ 60·(6/30), files exist, summary correct.

- [ ] Write failing tests → implement → pass → commit.

### Task 4: synthetic session fixture + plane.py  (Phase 2)

**Files:** Create `tests/synthetic.py`, `plane.py`, `tests/test_plane.py`; install open3d.

**Produces:**
- `tests/synthetic.py`: `make_synthetic_session(out_dir, n_frames=20, table_z=0.0, …) -> session_dir` —
  camera orbits ~40 cm above a checkerboard-textured table plane at known world
  height, looking down; renders RGB via `project_points` sampling of the
  checkerboard, depth as exact ray-plane distance at 256×192; writes via
  `SessionWriter` so it exercises the real schema. Also runnable:
  `python -m tests.synthetic --out sessions/synthetic`.
- `plane.py`:
  - `fit_table_plane(reader, stride=4, n_frames=12) -> (normal, d, inliers)` —
    pooled world-space point cloud (unproject_depth, voxel-downsampled) →
    `o3d.geometry.PointCloud.segment_plane(distance_threshold=0.008, ransac_n=3)`;
    flip normal to point up (toward mean camera position).
  - `table_frame_from_plane(normal, d, points_inlier) -> np.ndarray (4,4)` world→table:
    origin = inlier centroid projected to plane, Z = up normal, X = arbitrary
    in-plane axis, Y = Z×X (pure, unit-testable without open3d).
  - CLI: `python plane.py sessions/<ts>` → writes `table_frame.json`
    ({world_to_table 4x4, plane normal+d, extent estimate}), prints fit stats.

Tests: table_frame math pure tests (plane z=0 → identity-ish; tilted plane → transformed points have z≈0); end-to-end on synthetic session → recovered plane within 0.5° / 5 mm of ground truth.

- [ ] Write failing tests → implement → pass → commit.

### Task 5: coverage.py  (Phase 2)

**Files:** Create `coverage.py`, `tests/test_coverage.py`.

**Consumes:** `unproject_depth`, `world_to_table` from table_frame.json / plane.py.

**Produces:**
- `CoverageGrid(bounds_xy, cell_m=0.003)`: arrays `seen_count (H,W) uint16`,
  `angle_bins (H,W) uint8` (bitmask of 8 azimuth octants of the camera direction
  projected into the table plane), `.mark_frame(depth, K_depth, cam_to_world, world_to_table, z_band=0.02)`
  (unproject → table coords → in-band points → cell indices via floor((xy−min)/cell) → 
  bump count, OR azimuth bit), `.coverage_fraction(min_seen=2)`,
  `.observed_mask()`, `.save(path.npz)` / `.load`.
  Workspace hull = cells within convex hull of all observed cells;
  `.hull_coverage_fraction()` is the “coverage complete” metric.
- CLI: `python coverage.py sessions/<ts>` → replays session, prints final coverage %.

Tests: single synthetic frame marks the expected rectangle of cells; two opposing views set two distinct azimuth bits; out-of-band (above-plane) points not marked; fraction math exact on hand-built grids.

- [ ] Write failing tests → implement → pass → commit.

### Task 6: viewer.py  (Phase 3)

**Files:** Create `viewer.py`; modify `capture.py` (`--live-view` wires a frame tap).

**Consumes:** CoverageGrid, plane.py fits, SessionReader, capture frame queue.

**Produces:**
- `CoverageView(grid, world_to_table)`: keeps an RGB canvas (H,W,3) aligned to the
  grid; per frame, for observed cells, project cell centers → camera pixels
  (project_points with RGB-scaled K), sample colors; unseen cells gray (128).
  `.render() -> BGR image` with % overlay and "COVERAGE COMPLETE" banner at ≥ threshold (default 95% of hull).
- Replay mode: `python viewer.py sessions/<ts>` — fits plane on first N frames,
  then steps frames at ~2 Hz in a cv2 window (works fully offline).
- Live mode: `capture.py --live-view` runs plane fit lazily once enough frames
  exist (~20), then updates/render at ≤2 Hz on the main thread; capture never
  blocks on rendering (viewer reads the latest snapshot only, drops the rest).

Tests: CoverageView canvas colors observed cells from a synthetic frame (checkerboard cells get ~checkerboard colors), unseen stay gray; render smoke test headless (no imshow in tests).

- [ ] Write failing tests → implement → pass → commit; verify replay window manually-optional.

### Task 7: topdown.py  (Phase 4)

**Files:** Create `topdown.py`, `tests/test_topdown.py`.

**Produces:**
- Ortho mode (default): output grid at `--px-per-mm 2` over observed hull bounds; for
  each frame compute per-cell score = cos(incidence) × sharpness(Laplacian var of
  its rgb) / distance²; sample RGB at projected cell centers (cv2.remap batched);
  keep best-scoring observation per cell; write `sessions/<ts>/topdown.jpg`.
- `--mode best-frame`: score frames by (top-down-ness = cos angle(view dir, −normal),
  sharpness); rectify winner via homography from image plane to table plane
  (H from K, cam_to_world, world_to_table — pure helper `plane_homography(...)`,
  unit-tested) → cv2.warpPerspective → topdown.jpg.
- CLI: `python topdown.py sessions/<ts> [--mode ortho|best-frame] [--px-per-mm 2]`.

Tests: on synthetic session both modes reproduce the checkerboard (pattern
correlation with ground-truth texture > 0.8, straight edges); homography helper maps
4 known table corners to their exact projected pixels.

- [ ] Write failing tests → implement → pass → commit.

### Task 8: README + end-to-end offline verification

**Files:** Create `README.md`.

- README: setup, per-module usage, session directory layout, the one-real-capture →
  offline-iteration workflow, coordinate conventions.
- End-to-end: generate synthetic session → plane → coverage → viewer replay
  (headless smoke) → topdown both modes; confirm artifacts exist and look right.
- [ ] Verify → commit.

## Self-Review Notes

- Spec coverage: Phase 1→Tasks 1–3, Phase 2→Tasks 4–5, Phase 3→Task 6, Phase 4→Task 7, README/tests→Task 8. Throttle ✓, sync guarantee test ✓ (Task 3), minimal console ✓ (status line), CPU-only ✓, lego-cv untouched ✓.
- Types consistent: pose_mat always camera-to-world 4×4; world_to_table always 4×4; intrinsics dict keys fx/fy/cx/cy everywhere (record3d's `tx/ty` coeff names are mapped to cx/cy at the capture boundary, and only there).
- Open risk: record3d intrinsics-vs-RGB-resolution assumption — verified cheaply on first real capture (plane fit residuals); synthetic fixture is self-consistent either way.
