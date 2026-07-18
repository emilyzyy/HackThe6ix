# Calibrated Silhouette Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and evaluate a diagnostic-only calibrated multi-view cuboid fitter that can separate regular Brick and Plate candidates without changing production identities.

**Architecture:** `pipeline/model_fit.py` will own ARKit projection, regular cuboid models, original-mask loss calculation, and deterministic coarse-to-fine pose search. It consumes existing `Observation` masks plus frame/table calibration loaded from the session. `pipeline/multiview.py` will serialize diagnostic results and run-scoped overlays only after synthetic and projection tests pass; production arbitration remains unchanged until a separate promotion plan.

**Tech Stack:** Python 3.11, NumPy 2.4, OpenCV 5.0, pytest. No new dependency and no solvePnP.

## Global Constraints

- Geometry is diagnostic-only throughout this plan.
- Wrong exact identity is worse than unknown/review-needed.
- Production `brickognize` identities cannot change in this plan.
- Models are limited to appearance-supplied `Brick N x M` and `Plate N x M`.
- LDraw dimensions use 8.0 mm pitch, 9.6 mm brick height, and 3.2 mm plate height.
- Original segmentation masks, intrinsics, poses, and table coordinates are mandatory.
- A usable oblique view is mandatory for a reliable result.
- The only usable oblique view cannot be trimmed.
- At most one other outlier view may be trimmed.
- Output is run-scoped and cannot overwrite preserved baselines.
- Tests precede every production change.

---

### Task 1: Camera Calibration and Projection

**Files:**
- Create: `/Users/emily/lego-cv/pipeline/model_fit.py`
- Create: `/Users/emily/lego-cv/tests/test_model_fit.py`

**Interfaces:**
- `FrameCalibration(frame_id, intrinsics, rgb_size, camera_to_world, world_to_table, view_tilt_deg)`.
- `load_frame_calibrations(session_dir, views) -> dict[int, FrameCalibration]`.
- `project_table_points(points_table_xyz, calibration) -> (pixels, valid)`.

- [ ] Write tests that project a centered point through an identity ARKit camera, reject behind-camera points, and compare projected z=0 table points with the real session's stored frame-14 plane homography to within `1e-5` pixels.
- [ ] Run `.venv/bin/pytest tests/test_model_fit.py -k projection -v` and observe import/module RED.
- [ ] Implement `FrameCalibration` and JSONL/table loaders. Convert table to world with `inv(world_to_table)`, world to ARKit camera with `inv(camera_to_world)`, then use `x`, `-y`, `-z` as OpenCV pinhole coordinates.
- [ ] Run the projection tests and require GREEN before continuing.

Projection implementation:

```python
def project_table_points(points_table_xyz, calibration):
    points = np.asarray(points_table_xyz, dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    table_to_world = np.linalg.inv(calibration.world_to_table)
    world_to_camera = np.linalg.inv(calibration.camera_to_world)
    camera = (world_to_camera @ (table_to_world @ homogeneous.T)).T
    depth = -camera[:, 2]
    fx, fy, cx, cy = calibration.intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * camera[:, 0] / depth + cx
        v = fy * -camera[:, 1] / depth + cy
    pixels = np.column_stack([u, v])
    width, height = calibration.rgb_size
    valid = (
        (depth > 0) & np.isfinite(pixels).all(axis=1)
        & (u >= -0.5) & (u < width - 0.5)
        & (v >= -0.5) & (v < height - 0.5)
    )
    return pixels, valid
```

Commit after focused tests:

```bash
git add pipeline/model_fit.py tests/test_model_fit.py
git commit -m "feat: add calibrated model projection"
```

---

### Task 2: Regular Cuboid Models, Orientations, and Rasterization

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/model_fit.py`
- Modify: `/Users/emily/lego-cv/tests/test_model_fit.py`

**Interfaces:**
- `RegularModel.from_identification(part_id, name) -> RegularModel | None`.
- `resting_orientations(model) -> tuple[RestingOrientation, ...]`.
- `cuboid_vertices(center_xy, yaw_deg, dimensions_xyz) -> ndarray[8,3]`.
- `render_cuboid(mask_shape, calibration, vertices) -> bool mask`.

- [ ] Write RED tests for Brick 2 x 4 dimensions `(0.016, 0.032, 0.0096)`, Plate 2 x 4 height `0.0032`, rejection of irregular names, symmetry-deduplicated orientations, and nonempty projected cuboid masks.
- [ ] Implement the exact regular-name grammar `^(Brick|Plate) (\d+) x (\d+)$` and enumerate unique permutations of width, length, and height as the table x/y/z extents.
- [ ] Generate eight centered vertices from `z=0` to the vertical extent, rotate x/y by yaw, project them, take `cv2.convexHull`, and fill the polygon.
- [ ] Run all Task 2 tests GREEN and commit.

```bash
git add pipeline/model_fit.py tests/test_model_fit.py
git commit -m "feat: render regular part cuboids"
```

---

### Task 3: Inspectable Per-View Silhouette Loss

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/model_fit.py`
- Modify: `/Users/emily/lego-cv/tests/test_model_fit.py`

**Interfaces:**
- `score_silhouettes(observed, predicted) -> ViewFitLoss`.
- Fields: `iou`, `boundary_error`, `under_coverage`, `spill`, `total`.

- [ ] Write RED tests asserting zero loss for equal masks, larger under-coverage for a contained prediction, larger spill for an oversized prediction, and symmetric boundary error for equal translations in opposite directions.
- [ ] Implement IoU and asymmetric missing/spill fractions. Compute symmetric boundary loss with distance transforms of one-pixel morphological boundaries, normalized by observed-mask diagonal.
- [ ] Use the fixed transparent diagnostic total `0.35*(1-iou) + 0.20*boundary + 0.25*under + 0.20*spill`; retain all raw terms so this diagnostic weight choice is not mistaken for a calibrated production threshold.
- [ ] Run Task 3 tests GREEN and commit.

```bash
git add pipeline/model_fit.py tests/test_model_fit.py
git commit -m "feat: score projected silhouettes"
```

---

### Task 4: Deterministic Multi-View Pose Search and Ambiguity

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/model_fit.py`
- Modify: `/Users/emily/lego-cv/tests/test_model_fit.py`

**Interfaces:**
- `FitObservation(frame_id, mask_original, calibration, confidence, boundary_zone, sharpness)`.
- `fit_regular_candidate(model, observations, seed_xy, yaw_values=None) -> CandidateFit`.
- `compare_regular_candidates(models, observations, seed_xy) -> ModelFitEvidence`.

- [ ] Write synthetic RED tests where a side-resting Brick 2 x 4 mask prefers Brick over Plate, a side-resting Plate prefers Plate, Brick 2 x 2 prefers Brick 1 x 2, one corrupted auxiliary view may be trimmed, the only oblique view cannot be trimmed, and insufficient obliquity returns unreliable/ambiguous.
- [ ] Use coarse search at x/y offsets `[-0.012, -0.008, -0.004, 0, 0.004, 0.008, 0.012]` metres and yaw `0..165` by 15 degrees for each resting orientation. Refine the best eight hypotheses with x/y offsets `[-0.004..0.004]` by 0.001 and yaw offsets `[-7.5..7.5]` by 2.5 degrees.
- [ ] Weight views by `confidence * boundary_weight * area_weight`, with boundary weight 0.5 for boundary observations. Mark views oblique at tilt >=30 degrees. Aggregate the weighted mean, optionally trimming the single worst nonessential view only when at least three views remain.
- [ ] Return diagnostic reliability `unavailable`, `low`, or `diagnostic`; never return an accepted production identity or production margin threshold.
- [ ] Run all synthetic tests GREEN and commit.

```bash
git add pipeline/model_fit.py tests/test_model_fit.py
git commit -m "feat: fit regular models across views"
```

---

### Task 5: Diagnostic Serialization and Run-Scoped Overlays

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py`
- Modify: `/Users/emily/lego-cv/session_cli.py`
- Modify: `/Users/emily/lego-cv/tests/test_multiview_bridge.py`
- Modify: `/Users/emily/lego-cv/tests/test_session_cli.py`

**Interfaces:**
- Add `model_fit` evidence beneath each `id_crop` without modifying `brickognize`.
- Add optional CLI `--analysis-dir`; model-fit JSON and overlays go there while existing default output remains backward compatible.

- [ ] Write RED tests proving regular conflicting candidates trigger diagnostic fitting, irregular candidates serialize unavailable, final identities remain byte-for-byte unchanged, and overlays write only beneath the supplied analysis directory.
- [ ] Build candidate models only from accepted/weak per-view top candidates. Load calibration once per scan. Attach observed/predicted/difference overlays for each candidate/view.
- [ ] Assert in tests that geometry cannot enter `_apply_dimension_consistency`, `_apply_stud_advisory`, or final arbitration in this plan.
- [ ] Run focused bridge/CLI tests and the full 259+ CV suite GREEN, then commit.

```bash
git add pipeline/model_fit.py pipeline/multiview.py session_cli.py tests/test_model_fit.py tests/test_multiview_bridge.py tests/test_session_cli.py
git commit -m "feat: record diagnostic regular model fits"
```

---

### Task 6: Synthetic, Primary-Session, and Historical Evaluation Gate

**Files:**
- Generated: `sessions/20260717-233252/analysis-runs/model-fit-diagnostic/`
- Modify: `/Users/emily/lego-capture/PROGRESS.md`

- [ ] Run both full suites and verify the preserved baseline hashes.
- [ ] Run the target session with `--analysis-dir sessions/20260717-233252/analysis-runs/model-fit-diagnostic` and confirm 13 components plus unchanged production identities.
- [ ] Record Brick-versus-Plate and 2x2-versus-1x2 per-view losses, robust totals, best orientations, runner-up margins, and overlays for the two side targets.
- [ ] Run the same diagnostic on labelled anchors in `20260717-202103`, `20260717-005145`, and `20260716-221544` without tuning weights or margins after viewing target results.
- [ ] Report one of two explicit outcomes:
  - **Calibration evidence succeeds:** projection agrees, every synthetic case separates, known historical anchors prefer the labelled family, and no production identity changes. Write a separate Phase 3 promotion/arbitration plan.
  - **Calibration evidence fails:** keep geometry diagnostic-only, identify the failed calibration/mask assumption, and do not write a promotion plan.
- [ ] Record exact-ID errors and review-queue count separately. Never reduce review by forcing geometry guesses.
- [ ] Commit only the measured `PROGRESS.md` report; generated artifacts remain ignored.
