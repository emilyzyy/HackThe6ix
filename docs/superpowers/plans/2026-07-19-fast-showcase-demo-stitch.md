# Fast Showcase Demo Stitch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one demo command that records the accepted live yellow-mask scan, animates 95% to 100%, shows a polished analysis interstitial, prepares three to five safe confirmations through an early-exit CV path, and writes the final inventory handoff.

**Architecture:** One Record3D callback snapshots each frame once, sends the full-rate snapshot to the live renderer, and sends throttled snapshots to the existing session writer. After automatic completion, a coordinator runs table calibration, exports at most three ordered views, invokes a bounded fast-showcase scan, generates a disposable review workspace, and opens the existing compact confirmation UI. The teammate generator remains behind a JSON/URL handoff boundary.

**Tech Stack:** Python 3.11, OpenCV, NumPy, Record3D, Ultralytics YOLO segmentation, FastAPI, vanilla HTML/CSS/JavaScript, pytest.

## Global Constraints

- Live fallback masks remain cosmetic and never become confirmation evidence.
- Coverage is labeled workspace coverage, never percent of pieces identified.
- Display coverage reaches 95%, eases to exactly 100% over four seconds, then automatically stops capture.
- Fast showcase processes at most three selected frames and identifies at most twelve crops.
- Surface at most five safe identities; fewer than three is allowed rather than weakening safety gates.
- Preserve every historical session and review workspace.
- Do not modify the teammate generator; emit a handoff artifact and optional redirect only.
- Do not stage, commit, reset, merge, push, or discard the existing dirty worktrees.

---

### Task 1: Completion Ramp and Analysis Renderer

**Files:**
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Create: `/Users/emily/lego-capture/demo_analysis.py`
- Modify: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`
- Create: `/Users/emily/lego-capture/tests/test_demo_analysis.py`

**Interfaces:**
- Produces: `CoveragePresentation.update(measured_fraction: float, elapsed_s: float) -> float` and `complete: bool`.
- Produces: `render_analysis(frames: list[np.ndarray], stage: str, now_s: float, size_wh: tuple[int, int], masks: list[np.ndarray] = []) -> np.ndarray`.
- Consumes: BGR frames and optional authoritative boolean masks.

- [ ] **Step 1: Add failing coverage-ramp tests**

```python
def test_coverage_presentation_eases_from_95_to_100():
    display = CoveragePresentation(rate_per_s=0.04, completion_s=4.0)
    assert display.update(0.20, 23.75) == pytest.approx(0.95)
    assert display.update(0.20, 25.75) == pytest.approx(0.975)
    assert display.update(0.20, 27.75) == pytest.approx(1.0)
    assert display.complete

def test_coverage_hud_uses_larger_workspace_text(monkeypatch):
    # Record cv2.putText calls and require the workspace line scale >= 1.0.
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py -k 'coverage_presentation or larger_workspace_text' -x`

Expected: FAIL because `CoveragePresentation` and the larger scale do not exist.

- [ ] **Step 3: Implement the stateful presentation ramp**

```python
class CoveragePresentation:
    def __init__(self, rate_per_s=0.04, completion_s=4.0):
        self.rate_per_s = float(rate_per_s)
        self.completion_s = float(completion_s)
        self._ramp_started_at = None
        self.complete = False

    def update(self, measured_fraction, elapsed_s):
        base = max(min(float(measured_fraction), 0.95),
                   min(0.95, max(0.0, elapsed_s) * self.rate_per_s))
        if base < 0.95:
            return base
        if self._ramp_started_at is None:
            self._ramp_started_at = float(elapsed_s)
        progress = min(1.0, max(0.0, elapsed_s - self._ramp_started_at)
                       / self.completion_s)
        self.complete = progress >= 1.0
        return 0.95 + 0.05 * progress
```

Use font scale `1.05`, thickness `3`, and LEGO yellow `(0, 213, 255)` for the workspace line.

- [ ] **Step 4: Add failing analysis-renderer tests**

```python
def test_analysis_renderer_is_deterministic_and_preserves_size():
    out = render_analysis([FRAME_A, FRAME_B, FRAME_C],
                          "Finding clean LEGO pieces...", 1.25, (1280, 720))
    assert out.shape == (720, 1280, 3)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out) > 0

def test_analysis_indicator_changes_with_time():
    assert not np.array_equal(render_at(0.0), render_at(0.4))
```

- [ ] **Step 5: Implement `demo_analysis.py`**

Render a dark yellow-tinted background, three scaled/rotated photo cards, a pulsing yellow core, two expanding rings, three orbiting dots, `Analyzing your LEGO...`, and the supplied real stage label. Composite optional masks in yellow at 28% opacity. Use only `now_s` for animation so rendering never waits on processing.

- [ ] **Step 6: Run focused tests**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py tests/test_demo_analysis.py -x`

Expected: PASS.

---

### Task 2: Single-Client Recording Plus Live Glow

**Files:**
- Modify: `/Users/emily/lego-capture/capture.py`
- Create: `/Users/emily/lego-capture/demo_capture.py`
- Modify: `/Users/emily/lego-capture/tests/test_capture.py`
- Create: `/Users/emily/lego-capture/tests/test_demo_capture.py`

**Interfaces:**
- Produces: `Record3DSource(controller, dev_idx=0, preview_listener=None)`.
- Produces: `snapshot_to_live_frame(snapshot: dict, frame_id: int) -> dict`.
- Produces: `run_demo_capture(anchor_dir: Path, session_dir: Path, detector, diagnostics_path: Path | None) -> Path`.

- [ ] **Step 1: Write failing fan-out tests**

```python
def test_record3d_source_fans_same_snapshot_to_preview_and_writer():
    source = fake_source(preview_listener=preview.append)
    source._on_new_frame()
    assert len(preview) == 1
    assert controller.submitted[0]["timestamp"] == preview[0]["timestamp"]

def test_snapshot_to_live_frame_preserves_rgb_pose_depth_and_intrinsics():
    frame = snapshot_to_live_frame(_snapshot(3, 4.0), frame_id=9)
    assert frame["frame_id"] == 9
    assert frame["image"].shape == (24, 32, 3)
    assert frame["depth"].shape == (6, 8)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_capture.py tests/test_demo_capture.py -x`

Expected: FAIL because preview fan-out and `demo_capture.py` are absent.

- [ ] **Step 3: Refactor the Record3D callback minimally**

Snapshot buffers once, invoke `preview_listener(snapshot)` for every phone frame,
then apply `controller.offer(timestamp)` before submitting that same snapshot to
the writer. Catch listener exceptions, retain the latest error, and stop cleanly
without blocking the Record3D callback.

- [ ] **Step 4: Implement the combined capture loop**

`run_demo_capture` creates `CaptureController`, the full-rate latest-frame
buffer, accepted `LiveGlowSession`, coverage tracker, detector worker, and
`CoveragePresentation`. It renders until `presentation.complete`, disconnects
Record3D, stops workers, flushes `controller.stop()`, and returns the new session
path. It accepts `q` as a safe manual completion path but never deletes a partial
session.

- [ ] **Step 5: Verify the combined path without hardware**

Run: `.venv/bin/pytest -q tests/test_capture.py tests/test_demo_capture.py tests/test_live_scan_viewer.py -x`

Expected: PASS, with assertions that only one source object is constructed and
the session is flushed before return.

---

### Task 3: Bounded Early-Exit Showcase Scan

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py`
- Create: `/Users/emily/lego-cv/pipeline/fast_showcase.py`
- Create: `/Users/emily/lego-cv/fast_showcase_cli.py`
- Create: `/Users/emily/lego-cv/tests/test_fast_showcase.py`
- Modify: `/Users/emily/lego-cv/tests/test_multiview_bridge.py`

**Interfaces:**
- Produces: `rank_showcase_indices(instances, id_crops, limit=12) -> list[int]`.
- Extends: `scan_manifest(..., view_limit: int | None = None, identify_limit: int | None = None, fast_review: bool = False)`.
- Produces: `run_fast_showcase(manifest_path: Path, output_dir: Path, target=5, min_safe=3, max_views=3, identify_limit=12, event_sink=None) -> Path`.

- [ ] **Step 1: Add failing shortlist tests**

```python
def test_shortlist_caps_at_twelve_and_rejects_boundary_instances():
    indices = rank_showcase_indices(instances, id_crops, limit=12)
    assert len(indices) == 12
    assert boundary_index not in indices

def test_scan_manifest_identifies_only_shortlisted_crops(monkeypatch):
    result = scan_manifest(MANIFEST, detector="seg", identify_limit=3,
                           view_limit=1, fast_review=True)
    assert identify_call_sizes == [3]
    assert sum(c["brickognize"]["part_id"] is not None
               for c in result.id_crops) <= 3
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_fast_showcase.py tests/test_multiview_bridge.py -k 'shortlist or identify_limit or view_limit' -x`

Expected: FAIL because the bounded interfaces do not exist.

- [ ] **Step 3: Implement local pre-identification ranking**

Hard-veto incomplete/boundary detections and rank the remainder by segmentation
confidence, identification focus, crop area above the degenerate minimum,
workspace gate evidence, low overlap with neighbors, and spatial separation.
Return at most `limit` indices. Fill unselected identities with the existing
`UNKNOWN` value so result arrays remain aligned.

- [ ] **Step 4: Implement the early-exit runner**

Run with one selected view first. Write real stage events to `event_sink`.
Build the normal showcase payload. If fewer than `min_safe` pieces pass, retry
with the full three-view fallback, reusing one loaded `YoloSegModel` and the
Brickognize disk cache. Stop immediately once `min_safe` pass, while still
capping output at `target` and identification at twelve crops per final run.
Persist `fast-showcase-events.jsonl`, `scene_state.json`, `showcase.json`, and
the normal showcase gallery beneath `output_dir`.

- [ ] **Step 5: Add and run early-exit tests**

```python
def test_fast_showcase_stops_after_first_view_with_three_safe_results():
    path = run_fast_showcase(..., min_safe=3, max_views=3)
    assert attempted_view_limits == [1]
    assert json.loads(path.read_text())["selected_count"] == 3

def test_fast_showcase_expands_to_second_view_when_needed():
    run_fast_showcase(...)
    assert attempted_view_limits == [1, 2]
```

Run: `/Users/emily/lego-cv/.venv/bin/pytest -q tests/test_fast_showcase.py tests/test_showcase.py tests/test_multiview_bridge.py -x`

Expected: PASS.

---

### Task 4: Showcase Review Adapter and Handoff

**Files:**
- Create: `/Users/emily/lego-cv/pipeline/showcase_review.py`
- Modify: `/Users/emily/lego-cv/inventory_review_app.py`
- Modify: `/Users/emily/lego-cv/inventory_review_web/app.js`
- Create: `/Users/emily/lego-cv/tests/test_showcase_review.py`
- Modify: `/Users/emily/lego-cv/tests/test_inventory_review_app.py`

**Interfaces:**
- Produces: `write_showcase_workspace(showcase_path: Path, scene_state_path: Path, session_dir: Path, output_dir: Path, fixed_inventory: Path | None) -> Path`.
- Produces: `POST /api/sessions/{session_id}/handoff` returning the written handoff path.

- [ ] **Step 1: Add failing adapter geometry tests**

```python
def test_adapter_writes_only_selected_showcase_components(tmp_path):
    workspace = write_showcase_workspace(SHOWCASE, STATE, SESSION, tmp_path, None)
    review = load_review(workspace)
    assert len(review["components"]) == SHOWCASE_SELECTED_COUNT
    assert review["source_frames"][0]["image"].startswith("review-assets/")
    assert review["components"][0]["assets"]["views"][0]["mask"]

def test_adapter_mask_matches_source_frame_size():
    assert cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).shape == (height, width)
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_showcase_review.py tests/test_inventory_review_app.py -x`

Expected: FAIL because the adapter and handoff endpoint are absent.

- [ ] **Step 3: Implement disposable review assets**

Join `showcase.json` to `scene_state.json` by `instance_id`. Copy only referenced
source frames into the run directory, rasterize the exact source observation
mask, and call `new_review_document` with one component per selected result.
Set `showcase_only: true`; part-only confirmations may retain an empty color
without weakening standard inventory-review validation.

- [ ] **Step 4: Implement completion handoff**

The handoff endpoint requires every showcase component to be confirmed, then
atomically writes `handoff.json` containing schema version, session ID,
confirmation answers, fixed inventory path, and creation time. It never copies
or edits the fixed inventory.

- [ ] **Step 5: Complete the compact client flow**

After `confirmCurrent()` finds no unresolved components, POST the handoff,
replace the card with `Inventory ready`, pulse the yellow outline once, then
navigate to `generator_url` only when the handoff response includes one.

- [ ] **Step 6: Run adapter, API, and browser-contract tests**

Run: `.venv/bin/pytest -q tests/test_showcase_review.py tests/test_inventory_review_app.py -x`

Expected: PASS.

---

### Task 5: End-to-End Demo Coordinator

**Files:**
- Create: `/Users/emily/lego-capture/demo_flow.py`
- Create: `/Users/emily/lego-capture/tests/test_demo_flow.py`
- Modify: `/Users/emily/lego-capture/README.md`

**Interfaces:**
- Produces CLI: `demo_flow.py --anchor SESSION --fixed-inventory PATH [--generator-url URL] [--replay-session SESSION]`.
- Produces states: `SCANNING`, `COMPLETING`, `ANALYZING`, `CONFIRMING`, `HANDOFF`, `FAILED`.

- [ ] **Step 1: Add failing coordinator-order tests**

```python
def test_coordinator_flushes_before_processing_and_opens_review_last():
    coordinator.run()
    assert events == ["capture", "flush", "plane", "multiview",
                      "fast_showcase", "workspace", "serve", "open"]

def test_processing_failure_preserves_session_and_reports_stage():
    with pytest.raises(DemoStageError) as error:
        coordinator.run()
    assert error.value.stage == "fast_showcase"
    assert session_dir.exists()
```

- [ ] **Step 2: Verify RED**

Run: `.venv/bin/pytest -q tests/test_demo_flow.py -x`

Expected: FAIL because `demo_flow.py` is absent.

- [ ] **Step 3: Implement explicit subprocess commands**

After capture flush:

```text
lego-capture/.venv/bin/python plane.py <session>
lego-capture/.venv/bin/python multiview.py <session> --max-frames 3 --min-frames 1
lego-cv/.venv/bin/python fast_showcase_cli.py <session> --output <run-dir> --target 5 --min-safe 3 --identify-limit 12
```

Read JSONL stage events while rendering `render_analysis` at display cadence.
When the workspace exists, launch uvicorn with
`LEGO_INVENTORY_WORKSPACE=<workspace>` and open its local URL.

- [ ] **Step 4: Add replay mode**

`--replay-session` skips phone capture but exercises completion, analysis,
workspace generation, server launch, and browser handoff from an existing
session. It writes only beneath a new run-scoped directory.

- [ ] **Step 5: Document the one-command demo**

```bash
cd /Users/emily/lego-capture
.venv/bin/python demo_flow.py \
  --anchor sessions/20260718-165402 \
  --fixed-inventory /absolute/path/to/fixed-inventory.json
```

- [ ] **Step 6: Run integration tests**

Run: `.venv/bin/pytest -q tests/test_demo_flow.py tests/test_demo_capture.py tests/test_demo_analysis.py -x`

Expected: PASS.

---

### Task 6: Recorded Replay, Browser QA, and Full Verification

**Files:**
- Modify only if a verified defect is found in Tasks 1-5.
- Produce run-scoped artifacts beneath `/Users/emily/lego-capture/scratchpad/`.

**Interfaces:**
- Consumes: the completed `demo_flow.py --replay-session` path.
- Produces: timing evidence, screenshot/replay artifacts, and the exact phone command.

- [ ] **Step 1: Run both complete test suites**

Run: `cd /Users/emily/lego-capture && .venv/bin/pytest -q`

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest -q`

Expected: zero failures in both repositories.

- [ ] **Step 2: Run a crowded recorded-session replay**

Run the post-capture flow on `sessions/20260718-162834`, writing a fresh
run-scoped output. Record stage durations and selected/rejected counts. Pass if
the system stops at one to three views, identifies no more than twelve crops,
and surfaces only safety-gated results.

- [ ] **Step 3: Perform browser interaction QA**

Verify the full-width source image, exact yellow mask, compact card, Yes advance,
No correction slide, final `Inventory ready` state, handoff JSON, and absence of
console errors.

- [ ] **Step 4: Run scoped diff checks**

Run `git diff --check` on every modified implementation/test/spec/plan file in
both repositories. Do not stage or commit.

- [ ] **Step 5: Hand off the final phone command**

Provide the single `demo_flow.py` command and a concise physical acceptance
checklist. State recorded evidence separately from the remaining phone test.
