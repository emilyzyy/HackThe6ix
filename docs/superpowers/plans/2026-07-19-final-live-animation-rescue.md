# Final Live Animation Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the live LEGO glow continue for an arbitrary full-table phone pan by fixing completion-time flash activation, measuring every post-YOLO gate, and guaranteeing a depth-filtered screen-space animation when stale table geometry produces no visible glow.

**Architecture:** Preserve the accurate table-anchored path when it produces visible output. Add a bounded animation-only fallback that runs at most one extra full-RGB inference on a starving pass, rejects masks without plausible live depth/physical size, selects 3–4 separated regions with cooldown, and emits short yellow screen-space flashes. Keep the final batch CV pipeline and yellow confirmation UI separate until the live phone acceptance test passes.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Record3D RGB/depth/poses, Ultralytics through the existing `YoloSegModel`, pytest.

## Global Constraints

- This is the final live-animation architecture attempt; no model retraining, second MPS model, native iOS work, optical-flow subsystem, ARWorldMap, or new current-session plane fitter.
- Keep the accurate batch CV pipeline unchanged.
- Keep the existing latest-frame-only worker and never queue stale frames.
- Never run more than one extra full-RGB YOLO call on a starving segmentation pass.
- A screen fallback mask must pass real confidence, depth-support, and physical-size checks; large/background masks must not flash.
- The animation fallback may rediscover a piece after cooldown because it is not authoritative inventory identity.
- Use a yellow LEGO theme for live masks and scanner bar to match confirmation.
- Target 3–4 masks per fallback pass, 8% of image diagonal minimum separation, 1.5-second region cooldown, 0.45–0.50-second flash lifetime, and 0.75-second maximum result age.
- Preserve all user-owned dirty-tree changes and artifacts. Do not stage, commit, reset, discard, merge, or push.

## Acceptance gates

The rescue is accepted only if all of the following hold:

1. On a long crowded recorded replay, YOLO continues through the last sampled frames and visible animation does not remain at zero for more than two consecutive eligible detection periods while depth-plausible masks exist.
2. A deliberately invalid table/workspace transform still produces depth-plausible, spatially separated screen flashes.
3. An oversized polygon and a polygon without adequate depth support are rejected from the fallback.
4. The RGB display loop remains independent of inference and the latest-frame scheduler still drops stale frames.
5. On the real ~800-piece phone pan, segmentation-pass count continues increasing to the end, late-pan fallback activity is visible, and the terminal diagnostics identify every zero-output pass.
6. Only after Gate 5 passes do we process the new scan and point the yellow confirmation flow at top-confidence pieces from that same scan.

---

### Task 1: Make visible-output accounting and completion timing truthful

**Files:**
- Modify: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Modify: `/Users/emily/lego-capture/live_glow.py`
- Modify: `/Users/emily/lego-capture/tests/test_live_glow.py`

**Interfaces:**
- Consumes: activated `Track`/`ScreenFlash` objects, current homography, current frame shape, and inference submission/completion timestamps.
- Produces: `visible_glow_count(tracks, homography, image_shape, now) -> int`; `LiveGlowSession.detect(...) -> dict`; completion-time `activation_time` values.

- [ ] **Step 1: Write a failing test proving faded tracks are not visible**

```python
def test_live_render_count_excludes_faded_tracks():
    K = intrinsics_to_K(900.0, 900.0, 720.0, 540.0)
    pose = look_at_pose([0.0, 0.0, 0.4], [0.0, 0.0, 0.0])
    polygon = np.array([[-0.02, -0.02], [0.02, -0.02],
                        [0.02, 0.02], [-0.02, 0.02]])
    detector = _FixedImageDetector(project_polygon(
        polygon, table_to_image_homography(K, pose, np.eye(4))
    ))
    session = LiveGlowSession(
        detector, np.eye(4), min_confidence=0.6,
        activation_stagger_s=0.0,
    )
    frame = _frame(0, pose, K)
    session.detect(frame, now=0.0)
    _, count = session.render(frame, now=live_scan_viewer.FLASH_END_S)
    assert count == 0
```

- [ ] **Step 2: Run the test and confirm RED**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py -k excludes_faded -x`

Expected: FAIL because `render()` currently counts every activated track until its 30-second TTL, even after its pixels have faded.

- [ ] **Step 3: Add one visibility helper and use it for render counts**

Implement a helper that requires both a non-zero `glow_envelope(now - activation_time)` and a projected polygon intersecting the current frame. Filter table and screen tracks through that helper before returning the HUD/diagnostic count. Continue passing the same track lists to `render_glow`, which already skips faded/out-of-frame polygons.

- [ ] **Step 4: Write a failing completion-clock test**

```python
def test_detect_activates_at_inference_completion_not_submission():
    workspace = np.array([
        [-0.12, -0.12], [0.12, -0.12],
        [0.12, 0.12], [-0.12, 0.12],
    ])
    session = LiveGlowSession(
        _FixedImageDetector(_square_at(720, 540, half=20)),
        np.eye(4), workspace_table=workspace,
        min_confidence=0.6, activation_stagger_s=0.0,
        clock=lambda: 0.40,
    )
    frame = _depth_frame(depth_m=0.50, fx=900.0, fy=900.0)
    monkeypatch.setattr(
        live_scan_viewer, "_workspace_crop",
        lambda *args: (_ for _ in ()).throw(ValueError("outside")),
    )
    stats = session.detect(frame, now=0.0)
    assert stats["inference_latency_ms"] == 400.0
    assert session.screen_flashes[0].activation_time == 0.40
```

Use a workspace-outside/full-screen setup in the final test fixture so it exercises a screen flash without relying on invalid identity geometry.

- [ ] **Step 5: Run the completion-clock test and confirm RED**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py -k inference_completion -x`

Expected: FAIL because `LiveGlowSession` has no completion clock and uses the pre-inference `now` value.

- [ ] **Step 6: Activate new tracks/flashes with a completion timestamp**

Add an optional monotonic-relative `clock` callable to `LiveGlowSession`. In production, create `wall0` before the session and pass `clock=lambda: time.monotonic() - wall0`; in deterministic tests, omit it or inject a fake. Record `submitted_at`, `completed_at`, and `inference_latency_ms` in the returned pass statistics. Never add `torch.mps.synchronize()` to the normal hot path.

- [ ] **Step 7: Run focused tests**

Run: `.venv/bin/pytest -q tests/test_live_glow.py tests/test_live_scan_viewer.py -k 'faded or completion or glow_envelope' -x`

Expected: PASS.

---

### Task 2: Add a live depth/physical-size plausibility gate

**Files:**
- Modify: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`

**Interfaces:**
- Consumes: a full-RGB polygon, confidence, `frame["depth"]`, full-RGB intrinsics, and RGB/depth shapes.
- Produces: `screen_piece_evidence(polygon_px, confidence, frame) -> ScreenPieceEvidence` with `accepted`, `reason`, `depth_support`, `median_depth_m`, `width_m`, `length_m`, and `area_m2`.

- [ ] **Step 1: Write failing gate tests**

```python
def test_screen_piece_gate_accepts_depth_supported_lego_size():
    frame = _depth_frame(depth_m=0.50, fx=900.0, fy=900.0)
    evidence = live_scan_viewer.screen_piece_evidence(
        _square_at(720, 540, half=35), 0.90, frame
    )
    assert evidence.accepted
    assert evidence.reason == "accepted"

def test_screen_piece_gate_rejects_oversized_mask():
    frame = _depth_frame(depth_m=0.70, fx=900.0, fy=900.0)
    evidence = live_scan_viewer.screen_piece_evidence(
        _square_at(720, 540, half=220), 0.95, frame
    )
    assert not evidence.accepted
    assert evidence.reason == "oversized"

def test_screen_piece_gate_rejects_missing_depth_support():
    frame = _depth_frame(depth_m=0.0, fx=900.0, fy=900.0)
    evidence = live_scan_viewer.screen_piece_evidence(
        _square_at(720, 540, half=35), 0.90, frame
    )
    assert not evidence.accepted
    assert evidence.reason == "depth_unsupported"
```

- [ ] **Step 2: Run gate tests and confirm RED**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py -k screen_piece_gate -x`

Expected: FAIL because the evidence type/gate does not exist.

- [ ] **Step 3: Implement the gate**

Rasterize only the polygon's bounding region at depth resolution. Count finite depth samples in `[0.08, 1.5]` metres, require at least six valid samples and 25% polygon depth support, use their median depth, and convert `cv2.minAreaRect` pixel dimensions to metres with the mean focal length. Reuse `MAX_PIECE_DIMENSION_M=0.16` and `MAX_PIECE_FOOTPRINT_AREA_M2=0.008`. Reject confidence below the session's `min_confidence` before raster work.

- [ ] **Step 4: Add recorded depth to deterministic frames**

Change `RecordedFrameSource.frames()` to include `record.load_depth()` so replay exercises the same fallback gate as the phone path. Do not alter saved session files.

- [ ] **Step 5: Run gate and recorded-source tests**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py -k 'screen_piece_gate or recorded' -x`

Expected: PASS.

---

### Task 3: Add cooldown-aware, spatially separated screen flashes

**Files:**
- Modify: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`
- Modify: `/Users/emily/lego-capture/live_glow.py`
- Modify: `/Users/emily/lego-capture/tests/test_live_glow.py`

**Interfaces:**
- Consumes: depth-approved `(polygon, confidence)` candidates, image shape, and completion time.
- Produces: `ScreenFlashScheduler.select(candidates, image_shape, now) -> list`; `ScreenFlashScheduler.arm(selected, image_shape, now)`.

- [ ] **Step 1: Write failing scheduler tests**

```python
def test_screen_scheduler_spaces_and_cools_regions():
    scheduler = live_scan_viewer.ScreenFlashScheduler(
        max_count=4, min_spacing_fraction=0.08, cooldown_s=1.5
    )
    candidates = [
        (_square_at(100 + 180 * i, 200, half=20), 0.95 - 0.01 * i)
        for i in range(6)
    ]
    first = scheduler.select(candidates, (1080, 1440, 3), now=0.0)
    scheduler.arm(first, (1080, 1440, 3), now=0.0)
    second = scheduler.select(candidates, (1080, 1440, 3), now=0.5)
    assert len(first) == 4
    assert not {scheduler.cell(p, (1080, 1440, 3)) for p, _ in first} & \
        {scheduler.cell(p, (1080, 1440, 3)) for p, _ in second}

def test_screen_scheduler_fail_opens_when_every_region_is_cooling():
    scheduler = live_scan_viewer.ScreenFlashScheduler(max_count=4)
    candidates = [(_square_at(720, 540, half=20), 0.95)]
    first = scheduler.select(candidates, (1080, 1440, 3), now=0.0)
    scheduler.arm(first, (1080, 1440, 3), now=0.0)
    assert len(scheduler.select(
        candidates, (1080, 1440, 3), now=0.5, guarantee_one=True
    )) == 1
```

- [ ] **Step 2: Run scheduler tests and confirm RED**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py -k screen_scheduler -x`

Expected: FAIL because the scheduler does not exist.

- [ ] **Step 3: Implement the scheduler**

Use normalized coarse cells only for the 1.5-second cooldown and reuse `_select_spaced_masks` for farthest-point selection. Prefer ready cells; when `guarantee_one=True` and every depth-plausible candidate is cooling, select one highest-confidence candidate anyway so cooldown can never create another dark screen.

- [ ] **Step 4: Shorten and recolor the live effect**

Set the discovery flash end to `0.48` seconds and use LEGO yellow/gold BGR for `GLOW_HUE_BGR`, which also colors the scanner sweep. Update `test_glow_envelope_flashes_then_turns_off` to assert the new end time through the constant rather than a duplicated literal.

- [ ] **Step 5: Run scheduler and glow tests**

Run: `.venv/bin/pytest -q tests/test_live_glow.py tests/test_live_scan_viewer.py -k 'screen_scheduler or glow_envelope or scanner_sweep' -x`

Expected: PASS.

---

### Task 4: Guarantee a last-mile flash when the table path starves

**Files:**
- Modify: `/Users/emily/lego-capture/tests/test_live_scan_viewer.py`
- Modify: `/Users/emily/lego-capture/live_scan_viewer.py`

**Interfaces:**
- Consumes: table-path raw/selected/rejected counts, genuinely visible table/screen glow count, full live frame, and `ScreenFlashScheduler`.
- Produces: exactly one `detect()` result dictionary per pass with gate counts and a `fallback_reason`/`fallback_flash_count`.

- [ ] **Step 1: Write a failing starvation regression test**

```python
def test_raw_masks_rejected_by_table_still_emit_depth_safe_fallback(monkeypatch):
    workspace = np.array([
        [-0.12, -0.12], [0.12, -0.12],
        [0.12, 0.12], [-0.12, 0.12],
    ])
    detector = _RelativeCenterDetector(confidence=0.90, half=20)
    session, frame, _ = _workspace_session(detector, workspace)
    frame["depth"] = np.full((192, 256), 0.55, np.float32)
    monkeypatch.setattr(live_scan_viewer, "_workspace_overlap", lambda *a: 0.0)
    stats = session.detect(frame, now=0.0)
    _, visible = session.render(frame, now=0.05)
    assert stats["raw_mask_count"] > 0
    assert stats["workspace_rejected_count"] > 0
    assert stats["fallback_flash_count"] >= 1
    assert stats["fallback_reason"] == "no_visible_table_output"
    assert visible >= 1
```

- [ ] **Step 2: Run the starvation regression and confirm RED**

Run: `.venv/bin/pytest -q tests/test_live_scan_viewer.py -k rejected_by_table_still_emit -x`

Expected: FAIL because non-empty raw detections rejected downstream do not invoke fallback.

- [ ] **Step 3: Implement one bounded fallback decision**

After the table tracker update, compute actual visible table/screen output at completion time. If it is zero and the result is no older than 0.75 seconds, obtain full-frame detections (reuse the existing full-frame call if already made; otherwise run exactly one additional detector call), apply `screen_piece_evidence`, select with the scheduler, and append screen flashes at completion time. Do not trigger merely because `new_track_count == 0` when a recent glow is still visible.

- [ ] **Step 4: Return complete pass diagnostics**

Return a dictionary with at least:

```python
{
    "frame_id": frame["frame_id"],
    "inference_latency_ms": ...,
    "result_age_ms": ...,  # total latest-frame inference age; no queued work
    "raw_mask_count": ...,
    "selected_mask_count": ...,
    "workspace_rejected_count": ...,
    "size_rejected_count": ...,
    "accepted_table_count": ...,
    "new_track_count": ...,
    "visible_table_count": ...,
    "visible_screen_count": ...,
    "depth_rejected_count": ...,
    "fallback_reason": ...,
    "fallback_flash_count": ...,
}
```

Print one compact line per live segmentation pass and optionally append JSONL through a new `--diagnostics PATH` argument. The display thread must not write logs.

- [ ] **Step 5: Add a faded-track regression**

Detect/activate a table track at `t=0`, advance beyond `FLASH_END_S` while refreshing the same identity, then assert that a new pass reports zero visible table glows and emits a depth-safe screen flash. This specifically covers the old 30-second activate-once starvation case.

- [ ] **Step 6: Run live unit/integration tests**

Run: `.venv/bin/pytest -q tests/test_live_glow.py tests/test_live_scan_viewer.py -x`

Expected: PASS.

---

### Task 5: Prove the rescue on long crowded recorded data

**Files:**
- Modify only if a test exposes a defect: `/Users/emily/lego-capture/live_scan_viewer.py`
- Create artifact: `/Users/emily/lego-capture/scratchpad/final-live-animation-replay.mp4`
- Create artifact: `/Users/emily/lego-capture/scratchpad/final-live-animation-replay.jsonl`

**Interfaces:**
- Consumes: the existing crowded session `/Users/emily/lego-capture/sessions/20260718-162834` and the real model `/Users/emily/lego-cv/models/lego_seg.pt`.
- Produces: a full replay preview plus pass-level diagnostic evidence.

- [ ] **Step 1: Run all capture tests before the expensive replay**

Run: `.venv/bin/pytest -q`

Expected: all tests pass.

- [ ] **Step 2: Run a deterministic headless crowded replay**

Run:

```bash
cd /Users/emily/lego-capture
.venv/bin/python live_scan_viewer.py \
  sessions/20260718-162834 \
  --headless \
  --detect-every 10 \
  --out scratchpad/final-live-animation-replay.mp4 \
  --diagnostics scratchpad/final-live-animation-replay.jsonl
```

- [ ] **Step 3: Audit the diagnostic tail and dark-gap invariant**

Parse the JSONL and assert:

- completed frame IDs reach the final sampled portion of the recording;
- every pass with depth-plausible masks and zero table visibility either emits a fallback flash or records `stale_result` explicitly;
- no fallback polygon classified `oversized` or `depth_unsupported` is emitted;
- the maximum consecutive eligible zero-visible passes is two.

- [ ] **Step 4: Inspect start, middle, and final-third frames**

Extract representative frames from the MP4 and verify yellow masks are spatially separated, do not create crop rectangles, and remain present in the final third. If this fails, permit one threshold-only adjustment; do not introduce another architecture.

---

### Task 6: Run the one real full-box phone acceptance test

**Files:**
- Create artifact: `/Users/emily/lego-capture/scratchpad/final-phone-animation.jsonl`

**Interfaces:**
- Consumes: connected Record3D phone, dumped ~800-piece workspace, and existing anchor session geometry.
- Produces: the final go/no-go verdict for the live animation.

- [ ] **Step 1: Start the final phone viewer**

Run:

```bash
cd /Users/emily/lego-capture
.venv/bin/python live_scan_viewer.py \
  --live \
  --world-to-table sessions/20260719-032336 \
  --diagnostics scratchpad/final-phone-animation.jsonl
```

- [ ] **Step 2: Perform one deliberate pan**

Pan slowly for 20–30 seconds using the intended demo movement. Include one direction change and ensure new table regions enter the frame during the final third.

- [ ] **Step 3: Judge only the acceptance gates**

Pass if the RGB stays live, the segmentation-pass counter continues increasing, separated yellow masks appear during the final third, no obvious table/hand/large-object masks flash, and coverage/scanner UI continues. Do not reject for imperfect persistent identity during a fast motion; the fallback is intentionally cosmetic.

- [ ] **Step 4: Use the diagnostic log for a single decision**

If visual output fails, inspect the last ten JSONL events. Allow one numeric threshold correction only when the log proves `depth_unsupported`, `oversized`, or cooldown is the sole blocker. Otherwise stop changing the live architecture and preserve the diagnostic evidence.

---

### Task 7: Test confirmation on the new large scan, then stitch

**Files:**
- Reuse: `/Users/emily/lego-cv/inventory_review_web/`
- Reuse: the new scan's `showcase.json`, `scene_state.json`, and source-frame assets.
- Plan separately before implementation: a showcase-to-confirmation adapter and the final generation handoff.

**Interfaces:**
- Consumes: the accepted new full-box scan and its top-confidence showcase selections.
- Produces: yellow Yes/No confirmation over pieces from the same ~800-piece scan, followed by the apparent inventory handoff.

- [ ] **Step 1: Run the final batch CV path on the accepted new session**

Generate `scene_state.json`, `showcase.json`, and `debug/showcase_gallery.jpg` with `--showcase 5` using the existing `lego-cv/session_cli.py` path.

- [ ] **Step 2: Manually verify the five showcase identities**

Reject any known ambiguous plate/brick case before putting it in front of the audience. Fewer than five is acceptable.

- [ ] **Step 3: Point a disposable confirmation workspace at the new source frames**

Use the existing yellow mask UI without modifying the original session. The primary view is full-width with no permanent right inspector. Only the compact yellow confirmation card floats above the image; No swaps that same card into the type-only correction state and Back/Save restores the compact state. Verify the actual dense-pile frame loads, mask placement remains exact, Yes advances by confidence, and the compact correction flow works.

- [ ] **Step 4: Create the stitch plan only after both demos pass independently**

The stitch should automate: live scan completion → batch command → top showcase payload → yellow confirmation → apparent inventory handoff. Do not couple the live tracker to authoritative inventory identity.

## Execution order and stop rule

Execute Tasks 1–5 now. Task 6 is the user's one real-phone acceptance run. Do not start Task 7 stitching work until Task 6 passes. This preserves the remaining time for confirmation and integration instead of reopening the live architecture again.
