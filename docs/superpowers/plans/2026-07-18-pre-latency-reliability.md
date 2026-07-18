# Pre-Latency Inventory Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make final colors match Emily's physical 12-color inventory, prevent the confirmed same-frame duplicate from becoming two fused pieces, and choose locally sharp evidence for identification without changing Brickognize concurrency or adding DINO.

**Architecture:** Preserve the 24-color reference palette for per-view evidence and fusion hints, but add a separate 12-color output profile used only by final color aggregation. Suppress only near-identical accepted masks within one source view before fusion, retaining the rejected duplicate as provenance. Add mask-interior focus evidence to observations, use it in identification-view ranking, and retain the strongest weak view for review while leaving its identity unknown.

**Tech Stack:** Python 3.11, NumPy, OpenCV, scikit-image, pytest. No new dependency.

## Global Constraints

- Final output colors are exactly: red, orange, yellow, beige, brown, green, dark green, blue, white, light gray, dark gray, black.
- Keep the full 24-color palette for diagnostic evidence and association; do not alter segmentation color families.
- Never merge detections from different views in the duplicate-suppression step.
- Do not suppress adjacent pieces whose boxes overlap but masks do not.
- Do not lower the Brickognize 0.70 identity threshold.
- Do not change Brickognize request count, throttling, concurrency, DINO, or exemplar-bank behavior.
- Never overwrite historical session artifacts; real-session validation writes to a new `--analysis-dir`.

---

### Task 1: Inventory Color Output Profile

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/color.py`
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py`
- Test: `/Users/emily/lego-cv/tests/test_color.py`
- Test: `/Users/emily/lego-cv/tests/test_multiview.py`

**Interfaces:**
- Produces: `INVENTORY_COLOR_NAMES: tuple[str, ...]` and `INVENTORY_PALETTE: list[LegoColor]`.
- Consumes: `rank_color_candidates(rgb, colors=INVENTORY_PALETTE)` during final aggregation only.

- [ ] **Step 1: Write failing palette-profile tests**

```python
def test_inventory_palette_matches_physical_demo_bucket():
    assert tuple(color.name for color in INVENTORY_PALETTE) == (
        "red", "orange", "yellow", "beige", "brown", "green",
        "dark green", "blue", "white", "light gray", "dark gray", "black",
    )

def test_final_consensus_cannot_emit_dark_red():
    # Full reference evidence may describe dark-red pixels, but final output
    # must resolve them against the physical inventory profile.
    assert consensus_color(instance).name == "red"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_color.py tests/test_multiview.py -k 'inventory_palette or cannot_emit' -v`

Expected: import/assertion failure because the output profile does not exist and final aggregation still considers all 24 colors.

- [ ] **Step 3: Implement the minimal separate output profile**

```python
INVENTORY_COLOR_NAMES = (
    "red", "orange", "yellow", "beige", "brown", "green",
    "dark green", "blue", "white", "light gray", "dark gray", "black",
)
INVENTORY_PALETTE = [
    color for color in PALETTE if color.name in INVENTORY_COLOR_NAMES
]
```

Import `INVENTORY_PALETTE` in `pipeline.multiview` and pass it only to the final `rank_color_candidates` call. Make the legacy no-evidence branch rematch its winning reference RGB against `INVENTORY_PALETTE`. Include `output_palette` in final color provenance.

- [ ] **Step 4: Run focused and complete color/multiview tests**

Run: `.venv/bin/pytest tests/test_color.py tests/test_multiview.py -v`

Expected: all selected tests pass, and existing ambiguity/quality-weighting behavior remains green.

- [ ] **Step 5: Commit the independently testable color-profile change**

```bash
git add pipeline/color.py pipeline/multiview.py tests/test_color.py tests/test_multiview.py
git commit -m "fix: restrict final colors to inventory profile"
```

### Task 2: Conservative Same-View Duplicate Suppression

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/segdetect.py`
- Test: `/Users/emily/lego-cv/tests/test_segdetect.py`

**Interfaces:**
- Produces: `suppress_same_view_duplicates(detections: list[SegmentationDetection]) -> list[SegmentationDetection]`.
- Consumes: accepted mapped masks, view index, confidence, and gate provenance.

- [ ] **Step 1: Write failing suppression tests**

```python
def test_same_view_near_identical_masks_keep_higher_confidence():
    result = suppress_same_view_duplicates([strong, weak])
    assert result[0].accepted
    assert not result[1].accepted
    assert result[1].gate.reason == "duplicate_suppressed"

def test_overlapping_boxes_with_disjoint_piece_masks_are_preserved():
    result = suppress_same_view_duplicates([left_piece, right_piece])
    assert all(detection.accepted for detection in result)

def test_identical_masks_from_different_views_are_preserved():
    result = suppress_same_view_duplicates([first_view, second_view])
    assert all(detection.accepted for detection in result)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_segdetect.py -k 'duplicate or disjoint' -v`

Expected: import failure because `suppress_same_view_duplicates` does not exist.

- [ ] **Step 3: Implement inspectable mask-level suppression**

Within each view, compare accepted detections in descending confidence order. Mark the weaker detection `duplicate_suppressed` only when mask IoU is at least 0.65, smaller-mask containment is at least 0.85, mask area ratio is at least 0.70, and original-box IoU is at least 0.60. Preserve every detection in the returned audit list; only its gate acceptance changes. Call the helper once after `detect_original_instances` has gathered detections.

- [ ] **Step 4: Run focused segmentation and fusion tests**

Run: `.venv/bin/pytest tests/test_segdetect.py tests/test_multiview.py -v`

Expected: all tests pass; same-view fusion constraints remain unchanged.

- [ ] **Step 5: Commit the independently testable duplicate fix**

```bash
git add pipeline/segdetect.py tests/test_segdetect.py
git commit -m "fix: suppress duplicate masks before fusion"
```

### Task 3: Piece-Specific Focus Ranking and Best Weak Evidence

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py`
- Test: `/Users/emily/lego-cv/tests/test_multiview_bridge.py`

**Interfaces:**
- Produces: `_masked_focus_score(image, mask) -> float` and `Observation.identification_focus: float | None`.
- Consumes: original RGB pixels and the eroded instance interior.
- Produces provenance keys: `identification_focus`, `identification_quality`, and informational flag `best_weak_view_selected`.

- [ ] **Step 1: Write failing focus and weak-evidence tests**

```python
def test_masked_focus_score_prefers_sharp_piece_interior():
    assert _masked_focus_score(sharp, mask) > _masked_focus_score(blurred, mask)

def test_id_observation_prefers_local_sharpness_over_global_view_quality():
    chosen, _ = _identification_observation(instance, views)
    assert chosen is locally_sharp

def test_unresolved_identity_displays_highest_scoring_weak_view():
    identifications, crops, id_crops = _second_chance(...)
    assert identifications[0].part_id is None
    assert identifications[0].score == pytest.approx(0.68)
    assert id_crops[0]["frame_id"] == 2
    assert "best_weak_view_selected" in id_crops[0]["flags"]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_multiview_bridge.py -k 'local_sharpness or masked_focus or highest_scoring_weak' -v`

Expected: import/assertion failures because local focus and weak-review selection are absent.

- [ ] **Step 3: Implement local focus evidence and ranking**

Erode the instance mask before applying a Laplacian so the segmentation boundary cannot manufacture sharpness. Use the 80th percentile absolute Laplacian response as focus. Combine focus with original-frame fit, view tilt, segmentation confidence, and crop area in the identification ranking; use existing whole-view quality only as a fallback/tiebreaker. Serialize the focus and combined ranking score in crop provenance.

- [ ] **Step 4: Preserve unknown safety while selecting useful review evidence**

When consensus has no accepted part ID, select the internal record with the highest weak API score, breaking ties with identification quality. Update the displayed crop/provenance and append `best_weak_view_selected`; do not assign a part ID and do not lower `MIN_SCORE`.

- [ ] **Step 5: Run focused bridge tests**

Run: `.venv/bin/pytest tests/test_multiview_bridge.py -v`

Expected: all bridge tests pass.

- [ ] **Step 6: Commit the independently testable focus change**

```bash
git add pipeline/multiview.py tests/test_multiview_bridge.py
git commit -m "fix: rank identification views by local focus"
```

### Task 4: Full Regression and Run-Scoped Session Validation

**Files:**
- Modify: `/Users/emily/lego-capture/PROGRESS.md`
- Create: `/Users/emily/lego-capture/sessions/20260718-130245/analysis-runs/pre-latency-reliability/` via the existing CLI.

**Interfaces:**
- Consumes: session `20260718-130245` and existing cached Brickognize responses.
- Produces: run-scoped `multiview_result.json`, inventory, provenance, and contact sheets without altering historical outputs.

- [ ] **Step 1: Run complete repository tests**

Run in `/Users/emily/lego-capture`: `.venv/bin/pytest -q`

Expected: 80 passed.

Run in `/Users/emily/lego-cv`: `.venv/bin/pytest -q`

Expected: at least 294 plus the newly added tests, zero failures.

- [ ] **Step 2: Rerun the real session into a new analysis directory**

```bash
.venv/bin/python session_cli.py \
  /Users/emily/lego-capture/sessions/20260718-130245/multiview/manifest.json \
  --detector seg \
  --analysis-dir /Users/emily/lego-capture/sessions/20260718-130245/analysis-runs/pre-latency-reliability
```

Expected: historical artifacts remain untouched; output final colors are all within the 12-color profile; duplicate-suppression provenance exists; unresolved identities remain unknown while exposing their strongest weak view.

- [ ] **Step 3: Compare before/after evidence without inventory-order assumptions**

Check the confirmed white Plate 1×6 duplicate by its source detection IDs/geometry rather than component number. Report total fused count, suppressed source IDs, active final colors, unknown count, and whether the displayed evidence for the upside-down 2×2 pieces improved. Do not claim part-ID accuracy without labeled truth.

- [ ] **Step 4: Record the gate in PROGRESS.md and commit documentation**

```bash
git add PROGRESS.md docs/superpowers/plans/2026-07-18-pre-latency-reliability.md
git commit -m "docs: record pre-latency reliability gate"
```

