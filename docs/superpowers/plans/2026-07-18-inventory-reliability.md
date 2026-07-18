# Inventory Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make capture selection cover the complete admissible workspace, choose genuinely distinct 3D views, and prevent weak identification heuristics from overruling stronger evidence.

**Architecture:** `lego-capture` will choose candidates on a coarse hard-workspace canvas, then rectify only three to five selected frames at full resolution. `lego-cv` will review semantically ambiguous regular parts across original RGB views, aggregate metric and stud evidence under explicit precedence rules, and record shadow-only depth height without changing identities.

**Tech Stack:** Python 3.11, NumPy, OpenCV, Record3D 1.4.1, Ultralytics YOLO11 segmentation, requests, pytest.

## Global Constraints

- Color classification remains unchanged.
- The cosmetic Record3D live visualization remains unchanged.
- Selection retains a minimum of three and maximum of five exported views.
- Selection targets at least 99.5% of the valid-pixel union available inside the hard admissible region.
- The export canvas spans the complete hard-contour bounds.
- Stud evidence cannot independently replace or demote an identification.
- Depth height is shadow-only and cannot change inventory identities.
- Session `20260717-233252` remains a regression fixture, not segmentation training data.
- Historical records without confidence sidecars remain readable.
- Production behavior changes require a failing test observed before implementation.

---

### Task 1: Hard-Workspace Canvas and Coarse Candidate Coverage

**Files:**
- Modify: `/Users/emily/lego-capture/coverage.py:189-202`
- Modify: `/Users/emily/lego-capture/multiview.py:31-45,178-354`
- Modify: `/Users/emily/lego-capture/tests/test_coverage.py:177-225`
- Modify: `/Users/emily/lego-capture/tests/test_multiview.py:82-138`

**Interfaces:**
- Consumes: `CoverageGrid.admissible_workspace_mask(min_seen=2, close_cells=5)`.
- Produces: `CoverageGrid.admissible_workspace_bounds_xy(min_seen=2, close_cells=5, pad=0.0)`, `_workspace_mask(contours_xy, origin_xy, px_per_m, size_wh)`, and a manifest whose coverage denominator is the hard workspace rather than its bounding rectangle.

- [ ] **Step 1: Add failing hard-bounds and full-canvas tests**

```python
def test_admissible_workspace_bounds_cover_outer_region_not_dense_core():
    grid = CoverageGrid([[0.0, 0.2], [0.0, 0.2]], cell_m=0.01)
    grid.seen_count[1:19, 2:18] = 2
    grid.seen_count[7:13, 7:13] = 20
    assert grid.admissible_workspace_bounds_xy() == pytest.approx(
        [[0.02, 0.18], [0.01, 0.19]], abs=0.011
    )
    assert grid.workspace_bounds_xy() != grid.admissible_workspace_bounds_xy()


def test_export_canvas_uses_admissible_bounds(session, monkeypatch):
    from multiview import _canvas_geometry
    table, grid, origin, size = _canvas_geometry(session, 1000.0)
    hard = grid.admissible_workspace_bounds_xy()
    assert origin == pytest.approx((hard[0][0], hard[1][0]))
    assert size == (
        int(np.ceil((hard[0][1] - hard[0][0]) * 1000.0)),
        int(np.ceil((hard[1][1] - hard[1][0]) * 1000.0)),
    )
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_coverage.py::test_admissible_workspace_bounds_cover_outer_region_not_dense_core tests/test_multiview.py::test_export_canvas_uses_admissible_bounds -v`

Expected: FAIL because `admissible_workspace_bounds_xy` is absent and `_canvas_geometry` still uses `workspace_bounds_xy`.

- [ ] **Step 3: Implement hard-workspace bounds**

```python
def _bounds_for_mask(self, mask, pad=0.0):
    if not mask.any():
        return None
    ys, xs = np.nonzero(mask)
    (gx0, _), (gy0, _) = self.bounds
    return [[gx0 + xs.min() * self.cell_m - pad,
             gx0 + (xs.max() + 1) * self.cell_m + pad],
            [gy0 + ys.min() * self.cell_m - pad,
             gy0 + (ys.max() + 1) * self.cell_m + pad]]

def admissible_workspace_bounds_xy(self, min_seen=2, close_cells=5, pad=0.0):
    return self._bounds_for_mask(
        self.admissible_workspace_mask(min_seen, close_cells), pad
    )
```

Refactor `workspace_bounds_xy` to call `_bounds_for_mask`, then make `_canvas_geometry` clamp `admissible_workspace_bounds_xy()` to `table_frame["extent_xy"]`.

- [ ] **Step 4: Add a failing test proving candidate images are not retained at export resolution**

```python
def test_export_rectifies_all_candidates_coarsely_and_selected_views_fully(
    session, monkeypatch
):
    calls = []
    real = multiview._rectify
    def recording_rectify(record, table, origin, scale, size):
        calls.append((record.frame_id, scale, size))
        return real(record, table, origin, scale, size)
    monkeypatch.setattr(multiview, "_rectify", recording_rectify)
    _, manifest = export_multiview(session, px_per_mm=1.0, max_frames=3)
    full = [call for call in calls if call[1] == 1000.0]
    coarse = [call for call in calls if call[1] < 1000.0]
    assert len(full) == len(manifest["views"])
    assert len(coarse) == 16
```

- [ ] **Step 5: Run the coarse-selection test and verify RED**

Run: `.venv/bin/pytest tests/test_multiview.py::test_export_rectifies_all_candidates_coarsely_and_selected_views_fully -v`

Expected: FAIL because every frame is currently rectified and retained at the final scale.

- [ ] **Step 6: Implement two-resolution selection**

Use `SELECTION_PX_PER_M = 250.0`. Rasterize `hard_contours_table_xy` into a boolean mask at each scale:

```python
def _workspace_mask(contours_xy, origin_xy, px_per_m, size_wh):
    mask = np.zeros((size_wh[1], size_wh[0]), np.uint8)
    polygons = []
    for contour in contours_xy:
        points = np.rint(
            (np.asarray(contour, float) - np.asarray(origin_xy)) * px_per_m
        ).astype(np.int32)
        if len(points) >= 3:
            polygons.append(points)
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask.astype(bool)
```

For every source frame, rectify at `min(px_per_m, SELECTION_PX_PER_M)`, erode the valid mask, intersect it with the coarse workspace mask, retain only that mask and scalar metadata, and set `candidate.image=None`. After selection, rectify only selected records at `px_per_m`, intersect with the full-resolution workspace mask, and fill invalid pixels.

Compute manifest coverage as:

```python
workspace_pixels = max(1, int(workspace_mask.sum()))
union_coverage = float(union.sum() / workspace_pixels)
selected_coverage = float(selected_union.sum() / workspace_pixels)
selected_union_fraction = float(
    (selected_union & union).sum() / max(1, union.sum())
)
```

Serialize `selection["selected_union_fraction"]` and `workspace_geometry["hard_bounds_table_xy"]`.

- [ ] **Step 7: Run capture tests and commit Task 1**

Run: `.venv/bin/pytest tests/test_coverage.py tests/test_multiview.py -v`

Expected: PASS.

Commit:

```bash
git add coverage.py multiview.py tests/test_coverage.py tests/test_multiview.py
git commit -m "fix: select views over complete workspace"
```

---

### Task 2: Actual 3D Viewing-Ray Diversity

**Files:**
- Modify: `/Users/emily/lego-capture/multiview.py:31-177,214-276,294-349`
- Modify: `/Users/emily/lego-capture/tests/test_multiview.py:1-120`
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py:85-102,334-378,863-884`
- Modify: `/Users/emily/lego-cv/tests/test_multiview_bridge.py:116-151`

**Interfaces:**
- Produces: `viewing_ray_table_xyz`, `view_tilt_deg`, `view_ray_separation_deg`, and `selection_geometry` based on ray angles.
- Consumes downstream: `selection.angle_diverse` remains the compatibility flag, now derived from viewing rays.

- [ ] **Step 1: Add failing ray geometry tests**

```python
def test_view_ray_separation_detects_real_obliquity_not_bearing():
    near_top_a = _candidate(1, 10, FULL, bearing=0,
                            ray=(0.01, 0.0, -0.99995))
    near_top_b = _candidate(2, 10, FULL, bearing=180,
                            ray=(-0.01, 0.0, -0.99995))
    oblique = _candidate(3, 8, FULL, bearing=0,
                         ray=(0.70, 0.0, -0.714))
    assert view_ray_separation_deg(near_top_a.viewing_ray_table_xyz,
                                   near_top_b.viewing_ray_table_xyz) < 2
    assert view_ray_separation_deg(near_top_a.viewing_ray_table_xyz,
                                   oblique.viewing_ray_table_xyz) > 40


def test_confirmation_selection_maximizes_minimum_ray_separation():
    selected = select_covering_views([
        _candidate(0, 10, FULL, ray=(0, 0, -1)),
        _candidate(10, 8, FULL, ray=(0.1, 0, -0.995)),
        _candidate(20, 8, FULL, ray=(0.7, 0, -0.714)),
    ], max_frames=2, min_frames=2)
    assert [view.frame_id for view in selected] == [0, 20]
```

- [ ] **Step 2: Run ray tests and verify RED**

Run: `.venv/bin/pytest tests/test_multiview.py -k 'ray or confirmation_selection' -v`

Expected: FAIL because candidates have no ray field and selection still uses bearing.

- [ ] **Step 3: Implement normalized rays and selection geometry**

```python
def view_ray_separation_deg(first, second):
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))

def _camera_geometry(record, table_frame, target_xy):
    camera_world = np.append(record.pose_mat[:3, 3], 1.0)
    camera = (table_frame["world_to_table"] @ camera_world)[:3]
    target = np.array([target_xy[0], target_xy[1], 0.0])
    ray = target - camera
    ray /= np.linalg.norm(ray)
    bearing = float(np.degrees(np.arctan2(
        camera[1] - target[1], camera[0] - target[0]
    )) % 360.0)
    tilt = float(np.degrees(np.arctan2(
        np.linalg.norm(ray[:2]), abs(ray[2])
    )))
    return tuple(camera), bearing, tuple(ray), tilt
```

Add `viewing_ray_table_xyz` and `view_tilt_deg` to `ViewCandidate`. During confirmation selection, restrict candidates to quality at or above the median, then maximize minimum ray separation, time separation, quality, and negative frame ID in that order.

`selection_geometry` serializes pairwise ray separation and sets `angle_diverse` from the minimum ray separation against the existing 30-degree diagnostic threshold. Bearing separation remains serialized as deprecated diagnostic information but does not select or certify views.

- [ ] **Step 4: Add failing CV manifest-loading tests for the new optional fields**

```python
def test_load_views_accepts_view_ray_geometry(tmp_path):
    manifest_path = _write_session(tmp_path / "session", with_bridge=True)
    manifest = json.loads(manifest_path.read_text())
    manifest["version"] = 3
    manifest["views"][0]["viewing_ray_table_xyz"] = [0.3, 0.0, -0.9539]
    manifest["views"][0]["view_tilt_deg"] = 17.46
    manifest_path.write_text(json.dumps(manifest))
    _, views = load_views(manifest_path)
    assert views[0].viewing_ray_table_xyz == pytest.approx((0.3, 0, -0.9539))
    assert views[0].view_tilt_deg == pytest.approx(17.46)
```

- [ ] **Step 5: Implement backward-compatible CV loading and flags**

Add optional `viewing_ray_table_xyz` and `view_tilt_deg` fields to `RectifiedView`; load them with `None` defaults. `_apply_dimension_consistency` continues to consume `selection.angle_diverse`, so legacy manifests produce `view_diversity_unknown` and new manifests use the ray-derived value.

- [ ] **Step 6: Run both focused suites and commit Task 2 in each repository**

Run in `lego-capture`: `.venv/bin/pytest tests/test_multiview.py -v`

Run in `lego-cv`: `.venv/bin/pytest tests/test_multiview_bridge.py -v`

Expected: PASS.

Capture commit:

```bash
git add multiview.py tests/test_multiview.py
git commit -m "fix: select genuinely distinct camera views"
```

CV commit:

```bash
git add pipeline/multiview.py tests/test_multiview_bridge.py
git commit -m "feat: load viewing-ray diagnostics"
```

---

### Task 3: Confidence-Frame Persistence

**Files:**
- Modify: `/Users/emily/lego-capture/capture.py:42-135`
- Modify: `/Users/emily/lego-capture/session_io.py:1-121`
- Modify: `/Users/emily/lego-capture/tests/test_capture.py:27-68`
- Modify: `/Users/emily/lego-capture/tests/test_session_io.py:9-70`

**Interfaces:**
- Produces: optional `FrameRecord.confidence: str | None` and `FrameRecord.load_confidence() -> np.ndarray | None`.
- Preserves: historical JSONL records that omit `confidence`.

- [ ] **Step 1: Add failing writer/reader compatibility tests**

```python
def test_writer_reader_round_trip_with_confidence(tmp_path):
    rgb, depth, coeffs, pose = _fake_frame(0)
    confidence = np.full(depth.shape, 2, dtype=np.uint8)
    writer = SessionWriter(tmp_path / "sess")
    rec = writer.add_frame(rgb, depth, coeffs, pose, 1, 1.0,
                           confidence=confidence)
    writer.close()
    loaded = SessionReader(tmp_path / "sess").frames()[0]
    assert loaded.confidence == "frames/00000.confidence.npy"
    np.testing.assert_array_equal(loaded.load_confidence(), confidence)


def test_reader_accepts_historical_record_without_confidence(tmp_path):
    rgb, depth, coeffs, pose = _fake_frame(0)
    writer = SessionWriter(tmp_path / "sess")
    writer.add_frame(rgb, depth, coeffs, pose, 1, 1.0)
    writer.close()
    path = tmp_path / "sess/frames.jsonl"
    payload = json.loads(path.read_text())
    payload.pop("confidence", None)
    path.write_text(json.dumps(payload) + "\n")
    loaded = SessionReader(tmp_path / "sess").frames()[0]
    assert loaded.confidence is None
    assert loaded.load_confidence() is None
```

- [ ] **Step 2: Run persistence tests and verify RED**

Run: `.venv/bin/pytest tests/test_session_io.py -k confidence -v`

Expected: FAIL because the schema and writer do not accept confidence data.

- [ ] **Step 3: Implement the optional sidecar schema**

Add `confidence: str | None = None` after required `FrameRecord` fields and:

```python
def load_confidence(self):
    if self.confidence is None:
        return None
    return np.load(self.session_dir / self.confidence)

@classmethod
def from_json(cls, session_dir, line):
    data = json.loads(line)
    data.setdefault("confidence", None)
    data["pose_mat"] = np.array(data["pose_mat"])
    return cls(session_dir=Path(session_dir), **data)
```

`SessionWriter.add_frame(rgb_bgr, depth, coeffs, pose, device_type, timestamp, confidence=None)` writes `frames/<id>.confidence.npy` only when data is nonempty, includes the optional path in JSONL, and adds its bytes to the summary.

- [ ] **Step 4: Add failing capture-controller tests**

Update `_snapshot` with `"confidence": np.full((6, 8), 2, np.uint8)` and assert every retained record loads the matching confidence array. Add a unit-tested helper:

```python
def _copy_confidence_frame(session):
    getter = getattr(session, "get_confidence_frame", None)
    if getter is None:
        return None
    value = np.asarray(getter()).copy()
    return value if value.size else None
```

Test absent, empty, and populated getters.

- [ ] **Step 5: Implement synchronized capture of confidence**

Read confidence in `_on_new_frame` alongside RGB, depth, pose, and intrinsics. Flip it for TrueDepth exactly when depth is flipped. Pass it through the snapshot queue and into `SessionWriter.add_frame`.

- [ ] **Step 6: Run capture/session tests and commit Task 3**

Run: `.venv/bin/pytest tests/test_capture.py tests/test_session_io.py -v`

Expected: PASS.

Commit:

```bash
git add capture.py session_io.py tests/test_capture.py tests/test_session_io.py
git commit -m "feat: persist Record3D confidence frames"
```

---

### Task 4: Family-Aware Multi-View Identification

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py:572-795,886-949`
- Modify: `/Users/emily/lego-cv/tests/test_multiview_bridge.py:253-373`

**Interfaces:**
- Produces: `brickognize_primary`, `brickognize_views`, `brickognize_raw`, and `identity_view_disagreement`/`family_reviewed` flags in each ID-crop record.
- Consumes: original-frame observations already ranked by `_ranked_id_observations`.

- [ ] **Step 1: Add failing high-confidence family-review and consensus tests**

```python
def test_high_score_regular_segmentation_still_requires_family_review():
    primary = Identification("3021", "Plate 2 x 3", None, 0.90,
        candidates=(
            {"part_id": "3021", "name": "Plate 2 x 3", "score": 0.90},
            {"part_id": "3002", "name": "Brick 2 x 3", "score": 0.72},
        ))
    assert _needs_family_review(primary, detector="seg", metric_flag=None)


def test_two_alternate_frames_overrule_one_high_primary():
    entries = [
        {"frame_id": 1, "part_id": "3021", "name": "Plate 2 x 3", "score": 0.90},
        {"frame_id": 2, "part_id": "3002", "name": "Brick 2 x 3", "score": 0.88},
        {"frame_id": 3, "part_id": "3002", "name": "Brick 2 x 3", "score": 0.86},
    ]
    selected, disagreement, support = _select_identity_consensus(entries)
    assert selected["part_id"] == "3002"
    assert not disagreement
    assert support == {"3021": 1, "3002": 2}


def test_conflicting_single_alternate_preserves_high_primary_and_flags():
    entries = [
        {"frame_id": 1, "part_id": "3021", "name": "Plate 2 x 3", "score": 0.90},
        {"frame_id": 2, "part_id": "3002", "name": "Brick 2 x 3", "score": 0.91},
    ]
    selected, disagreement, support = _select_identity_consensus(entries)
    assert selected["part_id"] == "3021"
    assert disagreement
    assert support == {"3021": 1, "3002": 1}
```

- [ ] **Step 2: Run the focused review tests and verify RED**

Run: `.venv/bin/pytest tests/test_multiview_bridge.py -k 'high_score_regular or conflicting_single' -v`

Expected: FAIL because score 0.90 currently skips `_second_chance` and no per-view evidence is retained.

- [ ] **Step 3: Implement per-frame identity observations**

Add helpers:

```python
def _regular_family(name):
    parsed = parse_regular_dimensions(name)
    return None if parsed is None else parsed.kind, parsed.studs

def _family_ambiguous(identification):
    primary = _regular_family(identification.name)
    return primary is not None and any(
        (family := _regular_family(str(candidate.get("name", ""))))
        is not None and family[1] == primary[1] and family[0] != primary[0]
        for candidate in identification.candidates
    )

def _needs_family_review(identification, detector, metric_flag):
    return detector == "seg" and (
        identification.score < SEG_ID_REVIEW_SCORE
        or _family_ambiguous(identification)
        or metric_flag == "possible_non_canonical_pose"
    )

def _select_identity_consensus(entries):
    per_frame = {}
    for entry in entries:
        current = per_frame.get(entry["frame_id"])
        if current is None or entry["score"] > current["score"]:
            per_frame[entry["frame_id"]] = entry
    accepted = [entry for entry in per_frame.values()
                if entry["part_id"] is not None and entry["score"] >= MIN_SCORE]
    support = {}
    for entry in accepted:
        support[entry["part_id"]] = support.get(entry["part_id"], 0) + 1
    repeated = [entry for entry in accepted
                if support[entry["part_id"]] >= 2]
    if repeated:
        selected = max(repeated, key=lambda entry: (
            support[entry["part_id"]], entry["score"], entry["name"]
        ))
        return selected, False, support
    primary = entries[0]
    if primary["part_id"] is None or primary["score"] < MIN_SCORE:
        selected = max(accepted, key=lambda entry: entry["score"],
                       default=primary)
    else:
        selected = primary
    return selected, len({entry["part_id"] for entry in accepted}) > 1, support
```

Refactor `_second_chance` into `_review_identifications`. Review segmentation instances when the primary is below `SEG_ID_REVIEW_SCORE`, `_family_ambiguous(primary)` is true, or the preliminary metric decision flags `possible_non_canonical_pose`. Submit at most `SECOND_CHANCE_MAX_ALTERNATES` distinct observations. Context and masked variants from the same frame count as one identity vote.

Serialize every frame's selected top response with frame ID, source, part ID, name, score, and candidates. When consensus selects an alternate, update the displayed crop and provenance to that frame.

- [ ] **Step 4: Preserve the raw-to-reviewed decision chain**

In `scan_manifest`, serialize the `identify_all` output as `brickognize_primary` before review. After review, serialize consensus as `brickognize_raw`. Append review flags instead of allowing `_apply_dimension_consistency` to overwrite them.

- [ ] **Step 5: Run bridge and multiview tests and commit Task 4**

Run: `.venv/bin/pytest tests/test_multiview.py tests/test_multiview_bridge.py -v`

Expected: PASS.

Commit:

```bash
git add pipeline/multiview.py tests/test_multiview_bridge.py
git commit -m "feat: review ambiguous parts across views"
```

---

### Task 5: Stud Consensus and Conflict-Aware Arbitration

**Files:**
- Create: `/Users/emily/lego-cv/pipeline/evidence.py`
- Create: `/Users/emily/lego-cv/tests/test_evidence.py`
- Modify: `/Users/emily/lego-cv/pipeline/studs.py:17-232`
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py:797-884,920-949`
- Modify: `/Users/emily/lego-cv/tests/test_studs.py:72-223`
- Modify: `/Users/emily/lego-cv/tests/test_multiview_bridge.py:207-251`

**Interfaces:**
- Produces: `StudConsensus` and `arbitrate_identification(raw, dimension_decision, stud_proposal, stud_action, identity_support)`.
- Guarantees: a stud-only proposal cannot replace or demote an identity.

- [ ] **Step 1: Add failing transitional-luminance and variant-agreement tests**

```python
def test_transitional_dark_piece_uses_multiple_variants():
    image, mask = _dark_grid(2, 12)
    image[mask > 0] = np.clip(image[mask > 0] + 22, 0, 255)
    evidence = analyze_studs(image, radius_px=11, mask=mask)
    assert 50 <= evidence.median_luma <= 80
    assert len(evidence.variants) >= 2


def test_one_plausible_variant_is_not_high_reliability(monkeypatch):
    import pipeline.studs as studs
    centers = iter((
        ((30.0, 30.0), (60.0, 30.0), (30.0, 60.0), (60.0, 60.0)),
        (),
    ))
    monkeypatch.setattr(studs, "_hough_centers", lambda *args: next(centers))
    image = np.full((100, 100, 3), 100, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8)
    evidence = analyze_studs(image, 10, mask)
    assert evidence.reliability != "high"
```

- [ ] **Step 2: Run stud tests and verify RED**

Run: `.venv/bin/pytest tests/test_studs.py -k 'transitional or one_plausible' -v`

Expected: FAIL because luma 55 currently uses one bright variant and one plausible bright variant becomes high reliability.

- [ ] **Step 3: Implement overlapping preprocessing and strict reliability**

Run at least two variants for every crop. For luma up to 80, use CLAHE at `param2=18`, gamma 0.5 plus CLAHE at `param2=16`, and gamma 0.75 plus CLAHE at `param2=18`. Brighter crops use CLAHE at `param2=22` and gamma 0.75 plus CLAHE at `param2=20`.

High reliability requires at least two plausible variants and count spread no greater than `max(3, 0.25 * median_count)`. One plausible variant is medium at most. No plausible variants are low.

- [ ] **Step 4: Add failing arbitration tests**

```python
def test_stud_only_correction_is_preserved_as_conflict():
    outcome = arbitrate_identification(
        current=Identification("2445", "Plate 2 x 12", None, 0.868),
        dimension_action="compatible",
        stud_proposal=Identification("32064", "Technic Brick 1 x 12", None, 0.502),
        stud_action="corrected",
        identity_support={"32064": 0},
    )
    assert outcome.identification.part_id == "2445"
    assert outcome.action == "conflict_preserved"


def test_stud_and_two_view_identity_support_can_correct_unprotected_raw():
    outcome = arbitrate_identification(
        current=Identification("3008", "Brick 1 x 8", None, 0.79),
        dimension_action="flagged",
        stud_proposal=Identification("3028", "Plate 6 x 12", None, 0.75),
        stud_action="corrected",
        identity_support={"3028": 2},
    )
    assert outcome.identification.part_id == "3028"
    assert outcome.action == "corrected"
    assert outcome.supporting_categories == ("stud", "multi_view_identity")
```

- [ ] **Step 5: Run arbitration tests and verify RED**

Run: `.venv/bin/pytest tests/test_evidence.py -v`

Expected: FAIL because `pipeline.evidence` does not exist.

- [ ] **Step 6: Implement explicit arbitration**

```python
@dataclass(frozen=True)
class ArbitrationOutcome:
    identification: Identification
    action: str
    supporting_categories: tuple[str, ...]
    reason: str

def arbitrate_identification(current, dimension_action, stud_proposal,
                             stud_action, identity_support):
    if stud_proposal == current or stud_action in {
        "compatible", "unreliable", "ambiguous", "high-confidence",
        "skipped", "no-rescue"
    }:
        return ArbitrationOutcome(current, stud_action, (),
                                  "stud evidence did not propose a change")
    if stud_proposal.part_id is None:
        return ArbitrationOutcome(current, "conflict_preserved", ("stud",),
                                  "stud evidence cannot demote by itself")
    supports = ["stud"]
    if identity_support.get(stud_proposal.part_id, 0) >= 2:
        supports.append("multi_view_identity")
    if dimension_action in {"compatible", "corrected"}:
        return ArbitrationOutcome(current, "conflict_preserved", tuple(supports),
                                  "metric evidence supports the current identity")
    if len(supports) < 2:
        return ArbitrationOutcome(current, "conflict_preserved", tuple(supports),
                                  "a correction requires two evidence categories")
    return ArbitrationOutcome(stud_proposal, "corrected", tuple(supports),
                              "stud and multi-view identity evidence agree")
```

- [ ] **Step 7: Aggregate stud observations across distinct views**

Add `aggregate_stud_evidence(evidences)` to `studs.py`. Two medium/high observations with count spread within `max(3, 0.25 * median)` produce high consensus. One usable observation produces medium consensus at most. Conflicting usable observations produce low reliability.

Refactor `_apply_stud_advisory` to build masked original crops for up to three distinct ranked observations, serialize each per-view result, call `advise` with the aggregate only to generate a proposal, then call `arbitrate_identification`. Use `brickognize_views` to calculate distinct-frame identity support. Preserve both the proposal and final arbitration reason.

- [ ] **Step 8: Replace the old integration expectation with protected evidence behavior**

The former bridge test that expected a stud-only correction must now assert that the raw identity remains and `stud.action == "conflict_preserved"`. The two-category correction path is exercised by `test_stud_and_two_view_identity_support_can_correct_unprotected_raw` in `tests/test_evidence.py`.

- [ ] **Step 9: Run the CV suite and commit Task 5**

Run: `.venv/bin/pytest tests/test_studs.py tests/test_evidence.py tests/test_multiview_bridge.py tests/test_dimensions.py -v`

Expected: PASS.

Commit:

```bash
git add pipeline/evidence.py pipeline/studs.py pipeline/multiview.py tests/test_evidence.py tests/test_studs.py tests/test_multiview_bridge.py
git commit -m "fix: arbitrate conflicting identification evidence"
```

---

### Task 6: Shadow Depth Height and Training Attribution

**Files:**
- Create: `/Users/emily/lego-cv/pipeline/depth_height.py`
- Create: `/Users/emily/lego-cv/pipeline/attribution.py`
- Create: `/Users/emily/lego-cv/tests/test_depth_height.py`
- Create: `/Users/emily/lego-cv/tests/test_attribution.py`
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py:841-949`

**Interfaces:**
- Produces: `measure_instance_height(instance, manifest) -> DepthHeightEvidence` and `attribute_failure(FailureEvidence) -> str`.
- Guarantees: height evidence is serialized as `shadow_only` and never enters arbitration.

- [ ] **Step 1: Add failing historical and synthetic height tests**

```python
def test_historical_session_without_confidence_is_unavailable(tmp_path):
    evidence = measure_aligned_depth(
        depth=np.full((6, 8), 1.0, dtype=np.float32),
        confidence=None,
        mask_rgb=np.ones((6, 8), dtype=bool),
        intrinsics={"fx": 10.0, "fy": 10.0, "cx": 3.5, "cy": 2.5},
        rgb_size=[6, 8],
        pose_mat=np.eye(4),
        world_to_table=np.eye(4),
        frame_id=4,
    )
    assert evidence.reliability == "unavailable"
    assert "confidence" in evidence.reason


def test_confident_depth_reports_table_relative_height(tmp_path):
    depth = np.full((6, 8), 0.9904, dtype=np.float32)
    confidence = np.full((6, 8), 2, dtype=np.uint8)
    mask = np.ones((6, 8), dtype=bool)
    world_to_table = np.eye(4)
    world_to_table[2, 3] = 1.0
    evidence = measure_aligned_depth(
        depth=depth,
        confidence=confidence,
        mask_rgb=mask,
        intrinsics={"fx": 10.0, "fy": 10.0, "cx": 3.5, "cy": 2.5},
        rgb_size=[6, 8],
        pose_mat=np.eye(4),
        world_to_table=world_to_table,
        frame_id=4,
    )
    assert evidence.action == "shadow_only"
    assert evidence.median_mm == pytest.approx(9.6, abs=0.5)
    assert evidence.usable_pixels > 0
```

- [ ] **Step 2: Run height tests and verify RED**

Run: `.venv/bin/pytest tests/test_depth_height.py -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement self-contained aligned depth measurement**

Load `frames.jsonl`, `table_frame.json`, the selected record's depth sidecar, and optional confidence sidecar. Resize the RGB instance mask to depth resolution with nearest-neighbor sampling. Scale `fx`, `fy`, `cx`, and `cy` from RGB to depth resolution. For valid, confident masked pixels, unproject ARKit depth as:

```python
x = (u - cx) * depth / fx
y = (v - cy) * depth / fy
camera_points = np.column_stack([x, -y, -depth, np.ones_like(depth)])
world_points = (pose_mat @ camera_points.T).T
table_points = (world_to_table @ world_points.T).T
height_mm = table_points[:, 2] * 1000.0
```

Reject nonfinite, nonpositive depth and nonpositive confidence. Serialize usable pixel count, confidence distribution, median, 10th/90th percentiles, spread, frame IDs, reliability, and reason. No confidence sidecar means unavailable rather than an unfiltered estimate.

- [ ] **Step 4: Integrate height as serialization-only evidence**

Attach `depth_height = evidence.to_dict()` to every ID-crop after fusion. Do not pass it to dimension resolution, stud analysis, or final arbitration.

- [ ] **Step 5: Add failing attribution-order tests**

```python
@pytest.mark.parametrize((evidence, expected), [
    (FailureEvidence(False, False, False, False, False), "capture_coverage"),
    (FailureEvidence(True, False, False, False, False), "workspace_view_selection"),
    (FailureEvidence(True, True, False, False, False), "segmentation"),
    (FailureEvidence(True, True, True, False, False), "geometry_fusion"),
    (FailureEvidence(True, True, True, True, False), "identification_evidence"),
    (FailureEvidence(True, True, True, True, True), "no_failure"),
])
def test_failure_attribution_order(evidence, expected):
    assert attribute_failure(evidence) == expected
```

- [ ] **Step 6: Implement the deterministic attribution gate**

```python
@dataclass(frozen=True)
class FailureEvidence:
    source_frame_contains_piece: bool
    selected_roi_contains_piece: bool
    segmentation_mask_correct: bool
    component_count_correct: bool
    identity_correct: bool

def attribute_failure(evidence):
    if not evidence.source_frame_contains_piece:
        return "capture_coverage"
    if not evidence.selected_roi_contains_piece:
        return "workspace_view_selection"
    if not evidence.segmentation_mask_correct:
        return "segmentation"
    if not evidence.component_count_correct:
        return "geometry_fusion"
    if not evidence.identity_correct:
        return "identification_evidence"
    return "no_failure"
```

- [ ] **Step 7: Run focused and full CV tests and commit Task 6**

Run: `.venv/bin/pytest tests/test_depth_height.py tests/test_attribution.py tests/test_multiview_bridge.py -v`

Run: `.venv/bin/pytest -q`

Expected: PASS.

Commit:

```bash
git add pipeline/depth_height.py pipeline/attribution.py pipeline/multiview.py tests/test_depth_height.py tests/test_attribution.py tests/test_multiview_bridge.py
git commit -m "feat: record shadow height and failure attribution"
```

---

### Task 7: Real-Session Regression and Evidence Report

**Files:**
- Modify generated artifacts only under `/Users/emily/lego-capture/sessions/20260717-233252/` (gitignored).
- Modify if behavior needs a proven fix: the exact production and test files from Tasks 1-6, with a new failing regression test first.
- Modify: `/Users/emily/lego-capture/PROGRESS.md`
- Modify: `/Users/emily/lego-capture/CLAUDE_HANDOFF_2026-07-17.md`

**Interfaces:**
- Consumes: complete hard-workspace exporter and installed YOLO segmentation model.
- Produces: regenerated manifest, inventory, debug evidence, timings, and an attributed residual-error report.

- [ ] **Step 1: Preserve the prior generated result**

Run:

```bash
cp -R sessions/20260717-233252/multiview \
  sessions/20260717-233252/multiview-before-reliability
cp sessions/20260717-233252/inventory.json \
  sessions/20260717-233252/inventory-before-reliability.json
```

Expected: previous generated evidence remains available for comparison.

- [ ] **Step 2: Export the new selected views**

Run in `lego-capture`:

```bash
.venv/bin/python multiview.py sessions/20260717-233252
```

Expected: manifest canvas spans hard bounds, selection has three to five views, `selected_union_fraction >= 0.995`, and pairwise viewing-ray diagnostics are present.

- [ ] **Step 3: Verify the selector chose materially different rays and contains the orange region**

Run a read-only diagnostic that prints selected frame IDs, canvas bounds, ray angles, per-view segmentation counts, and the orange mask confidence. Expected: at least one selected original frame contains all 13 objects or the selected union includes the orange 1x8; minimum ray separation exceeds the former 12.85 degrees when eligible candidates allow it.

- [ ] **Step 4: Run the complete segmentation inventory**

Run in `lego-cv`:

```bash
.venv/bin/python session_cli.py \
  /Users/emily/lego-capture/sessions/20260717-233252 \
  --detector seg
```

Expected: 13 fused components before identity grouping, black part remains Plate 2x12, and every remaining wrong/ambiguous identity has per-view, metric, stud, and arbitration diagnostics.

- [ ] **Step 5: Diagnose any failed acceptance criterion before modifying code**

For each failure, use `attribute_failure` in order. Add one minimal automated test reproducing the responsible behavior, observe RED, make the smallest production change, observe GREEN, then rerun Steps 2-4. Do not add segmentation data unless a correct selected ROI visibly contains a piece and YOLO itself misses, merges, splits, or hallucinates its mask.

- [ ] **Step 6: Run final verification**

Run in `lego-capture`:

```bash
.venv/bin/pytest -q
git status --short
```

Run in `lego-cv`:

```bash
.venv/bin/pytest -q
git status --short
```

Expected: all tests pass; only intentional documentation changes or ignored generated artifacts remain.

- [ ] **Step 7: Document measured outcomes and commit documentation**

Record before/after selected frames, canvas size, component count, ray separations, identity results, remaining ambiguities, and whether the segmentation training gate was reached. Do not describe an unresolved ambiguity as fixed.

Commit:

```bash
git add PROGRESS.md CLAUDE_HANDOFF_2026-07-17.md
git commit -m "docs: record inventory reliability regression"
```
