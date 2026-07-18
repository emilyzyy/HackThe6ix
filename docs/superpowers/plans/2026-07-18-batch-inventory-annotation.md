# Batch Inventory Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and populate a lightweight local verifier that turns the twelve supplied Record3D sessions into an exact, human-confirmed part-and-color inventory.

**Architecture:** Add a review-only fast path to `lego-cv` that performs existing segmentation/fusion/color work but limits Brickognize to one best crop per component with bounded concurrency. Export immutable draft provenance plus review assets into run-scoped session directories, then serve a single aggregate review workspace through FastAPI with atomic annotation persistence and exact-count reconciliation. A minimal browser client presents numbered segmentation overlays and per-piece part-render, color, inclusion, and segmentation controls.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, OpenCV, NumPy, Requests, vanilla HTML/CSS/JavaScript, pytest.

## Global Constraints

- Source frames, normal session outputs, and historical analysis artifacts are read-only.
- New artifacts live under `sessions/<id>/analysis-runs/inventory-annotation-v1/` and `/Users/emily/lego-capture/inventory-review-20260718/`.
- Production scan behavior and its current Brickognize throttle remain unchanged; only review-draft mode uses 12 workers and zero fixed inter-request delay.
- Review mode retries `429` and transient server errors using the existing exponential backoff.
- Final colors are restricted to red, orange, yellow, beige, brown, green, dark green, blue, white, light gray, dark gray, and black.
- Original predictions remain immutable; human corrections are stored separately.
- Only included, reconciled, human-confirmed pieces enter the final aggregate.
- A confirmed part requires both a part ID and a human-readable name.
- Segmentation masks become training truth only when explicitly marked `correct`.
- Do not merge, push, reset, or overwrite any historical session artifact.

---

### Task 1: Exact-count annotation domain

**Files:**
- Create: `/Users/emily/lego-cv/training/inventory_review.py`
- Create: `/Users/emily/lego-cv/tests/training/test_inventory_review.py`

**Interfaces:**
- Consumes: draft component dictionaries produced by Task 3.
- Produces: `new_review_document(session_id, expected_count, components) -> dict`, `apply_component_review(document, component_id, patch) -> dict`, `add_manual_piece(document, part_id, part_name, color, included) -> dict`, `reconciliation(document) -> dict`, `lock_document(document) -> dict`, `aggregate_documents(documents) -> tuple[list[dict], list[dict], dict]`, and `atomic_write_json(path, payload) -> None`.

- [ ] **Step 1: Write failing tests for document creation and immutable predictions**

```python
def test_new_review_document_keeps_prediction_separate_from_review():
    component = {
        "component_id": "piece-000",
        "prediction": {"part_id": "3001", "part_name": "Brick 2 x 4", "color": "blue"},
        "candidates": [],
        "assets": {},
    }
    document = new_review_document("session-a", 1, [component])
    row = document["components"][0]
    assert row["prediction"]["part_id"] == "3001"
    assert row["review"] == {
        "confirmed": False,
        "part_id": "3001",
        "part_name": "Brick 2 x 4",
        "color": "blue",
        "included": True,
        "exclusion_reason": None,
        "segmentation": "unreviewed",
    }
```

- [ ] **Step 2: Run the focused test and verify failure**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/training/test_inventory_review.py::test_new_review_document_keeps_prediction_separate_from_review -v`

Expected: FAIL because `training.inventory_review` does not exist.

- [ ] **Step 3: Implement constants, document creation, and validation**

```python
INVENTORY_COLORS = (
    "red", "orange", "yellow", "beige", "brown", "green",
    "dark green", "blue", "white", "light gray", "dark gray", "black",
)
SEGMENTATION_STATES = (
    "unreviewed", "correct", "duplicate", "merged", "incomplete", "missed",
)

def new_review_document(session_id: str, expected_count: int,
                        components: list[dict]) -> dict:
    rows = []
    for component in components:
        prediction = copy.deepcopy(component["prediction"])
        rows.append({
            **copy.deepcopy(component),
            "review": {
                "confirmed": False,
                "part_id": prediction.get("part_id") or "",
                "part_name": prediction.get("part_name") or "",
                "color": prediction.get("color") or "",
                "included": True,
                "exclusion_reason": None,
                "segmentation": "unreviewed",
            },
        })
    return {
        "schema_version": 1,
        "session_id": session_id,
        "expected_physical_count": int(expected_count),
        "components": rows,
        "manual_pieces": [],
        "locked": False,
    }
```

- [ ] **Step 4: Write failing reconciliation tests**

```python
def test_reconciliation_distinguishes_excluded_physical_piece_from_duplicate():
    document = new_review_document("s", 2, [_component("a"), _component("b"), _component("c")])
    apply_component_review(document, "a", _confirmed(included=True))
    apply_component_review(document, "b", _confirmed(included=False, exclusion_reason="not in demo bank"))
    apply_component_review(document, "c", _confirmed(segmentation="duplicate"))
    assert reconciliation(document) == {
        "expected": 2, "accounted_physical": 2, "included": 1,
        "excluded": 1, "duplicates": 1, "manual_missed": 0,
        "unresolved": 0, "reconciled": True,
    }

def test_lock_rejects_unconfirmed_or_merged_entries():
    document = new_review_document("s", 1, [_component("a")])
    with pytest.raises(ValueError, match="unresolved"):
        lock_document(document)
```

- [ ] **Step 5: Implement review mutation, reconciliation, locking, and manual pieces**

`apply_component_review` must reject unknown component IDs, invalid colors, invalid segmentation states, missing part fields on included rows, and exclusion without a reason. Duplicate detections count as zero physical pieces. Included and excluded confirmed rows count as one physical piece. Manual missed pieces count as one physical piece. Merged and incomplete rows remain unresolved.

- [ ] **Step 6: Write and pass aggregate/export tests**

```python
def test_aggregate_sums_only_locked_included_rows():
    first = _locked_document([_reviewed("3001", "Brick 2 x 4", "blue", True)])
    second = _locked_document([
        _reviewed("3001", "Brick 2 x 4", "blue", True),
        _reviewed("3003", "Brick 2 x 2", "white", False),
    ])
    inventory, pieces, summary = aggregate_documents([first, second])
    assert inventory == [{
        "part_id": "3001", "name": "Brick 2 x 4",
        "color": "blue", "quantity": 2,
    }]
    assert len(pieces) == 3
    assert summary["included"] == 2
```

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/training/test_inventory_review.py -v`

Expected: all Task 1 tests PASS.

- [ ] **Step 7: Commit Task 1**

```bash
cd /Users/emily/lego-cv
git add training/inventory_review.py tests/training/test_inventory_review.py
git commit -m "feat: add exact-count inventory review model"
```

---

### Task 2: Review-only identification mode and candidate renders

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/identify.py`
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py`
- Modify: `/Users/emily/lego-cv/tests/test_identify.py`
- Modify: `/Users/emily/lego-cv/tests/test_multiview_bridge.py`

**Interfaces:**
- Consumes: existing Brickognize payloads and `scan_manifest` inputs.
- Produces: candidate dictionaries containing `img_url`; `scan_manifest(..., review_draft: bool = False, identify_workers: int = 4, identify_min_interval: float = 1.0)`.

- [ ] **Step 1: Write failing candidate-render parsing test**

```python
def test_parse_preserves_candidate_render_url():
    result = _parse({"items": [{
        "id": "3001", "name": "Brick 2 x 4", "score": 0.91,
        "img_url": "https://example.test/3001.webp",
    }]})
    assert result.candidates[0]["img_url"] == "https://example.test/3001.webp"
```

- [ ] **Step 2: Run the test and verify `img_url` is absent**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/test_identify.py::test_parse_preserves_candidate_render_url -v`

Expected: FAIL with `KeyError: 'img_url'`.

- [ ] **Step 3: Preserve the optional render URL in `_parse`**

```python
candidates = tuple({
    "part_id": item["id"],
    "name": item["name"],
    "score": float(item.get("score", 0.0)),
    "img_url": item.get("img_url"),
} for item in items[:5])
```

- [ ] **Step 4: Write failing review-draft scan test**

The test must monkeypatch `identify_all`, `_second_chance`, and `_apply_stud_advisory`, call `scan_manifest(..., review_draft=True, identify_workers=12, identify_min_interval=0.0)`, and assert:

```python
assert identify_all_call.kwargs["workers"] == 12
assert identify_all_call.kwargs["min_interval"] == 0.0
assert second_chance_calls == []
assert stud_advisory_calls == []
assert result.id_crops[0]["brickognize_raw"] == result.id_crops[0]["brickognize_primary"]
```

- [ ] **Step 5: Implement the review-only branch**

Keep `review_draft=False` as the production default. In review mode, call:

```python
identifications = identify_all(
    crops,
    workers=identify_workers,
    min_interval=identify_min_interval,
    cache_dir=cache_dir,
)
```

Serialize primary/raw/final identification from that one request, attach existing metric and color evidence needed for the worksheet, and skip second-chance identification, stud arbitration, dimension relabeling, depth height, and model fitting. Do not change the non-review branch.

- [ ] **Step 6: Run focused and full CV tests**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/test_identify.py tests/test_multiview_bridge.py -v`

Expected: focused tests PASS with no production-path regression.

- [ ] **Step 7: Commit Task 2**

```bash
cd /Users/emily/lego-cv
git add pipeline/identify.py pipeline/multiview.py tests/test_identify.py tests/test_multiview_bridge.py
git commit -m "feat: add fast human-review scan mode"
```

---

### Task 3: Run-scoped review bundle generation

**Files:**
- Create: `/Users/emily/lego-cv/training/inventory_review_assets.py`
- Create: `/Users/emily/lego-cv/tests/training/test_inventory_review_assets.py`
- Modify: `/Users/emily/lego-cv/session_cli.py`
- Modify: `/Users/emily/lego-cv/tests/test_session_cli.py`

**Interfaces:**
- Consumes: `MultiViewScanResult`, session directory, output directory, and expected count.
- Produces: `write_review_bundle(result, session_dir, output_dir, expected_count) -> Path`, where the returned path is `review.json`.

- [ ] **Step 1: Write failing asset-bundle test with one synthetic segmented component**

```python
def test_write_review_bundle_writes_overlay_crop_mask_and_document(tmp_path):
    result = fake_review_result(tmp_path)
    path = write_review_bundle(result, tmp_path / "session", tmp_path / "run", 1)
    document = json.loads(path.read_text())
    component = document["components"][0]
    assert Path(component["assets"]["crop"]).exists()
    assert Path(component["assets"]["mask"]).exists()
    assert Path(component["assets"]["masked_crop"]).exists()
    assert document["expected_physical_count"] == 1
```

- [ ] **Step 2: Run the test and verify missing module failure**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/training/test_inventory_review_assets.py -v`

Expected: FAIL because `inventory_review_assets` does not exist.

- [ ] **Step 3: Implement deterministic component assets**

For every fused instance, create:

```text
review-assets/piece-000/crop-00.jpg
review-assets/piece-000/mask-00.png
review-assets/piece-000/masked-00.png
```

Use the ranked original-frame observations, crop each original mask to its bounding box, and save at most three views. Candidate records are de-duplicated by part ID. The draft prediction comes from `id_crop["brickognize"]` and `consensus_color(instance).name`.

- [ ] **Step 4: Implement part-render caching**

For every candidate with an `img_url`, download at most once into:

```text
review-assets/part-renders/<part-id>.webp
```

Use a 15-second timeout. On failure, retain `render_path: null` and the original `img_url`; missing renders never abort bundle creation.

- [ ] **Step 5: Write failing CLI option tests**

Assert that `--inventory-review --expected-count 48 --analysis-dir <run>` calls `scan_manifest` with `review_draft=True`, `identify_workers=12`, `identify_min_interval=0.0`, and writes `review.json`. Assert both review flags require `--analysis-dir` and a positive expected count.

- [ ] **Step 6: Implement CLI arguments and bundle call**

Add:

```python
parser.add_argument("--inventory-review", action="store_true")
parser.add_argument("--expected-count", type=int)
parser.add_argument("--review-workers", type=int, default=12)
```

Review mode skips color worksheets and model-fit diagnostics, writes all ordinary diagnostic JSON beneath `--analysis-dir`, then calls `write_review_bundle`.

- [ ] **Step 7: Run Task 3 tests**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/training/test_inventory_review_assets.py tests/test_session_cli.py -v`

Expected: all Task 3 tests PASS.

- [ ] **Step 8: Commit Task 3**

```bash
cd /Users/emily/lego-cv
git add training/inventory_review_assets.py tests/training/test_inventory_review_assets.py session_cli.py tests/test_session_cli.py
git commit -m "feat: export run-scoped inventory review bundles"
```

---

### Task 4: Aggregate review workspace and FastAPI

**Files:**
- Create: `/Users/emily/lego-cv/inventory_review_app.py`
- Create: `/Users/emily/lego-cv/tests/test_inventory_review_app.py`

**Interfaces:**
- Consumes: a workspace JSON listing absolute `review.json` paths.
- Produces: `create_app(workspace_path: Path) -> FastAPI`, `create_app_from_env() -> FastAPI`, and endpoints for loading, updating, manually adding, locking, and exporting.

- [ ] **Step 1: Write failing API tests**

```python
def test_update_component_autosaves_and_preserves_prediction(workspace):
    client = TestClient(create_app(workspace))
    response = client.put("/api/sessions/s1/components/piece-000", json={
        "confirmed": True,
        "part_id": "3003",
        "part_name": "Brick 2 x 2",
        "color": "white",
        "included": True,
        "segmentation": "correct",
    })
    assert response.status_code == 200
    saved = json.loads(workspace.review_path.read_text())
    assert saved["components"][0]["prediction"]["part_id"] == "3001"
    assert saved["components"][0]["review"]["part_id"] == "3003"
```

Also test path traversal rejection, exclusion-reason validation, manual missing pieces, lock refusal, and aggregate export refusal while any session is unlocked.

- [ ] **Step 2: Run tests and verify missing app failure**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/test_inventory_review_app.py -v`

Expected: FAIL because `inventory_review_app` does not exist.

- [ ] **Step 3: Implement app factory and safe file serving**

`create_app_from_env` reads `LEGO_INVENTORY_WORKSPACE`, rejects a missing or
nonexistent path with a clear startup error, and delegates to `create_app`.

Routes:

```text
GET  /api/workspace
GET  /api/sessions/{session_id}
PUT  /api/sessions/{session_id}/components/{component_id}
POST /api/sessions/{session_id}/manual-pieces
PUT  /api/sessions/{session_id}/manual-pieces/{manual_id}
POST /api/sessions/{session_id}/lock
POST /api/export
GET  /assets/{session_id}/{asset_path:path}
GET  /
GET  /app.js
GET  /styles.css
```

Resolve asset paths and require `candidate.is_relative_to(review_dir.resolve())` before returning `FileResponse`.

- [ ] **Step 4: Implement atomic autosave and export files**

Every mutation calls `atomic_write_json`. `/api/export` calls `aggregate_documents` and writes `verified_inventory.json`, `verified_inventory.csv`, `verified_pieces.csv`, and `verification_summary.json` beneath the workspace directory.

- [ ] **Step 5: Run API tests**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/test_inventory_review_app.py -v`

Expected: all API tests PASS.

- [ ] **Step 6: Commit Task 4**

```bash
cd /Users/emily/lego-cv
git add inventory_review_app.py tests/test_inventory_review_app.py
git commit -m "feat: add local inventory review API"
```

---

### Task 5: Minimal detector-style browser client

**Files:**
- Create: `/Users/emily/lego-cv/inventory_review_web/index.html`
- Create: `/Users/emily/lego-cv/inventory_review_web/app.js`
- Create: `/Users/emily/lego-cv/inventory_review_web/styles.css`
- Modify: `/Users/emily/lego-cv/tests/test_inventory_review_app.py`

**Interfaces:**
- Consumes: Task 4 JSON endpoints.
- Produces: the approved two-pane review interface.

- [ ] **Step 1: Add failing static-route assertions**

```python
def test_browser_client_is_served(workspace):
    client = TestClient(create_app(workspace))
    assert "segmentation-stage" in client.get("/").text
    assert client.get("/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/styles.css").headers["content-type"].startswith("text/css")
```

- [ ] **Step 2: Create semantic HTML matching the approved mockup**

The page contains only session navigation, an overlay stage, selected crop/views, candidate render cards, manual part fields, the 12-color select, inclusion/exclusion controls, segmentation state, progress, and confirm/previous/next controls.

- [ ] **Step 3: Implement client state and autosave**

`app.js` loads `/api/workspace`, opens the first unresolved component, renders overlay frame choices and component candidate cards, sends `PUT` on confirmation, and moves to the next unresolved component. Keyboard shortcuts: Enter confirms, left/right arrows navigate, `E` toggles exclusion.

- [ ] **Step 4: Make count reconciliation visible but unobtrusive**

Show one line:

```text
Physical 48 · accounted 46 · included 44 · excluded 2 · unresolved 2
```

Disable the Lock button until `reconciled` is true and unresolved is zero. Provide `Add missed piece` only when the physical count is not yet accounted for.

- [ ] **Step 5: Run the API/static tests and manually smoke-test the fixture**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest tests/test_inventory_review_app.py -v`

Expected: all tests PASS. Start the fixture app and confirm select, exclude, refresh-resume, and navigation work in the browser.

- [ ] **Step 6: Commit Task 5**

```bash
cd /Users/emily/lego-cv
git add inventory_review_web tests/test_inventory_review_app.py
git commit -m "feat: add lightweight inventory annotation UI"
```

---

### Task 6: Batch preparation command and the twelve-session workspace

**Files:**
- Create: `/Users/emily/lego-capture/scripts/prepare_inventory_review.py`
- Create: `/Users/emily/lego-capture/tests/test_prepare_inventory_review.py`
- Modify: `/Users/emily/lego-capture/.gitignore`

**Interfaces:**
- Consumes: the twelve session IDs/counts, `lego-capture/multiview.py`, and `lego-cv/session_cli.py`.
- Produces: per-session run directories and `/Users/emily/lego-capture/inventory-review-20260718/workspace.json`.

- [ ] **Step 1: Write failing batch-config and command-construction tests**

```python
def test_batch_counts_total_780():
    batches = load_batches(FIXTURE_BATCHES)
    assert sum(batch.expected_count for batch in batches) == 780

def test_commands_are_run_scoped(tmp_path):
    export, review = commands_for(batch("20260718-161438", 48), tmp_path)
    assert "multiview.py" in export
    assert "--analysis-dir" in review
    assert "inventory-annotation-v1" in review
    assert "--expected-count" in review and "48" in review
```

- [ ] **Step 2: Implement resumable preparation**

For each batch, run multiview export only when `multiview/manifest.json` is absent. Run review generation only when its `review.json` is absent. After each successful session, atomically refresh `workspace.json`. Accept `--session` to retry one batch and `--dry-run` to print commands without mutation.

- [ ] **Step 3: Add generated workspace to `.gitignore`**

Add:

```text
inventory-review-*/
```

The source batch configuration is embedded in the preparation script and tested; generated review data is not committed.

- [ ] **Step 4: Run capture tests**

Run: `cd /Users/emily/lego-capture && .venv/bin/pytest tests/test_prepare_inventory_review.py -v`

Expected: all Task 6 tests PASS.

- [ ] **Step 5: Commit Task 6**

```bash
cd /Users/emily/lego-capture
git add .gitignore scripts/prepare_inventory_review.py tests/test_prepare_inventory_review.py
git commit -m "feat: prepare exact inventory review batches"
```

- [ ] **Step 6: Generate the real workspace**

Run:

```bash
cd /Users/emily/lego-capture
.venv/bin/python scripts/prepare_inventory_review.py \
  --capture-root /Users/emily/lego-capture \
  --cv-root /Users/emily/lego-cv \
  --workspace /Users/emily/lego-capture/inventory-review-20260718
```

Expected: every successful batch reports its fused-component count and creates a `review.json`. A session whose draft count differs from its physical count remains reviewable and visibly unreconciled; generation does not invent pieces.

---

### Task 7: Full verification and launch

**Files:**
- Modify: `/Users/emily/lego-capture/PROGRESS.md`

**Interfaces:**
- Consumes: Tasks 1–6 and the generated workspace.
- Produces: a tested launch command and documented limitations.

- [ ] **Step 1: Run all CV tests**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest -q`

Expected: all tests PASS; record the exact total.

- [ ] **Step 2: Run all capture tests**

Run: `cd /Users/emily/lego-capture && .venv/bin/pytest -q`

Expected: all tests PASS; record the exact total.

- [ ] **Step 3: Validate the generated workspace**

Run a read-only validation command that loads every `review.json`, confirms all twelve session IDs and expected counts, checks that each asset path remains beneath its run directory, and prints draft detected counts plus a total. No session is expected to be locked before Emily reviews it.

- [ ] **Step 4: Launch the real verifier**

Run:

```bash
cd /Users/emily/lego-cv
.venv/bin/uvicorn 'inventory_review_app:create_app_from_env' \
  --host 127.0.0.1 --port 8765
```

with `LEGO_INVENTORY_WORKSPACE=/Users/emily/lego-capture/inventory-review-20260718/workspace.json` in the environment. Open `http://127.0.0.1:8765/` and confirm the first session and first component load.

- [ ] **Step 5: Record evidence in PROGRESS.md**

Document the implementation commits, test totals, workspace path, per-session draft counts, review URL, and the explicit statement that draft predictions are not the final inventory until all sessions are human-confirmed and reconciled.

- [ ] **Step 6: Commit the progress checkpoint**

```bash
cd /Users/emily/lego-capture
git add PROGRESS.md
git commit -m "docs: record inventory review launch"
```
