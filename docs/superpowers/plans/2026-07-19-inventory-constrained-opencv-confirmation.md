# Inventory-Constrained OpenCV Confirmation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Constrain showcase questions to the authoritative 787-piece CSV, show only 3–4 confirmations, and make native OpenCV confirmation the default while retaining the web reviewer as a backup.

**Architecture:** Add a pure inventory catalog and constrain identities only at showcase construction, leaving detector/LiDAR/multiview code unchanged. Add a deterministic OpenCV confirmation state machine that consumes the existing review workspace and shares handoff persistence with FastAPI. Wire `demo_flow.py` to choose `opencv` by default or `web` explicitly.

**Tech Stack:** Python 3.11, OpenCV, NumPy, CSV/JSON, pytest, existing LEGO review documents

## Global Constraints

- Authoritative catalog: `/Users/emily/lego-capture/outputs/019f72d0-9cbe-7431-8f4b-7ed4084bbd13/lego_inventory_for_model_generation_20260719.csv`.
- Only the CSV's 32 normalized `piece_type` values may be shown or manually saved.
- Showcase target is four; minimum-safe count is three; never ask five.
- Do not change phone capture, LiDAR filtering, segmentation, multiview fusion, or Claude's performance/reliability commits.
- Keep the existing FastAPI/browser reviewer selectable with `--confirmation-ui web`.
- Native confirmation must produce the same `handoff.json` schema as the browser endpoint.

---

### Task 1: Inventory-constrained showcase selection

**Files:**
- Create: `/Users/emily/lego-cv/pipeline/inventory_constraints.py`
- Modify: `/Users/emily/lego-cv/pipeline/showcase.py`
- Modify: `/Users/emily/lego-cv/pipeline/fast_showcase.py`
- Modify: `/Users/emily/lego-cv/fast_showcase_cli.py`
- Test: `/Users/emily/lego-cv/tests/test_inventory_constraints.py`
- Test: `/Users/emily/lego-cv/tests/test_showcase.py`
- Test: `/Users/emily/lego-cv/tests/test_fast_showcase.py`

**Interfaces:**
- Produces: `InventoryCatalog.load(path: Path) -> InventoryCatalog`, `catalog.resolve(final: dict, raw: dict) -> dict | None`, and `allowed_names: tuple[str, ...]`.
- Extends: `build_showcase(result, n=4, *, colors=None, single_view_min_top1=SINGLE_VIEW_MIN_TOP1, inventory_catalog=None)` and `run_fast_showcase(manifest_path, output_dir, *, target=4, min_safe=3, max_views=3, identify_limit=12, weights=Path("models/lego_seg.pt"), inventory_csv=None)`.

- [x] **Step 1: Write failing catalog and selection tests**

```python
def test_catalog_reranks_disallowed_slope_to_allowed_plate(tmp_path):
    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text("piece_type,color,quantity\nPlate 2 x 2,red,3\n")
    catalog = InventoryCatalog.load(csv_path)
    resolved = catalog.resolve(
        {"part_id": "3039", "name": "Slope 45 2 x 2", "score": 0.91},
        {"candidates": [
            {"part_id": "3039", "name": "Slope 45 2 x 2", "score": 0.91},
            {"part_id": "3022", "name": "Plate 2 x 2", "score": 0.84},
        ]},
    )
    assert resolved == {"part_id": "3022", "name": "Plate 2 x 2", "score": 0.84}

def test_catalog_returns_none_when_no_inventory_candidate_exists(tmp_path):
    csv_path = tmp_path / "inventory.csv"
    csv_path.write_text("piece_type,color,quantity\nBrick 2 x 2,blue,3\n")
    catalog = InventoryCatalog.load(csv_path)
    resolved = catalog.resolve(
        {"part_id": "3039", "name": "Slope 45 2 x 2", "score": 0.91},
        {"candidates": []},
    )
    assert resolved is None
```

- [x] **Step 2: Verify the tests fail for missing inventory constraints**

Run: `.venv/bin/pytest tests/test_inventory_constraints.py tests/test_showcase.py tests/test_fast_showcase.py -q`

Expected: collection/import failure for `pipeline.inventory_constraints` and failed slope/cap assertions.

- [x] **Step 3: Implement CSV validation and inventory-only identity resolution**

`InventoryCatalog.load` must require `piece_type`, `color`, and `quantity`, reject non-positive/non-integer quantities, retain canonical CSV spelling, and normalize with `" ".join(name.lower().split())`. `resolve` must consider the surfaced identity plus raw candidates, sort by score descending, and return the highest-scoring allowed candidate or `None`.

In `build_showcase`, resolve identity before constructing `ShowcaseCandidate`; when resolution is `None`, construct it as unidentified so the existing veto/fallback cannot surface it. Recalculate top-1/top-2 from allowed raw candidates. Pass the catalog through strict and final showcase calls so both use identical filtering.

- [x] **Step 4: Add CLI plumbing and cap the selectable target**

Add `--inventory-csv PATH` to `fast_showcase_cli.py`, pass it to `run_fast_showcase`, load once before both attempts, and clamp the requested target to `min(4, max(0, target))`. Keep `min_safe=3`.

- [x] **Step 5: Run focused tests**

Run: `.venv/bin/pytest tests/test_inventory_constraints.py tests/test_showcase.py tests/test_fast_showcase.py -q`

Expected: all selected tests pass.

---

### Task 2: Shared handoff writer and native confirmation state machine

**Files:**
- Create: `/Users/emily/lego-cv/pipeline/showcase_handoff.py`
- Create: `/Users/emily/lego-cv/opencv_confirmation.py`
- Create: `/Users/emily/lego-cv/inventory_review_web/confirmation-mascot.png`
- Modify: `/Users/emily/lego-cv/inventory_review_app.py`
- Test: `/Users/emily/lego-cv/tests/test_showcase_handoff.py`
- Test: `/Users/emily/lego-cv/tests/test_opencv_confirmation.py`
- Test: `/Users/emily/lego-cv/tests/test_inventory_review_app.py`

**Interfaces:**
- Produces: `write_showcase_handoff(review_path: Path, document: dict) -> dict`.
- Produces: `ConfirmationController(workspace_path: Path, catalog: InventoryCatalog)` with `handle_key`, `handle_click`, `render`, `save_current`, and `complete`.
- CLI: `opencv_confirmation.py --workspace PATH --inventory-csv PATH`.

- [x] **Step 1: Write failing handoff-equivalence and interaction tests**

```python
def test_controller_rejects_invalid_text_and_accepts_inventory_suggestion(workspace, catalog):
    controller = ConfirmationController(workspace, catalog)
    controller.handle_key(ord("n"))
    for char in "slope":
        controller.handle_key(ord(char))
    assert controller.save_current() is False
    controller.correction_text = "Plate 2 x 2"
    assert controller.save_current() is True

def test_native_and_web_handoff_payloads_match(review_path, confirmed_document):
    result = write_showcase_handoff(review_path, confirmed_document)
    payload = json.loads(Path(result["handoff_path"]).read_text())
    assert payload["schema_version"] == 1
    assert payload["confirmed_pieces"][0]["accepted_prediction"] is True
```

- [x] **Step 2: Verify the tests fail for missing native reviewer and shared writer**

Run: `.venv/bin/pytest tests/test_showcase_handoff.py tests/test_opencv_confirmation.py tests/test_inventory_review_app.py -q`

Expected: import failures for both new modules.

- [x] **Step 3: Extract the existing FastAPI handoff logic without schema changes**

Move the body that validates `showcase_only`, verifies every component is confirmed, constructs `confirmed_pieces`, writes `handoff.json`, and returns `handoff_path`/`generator_url` into `write_showcase_handoff`. Convert its `ValueError` to HTTP 422 in the endpoint. Keep all existing response keys unchanged.

- [x] **Step 4: Implement the pure native interaction state machine**

Load the workspace's first review document, resolve its frame/mask assets relative to the review path, and maintain `mode`, `index`, `correction_text`, `suggestions`, `suggestion_index`, `status`, and button rectangles. `Yes` applies an existing showcase review with prediction fields. `No` enters correction mode. Save accepts only `catalog.canonical_name(text)`, applies a part-name-only review, atomically writes the review document, and advances. The final save calls `write_showcase_handoff`.

- [x] **Step 5: Implement deterministic OpenCV rendering and runtime loop**

Render to a 1280×720 canvas: aspect-fit the source frame, tint only mask pixels yellow at 55% alpha, draw the waypoint line/card/buttons, alpha-blend the committed mascot PNG, and render at most five correction suggestions. Use one mouse callback and `cv2.waitKeyEx`; always call `cv2.destroyAllWindows()` in `finally`. `Q` exits resumably and `Inventory ready` remains until Enter/Escape/Q.

- [x] **Step 6: Run focused native/UI tests**

Run: `.venv/bin/pytest tests/test_showcase_handoff.py tests/test_opencv_confirmation.py tests/test_inventory_review_app.py -q`

Expected: all selected tests pass and the existing web handoff contract remains green.

---

### Task 3: Stitch OpenCV confirmation into the demo with web fallback

**Files:**
- Modify: `/Users/emily/lego-capture/demo_flow.py`
- Modify: `/Users/emily/lego-capture/README.md`
- Modify: `/Users/emily/lego-capture/tests/test_demo_flow.py`

**Interfaces:**
- Extends: `build_processing_commands(capture_root, cv_root, session_dir, output_dir, *, fixed_inventory, generator_url, inventory_csv)`.
- Adds: `run_confirmation(workspace, cv_root, inventory_csv, mode, port) -> str | Path`.
- Adds: `build_parser() -> argparse.ArgumentParser` so CLI defaults are tested without starting capture.
- CLI: `--inventory-csv PATH` and `--confirmation-ui {opencv,web}`.

- [x] **Step 1: Write failing command/default/fallback tests**

```python
def test_demo_defaults_to_inventory_constrained_native_confirmation():
    args = build_parser().parse_args(["--replay-session", "session"])
    assert args.confirmation_ui == "opencv"
    assert args.inventory_csv.name == "lego_inventory_for_model_generation_20260719.csv"

def test_processing_command_caps_showcase_and_passes_inventory_csv(tmp_path):
    commands = build_processing_commands(
        tmp_path / "capture", tmp_path / "cv", tmp_path / "session",
        tmp_path / "output", fixed_inventory=None, generator_url=None,
        inventory_csv=tmp_path / "inventory.csv",
    )
    fast = commands[-1]
    assert fast[fast.index("--target") + 1] == "4"
    assert fast[fast.index("--min-safe") + 1] == "3"
    assert fast[fast.index("--inventory-csv") + 1].endswith("inventory.csv")
```

- [x] **Step 2: Verify demo-flow tests fail**

Run: `.venv/bin/pytest tests/test_demo_flow.py -q`

Expected: missing parser/options and target still equal to five.

- [x] **Step 3: Add native/default and browser/backup dispatch**

Validate the inventory CSV before starting a live scan. Pass it into the fast-showcase command. For `opencv`, run `[str(cv_root / ".venv/bin/python"), str(cv_root / "opencv_confirmation.py"), "--workspace", str(workspace), "--inventory-csv", str(inventory_csv)]` in the foreground and do not start FastAPI or open a browser. For `web`, call the unchanged `_start_confirmation`, including Claude's `_free_port` behavior.

- [x] **Step 4: Document both commands**

Primary:

```bash
.venv/bin/python demo_flow.py --anchor sessions/20260718-165402
```

Backup:

```bash
.venv/bin/python demo_flow.py --anchor sessions/20260718-165402 --confirmation-ui web
```

- [x] **Step 5: Run capture-side focused tests**

Run: `.venv/bin/pytest tests/test_demo_flow.py tests/test_demo_analysis.py tests/test_demo_capture.py -q`

Expected: all selected tests pass.

---

### Task 4: Regression and replay verification

**Files:**
- Modify only if a failing regression proves a requirement gap.

**Interfaces:**
- Consumes: the authoritative inventory CSV and a saved real session.
- Produces: an inventory-valid workspace, native handoff, and retained web backup.

- [x] **Step 1: Run both full suites**

Run: `cd /Users/emily/lego-cv && .venv/bin/pytest -q`

Run: `cd /Users/emily/lego-capture && .venv/bin/pytest -q`

Expected: zero failures in both repositories.

- [x] **Step 2: Run static checks**

Run: `python -m py_compile demo_flow.py /Users/emily/lego-cv/opencv_confirmation.py`

Run in each repository: `git diff --check`

Expected: exit code zero.

- [x] **Step 3: Replay the newest complete session through processing**

Run:

```bash
.venv/bin/python demo_flow.py \
  --replay-session sessions/20260719-093100 \
  --inventory-csv outputs/019f72d0-9cbe-7431-8f4b-7ed4084bbd13/lego_inventory_for_model_generation_20260719.csv
```

Expected: no browser opens, no slope appears, 3–4 inventory-valid questions are available, native correction rejects invalid text, and the final component writes `handoff.json`.

- [x] **Step 4: Verify the browser backup still launches**

Run the same replay with `--confirmation-ui web --port 8781` and stop after the fresh workspace loads.

Expected: the existing yellow web reviewer opens with the same inventory-valid 3–4 components.
