# Studio Inventory Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate one editable BrickLink Studio MPD with predicted parts and colors for all twelve inventory sessions.

**Architecture:** A focused Python exporter reads existing run-scoped `review.json` files, resolves each component to an LDraw part and color, and writes an MPD plus provenance CSV. Pure selection and serialization functions are tested independently before the CLI generates the real artifact.

**Tech Stack:** Python 3.11 standard library, pytest, LDraw MPD text format.

## Global Constraints

- Do not modify source frames or historical session outputs.
- Prefill the model's current part and color; use confirmed review values when present.
- Write generated output only beneath `inventory-review-20260718/studio/`.
- Do not merge, push, or reset either repository.

---

### Task 1: Tested LDraw resolution and serialization

**Files:**
- Create: `scripts/export_inventory_studio.py`
- Create: `tests/test_export_inventory_studio.py`

**Interfaces:**
- Consumes: the existing `review.json` component schema and `scripts.prepare_inventory_review.BATCHES`.
- Produces: `resolve_component(component) -> ResolvedPiece`, `ldraw_color(name) -> int`, and `build_mpd(sessions) -> tuple[str, list[dict[str, str]]]`.

- [ ] **Step 1: Write failing tests** for confirmed-value precedence, first-choice ambiguous colors, candidate fallback, conspicuous no-candidate placeholders, and one-reference-per-component MPD output.
- [ ] **Step 2: Run `pytest -q tests/test_export_inventory_studio.py`** and verify failure because the exporter module is absent.
- [ ] **Step 3: Implement the minimal exporter** using standard LDraw type-1 records, twelve session submodels, a non-overlapping grid, and provenance comments.
- [ ] **Step 4: Run `pytest -q tests/test_export_inventory_studio.py`** and verify all focused tests pass.

### Task 2: Generate and verify the real Studio artifact

**Files:**
- Generate: `inventory-review-20260718/studio/lego-inventory-draft.mpd`
- Generate: `inventory-review-20260718/studio/lego-inventory-draft.csv`
- Modify: `PROGRESS.md`

**Interfaces:**
- Consumes: all twelve `sessions/<id>/analysis-runs/inventory-annotation-v1/review.json` files.
- Produces: a Studio-openable MPD and an audit sidecar with 810 component rows.

- [ ] **Step 1: Run `python scripts/export_inventory_studio.py`** and require twelve input sessions.
- [ ] **Step 2: Validate** twelve MPD submodels, 810 component provenance records, 810 CSV rows, and 810 part records within session submodels.
- [ ] **Step 3: Run `pytest -q`** and verify the full capture suite passes.
- [ ] **Step 4: Record the artifact paths, counts, placeholder count, and verification evidence in `PROGRESS.md`**.
