# Weak Candidate Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rescue the black 3001 Brick 2 x 4 from retained weak appearance evidence plus compatible metric and stud evidence without changing accepted identities or increasing exact-ID errors.

**Architecture:** `pipeline.evidence` will aggregate one top appearance hypothesis per distinct frame, retain lower-ranked alternatives separately, and expose a unique weak leader only when at least two frames agree. The existing arbitration boundary will gain an unknown-only rescue branch that requires the weak leader plus both candidate-specific high-reliability metric compatibility and compatible stud evidence. `pipeline.multiview` will serialize and pass this evidence without changing the existing accepted-identity consensus path.

**Tech Stack:** Python 3.11, dataclasses, pytest, existing Brickognize/metric/stud evidence, cached real-session API responses.

## Global Constraints

- A wrong exact identity is worse than `unknown` or `review_needed`.
- Existing accepted identities do not enter the new weak-rescue branch.
- The existing acceptance threshold remains 0.70.
- The existing weak candidate floor remains 0.50.
- Only a distinct frame's top raw candidate counts as accepted or weak top support.
- Lower-ranked candidates remain alternatives and do not count as full votes.
- Weak rescue requires at least two agreeing top-hypothesis frames.
- Weak rescue requires both high-reliability compatible metric evidence and compatible medium/high stud evidence.
- Stud count and lattice remain one physical evidence category.
- High-reliability metric contradiction blocks rescue.
- Color is not an input to identity rescue.
- Plate 2445 and the upside-down 3003 remain protected.
- Component count remains exactly 13 in session `20260717-233252`.
- The ten currently correct session identities must not regress.
- Manual review remains at most seven components; identities are never forced to meet this budget.
- Production behavior changes require a failing test observed before implementation.
- Do not modify `lego-capture` production code in this phase.

---

### Task 1: Aggregate Accepted, Weak, and Alternative Appearance Support

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/evidence.py`
- Modify: `/Users/emily/lego-cv/tests/test_evidence.py`

**Interfaces:**
- Produces: `aggregate_appearance_support(entries, accepted_score=0.70, weak_score=0.50) -> dict`.
- Input entry schema: current `brickognize_views` dictionaries containing `frame_id`, `part_id`, `name`, `score`, and `candidates`.
- Output schema: `candidate_support` keyed by part ID plus `unique_top_candidate` and `unique_top_count`.
- Consumed by: Task 2 arbitration and Task 3 multiview serialization.

- [ ] **Step 1: Add a failing test for distinct-frame weak support and lower-ranked alternatives**

Append to `tests/test_evidence.py`:

```python
def test_appearance_support_counts_only_each_frames_top_raw_hypothesis():
    from pipeline.evidence import aggregate_appearance_support

    entries = [
        {
            "frame_id": 35, "part_id": None, "name": "unknown brick",
            "score": 0.615,
            "candidates": [
                {"part_id": "3001", "name": "Brick 2 x 4", "score": 0.615},
                {"part_id": "39789", "name": "Technic, Brick 2 x 4 with 3 Axle Holes", "score": 0.600},
            ],
        },
        {
            "frame_id": 54, "part_id": None, "name": "unknown brick",
            "score": 0.687,
            "candidates": [
                {"part_id": "3001", "name": "Brick 2 x 4", "score": 0.687},
                {"part_id": "39789", "name": "Technic, Brick 2 x 4 with 3 Axle Holes", "score": 0.578},
            ],
        },
        {
            "frame_id": 263, "part_id": None, "name": "unknown brick",
            "score": 0.699,
            "candidates": [
                {"part_id": "39789", "name": "Technic, Brick 2 x 4 with 3 Axle Holes", "score": 0.699},
            ],
        },
    ]

    result = aggregate_appearance_support(entries)

    assert result["unique_top_candidate"] == "3001"
    assert result["unique_top_count"] == 2
    assert result["candidate_support"]["3001"]["weak_frame_ids"] == [35, 54]
    assert result["candidate_support"]["39789"]["weak_frame_ids"] == [263]
    assert result["candidate_support"]["39789"]["alternative_frame_ids"] == [35, 54]
```

- [ ] **Step 2: Add a failing test proving accepted and weak observations are distinct states**

```python
def test_appearance_support_separates_accepted_and_weak_frames():
    from pipeline.evidence import aggregate_appearance_support

    result = aggregate_appearance_support([
        {
            "frame_id": 1, "part_id": "3001", "name": "Brick 2 x 4",
            "score": 0.81,
            "candidates": [
                {"part_id": "3001", "name": "Brick 2 x 4", "score": 0.81},
            ],
        },
        {
            "frame_id": 2, "part_id": None, "name": "unknown brick",
            "score": 0.64,
            "candidates": [
                {"part_id": "3001", "name": "Brick 2 x 4", "score": 0.64},
            ],
        },
    ])

    support = result["candidate_support"]["3001"]
    assert support["accepted_frame_ids"] == [1]
    assert support["weak_frame_ids"] == [2]
    assert support["top_frame_ids"] == [1, 2]
    assert support["best_score"] == 0.81
```

- [ ] **Step 3: Add a failing test proving a tie has no unique leader**

```python
def test_appearance_support_tie_has_no_unique_top_candidate():
    from pipeline.evidence import aggregate_appearance_support

    result = aggregate_appearance_support([
        {
            "frame_id": 1, "part_id": None, "name": "unknown brick",
            "score": 0.65,
            "candidates": [
                {"part_id": "3001", "name": "Brick 2 x 4", "score": 0.65},
            ],
        },
        {
            "frame_id": 2, "part_id": None, "name": "unknown brick",
            "score": 0.66,
            "candidates": [
                {"part_id": "39789", "name": "Technic, Brick 2 x 4 with 3 Axle Holes", "score": 0.66},
            ],
        },
    ])

    assert result["unique_top_candidate"] is None
    assert result["unique_top_count"] == 1
```

- [ ] **Step 4: Run the new tests and verify RED**

Run:

```bash
cd /Users/emily/lego-cv
.venv/bin/pytest tests/test_evidence.py -k appearance_support -v
```

Expected: FAIL because `aggregate_appearance_support` does not exist.

- [ ] **Step 5: Implement deterministic appearance aggregation**

Add to `pipeline/evidence.py`:

```python
ACCEPTED_APPEARANCE_SCORE = 0.70
WEAK_APPEARANCE_SCORE = 0.50


def _candidate_record(candidate):
    return {
        "part_id": candidate.get("part_id"),
        "name": str(candidate.get("name", "")),
        "score": float(candidate.get("score", 0.0)),
    }


def aggregate_appearance_support(
    entries,
    accepted_score=ACCEPTED_APPEARANCE_SCORE,
    weak_score=WEAK_APPEARANCE_SCORE,
):
    per_frame = {}
    for entry in entries:
        frame_id = int(entry["frame_id"])
        current = per_frame.get(frame_id)
        if current is None or float(entry.get("score", 0.0)) > float(
            current.get("score", 0.0)
        ):
            per_frame[frame_id] = entry

    support = {}

    def row_for(candidate):
        part_id = candidate.get("part_id")
        if part_id is None:
            return None
        part_id = str(part_id)
        return support.setdefault(part_id, {
            "part_id": part_id,
            "name": str(candidate.get("name", "")),
            "accepted_frame_ids": [],
            "weak_frame_ids": [],
            "top_frame_ids": [],
            "alternative_frame_ids": [],
            "best_score": 0.0,
        })

    for frame_id, entry in sorted(per_frame.items()):
        candidates = [
            _candidate_record(candidate)
            for candidate in entry.get("candidates", ())
            if candidate.get("part_id") is not None
        ]
        if entry.get("part_id") is not None:
            top = _candidate_record(entry)
            candidates = [
                top,
                *[candidate for candidate in candidates
                  if candidate["part_id"] != top["part_id"]],
            ]
        elif candidates:
            top = candidates[0]
        else:
            continue
        if top["score"] >= weak_score:
            row = row_for(top)
            row["top_frame_ids"].append(frame_id)
            state = (
                "accepted_frame_ids"
                if entry.get("part_id") is not None
                and top["score"] >= accepted_score
                else "weak_frame_ids"
            )
            row[state].append(frame_id)
            row["best_score"] = max(row["best_score"], top["score"])
        for candidate in candidates[1:]:
            if candidate["score"] < weak_score:
                continue
            row = row_for(candidate)
            row["alternative_frame_ids"].append(frame_id)
            row["best_score"] = max(row["best_score"], candidate["score"])

    counts = {
        part_id: len(row["top_frame_ids"])
        for part_id, row in support.items()
    }
    maximum = max(counts.values(), default=0)
    winners = sorted(
        part_id for part_id, count in counts.items() if count == maximum
    )
    return {
        "candidate_support": support,
        "unique_top_candidate": winners[0] if len(winners) == 1 else None,
        "unique_top_count": maximum,
    }
```

- [ ] **Step 6: Run Task 1 tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_evidence.py -k appearance_support -v
```

Expected: 3 tests PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add pipeline/evidence.py tests/test_evidence.py
git commit -m "feat: aggregate weak appearance support"
```

---

### Task 2: Require Weak Leader, Metric, and Stud Evidence for Rescue

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/evidence.py`
- Modify: `/Users/emily/lego-cv/tests/test_evidence.py`

**Interfaces:**
- Extends: `arbitrate_identification(..., appearance_evidence=None)`.
- Adds internal: `_candidate_metric_state(dimension_evidence, part_id) -> str` returning `supports`, `rejects`, or `unavailable`.
- Preserves every existing exact-identity arbitration path.

- [ ] **Step 1: Add a failing bounded-rescue test matching the black component**

```python
def test_unknown_is_rescued_by_unique_weak_leader_metric_and_studs():
    from pipeline.evidence import arbitrate_identification

    unknown = Identification(
        None, "unknown brick", None, 0.615,
        candidates=(
            {"part_id": "3001", "name": "Brick 2 x 4", "score": 0.615},
            {"part_id": "39789", "name": "Technic, Brick 2 x 4 with 3 Axle Holes", "score": 0.600},
        ),
    )
    proposal = Identification(
        "3001", "Brick 2 x 4", None, 0.615,
        candidates=unknown.candidates,
    )
    outcome = arbitrate_identification(
        current=unknown,
        dimension_action="unmodeled_candidate",
        dimension_evidence={
            "evidence": {"reliability": "high"},
            "decision": {"candidate_scores": [{
                "part_id": "3001", "name": "Brick 2 x 4",
                "modeled": True, "fits": True,
            }]},
        },
        stud_proposal=proposal,
        stud_action="rescued",
        identity_support={},
        appearance_evidence={
            "unique_top_candidate": "3001",
            "unique_top_count": 2,
            "candidate_support": {
                "3001": {"top_frame_ids": [35, 54]},
                "39789": {"top_frame_ids": [263]},
            },
        },
    )

    assert outcome.identification.part_id == "3001"
    assert outcome.action == "rescued"
    assert outcome.supporting_categories == (
        "weak_multi_view_appearance", "metric", "stud"
    )
```

- [ ] **Step 2: Add failing safety tests for contradiction, ties, and one-view guesses**

```python
def _weak_rescue_outcome(appearance_evidence, fits=True):
    from pipeline.evidence import arbitrate_identification

    unknown = Identification(
        None, "unknown brick", None, 0.62,
        candidates=(
            {"part_id": "3001", "name": "Brick 2 x 4", "score": 0.62},
        ),
    )
    proposal = Identification(
        "3001", "Brick 2 x 4", None, 0.62,
        candidates=unknown.candidates,
    )
    return arbitrate_identification(
        current=unknown,
        dimension_action="unmodeled_candidate",
        dimension_evidence={
            "evidence": {"reliability": "high"},
            "decision": {"candidate_scores": [{
                "part_id": "3001", "name": "Brick 2 x 4",
                "modeled": True, "fits": fits,
            }]},
        },
        stud_proposal=proposal,
        stud_action="rescued",
        identity_support={},
        appearance_evidence=appearance_evidence,
    )


def test_weak_rescue_is_blocked_by_high_reliability_metric_rejection():
    outcome = _weak_rescue_outcome({
        "unique_top_candidate": "3001", "unique_top_count": 2,
        "candidate_support": {"3001": {"top_frame_ids": [1, 2]}},
    }, fits=False)
    assert outcome.identification.part_id is None
    assert outcome.action == "conflict_preserved"


def test_weak_rescue_is_blocked_by_tied_top_support():
    outcome = _weak_rescue_outcome({
        "unique_top_candidate": None, "unique_top_count": 2,
        "candidate_support": {
            "3001": {"top_frame_ids": [1, 2]},
            "39789": {"top_frame_ids": [3, 4]},
        },
    })
    assert outcome.identification.part_id is None
    assert outcome.action == "conflict_preserved"


def test_weak_rescue_requires_two_distinct_top_frames():
    outcome = _weak_rescue_outcome({
        "unique_top_candidate": "3001", "unique_top_count": 1,
        "candidate_support": {"3001": {"top_frame_ids": [1]}},
    })
    assert outcome.identification.part_id is None
    assert outcome.action == "conflict_preserved"
```

- [ ] **Step 3: Run the new rescue tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_evidence.py -k 'unknown_is_rescued or weak_rescue' -v
```

Expected: FAIL because arbitration does not accept `appearance_evidence` and does not count candidate-specific compatible metric evidence.

- [ ] **Step 4: Implement candidate-specific metric state and the unknown-only rescue branch**

Add to `pipeline/evidence.py`:

```python
def _candidate_metric_state(dimension_evidence, part_id):
    if not dimension_evidence:
        return "unavailable"
    if dimension_evidence.get("evidence", {}).get("reliability") != "high":
        return "unavailable"
    for candidate in dimension_evidence.get("decision", {}).get(
        "candidate_scores", ()
    ):
        if candidate.get("part_id") != part_id:
            continue
        if candidate.get("modeled") is not True:
            return "unavailable"
        if candidate.get("fits") is True:
            return "supports"
        if candidate.get("fits") is False:
            return "rejects"
    return "unavailable"
```

Extend the arbitration signature with `appearance_evidence: dict | None = None`.
After rejecting a null stud proposal and before the existing accepted-identity
metric-protection branch, add:

```python
    metric_state = _candidate_metric_state(
        dimension_evidence, stud_proposal.part_id
    )
    if metric_state == "rejects":
        return ArbitrationOutcome(
            current,
            "conflict_preserved",
            tuple(supports),
            "high-reliability metric evidence rejects the proposed identity",
        )
    if current.part_id is None:
        appearance = appearance_evidence or {}
        unique_candidate = appearance.get("unique_top_candidate")
        unique_count = int(appearance.get("unique_top_count", 0))
        if (
            stud_action == "rescued"
            and stud_proposal.part_id == unique_candidate
            and unique_count >= 2
            and metric_state == "supports"
        ):
            return ArbitrationOutcome(
                stud_proposal,
                "rescued",
                ("weak_multi_view_appearance", "metric", "stud"),
                "unique weak appearance leader agrees with metric and stud evidence",
            )
        return ArbitrationOutcome(
            current,
            "conflict_preserved",
            tuple(supports),
            "weak rescue requires a two-view unique leader plus metric and stud support",
        )
```

Replace the duplicated manual metric-rejection scan later in the function with
`metric_state == "rejects"` while keeping its existing outcome and reason.

- [ ] **Step 5: Run all evidence tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_evidence.py -v
```

Expected: all old and new evidence tests PASS. Existing accepted correction and
protection tests must remain unchanged.

- [ ] **Step 6: Commit Task 2**

```bash
git add pipeline/evidence.py tests/test_evidence.py
git commit -m "fix: rescue corroborated weak candidates"
```

---

### Task 3: Serialize Appearance Evidence and Pass It Through Multiview Arbitration

**Files:**
- Modify: `/Users/emily/lego-cv/pipeline/multiview.py`
- Modify: `/Users/emily/lego-cv/tests/test_multiview_bridge.py`

**Interfaces:**
- Produces: `id_crop["appearance_support"]` for every component with identity views.
- Passes: serialized appearance support to `arbitrate_identification`.
- Preserves: current `identity_support`, `brickognize_views`, and final output schema.

- [ ] **Step 1: Add a failing serialization test to the family-review fixture**

Extend `test_family_review_uses_distinct_frame_consensus` with:

```python
    support = id_crops[0]["appearance_support"]
    assert support["unique_top_candidate"] == "3002"
    assert support["unique_top_count"] == 2
    assert support["candidate_support"]["3002"]["accepted_frame_ids"] == [2, 3]
```

- [ ] **Step 2: Add a failing one-view/early-return serialization test**

```python
def test_non_reviewed_identity_still_serializes_appearance_support():
    from pipeline.evidence import aggregate_appearance_support

    result = aggregate_appearance_support([{
        "frame_id": 7,
        "part_id": "2445",
        "name": "Plate 2 x 12",
        "score": 0.92,
        "candidates": [
            {"part_id": "2445", "name": "Plate 2 x 12", "score": 0.92},
        ],
    }])

    assert result["candidate_support"]["2445"]["accepted_frame_ids"] == [7]
    assert result["unique_top_candidate"] == "2445"
```

This pure assertion fixes the expected serialized schema used by the early
return path; the integration change below uses the same helper for all paths.

- [ ] **Step 3: Run the focused bridge test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_multiview_bridge.py::test_family_review_uses_distinct_frame_consensus -v
```

Expected: FAIL because `_second_chance` does not write `appearance_support`.

- [ ] **Step 4: Integrate appearance aggregation in both `_second_chance` paths**

Import `aggregate_appearance_support` with `arbitrate_identification`:

```python
from pipeline.evidence import (
    aggregate_appearance_support,
    arbitrate_identification,
)
```

In the no-review branch, after writing `brickognize_views` and
`identity_support`, add:

```python
            info["appearance_support"] = aggregate_appearance_support(entries)
```

In the reviewed branch, after `per_frame = _identity_entries_by_frame(entries)`
and before selecting the final provenance, add:

```python
        info["appearance_support"] = aggregate_appearance_support(per_frame)
```

- [ ] **Step 5: Pass appearance evidence into arbitration**

Extend the `_apply_stud_advisory` call:

```python
        outcome = arbitrate_identification(
            current=ident,
            dimension_action=dimension_action,
            dimension_evidence=info.get("metric_silhouette", {}),
            stud_proposal=proposal,
            stud_action=proposal_action,
            identity_support=info.get("identity_support", {}),
            appearance_evidence=info.get("appearance_support", {}),
        )
```

- [ ] **Step 6: Run focused multiview/evidence/stud tests**

Run:

```bash
.venv/bin/pytest tests/test_evidence.py tests/test_studs.py tests/test_multiview_bridge.py -v
```

Expected: PASS with no new warnings.

- [ ] **Step 7: Run the full CV suite and commit Task 3**

Run:

```bash
.venv/bin/pytest -q
```

Expected: 252 existing tests plus the new tests PASS, with only the pre-existing
Starlette warning.

Commit:

```bash
git add pipeline/evidence.py pipeline/multiview.py tests/test_evidence.py tests/test_multiview_bridge.py
git commit -m "feat: integrate weak appearance evidence"
```

---

### Task 4: Real-Session Promotion Gate and Phase 1 Report

**Files:**
- Generated only: `/Users/emily/lego-capture/sessions/20260717-233252/analysis-runs/phase1-weak-rescue/`
- Modify: `/Users/emily/lego-capture/PROGRESS.md`
- Modify: `/Users/emily/lego-capture/docs/superpowers/plans/2026-07-18-weak-candidate-rescue.md` only to check completed steps during execution.

**Interfaces:**
- Consumes: preserved baseline and cached Brickognize responses.
- Produces: candidate result, exact-ID comparison, review-queue count, commands,
  and hashes without modifying segmentation or capture behavior.

- [ ] **Step 1: Re-run both full suites immediately before the real session**

Run:

```bash
cd /Users/emily/lego-capture
.venv/bin/pytest -q

cd /Users/emily/lego-cv
.venv/bin/pytest -q
```

Expected: capture remains 80 PASS; CV passes all old and new tests with only the
known warning.

- [ ] **Step 2: Verify the preserved baseline before overwriting current outputs**

Run:

```bash
cd /Users/emily/lego-capture
shasum -a 256 -c \
  sessions/20260717-233252/analysis-baseline-residual-identities/SHA256SUMS
```

Expected: all 41 baseline files report `OK`.

- [ ] **Step 3: Run the real session through the unchanged segmentation path**

Run:

```bash
cd /Users/emily/lego-cv
.venv/bin/python session_cli.py \
  /Users/emily/lego-capture/sessions/20260717-233252 \
  --detector seg
```

Expected: 13 fused pieces. The run may update current session outputs, but the
verified baseline remains preserved.

- [ ] **Step 4: Copy the candidate result into a run-scoped directory**

Run:

```bash
mkdir -p /Users/emily/lego-capture/sessions/20260717-233252/analysis-runs/phase1-weak-rescue
cp /Users/emily/lego-capture/sessions/20260717-233252/inventory.json \
   /Users/emily/lego-capture/sessions/20260717-233252/multiview_result.json \
   /Users/emily/lego-capture/sessions/20260717-233252/topdown.jpg \
   /Users/emily/lego-capture/sessions/20260717-233252/analysis-runs/phase1-weak-rescue/
cp -R /Users/emily/lego-capture/sessions/20260717-233252/debug \
   /Users/emily/lego-capture/sessions/20260717-233252/analysis-runs/phase1-weak-rescue/
```

Expected: baseline and Phase 1 candidate artifacts exist side by side.

- [ ] **Step 5: Compare the 13 fused components by box IoU and exact identity**

Run:

```bash
cd /Users/emily/lego-capture
.venv/bin/python - <<'PY'
import json
from pathlib import Path

session = Path("sessions/20260717-233252")
baseline = json.loads((
    session / "analysis-baseline-residual-identities/multiview_result.json"
).read_text())["instances"]
candidate = json.loads((
    session / "analysis-runs/phase1-weak-rescue/multiview_result.json"
).read_text())["instances"]


def iou(first, second):
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


assert len(baseline) == len(candidate) == 13
remaining = set(range(len(candidate)))
matches = []
for old in baseline:
    index = max(remaining, key=lambda value: iou(old["box"], candidate[value]["box"]))
    overlap = iou(old["box"], candidate[index]["box"])
    assert overlap >= 0.70, (old["box"], candidate[index]["box"], overlap)
    remaining.remove(index)
    matches.append((old, candidate[index]))

black_box = [546, 472, 606, 560]
review_count = 0
for old, new in matches:
    old_id = old["id_crop"]["brickognize"]["part_id"]
    new_id = new["id_crop"]["brickognize"]["part_id"]
    flags = new["id_crop"].get("flags", [])
    print(old["box"], new["box"], old_id, new_id, flags)
    if old["box"] == black_box:
        assert old_id is None
        assert new_id == "3001"
        arbitration = new["id_crop"]["stud"]["arbitration"]
        assert arbitration["action"] == "rescued"
        assert arbitration["supporting_categories"] == [
            "weak_multi_view_appearance", "metric", "stud"
        ]
    else:
        assert new_id == old_id, (old["box"], old_id, new_id)
    if (
        new_id is None
        or "identity_view_disagreement" in flags
        or "ambiguous" in flags
        or "review_needed" in flags
    ):
        review_count += 1

assert any(
    new["id_crop"]["brickognize"]["part_id"] == "2445"
    for _, new in matches
)
assert review_count <= 7, review_count
print(f"matched=13 review_count={review_count}")
PY
```

The comparison passes only when:

- 13 baseline components match 13 candidate components.
- Box `[546,472,606,560]` changes from unknown to 3001 with action `rescued`.
- The ten baseline-correct identities remain unchanged.
- Plate 2445 remains 2445.
- No accepted component changes to a new incompatible exact identity.
- Unknown/ambiguous/review-needed count is at most seven.

If any condition fails, do not promote Phase 1. Return to the failing evidence
boundary with a new minimal RED test.

- [ ] **Step 6: Record Phase 1 results in `PROGRESS.md`**

Add a dated section containing:

- Test counts.
- Black component before/after identity and supporting categories.
- The complete list of any other identity changes.
- Component count.
- Exact-ID error count and review-queue count.
- Protected 2445 and upside-down 3003 outcomes.
- Remaining side-facing ambiguities.
- Statement that Phase 2 geometry is not yet promoted.

- [ ] **Step 7: Verify both repositories and commit the report**

Run:

```bash
git -C /Users/emily/lego-capture status --short
git -C /Users/emily/lego-cv status --short
```

Expected: only the intentional `PROGRESS.md` change is uncommitted in
`lego-capture`; `lego-cv` is clean after Tasks 1-3 commits. Generated session
artifacts are ignored.

Commit:

```bash
cd /Users/emily/lego-capture
git add PROGRESS.md
git commit -m "docs: record weak candidate rescue"
```

After this gate, write the separate calibrated-silhouette implementation plan
using the measured Phase 1 result. Do not modify geometry or arbitration further
inside this Phase 1 plan.
