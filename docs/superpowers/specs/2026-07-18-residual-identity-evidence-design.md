# Residual Identity Evidence Design

**Date:** 2026-07-18

**Status:** Approved design; implementation not started

**Primary regression session:** `20260717-233252`

## Objective

Resolve three residual part-identity failures without regressing the ten
currently correct identities or the exact 13-component count. This work is
strictly about part identity and evidence. Color classification, Record3D
connectivity, capture latency, live 3D quality, and YOLO retraining are out of
scope.

A change may enter production only when it passes focused tests, preserves both
full suites, preserves all ten correct identities in the primary regression,
and does not introduce a demonstrated regression in labelled historical checks.
Geometry that fails calibration or separation checks remains diagnostic-only.

## Reproduced Baseline

- `lego-capture`: 80 tests pass.
- `lego-cv`: 252 tests pass with one pre-existing Starlette warning.
- The target session has 13 fused components and 13 total inventory items.
- Ten identities are correct when color is ignored.
- Frame 14 has 13 usable masks; this is not a segmentation-training case.
- Confidence sidecars are absent, so depth height is unavailable and shadow-only.

Stable target boxes in the table canvas are:

| Expected | Fused box | Current result |
|---|---|---|
| 3001 Brick 2 x 4, side-facing blue | `[288, 312, 359, 353]` | 3020 Plate 2 x 4 |
| 3003 Brick 2 x 2, side-facing white | `[386, 353, 430, 391]` | 3004 Brick 1 x 2 |
| 3001 Brick 2 x 4, black | `[546, 472, 606, 560]` | unknown brick |

Regression assertions locate components using box IoU and source-instance
evidence rather than fragile list indexes. The current result is preserved in
`sessions/20260717-233252/analysis-baseline-residual-identities/` with hashes.

## Root Causes

### Black Brick 2 x 4

Raw Brickognize candidates are already retained below the 0.70 acceptance
threshold. Candidate 3001 is the top raw hypothesis in frames 35 and 54;
candidate 39789 is top in frame 263.

The accepted identity becomes `unknown brick`. Dimension resolution records
that candidate 3001 is compatible with the high-reliability 21.33 x 39.78 mm
measurement, but returns `unmodeled_candidate` because the accepted unknown
name is outside the regular-part grammar. Stud logic proposes rescuing 3001
from medium-reliability eight-stud evidence. Arbitration preserves unknown
because it counts only stud support; candidate-specific metric support is
stranded.

### Side-Facing Blue Brick 2 x 4

Accepted views vote 3020, 3001, and 3020. Brick 2 x 4 and Plate 2 x 4 share the
same planar footprint and stud layout. Table-plane measurements spread by about
9.2 mm on the short axis and are correctly low reliability. Tightening the
planar tolerance would turn pose distortion into false vetoes.

### Side-Facing White Brick 2 x 2

Accepted views provide one 3004 vote and one 3003 vote; a third is unknown.
Consensus therefore preserves the primary 3004 and flags disagreement. Its
table-plane measurements spread by roughly 11 x 22 mm, so they cannot safely
resolve the pose.

## Architecture

The existing capture, segmentation, fusion, color, and inventory flow remains.
Three explicit identity stages are added:

1. Appearance state distinguishes accepted identities, weak top hypotheses,
   lower-ranked alternatives, and physically contradicted hypotheses.
2. Candidate-specific physical evaluation scores evidence against an exact
   candidate rather than only the currently accepted identity.
3. Evidence arbitration selects, rescues, corrects, or leaves ambiguous using
   named evidence categories and explicit reliability.

The existing table-plane metric measurement remains soft evidence. It is not a
manufacturing-tolerance veto.

## Phase 1: Candidate-Driven Weak Rescue

For each distinct frame, retain the accepted identity when the top score is at
least 0.70, or the top weak hypothesis when its score is at least the existing
0.50 candidate floor. Lower-ranked candidates remain diagnostic alternatives
but do not count as equal full-frame votes. These inherited thresholds will not
be retuned solely from the target session.

A weak exact-part candidate may replace unknown only when:

1. It uniquely leads top-hypothesis support across distinct frames.
2. At least two independent physical evidence categories agree.
3. No high-reliability physical evidence rejects it.
4. No incompatible candidate has equal or greater distinct-frame top support.
5. The explanation records all supporting and competing candidates.

For the black component, the physical categories are candidate-specific regular
2 x 4 metric compatibility and one combined stud count/lattice category. Stud
count and lattice are not counted separately because they share detections.

The rescue is independent of color. There will be no rule like "black plus
eight studs means 3001." Accepted 2445 Plate 2 x 12 cannot enter the unknown-only
rescue path, and upside-down 3003 circle detections cannot generate or force a
candidate.

Geometry-only catalog hypothesis generation is deferred. All three desired
identities already appear in real appearance evidence. A future component with
no recoverable appearance candidate remains unknown or ambiguous.

## Phase 2: Calibrated Regular-Part Silhouette Fitting

The first model library contains only appearance-supplied regular `Brick N x M`
and `Plate N x M` candidates. It does not classify slopes, Technic features,
decorations, or arbitrary LDraw families.

Coarse dimensions follow official LDraw proportions:

- 20 LDU / 8.0 mm stud pitch.
- 24 LDU / 9.6 mm brick height.
- 8 LDU / 3.2 mm plate height.
- 12 LDU / 4.8 mm stud diameter.
- 4 LDU / 1.6 mm stud height.

The initial renderer uses a rectangular body. A separate stud envelope is
enabled only if tests show improved calibrated separation without regression.

For every observation, load the original instance mask, RGB intrinsics and
size, camera-to-world pose, table transform, segmentation confidence, boundary
status, sharpness, and view geometry. Convert ARKit camera coordinates (+X
right, +Y up, -Z forward) explicitly to OpenCV coordinates (+X right, +Y down,
+Z forward). Do not use solvePnP because a mask supplies no known 3D-to-2D
correspondences.

Before fitting, table-plane points projected by the new code must agree with
the stored plane homography. This checks matrix direction, axis signs,
intrinsic scaling, and pixel convention. It does not prove absence of residual
lens distortion. Systematic real-mask misalignment blocks production use and
triggers calibration diagnosis instead of threshold tuning.

For each candidate:

1. Enumerate distinct studs-up, studs-down, and side-resting orientations.
2. Seed table x/y/yaw from the fused component.
3. Search a bounded coarse x/y/yaw grid.
4. Refine the best hypotheses with a deterministic derivative-free optimizer.
5. Use one shared pose and orientation across every view.

Pose bounds remain local enough to prevent matching a neighbor while covering
the known raised-object table-homography shift.

The renderer projects the 3D model into a local original-frame crop and records
separate IoU, symmetric boundary distance, under-coverage, and spill penalties.
View weights include segmentation confidence, boundary status, mask area,
sharpness, and geometric usefulness.

A production-quality fit requires a usable genuinely oblique view. If only one
oblique view is usable, robust trimming cannot discard it. Aggregation may trim
at most one additional outlier and records both untrimmed and robust results.

Every fit records candidate dimensions, best orientation and pose, per-view
loss terms, weights, trimmed-view decision, total score, runner-up margin, and
observed/predicted/difference overlays. Output is written beneath a run-scoped
analysis directory and never overwrites the baseline.

No production margin threshold is chosen from the three targets alone. A fit
may affect identity only after synthetic and labelled historical calibration.
Otherwise it returns `ambiguous` or `review_needed`.

## Phase 3: Evidence Arbitration

Replace the stud-only proposal boundary with a candidate evidence ledger. Each
exact candidate stores:

- Accepted and weak appearance support by distinct frame.
- Lower-ranked appearance alternatives.
- 3D model-fit result and margin.
- Table-plane metric compatibility.
- One combined stud count/lattice result.
- Pose compatibility.
- Shadow-only depth evidence.
- High-reliability contradictions.

High-reliability contradictions veto a candidate. Unknown may be rescued by a
unique weak leader plus two agreeing physical categories. A calibrated
high-margin 3D fit may resolve Brick-versus-Plate or orientation disagreement
when no strong contradiction exists. Low-margin geometry cannot override
appearance and returns ambiguous. Depth never votes.

Every outcome records the previous and final identities, action, supporting
categories, contradictions, alternatives, and a human-readable reason.

## Artifact Safety

`session_cli.py` currently overwrites `inventory.json`, `topdown.jpg`,
`multiview_result.json`, and same-named debug images. Development runs use an
explicit run-scoped output or preserve a verified snapshot first. Model-fit
overlays always use run-scoped paths. Baseline and candidate end-to-end outputs
remain side by side with hashes and exact commands.

## Testing and Non-Regression Gates

Every behavior change follows red-green-refactor and must first produce the
expected failing test.

Phase 1 tests cover successful bounded 3001 rescue, high-reliability metric
rejection, tied weak support, lower-ranked candidates not counting as full
votes, protected 2445, and the upside-down 3003 false-circle case.

Geometry tests cover ARKit/OpenCV projection agreement with plane homographies,
side-facing Brick 2 x 4 over Plate, side-facing Plate over Brick, Brick 2 x 2
over Brick 1 x 2, one corrupt auxiliary view, protection of the only oblique
view, low-margin ambiguity, and overlay output. Candidate-selection tests use
fixed expected masks or analytical expectations as well as renderer-generated
data so they are not purely self-confirming.

On `20260717-233252`:

1. Component count remains exactly 13.
2. All ten currently correct identities remain correct.
3. The side blue target becomes 3001 only with calibrated high-margin geometry;
   otherwise it remains explicitly ambiguous.
4. The side white target becomes 3003 under the same condition.
5. The black target becomes 3001 through bounded weak rescue.
6. Plate 2445 remains Plate 2 x 12.
7. Every target has inspectable evidence and overlays.

Historical labelled anchors include the known side-facing regular brick and
black Plate 2 x 12 in `20260717-202103`, the known upright Brick 1 x 8 and Plate
6 x 12 evidence in `20260717-005145`, and the previously verified upright Brick
1 x 8 and Plate 6 x 12 in `20260716-221544`.

Reports separate unit/synthetic results, projection calibration, the primary
session, historical anchors, unresolved ambiguities, and inherited versus
data-derived thresholds.

## Promotion and Stop Conditions

Phase 1 may ship independently after it rescues the black 3001 and passes every
protected-case and non-regression gate.

Phase 2 stays diagnostic-only if projection is inconsistent, synthetic families
do not separate, known historical anchors prefer the wrong family, target-fixing
thresholds regress labelled identities, or separation appears only after tuning
to the three targets. Phase 3 must then ignore geometry, and the final report
must state that it was not a reliable improvement.

The work is complete only when promoted changes preserve 13 components, all ten
baseline-correct identities, both full test suites, and evidence beyond the
three target labels. Merely changing the three target outputs is insufficient.

## Primary References

- LDraw units: <https://www.ldraw.org/article/218.html>
- Official LDraw library: <https://library.ldraw.org/>
- OpenCV projection: <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
- Visual hulls: <https://doi.org/10.1109/34.273735>
- Rebrickable API: <https://rebrickable.com/api/v3/docs/>
