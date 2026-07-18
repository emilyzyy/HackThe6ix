# Inventory Reliability Design

**Date:** 2026-07-18

**Status:** Approved design; implementation not started

**Primary regression session:** `20260717-233252`

## Objective

Improve inventory correctness by repairing workspace coverage, view selection,
and evidence arbitration before changing color classification, the cosmetic live
3D display, or the segmentation training set.

The implementation must preserve uncertainty when the available observations do
not justify a unique LEGO element. It must not convert a weak heuristic into a
confident answer merely to increase the number of finalized identifications.

## Scope

This design covers:

1. Selecting frames that collectively cover the admissible table workspace.
2. Selecting genuinely different 3D viewing directions rather than different
   azimuth bearings around a nearly top-down camera position.
3. Preventing stud-count evidence from overruling stronger agreeing evidence.
4. Using multiple original RGB views to review brick-versus-plate and difficult
   side-facing identifications.
5. Establishing an error-attribution gate that determines when segmentation
   training data is actually warranted.
6. Capturing depth-confidence data and evaluating height evidence in shadow
   mode, without allowing unvalidated depth evidence to alter inventory results.

The following are explicitly out of scope:

- Color-classification changes.
- Cosmetic improvements to the live Record3D visualization.
- Blindly reducing the Brickognize request interval. Latency tuning follows the
  correctness work and requires rate-limit telemetry.
- Retraining the segmentation model without demonstrated correct-ROI mask
  failures.

## Ground Truth and Existing Evidence

Session `20260717-233252` contains these 13 pieces:

1. Orange 1x8 brick.
2. Gray 2x4 brick.
3. Blue 2x3 brick.
4. Side-facing blue 2x4 brick.
5. White 2x3 brick.
6. Side-facing white 2x2 brick.
7. Black 2x12 plate.
8. Upside-down blue 2x2 brick.
9. A second white 2x3 brick.
10. White 2x6 brick.
11. Black 2x4 brick.
12. Side-facing green 1x1 brick.
13. Yellow 2x2 brick.

The current pipeline produced 12 components and seven correct part types when
color was ignored.

The following observations are established by local diagnostics:

- Direct segmentation of original frames 0 and 325 produces 13 masks. The
  orange 1x8 is detected at confidence 0.931 and 0.969 respectively.
- The exported views 49, 283, and 302 omit the orange piece.
- The current fused canvas is approximately 17 by 21.5 cm, while the detected
  table region is approximately 48 by 43.3 cm.
- The selected views report large bearing separation, but their actual
  camera-to-workspace viewing-ray separations are only 4.68, 8.62, and 12.85
  degrees.
- The capture contains quality candidates with substantially more oblique
  viewing directions; pairwise ray separation reaches approximately 84 degrees.
- The black 2x12 plate is initially classified as Plate 2x12 at score 0.868,
  and metric dimensions support a 2x12 footprint. A later eight-circle stud
  estimate replaces it with Technic Brick 1x12 at score 0.502.
- The upside-down blue 2x2 produces five alleged studs, demonstrating that a
  Hough circle is not equivalent to a visible top stud.
- Several brick-versus-plate errors have Brickognize scores between 0.886 and
  0.908 and therefore bypass the existing second-chance threshold of 0.85.

These observations establish that the regression session is not evidence for a
segmentation retraining need. They do not establish that the segmenter will be
adequate for every future environment.

## Architecture

The pipeline will retain its broad capture-to-inventory flow, with explicit
responsibility boundaries:

1. **Workspace geometry** defines where legitimate objects may be located.
2. **Candidate observation metadata** represents coverage, quality, camera pose,
   and view direction without retaining every full-resolution rectification.
3. **Set selection** chooses a small collection of frames that jointly provides
   workspace coverage and useful view-direction separation.
4. **Selected-view export** rectifies only the chosen frames over the complete
   admissible workspace.
5. **Segmentation and fusion** produce object components without assigning LEGO
   identities.
6. **Identification evidence aggregation** combines Brickognize candidates,
   metric dimensions, stud observations, pose information, and multi-view
   agreement under explicit precedence rules.
7. **Diagnostics** record why an answer was accepted, preserved, changed, or
   left ambiguous.

Each boundary must expose enough structured diagnostics to attribute a failure
without rerunning the entire capture manually.

## 1. Complete Workspace Coverage

### Design

The selection workspace will be derived from the stable hard admissible/table
contour. Dense depth occupancy may inform confidence or quality but must not crop
the semantic workspace.

Candidate evaluation will use a coarse grid over that workspace. Per-frame
candidate state will contain only low-resolution coverage masks and scalar
metadata. Once a small set of frames has been selected, only those frames will
be rectified at the existing export resolution over the full admissible bounds.
The current selection limits remain a minimum of three and maximum of five
views for this phase; the defect is which views are chosen, not an established
need to increase the request count.

This two-resolution design avoids retaining high-resolution full-table canvases
for hundreds of candidates while ensuring that objects outside the densest depth
region remain selectable.

The hard admissible contour remains independent of LEGO detections. A
detector-derived workspace would be circular: an object omitted by selection
could not expand the region used to select it.

### Selection Requirements

- Compute candidate coverage over the same complete admissible region used by
  the exporter.
- Target at least 99.5% of the valid-pixel union available from all candidates
  inside the admissible region. The export canvas itself always spans 100% of
  the hard-contour bounds, even where no frame supplies valid pixels.
- Prefer sets that increase union coverage, subject to minimum image and pose
  quality.
- Serialize the admissible bounds, per-view coverage, selected-set union
  coverage, and uncovered regions.
- Preserve the existing exclusion of detached background clutter.

### Acceptance Criteria

- On `20260717-233252`, the selected set collectively contains the orange 1x8.
- The exported workspace contains the full hard admissible contour.
- Fusion produces 13 components before part identification on the regression
  session.
- Existing historical sessions do not lose previously detected components solely
  because of the workspace change.

## 2. True 3D View Diversity

### Design

For each candidate frame, compute the normalized ray from the camera center to a
stable target point at the hard-workspace center. Pairwise diversity is the
angle between these rays, not the difference between horizontal bearings.

Set selection will balance:

- Union workspace coverage.
- Image sharpness and existing pose/depth quality signals.
- Minimum pairwise viewing-ray separation.
- Candidate validity and rectifiability.

Coverage gain remains the primary selection criterion. When further candidates
provide less than the existing 0.2% minimum coverage gain and confirmation views
are still required, select among candidates at or above the median quality by
maximizing their minimum viewing-ray separation from the already selected set.
Use time separation and then quality as deterministic tie-breakers.

The selector must not claim diversity solely because a near-centered camera
moved around the workspace center. Horizontal bearing remains diagnostic
metadata only.

### Threshold Policy

No universal ray-separation threshold will be selected from this one session.
The new metric will first be reported across existing development and validation
sessions. A threshold may then be selected from those distributions without
tuning against an untouched held-out test set.

If a capture lacks adequate candidates, the output will report insufficient
angular coverage. It will not fabricate an `angle_diverse=true` result.

### Acceptance Criteria

- On the regression session, selected frames have materially greater viewing-ray
  separation than the current maximum of 12.85 degrees when quality candidates
  with greater separation are available.
- Coverage cannot be sacrificed below the workspace-coverage requirement merely
  to maximize angle.
- Selection diagnostics include every selected ray, pairwise ray angles,
  quality values, coverage values, and any insufficiency reason.

## 3. Conflict-Aware Evidence Arbitration

### Design

Identification evidence will be aggregated before a final result is selected.
Stud counting will no longer execute as an unconditional final correction pass.

Evidence sources are classified as follows:

- **Identity candidate evidence:** Brickognize labels and scores from each view.
- **Physical footprint evidence:** metric width/length compatibility.
- **Stud observation evidence:** counts and reliability from preprocessing
  variants and views.
- **Pose evidence:** whether the observed face plausibly exposes top studs.

The aggregator will preserve the current identity when identity and physical
footprint evidence agree and only stud evidence conflicts. A correction requires
at least two independent evidence categories supporting the same alternative
and no strong contradictory physical evidence. The eligible categories are
multi-view identity agreement, metric footprint compatibility, and reliable
multi-view stud agreement. Pose determines whether an observation is eligible;
it is not an independent vote for an identity.

Ambiguity is a valid output. When competing candidates remain credible, the
result will retain the leading candidate but record a review/ambiguity state
rather than silently rewriting the identity.

### Stud Reliability

- Run overlapping preprocessing variants around dark and transitional luminance
  rather than selecting exactly one path at a hard threshold.
- Record the count produced by each variant.
- Require cross-variant agreement for high reliability.
- Prefer agreement across more than one usable view when those views expose the
  stud face.
- Treat upside-down, underside, or uncertain-face observations as non-corrective.
- Preserve raw observations in diagnostics even when they are rejected.

### Acceptance Criteria

- The black 2x12 remains Plate 2x12.
- A single stud estimate cannot replace an identity that agrees with metric
  dimensions.
- The upside-down blue 2x2's circle detections cannot independently trigger a
  correction.
- Existing cases in which stud evidence safely vetoes an impossible footprint
  remain detectable as conflicts or supported corrections under the new rules.

## 4. Multi-View Family Review

### Design

The identifier will retain and submit high-quality original RGB crops from
genuinely different selected views. Fused top-down imagery remains useful for
footprint measurement, but it is not the sole semantic identification image.

Family-aware review applies to:

- Brick-versus-plate alternatives with the same footprint.
- Side-facing objects whose projected stud grid does not express their true
  footprint.
- Cases where the top candidates disagree across views.
- Poses for which metric or stud evidence is known to be unreliable.

These cases may receive additional Brickognize requests even when a first score
exceeds 0.85. The review decision is based on semantic ambiguity and pose, not
only on a global confidence threshold.

Multi-view aggregation will prefer a candidate supported by independent views
and compatible physical evidence. When views disagree without a reliable
tie-breaker, the output will be marked ambiguous/review-needed.

### Acceptance Criteria

- Original oblique crops are traceable to source frame IDs in the manifest.
- Brick/plate same-footprint cases receive family-aware review independent of the
  old 0.85 cutoff.
- Side-facing white 2x2, white 2x3, white 2x6, and yellow 2x2 regression errors
  expose their per-view candidates and the exact final arbitration reason.
- No acceptance criterion requires the system to invent a unique identity where
  the observations remain genuinely ambiguous.

## 5. Depth Height as a Shadow Experiment

### Design

New Record3D captures will save the confidence frame corresponding to each depth
frame as a NumPy sidecar and add an optional `confidence` path to that frame's
JSONL record. Historical records without that field remain valid. A diagnostic
experiment will compute table-relative height distributions inside object masks,
filtered by valid depth and confidence.

This evidence remains shadow-only until calibration with known bricks and plates
demonstrates separation that is stable across position, surface, color, view
angle, and capture session. The experiment must report coverage and uncertainty,
not only a median height.

Depth height will not change an inventory identity in this implementation phase.

### Acceptance Criteria

- Newly captured depth frames have corresponding confidence data when supplied
  by the Record3D API.
- Missing confidence data remains supported for historical sessions.
- Shadow diagnostics record usable pixel count, confidence distribution,
  table-relative height summary, and rejection reason.
- No production identity changes because of shadow height evidence.

## 6. Segmentation Training Decision Gate

The following attribution order is mandatory:

1. If no source frame contains the object, classify it as a capture-coverage
   failure.
2. If source frames contain it but selected/exported views omit it, classify it
   as a workspace/view-selection failure.
3. If the correct exported ROI contains it but YOLO misses, merges, splits, or
   hallucinates a mask, classify it as a segmentation failure.
4. If masks are correct but fusion produces the wrong component count, classify
   it as a geometry/fusion failure.
5. If components are correct but LEGO identities are wrong, classify it as an
   identification/evidence failure.

Only item 3 justifies adding segmentation training data.

When item 3 is demonstrated, collect complete independently captured frames with
verified polygons. Target the demonstrated conditions, including dark pieces,
touching same-color pieces, side and upside-down poses, new surfaces, and
in-workspace negative clutter. Do not use isolated Brickognize identification
crops as if they were full-scene segmentation examples. Split new data by
capture session and preserve untouched sessions for evaluation.

Session `20260717-233252` will initially remain a regression fixture, not become
training data.

## Diagnostics and Error Handling

Every final component will retain a structured evidence trail containing:

- Source and selected frame IDs.
- Workspace and per-view coverage metadata.
- Viewing rays and pairwise ray angles.
- Raw Brickognize candidates per view.
- Metric compatibility decisions.
- Stud variant/view observations and reliability classification.
- Arbitration outcome and reason.
- Ambiguity or review-needed flags.
- Shadow depth-height diagnostics when available.

Failures in optional evidence sources must degrade gracefully. A missing
confidence frame, unusable oblique crop, or failed extra identification request
must not discard an otherwise valid component. It must be recorded as unavailable
evidence.

## Testing Strategy

Implementation will follow test-driven development for each behavior change.

### Unit Tests

- Hard-workspace bounds remain complete when dense occupancy is localized.
- Viewing-ray separation distinguishes real oblique views from azimuth-only
  movement.
- Set selection balances coverage and ray separation deterministically.
- Stud evidence cannot overrule agreeing identity and metric evidence.
- Cross-variant stud disagreement lowers reliability.
- Family-aware cases trigger review despite scores above 0.85.
- Missing confidence frames are accepted for historical sessions.
- The training-decision classifier assigns synthetic failures to the correct
  pipeline stage.

### Regression Tests

- Re-run `20260717-233252` through export, fusion, and inventory generation.
- Assert 13 pre-identification components.
- Assert inclusion of the orange 1x8 region.
- Assert the black piece's final type is Plate 2x12.
- Record, rather than conceal, unresolved brick/plate ambiguity.
- Run the existing capture and CV test suites after each repository change.

### Historical Evaluation

Run selection and inventory diagnostics on existing development sessions. Compare
component count, workspace coverage, angular separation, and identity regressions.
Do not tune thresholds using an untouched held-out evaluation session.

## Delivery Sequence

1. Complete-workspace geometry and coarse candidate coverage.
2. True 3D viewing-ray metadata and set selection.
3. Regression export/fusion validation on `20260717-233252`.
4. Conflict-aware evidence aggregation and stud reliability.
5. Multi-view family review and ambiguity reporting.
6. Confidence-frame capture and shadow height diagnostics.
7. End-to-end and historical regression analysis.
8. Segmentation training only if the decision gate exposes correct-ROI failures.
9. Brickognize latency benchmarking with HTTP 429 telemetry after correctness is
   stable.

## Definition of Done

This phase is complete when the selected views cover the full admissible
workspace, use genuine 3D view diversity when available, produce 13 components
for the primary regression session, preserve the black 2x12 plate against weak
stud evidence, and expose defensible evidence trails for remaining identification
decisions. Depth remains shadow-only, color remains unchanged, and any request
for additional segmentation training is supported by a demonstrated
correct-ROI mask failure.
