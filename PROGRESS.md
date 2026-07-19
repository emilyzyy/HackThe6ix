# LEGO Scanner — Progress

Covers both repos (`lego-capture`, `lego-cv`), branch
`codex/option-a-color-detection` in each. The latest reliability regression,
`sessions/20260717-233252`, now produces all 13 physical components; the older
`005145` touching-white-piece regression remains fixed at 8/8.

Note: the mission brief said to read CLAUDE.md first — no CLAUDE.md exists in
either repo or $HOME as of 2026-07-17; state was read from git history,
docs/superpowers specs+plans, and session artifacts instead.

## Mission status

| Phase | Scope | Status |
|---|---|---|
| 1 — Bridge (identify from original crops) | manifest v2 + original-crop identification + debug overhaul + pad fix | DONE — approved 2026-07-17 |
| 2 — Stud-count advisory tiebreaker | HoughCircles advisory + second-chance identification | DONE — approved 2026-07-17 |
| 3 — Trained separator (yolo11n-seg) | verified dataset + local MPS training + original-frame runtime + product evaluation | IMPLEMENTED — held-out `155228` remains 17/19 |
| 4 — Reliability-gated ID evidence | acceptance worksheet + dark studs + metric silhouettes + angle diversity + dimension flags | IMPLEMENTED 2026-07-17 — color unchanged |
| 5 — Inventory reliability | hard workspace + true view rays + conflict-aware evidence + confidence/depth shadow data + failure attribution | IMPLEMENTED 2026-07-18 — `233252` is 13/13 components; 3 part IDs remain ambiguous/wrong |

## Known constraints (do NOT)

Tune seam blending; redetect the mosaic; single global best frame; classical
detector on raw frames without ROI (over-detects 14-26 vs 8); blanket
drop-small-boxes; merge/push branches.

## Acceptance sessions (ground truth)

| Session | Pieces visible | Expected after Phase 1 |
|---|---|---|
| 20260716-212422 | 7 | 7 |
| 20260716-221544 | 8 | 8 |
| 20260716-234326 | 8 | 8 |
| 20260717-005145 | 8 | 7 (known same-color merge — Phase 3 fixes) |

## Phase 1 GATE results (2026-07-17) — APPROVED

Counts unchanged 7/8/8/7 ✓. Unknowns 9 → 4 ✓. Orange 1x8: fixed on 005145
(was 1x10) ✓ and kept on 221544 ✓; regressed to unknown on 234326 and reads
2x8 on 212422. Gray 6x12 plate: correct on 221544/234326, reads "Brick 1x8"
on 005145. Crop galleries (`sessions/<ts>/debug/crop_gallery.jpg`) show all
remaining errors are stud-count/aspect class errors on CLEAN single-piece
crops — Phase 2's designed target. Recommendation: approve and proceed to
Phase 2 rather than tuning crops further.

| Session | Before | After |
|---|---|---|
| 212422 | 7 pcs, 4 unknown | 7 pcs, 1 unknown (white plate at 0.67, just under the 0.70 gate); orange beam "2x8" (true 1x8) |
| 221544 | 8 pcs, 1 unknown | 8 pcs, 1 unknown; 1x8 orange ✓, 6x12 gray ✓ |
| 234326 | 8 pcs, 2 unknown | 8 pcs, 2 unknown (orange beam now unknown — was correct pre-bridge; red 2x2) |
| 005145 | 7 pcs, 1x10 misID | 7 pcs, 0 unknown; **1x8 ✓**; gray 6x12 reads "Brick 1x8"; white merge persists (known, Phase 3) |

Phase 1 deliverables, all committed on codex/option-a-color-detection:
- lego-capture: symmetric 25 mm workspace pad; manifest v2 (per-view
  homography canvas→original px, original rgb path/size, raise_uv_per_cm).
- lego-cv: pipeline/original_crop.py (AABB mapping, fit check, quad
  masking); ID view chosen by obliquity/sharpness with original-frame fit
  (canvas completeness demoted); identification from masked original crops;
  provenance (`id_crop`) in multiview_result.json; debug overhaul —
  box-free mosaic + debug/crop_gallery.jpg + per-frame overlays.
- Tests: lego-capture 61 passing, lego-cv 100 passing.

## Phase 2 GATE results (2026-07-17) — APPROVED

Counts stable 7/8/8/7 ✓. Orange beam correct in 3 of 4 sessions
(221544 ✓, 234326 ✓ fixed, 005145 ✓; 212422 still "2x8" — see below).
221544 is now perfect: 8/8 identified, zero unknowns.

Acceptance cases:
1. 234326 orange regression → **FIXED**: Brick 1x8 (0.84) via second-chance
   identification (weak chosen-crop IDs retry alternate observations' crops;
   all alternates evaluated, best score wins).
2. 212422 "2x8"-for-1x8 → NOT FIXED, correctly skipped: the only view is a
   side view (stud check's own "not camera-facing" rule; Hough finds 0-1
   circles). Single-view session — no alternate crop exists.
3. 005145 1x8-for-6x12 gray → confident-wrong **eliminated** (demoted to
   unknown): stud count 75 contradicts every 8-stud candidate; "Plate 6x12"
   is not among Brickognize's candidates for that crop because the crop
   contains BOTH the plate and the orange beam (merged observation box —
   the Phase 3 detector's territory).
4. 212422 white at 0.67 → NOT rescued: white-on-white studs give Hough <=1
   circle (unreliable -> advisory skips; global threshold untouched per
   instruction). The analogous white in 221544 WAS recovered (0.79) by
   second-chance. Single-view session again.

Assessment: both unfixed cases are single-view limitations of session
212422, not advisory bugs. Options for Emily: (a) accept and proceed to
Phase 3 (better per-frame boxes also improve crops), (b) recapture 212422
with a fuller sweep so multiview selects >=2 views.

Phase 2 deliverables (lego-cv, committed): pipeline/studs.py (HoughCircles
count, 3x contradiction band, rescue/correct/demote, never overrides
>=0.90); Identification.candidates; second-chance identification;
stud provenance in multiview_result.json. 118 tests passing (real 6x12 and
1x8 crops as fixtures).

## Phase 3 result (2026-07-17) — IMPLEMENTED AND EVALUATED

Human review covered 71 original RGB frames and 1,157 individual-piece
polygons. Whole-session splits contain 48 train, 15 validation, and 8 held-out
white-table frames with no session leakage. The content-addressed dataset hash
is `a751e0b2b456f867c62c6c50467ba3bc635869d4fe2706ac9ced31579a0c1a8c`.

`yolo11n-seg` trained locally on Apple MPS for 100 epochs in 597 seconds. The
installed checkpoint hash is
`c792bdad68351b6930d6e67dd41282b497a7c06b05ee7107dbe416b2ff4ea689`.
Frame-level mask mAP50-95 is 0.847 train-seen, 0.837 whole-session validation,
and 0.877 on the held-out `white_round` environment. These are segmentation
metrics, not inventory accuracy.

Runtime behavior:
- manifest v3 exports a detector-independent repeated table-plane workspace;
- YOLO runs on projected crops of each original RGB frame, never on a stitched
  mosaic;
- masks retain source IDs, gate evidence, original/canvas geometry, and crop
  provenance through table-coordinate fusion;
- the capture exporter keeps at least three confirmation views and now chooses
  circular camera-bearing diversity after coverage is complete;
- boundary-only segmented evidence needs multi-view support, while a strong
  complete singleton is still allowed where other views do not cover it;
- identification uses isolated mask crops and may review a moderate result
  with real padded context only when no neighboring mask enters that crop;
- `session_cli.py --detector auto` prefers the installed verified segmenter and
  writes physical-workspace/gate overlays plus the existing crop gallery.

Product count results use manually inspected historical counts or the maximum
human-verified source-frame count as the reference. Train-seen/historical:
7/7 (`212422`), 8/8 (`221544`), 8/8 (`234326`), and 8/8 (`005145`). The target
now separates both touching white pieces and identifies the orange piece as
Brick 1 x 8 (0.895). All five untouched validation sessions are count-exact:
14/14, 7/7, 23/23, 8/8, and 15/15; `152458` retains one `unknown brick`.

Held-out `white_round` remains separate: `153942` is 11/11, while `155228` is
17/19 with one unknown. Both have weak plane fits (0.587 and 0.705 inliers).
The `155228` shortfall is classified as geometry plus fusion/count mismatch:
per-view accepted-mask counts were 24, 19, and 18, but unstable projection and
edge evidence yielded 17 fused records. It was not retuned after observing the
held-out result.

Reports and operating guidance:
- `lego-cv/training/runs/product-evaluation/product-evaluation.{json,md}`
- `lego-cv/docs/phase3-capture-quality.md`
- `lego-cv/models/lego_seg.metadata.json`

Exact next live command:

```bash
cd /Users/emily/lego-capture
.venv/bin/python multiview.py sessions/<timestamp>

cd /Users/emily/lego-cv
.venv/bin/python session_cli.py \
  /Users/emily/lego-capture/sessions/<timestamp> --detector auto
```

Latest verification at this gate: 68 capture tests and 225 CV tests pass.

## Phase 4 reliability evidence (2026-07-17) — IMPLEMENTED

Scope stops after dimension consistency. Color correction, speed work,
rendering, confirmation UI, and large-batch expansion were deliberately not
changed.

Acceptance and provenance:
- `lego-cv/training/acceptance/20260717-202103.json` contains 29 stable
  table-anchor rows. Every row is `user_verified: false`, so current acceptance
  accuracy is intentionally `null` until Emily reviews it.
- `multiview_result.json` now preserves raw Brickognize candidates, metric
  evidence/action, structured stud evidence/action, final ID, ambiguity flags,
  selected observation frames, unchanged final color, and stage timings.

Dark-piece stud result:
- The black Plate 2 x 12 (`2445`, raw score 0.8905) has masked median luminance
  34/255. Dark CLAHE and gamma+CLAHE variants count 22 and 24 studs; the robust
  count is 23, reliability is high, action is compatible, and final ID remains
  `2445`.
- The dark-blue Plate 4 x 12 produces 48 studs from both dark variants.
- Low/disagreeing undercounts remain visible in provenance but cannot demote an
  ID. High-reliability undercount and overcount evidence remains bidirectional.

Metric silhouette and dimension result:
- Shadow calibration on `20260717-202103`: black 2 x 12 = 19.6 x 98.8 mm;
  dark-blue 4 x 12 = approximately 42.3 x 102.9 mm; light-blue 6 x 12 remained
  inside the conservative projection bound; side-lying blue piece =
  approximately 17.0 x 30.2 mm.
- The initial consistency tolerance is `max(8 mm, 35% of nominal side)`. This
  is a view-projection allowance, not manufacturing tolerance. It keeps the
  audited correct large plates compatible while exposing the raw 1 x 3 versus
  2 x 3 small-side conflict.
- The side-lying blue raw result remains `3622` Brick 1 x 3 because Brickognize
  supplied no 2 x 3 candidate. It is no longer silently accepted: action is
  `flagged`, flag is `possible_non_canonical_pose`, and the proposed family is
  `2 x 3`. No identity is invented.

Angle-diverse capture result:
- Re-export selected frames 387, 374, and 72 with bearings approximately
  107.3, 292.6, and 198.2 degrees. Pairwise separation is about 91–175 degrees,
  so `selection.angle_diverse` is true. Legacy version-3 manifests still load
  with geometry unavailable and receive `view_diversity_unknown` provenance.

Split-safe regression gate:
- TRAIN-SEEN `005145`: 8/8, zero unknown, no raw-to-final ID change.
- VALIDATION wood `150212`: 14/14, zero unknown, no raw-to-final ID change.
- VALIDATION gray `154929`: 15/15, zero unknown, no raw-to-final ID change.
- TEST-ENVIRONMENT `155228`: still 17/19, zero unknown. One raw 1 x 6 was
  changed to an existing 2 x 6 API candidate by frozen high-reliability metric
  evidence; this was recorded only after thresholds were frozen and was not
  used for tuning.
- Evaluation artifact:
  `lego-cv/training/runs/reliability-evidence-20260717/` (gitignored).

## Phase 5 inventory reliability (2026-07-18) — IMPLEMENTED

Scope deliberately excludes color correction, cosmetic live-3D rendering,
Brickognize latency tuning, and unproven segmentation retraining.

Capture and selection changes:
- The export canvas now uses the hard admissible workspace, not the dense depth
  core. On `233252` it grew from 340 x 430 to 961 x 868 pixels and spans roughly
  48.3 x 43.5 cm.
- Candidate coverage is evaluated at 250 px/m; only the selected views are
  rectified at the final 2000 px/m scale.
- Selection uses normalized 3D camera-to-workspace rays. Frames 14, 54, and 35
  are coverage anchors; frames 263 and 231 are oblique confirmations at 30.3
  and 56.8 degrees tilt. Confirmation-involving ray separation is at least
  37.9 degrees and reaches 77.7 degrees.
- The five-view cap cannot satisfy the original 99.5% target on this recording.
  The best five coverage-only candidates reach about 98.7% of the recorded
  valid-pixel union, while nine are required for 99.5%. The selected 3+2
  coverage/confirmation set reaches 97.50% of the recorded union (93.13% of the
  hard workspace); the manifest records `coverage_complete: false` and the
  insufficiency reason instead of claiming success.
- New Record3D captures persist optional confidence sidecars. Historical frames
  remain readable; `233252` predates the sidecars, so height evidence is
  correctly serialized as shadow-only/unavailable and never changes identity.

Identification and regression result:
- Fusion initially produced a false 14th component: an oblique blue detection
  and an oblique white detection cross-matched because table-plane projection
  shifts raised bricks. A confirmation-only reassignment pass now attaches each
  observation to its distinct, same-family coverage anchor only under a strong
  absolute geometry match, while preserving one observation per view and
  leaving ambiguous or single-anchor groups untouched.
- `233252` now produces exactly 13 components from 57 accepted detections across
  five views. Frame 14 alone visibly contains 13 correct masks, so this session
  is not a segmentation-training case.
- Ten of 13 part identities are correct when color is ignored. Residuals are:
  side blue 2x4 -> Plate 2x4 (two Plate votes, one Brick vote); side white 2x2
  -> Brick 1x2 (one 1x2 vote, one 2x2 vote, explicitly flagged view
  disagreement); black 2x4 -> unknown (all reviewed Brickognize scores below
  threshold despite a high-reliability approximately 2x4 metric footprint and
  medium eight-stud evidence).
- The black 2x12 remains part `2445`, Plate 2x12. Stud evidence can no longer
  demote or overwrite stronger identity/metric evidence, and the upside-down
  blue 2x2's circle detections remain non-corrective.
- Failure attribution is therefore: capture pass -> selection pass ->
  segmentation pass -> fusion pass after repair -> identification/evidence for
  the three residuals. Additional YOLO training crops are not warranted by this
  regression.

Latest verification: 80 capture tests and 252 CV tests pass; the CV suite has
one pre-existing Starlette deprecation warning.

## Phase 6A bounded weak-candidate rescue (2026-07-18) — PROMOTED

This phase changes only the handling of subthreshold appearance hypotheses.
It does not yet use or promote 3D silhouette fitting.

TDD and regression result:
- `lego-capture`: 80 tests pass.
- `lego-cv`: 259 tests pass with the same pre-existing Starlette warning.
- The preserved pre-run artifact checksum verified all 41 baseline files.
- The new run produced exactly 13 fused components. Box-IoU comparison matched
  all 13 to the preserved baseline with unchanged fused boxes.
- The black component at `[546, 472, 606, 560]` changed from unknown to LDraw
  `3001` Brick 2 x 4. Its arbitration categories are
  `weak_multi_view_appearance`, `metric`, and `stud`: 3001 is the top weak
  hypothesis in frames 35 and 54, high-reliability candidate-specific metric
  evidence is compatible, and medium-reliability stud consensus supports the
  same candidate.
- No other identity changed. The ten previously correct identities remain
  correct, black Plate 2 x 12 remains `2445`, and the upside-down blue 2 x 2
  remains `3003` without a circle-driven correction.
- The measured final-review queue is one component under the current explicit
  disagreement/unknown flags, below the seven-piece usability budget.
- Exact identity accuracy on the primary session is now 11/13 when color is
  ignored. The side blue 2 x 4 still reads `3020`, and the side white 2 x 2
  still reads `3004`. These are not described as fixed; calibrated silhouette
  fitting remains required, and uncertain geometry must prefer review/unknown
  over a forced exact identity.

Thresholds were not tuned from this inventory. The rescue uses the existing
0.70 acceptance threshold and 0.50 candidate floor. Lower-ranked candidates do
not count as full distinct-frame votes. The run-scoped candidate artifacts are
under `sessions/20260717-233252/analysis-runs/phase1-weak-rescue/`.

## Phase 6B calibrated silhouette fitting (2026-07-18) — DIAGNOSTIC ONLY

This phase implemented calibrated multi-view cuboid fitting for exact regular
`Brick N x M` and `Plate N x M` candidates. It is opt-in through
`--analysis-dir`, runs only after final identity arbitration, and cannot feed
dimension consistency, stud advice, or production identity selection. A
diagnostic run writes its inventory, result JSON, gallery, and overlays beneath
the supplied directory without rewriting the session's normal outputs.

Calibration and synthetic gates:
- ARKit pose/intrinsics projection agrees with the stored frame-14 table-plane
  homography to `1e-5` pixels.
- Synthetic masks separate side-resting Brick 2 x 4 from Plate 2 x 4 in both
  directions, Brick 2 x 2 from Brick 1 x 2, tolerate at most one corrupt
  nonessential auxiliary view, never trim the sole oblique view, and return low
  reliability without an oblique view.
- The first real implementation was rejected on runtime: it produced no
  diagnostic instance after nearly three minutes. TDD-bounded original-mask
  crops reduced the final completed 13-piece primary run to 130.11 seconds. This is
  still an offline diagnostic and adds zero work to a normal scan.

Primary session result (`20260717-233252`):
- Exactly 13 components were retained and all production identities remained
  byte-for-byte equal to the promoted Phase 6A run: 11/13 exact IDs remain
  correct, Plate 2 x 12 remains `2445`, and the black 2 x 4 remains rescued as
  `3001`.
- The side blue target at `[288, 312, 359, 353]` prefers `3001` Brick 2 x 4
  with loss 0.0820 over `3020` Plate 2 x 4 at 0.1631 (margin 0.0811).
- The side white target at `[386, 353, 430, 391]` prefers `3003` Brick 2 x 2
  with loss 0.1779 over `3004` Brick 1 x 2 at 0.2939 (margin 0.1161).
- Protected comparison anchors also prefer their existing identities: `2445`
  over `3034`, `3001` over `3010`, and `3003` over `3022`.
- Diagnostic-quality disagreement sets `diagnostic_model_disagreement` and
  recommends review without changing the exact ID. Exactly the two residual
  targets are recommended for review, below the seven-piece usability limit,
  rather than forcing a guess.

Historical gate:
- `20260717-202103`: the known side-facing Brick 2 x 3 prefers `3002` over the
  misleading `3622` by 0.0770. Plate `2445` has no second accepted/weak regular
  candidate and is correctly unavailable rather than force-compared.
- `20260717-005145`: known Plate 6 x 12 prefers `3028` over `2445` by 0.1627;
  Brick 1 x 8 has no second regular candidate and is unavailable.
- `20260716-221544`: known Plate 6 x 12 prefers `3028` over `3033` by 0.0409;
  Brick 1 x 8 again has no second regular candidate and is unavailable.
- Camera-pose/table-frame derivation shows the legacy optical tilts are only
  0.6-2.1, 2.6-6.8, and 7.5-11.4 degrees respectively. These captures contain
  no mandatory >=30-degree oblique anchor, so all historical comparisons are
  low-reliability even though each available comparison prefers its known
  family.

Promotion decision: the diagnostic is a real investigative improvement and
correctly ranks both primary residuals, but the approved historical obliquity
gate is not satisfied. Geometry therefore remains diagnostic-only; no Phase 3
identity-arbitration plan was written and the two production misreads were not
silently relabelled. Run-scoped evidence is under each evaluated session's
`analysis-runs/model-fit-diagnostic/` directory.

## Phase 7A color evidence and ground-truth gate (2026-07-18) — COMPLETE

This phase deliberately changes no color decision. It records the evidence
needed to locate each failure before correction thresholds are designed.

- Every fused component now serializes per-view raw and white-patch-corrected
  RGB, corrected Lab, the top three palette matches with CIEDE2000 distances
  and best/second margin, luminance quantiles and clip fractions, background
  neutrality, segmentation/view metadata, and diagnostic shadow/specular,
  oblique, boundary, and low-confidence flags.
- The current final provenance remains explicitly
  `strategy: unweighted_name_vote`; the color output is not represented as a
  new decision system yet.
- `session_cli.py --color-worksheet --analysis-dir <run>` writes one crop per
  fused component plus CSV and Markdown worksheets beneath the run directory.
  Color-only worksheet runs skip the opt-in 130-second model-fit diagnostic.
- `python -m training.color_acceptance <worksheet.csv>` scores only rows with a
  human-filled truth color and reports exact accuracy, ambiguity containing the
  truth, confident wrongs, per-color results, and the confusion summary.
- The primary run contains 13 blank-truth worksheet rows, 13 thumbnails, and
  56 complete per-view color-evidence records. Its component boxes, part IDs,
  colors, and inventory are unchanged from the promoted Phase 6A baseline.
- Run-scoped artifact:
  `sessions/20260717-233252/analysis-runs/color-phase1-evidence/`.
- Verification at this gate: `lego-capture` 80 tests pass; `lego-cv` 285 tests
  pass with the existing Starlette warning.

Emily supplied all 13 truth labels for `233252`. The measured Phase 1 baseline
is 8/13 exact (61.5%), 8/13 acceptable, and five confidently wrong colors.

## Phase 7B trustworthy color decision (2026-07-18) — PROMOTED ON PRIMARY LABELLED SESSION

Root cause and implementation:
- Raw samples already put every named failure in the correct broad family. The
  legacy white-patch step was the primary entry point for all five confident
  errors: the warm table reference was forced toward white, boosting blue and
  clipping bright channels. Palette shade boundaries and repeated unweighted
  votes were secondary contributors.
- Session-level, brightness-preserving gray-world correction now replaces the
  exposure-forcing decision path. It verifies reference brightness/chroma/view
  consistency and falls back to raw when the table is not a defensible neutral
  reference. Per-object safeguards reject chroma amplification, excessive hue
  shift, and needless correction of strongly chromatic samples. The local
  OpenCV lacks `cv2.xphoto`, so the equivalent channel gains are implemented
  directly without changing OpenCV distributions.
- Low-chroma evidence is restricted to the neutral palette. Chromatic evidence
  excludes neutrals and uses hue-first scoring. Small aggregate margins and
  shade/exposure overlaps emit `ambiguous(colorA/colorB)` rather than a
  confident error.
- Final color aggregation is quality weighted; shadowed, specular, oblique,
  boundary, and low-confidence observations are down-weighted with the exact
  weights/reasons persisted. The old per-view name remains only as a fusion
  hint, so color changes cannot alter the established geometry association.

Primary labelled gate (`20260717-233252`):
- Exactly 13 components, all 13 boxes, and all 13 part IDs match Phase 1.
- Exact color accuracy rises from 8/13 (61.5%) to 9/13 (69.2%). Acceptable
  correct-or-ambiguity-containing-truth rises from 8/13 to 13/13 (100%).
  Confident wrong colors fall from five to zero.
- Dark gray and yellow are now exact. Both black pieces are
  `ambiguous(black/dark gray)`, green is `ambiguous(dark green/green)`, and
  orange is conservatively `ambiguous(orange/red)`. All prior blue and white
  successes remain exact.
- The final run has 56 complete weighted per-view provenance records. It is at
  `sessions/20260717-233252/analysis-runs/color-phase2-final/`; detailed failure
  attribution is in `color_diagnosis.md` and the score is in
  `color-labels/after_score.json`.

Regression gates: 294 CV tests and 80 capture tests pass; all 41 preserved
historical artifact checksums pass; `IMG_4841.jpg` remains unchanged. This is
labelled-session evidence only, not proof across all lighting. No additional
session has human color truth yet, so the requested cross-session no-confusion
gate remains unmeasured rather than claimed.

## Phase 8 pre-latency reliability (2026-07-18) — IMPLEMENTED, REAL-SESSION REVIEW PENDING

Scope deliberately excludes Brickognize throttling/concurrency and DINO or an
exemplar bank. It addresses the approved palette, duplicate-mask, and
piece-specific crop-quality work first.

Implementation:
- The full 24-color reference palette remains unchanged for per-view evidence,
  diagnostics, and fusion hints. Final inventory aggregation now uses a
  separate physical 12-color profile: red, orange, yellow, beige, brown,
  green, dark green, blue, white, light gray, dark gray, and black. Final
  ambiguity can only contain two names from that profile. Provenance records
  the active `output_palette`.
- Accepted detections from one source view now undergo conservative mask-level
  duplicate suppression before fusion. A lower-confidence mask is retained in
  the audit output but gated as `duplicate_suppressed` only when mask IoU is at
  least 0.65, smaller-mask containment is at least 0.85, mask area ratio is at
  least 0.70, and original-box IoU is at least 0.60. Different views and
  overlapping boxes with disjoint masks are protected by regression tests.
- Identification view ranking now measures the 80th-percentile absolute
  Laplacian response inside an eroded piece mask, avoiding an artificial score
  from the segmentation edge. The ranking combines local focus, crop size,
  view tilt, segmentation confidence, original-frame fit, and the legacy
  whole-view score. The exact factors are persisted as
  `identification_focus` and `identification_quality`.
- If all Brickognize observations remain below 0.70, production identity stays
  unknown. The gallery/provenance now selects the strongest weak observation
  and adds `best_weak_view_selected`; no threshold was lowered.

Run-scoped validation (`20260718-130245`):
- Output is under
  `sessions/20260718-130245/analysis-runs/pre-latency-reliability/`; normal
  session artifacts were not rewritten.
- Final output contains no excluded shade name. It contains approved profile
  names plus three `ambiguous(dark green/green)` results; both sides of that
  ambiguity are in the physical profile. This is a candidate-space guarantee,
  not color-accuracy evidence because this session has no human color truth.
- Eight lower-confidence same-view masks were marked
  `duplicate_suppressed`, paired with stronger masks at original-box IoU
  0.647-0.706. Fused count falls from 60 to 56.
- The user-confirmed duplicate white Plate 1 x 6 formerly represented by
  pieces 003 and 004 is now one component with observations from frames 503,
  12, and 306 and final ID `3666` Plate 1 x 6.
- Two additional clear before/after consolidation candidates are former pairs
  013/014 (both Brick 1 x 2) and 046/047 (both unknown). The fourth count
  reduction involves reassociation in the dense red cluster around canvas
  y=710. These three are not claimed correct until compared with the physical
  mat; the gate overlays preserve every suppressed source ID for review.
- The upside-down dark-gray Brick 2 x 2 stays unknown, but its displayed weak
  evidence improves from frame 503 score 0.546 to frame 12 score 0.682, with
  `3003` Brick 2 x 2 still the top candidate and
  `best_weak_view_selected` recorded. The upside-down white Brick 2 x 2 stays
  unknown at 0.640; it already used its strongest available view.
- Seven of ten unknowns selected more useful weak-review evidence. Unknown
  count remains ten, as intended by the unchanged safety threshold.
- The validation took 192.76 seconds, but 165.00 seconds was the opt-in
  diagnostic model-fit path triggered by `--analysis-dir`; cached
  identification took 9.45 seconds. This timing is not the normal demo path
  and is not evidence that production latency has been fixed.

Regression gate: 302 CV tests and 80 capture tests pass. Plan and red/green
steps are in
`docs/superpowers/plans/2026-07-18-pre-latency-reliability.md`.

## Phase 9 exact batch inventory review (2026-07-18) — IMPLEMENTED, HUMAN REVIEW PENDING

The CV scanner and the partner's model-generation inventory are now disjoint.
Twelve physically isolated batches containing 780 counted pieces were captured.
The scanner supplies draft detections and part/color suggestions; Emily's
confirmation is the source of truth for the fixed partner inventory.

Implementation:
- Review-draft mode keeps production scanning unchanged but performs only one
  best-crop Brickognize request per fused component, with 12 bounded workers,
  zero fixed inter-request delay, and existing retry/backoff behavior.
- Brickognize candidate render URLs are preserved. Review bundles cache their
  part renders and export up to three masked original-frame crops per component.
- A local FastAPI verifier shows a numbered segmentation overview, candidate
  renders, editable part ID/name, the physical 12-color selector, segmentation
  disposition, and an explicit include/exclude decision. Every mutation is
  atomically autosaved.
- Exact-count reconciliation distinguishes included physical pieces, excluded
  accidental physical pieces, zero-count duplicate detections, and manually
  added missed pieces. Merged/incomplete/unconfirmed rows block locking.
- Aggregate JSON/CSV export is refused until all twelve sessions are reconciled
  and locked. Only included human-confirmed rows enter the final inventory.
- Confirmed identity/color rows may become future exemplar truth. A mask is
  eligible for segmentation training only when explicitly marked correct.

Generated workspace:
`/Users/emily/lego-capture/inventory-review-20260718/workspace.json`.
The verifier runs at `http://127.0.0.1:8765/` while its local Uvicorn process is
active. All generated session outputs are run-scoped beneath
`sessions/<id>/analysis-runs/inventory-annotation-v1/`.

Draft count evidence (physical -> fused draft):
- `161438`: 48 -> 45 (-3)
- `161851`: 74 -> 91 (+17)
- `162215`: 83 -> 91 (+8)
- `162458`: 64 -> 64
- `162834`: 88 -> 89 (+1)
- `163214`: 85 -> 86 (+1)
- `163451`: 63 -> 64 (+1)
- `163723`: 67 -> 70 (+3)
- `164119`: 86 -> 88 (+2)
- `164730`: 57 -> 57
- `165118`: 39 -> 39
- `165402`: 26 -> 26

The drafts total 810 components versus 780 physical pieces. This is not an
inventory result: it is the workload presented for human reconciliation. The
large positive deltas in `161851` and `162215` likely contain duplicate tracks;
the negative delta in `161438` requires three manual missed-piece entries or a
targeted rescan. Exact draft counts can still conceal a duplicate paired with a
miss, so every row remains review-required.

Verification: all twelve workspace entries and every referenced crop/mask asset
validated beneath its run directory; JavaScript syntax passed; live-browser
checks passed for all-session loading, numbered-shape selection, next/previous
navigation, candidate renders, color controls, and session switching. Regression
gates are 322 CV tests and 83 capture tests passing. The sole CV warning is the
pre-existing FastAPI/Starlette `httpx` deprecation warning.

## Phase 10 BrickLink Studio draft export (2026-07-18) — GENERATED

The rejected browser verifier is no longer required for routine correction. A
new exporter writes the current draft as ordinary editable LDraw pieces, with
the model-selected part and color prefilled. BrickLink Studio can replace a part,
repaint it, delete accidental/duplicate pieces, and add missed pieces directly.

Generated outputs:
- `inventory-review-20260718/studio/lego-inventory-draft.mpd`
- `inventory-review-20260718/studio/lego-inventory-draft.csv`

The MPD contains one main model and twelve session-named submodels. It contains
all 810 fused draft components exactly once. Of these, 766 have a predicted,
confirmed, or best-candidate part; the 44 components with no part candidate use
a conspicuous magenta Brick 1 x 1 placeholder. Ambiguous color strings use the
model's first-listed color choice. Four edits already confirmed in the browser
draft are respected; all other pieces use current model output.

Focused verification: 7 exporter tests pass. Structural verification found 12
session submodels, 810 component provenance records, 810 session part records,
810 CSV rows, and 44 placeholders. Full capture regression gate: 90 tests pass.
Design and execution plan are in `docs/superpowers/specs/2026-07-18-studio-inventory-export-design.md`
and `docs/superpowers/plans/2026-07-18-studio-inventory-export.md`.

## Phase 11 standalone demo viewer (2026-07-19) — P1 COMPLETE, AWAITING APPROVAL

Demo pivot: the partner's generator now uses a pre-counted fixed inventory, so
live CV accuracy is no longer load-bearing. The CV pipeline becomes a
standalone demo — live scan with a camera-following glow, spatial coverage to
completion, fused apparent inventory, then a confirmation of the safest few
pieces. P1 (scene state + timing) is built; P2 (viewer) and P3 (showcase) are
scoped and pending approval.

### P1 audit — what already exists vs. what P1 added

Already present (reused, not rebuilt): per-observation source frame id
(`Observation.frame_id`), image-space bbox (`box_original`), instance masks
(`mask_original`/`polygons_canvas`), real per-detection YOLO confidence
(`Observation.confidence`, seg only), and stage timings (`timings_s`). The
manifest already carries table geometry (`origin_xy`, `px_per_m`, `size_wh`,
hard workspace contours) and spatial coverage (`selected_coverage`,
`union_coverage`, `coverage_complete`).

Genuinely missing, so P1 added it in `lego-cv/pipeline/scene_state.py`
(new `scene_state.json`, written by default; `--no-scene-state` opts out):
- **Stable instance ids.** Fused instances had no id — only a positional index.
  `instance_id` is now that deterministic sorted index, made explicit. Honest
  limit: stable across one scan's artifacts, re-derived if the session is
  re-scanned; there is no persistent cross-scan identity and none was invented.
- **Per-observation timestamps.** Not carried on `Observation`; joined from
  `frames.jsonl` by frame id (validated present, 44 s span on `233252`).
- **Table-coordinate footprint per instance** (`table_polygon_xy`,
  `table_box_xy`, `table_centroid_xy`) — canvas px → table metres — so a UI can
  project a piece into any frame.

Signals we deliberately did NOT invent: there is no learned "mask completeness"
or per-instance segmentation-quality score. `complete` is the real
box-inside-workspace-margin boolean; `confidence` is the real YOLO box score
and is exported as `null` for the classical `cv` detector rather than faked.

### P1 real timing — end of capture → fused inventory (Apple MPS, warm cache)

| stage | 233252 (13 pieces) | 162834 (89 pieces) |
|---|---:|---:|
| view selection (`multiview.py`) | 4.9 s | 8.6 s |
| YOLO detect (5 frames) | 5.0 s | 34.6 s |
| fuse | 0.00 s | 0.15 s |
| Brickognize identify | 3.9 s | 257.6 s |
| evidence + color | 0.8 s | 3.8 s |
| **scan total** | **9.8 s** | **296.1 s** |

The decisive findings: **fusion is effectively free**, but everything is a
**post-capture batch** — view selection needs the full coverage grid, so the 5
detected frames are not even known until capture ends. Brickognize dominates
and is network-rate-limited (~1 req/s), scaling with piece count (4 min for 89
pieces). Live per-frame segmentation is far too slow for an interactive
overlay (5–35 s for 5 frames).

### P1 — how the glow must be driven (recommendation)

Fusion cannot run incrementally during the scan, so the glow is **not** driven
by live detection. Recommended and validated mechanism: after the batch scan,
**replay the recorded frames in capture order and project each already-fused
piece's table polygon into the current frame via that frame's pose +
intrinsics**; a piece lights the first frame it becomes visible in and stays
lit. Because frames are in capture order along the camera's path, pieces light
in a spatial trail that follows the camera — the requested zigzag — while
reusing existing fusion identity instead of building a live tracker.

Viability proven, not asserted: instance 0's exported table polygon projects
cleanly into 4/4 arbitrary NON-selected frames (14/35/54/231/263 were the only
detected frames; tested 50/120/200/280), centroid tracking the camera
(405→301→615→540 px). This is a post-scan replay, not a true first-pass live
overlay — stated plainly so P2 commits to it knowingly.

### P1 gate status

`scene_state.json` (v1) validated on 2 real sessions (`233252` 13 inst,
`154929` 15 inst). Regression: **90 capture + 369 cv tests pass** (was 90 +
364; +5 new scene-state tests, one pre-existing Starlette warning). No existing
schema or behavior changed; artifact is purely additive. Historical session
artifacts untouched (validation written to run-scoped scratch dirs);
`IMG_4841.jpg` preserved. Committed on `codex/option-a-color-detection`; not
merged/pushed. **Stopping for approval before P2.**

### P2 — showcase selection + confirmation payload (2026-07-19) — COMPLETE, AWAITING APPROVAL

Order updated per approval: P2 = showcase now, P3 = true live glow next.

Surfaces up to N (default 5) of the SAFEST identified pieces for end-of-scan
confirmation (inverting the old surface-the-uncertain logic), in
`lego-cv/pipeline/showcase.py`, wired as `session_cli.py --showcase N`. Writes
`showcase.json` (+ `debug/showcase_gallery.jpg`); purely additive, computed
from the completed `result` after inventory is built, so it cannot change
inventory. Conservative funnel, real signals only:
- **Cheap local hard vetoes** (prefilter): unidentified; real warning flags
  `identity_view_disagreement` / `dimension_mismatch` /
  `possible_non_canonical_pose`; chosen-view crop flags `possible_shadow` /
  `possible_specular` / `oblique_view` / `boundary_view`; boundary-zone crop;
  incomplete crop; degenerate area. `family_reviewed` and
  `view_diversity_unknown` are explicitly NOT vetoes (they fire on ~every
  piece).
- **ID-quality floors**: Brickognize top1 >= 0.80; top1-vs-next-different-part
  margin >= 0.10; multi-view agreement >= 2 OR an exceptional single view
  (top1 >= 0.90, margin >= 0.15) — single view is not auto-rejected; real YOLO
  confidence >= 0.50 when present (null for cv detector is not treated as low).
- **Safety ranking** with margin weighted highest; keep top N; below-cut safe
  survivors and strongest rejects both recorded with reasons.
- **Colour**: confirmation question is part-identity only; predicted colour +
  a `color_reliable` flag (false for `ambiguous(...)`) ride in the payload for
  optional UI use, never the question.

Also updated the Brickognize identify path to **bounded concurrency** (10
workers, was 1 req/s serial) — result-preserving (same crops/cache/responses),
retry/backoff handles 429s.

Real-session results (eye-checked crops in scratch galleries):

| session | pieces | surfaced | scan time (warm) | identify (warm) |
|---|---:|---:|---:|---:|
| 233252 | 13 | 5/5 | 5.9 s | 1.0 s |
| 154929 | 15 | 3/5 | 3.4 s | 0.8 s |
| 162834 (pile) | 89 | 5/5 | 44.7 s | 6.3 s |

Bounded concurrency: 89-piece identify **257.6 s → 6.3 s**. The funnel is
brutally picky — it rejected several 0.90–0.92-confidence pieces purely on
crop-quality flags (oblique / specular / boundary / non-canonical pose), e.g.
233252 #12 (0.92, oblique) and #7 (0.91, boundary); 162834 #31 (0.92, side-
lying). Every surfaced piece's stud count and part type match the crop on eye
inspection. Two carry the known plate-vs-brick height ambiguity that a
top-down crop can't fully resolve (233252 #9 green 1x1; 162834 #86 red
"Plate 2x2" that could be a Brick 2x2) — flagged for Emily's physical check,
which is exactly what the confirmation step exists for. This is evidence on
these sessions, not a guarantee on fresh piles.

Regression: **90 capture + 389 cv tests pass** (+20 showcase; no inventory
behaviour or schema changed). Historical session artifacts preserved (validated
in-process / with backup-restore); `IMG_4841.jpg` untouched. Committed, not
merged/pushed. **Stopping for approval before P3 (true live glow).**

### P3 — true live scan glow, vertical slice (2026-07-19) — COMPLETE, AWAITING APPROVAL

Live segmentation-mask glow that follows the camera, NOT static boxes. New
`lego-capture/live_glow.py` (pure engine, tested) + `live_scan_viewer.py`
(runner). Data flow, identical for live phone and recorded replay:

    frame + pose -> async sampled YOLO -> back-project mask to table plane
                 -> GlowTracker (one table-space track per piece)
    every UI tick -> project active tracks into current frame -> glow

Reuses the P1 geometry exactly: for on-plane points, image<->table is the
`topdown.plane_homography` (unit scale), so a detected mask is back-projected
to the table and re-projected into any later frame — the mask stays attached
to the physical piece as the camera moves (proven to sub-pixel in tests).
`GlowTracker` associates async detections to persistent tracks, activates each
**once**, and staggers new activations along the camera's travel direction (the
scan pulse / zigzag). `glow_envelope` animates a bright discovery pulse that
settles to a subtler persistent highlight, sampled every UI frame independent
of the slower detection cadence. Single consistent scan hue (no per-piece
colour — that is the confirmation step's job). Rendering never blocks on YOLO:
the on-screen path runs detection in a worker on the latest frame only (stale
dropped); the headless preview samples detection inline deterministically.

Demonstrated on a real recorded scan (`20260717-233252`, 326 frames, actual
`lego_seg.pt`): headless run wrote a 326-frame mp4 preview. Sample frames
confirm the target visual — the full phone camera feed with translucent cyan
masks traced on each physical piece's actual segmented shape (orange 1x8 beam,
blue/white/teal bricks, the green 1x1, the yellow 2x2), a discovery counter,
progressive activation (e.g. the centre white 1x2 not yet lit while 12 others
are), and masks that stay attached to pieces as the camera pans between frames.
The slice already extends past the minimal single-piece target to multi-piece
progressive activation. Runs from the lego-cv venv (YOLO/ultralytics) with
lego-capture on PYTHONPATH; `--live` drives the Record3D phone with a prior
session's `table_frame.json` for anchoring.

Honest limits of this slice: table anchoring reuses a recorded/prior plane fit
(live-only incremental plane fitting is not built — the phone path expects a
`--world-to-table` session); a track drops after a 4 s no-re-detection TTL, so
a piece that leaves the frame stops glowing until re-seen (it cannot glow while
off-screen); this was demonstrated headless on recorded frames with real poses
(faithful to the live data flow) — an on-device live run needs the phone.

Regression: **101 capture + 389 cv tests pass** (+11 live-glow/viewer; cv
untouched this phase). No P1/P2 behaviour or schemas changed. Committed on
`codex/option-a-color-detection`; not merged/pushed. **Stopping at the P3 gate.**

## Log

- 2026-07-18 (source-frame inventory review): replaced unsafe cross-frame
  review associations with a new run-scoped waypoint workspace at
  `inventory-waypoint-validation-20260718/workspace.json`. Every review item
  now has exactly one untouched source frame and one view; secondary frames
  are used only through the explicit Add missed piece picker. Batch
  `20260718-161438` uses original frame 0 with 45 primary waypoints; 44 of 49
  completed confirmations migrated by exact reviewed-shape ID, with five old
  confirmations absent from the primary frame. Batch `20260718-161851` shapes
  `00002:086` and `00200:016` are separate frame-2/frame-200 records and no
  longer share a component. The previous `inventory-validation-v2` reviews
  retained aggregate SHA-256
  `c4a5332eb045e85e5ba4a0ca7411d4c7a073d550840db18086c28e9980065c94`.
  The live verifier at `http://127.0.0.1:8766/` now serves the waypoint
  workspace. Verification: 351 CV tests passed; JavaScript syntax, live Batch
  1 full-frame rendering, the secondary-frame picker, and Batch 2 source
  isolation passed.

- 2026-07-17: Phase 1 plan written
  (`lego-cv/docs/superpowers/plans/2026-07-17-phase1-original-crop-bridge.md`).
  Key finding: multiview manifest v1 carries no canvas->original geometry;
  Bridge = manifest v2 (homography + rgb path + raise-direction) + lego-cv
  original-crop module.
- 2026-07-17 (later): Phase 1 built in 3 steps — (a) original AABB crops,
  (b) ID-view selection by obliquity + original-frame fit, (c) quad masking
  of neighbor pieces. Each step validated against the 4 real sessions via
  the new crop galleries. Stopped at GATE (see above).
- 2026-07-17 (Phase 2): evidence-first — reconstructed the 4 case crops,
  read cached Brickognize candidates, tuned Hough on real crops before
  writing code. Key evidence: correct 1x8 existed in 234326's alternate
  views (f90 0.84); counts ~2x noisy on beams => 3x contradiction band.
- 2026-07-17 (Phase 2 GATE): APPROVED by Emily. No further tuning against
  212422/005145 residuals (view-count / merged-crop limits — Phase 3).
- 2026-07-17 (Phase 3): plan written
  (lego-cv/docs/superpowers/plans/2026-07-17-phase3-trained-separator.md),
  superseded by the approved execution handoff and expanded session-safe plan.
- 2026-07-17 (Phase 3 execution): 71 frames human-reviewed, 1,157 masks
  imported, local MPS training completed, original-frame segmentation/fusion
  integrated, target and whole-session validation regressions passed, and the
  held-out white-table shortfall recorded without tuning on it.
