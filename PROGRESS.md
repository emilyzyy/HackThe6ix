# LEGO Scanner — Progress

Covers both repos (`lego-capture`, `lego-cv`), branch
`codex/option-a-color-detection` in each. Primary failing acceptance case:
`sessions/20260717-005145` (8 pieces; previously 7 because two white pieces
merged) is now fixed at 8 separate fused instances.

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

## Log

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
