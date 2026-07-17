# LEGO Scanner — Progress

Covers both repos (`lego-capture`, `lego-cv`), branch
`codex/option-a-color-detection` in each. Primary failing acceptance case:
`sessions/20260717-005145` (8 pieces; currently 7 — two white plates merge).

Note: the mission brief said to read CLAUDE.md first — no CLAUDE.md exists in
either repo or $HOME as of 2026-07-17; state was read from git history,
docs/superpowers specs+plans, and session artifacts instead.

## Mission status

| Phase | Scope | Status |
|---|---|---|
| 1 — Bridge (identify from original crops) | manifest v2 + original-crop identification + debug overhaul + pad fix | IN PROGRESS |
| 2 — Stud-count advisory tiebreaker | HoughCircles sanity check on least-oblique crop | not started |
| 3 — Trained separator (yolo11n-seg) | dataset pipeline + Colab notebook + integration | not started; needs Emily (labeling, Colab) |

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

## Phase 1 GATE results (2026-07-17) — AWAITING EMILY'S APPROVAL

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
