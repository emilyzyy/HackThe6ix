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

## Log

- 2026-07-17: Phase 1 plan written
  (`lego-cv/docs/superpowers/plans/2026-07-17-phase1-original-crop-bridge.md`).
  Key finding: multiview manifest v1 carries no canvas->original geometry;
  Bridge = manifest v2 (homography + rgb path + raise-direction) + lego-cv
  original-crop module. Building now; will stop at GATE with before/after.
