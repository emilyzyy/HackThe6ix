# Studio Inventory Export Design

**Date:** 2026-07-18

## Goal

Replace the rejected browser-based verifier with one BrickLink Studio-compatible
MPD containing the draft inventory from all twelve scan sessions. The CV model's
part and color are prefilled; Emily corrects parts/colors, deletes unwanted or
duplicate pieces, and adds missed pieces directly in Studio.

## Format

LDraw line type 1 natively records both the color code and referenced part file.
The export therefore uses normal editable Studio pieces, not rendered images.
One MPD contains a main model and one named submodel per session. Pieces are
spread on a wide grid so they do not connect or overlap.

For each component, a confirmed review value wins. Otherwise the production
prediction is used. If production part identity is unknown but candidates exist,
the highest-ranked candidate is used. If no candidate exists, the exporter uses
a magenta 1x1 brick placeholder and marks it in provenance so it cannot look like
a confident inventory prediction. Ambiguous colors use the first listed color,
which is the model's leading color choice.

## Provenance and output

Comments before each LDraw part record the session, component, predicted status,
part ID, and color. A sidecar CSV records the same mapping and grid coordinates;
the Studio workflow does not require viewing the CSV. The generated files live
under `inventory-review-20260718/studio/` and do not modify historical session
artifacts.

## Acceptance criteria

- Studio can open the MPD as editable parts split into twelve session submodels.
- Each non-placeholder component displays the model-selected part and color.
- Confirmed browser edits, if any, are respected.
- Unknown components with candidates use the best candidate; components with no
  candidate are conspicuous magenta placeholders.
- The export contains all 810 draft components exactly once.
- Focused exporter tests and the capture repository test suite pass.
