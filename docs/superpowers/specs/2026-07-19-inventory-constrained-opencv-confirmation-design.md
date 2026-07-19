# Inventory-Constrained OpenCV Confirmation Design

## Goal

Unify the hackathon demo into one visual flow—live OpenCV scan, OpenCV analysis, and OpenCV confirmation—without changing the proven phone capture, LiDAR filtering, segmentation, multiview processing, or Claude's recent reliability fixes. Keep the current browser confirmation workflow available as an explicit backup.

## Authoritative Inventory

The confirmation catalog is:

`/Users/emily/lego-capture/outputs/019f72d0-9cbe-7431-8f4b-7ed4084bbd13/lego_inventory_for_model_generation_20260719.csv`

It contains 787 pieces across 32 allowed piece types. Its `piece_type` values are the only identities that may be presented or manually submitted during showcase confirmation. Matching is case-insensitive and normalizes repeated whitespace around dimensions, while display text retains the CSV spelling.

The inventory constraint applies only to the 3–4-piece showcase confirmation path. It does not modify detector training, segmentation, LiDAR filtering, multiview fusion, or the complete inventory artifacts.

## Confirmation Selection

The stitched demo requests four confirmations and requires three when at least three inventory-valid identified detections exist.

For each detected instance, showcase construction examines the final identity and the raw ranked recognition candidates. It selects the highest-scoring candidate whose normalized part name exists in the inventory CSV. Scores and ambiguity margins are recalculated within the allowed candidate set. A disallowed result such as `Slope 45 2 x 2` is never surfaced; if no allowed alternative exists, that instance is excluded from confirmation.

The existing strict safety funnel remains first priority. If fewer than three strict candidates survive, the existing real-mask best-effort fallback may fill the list using only inventory-valid identified candidates. The final list contains at most four pieces. If fewer than three valid candidates exist, the UI shows only the available valid pieces and never invents or substitutes an unavailable type.

## OpenCV Confirmation UI

A new native OpenCV reviewer consumes the same disposable showcase workspace currently consumed by the web application. It does not rerun segmentation or recognition.

For each component it renders:

- the original source camera frame;
- the exact yellow mask crop at its original-frame location;
- a yellow waypoint dot and line;
- the compact yellow confirmation card;
- a pre-rendered transparent PNG version of the existing yellow waving mascot;
- compact clickable `Yes` and `No` buttons.

The interface supports mouse and keyboard input:

- `Yes` button or `Y`: accept the inventory-valid prediction;
- `No` button or `N`: open correction mode;
- typing: filter inventory names from the 32-type allowlist;
- Up/Down: move through suggestions;
- Tab: autocomplete the highlighted suggestion;
- Enter or the `Save` button: submit the highlighted exact inventory type;
- Backspace: edit;
- Escape or the `Back` button: return without changing the piece.

Free text that does not resolve to an exact allowed inventory type cannot be saved. A corrected name may have a blank part ID because the authoritative CSV contains names rather than part IDs; the existing showcase review schema already permits part-name-only confirmation.

The reviewer uses a small explicit state machine (`review`, `correction`, `complete`, `error`) and a single OpenCV mouse callback. Rendering is deterministic from state, which keeps interaction logic testable without opening a real window.

## Persistence and Handoff

Confirmation updates use the existing review-document functions rather than duplicating validation rules. The browser endpoint's handoff construction is extracted into a pure shared function so both the web backup and OpenCV path produce the same `handoff.json` schema, confirmed-piece payload, fixed-inventory reference, and optional generator URL.

After the last confirmation, OpenCV writes the same handoff artifact and shows `Inventory ready`. If a generator URL is configured, the handoff records it; the confirmation UI itself never opens a browser.

## Demo Integration and Backup

`demo_flow.py` gains:

- `--inventory-csv PATH`, defaulting to the authoritative 787-piece CSV;
- `--confirmation-ui {opencv,web}`, defaulting to `opencv`;
- a showcase target of four and minimum-safe count of three.

The OpenCV path invokes the reviewer with the generated workspace and remains in the terminal until confirmation completes. The existing FastAPI/browser startup code stays intact and is selected with `--confirmation-ui web`. Claude's stale-port cleanup remains unchanged for that backup path.

The normal path becomes:

`phone scan → animated analysis → inventory-constrained 3–4-piece showcase → OpenCV confirmation → handoff.json`

## Failure Handling

- Missing or malformed inventory CSV: fail before scanning with a concise terminal error listing the expected path and columns.
- No inventory-valid detections: show an OpenCV error card, write no fabricated handoff, and exit cleanly.
- Missing frame or mask asset: identify the component and missing path, close all OpenCV windows in `finally`, and preserve the workspace for replay.
- User closes the window or presses `Q`: save completed confirmations, close cleanly, and leave the workspace resumable.
- Web backup mode remains available if the native reviewer fails during rehearsal.

## Testing

Focused tests cover:

1. CSV parsing, normalization, quantity aggregation, and rejection of malformed files.
2. Re-ranking a disallowed slope to the best allowed raw candidate.
3. Excluding detections with no inventory-valid identity.
4. Selecting three to four confirmations and never five.
5. Inventory-constrained strict and best-effort fallback behavior.
6. OpenCV review/correction state transitions, keyboard editing, suggestion navigation, and mouse hit targets.
7. Rejection of invalid correction text and acceptance of an exact allowlisted type.
8. Exact-mask overlay and card rendering on synthetic images.
9. Handoff equivalence between browser and OpenCV paths.
10. Replay integration that reaches the native reviewer without starting a web server.

The complete capture and CV test suites must pass before the native path becomes the default.
