# Fast Showcase Demo Stitch Design

**Date:** 2026-07-19  
**Status:** Approved direction; implementation pending  
**Repositories:** `/Users/emily/lego-capture`, `/Users/emily/lego-cv`

## Goal

Turn the accepted live yellow-mask scan and compact confirmation UI into one
coherent demo flow:

`scan and record -> 100% -> animated analysis -> 3-5 safe confirmations -> inventory handoff`

The critical path optimizes for a few highly trustworthy showcase identities,
not exhaustive identification of the approximately 800-piece pile. The live
glow remains cosmetic. Only post-capture detections may be shown as explicit
part predictions.

## Non-goals

- Do not identify every visible piece before confirmation.
- Do not use animation-only screen fallback masks as authoritative evidence.
- Do not change the teammate's model-generation implementation.
- Do not claim that scan coverage is the fraction of LEGO pieces identified.
- Do not surface a weak prediction merely to reach five confirmations.

## User experience

### 1. Scanning

A single demo process owns the Record3D connection. Every retained RGB, depth,
pose, and intrinsics snapshot is written through the existing
`CaptureController` while the same in-memory frame feeds the accepted live glow
renderer.

The HUD uses larger coverage text. Its display animation rises normally to 95%,
then eases from 95% to 100% over four seconds. Reaching 100% automatically ends
the recording and flushes the saved session. The coverage value is presentation
progress; the LiDAR coverage tracker still supplies legitimate workspace
coverage and the UI must continue to call it workspace coverage.

### 2. Analyzing

The camera feed transitions directly into a pulsing **Analyzing your LEGO...**
screen. It does not freeze one best frame.

Three real frames from the just-finished scan animate into a gently moving,
slightly rotated photo stack. Status copy follows real work:

1. `Selecting the clearest views...`
2. `Finding clean LEGO pieces...`
3. `Matching LEGO types...`
4. `Preparing your confirmations...`

A centered LEGO-yellow analysis indicator remains visible above the stack: a
soft glowing core with two expanding pulse rings and three orbiting dots. The
current real stage label sits directly beneath it. The motion is continuous and
time-based, not tied to a fabricated numeric processing percentage.

As authoritative segmentation results become available, yellow translucent
piece silhouettes may stamp onto the current photo. This is decorative feedback
driven by real results; the animation continues independently so slow inference
never freezes the screen.

### 3. Confirmation

When at least one safe result exists and processing is complete, the analysis
window yields to the existing full-width browser review. Only the compact yellow
confirmation card is visible. It points to the actual segmentation mask, not a
box. `Yes` advances. `No` replaces the card contents with the existing type-only
correction field.

The target is three to five confirmations. Fewer is acceptable when the safety
gate abstains.

### 4. Handoff

After the last confirmation, the UI writes a small handoff artifact containing
the scan session, confirmed showcase pieces, and the configured fixed inventory
reference. It then displays an inventory-ready transition. A configurable
generator URL may be opened afterward; no generator code or inventory contents
are modified by this work.

## Architecture

### Demo coordinator

A thin coordinator owns the state machine:

`SCANNING -> COMPLETING -> ANALYZING -> CONFIRMING -> HANDOFF`

It launches one Record3D client, starts and stops worker threads, flushes the
session before offline processing, reports real stage events to the animation,
starts the local confirmation service, and opens the generated review URL.

The coordinator does not contain detector, fusion, identification, or browser
business logic. Those remain in their existing modules.

### Combined capture and glow

The existing capture snapshot boundary remains authoritative: buffers and pose
are copied once inside the Record3D callback. The snapshot is then offered to:

- the bounded session writer at approximately 6 retained frames per second;
- the latest-frame live glow renderer and asynchronous animation detector;
- the live LiDAR workspace coverage tracker.

This avoids the current impossible configuration where `capture.py` and
`live_scan_viewer.py --live` compete for Record3D's single-client stream.

The saved session receives its own post-capture table-plane computation. The
old anchor may be used for live animation only; it is not silently copied as the
new scan's authoritative calibration.

### Early-exit showcase processor

The processor reuses the existing multiview export and YOLO segmentation code,
but it does not call the current monolithic path that identifies every fused
instance.

1. Select at most three sharp, well-covered, mostly top-down frames in priority
   order.
2. Segment the best frame first inside the projected LiDAR workspace.
3. Locally shortlist clean candidates using real evidence: segmentation
   confidence, complete/non-boundary mask, workspace overlap, useful crop size,
   sharpness, low neighbor overlap, plausible LEGO physical size, and spatial
   separation.
4. If the first view is insufficient, run the full three-view fallback. This
   avoids the slower redundant `one -> two -> three` replay pattern.
5. Deduplicate repeated observations in table coordinates.
6. Identify only a bounded shortlist, initially at most twelve crops, with the
   existing concurrent Brickognize client.
7. Apply the existing showcase safety concepts: accepted identity, strict top-1
   confidence, margin over a different candidate, no disagreement/dimension/
   pose warning, and clean crop evidence.
8. Stop when five safe results exist or the three-frame budget is exhausted.

Single-view fast-showcase results must clear a top-1 score of 0.85, retain the
existing 0.15 alternative margin, and pass every existing hard veto. Additional
views can add agreement, but are not mandatory when the first frame already
provides enough clean candidates.

### Showcase-to-confirmation adapter

The adapter converts selected results from the same scan into the existing
review document shape. Each component includes:

- source frame path and dimensions;
- exact source-frame mask polygon or raster mask;
- source-frame box for crop loading only;
- predicted part ID/name and score;
- reliable color only when the current color evidence marks it reliable;
- confirmation state and optional corrected type.

The adapter creates a disposable, run-scoped workspace. It never rewrites the
recorded session, historical review workspaces, or manually validated outputs.

## Timing budget

Historical five-view runs spent 7.7-36.9 seconds in segmentation and 2.6-8.8
seconds identifying 26-91 instances. The design removes all-instance
identification and normally avoids four of five segmentation views.

The initial target on the current Mac is:

- session finalization and view selection: 1-4 seconds;
- one authoritative view or the three-view fallback: 4-30 seconds;
- identification of at most twelve crops: 1-4 seconds;
- adapter and UI handoff: under 2 seconds;
- typical total analyzing time: 10-20 seconds.

These are acceptance targets, not invented progress percentages. Real timing is
recorded per stage and the photo-stack animation continues if a stage overruns.

## Failure behavior

- If the first frame yields too few candidates, process the next frame.
- If fewer than three identities pass after the view budget, show the safe
  subset rather than weakening thresholds.
- If Brickognize is unavailable, use only valid cached responses; otherwise
  explain that no identities were safe enough and preserve the session.
- If processing fails, keep the session and display a retry action with the
  failing stage. Never return to Record3D or rescan automatically.
- Cancellation closes workers and the local service without deleting artifacts.

## Outputs

Each demo run keeps its artifacts under the new session or a run-scoped analysis
directory:

- synchronized captured frames and `frames.jsonl`;
- table calibration and multiview selection metadata;
- fast-showcase timing/event log;
- authoritative candidate masks and rejection reasons;
- `showcase.json` for the selected safe pieces;
- disposable confirmation `workspace.json` and `review.json`;
- confirmation answers and a final handoff JSON.

## Testing and acceptance

### Automated

- Coverage display holds at 95%, eases monotonically to exactly 100% in four
  seconds, and uses the larger HUD font.
- One frame snapshot can be written and rendered without a second Record3D
  client.
- State transitions are ordered and cannot enter confirmation before the
  session is flushed and the showcase payload exists.
- The fast processor stops after enough safe candidates and never identifies
  more than the configured shortlist limit.
- Unsafe, boundary, oversized, overlapping, and low-margin candidates abstain.
- The adapter preserves source-frame geometry and mask alignment.
- Confirmation writes Yes/No/correction results and produces the handoff
  artifact.

### Recorded-session integration

Run the coordinator's post-capture half on an existing crowded session. Verify
the analyzing animation remains responsive, stage timings are emitted, three to
five safe candidates appear when available, mask overlays align, and historical
session artifacts remain unchanged.

### Final phone acceptance

Perform one full demo sweep over the approximately 800-piece layout. Pass when:

- live glow, scanner bar, and larger coverage text remain responsive;
- 95% rises smoothly to 100% and automatically enters analysis;
- the photo stack stays animated throughout processing;
- confirmation opens within the target timing or reports the measured overrun;
- every surfaced mask points to the intended physical piece;
- the confirmation flow ends at the inventory-ready handoff.
