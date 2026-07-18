# Batch Inventory Annotation Design

**Date:** 2026-07-18

## Goal

Turn twelve already-recorded Record3D sessions into one human-verified inventory
with exact quantities by part and color. The computer vision output is a draft;
Emily's confirmation is the source of truth.

This tool is intentionally similar to the earlier detector-dataset review. It is
not an analytics dashboard. A reviewer clicks a numbered detected shape, confirms
or corrects its rendered part and color, and either includes or excludes the
physical piece from the final inventory.

## Input batches

| Session | Physical pieces |
| --- | ---: |
| `20260718-161438` | 48 |
| `20260718-161851` | 74 |
| `20260718-162215` | 83 |
| `20260718-162458` | 64 |
| `20260718-162834` | 88 |
| `20260718-163214` | 85 |
| `20260718-163451` | 63 |
| `20260718-163723` | 67 |
| `20260718-164119` | 86 |
| `20260718-164730` | 57 |
| `20260718-165118` | 39 |
| `20260718-165402` | 26 |
| **Total physically counted** | **780** |

Each physical batch remains isolated in its session-labelled container until
that session is reconciled and locked.

## Scope

The first version will:

1. Generate run-scoped draft detections and fused components without replacing
   historical session artifacts.
2. Show a session image with numbered segmentation outlines and allow selecting
   one component at a time.
3. Show the selected masked crop, alternate crop views when available, the
   predicted part, top candidate part renders, and the predicted color.
4. Let the reviewer confirm or correct the part and color.
5. Let the reviewer exclude an accidental physical LEGO piece from the final
   inventory while retaining it in the audit record.
6. Record segmentation disposition as `correct`, `duplicate`, `merged`,
   `incomplete`, or `missed` when applicable.
7. Reconcile every session against its supplied physical count.
8. Export one locked, aggregated inventory containing only included and confirmed
   pieces.

This version will not train DINO, retrain YOLO, alter production identity rules,
or feed the partner's model directly.

## Draft generation

Draft artifacts are written beneath:

```text
sessions/<session-id>/analysis-runs/inventory-annotation-v1/
```

Normal session outputs and source frames remain untouched. Draft generation uses
the existing detector, tracking/fusion, crop-quality selection, color system, and
Brickognize cache. Because every answer receives human review, the draft path may
request only the best one or two crops rather than running expensive diagnostic
model fitting or exhaustive alternate-view identification.

Brickognize responses already contain `img_url` renders for their candidate part
IDs. The generator preserves the top candidate records and downloads their
renders into the run directory so the verifier remains usable after generation
without depending on live image URLs. A prediction below the normal production
acceptance threshold still appears as a candidate for human selection; it is not
silently promoted into production output.

## User interface

The verifier is a small local web application served by the existing Python
environment.

### Main view

- Left: best session view with numbered segmentation outlines.
- Right: the currently selected component.
- Header: session ID, expected physical count, detected component count, and
  review progress.

### Selected component controls

- Masked crop and available alternate views.
- Predicted part ID, name, score, and reference render.
- Top Brickognize candidates as selectable render cards.
- A searchable/manual part-ID field for a correct part absent from the candidate
  list.
- One color selector restricted to: red, orange, yellow, beige, brown, green,
  dark green, blue, white, light gray, dark gray, and black.
- `Include in final inventory`, enabled by default.
- `Exclude accidental physical piece`, which records an exclusion reason and
  removes that piece only from the final aggregate.
- Optional segmentation disposition.
- `Confirm + next`, with keyboard shortcuts for confirmation and navigation.

The interface autosaves after every action and resumes at the first unresolved
component.

## Exact-count reconciliation

The physical count and final inventory count are deliberately different totals:

- An **included** confirmed component represents one physical piece and one final
  inventory piece.
- An **excluded** confirmed component represents one physical piece but zero final
  inventory pieces.
- A **duplicate detection** represents zero physical pieces and zero final pieces.
- A **missed piece** must be added manually with part and color, or the batch must
  be rescanned while its physical container remains isolated.
- A **merged detection** cannot be locked until it is corrected into separate
  physical entries or replaced by manually entered pieces.

A session can be locked only when its included pieces, excluded physical pieces,
and manually added missed pieces account for the supplied physical count after
duplicate detections are removed. Every inventory-bearing entry must have a
confirmed part and color.

This allows Emily to remove unwanted blocks without making the detector appear to
have missed part of the original physical batch.

## Stored annotations

Each session receives a versioned annotation JSON containing:

- session ID and expected physical count;
- component ID and stable source anchors;
- crop, mask, overlay, and alternate-view paths;
- original predictions and all displayed candidates;
- confirmed part ID, part name, and color;
- include/exclude status and optional reason;
- segmentation disposition;
- reviewer and timestamps;
- manual additions for missed physical pieces;
- lock status and reconciliation totals.

Original predictions remain immutable inside the provenance record. Corrections
are stored separately so future model evaluation can compare prediction against
human truth.

## Outputs

After all twelve sessions are locked, the tool writes:

1. `verified_inventory.json`: aggregate quantities by confirmed part and color.
2. `verified_inventory.csv`: the same aggregate in a shareable table.
3. `verified_pieces.csv`: one audit row per physical piece, including excluded
   pieces.
4. `verification_summary.json`: expected, detected, duplicate, missed, excluded,
   included, and unresolved totals per session.

Only `included` confirmed rows contribute to `verified_inventory.*`.

## Future training-data use

Confirmed part and color labels can be used as classification/exemplar truth.
Segmentation masks become training truth only when the reviewer explicitly marks
the segmentation `correct` or corrects it in the existing LabelMe workflow.
Merely confirming a part ID does not validate its mask.

Dataset exports must split by whole session so crops from the same capture do not
leak across training and evaluation sets.

## Error handling and safety

- Source sessions and historical outputs are read-only.
- All generation and annotation writes are run-scoped and atomic.
- Autosave writes to a temporary file and replaces the annotation file only after
  valid JSON is complete.
- The aggregate export refuses unresolved, unreconciled, or unlocked sessions.
- Missing candidate renders do not block labeling; the crop, ID, and name remain
  available.
- A manual part entry requires both an ID and a human-readable name.

## Acceptance criteria

- All twelve supplied sessions appear with the correct expected counts.
- A reviewer can select every detected shape and confirm or correct part and color.
- An accidental physical piece can be excluded without breaking physical-count
  reconciliation.
- Duplicate, missed, merged, and incomplete detections remain explicitly visible.
- Closing and reopening the tool loses no confirmed work.
- No aggregate export is possible until all 780 physical pieces are accounted for.
- The final inventory contains only included, human-confirmed rows and exact
  quantities equal to the sum of those rows.
- Existing capture and CV test suites continue to pass.
