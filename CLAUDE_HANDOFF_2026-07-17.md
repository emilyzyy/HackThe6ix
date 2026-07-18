# LEGO Scanner CV — full context handoff back to Claude Code

**Prepared:** 2026-07-17

**Purpose:** Continue the project without reconstructing the history from chat.

**Start Claude Code in:** /Users/emily/lego-capture, but read both repositories named below.

## 1. Executive status

The CV side of the hackathon project is now a working original-frame, multi-view LEGO inventory pipeline:

1. Record3D captures high-resolution RGB, low-resolution LiDAR depth, camera intrinsics, and pose.
2. Depth is used only to estimate the table and a physical workspace. It is not detailed enough to separate LEGO pieces.
3. A locally trained **YOLO11n instance-segmentation model** finds separate LEGO instances in selected original RGB frames. It does not run on the stitched/rectified mosaic.
4. Predictions are gated by the physical workspace and fused across views in table coordinates.
5. Brickognize identifies each fused piece from original, unwarped, mask-isolated crops.
6. Custom mask-aware color analysis assigns a LEGO color.
7. The result is inventory JSON for Emily's teammate's text-to-build/generation half.

The current model is single-class **lego_piece**. YOLO separates instances; it does not determine the part ID or color. Brickognize still performs part identification. The custom color pipeline still performs color recognition. Stud counting remains advisory/second-chance evidence only.

The target regression session, **20260717-005145**, now returns all eight physical pieces. The touching white pieces remain separate, the large gray Plate 6 x 12 is found, and the orange item is identified as Brick 1 x 8 with score 0.895.

The newer reliability regression, **20260717-233252**, now returns all 13
physical components from five selected views. Ten of 13 part identities are
correct when color is ignored; the three residual identity failures retain
per-view evidence rather than being silently forced.

Latest verified test totals before this handoff:

- lego-capture: **80 passed**
- lego-cv: **252 passed**
- One non-failing Starlette deprecation warning may appear in the CV suite.

## 2. Repositories and current state

There are two separate Git repositories and two separate virtual environments:

| Repository | Responsibility | Current branch |
|---|---|---|
| /Users/emily/lego-capture | Record3D capture, table plane, coverage, workspace geometry, top-down/multiview export | codex/option-a-color-detection |
| /Users/emily/lego-cv | segmentation, geometry gate, multiview fusion, Brickognize, color, inventory, training/evaluation | codex/option-a-color-detection |

At the time this file was prepared, both repositories were clean before adding this handoff. Do not merge, push, delete branches, or create another branch unless Emily explicitly asks.

Important local artifacts:

- Installed model: /Users/emily/lego-cv/models/lego_seg.pt
- Tracked model metadata: /Users/emily/lego-cv/models/lego_seg.metadata.json
- Training runs: /Users/emily/lego-cv/training/runs/
- YOLO dataset: /Users/emily/lego-cv/training/data/yolo/lego-seg-a751e0b2b456/
- Product evaluation: /Users/emily/lego-cv/training/runs/product-evaluation/product-evaluation.md
- Capture guidance: /Users/emily/lego-cv/docs/phase3-capture-quality.md
- Ongoing project log: /Users/emily/lego-capture/PROGRESS.md
- Phase 3 implementation plan: /Users/emily/lego-cv/docs/superpowers/plans/2026-07-17-phase3-yolo11-seg-execution.md
- Preserve this unrelated user file: /Users/emily/lego-cv/IMG_4841.jpg

The model checkpoint and training runs are intentionally Git-ignored. The metadata and code are tracked.

## 3. Product and team context

This is a two-person Hack the 6ix project aimed at children:

- The user scans a batch of LEGO pieces with a LiDAR iPhone Pro.
- Emily owns the entire scan-to-inventory CV pipeline.
- Her teammate owns the prompt-to-build half, which consumes inventory JSON and generates a buildable model/instructions, likely with Gemini and three.js.

Emily wants:

- accurate inventory above all else;
- plain explanations of technical tradeoffs;
- exact copy/paste terminal commands;
- explicit labels for physical steps only she can perform;
- small phases with test gates;
- no confident claims based only on a pretty debug image;
- future support for larger batches that do not fit in one fixed camera view and for multiple environments.

Do not re-propose these rejected ideas:

- replacing Brickognize with a custom part classifier;
- using LDraw to identify an unknown part;
- full SLAM or full 3D reconstruction for the hackathon;
- Scaniverse/RebrickNet/Brickit as an open integration;
- MongoDB as image storage;
- treating coarse Record3D depth as piece-level geometry.

## 4. Exact chronology since the first Claude-to-Codex transfer

### 4.1 What Codex inherited when Claude first ran out

The initial pipeline already captured Record3D sessions, fitted the table plane, produced a stitched top-down, ran a classical OpenCV contour detector, sent crops to Brickognize, estimated color, and wrote inventory JSON.

The concrete real scan was **20260716-212422**. It demonstrated that the end-to-end path worked, but exposed two immediate bugs:

1. Sparse background observations enlarged the coverage hull, so completion stayed near 95%.
2. The top-down included desk edge, laptop/floor/background, and the detector boxed that material as unknown bricks.

Claude had started the workspace-bounding task but ran out of tokens before completing/validating it. Codex took over at that point.

### 4.2 Codex phase A — workspace and conservative background filtering

Capture commit:

- 3969542 — fix: bound coverage and topdown to dense workspace

CV commit:

- 5ca5c2e — fix: filter implausible background detections

The capture fix replaced the full observed convex hull as the practical workspace with a bounded dense-observation region. Coverage completion and the top-down crop use that bounded workspace. Enclosed unswept holes still count as incomplete, while sparse far-away observations no longer keep the scan permanently at about 95%.

The CV fix added conservative post-detection size/position filtering for implausibly huge or isolated boxes. This removed obvious desk-edge/laptop detections without making depth a piece detector.

This solved the original background-hull problem but did not solve the fundamental instance-separation problem: classical contours still merged nearby/touching pieces and still reacted to mosaic artifacts.

### 4.3 Codex phase B — “Option A,” classical color-aware separation

Design/implementation commits:

- 3b57904 — docs: design color-aware instance detection
- 6a58f25 — docs: plan color-aware detection option
- fd79a72 — feat: split nearby pieces by color

Why this was tried:

- The UI was producing “cluster: separate pieces and re-scan” even when pieces were merely close.
- Different-color adjacent pieces contained useful separation evidence.
- Emily correctly objected that physically re-separating already distinct pieces was unrealistic.
- A gray piece was missed in a test, while seam/overlay fragments became unknown bricks.

Option A added color-aware classical splitting/watershed behavior. It helped some differently colored contacts and reduced some cluster warnings.

Subsequent limitation:

- It could not reliably split same-color contacts.
- Lighting, shadows, white pieces, gray pieces, and specular highlights made color boundaries inconsistent.
- It was still operating on rectified/stitched imagery, so it inherited geometry distortion and seam artifacts.
- Therefore this was retained as a fallback, not accepted as the primary accuracy path.

### 4.4 The top-down representation experiments and why they were abandoned for detection

There were three successive attempts:

#### Attempt 1 — stitched top-down

Problem:

- Repeated frames were projected/overlaid.
- Slight calibration, pose, and table-plane error created ghost layers.
- Stitch boundaries and overlapping ends created small rectangular/contour artifacts.
- The detector treated those artifacts as unknown bricks.

The user suggested cropping overlap ends and blending/gluing border pixels. That idea was researched and implemented in the form of coverage-aware rectified views and invalid-border handling:

- 96ef438 — feat: export coverage-aware rectified views
- 420430f — fix: fill invalid multiview borders

Feathering/blending can hide a visible seam, but it cannot make two misregistered views of a raised 3D object agree. LEGO pieces are above the table plane, so a homography calculated for the table warps and shifts them differently between views.

#### Attempt 2 — one clean “best frame”

Capture commit:

- aee0cbd — feat: use clean best frame for detection

Benefit:

- No stitch seams.
- No multi-frame ghost layers.
- Less shape distortion than the mosaic.

Failure:

- The best frame could cut off a real piece or omit part of the batch.
- This made the inventory incomplete, which is worse than a visually imperfect image.
- Emily explicitly established that accurate inventory is the highest priority.

#### Attempt 3 — cleaner object-aware mosaic/rectified views

This reduced some hard rectangular edges but exposed the deeper issue:

- Plane rectification is exact only for the table plane.
- Raised LEGO geometry is subject to parallax.
- The orange physical Brick 1 x 8 appeared stretched enough to be identified as 1 x 10.
- A yellow piece was visibly contorted/partially reconstructed.
- Blending can make the boundary look smoother while still changing the object shape.

Conclusion:

**A mosaic is useful for human visualization and coverage, but not trustworthy as the source of truth for piece detection or identification.**

### 4.5 Codex phase C — original-frame multiview inventory architecture

Design/implementation commits in lego-cv:

- 3decb64 — docs: design multiview inventory fusion
- dea5cf8 — docs: plan multiview inventory fusion
- 2a86103 — feat: fuse multiview piece observations
- 415b55e — feat: scan multiview capture sessions
- c462f97 — fix: stabilize real-session multiview fusion

The architectural shift was:

1. Select several original RGB frames.
2. Detect each original/rectified view independently.
3. Map each observation to the shared table coordinate system.
4. Fuse observations that represent the same physical piece.
5. Keep the mosaic only as a debug/coverage visualization.

This solved the “one frame cuts something off” problem because a piece can be seen in another frame. It also avoided asking the detector to interpret blended pixels.

The first version still used rectified evidence for some operations and had crop/fusion limitations. That became the starting point for the next Claude phase.

### 4.6 Claude interlude — inherited Phase 1 and Phase 2 work

Emily returned temporarily to Claude Code after receiving a comprehensive Codex summary. When she later returned to Codex, the supplied handoff said these phases were already built and approved.

#### Phase 1 — original unwarped crops

Capture commit:

- 589dd97 — feat: manifest v2 with bridge geometry; symmetric 25 mm workspace pad

CV commits:

- 9fd6758 — original_crop: map canvas boxes to padded original-frame AABBs
- 1d83764 — identify fused instances from original unwarped crops
- b74e4c8 — debug overhaul: crop gallery + original-frame overlays, box-free mosaic
- 5c354e4 — ID-view selection by obliquity/sharpness/original-frame fit
- 5b4dd65 — mask ID crops to the piece’s projected quad

This ensured Brickognize saw clean pixels from an original camera frame instead of a warped top-down object.

#### Phase 2 — stud advisory and second-chance identification

CV commits:

- 9810e37 — stud-count advisory module + candidate lists
- 22a83fa — wire stud advisory + second-chance identification
- 1d3d7a9 — second chance evaluates all alternates and keeps best score

Stud estimates were deliberately not made the primary identifier. They are noisy, especially on beams and oblique views. They help reject a grossly inconsistent Brickognize candidate or justify checking alternate views.

The Phase 2 gate was approved. Residual touching/merged-crop problems were assigned to Phase 3 instance segmentation rather than endlessly tuning heuristics.

### 4.7 Codex phase D — approved Phase 3 trained separator

Emily supplied an approved detailed Phase 3 execution handoff and a set of new Record3D captures. Codex executed the plan.

#### 4.7.1 Catalog and audit

CV commits:

- 012c841 — feat: catalog and audit segmentation sessions
- ebdb1d9 — feat: select diverse original training frames

The audit covered 20 session IDs. Nineteen were usable. **20260717-153911 contained zero frames** and was excluded.

The selector chose time-diverse, sharp, useful original frames rather than adjacent near-duplicates. It preserved whole-session split boundaries so frames from one scan could not leak across train/validation/test.

#### 4.7.2 Annotation bootstrap and human review

CV commits:

- 4ef04fb — feat: validate and assemble YOLO segmentation data
- 1fea3a8 — feat: bootstrap reviewed segmentation annotations

Final reviewed dataset:

- 71 original RGB images
- 1,157 verified instance polygons
- one semantic class: lego_piece
- 48 train images
- 15 validation images
- 8 held-out test-environment images

Important Labelme clarification:

- “lego_piece (31)” in the Labelme side panel was an instance/display count, not a class name.
- The semantic label must remain exactly **lego_piece** for every polygon.
- The final reviewed JSON files contain only lego_piece labels.

Partial/off-frame images:

- It was acceptable for a selected training image not to contain the entire batch.
- Every visible LEGO instance, including a genuinely partial edge instance, should be outlined.
- Nothing outside the image should be invented.
- Whole-session diversity provides other complete views.

Annotation import bug:

- Nine polygon vertices in four Labelme files landed exactly at x = image width or y = image height.
- The strict importer initially rejected them as out of bounds.
- Labelme can legitimately place a boundary point on that inclusive edge.
- The importer was fixed with tests to accept and clamp an exact boundary coordinate to the last valid pixel while still rejecting a true overshoot.

The noisy Labelme terminal DEBUG output was benign GUI logging. Saving/quitting the GUI and then stopping the terminal process was sufficient; all 71 reviewed files were imported and verified.

#### 4.7.3 YOLO training

CV commit:

- 3245b89 — feat: train reproducible YOLO11 segmenter

Training details:

- base model: yolo11n-seg.pt
- device: Apple MPS
- epochs: 100
- image size: 640
- batch: 4
- seed: 17
- full run duration: 597.2 seconds
- dataset hash: a751e0b2b456f867c62c6c50467ba3bc635869d4fe2706ac9ced31579a0c1a8c
- installed checkpoint SHA-256: c792bdad68351b6930d6e67dd41282b497a7c06b05ee7107dbe416b2ff4ea689

Frame-level mask mAP50-95:

- train-seen: 0.8465
- validation: 0.8366
- held-out environment: 0.8770

These are segmentation metrics, not inventory-accuracy metrics. Product-level session counts below are the meaningful end-to-end evidence.

#### 4.7.4 Runtime integration

Capture commits:

- 7a09cce — feat: export physical workspace geometry
- 5f2bd36 — feat: retain diverse confirmation views

CV commits:

- 11eea70 — feat: integrate original-frame segmented inventory
- 687a665 — feat: evaluate segmented multiview inventory
- ddf18f8 — feat: review moderate IDs with safe source context

Current runtime behavior:

- Capture manifest v3 exports a detector-independent outer physical workspace from repeated table-plane depth observations.
- The dense visualization canvas remains separate from that hard physical workspace.
- The exporter retains at least three time-diverse confirmation views when possible and at most five.
- YOLO runs on the original RGB region projected from the physical workspace, never on the stitched mosaic.
- Each mask carries source-frame ID, confidence, original polygon/mask, canvas/table geometry, gate evidence, and rejection reason.
- Predictions outside the physical workspace are rejected before fusion.
- Observations from the same source frame are never fused together; this prevents two real adjacent pieces in one frame becoming one record.
- Geometry-strong observations can fuse across color-family disagreement, which fixed one large plate being split into blue and gray records.
- Mask-aware color is voted across retained observations.
- Brickognize gets a true mask-isolated crop from an original frame.
- A moderate Brickognize result may be rechecked with real padded context only when no neighboring segmentation mask enters that context.
- session_cli.py --detector auto prefers the verified local segmenter and falls back to the classical detector if the model is absent.

Debug overlays:

- cyan: physical workspace
- magenta: segmentation inference ROI
- green: accepted instance evidence
- red: rejected evidence, annotated with source/confidence/workspace-overlap/reason

## 5. Detailed problem → fix → follow-on problem ledger

This section is intentionally repetitive. It preserves the causal chain so a future agent does not reintroduce a previously rejected fix.

### Problem 1 — Record3D connects but no callback ever fires

Evidence:

- Instrumentation before throttling/copy/queue logic printed zero callbacks.
- A minimal probe using demo-main.py’s exact callback/event wiring also got zero callbacks.
- A stale capture.py process and simultaneous demo client were found.
- Record3D lists the device and connect() succeeds even when another client owns the stream.

Root cause:

- Record3D streams to exactly one client.
- A second client can connect silently but receives no frames.
- After repeated abrupt connects/disconnects, the phone app can also wedge or stop streaming when backgrounded/locked.

Fix:

- Removed temporary instrumentation.
- Added a one-time five-second zero-frame watchdog in capture.py.
- Warning names the real checks.
- Commit ead3020.

Physical recovery:

1. Stop every demo-main.py and capture.py.
2. Keep the iPhone unlocked.
3. Bring Record3D to foreground.
4. Toggle USB streaming off/on or relaunch Record3D.
5. Run capture.py alone.

No Python callback/queue rewrite is warranted unless callbacks are proven to arrive and frames are then lost.

### Problem 2 — coverage stuck around 95% and top-down included background

Root cause:

- A convex hull over all observed plane cells included sparse background corners and objects beyond the intended scan.

Fix:

- Dense bounded workspace/component logic.
- Completion, crop, and workspace percentage use the bounded region.
- Enclosed missing holes remain incomplete.

Follow-on:

- A dense image canvas is not the same thing as a hard physical table workspace.
- Phase 3 therefore added a separate physical workspace contour for detector gating rather than reusing only the visualization crop.

### Problem 3 — giant background boxes became “unknown brick”

Root cause:

- Classical contours saw desk/laptop/floor/edge geometry as foreground.

Fix:

- Conservative size and main-region postfilter.

Follow-on:

- Global size thresholds can delete real large plates or tiny pieces.
- This filter is useful as a fallback but is not the primary Phase 3 defense.
- Physical workspace gating plus multi-view evidence is now preferred.

### Problem 4 — nearby/touching pieces were merged and the UI demanded a rescan

Root cause:

- The contour detector reasoned about one connected blob, not physical instances.

Fix attempt:

- Color-aware splitting helped different-color neighbors.

Follow-on:

- Same-color touching pieces and variable lighting remained unsolved.
- The “cluster: separate pieces and re-scan” advice was not acceptable product behavior.
- YOLO instance segmentation replaced this as the primary separator.

### Problem 5 — gray piece missed; small ghost rectangles became unknown bricks

Root cause:

- Gray-on-neutral contrast is difficult for classical thresholds.
- Mosaic overlaps and invalid rectified borders produced contours unrelated to physical pieces.

Fix attempts:

- Color-aware splitting and conservative contour filters.
- Invalid-border fill and cleaner rectified views.

Follow-on:

- These could reduce visual artifacts but not resolve parallax or create missing visual evidence.
- Original-frame segmentation is the current solution.

### Problem 6 — best-frame detection was clean but incomplete

Root cause:

- One selected image did not contain every physical piece.

Fix:

- Detect across several time-diverse original frames and fuse in table coordinates.

Follow-on:

- Fusion introduced its own duplicate/split/boundary-evidence problems, addressed below.

### Problem 7 — stitched/rectified pieces were contorted

Evidence:

- Orange Brick 1 x 8 appeared long enough to be called 1 x 10.
- A yellow piece was badly reconstructed.

Root cause:

- The homography describes the table plane, not the raised surface of a LEGO part.
- Camera pose/plane error plus parallax misaligns piece pixels across views.

Decision:

- Never use mosaic pixels as the detector/identifier source of truth.
- Keep the top-down as visualization/coverage only.

### Problem 8 — original-frame multiview could double-count or merge incorrectly

Root cause:

- The same physical piece appears in multiple frames.
- Nearby pieces can overlap after imperfect table projection.

Fix:

- Fuse observations in table coordinates with geometry/color evidence.
- Preserve source IDs.
- Enforce the invariant that two observations from the same source frame cannot fuse.

Follow-on:

- A large plate looked blue in one view and gray in another and split into two records.
- Strong geometry now overrides color-family shift when overlap/distance is decisive.

### Problem 9 — selected views admitted cable/clothing/table-edge proposals

Evidence:

- Historical session 212422 temporarily produced 9 records against 7 physical pieces.

Tempting but rejected fix:

- Raising the global confidence cutoff. This would also delete real small/edge pieces.

Fix:

- Physical workspace gate.
- Boundary-only evidence needs supporting observations from multiple views.
- A strong, complete singleton is still allowed if other views do not cover it.
- Image/workspace edge handling uses an explicit zero-valued erosion border.

Follow-on:

- Initial boundary filtering created extra weak singletons and could suppress real edge pieces.
- The final rule distinguishes boundary-only, weak incomplete singleton, and strong complete singleton evidence instead of using one hard cutoff.

### Problem 10 — 150212 returned 16 records instead of 14

Root causes:

- One large plate split into blue and gray fusion groups.
- A false top-edge proposal survived.

Fix:

- Strong geometry overrides color disagreement at high overlap/low normalized distance.
- Boundary evidence uses multi-view majority/support rules.

Result:

- 150212 returned 14/14.

### Problem 11 — Brickognize misidentified the orange 1 x 8

Earlier behavior:

- Neutral/masked crops sometimes produced Plate 1 x 8 around 0.787.
- Stud advice alone did not consistently rescue it.

Evidence:

- Isolated raw padded context from another original frame returned Brick 1 x 8 as high as 0.895.

Fix:

- For moderate-confidence IDs, review up to three original source observations.
- Raw context is allowed only when no neighboring mask enters the padded crop.
- Touching pieces remain mask-isolated.

Result:

- 005145 orange piece is Brick 1 x 8, part 3008, score 0.895.

### Problem 12 — exact Labelme boundary coordinates failed import

Root cause:

- Labelme emitted x = width or y = height for a point lying on the visible outer border.
- The importer treated all coordinates greater than width - 1/height - 1 as malformed.

Fix:

- Exact inclusive boundary values are accepted and clamped.
- True overshoots remain errors.
- Tests cover both cases.

### Problem 13 — Labelme showed lego_piece and lego_piece (32)

Explanation:

- Parenthesized numbers in the UI represented instances/display entries, not intended semantic categories.

Resolution:

- Final labels were normalized/verified as exactly lego_piece.
- The YOLO dataset contains one class only.

### Problem 14 — selected training views sometimes showed only part of the batch

Answer:

- That is acceptable for instance segmentation.
- Annotate every visible instance and visible partial instance accurately.
- Do not invent an off-frame polygon.
- Multiple selected frames and whole-session grouping provide complementary views.

Risk:

- A runtime scan still needs each physical piece fully visible in at least one clean view, preferably two or three.

### Problem 15 — held-out round white table remains 17/19

Session:

- 20260717-155228

Evidence:

- Individual selected views had 24, 19, and 18 accepted masks.
- Fusion produced 17 records.
- Table-plane inlier fraction was only 0.705.
- The other white-round session, 153942, had 0.587 plane inliers but reached 11/11.

Classification:

- Geometry/fusion count mismatch, not simply “YOLO missed two pieces.”
- One 155228 item also remained an identification unknown.

Decision:

- Do not tune against this test set and continue calling it held out.
- Collect a new round-table validation group for future geometry improvements.

### Problem 16 — shell command with angle-bracket placeholder failed

The command:

    sessions/<NEW_SESSION_ID>

caused zsh parse error because < and > are shell redirection operators. Use an actual session ID or a quoted variable, shown in the runbook below.

## 6. Current model, dataset, and split

Model:

- architecture: YOLO11n-seg
- class count: 1
- class name: lego_piece
- role: instance separation only

Dataset split by entire session:

### Train sessions

- 20260717-005145
- 20260717-145941
- 20260717-151014
- 20260717-151357
- 20260717-151611
- 20260717-152048
- 20260717-152202
- 20260717-152340
- 20260717-152721
- 20260717-153020
- 20260717-153313
- 20260717-153423
- 20260717-154428

### Validation sessions

- 20260717-150212
- 20260717-150350
- 20260717-152458
- 20260717-153200
- 20260717-154929

### Held-out environment sessions

- 20260717-153942
- 20260717-155228

The held-out sessions are the white-round-table environment. Do not silently move them into training and still present their old results as unseen generalization.

## 7. Product-level evaluation

References are manually inspected historical counts or the maximum count in human-verified source frames. They are not exhaustive 3D ground truth.

### Historical/train-seen

| Session | Reference | Fused | Unknown |
|---|---:|---:|---:|
| 20260716-212422 | 7 | 7 | 0 |
| 20260716-221544 | 8 | 8 | 0 |
| 20260716-234326 | 8 | 8 | 0 |
| 20260717-005145 | 8 | 8 | 0 |

These are regression results, not generalization evidence.

### Untouched validation sessions

| Session | Environment | Reference | Fused | Unknown |
|---|---|---:|---:|---:|
| 20260717-150212 | wood | 14 | 14 | 0 |
| 20260717-150350 | wood | 7 | 7 | 0 |
| 20260717-152458 | wood | 23 | 23 | 1 |
| 20260717-153200 | wood | 8 | 8 | 0 |
| 20260717-154929 | gray | 15 | 15 | 0 |

152458 is count-exact; its residual is an identification unknown, not a separation miss.

### Held-out white-round environment

| Session | Reference | Fused | Unknown | Note |
|---|---:|---:|---:|---|
| 20260717-153942 | 11 | 11 | 0 | weak geometry risk |
| 20260717-155228 | 19 | 17 | 1 | weak geometry + fusion mismatch |

### 20260717-233252 reliability regression (2026-07-18)

This 13-piece session validates the hard-workspace, true-view-ray, evidence,
and training-attribution work. The previous exporter selected three nearly
top-down views over a 340 x 430 dense-core canvas and produced 12 inventory
components. The new 961 x 868 canvas spans the approximately 48.3 x 43.5 cm
hard admissible workspace and includes the previously omitted orange 1x8.

Selected frames are coverage anchors 14/54/35 and oblique confirmations
263/231. The confirmations are 30.3 and 56.8 degrees from top-down;
confirmation-involving ray separations range from 37.9 to 77.7 degrees. The
five-view cap is mathematically incompatible with the original 99.5% union
target on this recording: the best five coverage-only candidates reach about
98.7%, and nine are required for 99.5%. The selected 3+2 set reaches 97.50% of
the recorded union and honestly serializes `coverage_complete: false`.

YOLO sees 13 accepted masks in frame 14. Across all selected views it produces
57 accepted detections, which now fuse to exactly 13 components. A prior 14th
component was not a hallucinated mask: two different raised pieces overlapped
after oblique table-plane projection and cross-matched. Confirmation-only
groups are now atomically reassigned only to distinct same-family coverage
anchors under a strong absolute geometry match; ambiguous or single-anchor
groups remain separate.

Part-ID result, ignoring the explicitly out-of-scope color field:

- 10/13 correct.
- Side blue 2x4 remains Plate 2x4: two reviewed views vote Plate, one votes
  Brick.
- Side white 2x2 remains Brick 1x2: one reviewed view votes 1x2 and one votes
  2x2, recorded as `identity_view_disagreement`.
- Black 2x4 remains unknown: all reviewed Brickognize results are below the
  identity threshold, while metric and stud evidence correctly preserve the
  conflict rather than inventing an answer.
- Black 2x12 remains part 2445, Plate 2x12; stud evidence cannot overwrite it.

Attribution for this session is capture pass -> selection pass -> segmentation
pass -> fusion pass after repair -> identification/evidence for the three
residuals. It is not evidence for more YOLO training crops. Historical depth
has no confidence sidecars, so height is recorded as shadow-only/unavailable;
new captures persist confidence when Record3D supplies it.

## 8. Exact runbook

### 8.1 Physical capture [EMILY]

From the capture repository:

~~~bash
cd /Users/emily/lego-capture
.venv/bin/python capture.py --live-view
~~~

Press Enter in the terminal or q in the live coverage window to stop.

If frames remain at zero for five seconds:

~~~bash
ps aux | grep -E 'demo-main|capture.py'
~~~

Then stop any stale competing client. On the phone, keep Record3D foregrounded/unlocked and toggle USB streaming or relaunch the app.

### 8.2 Export and scan a new session

Replace the example value, not the variable name:

~~~bash
SESSION_ID=20260717-XXXXXX

cd /Users/emily/lego-capture
.venv/bin/python multiview.py "sessions/$SESSION_ID"

cd /Users/emily/lego-cv
.venv/bin/python session_cli.py \
  "/Users/emily/lego-capture/sessions/$SESSION_ID" \
  --detector auto
~~~

For a known historical regression session:

~~~bash
SESSION_ID=20260716-234326
~~~

For the newest session on disk, this can also fill the variable automatically:

~~~bash
SESSION_ID=$(basename "$(ls -1dt /Users/emily/lego-capture/sessions/20* | head -1)")
echo "$SESSION_ID"
~~~

Check the printed ID before running the pipeline. Do not type angle brackets into zsh.

### 8.3 Inspect results

Primary files in the capture session:

- inventory.json
- debug/crop_gallery.jpg
- debug/frame_*_seg_gate.jpg
- multiview/manifest.json

For the target regression:

- /Users/emily/lego-capture/sessions/20260717-005145/inventory.json
- /Users/emily/lego-capture/sessions/20260717-005145/debug/crop_gallery.jpg

### 8.4 Run tests

~~~bash
cd /Users/emily/lego-capture
.venv/bin/pytest -q

cd /Users/emily/lego-cv
.venv/bin/pytest -q
~~~

### 8.5 Force detector modes

~~~bash
cd /Users/emily/lego-cv

# Preferred installed segmenter, with classical fallback if unavailable
.venv/bin/python session_cli.py /absolute/session/path --detector auto

# Require segmentation
.venv/bin/python session_cli.py /absolute/session/path --detector seg

# Classical fallback for comparison
.venv/bin/python session_cli.py /absolute/session/path --detector cv
~~~

## 9. Capture guidance supported by current evidence

For a demo or real scan:

1. Spread pieces in one layer. Touching is allowed; complete occlusion is not recoverable.
2. Keep loose non-LEGO clutter off the active tabletop, especially cables, clothing, hands, laptop edges, and tools.
3. Leave visible bare table around the batch so depth can repeatedly fit the plane and physical boundary.
4. Use a smooth mostly top-down sweep with modest angle variation.
5. Make every piece fully visible in at least two views, preferably three.
6. Include closer overlapping passes for small and edge pieces.
7. Move slowly enough for crisp stud edges and avoid harsh glare/shadows.
8. Do not trust coverage percentage as proof that an occluded piece was seen.

The runtime supports a layout larger than one fixed camera view. It selects and segments up to five original frames and fuses them. For a larger batch, sweep overlapping regions and keep transition pieces visible in multiple views. If the batch is too large to preserve clear details and stable table geometry, use multiple sessions and merge inventories at the application layer.

Do not impose a universal Laplacian sharpness threshold across surfaces. Gray sessions had much higher raw sharpness numbers than wood/white partly because surface texture changes the metric.

## 10. Current known limitations

- 155228 remains 17/19 because of weak geometry/fusion, not because the training run should simply be longer.
- 152458 has one identification unknown.
- Bare-table zero-LEGO behavior needs an explicit product test.
- Severe clutter inside the valid physical workspace needs more negative training/evaluation.
- Transparent, reflective, stacked, mostly occluded, flipped, and unusual non-rectangular pieces are not robustly validated.
- Color is mask-aware and multi-view-voted, but there is no per-piece color ground-truth benchmark across lighting environments.
- Brickognize is closed-source and rate-limited; local caching and conservative multi-view review are important.
- A piece whose informative side is never visible cannot be identified by software.
- Table-coordinate fusion depends on a stable plane. Small/round/featureless tables remain a geometry risk.

## 11. Recommended next work, in priority order

1. **Run one genuinely new live session** with the current model and inspect inventory plus every segmentation gate overlay.
2. **Do not tune on 155228.** Capture new round/small-table validation layouts with strong bare-plane visibility.
3. Improve table/workspace geometry and fusion using new data, keeping segmentation metrics separate from geometry failures.
4. Add explicit bare-table and in-workspace negative-clutter product tests.
5. Capture larger batches and new surfaces/lighting to verify the multi-frame design outside the July 17 data.
6. Build the confirmation/review UI for moderate IDs and unknowns.
7. Lock the inventory JSON contract with the teammate’s generator.
8. Only then consider retraining with additional diverse verified masks.

High-value future training captures:

- new surfaces and lighting with multiple independent layouts;
- same-color touching pairs/triples;
- edge, corner, and partial-overlap cases;
- small/upright/flipped/unusual pieces;
- negative tabletop clutter deliberately left unlabeled;
- new round-table sessions reserved as validation.

## 12. Important architecture invariants

Preserve these unless new evidence justifies a design change:

- Depth fits table/workspace only.
- Detection and identification operate on original RGB, not stitched mosaic pixels.
- YOLO separates instances but does not identify part IDs.
- Brickognize remains the part-ID service.
- Stud counting is advisory, not load-bearing.
- The classical detector remains a fallback.
- Same-source-frame observations cannot fuse.
- Physical workspace gating is independent of the selected detector.
- Strong edge pieces must not be removed by one global size/confidence cutoff.
- A moderate raw-context Brickognize retry is allowed only when the crop contains no neighboring piece mask.
- Evaluation results must state whether sessions are train-seen, validation, or held out.

## 13. Commit map

### lego-capture, first transfer onward

- ead3020 — warn when connected but receiving no frames
- 3969542 — bound coverage and top-down to dense workspace
- aee0cbd — use clean best frame for detection
- 96ef438 — export coverage-aware rectified views
- 420430f — fill invalid multiview borders
- 589dd97 — manifest v2 with original-frame bridge geometry
- 7a09cce — export physical workspace geometry
- 5f2bd36 — retain diverse confirmation views
- 8ba0e51 — record Phase 3 product results

### lego-cv, first transfer onward

- 5ca5c2e — filter implausible background detections
- 3b57904 / 6a58f25 / fd79a72 — design, plan, and implement color-aware classical splitting
- 3decb64 / dea5cf8 / 2a86103 / 415b55e / c462f97 — original multiview design, fusion, scanning, and stabilization
- 9fd6758 / 1d83764 / b74e4c8 / 5c354e4 / 5b4dd65 — original unwarped crop bridge and debug work
- 9810e37 / 22a83fa / 1d3d7a9 — stud advisory and second-chance identification
- 012c841 — catalog/audit sessions
- ebdb1d9 — select diverse original frames
- 4ef04fb — validate and assemble YOLO dataset
- 1fea3a8 — bootstrap reviewed annotations
- 3245b89 — reproducible YOLO11 segmentation training
- 11eea70 — integrate original-frame segmented inventory
- 687a665 — evaluate segmented multiview inventory
- ddf18f8 — moderate-ID review with safe source context

## 14. What Claude should read first

In order:

1. This file.
2. /Users/emily/lego-capture/PROGRESS.md
3. /Users/emily/lego-cv/docs/phase3-capture-quality.md
4. /Users/emily/lego-cv/training/runs/product-evaluation/product-evaluation.md
5. /Users/emily/lego-cv/models/lego_seg.metadata.json
6. /Users/emily/lego-cv/pipeline/segdetect.py
7. /Users/emily/lego-cv/pipeline/multiview.py
8. /Users/emily/lego-cv/session_cli.py
9. /Users/emily/lego-capture/multiview.py
10. /Users/emily/lego-capture/coverage.py

Before changing anything, run both test suites and inspect Git status in both repositories. Preserve existing user files and do not treat ignored model/training artifacts as disposable.
