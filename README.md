# lego-capture — Stage 0 of the LEGO scanner

Records synchronized RGB / depth / pose sessions from a LiDAR iPhone over the
record3d USB stream, tracks table coverage live while you scan, and exports a
high-res orthographic top-down image of the workspace for the detection
pipeline in `lego-cv/`.

Depth from the phone is coarse (~256×192) and is used **only** to find the
table plane and mark coverage. All visual detail comes from the high-res RGB
channel projected onto that plane.

## Setup

```bash
source .venv/bin/activate       # python 3.11: record3d, numpy, opencv, open3d, pytest
```

On the iPhone: open the Record3D app, enable USB streaming, connect the cable.

## Capture (needs the phone)

```bash
python capture.py [--fps 6] [--out sessions] [--dev 0] [--live-view]
```

Press Enter to stop (or `q` in the coverage window with `--live-view`).
Frames are retained at ~6 fps regardless of stream rate. A status line updates
every 2 s; on stop you get a summary (frames, duration, disk size).

With `--live-view`, once ~20 frames exist the table plane is fitted and a
coverage window opens: gray = unseen, RGB aerial map = seen, plus a coverage %
and a COVERAGE COMPLETE banner once ≥95 % of the dense workspace is observed.
Sparse desk/floor observations are excluded from that workspace.

## Session layout

```
sessions/<YYYYMMDD-HHMMSS>/
  frames/00042.rgb.jpg        # full-res RGB (JPEG q95, BGR on disk via cv2)
  frames/00042.depth.npy      # float32 depth, meters
  frames.jsonl                # one line per frame: paths, intrinsics,
                              #   quaternion+translation AND 4x4 camera-to-world
  table_frame.json            # plane + world→table 4x4 (written by plane.py)
  coverage.npz                # coverage grid (written by coverage.py)
  topdown.jpg + topdown.json  # final export (written by topdown.py)
```

Every frame's pose/intrinsics are snapshotted **inside the stream callback**
at the same instant as its pixels — the metadata always describes exactly its
own frame.

## Offline pipeline (works on any recorded session)

```bash
python plane.py sessions/<ts>                 # fit table plane → table_frame.json
python coverage.py sessions/<ts>              # replay → coverage %, coverage.npz
python viewer.py sessions/<ts> [--fps 2]      # replay the coverage display
python topdown.py sessions/<ts>               # clean best-frame → topdown.jpg
python topdown.py sessions/<ts> --mode ortho --out topdown_composite.jpg
                                               # stitched diagnostic composite
```

`topdown.jpg` is cropped to the bounded workspace by default, keeping desk,
laptop, and floor clutter out of the `lego-cv` hand-off. Use
`python topdown.py <session> --no-workspace` for an uncropped diagnostic view.
The default best-frame export fills pixels outside the rectified camera view
with the median table color, so black warp borders cannot become false
detections. The stitched orthographic mode remains available for diagnostics,
but is not the detector input by default.
`topdown.json` documents the
mapping: pixel (u, v) ↔ table point `origin_xy + (u, v) / px_per_m`.

## No phone? Synthetic sessions

```bash
python -m tests.synthetic --out sessions/synthetic --frames 20
```

Generates a full fake session (checkerboard table, known plane, orbiting
camera, exact depth) through the real writer — the whole offline pipeline runs
on it. This is also how the test suite verifies Phases 2–4; after your first
real capture, iterate on the recorded session instead.

## Tests

```bash
pytest            # 54 tests, no device needed
```

## Coordinate conventions

- Poses are **camera-to-world** 4×4 (ARKit: +X right, +Y up, camera looks
  down −Z; image v grows downward).
- record3d intrinsics coeffs `tx/ty` are renamed `cx/cy` at the capture
  boundary; stored intrinsics correspond to the RGB image size and are
  rescaled to depth resolution via `transforms.scale_intrinsics`.
- The table frame puts the fitted plane at z = 0, origin at the observed
  centroid, z pointing up toward the cameras.

**Assumption to verify on the first real capture:** intrinsics correspond to
the RGB (not depth) image. Check: `python plane.py sessions/<ts>` should
report a high inlier fraction (>60 % for a table-dominated scene). If it's
tiny, flip the assumption in `plane.py`/`coverage.py` (scale from depth size
instead).
