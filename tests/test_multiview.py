import json

import cv2
import numpy as np
import pytest

from multiview import ViewCandidate, export_multiview, select_covering_views
from plane import compute_table_frame
from tests.synthetic import make_synthetic_session


def _candidate(frame_id, quality, valid):
    image = np.full((*valid.shape, 3), frame_id, dtype=np.uint8)
    return ViewCandidate(frame_id, quality, image, valid)


def test_selection_reaches_union_and_prefers_sharp_coverage_tie():
    left = np.zeros((2, 4), dtype=bool)
    left[:, :2] = True
    right = np.zeros((2, 4), dtype=bool)
    right[:, 2:] = True
    candidates = [
        _candidate(1, 1.0, left),
        _candidate(2, 2.0, left),
        _candidate(3, 1.0, right),
    ]

    selected = select_covering_views(candidates, max_frames=3)

    assert [view.frame_id for view in selected] == [2, 3]


@pytest.fixture
def session(tmp_path):
    out = tmp_path / "session"
    make_synthetic_session(out, n_frames=16, seed=21)
    compute_table_frame(out)
    return out


def test_export_writes_versioned_common_canvas_manifest(session):
    path, manifest = export_multiview(
        session, px_per_mm=1.0, max_frames=3, edge_margin_px=2
    )

    assert path.exists()
    assert json.loads(path.read_text()) == manifest
    assert manifest["version"] == 1
    assert 1 <= len(manifest["views"]) <= 3
    assert 0 < manifest["selected_coverage"] <= manifest["union_coverage"] <= 1
    assert all((session / view["image"]).exists() for view in manifest["views"])
    assert all((session / view["mask"]).exists() for view in manifest["views"])
    assert len({tuple(view["size_wh"]) for view in manifest["views"]}) == 1


def test_export_fills_pixels_outside_each_valid_footprint(session):
    _, manifest = export_multiview(
        session, px_per_mm=1.0, max_frames=2, edge_margin_px=2
    )
    view = manifest["views"][0]
    image = cv2.imread(str(session / view["image"]))
    valid = cv2.imread(str(session / view["mask"]), cv2.IMREAD_GRAYSCALE) > 0

    assert (~valid).any()
    assert float(image[~valid].mean()) > 40.0
