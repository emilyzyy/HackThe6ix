import numpy as np

from demo_analysis import render_analysis


def _frames():
    return [
        np.full((180, 240, 3), (30, 60, 90), dtype=np.uint8),
        np.full((180, 240, 3), (90, 50, 20), dtype=np.uint8),
        np.full((180, 240, 3), (20, 100, 40), dtype=np.uint8),
    ]


def test_analysis_renderer_preserves_requested_size_and_dtype():
    out = render_analysis(
        _frames(), "Finding clean LEGO pieces...", 1.25, (960, 540)
    )

    assert out.shape == (540, 960, 3)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out) > 0


def test_analysis_indicator_changes_with_time():
    early = render_analysis(_frames(), "Selecting the clearest views...", 0.0,
                            (960, 540))
    later = render_analysis(_frames(), "Selecting the clearest views...", 0.4,
                            (960, 540))

    assert not np.array_equal(early, later)


def test_analysis_renderer_accepts_authoritative_mask_overlay():
    mask = np.zeros((180, 240), dtype=bool)
    mask[50:120, 80:170] = True

    plain = render_analysis(_frames(), "Finding clean LEGO pieces...", 0.2,
                            (960, 540))
    masked = render_analysis(_frames(), "Finding clean LEGO pieces...", 0.2,
                             (960, 540), masks=[mask])

    assert not np.array_equal(plain, masked)
