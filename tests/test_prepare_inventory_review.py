from pathlib import Path

from scripts.prepare_inventory_review import (
    BATCHES,
    Batch,
    commands_for,
    selected_batches,
)


def test_supplied_batch_counts_total_780():
    assert len(BATCHES) == 12
    assert sum(batch.expected_count for batch in BATCHES) == 780
    assert BATCHES[0] == Batch("20260718-161438", 48)
    assert BATCHES[-1] == Batch("20260718-165402", 26)


def test_commands_are_run_scoped_and_enable_fast_review(tmp_path):
    capture_root = tmp_path / "lego-capture"
    cv_root = tmp_path / "lego-cv"
    batch = Batch("20260718-161438", 48)

    export, review = commands_for(batch, capture_root, cv_root)

    assert export == [
        str(capture_root / ".venv/bin/python"),
        str(capture_root / "multiview.py"),
        str(capture_root / "sessions/20260718-161438"),
    ]
    assert "--analysis-dir" in review
    assert str(
        capture_root / "sessions/20260718-161438/analysis-runs/inventory-annotation-v1"
    ) in review
    assert review[review.index("--expected-count") + 1] == "48"
    assert "--inventory-review" in review
    assert review[review.index("--review-workers") + 1] == "12"


def test_selected_batches_rejects_unknown_session():
    assert selected_batches(None) == list(BATCHES)
    assert selected_batches("20260718-165402") == [BATCHES[-1]]
    try:
        selected_batches("missing")
    except ValueError as error:
        assert "unknown batch session" in str(error)
    else:
        raise AssertionError("unknown batch was accepted")
