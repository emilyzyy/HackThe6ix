import csv
import subprocess
import sys
from pathlib import Path

from scripts.export_inventory_studio import (
    build_mpd,
    export_studio_inventory,
    ldraw_color,
    resolve_component,
)


def test_exporter_can_run_directly_by_script_path():
    repository = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(repository / "scripts/export_inventory_studio.py"), "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Export inventory-review drafts" in result.stdout


def component(
    *,
    component_id="piece-000",
    part_id="3001",
    color="blue",
    candidates=None,
    review=None,
):
    return {
        "component_id": component_id,
        "prediction": {
            "part_id": part_id,
            "part_name": "Predicted part",
            "color": color,
        },
        "candidates": candidates or [],
        "review": review or {"confirmed": False},
    }


def test_ldraw_color_uses_models_first_ambiguous_choice():
    assert ldraw_color("blue") == (1, "blue")
    assert ldraw_color("ambiguous(dark green/green)") == (288, "dark green")
    assert ldraw_color("ambiguous(beige/yellow)") == (19, "beige")


def test_confirmed_review_part_and_color_override_prediction():
    resolved = resolve_component(
        component(
            part_id="3001",
            color="red",
            review={
                "confirmed": True,
                "included": True,
                "part_id": "3009",
                "part_name": "Brick 1 x 6",
                "color": "dark green",
            },
        )
    )

    assert resolved.part_id == "3009"
    assert resolved.color_code == 288
    assert resolved.color_name == "dark green"
    assert resolved.source == "confirmed"
    assert resolved.placeholder is False


def test_unknown_identity_uses_best_candidate_and_predicted_color():
    resolved = resolve_component(
        component(
            part_id="unknown",
            color="yellow",
            candidates=[
                {"part_id": "3003", "part_name": "Brick 2 x 2", "score": 0.62},
                {"part_id": "3022", "part_name": "Plate 2 x 2", "score": 0.41},
            ],
        )
    )

    assert resolved.part_id == "3003"
    assert resolved.color_code == 14
    assert resolved.source == "candidate"
    assert resolved.placeholder is False


def test_no_candidate_becomes_conspicuous_placeholder():
    resolved = resolve_component(component(part_id="unknown", color="white"))

    assert resolved.part_id == "3005"
    assert resolved.color_code == 26
    assert resolved.color_name == "magenta placeholder"
    assert resolved.source == "placeholder"
    assert resolved.placeholder is True


def test_mpd_contains_one_editable_part_per_component_and_session_submodels():
    mpd, rows = build_mpd(
        [
            {
                "session_id": "session-a",
                "components": [
                    component(component_id="piece-000", part_id="3001", color="red"),
                    component(component_id="piece-001", part_id="3020", color="blue"),
                ],
            },
            {
                "session_id": "session-b",
                "components": [
                    component(component_id="piece-000", part_id="3003", color="white")
                ],
            },
        ]
    )

    assert "0 FILE session-session-a.ldr" in mpd
    assert "0 FILE session-session-b.ldr" in mpd
    assert mpd.count("0 !LEGO_CV COMPONENT") == 3
    assert "1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat" in mpd
    assert "1 1 300 0 0 1 0 0 0 1 0 0 0 1 3020.dat" in mpd
    assert len(rows) == 3


def test_export_writes_mpd_and_csv_sidecar(tmp_path):
    review_path = tmp_path / "review.json"
    review_path.write_text(
        '{"session_id":"session-a","components":['
        '{"component_id":"piece-000","prediction":'
        '{"part_id":"3001","part_name":"Brick 2 x 4","color":"red"},'
        '"candidates":[],"review":{"confirmed":false}}]}',
        encoding="utf-8",
    )

    mpd_path, csv_path = export_studio_inventory([review_path], tmp_path / "out")

    assert mpd_path.name == "lego-inventory-draft.mpd"
    assert "1 4 0 0 0" in mpd_path.read_text(encoding="utf-8")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {
            "session_id": "session-a",
            "component_id": "piece-000",
            "part_id": "3001",
            "color": "red",
            "ldraw_color": "4",
            "source": "prediction",
            "placeholder": "false",
            "x": "0",
            "y": "0",
            "z": "0",
        }
    ]
