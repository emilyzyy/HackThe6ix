from pathlib import Path

from demo_flow import build_processing_commands


def test_processing_commands_calibrate_select_three_views_and_fast_showcase(
    tmp_path,
):
    capture_root = tmp_path / "lego-capture"
    cv_root = tmp_path / "lego-cv"
    session = capture_root / "sessions" / "scan"
    output = session / "analysis-runs" / "fast-showcase-demo"

    commands = build_processing_commands(
        capture_root, cv_root, session, output,
        fixed_inventory=tmp_path / "fixed.json",
        generator_url="http://127.0.0.1:9000/build",
    )

    assert len(commands) == 3
    assert commands[0][1].endswith("plane.py")
    assert commands[1][1].endswith("multiview.py")
    assert commands[1][commands[1].index("--max-frames") + 1] == "3"
    assert commands[1][commands[1].index("--min-frames") + 1] == "1"
    fast = commands[2]
    assert fast[1].endswith("fast_showcase_cli.py")
    assert fast[fast.index("--identify-limit") + 1] == "12"
    assert fast[fast.index("--workspace") + 1] == str(output / "review")
    assert "--fixed-inventory" in fast
    assert "--generator-url" in fast


def test_processing_commands_allow_demo_without_generator_configuration(tmp_path):
    commands = build_processing_commands(
        tmp_path / "capture", tmp_path / "cv", tmp_path / "session",
        tmp_path / "output", fixed_inventory=None, generator_url=None,
    )

    assert "--fixed-inventory" not in commands[-1]
    assert "--generator-url" not in commands[-1]
