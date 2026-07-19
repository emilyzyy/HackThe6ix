from pathlib import Path

from demo_flow import (
    CAPTURE_ROOT,
    CV_ROOT,
    DEFAULT_INVENTORY_CSV,
    build_parser,
    build_processing_commands,
    run_confirmation,
    validate_inventory_csv,
)


def test_demo_defaults_to_vendored_cv_pipeline():
    assert CV_ROOT == CAPTURE_ROOT / "cv_pipeline"
    assert (CV_ROOT / "fast_showcase_cli.py").exists()
    assert (CV_ROOT / "opencv_confirmation.py").exists()


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
        inventory_csv=tmp_path / "inventory.csv",
    )

    assert len(commands) == 3
    assert commands[0][1].endswith("plane.py")
    assert commands[1][1].endswith("multiview.py")
    assert commands[1][commands[1].index("--max-frames") + 1] == "3"
    assert commands[1][commands[1].index("--min-frames") + 1] == "1"
    fast = commands[2]
    assert fast[1].endswith("fast_showcase_cli.py")
    assert fast[fast.index("--identify-limit") + 1] == "12"
    assert fast[fast.index("--target") + 1] == "4"
    assert fast[fast.index("--min-safe") + 1] == "3"
    assert fast[fast.index("--inventory-csv") + 1] == str(
        tmp_path / "inventory.csv"
    )
    assert fast[fast.index("--workspace") + 1] == str(output / "review")
    assert "--fixed-inventory" in fast
    assert "--generator-url" in fast


def test_processing_commands_allow_demo_without_generator_configuration(tmp_path):
    commands = build_processing_commands(
        tmp_path / "capture", tmp_path / "cv", tmp_path / "session",
        tmp_path / "output", fixed_inventory=None, generator_url=None,
        inventory_csv=tmp_path / "inventory.csv",
    )

    assert "--fixed-inventory" not in commands[-1]
    assert "--generator-url" not in commands[-1]


def test_demo_defaults_to_inventory_constrained_native_confirmation():
    args = build_parser().parse_args(["--replay-session", "session"])

    assert args.confirmation_ui == "opencv"
    assert args.inventory_csv == DEFAULT_INVENTORY_CSV


def test_native_confirmation_runs_foreground_without_starting_web(
    tmp_path, monkeypatch,
):
    calls = []

    class Completed:
        returncode = 0

    monkeypatch.setattr(
        "demo_flow.subprocess.run",
        lambda command, **kwargs: calls.append((command, kwargs)) or Completed(),
    )
    monkeypatch.setattr(
        "demo_flow._start_confirmation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("web confirmation must not start")
        ),
    )

    result = run_confirmation(
        tmp_path / "workspace.json", tmp_path / "cv",
        tmp_path / "inventory.csv", "opencv", 8770,
    )

    assert result == tmp_path / "workspace.json"
    command, options = calls[0]
    assert command[1] == str(tmp_path / "cv" / "opencv_confirmation.py")
    assert command[command.index("--inventory-csv") + 1] == str(
        tmp_path / "inventory.csv"
    )
    assert options["check"] is False


def test_native_confirmation_returns_handoff_only_when_file_exists(
    tmp_path, monkeypatch,
):
    workspace = tmp_path / "review" / "workspace.json"
    workspace.parent.mkdir()
    handoff = workspace.parent / "handoff.json"

    class Completed:
        returncode = 0

    def run(command, **kwargs):
        handoff.write_text("{}")
        return Completed()

    monkeypatch.setattr("demo_flow.subprocess.run", run)

    assert run_confirmation(
        workspace, tmp_path / "cv", tmp_path / "inventory.csv", "opencv", 8770
    ) == handoff


def test_web_confirmation_backup_uses_existing_server_path(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "demo_flow._start_confirmation",
        lambda workspace, cv_root, port: "http://127.0.0.1:8781/",
    )

    result = run_confirmation(
        tmp_path / "workspace.json", tmp_path / "cv",
        tmp_path / "inventory.csv", "web", 8781,
    )

    assert result == "http://127.0.0.1:8781/"


def test_inventory_preflight_rejects_invalid_quantity_before_capture(tmp_path):
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(
        "piece_type,color,quantity\nBrick 2 x 4,red,2\nPlate 2 x 2,blue,0\n"
    )

    try:
        validate_inventory_csv(inventory)
    except ValueError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("zero inventory quantity must fail preflight")
