#!/usr/bin/env python3
"""One-command LEGO scan, analysis, confirmation, and inventory handoff."""
from __future__ import annotations

import argparse
import csv
import os
import signal
import socket
import subprocess
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path

import cv2

from demo_analysis import render_analysis
from session_io import SessionReader


CAPTURE_ROOT = Path(__file__).resolve().parent
CV_ROOT = Path("/Users/emily/lego-cv")
DEFAULT_INVENTORY_CSV = (
    CAPTURE_ROOT
    / "outputs/019f72d0-9cbe-7431-8f4b-7ed4084bbd13"
    / "lego_inventory_for_model_generation_20260719.csv"
)


def build_processing_commands(
    capture_root: Path,
    cv_root: Path,
    session_dir: Path,
    output_dir: Path,
    *,
    fixed_inventory: Path | None,
    generator_url: str | None,
    inventory_csv: Path,
) -> list[list[str]]:
    capture_root = Path(capture_root)
    cv_root = Path(cv_root)
    session_dir = Path(session_dir)
    output_dir = Path(output_dir)
    commands = [
        [str(capture_root / ".venv/bin/python"),
         str(capture_root / "plane.py"), str(session_dir)],
        [str(capture_root / ".venv/bin/python"),
         str(capture_root / "multiview.py"), str(session_dir),
         "--max-frames", "3", "--min-frames", "1"],
        [str(cv_root / ".venv/bin/python"),
         str(cv_root / "fast_showcase_cli.py"), str(session_dir),
         "--output", str(output_dir),
         "--target", "4", "--min-safe", "3",
         "--max-views", "3", "--identify-limit", "12",
         "--inventory-csv", str(inventory_csv),
         "--workspace", str(output_dir / "review")],
    ]
    if fixed_inventory is not None:
        commands[-1].extend(["--fixed-inventory", str(fixed_inventory)])
    if generator_url:
        commands[-1].extend(["--generator-url", generator_url])
    return commands


def _analysis_frames(session_dir: Path) -> list:
    records = SessionReader(session_dir).frames()
    if not records:
        raise ValueError(f"captured session has no frames: {session_dir}")
    indices = sorted({0, len(records) // 2, len(records) - 1})
    return [records[index].load_rgb() for index in indices]


def _run_animated(command: list[str], frames: list, stage: str) -> str:
    result = {}

    def worker():
        result["completed"] = subprocess.run(
            command,
            cwd=str(Path(command[1]).resolve().parent),
            text=True,
            capture_output=True,
            check=False,
        )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    started = time.monotonic()
    height, width = frames[0].shape[:2]
    while thread.is_alive():
        image = render_analysis(
            frames, stage, time.monotonic() - started, (width, height)
        )
        cv2.imshow("LEGO Scanner", image)
        cv2.waitKey(16)
    thread.join()
    completed = result["completed"]
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.returncode:
        if completed.stderr:
            print(completed.stderr, end="")
        raise RuntimeError(
            f"processing command failed ({completed.returncode}): "
            + " ".join(command)
        )
    return completed.stdout


def _wait_for_port(port: int, timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise RuntimeError(f"confirmation server did not open port {port}")


def _free_port(port: int) -> None:
    """Kill any stale confirmation server still holding the port, so a new run
    never opens a previous run's (already-confirmed) workspace."""
    try:
        pids = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}"],
            capture_output=True, text=True, check=False,
        ).stdout.split()
    except FileNotFoundError:
        return
    for pid in pids:
        try:
            os.kill(int(pid), signal.SIGKILL)
        except (ProcessLookupError, ValueError):
            pass
    if pids:
        time.sleep(0.4)  # let the OS release the socket


def _start_confirmation(workspace: Path, cv_root: Path, port: int) -> str:
    _free_port(port)
    env = dict(os.environ)
    env["LEGO_INVENTORY_WORKSPACE"] = str(Path(workspace).resolve())
    subprocess.Popen(
        [str(cv_root / ".venv/bin/python"), "-m", "uvicorn",
         "inventory_review_app:create_app_from_env",
         "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(cv_root), env=env,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _wait_for_port(port)
    url = f"http://127.0.0.1:{port}/"
    webbrowser.open(url)
    return url


def validate_inventory_csv(path: Path) -> Path:
    path = Path(path).resolve()
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            required = {"piece_type", "color", "quantity"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "inventory CSV missing columns: "
                    + ", ".join(sorted(missing))
                )
            row_count = 0
            for row_number, row in enumerate(reader, start=2):
                row_count += 1
                piece_type = " ".join((row.get("piece_type") or "").split())
                color = " ".join((row.get("color") or "").split())
                if not piece_type or not color:
                    raise ValueError(
                        f"inventory CSV row {row_number} has an empty type or color"
                    )
                try:
                    quantity = int(row.get("quantity") or "")
                except ValueError as error:
                    raise ValueError(
                        f"inventory CSV row {row_number} quantity must be an integer"
                    ) from error
                if quantity <= 0:
                    raise ValueError(
                        f"inventory CSV row {row_number} quantity must be positive"
                    )
            if row_count == 0:
                raise ValueError("inventory CSV contains no pieces")
    except OSError as error:
        raise ValueError(f"cannot read inventory CSV {path}: {error}") from error
    return path


def run_confirmation(
    workspace: Path,
    cv_root: Path,
    inventory_csv: Path,
    mode: str,
    port: int,
) -> str | Path:
    if mode == "web":
        return _start_confirmation(workspace, cv_root, port)
    command = [
        str(Path(cv_root) / ".venv/bin/python"),
        str(Path(cv_root) / "opencv_confirmation.py"),
        "--workspace", str(workspace),
        "--inventory-csv", str(inventory_csv),
    ]
    completed = subprocess.run(
        command, cwd=str(cv_root), text=True, check=False
    )
    if completed.returncode:
        raise RuntimeError(
            f"native confirmation failed ({completed.returncode}): "
            + " ".join(command)
        )
    handoff = Path(workspace).parent / "handoff.json"
    return handoff if handoff.is_file() else Path(workspace)


def run_post_capture(
    session_dir: Path,
    *,
    capture_root: Path = CAPTURE_ROOT,
    cv_root: Path = CV_ROOT,
    fixed_inventory: Path | None = None,
    generator_url: str | None = None,
    inventory_csv: Path = DEFAULT_INVENTORY_CSV,
    confirmation_ui: str = "opencv",
    port: int = 8770,
) -> tuple[Path, str | Path]:
    session_dir = Path(session_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = session_dir / "analysis-runs" / f"fast-showcase-demo-{stamp}"
    commands = build_processing_commands(
        capture_root, cv_root, session_dir, output_dir,
        fixed_inventory=fixed_inventory, generator_url=generator_url,
        inventory_csv=inventory_csv,
    )
    frames = _analysis_frames(session_dir)
    stages = (
        "Selecting the clearest views...",
        "Selecting the clearest views...",
        "Finding clean pieces and matching LEGO types...",
    )
    for index, (command, stage) in enumerate(zip(commands, stages)):
        if index == 0 and (session_dir / "table_frame.json").exists():
            continue
        if index == 1 and (session_dir / "multiview/manifest.json").exists():
            continue
        _run_animated(command, frames, stage)
    final = render_analysis(
        frames, "Preparing your confirmations...", 0.7,
        (frames[0].shape[1], frames[0].shape[0]),
    )
    cv2.imshow("LEGO Scanner", final)
    cv2.waitKey(350)
    cv2.destroyAllWindows()
    workspace = output_dir / "review" / "workspace.json"
    if not workspace.is_file():
        raise RuntimeError(f"showcase did not produce a workspace: {workspace}")
    return workspace, run_confirmation(
        workspace, cv_root, inventory_csv, confirmation_ui, port
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--replay-session", type=Path)
    parser.add_argument("--fixed-inventory", type=Path)
    parser.add_argument("--generator-url")
    parser.add_argument(
        "--inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV
    )
    parser.add_argument(
        "--confirmation-ui", choices=("opencv", "web"), default="opencv"
    )
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--cv-root", type=Path, default=CV_ROOT)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        inventory_csv = validate_inventory_csv(args.inventory_csv)
    except ValueError as error:
        parser.error(str(error))
    if args.replay_session is not None:
        session_dir = args.replay_session
    else:
        if args.anchor is None:
            parser.error("live demo requires --anchor")
        from demo_capture import run_demo_capture
        from live_scan_viewer import YoloMaskDetector

        session_dir = (
            CAPTURE_ROOT / "sessions" / datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        detector = YoloMaskDetector(
            args.cv_root / "models/lego_seg.pt", args.cv_root
        )
        run_demo_capture(args.anchor, session_dir, detector)
    workspace, url = run_post_capture(
        session_dir,
        cv_root=args.cv_root,
        fixed_inventory=args.fixed_inventory,
        generator_url=args.generator_url,
        inventory_csv=inventory_csv,
        confirmation_ui=args.confirmation_ui,
        port=args.port,
    )
    print(f"workspace: {workspace}")
    print(f"confirmation: {url}")


if __name__ == "__main__":
    main()
