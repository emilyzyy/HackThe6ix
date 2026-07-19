#!/usr/bin/env python3
"""One-command LEGO scan, analysis, confirmation, and inventory handoff."""
from __future__ import annotations

import argparse
import os
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


def build_processing_commands(
    capture_root: Path,
    cv_root: Path,
    session_dir: Path,
    output_dir: Path,
    *,
    fixed_inventory: Path | None,
    generator_url: str | None,
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
         "--target", "5", "--min-safe", "3",
         "--max-views", "3", "--identify-limit", "12",
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


def _start_confirmation(workspace: Path, cv_root: Path, port: int) -> str:
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


def run_post_capture(
    session_dir: Path,
    *,
    capture_root: Path = CAPTURE_ROOT,
    cv_root: Path = CV_ROOT,
    fixed_inventory: Path | None = None,
    generator_url: str | None = None,
    port: int = 8770,
) -> tuple[Path, str]:
    session_dir = Path(session_dir).resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = session_dir / "analysis-runs" / f"fast-showcase-demo-{stamp}"
    commands = build_processing_commands(
        capture_root, cv_root, session_dir, output_dir,
        fixed_inventory=fixed_inventory, generator_url=generator_url,
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
    return workspace, _start_confirmation(workspace, cv_root, port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path)
    parser.add_argument("--replay-session", type=Path)
    parser.add_argument("--fixed-inventory", type=Path)
    parser.add_argument("--generator-url")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--cv-root", type=Path, default=CV_ROOT)
    args = parser.parse_args(argv)
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
        port=args.port,
    )
    print(f"workspace: {workspace}")
    print(f"confirmation: {url}")


if __name__ == "__main__":
    main()
