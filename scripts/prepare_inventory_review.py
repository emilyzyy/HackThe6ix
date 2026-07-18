#!/usr/bin/env python3
"""Prepare the twelve exact-count inventory batches for human review."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Batch:
    session_id: str
    expected_count: int


BATCHES = (
    Batch("20260718-161438", 48),
    Batch("20260718-161851", 74),
    Batch("20260718-162215", 83),
    Batch("20260718-162458", 64),
    Batch("20260718-162834", 88),
    Batch("20260718-163214", 85),
    Batch("20260718-163451", 63),
    Batch("20260718-163723", 67),
    Batch("20260718-164119", 86),
    Batch("20260718-164730", 57),
    Batch("20260718-165118", 39),
    Batch("20260718-165402", 26),
)


def selected_batches(session_id: str | None) -> list[Batch]:
    if session_id is None:
        return list(BATCHES)
    matches = [batch for batch in BATCHES if batch.session_id == session_id]
    if not matches:
        raise ValueError(f"unknown batch session: {session_id}")
    return matches


def run_dir(capture_root: Path, batch: Batch) -> Path:
    return (
        Path(capture_root) / "sessions" / batch.session_id
        / "analysis-runs" / "inventory-annotation-v1"
    )


def commands_for(
    batch: Batch, capture_root: Path, cv_root: Path
) -> tuple[list[str], list[str]]:
    capture_root = Path(capture_root)
    cv_root = Path(cv_root)
    session_dir = capture_root / "sessions" / batch.session_id
    export = [
        str(capture_root / ".venv/bin/python"),
        str(capture_root / "multiview.py"),
        str(session_dir),
    ]
    review = [
        str(cv_root / ".venv/bin/python"),
        str(cv_root / "session_cli.py"),
        str(session_dir),
        "--detector", "auto",
        "--analysis-dir", str(run_dir(capture_root, batch)),
        "--inventory-review",
        "--expected-count", str(batch.expected_count),
        "--review-workers", "12",
    ]
    return export, review


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def refresh_workspace(capture_root: Path, workspace_dir: Path) -> Path:
    sessions = []
    for batch in BATCHES:
        review_path = run_dir(capture_root, batch) / "review.json"
        if review_path.exists():
            sessions.append({
                "session_id": batch.session_id,
                "expected_count": batch.expected_count,
                "review_path": str(review_path.resolve()),
            })
    path = Path(workspace_dir) / "workspace.json"
    _atomic_json(path, {
        "schema_version": 1,
        "name": "LEGO inventory 2026-07-18",
        "expected_physical_total": sum(
            batch.expected_count for batch in BATCHES
        ),
        "sessions": sessions,
    })
    return path


def prepare(
    capture_root: Path,
    cv_root: Path,
    workspace_dir: Path,
    *,
    session_id: str | None = None,
    dry_run: bool = False,
) -> Path | None:
    capture_root = Path(capture_root).resolve()
    cv_root = Path(cv_root).resolve()
    workspace_dir = Path(workspace_dir).resolve()
    batches = selected_batches(session_id)
    for batch in batches:
        session_dir = capture_root / "sessions" / batch.session_id
        if not session_dir.is_dir():
            raise FileNotFoundError(f"missing session directory: {session_dir}")
        export, review = commands_for(batch, capture_root, cv_root)
        manifest = session_dir / "multiview" / "manifest.json"
        review_path = run_dir(capture_root, batch) / "review.json"
        if not manifest.exists():
            print(f"[{batch.session_id}] export: {shlex.join(export)}", flush=True)
            if not dry_run:
                subprocess.run(export, cwd=capture_root, check=True)
        else:
            print(f"[{batch.session_id}] multiview manifest already exists", flush=True)
        if not review_path.exists():
            print(f"[{batch.session_id}] review: {shlex.join(review)}", flush=True)
            if not dry_run:
                subprocess.run(review, cwd=cv_root, check=True)
        else:
            print(f"[{batch.session_id}] review bundle already exists", flush=True)
        if not dry_run:
            refresh_workspace(capture_root, workspace_dir)
    return None if dry_run else refresh_workspace(capture_root, workspace_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cv-root", type=Path, default=Path("/Users/emily/lego-cv"))
    parser.add_argument(
        "--workspace", type=Path,
        default=Path("/Users/emily/lego-capture/inventory-review-20260718"),
    )
    parser.add_argument("--session", choices=[batch.session_id for batch in BATCHES])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    path = prepare(
        args.capture_root, args.cv_root, args.workspace,
        session_id=args.session, dry_run=args.dry_run,
    )
    if path is not None:
        print(f"workspace: {path}", flush=True)


if __name__ == "__main__":
    main()
