#!/usr/bin/env python3
"""Export inventory-review drafts as editable BrickLink Studio/LDraw pieces."""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__:
    from .prepare_inventory_review import BATCHES, run_dir
else:
    from prepare_inventory_review import BATCHES, run_dir


LDRAW_COLORS = {
    "black": 0,
    "blue": 1,
    "green": 2,
    "red": 4,
    "brown": 6,
    "light gray": 7,
    "dark gray": 8,
    "yellow": 14,
    "white": 15,
    "beige": 19,
    "orange": 25,
    "dark green": 288,
}

COLOR_ALIASES = {
    "tan": "beige",
    "light grey": "light gray",
    "dark grey": "dark gray",
}

PLACEHOLDER_PART = "3005"
PLACEHOLDER_COLOR = 26  # Magenta is outside Emily's inventory palette.
PART_ID = re.compile(r"^[A-Za-z0-9_-]+$")
GRID_COLUMNS = 10
GRID_SPACING = 300
SESSION_COLUMNS = 4
SESSION_SPACING = 3600
IDENTITY = "1 0 0 0 1 0 0 0 1"


@dataclass(frozen=True)
class ResolvedPiece:
    part_id: str
    color_code: int
    color_name: str
    source: str
    placeholder: bool
    included: bool = True


def ldraw_color(value: str | None) -> tuple[int, str]:
    """Resolve the model's leading inventory color to an LDraw code."""
    normalized = (value or "").strip().lower()
    if normalized.startswith("ambiguous(") and normalized.endswith(")"):
        normalized = normalized[len("ambiguous(") : -1].split("/", 1)[0].strip()
    normalized = COLOR_ALIASES.get(normalized, normalized)
    code = LDRAW_COLORS.get(normalized)
    if code is None:
        return PLACEHOLDER_COLOR, "magenta placeholder"
    return code, normalized


def _valid_part_id(value: object) -> str | None:
    candidate = str(value or "").strip()
    if candidate.lower() in {"", "unknown", "none", "null"}:
        return None
    return candidate if PART_ID.fullmatch(candidate) else None


def resolve_component(component: dict) -> ResolvedPiece:
    """Choose the editable Studio part/color represented by a draft component."""
    review = component.get("review") or {}
    confirmed = bool(review.get("confirmed"))
    included = not confirmed or review.get("included", True) is not False
    prediction = component.get("prediction") or {}

    if confirmed:
        reviewed_part = _valid_part_id(review.get("part_id"))
        if reviewed_part is not None:
            color_code, color_name = ldraw_color(review.get("color"))
            return ResolvedPiece(
                reviewed_part,
                color_code,
                color_name,
                "confirmed",
                color_name == "magenta placeholder",
                included,
            )

    predicted_part = _valid_part_id(prediction.get("part_id"))
    if predicted_part is not None:
        color_code, color_name = ldraw_color(prediction.get("color"))
        return ResolvedPiece(
            predicted_part,
            color_code,
            color_name,
            "prediction",
            color_name == "magenta placeholder",
            included,
        )

    for candidate in component.get("candidates") or []:
        candidate_part = _valid_part_id(candidate.get("part_id"))
        if candidate_part is not None:
            color_code, color_name = ldraw_color(prediction.get("color"))
            return ResolvedPiece(
                candidate_part,
                color_code,
                color_name,
                "candidate",
                color_name == "magenta placeholder",
                included,
            )

    return ResolvedPiece(
        PLACEHOLDER_PART,
        PLACEHOLDER_COLOR,
        "magenta placeholder",
        "placeholder",
        True,
        included,
    )


def _safe_token(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown"))


def build_mpd(session_payloads: Iterable[dict]) -> tuple[str, list[dict[str, str]]]:
    """Serialize review payloads into one MPD and return component provenance rows."""
    sessions = list(session_payloads)
    lines = [
        "0 FILE lego-inventory-draft.ldr",
        "0 LEGO CV draft inventory",
        "0 Name: lego-inventory-draft.ldr",
        "0 Author: LEGO CV pipeline",
        "0 !LDRAW_ORG Model",
        "0 // Open a session submodel in Studio to correct, delete, or add pieces.",
    ]
    for index, payload in enumerate(sessions):
        session_id = _safe_token(payload.get("session_id"))
        x = (index % SESSION_COLUMNS) * SESSION_SPACING
        z = (index // SESSION_COLUMNS) * SESSION_SPACING
        lines.append(f"1 16 {x} 0 {z} {IDENTITY} session-{session_id}.ldr")
    lines.append("0 NOFILE")

    rows: list[dict[str, str]] = []
    for payload in sessions:
        session_id = _safe_token(payload.get("session_id"))
        lines.extend(
            [
                f"0 FILE session-{session_id}.ldr",
                f"0 Inventory batch {session_id}",
                f"0 Name: session-{session_id}.ldr",
                "0 !LDRAW_ORG Model",
            ]
        )
        exported_index = 0
        for component in payload.get("components") or []:
            resolved = resolve_component(component)
            if not resolved.included:
                continue
            component_id = _safe_token(component.get("component_id"))
            x = (exported_index % GRID_COLUMNS) * GRID_SPACING
            z = (exported_index // GRID_COLUMNS) * GRID_SPACING
            exported_index += 1
            lines.append(
                "0 !LEGO_CV COMPONENT "
                f"{component_id} SOURCE {resolved.source} "
                f"COLOR {_safe_token(resolved.color_name)}"
            )
            lines.append(
                f"1 {resolved.color_code} {x} 0 {z} {IDENTITY} {resolved.part_id}.dat"
            )
            rows.append(
                {
                    "session_id": session_id,
                    "component_id": component_id,
                    "part_id": resolved.part_id,
                    "color": resolved.color_name,
                    "ldraw_color": str(resolved.color_code),
                    "source": resolved.source,
                    "placeholder": str(resolved.placeholder).lower(),
                    "x": str(x),
                    "y": "0",
                    "z": str(z),
                }
            )
        lines.append("0 NOFILE")
    return "\n".join(lines) + "\n", rows


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def export_studio_inventory(
    review_paths: Iterable[Path], output_dir: Path
) -> tuple[Path, Path]:
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in review_paths]
    mpd, rows = build_mpd(payloads)
    output_dir = Path(output_dir)
    mpd_path = output_dir / "lego-inventory-draft.mpd"
    csv_path = output_dir / "lego-inventory-draft.csv"
    _atomic_text(mpd_path, mpd)

    csv_buffer = io.StringIO(newline="")
    fieldnames = [
        "session_id",
        "component_id",
        "part_id",
        "color",
        "ldraw_color",
        "source",
        "placeholder",
        "x",
        "y",
        "z",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(csv_path, csv_buffer.getvalue())
    return mpd_path, csv_path


def main(argv: list[str] | None = None) -> None:
    capture_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, default=capture_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=capture_root / "inventory-review-20260718" / "studio",
    )
    args = parser.parse_args(argv)

    review_paths = [run_dir(args.capture_root, batch) / "review.json" for batch in BATCHES]
    missing = [path for path in review_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing inventory review files:\n" + "\n".join(str(path) for path in missing)
        )
    mpd_path, csv_path = export_studio_inventory(review_paths, args.output_dir)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    placeholders = sum(row["placeholder"] == "true" for row in rows)
    print(f"Studio model: {mpd_path}")
    print(f"Provenance CSV: {csv_path}")
    print(f"Draft pieces: {len(rows)}; placeholders needing part review: {placeholders}")


if __name__ == "__main__":
    main()
