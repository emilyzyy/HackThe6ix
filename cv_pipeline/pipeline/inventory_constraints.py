"""Inventory allowlist used only by the compact confirmation showcase."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


def normalize_piece_name(value: object) -> str:
    return " ".join(str(value or "").lower().split())


@dataclass(frozen=True)
class InventoryCatalog:
    _canonical: dict[str, str]
    _quantities: dict[str, int]

    @classmethod
    def load(cls, path: Path) -> "InventoryCatalog":
        path = Path(path)
        try:
            stream = path.open(newline="", encoding="utf-8-sig")
        except OSError as error:
            raise ValueError(f"cannot read inventory CSV {path}: {error}") from error
        with stream:
            reader = csv.DictReader(stream)
            required = {"piece_type", "color", "quantity"}
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    "inventory CSV missing columns: " + ", ".join(sorted(missing))
                )
            canonical: dict[str, str] = {}
            quantities: dict[str, int] = {}
            for line_number, row in enumerate(reader, start=2):
                display = " ".join(str(row.get("piece_type") or "").split())
                key = normalize_piece_name(display)
                if not key:
                    raise ValueError(
                        f"inventory CSV line {line_number} has empty piece_type"
                    )
                try:
                    quantity = int(str(row.get("quantity") or ""))
                except ValueError as error:
                    raise ValueError(
                        f"inventory CSV line {line_number} quantity must be an integer"
                    ) from error
                if quantity <= 0:
                    raise ValueError(
                        f"inventory CSV line {line_number} quantity must be positive"
                    )
                canonical.setdefault(key, display)
                quantities[key] = quantities.get(key, 0) + quantity
        if not canonical:
            raise ValueError("inventory CSV contains no pieces")
        return cls(canonical, quantities)

    @property
    def allowed_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._canonical.values()))

    def canonical_name(self, value: object) -> str | None:
        return self._canonical.get(normalize_piece_name(value))

    def quantity_for(self, value: object) -> int:
        return self._quantities.get(normalize_piece_name(value), 0)

    def ranked_candidates(self, final: dict, raw: dict) -> list[dict]:
        by_identity: dict[tuple[str, str], dict] = {}
        for row in [final, *(raw.get("candidates", []) or [])]:
            canonical = self.canonical_name(row.get("name"))
            if canonical is None:
                continue
            candidate = {
                "part_id": row.get("part_id"),
                "name": canonical,
                "score": float(row.get("score") or 0.0),
            }
            key = (str(candidate["part_id"] or ""), canonical)
            previous = by_identity.get(key)
            if previous is None or candidate["score"] > previous["score"]:
                by_identity[key] = candidate
        return sorted(
            by_identity.values(), key=lambda row: row["score"], reverse=True
        )

    def resolve(self, final: dict, raw: dict) -> dict | None:
        ranked = self.ranked_candidates(final, raw)
        return ranked[0] if ranked else None
