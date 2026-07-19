"""Shared persistence for browser and native showcase confirmation."""
from __future__ import annotations

from pathlib import Path

from pipeline.inventory_constraints import normalize_piece_name
from training.inventory_review import atomic_write_json


def write_showcase_handoff(review_path: Path, document: dict) -> dict:
    if not document.get("showcase_only"):
        raise ValueError("handoff is only available for showcases")
    rows = document.get("components", [])
    if not rows or any(
        not (row.get("review") or {}).get("confirmed") for row in rows
    ):
        raise ValueError("confirm every showcase piece before handoff")
    allowed = {
        normalize_piece_name(name): name
        for name in document.get("inventory_piece_types", [])
    }
    confirmed = []
    for row in rows:
        review = row["review"]
        prediction = row.get("prediction", {})
        part_name = review.get("part_name") or "unknown brick"
        if allowed:
            canonical = allowed.get(normalize_piece_name(part_name))
            if canonical is None:
                raise ValueError(
                    f"{part_name!r} is not present in the inventory"
                )
            part_name = canonical
        confirmed.append({
            "component_id": row["component_id"],
            "part_id": review.get("part_id") or None,
            "part_name": part_name,
            "color": review.get("color") or None,
            "accepted_prediction": (
                review.get("part_id") == prediction.get("part_id")
                and review.get("part_name") == prediction.get("part_name")
            ),
        })
    handoff_path = Path(review_path).parent / "handoff.json"
    atomic_write_json(handoff_path, {
        "schema_version": 1,
        "session_id": document["session_id"],
        "fixed_inventory": document.get("fixed_inventory"),
        "confirmed_pieces": confirmed,
    })
    return {
        "handoff_path": str(handoff_path),
        "generator_url": document.get("generator_url"),
    }
