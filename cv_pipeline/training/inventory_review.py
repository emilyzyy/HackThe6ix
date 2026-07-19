"""Human-reviewed, exact-count inventory annotations.

Predictions are immutable provenance.  Reviewer decisions are stored in a
separate ``review`` object and are the only values used by final aggregation.
"""
from __future__ import annotations

import copy
import json
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


INVENTORY_COLORS = (
    "red", "orange", "yellow", "beige", "brown", "green",
    "dark green", "blue", "white", "light gray", "dark gray", "black",
)
SEGMENTATION_STATES = (
    "unreviewed", "correct", "duplicate", "merged", "incomplete", "missed",
)
_REVIEW_FIELDS = {
    "confirmed", "part_id", "part_name", "color", "included",
    "exclusion_reason", "segmentation",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_review(prediction: dict) -> dict:
    return {
        "confirmed": False,
        "part_id": prediction.get("part_id") or "",
        "part_name": prediction.get("part_name") or "",
        "color": prediction.get("color") or "",
        "included": True,
        "exclusion_reason": None,
        "segmentation": "unreviewed",
    }


def new_review_document(
    session_id: str, expected_count: int, components: list[dict]
) -> dict:
    if not session_id:
        raise ValueError("session_id is required")
    if int(expected_count) <= 0:
        raise ValueError("expected physical count must be positive")
    rows = []
    seen = set()
    for component in components:
        row = copy.deepcopy(component)
        component_id = row.get("component_id")
        if not component_id or component_id in seen:
            raise ValueError("component IDs must be present and unique")
        seen.add(component_id)
        prediction = row.get("prediction") or {}
        row["prediction"] = copy.deepcopy(prediction)
        row["review"] = _default_review(prediction)
        rows.append(row)
    return {
        "schema_version": 1,
        "session_id": session_id,
        "expected_physical_count": int(expected_count),
        "components": rows,
        "manual_pieces": [],
        "locked": False,
        "created_at": _now(),
        "updated_at": _now(),
        "locked_at": None,
    }


def _validate_review(review: dict, *, require_color: bool = True) -> None:
    segmentation = review.get("segmentation")
    if segmentation not in SEGMENTATION_STATES:
        raise ValueError(f"invalid segmentation state: {segmentation}")
    if not review.get("confirmed"):
        return
    included = bool(review.get("included"))
    if segmentation == "duplicate" and included:
        raise ValueError("duplicate detections cannot be included")
    if included:
        if not str(review.get("part_name") or "").strip():
            raise ValueError("included piece requires part name")
        if require_color and review.get("color") not in INVENTORY_COLORS:
            raise ValueError("included piece requires a valid inventory color")
    elif not str(review.get("exclusion_reason") or "").strip():
        raise ValueError("excluded piece requires an exclusion reason")


def apply_component_review(
    document: dict, component_id: str, patch: dict
) -> dict:
    if document.get("locked"):
        raise ValueError("locked review documents cannot be edited")
    unknown = set(patch) - _REVIEW_FIELDS
    if unknown:
        raise ValueError(f"unknown review fields: {sorted(unknown)}")
    row = next(
        (item for item in document.get("components", [])
         if item.get("component_id") == component_id),
        None,
    )
    if row is None:
        raise KeyError(f"unknown component: {component_id}")
    updated = {**row["review"], **copy.deepcopy(patch)}
    _validate_review(
        updated, require_color=not bool(document.get("showcase_only"))
    )
    row["review"] = updated
    row["reviewed_at"] = _now()
    document["updated_at"] = _now()
    return row


def add_manual_piece(
    document: dict,
    *,
    part_id: str,
    part_name: str,
    color: str,
    included: bool = True,
    exclusion_reason: str | None = None,
) -> dict:
    if document.get("locked"):
        raise ValueError("locked review documents cannot be edited")
    index = len(document.setdefault("manual_pieces", []))
    row = {
        "manual_id": f"manual-{index:03d}",
        "source": "human-added-missed-piece",
        "review": {
            "confirmed": True,
            "part_id": str(part_id).strip(),
            "part_name": str(part_name).strip(),
            "color": color,
            "included": bool(included),
            "exclusion_reason": exclusion_reason,
            "segmentation": "missed",
        },
        "reviewed_at": _now(),
    }
    _validate_review(row["review"])
    document["manual_pieces"].append(row)
    document["updated_at"] = _now()
    return row


def add_waypoint_component(document: dict, waypoint_id: str) -> dict:
    """Promote one secondary-frame waypoint into independent review."""
    if document.get("locked"):
        raise ValueError("locked review documents cannot be edited")
    pool_row = next(
        (
            row for row in document.get("waypoint_pool", [])
            if row.get("waypoint_id") == waypoint_id
        ),
        None,
    )
    if pool_row is None:
        raise KeyError(f"unknown waypoint: {waypoint_id}")
    component = copy.deepcopy(pool_row["component"])
    component_id = component.get("component_id")
    if any(
        row.get("component_id") == component_id
        for row in document.get("components", [])
    ):
        raise ValueError(f"waypoint already added: {waypoint_id}")
    component["review"] = _default_review(component.get("prediction") or {})
    document.setdefault("components", []).append(component)
    document["updated_at"] = _now()
    return component


def add_manual_waypoint_component(document: dict, component: dict) -> dict:
    """Add one manually boxed source-frame region as an unconfirmed item."""
    if document.get("locked"):
        raise ValueError("locked review documents cannot be edited")
    row = copy.deepcopy(component)
    component_id = str(row.get("component_id") or "")
    if not component_id or any(
        item.get("component_id") == component_id
        for item in document.get("components", [])
    ):
        raise ValueError("manual waypoint component ID must be unique")
    row["review"] = _default_review(row.get("prediction") or {})
    row["review"]["segmentation"] = "missed"
    document.setdefault("components", []).append(row)
    document["updated_at"] = _now()
    return row


def _pair_key(component_a: str, component_b: str) -> str:
    return "::".join(sorted((str(component_a), str(component_b))))


def _box_overlap(box_a: list, box_b: list) -> tuple[float, float]:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0, 0.0
    try:
        ax0, ay0, ax1, ay1 = map(float, box_a)
        bx0, by0, bx1, by1 = map(float, box_b)
    except (TypeError, ValueError):
        return 0.0, 0.0
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0, 0.0
    intersection = (
        max(0.0, min(ax1, bx1) - max(ax0, bx0))
        * max(0.0, min(ay1, by1) - max(ay0, by0))
    )
    if intersection <= 0.0:
        return 0.0, 0.0
    union = area_a + area_b - intersection
    return intersection / union, intersection / min(area_a, area_b)


def _overlap_component(row: dict) -> dict:
    review = row.get("review") or {}
    anchor = row.get("source_anchor") or {}
    view = next(iter((row.get("assets") or {}).get("views") or []), {})
    return {
        "component_id": str(row.get("component_id") or ""),
        "box_original": list(anchor.get("box_original") or []),
        "part_id": str(review.get("part_id") or ""),
        "part_name": str(review.get("part_name") or ""),
        "color": str(review.get("color") or ""),
        "crop": view.get("crop") or view.get("masked_crop"),
    }


def overlap_candidates(document: dict) -> list[dict]:
    """Return unresolved, same-frame overlap pairs for explicit review."""
    dismissed = {
        str(row.get("pair_key") or "")
        for row in (document.get("overlap_cleanup") or {}).get("decisions", [])
        if row.get("action") == "separate"
    }
    eligible = []
    for row in document.get("components", []):
        review = row.get("review") or {}
        anchor = row.get("source_anchor") or {}
        if not (
            review.get("confirmed")
            and review.get("included")
            and review.get("segmentation") != "duplicate"
            and anchor.get("frame_id") is not None
            and len(anchor.get("box_original") or []) == 4
        ):
            continue
        eligible.append(row)
    eligible.sort(key=lambda row: str(row.get("component_id") or ""))
    candidates = []
    for index, first in enumerate(eligible):
        first_anchor = first["source_anchor"]
        for second in eligible[index + 1:]:
            second_anchor = second["source_anchor"]
            if int(first_anchor["frame_id"]) != int(second_anchor["frame_id"]):
                continue
            pair_key = _pair_key(first["component_id"], second["component_id"])
            if pair_key in dismissed:
                continue
            iou, containment = _box_overlap(
                first_anchor["box_original"], second_anchor["box_original"]
            )
            if iou < 0.20 and containment < 0.55:
                continue
            candidates.append({
                "pair_key": pair_key,
                "frame_id": int(first_anchor["frame_id"]),
                "iou": iou,
                "containment": containment,
                "a": _overlap_component(first),
                "b": _overlap_component(second),
            })
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["containment"]),
            -float(row["iou"]),
            row["pair_key"],
        ),
    )


def apply_overlap_decision(
    document: dict,
    component_a: str,
    component_b: str,
    action: str,
) -> dict:
    """Persist one explicit keep/separate decision for an overlap pair."""
    if document.get("locked"):
        raise ValueError("locked review documents cannot be edited")
    if action not in {"keep_a", "keep_b", "separate"}:
        raise ValueError(f"invalid overlap action: {action}")
    pair_key = _pair_key(component_a, component_b)
    candidate = next(
        (
            row for row in overlap_candidates(document)
            if row["pair_key"] == pair_key
        ),
        None,
    )
    if candidate is None:
        raise ValueError(
            f"pair is not a current overlap candidate: {pair_key}"
        )
    decision = {
        "pair_key": pair_key,
        "action": action,
        "component_a": str(component_a),
        "component_b": str(component_b),
        "kept_component_id": None,
        "removed_component_id": None,
        "removed_review": None,
        "decided_at": _now(),
    }
    if action != "separate":
        kept_id = str(component_a if action == "keep_a" else component_b)
        removed_id = str(component_b if action == "keep_a" else component_a)
        removed = next(
            row for row in document.get("components", [])
            if row.get("component_id") == removed_id
        )
        decision["kept_component_id"] = kept_id
        decision["removed_component_id"] = removed_id
        decision["removed_review"] = copy.deepcopy(removed.get("review") or {})
        apply_component_review(document, removed_id, {
            "confirmed": True,
            "part_id": "",
            "part_name": "",
            "color": "",
            "included": False,
            "exclusion_reason": (
                "duplicate overcount candidate; kept component " + kept_id
            ),
            "segmentation": "duplicate",
        })
    document.setdefault("overlap_cleanup", {}).setdefault(
        "decisions", []
    ).append(decision)
    document["updated_at"] = decision["decided_at"]
    return decision


def reconciliation(document: dict) -> dict:
    included = excluded = duplicates = unresolved = 0
    accounted = 0
    for row in document.get("components", []):
        review = row.get("review") or {}
        segmentation = review.get("segmentation", "unreviewed")
        if not review.get("confirmed"):
            unresolved += 1
            continue
        if segmentation == "duplicate":
            duplicates += 1
            continue
        if segmentation in {"merged", "incomplete"}:
            unresolved += 1
            continue
        accounted += 1
        if review.get("included"):
            included += 1
        else:
            excluded += 1
    manual = len(document.get("manual_pieces", []))
    for row in document.get("manual_pieces", []):
        _validate_review(row.get("review") or {})
        if not row["review"].get("confirmed"):
            unresolved += 1
            continue
        accounted += 1
        if row["review"].get("included"):
            included += 1
        else:
            excluded += 1
    expected = int(document.get("expected_physical_count", 0))
    segmentations_total = len(document.get("components", []))
    return {
        "expected": expected,
        "segmentations_total": segmentations_total,
        "segmentations_active": segmentations_total - duplicates,
        "segmentations_removed": duplicates,
        "accounted_physical": accounted,
        "included": included,
        "excluded": excluded,
        "duplicates": duplicates,
        "manual_missed": manual,
        "unresolved": unresolved,
        "reconciled": unresolved == 0 and accounted == expected,
    }


def lock_document(document: dict) -> dict:
    status = reconciliation(document)
    if status["unresolved"]:
        raise ValueError(
            f"cannot lock with {status['unresolved']} unresolved entries"
        )
    if status["accounted_physical"] != status["expected"]:
        raise ValueError(
            "cannot lock until physical count matches: "
            f"{status['accounted_physical']} != {status['expected']}"
        )
    document["locked"] = True
    document["locked_at"] = _now()
    document["updated_at"] = document["locked_at"]
    return status


def _piece_rows(document: dict) -> list[dict]:
    rows = []
    session_id = document["session_id"]
    for component in document.get("components", []):
        review = component["review"]
        if review.get("segmentation") == "duplicate":
            continue
        rows.append({
            "session_id": session_id,
            "piece_id": component["component_id"],
            "source": "detected",
            **copy.deepcopy(review),
        })
    for manual in document.get("manual_pieces", []):
        rows.append({
            "session_id": session_id,
            "piece_id": manual["manual_id"],
            "source": "manual-missed",
            **copy.deepcopy(manual["review"]),
        })
    return rows


def aggregate_documents(
    documents: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    if any(not document.get("locked") for document in documents):
        raise ValueError("all review documents must be locked before export")
    quantities: dict[tuple[str, str, str], int] = defaultdict(int)
    pieces = []
    total_expected = 0
    for document in documents:
        total_expected += int(document["expected_physical_count"])
        for row in _piece_rows(document):
            pieces.append(row)
            if row["included"]:
                key = (row["part_id"], row["part_name"], row["color"])
                quantities[key] += 1
    inventory = [
        {"part_id": part_id, "name": name, "color": color, "quantity": count}
        for (part_id, name, color), count in sorted(quantities.items())
    ]
    summary = {
        "sessions": len(documents),
        "expected_physical": total_expected,
        "included": sum(bool(row["included"]) for row in pieces),
        "excluded": sum(not bool(row["included"]) for row in pieces),
        "inventory_groups": len(inventory),
    }
    return inventory, pieces, summary


def atomic_write_json(path: Path, payload: dict | list) -> None:
    path = Path(path)
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
