"""Bounded early-exit processing for the end-of-scan showcase."""
from __future__ import annotations

from collections.abc import Callable
import json
import time
from pathlib import Path

import cv2
import numpy as np


def _box_area(box) -> int:
    x0, y0, x1, y1 = map(int, box)
    return max(0, x1 - x0) * max(0, y1 - y0)


def rank_showcase_indices(instances, limit: int = 12) -> list[int]:
    """Rank locally clean instances before any network identification call."""
    ranked = []
    for index, instance in enumerate(instances):
        best = instance.best
        gate = best.gate_provenance or {}
        if not best.complete or gate.get("boundary_zone"):
            continue
        confidence = 0.70 if best.confidence is None else float(best.confidence)
        focus = 0.0 if best.identification_focus is None else float(
            best.identification_focus
        )
        area = _box_area(best.box)
        if area < 1024:
            continue
        ranked.append(((confidence, focus, area, -index), index))
    ranked.sort(reverse=True)
    return [index for _, index in ranked[:max(0, int(limit))]]


def early_exit(
    attempt: Callable[[int], dict], *, min_safe: int = 3, max_views: int = 3
) -> dict:
    """Expand the authoritative view budget only while confirmations starve."""
    result = {"selected_count": 0, "selected": []}
    maximum = max(1, int(max_views))
    view_limits = [1] if maximum == 1 else [1, maximum]
    for view_limit in view_limits:
        result = attempt(view_limit)
        # Expand views only while genuinely-safe pieces are short; a
        # best-effort backfill does not count toward stopping early.
        strict = int(result.get("strict_count", result.get("selected_count", 0)))
        if strict >= int(min_safe):
            break
    return result


def write_selected_masks(result, payload: dict, output_dir: Path) -> None:
    """Persist exact source-frame crop masks for the compact review overlay."""
    output_dir = Path(output_dir)
    mask_dir = output_dir / "showcase-masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    for item in payload.get("selected", []):
        instance = result.instances[int(item["instance_id"])]
        frame_id = int(item["best_frame_id"])
        observation = next(
            (
                row for row in instance.observations
                if int(row.frame_id) == frame_id and row.mask_original is not None
            ),
            None,
        )
        if observation is None:
            raise ValueError(
                f"showcase instance {item['instance_id']} has no source mask"
            )
        x0, y0, x1, y1 = map(int, item["box_original"])
        mask = np.asarray(observation.mask_original, dtype=bool)[y0:y1, x0:x1]
        if not mask.any():
            raise ValueError(
                f"showcase instance {item['instance_id']} has an empty crop mask"
            )
        path = mask_dir / f"piece-{int(item['instance_id']):03d}.png"
        if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
            raise OSError(f"could not write showcase mask: {path}")
        item["mask"] = path.relative_to(output_dir).as_posix()


def run_fast_showcase(
    manifest_path: Path,
    output_dir: Path,
    *,
    target: int = 4,
    min_safe: int = 3,
    max_views: int = 3,
    identify_limit: int = 12,
    weights: Path = Path("models/lego_seg.pt"),
    inventory_csv: Path | None = None,
) -> Path:
    """Run bounded authoritative attempts and persist the normal UI artifacts."""
    from pipeline.multiview import consensus_color, scan_manifest
    from pipeline.inventory_constraints import InventoryCatalog
    from pipeline.scene_state import write_scene_state
    from pipeline.segdetect import YoloSegModel
    from pipeline.showcase import build_showcase, write_showcase

    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text())
    session_dir = Path(manifest["session_dir"])
    model = YoloSegModel(weights=weights)
    inventory_catalog = (
        None if inventory_csv is None else InventoryCatalog.load(inventory_csv)
    )
    target = min(4, max(0, int(target)))
    events_path = output_dir / "fast-showcase-events.jsonl"
    final_result = None
    final_payload = {"selected_count": 0, "selected": []}

    def emit(stage: str, **details) -> None:
        event = {"stage": stage, "timestamp": time.time(), **details}
        with events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")
        print("DEMO_EVENT " + json.dumps(event, sort_keys=True), flush=True)

    def attempt(view_limit: int) -> dict:
        nonlocal final_result, final_payload
        emit("finding_pieces", view_limit=view_limit)
        attempt_started = time.perf_counter()
        final_result = scan_manifest(
            manifest_path,
            detector="seg",
            seg_model=model,
            review_draft=True,
            identify_workers=12,
            identify_min_interval=0.0,
            view_limit=view_limit,
            identify_limit=identify_limit,
            # Only the highest-confidence masks per view can become confident
            # confirmations; bound the per-instance geometry work to them.
            max_detections_per_view=24,
        )
        # Surface the per-stage breakdown so a future regression is obvious in
        # the demo logs (segmentation vs identification vs fusion).
        stage_ms = {
            key: round(value * 1000.0)
            for key, value in final_result.timings_s.items()
        }
        emit(
            "matching_types",
            view_limit=view_limit,
            instances=len(final_result.instances),
            scan_stage_ms=stage_ms,
        )
        colors = [
            consensus_color(instance).name for instance in final_result.instances
        ]
        final_payload = build_showcase(
            final_result,
            target,
            colors=colors,
            single_view_min_top1=0.85,
            inventory_catalog=inventory_catalog,
        )
        emit(
            "attempt_complete",
            view_limit=view_limit,
            selected_count=final_payload["selected_count"],
            attempt_ms=round((time.perf_counter() - attempt_started) * 1000.0),
        )
        return final_payload

    emit("selecting_views")
    early_exit(attempt, min_safe=min_safe, max_views=max_views)
    if final_result is None:
        raise RuntimeError("fast showcase produced no processing attempt")
    emit("preparing_confirmations", selected_count=final_payload["selected_count"])
    colors = [consensus_color(instance).name for instance in final_result.instances]
    scene_path = write_scene_state(
        final_result,
        session_dir,
        output_dir,
        detector="seg",
        predicted_colors=colors,
    )
    showcase_path = write_showcase(
        final_result,
        output_dir,
        target,
        colors=colors,
        scene_state_ref=scene_path.name,
        single_view_min_top1=0.85,
        inventory_catalog=inventory_catalog,
    )
    payload = json.loads(showcase_path.read_text())
    write_selected_masks(final_result, payload, output_dir)
    showcase_path.write_text(json.dumps(payload, indent=2) + "\n")
    emit("showcase_ready", path=str(showcase_path))
    return showcase_path
