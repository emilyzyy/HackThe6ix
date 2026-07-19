"""Adapt a conservative showcase into the compact confirmation workspace."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from pipeline.inventory_constraints import InventoryCatalog
from training.inventory_review import atomic_write_json, new_review_document


def write_showcase_workspace(
    showcase_path: Path,
    session_dir: Path,
    output_dir: Path,
    fixed_inventory: Path | None,
    generator_url: str | None = None,
    inventory_csv: Path | None = None,
) -> Path:
    showcase_path = Path(showcase_path).resolve()
    session_dir = Path(session_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = json.loads(showcase_path.read_text()).get("selected", [])
    catalog = InventoryCatalog.load(inventory_csv) if inventory_csv else None

    source_frames = {}
    components = []
    for index, item in enumerate(selected):
        frame_id = int(item["best_frame_id"])
        source = (session_dir / item["rgb"]).resolve()
        if not source.is_relative_to(session_dir) or not source.is_file():
            raise ValueError(f"missing showcase source frame: {item.get('rgb')}")
        frame_rel = Path("review-assets") / "frames" / source.name
        frame_path = output_dir / frame_rel
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        if not frame_path.exists():
            shutil.copy2(source, frame_path)
        image = cv2.imread(str(frame_path))
        if image is None:
            raise ValueError(f"invalid showcase source frame: {source}")
        source_frames[frame_id] = {
            "frame_id": frame_id,
            "image": frame_rel.as_posix(),
            "size_wh": [int(image.shape[1]), int(image.shape[0])],
        }

        box = [int(value) for value in item["box_original"]]
        x0, y0, x1, y1 = box
        mask_source = (showcase_path.parent / item["mask"]).resolve()
        if (
            not mask_source.is_relative_to(showcase_path.parent)
            or not mask_source.is_file()
        ):
            raise ValueError(f"missing showcase mask: {item.get('mask')}")
        component_id = f"showcase-{index:03d}"
        component_dir = (
            output_dir / "review-assets" / "components" / component_id
        )
        component_dir.mkdir(parents=True, exist_ok=True)
        mask_rel = component_dir.relative_to(output_dir) / "mask.png"
        shutil.copy2(mask_source, output_dir / mask_rel)
        crop_rel = component_dir.relative_to(output_dir) / "crop.jpg"
        cv2.imwrite(str(output_dir / crop_rel), image[y0:y1, x0:x1])
        color = item.get("color") if item.get("color_reliable") else ""
        components.append({
            "component_id": component_id,
            "source_anchor": {
                "frame_id": frame_id,
                "shape_id": item.get("source_instance_id"),
                "box_original": box,
            },
            "prediction": {
                "part_id": item.get("part_id"),
                "part_name": item.get("part_name") or "unknown brick",
                "color": color or "",
                "score": float(item.get("top1_score") or 0.0),
            },
            "candidates": [],
            "assets": {"views": [{
                "frame_id": frame_id,
                "crop_box_original": box,
                "crop": crop_rel.as_posix(),
                "mask": mask_rel.as_posix(),
            }]},
        })

    document = new_review_document(
        session_dir.name, max(1, len(components)), components
    )
    document.update({
        "showcase_only": True,
        "run_dir": str(output_dir),
        "primary_frame_id": (
            int(components[0]["source_anchor"]["frame_id"])
            if components else None
        ),
        "source_frames": list(source_frames.values()),
        "draft_component_count": len(components),
        "fixed_inventory": (
            None if fixed_inventory is None else str(Path(fixed_inventory).resolve())
        ),
        "generator_url": generator_url,
        "inventory_piece_types": (
            list(catalog.allowed_names) if catalog is not None else []
        ),
    })
    review_path = output_dir / "review.json"
    atomic_write_json(review_path, document)
    workspace_path = output_dir / "workspace.json"
    atomic_write_json(workspace_path, {
        "schema_version": 1,
        "name": f"Fast showcase {session_dir.name}",
        "expected_physical_total": len(components),
        "sessions": [{
            "session_id": session_dir.name,
            "expected_count": len(components),
            "review_path": str(review_path),
        }],
    })
    return workspace_path
