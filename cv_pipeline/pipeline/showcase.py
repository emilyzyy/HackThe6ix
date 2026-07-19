"""Showcase selection: surface the few SAFEST identified pieces for the user
to physically confirm at the end of a scan.

This inverts the old "surface the uncertain ones" logic. Abstention is cheap
(~100 pieces, we need <=5), so the funnel is brutally conservative: a cheap
local prefilter hard-vetoes anything with a real warning flag, boundary crop,
unusable view, or missing identity; survivors are then scored on identification
quality (Brickognize top-1 confidence, top-1-vs-alternative margin, multi-view
agreement, YOLO confidence, crop quality) and only the top N are kept. Fewer
than N — including zero — is a correct outcome, not a failure.

Every signal used here is one the pipeline genuinely produces. No new
Brickognize calls are made: identities are reused from the completed scan, so
the "cheap local prefilter before identification" is satisfied trivially in the
demo path while the funnel structure stays valid if run standalone.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from pipeline.inventory_constraints import InventoryCatalog

# --- funnel thresholds (documented, not tuned against inventory ordering) ---
# Floors sit deliberately ABOVE the 0.70 identification-acceptance threshold so
# "safest" means clearly-accepted, not marginally-accepted.
SAFE_MIN_TOP1 = 0.80
SAFE_MIN_MARGIN = 0.10
MULTIVIEW_MIN_AGREE = 2
# A single exceptionally clean canonical view MAY still qualify — we do not make
# >=2 views mandatory, or we would only ever favour pieces that stayed in frame
# longest. It just has to clear a higher bar.
SINGLE_VIEW_MIN_TOP1 = 0.90
SINGLE_VIEW_MIN_MARGIN = 0.15
SEG_CONF_MIN = 0.50
MIN_CROP_AREA_PX = 1024  # 32x32; only rejects degenerate crops, keeps 1x1s

# Real warning flags = hard vetoes, never scoring inputs. Note: family_reviewed
# and view_diversity_unknown/insufficient_view_diversity are NOT vetoes — they
# fire on nearly every piece and mean "a review path ran", not "unsafe".
VETO_INSTANCE_FLAGS = frozenset({
    "identity_view_disagreement",
    "dimension_mismatch",
    "possible_non_canonical_pose",
})
# Per-view crop-quality flags on the chosen identification view.
VETO_VIEW_FLAGS = frozenset({
    "possible_shadow",
    "possible_specular",
    "oblique_view",
    "boundary_view",
})


@dataclass
class ShowcaseCandidate:
    instance_id: int
    part_id: str | None
    part_name: str | None
    top1_score: float
    top2_score: float
    agreement: int
    num_views: int
    seg_confidence: float | None
    complete: bool
    boundary_zone: bool
    crop_area_px: int
    focus: float | None
    instance_flags: tuple[str, ...]
    view_flags: tuple[str, ...]
    color: str | None
    color_reliable: bool
    # display/reference payload passed straight through to the UI
    refs: dict = field(default_factory=dict)

    @property
    def margin(self) -> float:
        return self.top1_score - self.top2_score

    @property
    def single_view(self) -> bool:
        return self.agreement < MULTIVIEW_MIN_AGREE


def _veto_reasons(candidate: ShowcaseCandidate) -> list[str]:
    """Cheap local hard vetoes. Any reason here disqualifies the candidate."""
    reasons = []
    if candidate.part_id is None:
        reasons.append("unidentified")
    for flag in candidate.instance_flags:
        if flag in VETO_INSTANCE_FLAGS:
            reasons.append(f"flag:{flag}")
    for flag in candidate.view_flags:
        if flag in VETO_VIEW_FLAGS:
            reasons.append(f"crop_flag:{flag}")
    if candidate.boundary_zone:
        reasons.append("boundary_crop")
    if not candidate.complete:
        reasons.append("incomplete_crop")
    if candidate.crop_area_px < MIN_CROP_AREA_PX:
        reasons.append("tiny_crop")
    return reasons


def _quality_reasons(
    candidate: ShowcaseCandidate,
    *,
    single_view_min_top1: float = SINGLE_VIEW_MIN_TOP1,
) -> list[str]:
    """Identification-quality floors applied to survivors of the prefilter."""
    reasons = []
    if candidate.top1_score < SAFE_MIN_TOP1:
        reasons.append("low_confidence")
    if candidate.margin < SAFE_MIN_MARGIN:
        reasons.append("low_margin")
    if candidate.single_view:
        if (
            candidate.top1_score < single_view_min_top1
            or candidate.margin < SINGLE_VIEW_MIN_MARGIN
        ):
            reasons.append("single_view_not_exceptional")
    if (
        candidate.seg_confidence is not None
        and candidate.seg_confidence < SEG_CONF_MIN
    ):
        reasons.append("low_seg_confidence")
    return reasons


def _safety_score(candidate: ShowcaseCandidate) -> tuple[float, dict]:
    """Rank survivors. Margin is weighted highest (it matters more than raw
    score); multi-view agreement and crop quality break ties."""
    margin_score = min(candidate.margin, 0.5) / 0.5
    conf_score = candidate.top1_score
    agree_score = 1.0 if not candidate.single_view else 0.5
    seg_score = (
        candidate.seg_confidence if candidate.seg_confidence is not None
        else 0.7  # cv detector has no YOLO confidence; neutral, not a bonus
    )
    quality_score = (0.5 if candidate.complete else 0.0) + (
        0.5 if not candidate.boundary_zone else 0.0
    )
    score = (
        0.35 * margin_score
        + 0.30 * conf_score
        + 0.15 * agree_score
        + 0.12 * seg_score
        + 0.08 * quality_score
    )
    components = {
        "margin_score": round(margin_score, 4),
        "confidence_score": round(conf_score, 4),
        "agreement_score": agree_score,
        "seg_confidence_score": round(float(seg_score), 4),
        "crop_quality_score": quality_score,
        "safety_score": round(score, 4),
    }
    return score, components


def _selection_reasons(candidate: ShowcaseCandidate) -> list[str]:
    reasons = [
        f"brickognize top1 {candidate.top1_score:.2f} "
        f"(margin {candidate.margin:.2f} over next part)",
    ]
    if not candidate.single_view:
        reasons.append(
            f"{candidate.agreement} of {candidate.num_views} views agree on "
            f"{candidate.part_name}"
        )
    else:
        reasons.append(
            f"single exceptional view ({candidate.part_name}) clears the "
            "strict single-view bar"
        )
    if candidate.seg_confidence is not None:
        reasons.append(f"YOLO confidence {candidate.seg_confidence:.2f}")
    if candidate.complete and not candidate.boundary_zone:
        reasons.append("clean fully-contained crop")
    return reasons


def select_showcase(
    candidates: list[ShowcaseCandidate], n: int = 5,
    *,
    single_view_min_top1: float = SINGLE_VIEW_MIN_TOP1,
) -> dict:
    """Run the full funnel and return the selected pieces + rejection audit."""
    survivors = []
    rejected = []
    for candidate in candidates:
        veto = _veto_reasons(candidate)
        if veto:
            rejected.append((candidate, "prefilter", veto, None))
            continue
        quality = _quality_reasons(
            candidate, single_view_min_top1=single_view_min_top1
        )
        if quality:
            rejected.append((candidate, "quality_floor", quality, None))
            continue
        score, components = _safety_score(candidate)
        survivors.append((score, components, candidate))

    survivors.sort(key=lambda item: item[0], reverse=True)
    selected = survivors[:max(0, n)]
    # Survivors that were safe but ranked below the cut are recorded too.
    for score, components, candidate in survivors[max(0, n):]:
        rejected.append((candidate, "below_cut", ["outside_top_n"], components))

    selected_payload = []
    for score, components, candidate in selected:
        selected_payload.append({
            "instance_id": candidate.instance_id,
            "part_id": candidate.part_id,
            "part_name": candidate.part_name,
            "top1_score": candidate.top1_score,
            "top2_score": candidate.top2_score,
            "margin": round(candidate.margin, 4),
            "agreement": candidate.agreement,
            "num_views": candidate.num_views,
            "seg_confidence": candidate.seg_confidence,
            "color": candidate.color,
            "color_reliable": candidate.color_reliable,
            # The confirmation question is part-identity only; colour is known
            # shaky, so it rides in the payload for optional UI use, not the ask.
            "question": f"I think this is a {candidate.part_name} — is that right?",
            "safety": components,
            "selection_reasons": _selection_reasons(candidate),
            **candidate.refs,
        })

    strict_count = len(selected_payload)  # pieces that cleared the strict bar
    # Best-effort fallback: if the strict funnel cannot fill the tiny demo
    # review, surface the highest-confidence remaining IDENTIFIED pieces
    # (real model output, real masks) up to n.  ``strict_count`` stays separate
    # so early-exit still expands views before accepting this fallback.
    if len(selected_payload) < max(0, n):
        selected_ids = {row["instance_id"] for row in selected_payload}
        backfill = sorted(
            (
                c for c in candidates
                if c.instance_id not in selected_ids
                and c.part_id is not None
                and c.refs.get("rgb")
            ),
            key=lambda c: (c.top1_score, c.margin),
            reverse=True,
        )[:max(0, n - len(selected_payload))]
        for candidate in backfill:
            _, components = _safety_score(candidate)
            selected_payload.append({
                "instance_id": candidate.instance_id,
                "part_id": candidate.part_id,
                "part_name": candidate.part_name,
                "top1_score": candidate.top1_score,
                "top2_score": candidate.top2_score,
                "margin": round(candidate.margin, 4),
                "agreement": candidate.agreement,
                "num_views": candidate.num_views,
                "seg_confidence": candidate.seg_confidence,
                "color": candidate.color,
                "color_reliable": candidate.color_reliable,
                "question": f"I think this is a {candidate.part_name} — is that right?",
                "safety": components,
                "selection_reasons": ["best-effort fallback (no piece cleared the strict bar)"],
                "low_confidence": True,
                **candidate.refs,
            })

    # Strongest rejects first: highest top1 among rejected, so the audit shows
    # the near-misses rather than the obvious junk.
    rejected.sort(key=lambda item: item[0].top1_score, reverse=True)
    rejected_payload = [
        {
            "instance_id": candidate.instance_id,
            "part_id": candidate.part_id,
            "part_name": candidate.part_name,
            "top1_score": candidate.top1_score,
            "margin": round(candidate.margin, 4),
            "stage": stage,
            "rejection_reasons": reasons,
            **({"safety": components} if components else {}),
        }
        for candidate, stage, reasons, components in rejected
    ]

    return {
        "requested": n,
        "selected_count": len(selected_payload),
        "strict_count": strict_count,
        "selected": selected_payload,
        "rejected": rejected_payload,
    }


def _margin_over_alternative(brickognize: dict, raw: dict) -> tuple[float, float]:
    """(top1, top2) where top1 is the SURFACED answer's score and top2 is the
    best candidate naming a DIFFERENT part — the real ambiguity margin."""
    final_pid = brickognize.get("part_id")
    top1 = float(brickognize.get("score") or 0.0)
    alternatives = [
        float(candidate.get("score") or 0.0)
        for candidate in raw.get("candidates", [])
        if candidate.get("part_id") != final_pid
    ]
    top2 = max(alternatives) if alternatives else 0.0
    return top1, top2


def _chosen_view_flags(instance, id_crop) -> tuple[str, ...]:
    """Crop-quality flags of the exact view whose crop the user will see."""
    frame_id = id_crop.get("frame_id")
    for observation in instance.observations:
        if observation.frame_id == frame_id:
            evidence = observation.color_evidence or {}
            return tuple(evidence.get("flags", []))
    return ()


def build_showcase(
    result,
    n: int = 5,
    *,
    colors: list[str] | None = None,
    single_view_min_top1: float = SINGLE_VIEW_MIN_TOP1,
    inventory_catalog: InventoryCatalog | None = None,
) -> dict:
    """Extract candidates from a completed scan and run the funnel."""
    colors = colors or []
    candidates: list[ShowcaseCandidate] = []
    for instance_id, (instance, id_crop) in enumerate(
        zip(result.instances, result.id_crops)
    ):
        brickognize = id_crop.get("brickognize", {})
        raw = id_crop.get("brickognize_raw", {})
        if inventory_catalog is None:
            top1, top2 = _margin_over_alternative(brickognize, raw)
        else:
            allowed = inventory_catalog.ranked_candidates(brickognize, raw)
            brickognize = allowed[0] if allowed else {}
            top1 = float(brickognize.get("score") or 0.0)
            alternatives = [
                float(row.get("score") or 0.0)
                for row in allowed[1:]
                if row.get("part_id") != brickognize.get("part_id")
            ]
            top2 = max(alternatives) if alternatives else 0.0
        support = id_crop.get("identity_support", {}) or {}
        agreement = int(support.get(brickognize.get("part_id"), 0))
        gate = id_crop.get("gate", {}) or {}
        box_original = id_crop.get("box_original")
        area = (
            (box_original[2] - box_original[0]) * (box_original[3] - box_original[1])
            if box_original else 0
        )
        color = colors[instance_id] if instance_id < len(colors) else None
        color_reliable = bool(color) and not str(color).startswith("ambiguous(")
        candidates.append(ShowcaseCandidate(
            instance_id=instance_id,
            part_id=brickognize.get("part_id"),
            part_name=brickognize.get("name"),
            top1_score=top1,
            top2_score=top2,
            agreement=agreement,
            num_views=len(instance.observations),
            seg_confidence=id_crop.get("segmentation_confidence"),
            complete=bool(instance.best.complete),
            boundary_zone=bool(gate.get("boundary_zone", False)),
            crop_area_px=int(area),
            focus=instance.best.identification_focus,
            instance_flags=tuple(id_crop.get("flags", [])),
            view_flags=_chosen_view_flags(instance, id_crop),
            color=color,
            color_reliable=color_reliable,
            refs={
                "best_frame_id": id_crop.get("frame_id"),
                "id_source": id_crop.get("source"),
                "rgb": id_crop.get("rgb"),
                "box_original": box_original,
                "box_canvas": list(instance.best.box),
                "source_instance_id": id_crop.get("source_instance_id"),
                "observation_frames": [
                    observation.frame_id for observation in instance.observations
                ],
            },
        ))
    return select_showcase(
        candidates, n, single_view_min_top1=single_view_min_top1
    )


def write_showcase(
    result, output_dir, n: int = 5, *, colors: list[str] | None = None,
    scene_state_ref: str = "scene_state.json",
    single_view_min_top1: float = SINGLE_VIEW_MIN_TOP1,
    inventory_catalog: InventoryCatalog | None = None,
) -> Path:
    payload = build_showcase(
        result,
        n,
        colors=colors,
        single_view_min_top1=single_view_min_top1,
        inventory_catalog=inventory_catalog,
    )
    payload["scene_state"] = scene_state_ref  # viewer joins by instance_id
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "showcase.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
