"""Deterministic attribution gate for inventory failures."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureEvidence:
    source_frame_contains_piece: bool
    selected_roi_contains_piece: bool
    segmentation_mask_correct: bool
    component_count_correct: bool
    identity_correct: bool


def attribute_failure(evidence: FailureEvidence) -> str:
    if not evidence.source_frame_contains_piece:
        return "capture_coverage"
    if not evidence.selected_roi_contains_piece:
        return "workspace_view_selection"
    if not evidence.segmentation_mask_correct:
        return "segmentation"
    if not evidence.component_count_correct:
        return "geometry_fusion"
    if not evidence.identity_correct:
        return "identification_evidence"
    return "no_failure"
