"""Conservative dimension consistency for regular rectangular LEGO names."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from pipeline.footprint import MetricSilhouette
from pipeline.identify import Identification


MIN_CANDIDATE_SCORE = 0.50
ABSOLUTE_TOLERANCE_MM = 8.0
FRACTIONAL_TOLERANCE = 0.35

_REGULAR_NAME = re.compile(r"^(Brick|Plate) (\d+) x (\d+)$")


@dataclass(frozen=True)
class RegularDimensions:
    kind: str
    studs_short: int
    studs_long: int
    short_mm: float
    long_mm: float

    @property
    def studs(self) -> tuple[int, int]:
        return self.studs_short, self.studs_long

    @property
    def dimensions_mm(self) -> tuple[float, float]:
        return self.short_mm, self.long_mm


@dataclass(frozen=True)
class CandidateDimensionScore:
    part_id: str | None
    name: str
    score: float
    modeled: bool
    expected_mm: tuple[float, float] | None
    error_mm: tuple[float, float] | None
    tolerance_mm: tuple[float, float] | None
    fits: bool | None
    normalized_error: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DimensionDecision:
    identification: Identification
    action: str
    flag: str | None
    candidate_scores: tuple[CandidateDimensionScore, ...]
    proposed_family: str | None
    reason: str
    tolerance_policy: dict

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "flag": self.flag,
            "candidate_scores": [
                candidate.to_dict() for candidate in self.candidate_scores
            ],
            "proposed_family": self.proposed_family,
            "reason": self.reason,
            "tolerance_policy": self.tolerance_policy,
            "selected": {
                "part_id": self.identification.part_id,
                "name": self.identification.name,
                "score": self.identification.score,
            },
        }


def parse_regular_dimensions(name: str) -> RegularDimensions | None:
    match = _REGULAR_NAME.fullmatch(name or "")
    if not match:
        return None
    first, second = sorted((int(match.group(2)), int(match.group(3))))
    return RegularDimensions(
        kind=match.group(1),
        studs_short=first,
        studs_long=second,
        short_mm=first * 8.0 - 0.2,
        long_mm=second * 8.0 - 0.2,
    )


def _tolerance(expected_mm: float) -> float:
    return max(ABSOLUTE_TOLERANCE_MM, FRACTIONAL_TOLERANCE * expected_mm)


def _score_candidate(
    candidate: dict,
    metric: MetricSilhouette,
) -> CandidateDimensionScore:
    dimensions = parse_regular_dimensions(str(candidate.get("name", "")))
    if dimensions is None:
        return CandidateDimensionScore(
            part_id=candidate.get("part_id"),
            name=str(candidate.get("name", "")),
            score=float(candidate.get("score", 0.0)),
            modeled=False,
            expected_mm=None,
            error_mm=None,
            tolerance_mm=None,
            fits=None,
            normalized_error=None,
        )
    expected = dimensions.dimensions_mm
    measured = tuple(sorted((float(metric.short_mm), float(metric.long_mm))))
    errors = tuple(abs(actual - nominal) for actual, nominal in zip(
        measured, expected
    ))
    tolerances = tuple(_tolerance(value) for value in expected)
    normalized = sum(
        error / tolerance for error, tolerance in zip(errors, tolerances)
    )
    return CandidateDimensionScore(
        part_id=candidate.get("part_id"),
        name=str(candidate["name"]),
        score=float(candidate.get("score", 0.0)),
        modeled=True,
        expected_mm=expected,
        error_mm=errors,
        tolerance_mm=tolerances,
        fits=all(
            error <= tolerance
            for error, tolerance in zip(errors, tolerances)
        ),
        normalized_error=normalized,
    )


def _candidate_rows(identification: Identification) -> list[dict]:
    rows = [{
        "part_id": identification.part_id,
        "name": identification.name,
        "score": identification.score,
    }]
    keys = {(identification.part_id, identification.name)}
    for candidate in identification.candidates:
        key = (candidate.get("part_id"), candidate.get("name"))
        if key in keys:
            continue
        keys.add(key)
        rows.append(dict(candidate))
    return rows


def _proposed_family(
    metric: MetricSilhouette,
    raw_dimensions: RegularDimensions | None,
) -> str | None:
    if metric.short_mm is None or metric.long_mm is None:
        return None
    measured = tuple(sorted((metric.short_mm, metric.long_mm)))
    proposed = []
    for index, value in enumerate(measured):
        if raw_dimensions is not None:
            expected = raw_dimensions.dimensions_mm[index]
            if abs(value - expected) <= _tolerance(expected):
                proposed.append(raw_dimensions.studs[index])
                continue
        proposed.append(max(1, int(round((value + 0.2) / 8.0))))
    short, long = sorted(proposed)
    return f"{short} x {long}"


def _from_score(
    selected: CandidateDimensionScore,
    base: Identification,
) -> Identification:
    return Identification(
        selected.part_id,
        selected.name,
        None,
        selected.score,
        candidates=base.candidates,
    )


def resolve_dimensions(
    identification: Identification,
    metric: MetricSilhouette,
) -> DimensionDecision:
    policy = {
        "absolute_tolerance_mm": ABSOLUTE_TOLERANCE_MM,
        "fractional_tolerance": FRACTIONAL_TOLERANCE,
        "source": "conservative_metric_silhouette_audit_bound",
    }
    raw_dimensions = parse_regular_dimensions(identification.name)
    proposed = _proposed_family(metric, raw_dimensions)
    if metric.reliability in {"low", "unavailable"}:
        return DimensionDecision(
            identification,
            "insufficient_metric_evidence",
            None,
            (),
            proposed,
            f"metric silhouette reliability is {metric.reliability}",
            policy,
        )
    if metric.short_mm is None or metric.long_mm is None:
        return DimensionDecision(
            identification,
            "insufficient_metric_evidence",
            None,
            (),
            None,
            "metric silhouette has no aggregate dimensions",
            policy,
        )

    scores = tuple(
        _score_candidate(candidate, metric)
        for candidate in _candidate_rows(identification)
    )
    raw_score = scores[0]
    if not raw_score.modeled:
        return DimensionDecision(
            identification,
            "unmodeled_candidate",
            None,
            scores,
            proposed,
            "raw name is outside exact Brick/Plate N x M grammar",
            policy,
        )
    if raw_score.fits:
        return DimensionDecision(
            identification,
            "compatible",
            None,
            scores,
            proposed,
            "raw candidate is within conservative projection tolerance",
            policy,
        )
    if metric.reliability == "medium":
        return DimensionDecision(
            identification,
            "flagged",
            "dimension_mismatch",
            scores,
            proposed,
            "medium-reliability mismatch cannot rerank candidates",
            policy,
        )

    compatible = [
        score for score in scores[1:]
        if score.modeled and score.fits
        and score.score >= MIN_CANDIDATE_SCORE
    ]
    if compatible:
        selected = min(
            compatible,
            key=lambda score: (
                float(score.normalized_error), -score.score, score.name
            ),
        )
        return DimensionDecision(
            _from_score(selected, identification),
            "corrected",
            None,
            scores,
            proposed,
            "high-reliability dimensions favor an existing API candidate",
            policy,
        )

    fitting_sides = sum(
        error <= tolerance
        for error, tolerance in zip(
            raw_score.error_mm, raw_score.tolerance_mm
        )
    )
    flag = (
        "possible_non_canonical_pose" if fitting_sides == 1
        else "dimension_mismatch"
    )
    return DimensionDecision(
        identification,
        "flagged",
        flag,
        scores,
        proposed,
        "all supplied modeled candidates conflict; raw result preserved",
        policy,
    )
