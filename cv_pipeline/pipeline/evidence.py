"""Conflict-aware arbitration for independent identification evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from pipeline.identify import Identification


ACCEPTED_APPEARANCE_SCORE = 0.70
WEAK_APPEARANCE_SCORE = 0.50


def _candidate_record(candidate):
    return {
        "part_id": candidate.get("part_id"),
        "name": str(candidate.get("name", "")),
        "score": float(candidate.get("score", 0.0)),
    }


def aggregate_appearance_support(
    entries,
    accepted_score=ACCEPTED_APPEARANCE_SCORE,
    weak_score=WEAK_APPEARANCE_SCORE,
):
    """Separate top per-frame support from lower-ranked alternatives."""
    per_frame = {}
    for entry in entries:
        frame_id = int(entry["frame_id"])
        current = per_frame.get(frame_id)
        if current is None or float(entry.get("score", 0.0)) > float(
            current.get("score", 0.0)
        ):
            per_frame[frame_id] = entry

    support = {}

    def row_for(candidate):
        part_id = candidate.get("part_id")
        if part_id is None:
            return None
        part_id = str(part_id)
        return support.setdefault(part_id, {
            "part_id": part_id,
            "name": str(candidate.get("name", "")),
            "accepted_frame_ids": [],
            "weak_frame_ids": [],
            "top_frame_ids": [],
            "alternative_frame_ids": [],
            "best_score": 0.0,
        })

    for frame_id, entry in sorted(per_frame.items()):
        candidates = [
            _candidate_record(candidate)
            for candidate in entry.get("candidates", ())
            if candidate.get("part_id") is not None
        ]
        if entry.get("part_id") is not None:
            top = _candidate_record(entry)
            candidates = [
                top,
                *[
                    candidate for candidate in candidates
                    if candidate["part_id"] != top["part_id"]
                ],
            ]
        elif candidates:
            top = candidates[0]
        else:
            continue
        if top["score"] >= weak_score:
            row = row_for(top)
            row["top_frame_ids"].append(frame_id)
            state = (
                "accepted_frame_ids"
                if entry.get("part_id") is not None
                and top["score"] >= accepted_score
                else "weak_frame_ids"
            )
            row[state].append(frame_id)
            row["best_score"] = max(row["best_score"], top["score"])
        for candidate in candidates[1:]:
            if candidate["score"] < weak_score:
                continue
            row = row_for(candidate)
            row["alternative_frame_ids"].append(frame_id)
            row["best_score"] = max(row["best_score"], candidate["score"])

    counts = {
        part_id: len(row["top_frame_ids"])
        for part_id, row in support.items()
    }
    maximum = max(counts.values(), default=0)
    winners = sorted(
        part_id for part_id, count in counts.items() if count == maximum
    )
    return {
        "candidate_support": support,
        "unique_top_candidate": winners[0] if len(winners) == 1 else None,
        "unique_top_count": maximum,
    }


@dataclass(frozen=True)
class ArbitrationOutcome:
    identification: Identification
    action: str
    supporting_categories: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["identification"] = {
            "part_id": self.identification.part_id,
            "name": self.identification.name,
            "score": self.identification.score,
        }
        return payload


def _candidate_metric_state(dimension_evidence, part_id):
    if not dimension_evidence:
        return "unavailable"
    if dimension_evidence.get("evidence", {}).get("reliability") != "high":
        return "unavailable"
    for candidate in dimension_evidence.get("decision", {}).get(
        "candidate_scores", ()
    ):
        if candidate.get("part_id") != part_id:
            continue
        if candidate.get("modeled") is not True:
            return "unavailable"
        if candidate.get("fits") is True:
            return "supports"
        if candidate.get("fits") is False:
            return "rejects"
    return "unavailable"


def arbitrate_identification(
    current: Identification,
    dimension_action: str,
    stud_proposal: Identification,
    stud_action: str,
    identity_support: dict[str, int],
    dimension_evidence: dict | None = None,
    appearance_evidence: dict | None = None,
) -> ArbitrationOutcome:
    """Require two evidence categories before accepting a stud proposal."""
    if (
        stud_proposal.part_id == current.part_id
        and stud_proposal.name == current.name
    ) or stud_action not in {"corrected", "rescued", "demoted"}:
        return ArbitrationOutcome(
            current,
            stud_action,
            (),
            "stud evidence did not propose an identity change",
        )

    supports = ["stud"]
    if (
        stud_proposal.part_id is not None
        and identity_support.get(stud_proposal.part_id, 0) >= 2
    ):
        supports.append("multi_view_identity")

    if stud_proposal.part_id is None:
        return ArbitrationOutcome(
            current,
            "conflict_preserved",
            tuple(supports),
            "stud evidence cannot demote an identity by itself",
        )
    metric_state = _candidate_metric_state(
        dimension_evidence, stud_proposal.part_id
    )
    if metric_state == "rejects":
        return ArbitrationOutcome(
            current,
            "conflict_preserved",
            tuple(supports),
            "high-reliability metric evidence rejects the proposed identity",
        )
    if current.part_id is None:
        appearance = appearance_evidence or {}
        unique_candidate = appearance.get("unique_top_candidate")
        unique_count = int(appearance.get("unique_top_count", 0))
        if (
            stud_action == "rescued"
            and stud_proposal.part_id == unique_candidate
            and unique_count >= 2
            and metric_state == "supports"
        ):
            return ArbitrationOutcome(
                stud_proposal,
                "rescued",
                ("weak_multi_view_appearance", "metric", "stud"),
                "unique weak appearance leader agrees with metric and stud evidence",
            )
        return ArbitrationOutcome(
            current,
            "conflict_preserved",
            tuple(supports),
            "weak rescue requires a two-view unique leader plus metric and stud support",
        )
    if dimension_action in {"compatible", "corrected"}:
        return ArbitrationOutcome(
            current,
            "conflict_preserved",
            tuple(supports),
            "metric evidence supports the current identity",
        )
    if len(supports) < 2:
        return ArbitrationOutcome(
            current,
            "conflict_preserved",
            tuple(supports),
            "an identity change requires two evidence categories",
        )
    return ArbitrationOutcome(
        stud_proposal,
        "corrected" if current.part_id is not None else "rescued",
        tuple(supports),
        "stud and multi-view identity evidence agree",
    )
