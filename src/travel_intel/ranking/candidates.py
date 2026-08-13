"""Candidate generation: the hard filters, applied before anything is scored.

Only rules that are *unarguably* disqualifying live here. "Too expensive to be sensible" is
a judgement and belongs in scoring; "cannot physically host the party" and "costs more than
the entire budget" are facts, and letting them reach the ranking would mean asking the
scorer to express, through weights, something the domain already knows for certain.

Every drop is counted by reason, so the funnel is inspectable rather than a black box.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from travel_intel.domain.errors import NoCandidatesError
from travel_intel.domain.models import Accommodation, TripRequest

REASON_CAPACITY = "insufficient_capacity"
REASON_UNAFFORDABLE = "exceeds_total_budget"


@dataclass(frozen=True)
class CandidateSet:
    """Survivors plus the funnel that produced them."""

    records: tuple[Accommodation, ...]
    retrieved: int
    dropped: dict[str, int]

    @property
    def kept(self) -> int:
        return len(self.records)


def generate_candidates(
    records: tuple[Accommodation, ...],
    request: TripRequest,
) -> CandidateSet:
    """Filter retrieved accommodations down to those that could actually be booked.

    Raises `NoCandidatesError` when nothing survives: an empty recommendation with a
    confident tone is worse than an explicit "your budget does not reach this destination".
    """
    kept: list[Accommodation] = []
    dropped: Counter[str] = Counter()

    for record in records:
        if record.max_occupancy < request.travelers:
            dropped[REASON_CAPACITY] += 1
            continue
        if record.total_cost(request.nights) > request.budget_total:
            dropped[REASON_UNAFFORDABLE] += 1
            continue
        kept.append(record)

    if not kept:
        raise NoCandidatesError(
            f"no accommodation in {request.destination} fits {request.travelers} travelers "
            f"within {request.budget_total:.0f} {request.currency.value} "
            f"for {request.nights} nights "
            f"(filtered out: {dict(dropped)} of {len(records)} retrieved)"
        )

    return CandidateSet(records=tuple(kept), retrieved=len(records), dropped=dict(dropped))
