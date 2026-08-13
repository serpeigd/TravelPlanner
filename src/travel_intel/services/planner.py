"""Assembling a valid, costed trip out of ranked candidates.

The ranking says which accommodation is *best*; this module answers whether the resulting
trip is *deliverable*, and that is a different question. The highest-ranked property can
still push the total over budget once food, transport and activities are added, so the
planner walks the ranking in order and returns the first option whose complete plan passes
every hard constraint.

That walk is the mechanism behind the reliability claim. A budget-violating plan is not
"discouraged by a prompt" or "penalised in a score" — it is never constructed, because
`validate_plan` sits between the ranking and everything downstream, including the LLM.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from travel_intel.budget import DEFAULT_BUDGET_POLICY, BudgetPolicy
from travel_intel.constraints import validate_plan
from travel_intel.domain.enums import Severity
from travel_intel.domain.errors import NoCandidatesError
from travel_intel.domain.models import (
    Activity,
    BudgetBreakdown,
    ConstraintReport,
    ScoredAccommodation,
    TripRequest,
)
from travel_intel.planning.assumptions import DailyCostAssumptions, assumptions_for
from travel_intel.planning.costs import compose_budget
from travel_intel.planning.itinerary import ActivityPlan, build_itinerary
from travel_intel.retrieval.base import destination_key

MAX_ALTERNATIVES = 4


@dataclass(frozen=True)
class RejectedOption:
    """A ranked option whose full plan broke a hard constraint."""

    accommodation_id: str
    name: str
    rank: int
    codes: tuple[str, ...]


@dataclass(frozen=True)
class PlannedTrip:
    request: TripRequest
    recommended: ScoredAccommodation
    alternatives: tuple[ScoredAccommodation, ...]
    itinerary: ActivityPlan
    budget: BudgetBreakdown
    constraints: ConstraintReport
    assumptions: DailyCostAssumptions
    rejected: tuple[RejectedOption, ...]
    """Options the ranking preferred but the constraints refused, kept for explainability."""


def plan_trip(
    ranked: tuple[ScoredAccommodation, ...],
    activities: tuple[Activity, ...],
    request: TripRequest,
    *,
    policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
    retrieval_warnings: Sequence[str] = (),
    max_alternatives: int = MAX_ALTERNATIVES,
) -> PlannedTrip:
    """Return the best-ranked trip that satisfies every hard constraint.

    The itinerary is built once and shared across candidates: with no activity coordinates
    in the data, the choice of activities genuinely does not depend on which hotel is
    booked. Pretending otherwise would be a fabricated dependency (see docs/data.md).

    Raises `NoCandidatesError` when nothing validates — the alternative would be returning
    a plan the system has just proven to be impossible.
    """
    if not ranked:
        raise NoCandidatesError("nothing to plan: the ranking is empty")

    assumptions = assumptions_for(destination_key(request.destination))
    itinerary = build_itinerary(activities, request, policy=policy)

    valid: list[tuple[ScoredAccommodation, BudgetBreakdown, ConstraintReport]] = []
    rejected: list[RejectedOption] = []

    for option in ranked:
        budget = compose_budget(
            request,
            accommodation_cost=option.total_cost,
            activities_cost=itinerary.total_cost,
            assumptions=assumptions,
        )
        report = validate_plan(
            request,
            option,
            budget,
            itinerary,
            retrieval_warnings=retrieval_warnings,
            policy=policy,
        )
        if report.is_valid:
            valid.append((option, budget, report))
        else:
            rejected.append(
                RejectedOption(
                    accommodation_id=option.accommodation.id,
                    name=option.accommodation.name,
                    rank=option.rank,
                    codes=tuple(v.code for v in report.violations if v.severity is Severity.HARD),
                )
            )

    if not valid:
        raise NoCandidatesError(_no_valid_plan_message(request, rejected))

    recommended, budget, report = valid[0]
    alternatives = tuple(option for option, _, _ in valid[1 : 1 + max_alternatives])
    return PlannedTrip(
        request=request,
        recommended=recommended,
        alternatives=alternatives,
        itinerary=itinerary,
        budget=budget,
        constraints=report,
        assumptions=assumptions,
        rejected=tuple(rejected),
    )


def _no_valid_plan_message(request: TripRequest, rejected: list[RejectedOption]) -> str:
    reasons = Counter(code for option in rejected for code in option.codes)
    dominant = ", ".join(f"{code} ({count})" for code, count in reasons.most_common())
    return (
        f"no plan for {request.destination} satisfies every hard constraint within "
        f"{request.budget_total:.0f} {request.currency.value}: "
        f"{len(rejected)} candidates rejected [{dominant}]"
    )
