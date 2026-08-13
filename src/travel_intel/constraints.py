"""Constraint validation.

The reliability guarantee of this system lives in this module. Hard constraints are checked
here, in plain Python, against numbers computed in plain Python — *before* any text is
generated and independently of anything a model says. An LLM cannot talk a €2,700 plan into
a €2,500 budget, because by the time the LLM is called the plan has already been rejected.

Hard vs soft is a real distinction, not a severity label:

- **Hard** — the plan is not deliverable. Over budget, too small for the party, priced in
  the wrong currency. A plan with any of these is never returned.
- **Soft** — the plan is deliverable but the user should know. Over the accommodation
  allowance, a preference the itinerary could not cover, prices captured for other dates.
  These are warnings attached to a valid answer, not reasons to refuse one.
"""

from __future__ import annotations

from collections.abc import Sequence

from travel_intel.budget import DEFAULT_BUDGET_POLICY, BudgetPolicy
from travel_intel.domain.enums import BudgetCategory, Severity
from travel_intel.domain.models import (
    BudgetBreakdown,
    ConstraintReport,
    ConstraintViolation,
    ScoredAccommodation,
    TripRequest,
)
from travel_intel.planning.itinerary import ActivityPlan

BUDGET_EXCEEDED = "budget_exceeded"
INSUFFICIENT_CAPACITY = "insufficient_capacity"
CURRENCY_MISMATCH = "currency_mismatch"
ACCOMMODATION_OVER_ALLOWANCE = "accommodation_over_allowance"
PREFERENCE_NOT_COVERED = "preference_not_covered"
NO_ACTIVITIES_PLANNED = "no_activities_planned"
LOW_DATA_COMPLETENESS = "low_data_completeness"
STALE_DATA = "stale_data"

LOW_COMPLETENESS_THRESHOLD = 0.5
"""Below this we know less than half of what we would want about the recommended property."""


def validate_plan(
    request: TripRequest,
    recommended: ScoredAccommodation,
    budget: BudgetBreakdown,
    itinerary: ActivityPlan,
    *,
    retrieval_warnings: Sequence[str] = (),
    policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
) -> ConstraintReport:
    """Check a complete costed plan against everything the request actually demanded."""
    violations: list[ConstraintViolation] = [
        *_hard_violations(request, recommended, budget),
        *_soft_violations(request, recommended, budget, itinerary, retrieval_warnings, policy),
    ]
    return ConstraintReport(violations=tuple(violations))


def _hard_violations(
    request: TripRequest,
    recommended: ScoredAccommodation,
    budget: BudgetBreakdown,
) -> list[ConstraintViolation]:
    violations: list[ConstraintViolation] = []
    accommodation = recommended.accommodation

    if budget.total > request.budget_total:
        violations.append(
            ConstraintViolation(
                code=BUDGET_EXCEEDED,
                severity=Severity.HARD,
                message=(
                    f"the plan costs {budget.total:.2f} {request.currency.value}, "
                    f"over the stated budget of {request.budget_total:.2f}"
                ),
                observed=budget.total,
                limit=request.budget_total,
            )
        )

    if accommodation.max_occupancy < request.travelers:
        violations.append(
            ConstraintViolation(
                code=INSUFFICIENT_CAPACITY,
                severity=Severity.HARD,
                message=(
                    f"'{accommodation.name}' sleeps {accommodation.max_occupancy}, "
                    f"the party is {request.travelers}"
                ),
                observed=float(accommodation.max_occupancy),
                limit=float(request.travelers),
            )
        )

    if accommodation.currency is not request.currency:
        violations.append(
            ConstraintViolation(
                code=CURRENCY_MISMATCH,
                severity=Severity.HARD,
                message=(
                    f"'{accommodation.name}' is priced in {accommodation.currency.value}, "
                    f"the budget is in {request.currency.value}; "
                    "the two are not comparable without a conversion rate"
                ),
            )
        )

    return violations


def _soft_violations(
    request: TripRequest,
    recommended: ScoredAccommodation,
    budget: BudgetBreakdown,
    itinerary: ActivityPlan,
    retrieval_warnings: Sequence[str],
    policy: BudgetPolicy,
) -> list[ConstraintViolation]:
    violations: list[ConstraintViolation] = []
    allowance = policy.allowance(BudgetCategory.ACCOMMODATION, request.budget_total)

    if budget.accommodation > allowance:
        overrun = budget.accommodation - allowance
        violations.append(
            ConstraintViolation(
                code=ACCOMMODATION_OVER_ALLOWANCE,
                severity=Severity.SOFT,
                message=(
                    f"accommodation takes {budget.accommodation:.2f}, "
                    f"{overrun:.2f} over its {allowance:.2f} share; "
                    "the total still fits, so this is a trade-off, not a breach"
                ),
                observed=budget.accommodation,
                limit=allowance,
            )
        )

    if not itinerary.selected:
        violations.append(
            ConstraintViolation(
                code=NO_ACTIVITIES_PLANNED,
                severity=Severity.SOFT,
                message="no activity fit the preferences and the activities budget",
            )
        )

    for preference in itinerary.uncovered_preferences:
        violations.append(
            ConstraintViolation(
                code=PREFERENCE_NOT_COVERED,
                severity=Severity.SOFT,
                message=f"nothing in the itinerary covers '{preference.value}'",
            )
        )

    completeness = recommended.accommodation.data_completeness
    if completeness < LOW_COMPLETENESS_THRESHOLD:
        violations.append(
            ConstraintViolation(
                code=LOW_DATA_COMPLETENESS,
                severity=Severity.SOFT,
                message=(
                    f"only {completeness:.0%} of the decision-relevant fields are known for "
                    f"'{recommended.accommodation.name}'"
                ),
                observed=completeness,
                limit=LOW_COMPLETENESS_THRESHOLD,
            )
        )

    violations.extend(
        ConstraintViolation(code=STALE_DATA, severity=Severity.SOFT, message=warning)
        for warning in retrieval_warnings
    )

    return violations
