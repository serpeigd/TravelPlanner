"""Budget composition.

Four lines, two of them priced from retrieved data and two of them estimated. The split
matters more than the arithmetic: `accommodation` and `activities` are what the traveller
would actually be charged, `food` and `transport` are planning assumptions carrying
`Provenance.SYNTHETIC`.

All of it is plain arithmetic in Python. No LLM computes a number that ends up in a budget.
"""

from __future__ import annotations

from travel_intel.domain.models import BudgetBreakdown, TripRequest
from travel_intel.planning.assumptions import DailyCostAssumptions


def compose_budget(
    request: TripRequest,
    *,
    accommodation_cost: float,
    activities_cost: float,
    assumptions: DailyCostAssumptions,
) -> BudgetBreakdown:
    """Assemble the four budget lines for a trip.

    Food and transport are charged for `nights` days rather than the calendar span: the
    arrival and departure days are partial and travelling, so counting both in full would
    inflate the estimate by roughly a day for every trip.
    """
    person_days = request.travelers * request.nights
    return BudgetBreakdown(
        budget_total=request.budget_total,
        accommodation=accommodation_cost,
        activities=activities_cost,
        food=assumptions.food * person_days,
        transport=assumptions.transport * person_days,
    )
