"""Daily cost assumptions for the budget lines nobody quotes a price for.

Accommodation and activities have **retrieved prices**. Food and local transport do not:
no provider quotes what a traveller will spend on dinner. These numbers are therefore
*assumptions*, and the system says so — they carry `Provenance.SYNTHETIC` all the way to
the user, and `DataSourceInfo` separates them from the retrieved lines in the response.

The tempting alternative is to set food and transport equal to their budget shares. That
would be circular: the plan would then consume exactly 100 % of the budget by construction,
utilisation would always be 1.0, and "does this trip fit the budget?" would stop being a
question the system can answer. Independent estimates are what make the check meaningful.

In production these would come from historical spend data per destination and traveller
segment — the kind of dataset a travel company already has and this project does not.
"""

from __future__ import annotations

from dataclasses import dataclass

from travel_intel.domain.enums import Provenance
from travel_intel.domain.models import DataSourceInfo


@dataclass(frozen=True)
class DailyCostAssumptions:
    """Per-person, per-day planning estimates in EUR."""

    food: float
    transport: float
    rationale: str

    def __post_init__(self) -> None:
        if self.food <= 0 or self.transport <= 0:
            raise ValueError("daily cost assumptions must be positive")

    def source(self) -> DataSourceInfo:
        return DataSourceInfo(
            name=f"planning assumptions ({self.rationale})",
            provenance=Provenance.SYNTHETIC,
            record_count=2,
        )


DEFAULT_DAILY_COSTS = DailyCostAssumptions(
    food=35.0,
    transport=10.0,
    rationale="mid-range city trip: casual lunch and restaurant dinner, public transport only",
)
"""Fallback for destinations we have no specific figures for."""

DAILY_COSTS_BY_DESTINATION: dict[str, DailyCostAssumptions] = {
    "tokyo": DailyCostAssumptions(
        food=45.0,
        # ~EUR 6 for a subway day pass, the rest covering longer hops and the airport
        # transfer amortised across the stay.
        transport=12.0,
        rationale="Tokyo mid-range: konbini breakfast, casual lunch, izakaya dinner, subway",
    ),
}


def assumptions_for(destination_key: str) -> DailyCostAssumptions:
    """Assumptions for a destination, falling back to the generic city-trip figures."""
    return DAILY_COSTS_BY_DESTINATION.get(destination_key, DEFAULT_DAILY_COSTS)
