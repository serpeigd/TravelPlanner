"""The golden set.

Ten requests chosen to exercise the decisions the system actually has to make, including the
ones where the right answer is **to refuse**. A suite of only happy paths measures nothing:
the interesting question is not whether a comfortable budget produces a plan, it is whether
an impossible one produces an honest refusal instead of a confident fiction.

Each scenario states what it is probing, so a failure says what broke rather than just which
number moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum

from travel_intel.domain.enums import Preference
from travel_intel.domain.models import TripRequest

START = date(2026, 9, 10)
END = date(2026, 9, 17)


class Expected(StrEnum):
    PLAN = "plan"
    """A complete, valid plan must come back."""
    REFUSAL = "refusal"
    """No plan can satisfy the request, and the system must say so rather than improvise."""


@dataclass(frozen=True)
class Scenario:
    id: str
    request: TripRequest
    expected: Expected
    probes: str
    """What this case is testing. Read it when the scenario fails."""


def _request(**overrides: object) -> TripRequest:
    payload: dict[str, object] = {
        "destination": "Tokyo",
        "start_date": START,
        "end_date": END,
        "travelers": 2,
        "budget_total": 2500,
        "preferences": (Preference.FOOD, Preference.CULTURE, Preference.NATURE),
    }
    payload.update(overrides)
    return TripRequest.model_validate(payload)


GOLDEN_SET: tuple[Scenario, ...] = (
    Scenario(
        id="reference",
        request=_request(),
        expected=Expected.PLAN,
        probes="the demo case: comfortable budget, three preferences, all satisfiable",
    ),
    Scenario(
        id="tight-budget",
        request=_request(budget_total=2100),
        expected=Expected.PLAN,
        probes="options the ranking prefers become undeliverable once the full plan is costed",
    ),
    Scenario(
        id="very-tight-budget",
        request=_request(budget_total=1600),
        expected=Expected.PLAN,
        probes="the cheap end of the market, where the fixed food and transport floor bites",
    ),
    Scenario(
        id="impossible-budget",
        request=_request(budget_total=800),
        expected=Expected.REFUSAL,
        probes="a budget below the food and transport floor alone: refusal, not a bad plan",
    ),
    Scenario(
        id="generous-budget",
        request=_request(budget_total=5000),
        expected=Expected.PLAN,
        probes="whether more money quietly buys a worse-value recommendation",
    ),
    Scenario(
        id="single-preference",
        request=_request(preferences=(Preference.FOOD,)),
        expected=Expected.PLAN,
        probes="coverage with one interest, where the reserve-a-slot pass has little to do",
    ),
    Scenario(
        id="no-preferences",
        request=_request(preferences=()),
        expected=Expected.PLAN,
        probes="an empty preference list means 'no opinion', not 'nothing qualifies'",
    ),
    Scenario(
        id="unmatchable-preference",
        request=_request(preferences=(Preference.SHOPPING,)),
        expected=Expected.PLAN,
        probes="nothing in the snapshot is tagged shopping: warn, still deliver a stay",
    ),
    Scenario(
        id="short-stay",
        request=_request(end_date=date(2026, 9, 13)),
        expected=Expected.PLAN,
        probes="three nights against a snapshot captured for seven: extrapolation warnings",
    ),
    Scenario(
        id="party-too-large",
        request=_request(travelers=4),
        expected=Expected.REFUSAL,
        probes="capacity is unverified above the searched party size, so refuse",
    ),
    Scenario(
        id="unknown-destination",
        request=_request(destination="Kyoto"),
        expected=Expected.REFUSAL,
        probes="no snapshot for the destination: an explicit error, not an empty plan",
    ),
)
