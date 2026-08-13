"""Transparent scoring.

Every candidate gets a small set of [0, 1] factors and a weighted average. No model decides
the ordering, no LLM is consulted, and the exact weights that produced each number travel
with the result so a user — or an interviewer — can take the score apart.

The formula, and the reasoning behind each weight, is in `docs/ranking.md`.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import pandas as pd

from travel_intel.budget import DEFAULT_BUDGET_POLICY, BudgetPolicy
from travel_intel.domain.models import (
    Accommodation,
    ScoreBreakdown,
    ScoredAccommodation,
    TripRequest,
)
from travel_intel.features.accommodations import build_accommodation_features

DEFAULT_WEIGHTS: dict[str, float] = {
    "budget_fit": 0.25,
    "value_for_money": 0.20,
    "rating": 0.20,
    "location": 0.15,
    "preference_match": 0.12,
    "data_completeness": 0.08,
}
"""Relative importance of each factor. See docs/ranking.md for the argument behind each.

`value_for_money` is listed even though the price model only arrives in M5. Until then the
factor is `None` for every candidate and its weight is redistributed, so switching the model
on is a drop-in change rather than a re-tuning of everything else.
"""

BUDGET_OVERRUN_TOLERANCE = 0.5
"""How far past its allowance a room may go before `budget_fit` hits zero (+50 %).

Beyond that the room is eating the food and activities budget, and no rating compensates.
"""

MAX_USEFUL_DISTANCE_KM = 10.0
"""Distance from the centre at which `location` hits zero.

In Tokyo terms, past ~10 km from Tokyo Station every outing becomes a planned commute rather
than a walk or a short hop, which is a different kind of trip.
"""


def budget_fit(budget_ratio: float) -> float:
    """Score how a room's total cost sits against its share of the budget.

    Full marks at or under the allowance — money left over is a benefit, not a reason to
    rank one option above another. Being *cheap* is not the same as being *good value*; that
    is `value_for_money`'s job, and conflating the two would double-count price.
    """
    if budget_ratio <= 1.0:
        return 1.0
    overrun = (budget_ratio - 1.0) / BUDGET_OVERRUN_TOLERANCE
    return max(0.0, 1.0 - overrun)


def location_score(distance_km: float) -> float:
    """Linear decay from the city centre to `MAX_USEFUL_DISTANCE_KM`."""
    return max(0.0, 1.0 - distance_km / MAX_USEFUL_DISTANCE_KM)


def weighted_score(
    factors: Mapping[str, float | None],
    weights: Mapping[str, float],
) -> tuple[float, dict[str, float]]:
    """Weighted average over the factors that could actually be computed.

    Missing factors are dropped and the remaining weights renormalised, so the score always
    means "out of the evidence available" rather than "penalised for what we don't know".
    Returns the score together with the effective weights actually applied.
    """
    available = {name: value for name, value in factors.items() if value is not None}
    if not available:
        raise ValueError("cannot score a candidate with no computable factors")

    active = {name: weights[name] for name in available if weights.get(name, 0.0) > 0.0}
    total_weight = sum(active.values())
    if total_weight <= 0.0:
        raise ValueError("computable factors carry no weight")

    effective = {name: weight / total_weight for name, weight in active.items()}
    score = sum(effective[name] * available[name] for name in effective)
    return round(score, 4), {name: round(weight, 4) for name, weight in effective.items()}


def rank_accommodations(
    records: tuple[Accommodation, ...],
    request: TripRequest,
    *,
    policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
    weights: Mapping[str, float] | None = None,
) -> tuple[ScoredAccommodation, ...]:
    """Score and order candidates, best first.

    Ties break on accommodation id so that the same input always produces the same order —
    a ranking that reshuffles between runs cannot be evaluated.
    """
    applied = dict(weights or DEFAULT_WEIGHTS)
    features = build_accommodation_features(records, request, policy)

    scored: list[ScoredAccommodation] = []
    for record in records:
        breakdown = ScoreBreakdown(
            budget_fit=budget_fit(_number(features.at[record.id, "budget_ratio"])),
            data_completeness=_number(features.at[record.id, "completeness"]),
            rating=_scale_rating(_optional(features.at[record.id, "rating_shrunk"])),
            location=_scale_distance(_optional(features.at[record.id, "distance_km"])),
            preference_match=_optional(features.at[record.id, "preference_match"]),
        )
        overall, effective = weighted_score(breakdown.as_factors(), applied)
        scored.append(
            ScoredAccommodation(
                accommodation=record,
                scores=breakdown,
                overall=overall,
                weights=effective,
                total_cost=_number(features.at[record.id, "total_cost"]),
                rank=1,  # replaced below, once the full ordering is known
            )
        )

    scored.sort(key=lambda item: (-item.overall, item.accommodation.id))
    return tuple(
        item.model_copy(update={"rank": position}) for position, item in enumerate(scored, start=1)
    )


def _number(value: object) -> float:
    """Read a pandas cell that must be present as a float."""
    return float(value)  # type: ignore[arg-type]


def _optional(value: object) -> float | None:
    """Convert a possibly-missing pandas cell into `float | None`."""
    if value is None:
        return None
    number = _number(value)
    return None if math.isnan(number) else number


def _scale_rating(rating: float | None) -> float | None:
    """0-10 guest rating onto the shared [0, 1] factor scale."""
    return None if rating is None else min(1.0, max(0.0, rating / 10.0))


def _scale_distance(distance_km: float | None) -> float | None:
    return None if distance_km is None else location_score(distance_km)


def features_frame(
    records: tuple[Accommodation, ...],
    request: TripRequest,
    policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
) -> pd.DataFrame:
    """Expose the underlying feature table for inspection and evaluation."""
    return build_accommodation_features(records, request, policy)
