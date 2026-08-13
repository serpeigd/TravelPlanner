"""Itinerary construction.

This is where the user's `culture` and `nature` preferences are actually served — a hotel
carries almost no evidence about them, activities carry all of it (see `docs/ranking.md`).

The selection is **coverage-first, then greedy**. A plain greedy pass over a score would
happily fill a week with seven food tours for someone who asked for food, culture *and*
nature: each pick is locally optimal and the plan as a whole is wrong. So the first pass
reserves one slot per stated preference, and only then does the remainder get filled by
score. That ordering is the whole reason preference coverage is measurable later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta

from travel_intel.budget import DEFAULT_BUDGET_POLICY, BudgetPolicy
from travel_intel.domain.enums import BudgetCategory, Preference
from travel_intel.domain.models import Activity, ItineraryDay, TripRequest
from travel_intel.ranking.scoring import weighted_score
from travel_intel.stats import shrink

ACTIVITY_WEIGHTS: dict[str, float] = {
    "preference_match": 0.45,
    "rating": 0.30,
    "affordability": 0.15,
    "evidence": 0.10,
}
"""Preference dominates: an activity exists in the plan because the traveller wants that
kind of thing. Rating is a quality filter within that, and affordability only breaks ties
between comparable options — the hard budget check happens separately, on the whole plan.

`evidence` is the counterpart of `data_completeness` in the accommodation ranking, and it is
here to close an asymmetry. When an activity has no reviews its `rating` factor is dropped
and the weight redistributed — which, without a counterweight, *rewards* being unknown: a
€198 tour with zero reviews was outranking a well-reviewed one purely by being tagged with
two of the user's interests. Charging the missing evidence once, explicitly, is the same
treatment hotels already get.
"""

ACTIVITIES_PER_DAY = 1
"""One booked activity per day, deliberately.

A plan that fills every hour is a plan nobody follows. One anchor per day leaves the rest
of the day for the city itself, and keeps the activities line inside its budget share.
"""

ACTIVITY_RATING_PRIOR_WEIGHT = 25.0
"""Shrinkage prior for activity ratings, in reviews.

Same treatment as hotel ratings, two orders of magnitude smaller. Activity review counts in
this dataset run 0-2,316 with a median of 7, so the hotel prior of 200 would flatten every
activity onto the market mean. At 25, a 5.0 from a single review lands near the mean while a
4.69 from 489 reviews keeps its own score — which is the whole point, since without it a
one-review perfect score outranks a genuinely well-reviewed tour.
"""

COST_EFFICIENCY_OFFSET = 1.0
"""Added to cost when ranking by score-per-euro, so free activities do not divide by zero
and so a €2 ticket does not score infinitely better than a €4 one."""


@dataclass(frozen=True)
class ActivityPlan:
    days: tuple[ItineraryDay, ...]
    selected: tuple[Activity, ...]
    total_cost: float
    covered_preferences: tuple[Preference, ...]
    uncovered_preferences: tuple[Preference, ...]
    considered: int
    """How many activities were eligible, for the funnel in the quality signals."""


def market_mean_rating(activities: Sequence[Activity]) -> float | None:
    """Mean rating across the activities on the table, or None if none is rated."""
    rated = [a.rating for a in activities if a.rating is not None]
    return sum(rated) / len(rated) if rated else None


def score_activity(
    activity: Activity,
    request: TripRequest,
    allowance: float,
    market_mean: float | None,
) -> float:
    """Score one activity for this request, on the same [0, 1] convention as the ranking."""
    cost = activity.cost_for(request.travelers)
    preference_match = (
        len(set(activity.categories) & set(request.preferences)) / len(request.preferences)
        if request.preferences
        else None
    )
    factors: dict[str, float | None] = {
        "preference_match": preference_match,
        "rating": _shrunk_rating(activity, market_mean),
        "affordability": max(0.0, 1.0 - cost / allowance) if allowance > 0 else 0.0,
        "evidence": _evidence(activity),
    }
    score, _ = weighted_score(factors, ACTIVITY_WEIGHTS)
    return score


def _evidence(activity: Activity) -> float:
    """How much is actually known about this activity's quality, on [0, 1].

    Measured in reviews relative to the shrinkage prior: at `ACTIVITY_RATING_PRIOR_WEIGHT`
    reviews the rating stands on its own and the factor saturates.
    """
    return min(1.0, float(activity.review_count or 0) / ACTIVITY_RATING_PRIOR_WEIGHT)


def _shrunk_rating(activity: Activity, market_mean: float | None) -> float | None:
    """Activity rating on [0, 1], shrunk toward the market mean by review volume."""
    if activity.rating is None or market_mean is None:
        return None
    evidence = float(activity.review_count or 0)
    shrunk = shrink(activity.rating, evidence, market_mean, ACTIVITY_RATING_PRIOR_WEIGHT)
    return min(1.0, max(0.0, shrunk / 10.0))


def build_itinerary(
    activities: tuple[Activity, ...],
    request: TripRequest,
    *,
    policy: BudgetPolicy = DEFAULT_BUDGET_POLICY,
) -> ActivityPlan:
    """Pick activities and lay them out across the stay.

    Never exceeds the activities allowance: an activity that does not fit is skipped rather
    than trimmed, because there is no such thing as three quarters of a museum ticket.
    """
    allowance = policy.allowance(BudgetCategory.ACTIVITIES, request.budget_total)
    eligible = _eligible(activities, request)
    market_mean = market_mean_rating(eligible)
    scores = {a.id: score_activity(a, request, allowance, market_mean) for a in eligible}

    by_score = sorted(eligible, key=lambda a: (-scores[a.id], a.id))
    by_efficiency = sorted(
        eligible,
        key=lambda a: (
            -scores[a.id] / (a.cost_for(request.travelers) + COST_EFFICIENCY_OFFSET),
            a.id,
        ),
    )

    selected: list[Activity] = []
    spent = 0.0
    slots = request.nights * ACTIVITIES_PER_DAY

    def try_add(activity: Activity) -> bool:
        nonlocal spent
        if len(selected) >= slots:
            return False
        cost = activity.cost_for(request.travelers)
        if spent + cost > allowance:
            return False
        selected.append(activity)
        spent = round(spent + cost, 2)
        return True

    # Pass 1 - one slot reserved for each stated preference, best first. Quality wins here:
    # covering a preference at all matters more than covering it cheaply.
    for preference in request.preferences:
        for activity in by_score:
            if activity in selected:
                continue
            if preference in activity.categories and try_add(activity):
                break

    # Pass 2 - fill the remaining days by score per euro, the standard greedy approximation
    # to a knapsack. Ranking pass 2 by raw score instead would spend the rest of the budget
    # on one or two headline tours and leave days empty; efficiency buys more of the trip.
    for activity in by_efficiency:
        if activity not in selected:
            try_add(activity)

    covered = tuple(
        preference
        for preference in request.preferences
        if any(preference in activity.categories for activity in selected)
    )
    uncovered = tuple(p for p in request.preferences if p not in covered)

    return ActivityPlan(
        days=_lay_out(selected, request),
        selected=tuple(selected),
        total_cost=round(spent, 2),
        covered_preferences=covered,
        uncovered_preferences=uncovered,
        considered=len(eligible),
    )


def _eligible(activities: tuple[Activity, ...], request: TripRequest) -> tuple[Activity, ...]:
    """Keep activities that match at least one stated preference.

    With no preferences stated, everything is eligible — an empty preference list means "no
    opinion", not "nothing qualifies".
    """
    if not request.preferences:
        return activities
    wanted = set(request.preferences)
    return tuple(a for a in activities if wanted & set(a.categories))


def _lay_out(selected: list[Activity], request: TripRequest) -> tuple[ItineraryDay, ...]:
    """Spread the selection over the stay, one per day, in selection order.

    Days without an activity are still emitted. A seven-night trip with five booked
    activities has two free days, and saying so is more useful than hiding them.
    """
    days: list[ItineraryDay] = []
    for offset in range(request.nights):
        on_this_day = selected[offset : offset + ACTIVITIES_PER_DAY]
        days.append(
            ItineraryDay(
                day_index=offset + 1,
                day=request.start_date + timedelta(days=offset),
                activity_ids=tuple(a.id for a in on_this_day),
                estimated_cost=round(sum(a.cost_for(request.travelers) for a in on_this_day), 2),
            )
        )
    return tuple(days)
