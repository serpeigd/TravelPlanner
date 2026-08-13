"""Itinerary construction, budget composition and cost assumptions."""

from datetime import date

import pytest

from travel_intel.budget import DEFAULT_BUDGET_POLICY
from travel_intel.config import Settings
from travel_intel.domain.enums import BudgetCategory, Preference, Provenance
from travel_intel.domain.models import Activity, TripRequest
from travel_intel.planning.assumptions import (
    DEFAULT_DAILY_COSTS,
    DailyCostAssumptions,
    assumptions_for,
)
from travel_intel.planning.costs import compose_budget
from travel_intel.planning.itinerary import build_itinerary, market_mean_rating, score_activity
from travel_intel.retrieval.snapshot import SnapshotProvider


def make_request(**overrides: object) -> TripRequest:
    payload: dict[str, object] = {
        "destination": "Tokyo",
        "start_date": date(2026, 9, 10),
        "end_date": date(2026, 9, 17),
        "travelers": 2,
        "budget_total": 2500,
        "preferences": [Preference.FOOD, Preference.CULTURE, Preference.NATURE],
    }
    payload.update(overrides)
    return TripRequest(**payload)  # type: ignore[arg-type]


@pytest.fixture
def activities() -> tuple[Activity, ...]:
    provider = SnapshotProvider(Settings().fixtures_dir)
    return provider.search_activities(make_request()).records


def make_activity(**overrides: object) -> Activity:
    payload: dict[str, object] = {
        "id": "a1",
        "name": "Something",
        "provider": "fixture",
        "provenance": Provenance.SNAPSHOT,
        "categories": (Preference.FOOD,),
        "price_per_person": 20.0,
        "rating": 9.0,
        "review_count": 100,
    }
    payload.update(overrides)
    return Activity(**payload)  # type: ignore[arg-type]


class TestAssumptions:
    def test_tokyo_has_its_own_figures(self) -> None:
        assert assumptions_for("tokyo").food == 45.0

    def test_unknown_destination_falls_back(self) -> None:
        assert assumptions_for("atlantis") is DEFAULT_DAILY_COSTS

    def test_non_positive_costs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            DailyCostAssumptions(food=0.0, transport=10.0, rationale="broken")

    def test_estimates_are_declared_synthetic(self) -> None:
        source = assumptions_for("tokyo").source()
        assert source.provenance is Provenance.SYNTHETIC


class TestBudgetComposition:
    def test_lines_and_total(self) -> None:
        budget = compose_budget(
            make_request(),
            accommodation_cost=1201.06,
            activities_cost=427.46,
            assumptions=assumptions_for("tokyo"),
        )
        # 45 EUR food and 12 EUR transport per person per day, 2 travelers, 7 nights.
        assert budget.food == 630.0
        assert budget.transport == 168.0
        assert budget.total == pytest.approx(2426.52)
        assert budget.remaining == pytest.approx(73.48)

    def test_estimates_are_independent_of_the_budget(self) -> None:
        """Food and transport must not be derived from their budget shares.

        If they were, the plan would consume 100 % of any budget by construction and
        "does this fit?" would stop being answerable. Doubling the budget must leave the
        estimated lines untouched.
        """
        assumptions = assumptions_for("tokyo")
        lean = compose_budget(
            make_request(budget_total=2500),
            accommodation_cost=600.0,
            activities_cost=200.0,
            assumptions=assumptions,
        )
        rich = compose_budget(
            make_request(budget_total=5000),
            accommodation_cost=600.0,
            activities_cost=200.0,
            assumptions=assumptions,
        )
        assert lean.food == rich.food
        assert lean.transport == rich.transport
        assert lean.utilization > rich.utilization

    def test_estimates_scale_with_party_and_length(self) -> None:
        assumptions = assumptions_for("tokyo")
        base = compose_budget(
            make_request(),
            accommodation_cost=0.01,
            activities_cost=0.0,
            assumptions=assumptions,
        )
        bigger = compose_budget(
            make_request(travelers=4),
            accommodation_cost=0.01,
            activities_cost=0.0,
            assumptions=assumptions,
        )
        assert bigger.food == pytest.approx(base.food * 2)


class TestActivityScoring:
    def test_shrinkage_discounts_a_perfect_score_from_one_review(self) -> None:
        request = make_request(preferences=[Preference.FOOD])
        one_review = make_activity(id="thin", rating=10.0, review_count=1)
        well_reviewed = make_activity(id="thick", rating=9.0, review_count=500)
        pool = [one_review, well_reviewed]
        mean = market_mean_rating(pool)
        assert mean is not None
        assert score_activity(well_reviewed, request, 500.0, mean) > score_activity(
            one_review, request, 500.0, mean
        )

    def test_an_unreviewed_activity_is_charged_for_the_missing_evidence(self) -> None:
        """Regression: dropping the rating factor must not *reward* being unknown.

        Both activities cover the same single preference at the same price. The unrated one
        loses its `rating` factor, and without the `evidence` counterweight the
        redistribution used to leave it ahead of a genuinely well-reviewed alternative.
        """
        request = make_request(preferences=[Preference.FOOD])
        unknown = make_activity(id="unknown", rating=None, review_count=0)
        known = make_activity(id="known", rating=9.0, review_count=200)
        mean = market_mean_rating([unknown, known])
        assert mean is not None
        assert score_activity(known, request, 500.0, mean) > score_activity(
            unknown, request, 500.0, mean
        )

    def test_covering_two_preferences_beats_covering_one(self) -> None:
        request = make_request()
        broad = make_activity(id="broad", categories=(Preference.FOOD, Preference.CULTURE))
        narrow = make_activity(id="narrow", categories=(Preference.FOOD,))
        mean = market_mean_rating([broad, narrow])
        assert mean is not None
        assert score_activity(broad, request, 500.0, mean) > score_activity(
            narrow, request, 500.0, mean
        )


class TestItinerary:
    def test_covers_every_stated_preference(self, activities: tuple[Activity, ...]) -> None:
        plan = build_itinerary(activities, make_request())
        assert plan.uncovered_preferences == ()
        assert set(plan.covered_preferences) == {
            Preference.FOOD,
            Preference.CULTURE,
            Preference.NATURE,
        }

    def test_never_exceeds_the_activities_allowance(self, activities: tuple[Activity, ...]) -> None:
        request = make_request()
        allowance = DEFAULT_BUDGET_POLICY.allowance(BudgetCategory.ACTIVITIES, request.budget_total)
        plan = build_itinerary(activities, request)
        assert plan.total_cost <= allowance

    def test_emits_one_day_per_night_including_empty_ones(
        self, activities: tuple[Activity, ...]
    ) -> None:
        request = make_request()
        plan = build_itinerary(activities, request)
        assert len(plan.days) == request.nights
        assert [d.day_index for d in plan.days] == list(range(1, request.nights + 1))
        assert plan.days[0].day == request.start_date

    def test_day_costs_sum_to_the_plan_total(self, activities: tuple[Activity, ...]) -> None:
        plan = build_itinerary(activities, make_request())
        assert sum(d.estimated_cost for d in plan.days) == pytest.approx(plan.total_cost)

    def test_fills_the_stay_rather_than_spending_it_all_on_headline_tours(
        self, activities: tuple[Activity, ...]
    ) -> None:
        """The fill pass ranks by score per euro, not by raw score.

        Ranking pass 2 by raw score left three days empty with the allowance exhausted; the
        knapsack-style ordering buys more of the trip for the same money.
        """
        request = make_request()
        plan = build_itinerary(activities, request)
        assert len(plan.selected) == request.nights

    def test_selection_is_deterministic(self, activities: tuple[Activity, ...]) -> None:
        request = make_request()
        first = build_itinerary(activities, request)
        second = build_itinerary(tuple(reversed(activities)), request)
        assert [a.id for a in first.selected] == [a.id for a in second.selected]

    def test_no_preferences_means_no_opinion_not_no_candidates(
        self, activities: tuple[Activity, ...]
    ) -> None:
        plan = build_itinerary(activities, make_request(preferences=[]))
        assert plan.considered == len(activities)
        assert plan.selected

    def test_unmatchable_preference_is_reported_not_hidden(
        self, activities: tuple[Activity, ...]
    ) -> None:
        # Nothing in the snapshot is tagged SHOPPING.
        plan = build_itinerary(activities, make_request(preferences=[Preference.SHOPPING]))
        assert plan.selected == ()
        assert plan.uncovered_preferences == (Preference.SHOPPING,)

    def test_an_activity_that_does_not_fit_is_skipped_whole(self) -> None:
        request = make_request(budget_total=100)  # activities allowance: 20 EUR
        cheap = make_activity(id="cheap", price_per_person=5.0)
        dear = make_activity(id="dear", price_per_person=500.0)
        plan = build_itinerary((cheap, dear), request)
        assert [a.id for a in plan.selected] == ["cheap"]
        assert plan.total_cost == 10.0
