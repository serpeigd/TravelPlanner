"""Constraint validation and the planner walk.

The property these tests exist to protect: **a plan that breaks a hard constraint is never
returned**. Not discouraged, not down-ranked — not constructed.
"""

from datetime import date

import pytest

from travel_intel.config import Settings
from travel_intel.constraints import (
    ACCOMMODATION_OVER_ALLOWANCE,
    BUDGET_EXCEEDED,
    CURRENCY_MISMATCH,
    INSUFFICIENT_CAPACITY,
    LOW_DATA_COMPLETENESS,
    NO_ACTIVITIES_PLANNED,
    PREFERENCE_NOT_COVERED,
    STALE_DATA,
    validate_plan,
)
from travel_intel.domain.enums import Preference, Provenance, Severity
from travel_intel.domain.errors import NoCandidatesError
from travel_intel.domain.models import (
    Accommodation,
    Activity,
    BudgetBreakdown,
    ScoreBreakdown,
    ScoredAccommodation,
    TripRequest,
)
from travel_intel.planning.assumptions import assumptions_for
from travel_intel.planning.itinerary import build_itinerary
from travel_intel.ranking import generate_candidates, rank_accommodations
from travel_intel.retrieval.snapshot import SnapshotProvider
from travel_intel.services import plan_trip


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


def make_scored(**overrides: object) -> ScoredAccommodation:
    payload: dict[str, object] = {
        "id": "h1",
        "name": "Test Hotel",
        "provider": "fixture",
        "provenance": Provenance.SNAPSHOT,
        "price_per_night": 100.0,
        "max_occupancy": 2,
        "rating": 8.5,
        "review_count": 500,
        "stars": 3,
        "neighborhood": "Somewhere",
        "distance_to_center_km": 2.0,
        "source_url": "https://example.invalid/h1",
    }
    payload.update(overrides)
    accommodation = Accommodation(**payload)  # type: ignore[arg-type]
    return ScoredAccommodation(
        accommodation=accommodation,
        scores=ScoreBreakdown(budget_fit=1.0, data_completeness=accommodation.data_completeness),
        overall=0.8,
        weights={"budget_fit": 0.75, "data_completeness": 0.25},
        total_cost=accommodation.total_cost(7),
        rank=1,
    )


def make_budget(**overrides: float) -> BudgetBreakdown:
    payload: dict[str, float] = {
        "budget_total": 2500.0,
        "accommodation": 700.0,
        "activities": 400.0,
        "food": 630.0,
        "transport": 168.0,
    }
    payload.update(overrides)
    return BudgetBreakdown(**payload)


@pytest.fixture
def activities() -> tuple[Activity, ...]:
    return SnapshotProvider(Settings().fixtures_dir).search_activities(make_request()).records


def codes(report: object) -> set[str]:
    return {v.code for v in report.violations}  # type: ignore[attr-defined]


class TestHardConstraints:
    def test_over_budget_is_hard_and_invalidates(self, activities: tuple[Activity, ...]) -> None:
        request = make_request()
        report = validate_plan(
            request,
            make_scored(),
            make_budget(accommodation=2000.0),
            build_itinerary(activities, request),
        )
        assert report.is_valid is False
        assert BUDGET_EXCEEDED in codes(report)
        violation = next(v for v in report.violations if v.code == BUDGET_EXCEEDED)
        assert violation.severity is Severity.HARD
        assert violation.limit == 2500.0

    def test_capacity_is_hard(self, activities: tuple[Activity, ...]) -> None:
        request = make_request(travelers=2)
        report = validate_plan(
            request,
            make_scored(max_occupancy=1),
            make_budget(),
            build_itinerary(activities, request),
        )
        assert report.is_valid is False
        assert INSUFFICIENT_CAPACITY in codes(report)

    def test_currency_mismatch_is_hard(self, activities: tuple[Activity, ...]) -> None:
        """Two amounts in different currencies are not comparable without a rate.

        Silently treating them as equal is exactly the class of error this layer exists to
        stop, so it blocks the plan rather than warning about it.
        """
        request = make_request()
        report = validate_plan(
            request, make_scored(), make_budget(), build_itinerary(activities, request)
        )
        assert CURRENCY_MISMATCH not in codes(report)  # both EUR in the fixture

    def test_a_plan_exactly_on_budget_is_valid(self, activities: tuple[Activity, ...]) -> None:
        request = make_request()
        budget = make_budget(accommodation=1302.0)  # totals exactly 2500
        assert budget.total == 2500.0
        report = validate_plan(request, make_scored(), budget, build_itinerary(activities, request))
        assert BUDGET_EXCEEDED not in codes(report)


class TestSoftConstraints:
    def test_over_allowance_warns_without_invalidating(
        self, activities: tuple[Activity, ...]
    ) -> None:
        request = make_request()
        report = validate_plan(
            request,
            make_scored(),
            make_budget(accommodation=1200.0),  # over the 1125 share, under the total
            build_itinerary(activities, request),
        )
        assert report.is_valid is True
        assert ACCOMMODATION_OVER_ALLOWANCE in codes(report)

    def test_uncovered_preference_warns(self, activities: tuple[Activity, ...]) -> None:
        request = make_request(preferences=[Preference.SHOPPING])
        report = validate_plan(
            request, make_scored(), make_budget(), build_itinerary(activities, request)
        )
        assert report.is_valid is True
        assert {PREFERENCE_NOT_COVERED, NO_ACTIVITIES_PLANNED} <= codes(report)

    def test_thin_data_on_the_recommendation_warns(self, activities: tuple[Activity, ...]) -> None:
        request = make_request()
        bare = make_scored(rating=None, review_count=None, stars=None, neighborhood=None)
        report = validate_plan(request, bare, make_budget(), build_itinerary(activities, request))
        assert report.is_valid is True
        assert LOW_DATA_COMPLETENESS in codes(report)

    def test_retrieval_warnings_surface_as_soft_violations(
        self, activities: tuple[Activity, ...]
    ) -> None:
        request = make_request()
        report = validate_plan(
            request,
            make_scored(),
            make_budget(),
            build_itinerary(activities, request),
            retrieval_warnings=["snapshot prices were captured for other dates"],
        )
        assert report.is_valid is True
        assert STALE_DATA in codes(report)

    def test_a_clean_plan_has_no_violations(self, activities: tuple[Activity, ...]) -> None:
        request = make_request()
        report = validate_plan(
            request, make_scored(), make_budget(), build_itinerary(activities, request)
        )
        assert report.violations == ()
        assert report.is_valid is True


class TestPlannerWalk:
    @pytest.fixture
    def ranked(self) -> object:
        provider = SnapshotProvider(Settings().fixtures_dir)
        request = make_request()
        records = provider.search_accommodations(request).records
        return rank_accommodations(generate_candidates(records, request).records, request)

    def test_end_to_end_plan_for_the_demo_request(
        self, ranked: object, activities: tuple[Activity, ...]
    ) -> None:
        request = make_request()
        trip = plan_trip(ranked, activities, request)  # type: ignore[arg-type]
        assert trip.constraints.is_valid
        assert trip.budget.total <= request.budget_total
        assert trip.recommended.rank == 1
        assert len(trip.alternatives) == 4

    def test_options_the_ranking_preferred_can_be_refused_by_the_constraints(
        self, ranked: object, activities: tuple[Activity, ...]
    ) -> None:
        """The walk, exercised: a tighter budget makes top-ranked options undeliverable.

        Each of these passed candidate generation — the room alone fits the budget — and
        only fails once food, transport and activities are added. That is precisely the
        error a plausible-sounding model would make, and it is caught in code.
        """
        request = make_request(budget_total=2100)
        trip = plan_trip(ranked, activities, request)  # type: ignore[arg-type]
        assert trip.rejected
        assert all(BUDGET_EXCEEDED in option.codes for option in trip.rejected)
        assert trip.recommended.rank > 1
        assert trip.budget.total <= 2100

    def test_rejected_options_are_reported_not_hidden(
        self, ranked: object, activities: tuple[Activity, ...]
    ) -> None:
        trip = plan_trip(ranked, activities, make_request(budget_total=2100))  # type: ignore[arg-type]
        assert all(option.name and option.rank >= 1 for option in trip.rejected)

    @pytest.mark.parametrize("budget", [1600, 1800, 2100, 2500, 4000])
    def test_no_budget_ever_produces_a_violating_plan(
        self, ranked: object, activities: tuple[Activity, ...], budget: int
    ) -> None:
        """The invariant the whole layer exists for, across the plausible budget range."""
        request = make_request(budget_total=budget)
        trip = plan_trip(ranked, activities, request)  # type: ignore[arg-type]
        assert trip.constraints.is_valid
        assert trip.budget.total <= budget

    def test_an_impossible_budget_raises_instead_of_answering(
        self, ranked: object, activities: tuple[Activity, ...]
    ) -> None:
        request = make_request(budget_total=900)  # under the fixed food + transport floor
        with pytest.raises(NoCandidatesError, match="hard constraint"):
            plan_trip(ranked, activities, request)  # type: ignore[arg-type]

    def test_an_empty_ranking_raises(self, activities: tuple[Activity, ...]) -> None:
        with pytest.raises(NoCandidatesError, match="ranking is empty"):
            plan_trip((), activities, make_request())

    def test_assumptions_travel_with_the_plan(
        self, ranked: object, activities: tuple[Activity, ...]
    ) -> None:
        trip = plan_trip(ranked, activities, make_request())  # type: ignore[arg-type]
        assert trip.assumptions is assumptions_for("tokyo")
        assert trip.assumptions.source().provenance is Provenance.SYNTHETIC
