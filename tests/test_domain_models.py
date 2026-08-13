"""Contract tests: the invariants the rest of the pipeline is allowed to assume."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError

from travel_intel.domain.enums import Preference, Provenance, Severity
from travel_intel.domain.models import (
    Accommodation,
    Activity,
    BudgetBreakdown,
    ConstraintReport,
    ConstraintViolation,
    TripRequest,
)


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


class TestTripRequest:
    def test_derived_fields(self) -> None:
        req = make_request()
        assert req.nights == 7
        assert req.budget_per_night == pytest.approx(357.14)
        assert req.budget_per_person == pytest.approx(1250.0)

    def test_end_date_must_follow_start_date(self) -> None:
        with pytest.raises(ValidationError, match="end_date must be after start_date"):
            make_request(end_date=date(2026, 9, 10))

    def test_trip_length_is_capped(self) -> None:
        with pytest.raises(ValidationError, match="must not exceed"):
            make_request(end_date=date(2026, 12, 31))

    def test_budget_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            make_request(budget_total=0)

    def test_preferences_are_deduplicated_preserving_order(self) -> None:
        req = make_request(preferences=[Preference.FOOD, Preference.FOOD, Preference.ART])
        assert req.preferences == (Preference.FOOD, Preference.ART)

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_request(buget_total=2500)  # typo must not pass silently

    def test_is_immutable(self) -> None:
        req = make_request()
        with pytest.raises(ValidationError):
            req.travelers = 4  # type: ignore[misc]

    def test_money_is_rounded_to_cents(self) -> None:
        assert make_request(budget_total=2500.129).budget_total == 2500.13


class TestAccommodation:
    def _hotel(self, **overrides: object) -> Accommodation:
        payload: dict[str, object] = {
            "id": "h1",
            "name": "Shinjuku Base",
            "provider": "fixture",
            "provenance": Provenance.SNAPSHOT,
            "price_per_night": 180.0,
            "max_occupancy": 2,
        }
        payload.update(overrides)
        return Accommodation(**payload)  # type: ignore[arg-type]

    def test_total_cost_scales_with_nights(self) -> None:
        assert self._hotel().total_cost(7) == 1260.0

    def test_total_cost_rejects_zero_nights(self) -> None:
        with pytest.raises(ValueError, match="nights must be >= 1"):
            self._hotel().total_cost(0)

    def test_completeness_is_zero_when_only_required_fields_are_present(self) -> None:
        assert self._hotel().data_completeness == 0.0

    def test_completeness_grows_with_known_fields(self) -> None:
        hotel = self._hotel(
            rating=8.6,
            review_count=1200,
            neighborhood="Shinjuku",
            distance_to_center_km=2.4,
            source_url="https://example.invalid/h1",
            retrieved_at=datetime(2026, 8, 1, 12, 0),
        )
        # 5 of the 7 completeness fields present (location and stars are still missing).
        assert hotel.data_completeness == pytest.approx(5 / 7, abs=1e-4)

    def test_rating_scale_is_enforced(self) -> None:
        with pytest.raises(ValidationError):
            self._hotel(rating=11.0)


class TestActivity:
    def test_cost_scales_with_travelers(self) -> None:
        activity = Activity(
            id="a1",
            name="Tsukiji food walk",
            provider="fixture",
            provenance=Provenance.SNAPSHOT,
            categories=(Preference.FOOD,),
            price_per_person=45.5,
        )
        assert activity.cost_for(2) == 91.0

    def test_free_activities_are_valid(self) -> None:
        activity = Activity(
            id="a2",
            name="Meiji Shrine",
            provider="fixture",
            provenance=Provenance.SNAPSHOT,
        )
        assert activity.cost_for(3) == 0.0


class TestBudgetBreakdown:
    def _budget(self, **overrides: float) -> BudgetBreakdown:
        payload: dict[str, float] = {
            "budget_total": 2500.0,
            "accommodation": 1260.0,
            "activities": 380.0,
            "food": 560.0,
            "transport": 150.0,
        }
        payload.update(overrides)
        return BudgetBreakdown(**payload)

    def test_total_and_remaining(self) -> None:
        budget = self._budget()
        assert budget.total == 2350.0
        assert budget.remaining == 150.0
        assert budget.utilization == 0.94

    def test_overspend_is_reported_as_negative_remaining(self) -> None:
        budget = self._budget(accommodation=2000.0)
        assert budget.remaining < 0
        assert budget.utilization > 1.0

    def test_categories_sum_to_total(self) -> None:
        budget = self._budget()
        assert sum(budget.by_category().values()) == pytest.approx(budget.total)


class TestConstraintReport:
    def test_empty_report_is_valid(self) -> None:
        assert ConstraintReport().is_valid is True

    def test_soft_violation_does_not_invalidate(self) -> None:
        report = ConstraintReport(
            violations=(
                ConstraintViolation(
                    code="preference_partially_matched",
                    severity=Severity.SOFT,
                    message="No nature activity within range",
                ),
            )
        )
        assert report.is_valid is True
        assert report.hard_violation_count == 0

    def test_hard_violation_invalidates(self) -> None:
        report = ConstraintReport(
            violations=(
                ConstraintViolation(
                    code="budget_exceeded",
                    severity=Severity.HARD,
                    message="Plan costs more than the stated budget",
                    observed=2700.0,
                    limit=2500.0,
                ),
            )
        )
        assert report.is_valid is False
        assert report.hard_violation_count == 1
