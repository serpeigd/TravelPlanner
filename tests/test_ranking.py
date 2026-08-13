"""Candidate generation, feature engineering and scoring, against the real Tokyo snapshot."""

from datetime import date
from itertools import pairwise

import pytest

from travel_intel.budget import DEFAULT_BUDGET_POLICY, BudgetPolicy
from travel_intel.config import Settings
from travel_intel.domain.enums import BudgetCategory, Preference
from travel_intel.domain.errors import NoCandidatesError
from travel_intel.domain.models import Accommodation, TripRequest
from travel_intel.features.accommodations import MODEL_FEATURES, build_accommodation_features
from travel_intel.ranking.candidates import (
    REASON_CAPACITY,
    REASON_UNAFFORDABLE,
    generate_candidates,
)
from travel_intel.ranking.scoring import (
    DEFAULT_WEIGHTS,
    budget_fit,
    location_score,
    rank_accommodations,
    weighted_score,
)
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
def retrieved() -> tuple[Accommodation, ...]:
    provider = SnapshotProvider(Settings().fixtures_dir)
    return provider.search_accommodations(make_request()).records


class TestBudgetPolicy:
    def test_default_shares_sum_to_one(self) -> None:
        assert sum(DEFAULT_BUDGET_POLICY.as_dict().values()) == pytest.approx(1.0)

    def test_shares_that_do_not_sum_to_one_are_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            BudgetPolicy(accommodation=0.9, food=0.25, activities=0.20, transport=0.10)

    def test_non_positive_share_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            BudgetPolicy(accommodation=0.75, food=0.25, activities=0.0, transport=0.0)

    def test_allowance(self) -> None:
        allowance = DEFAULT_BUDGET_POLICY.allowance(BudgetCategory.ACCOMMODATION, 2500.0)
        assert allowance == 1125.0


class TestCandidateGeneration:
    def test_drops_options_costing_more_than_the_entire_budget(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        candidates = generate_candidates(retrieved, make_request())
        assert candidates.retrieved == 30
        assert candidates.dropped[REASON_UNAFFORDABLE] == 4
        assert candidates.kept == 26
        assert all(r.total_cost(7) <= 2500 for r in candidates.records)

    def test_capacity_is_a_hard_filter(self, retrieved: tuple[Accommodation, ...]) -> None:
        # The snapshot was searched for 2 guests, so capacity above that is unverified.
        with pytest.raises(NoCandidatesError, match="no accommodation"):
            generate_candidates(retrieved, make_request(travelers=3))

    def test_capacity_drops_are_counted(self) -> None:
        request = make_request(travelers=2)
        small = Accommodation(
            id="tiny",
            name="Single room",
            provider="fixture",
            provenance="snapshot",  # type: ignore[arg-type]
            price_per_night=50.0,
            max_occupancy=1,
        )
        big = Accommodation(
            id="ok",
            name="Double room",
            provider="fixture",
            provenance="snapshot",  # type: ignore[arg-type]
            price_per_night=50.0,
            max_occupancy=2,
        )
        candidates = generate_candidates((small, big), request)
        assert candidates.dropped == {REASON_CAPACITY: 1}
        assert candidates.kept == 1

    def test_impossible_budget_raises_rather_than_returning_nothing(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        with pytest.raises(NoCandidatesError, match="filtered out"):
            generate_candidates(retrieved, make_request(budget_total=100))


class TestFeatures:
    def test_indexed_by_accommodation_id(self, retrieved: tuple[Accommodation, ...]) -> None:
        frame = build_accommodation_features(retrieved, make_request())
        assert frame.index.name == "id"
        assert len(frame) == len(retrieved)

    def test_every_model_feature_is_present(self, retrieved: tuple[Accommodation, ...]) -> None:
        frame = build_accommodation_features(retrieved, make_request())
        assert set(MODEL_FEATURES) <= set(frame.columns)

    def test_budget_ratio_is_relative_to_the_accommodation_allowance(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        frame = build_accommodation_features(retrieved, make_request())
        # EUR 643.81 for the stay against a EUR 1,125 allowance.
        assert frame.at["1118391", "budget_ratio"] == pytest.approx(0.5723, abs=1e-3)

    def test_shrinkage_pulls_thin_evidence_toward_the_market(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        frame = build_accommodation_features(retrieved, make_request())
        # 9.2 from 136 reviews: strong claim, weak evidence -> pulled down noticeably.
        thin = frame.at["13519990", "rating_shrunk"]
        assert thin < 9.2 - 0.3
        # 8.2 from 23,178 reviews: the property's own score dominates the prior.
        thick = frame.at["6416023", "rating_shrunk"]
        assert abs(thick - 8.2) < 0.05

    def test_shrinkage_never_invents_a_rating(self, retrieved: tuple[Accommodation, ...]) -> None:
        frame = build_accommodation_features(retrieved, make_request())
        assert frame.at["14451771", "rating_shrunk"] != frame.at["14451771", "rating_shrunk"]

    def test_preference_match_uses_only_evidence_bearing_preferences(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        frame = build_accommodation_features(retrieved, make_request())
        # FOOD and NATURE carry property-level evidence; CULTURE does not and is excluded.
        # This hotel has a Restaurant (FOOD) but no Garden or Terrace (NATURE).
        assert frame.at["1118391", "preference_match"] == pytest.approx(0.5)
        # No restaurant, no garden.
        assert frame.at["10560836", "preference_match"] == pytest.approx(0.0)

    def test_preference_match_is_missing_when_no_preference_has_evidence(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        request = make_request(preferences=[Preference.CULTURE, Preference.HISTORY])
        frame = build_accommodation_features(retrieved, request)
        assert frame["preference_match"].isna().all()

    def test_empty_candidate_set_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty candidate set"):
            build_accommodation_features((), make_request())


class TestFactorCurves:
    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.3, 1.0), (1.0, 1.0), (1.25, 0.5), (1.5, 0.0), (3.0, 0.0)],
    )
    def test_budget_fit(self, ratio: float, expected: float) -> None:
        assert budget_fit(ratio) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("distance", "expected"),
        [(0.0, 1.0), (2.5, 0.75), (10.0, 0.0), (25.0, 0.0)],
    )
    def test_location(self, distance: float, expected: float) -> None:
        assert location_score(distance) == pytest.approx(expected)


class TestWeightedScore:
    def test_all_factors_available(self) -> None:
        score, effective = weighted_score({"a": 1.0, "b": 0.0}, {"a": 0.75, "b": 0.25})
        assert score == pytest.approx(0.75)
        assert effective == {"a": 0.75, "b": 0.25}

    def test_missing_factor_redistributes_its_weight(self) -> None:
        score, effective = weighted_score({"a": 0.8, "b": None}, {"a": 0.5, "b": 0.5})
        assert score == pytest.approx(0.8)  # not 0.4: absence is not a zero
        assert effective == {"a": 1.0}

    def test_effective_weights_always_sum_to_one(self) -> None:
        _, effective = weighted_score(
            {"a": 0.5, "b": 0.5, "c": None}, {"a": 0.2, "b": 0.3, "c": 0.5}
        )
        assert sum(effective.values()) == pytest.approx(1.0)

    def test_no_computable_factor_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no computable factors"):
            weighted_score({"a": None}, {"a": 1.0})


class TestRanking:
    def test_ranks_are_contiguous_and_ordered(self, retrieved: tuple[Accommodation, ...]) -> None:
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        assert [item.rank for item in ranked] == list(range(1, len(ranked) + 1))
        assert all(a.overall >= b.overall for a, b in pairwise(ranked))

    def test_scores_stay_on_the_unit_interval(self, retrieved: tuple[Accommodation, ...]) -> None:
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        assert all(0.0 <= item.overall <= 1.0 for item in ranked)

    def test_is_deterministic(self, retrieved: tuple[Accommodation, ...]) -> None:
        request = make_request()
        candidates = generate_candidates(retrieved, request).records
        first = rank_accommodations(candidates, request)
        second = rank_accommodations(tuple(reversed(candidates)), request)
        assert [i.accommodation.id for i in first] == [i.accommodation.id for i in second]

    def test_effective_weights_are_recorded_and_normalised(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        for item in ranked:
            assert sum(item.weights.values()) == pytest.approx(1.0)

    def test_price_model_factor_is_absent_until_it_exists(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        assert "value_for_money" in DEFAULT_WEIGHTS
        assert all(item.scores.value_for_money is None for item in ranked)
        assert all("value_for_money" not in item.weights for item in ranked)

    def test_unrated_property_keeps_its_other_factors(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        unrated = next(i for i in ranked if i.accommodation.id == "14451771")
        assert unrated.scores.rating is None
        assert "rating" not in unrated.weights
        assert unrated.scores.budget_fit == 1.0

    def test_a_cheaper_better_rated_hotel_outranks_a_dearer_worse_one(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        position = {item.accommodation.id: item.rank for item in ranked}
        # APA Asakusa Tawaramachi: EUR 533.60, 8.3 from 7,388 reviews, restaurant.
        # Bespoke Shinjuku:        EUR 1,069.79, 8.7 from 1,984 reviews, no restaurant.
        assert position["2295449"] < position["4007031"]

    def test_every_ranked_option_respects_the_hard_budget(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        assert all(item.total_cost <= request.budget_total for item in ranked)

    def test_allowance_overrun_is_a_trade_off_not_a_veto(
        self, retrieved: tuple[Accommodation, ...]
    ) -> None:
        """The category split is a policy; only the *total* budget is a hard constraint.

        The top-ranked option for the demo request costs slightly more than the 45 %
        accommodation allowance and still wins, because its rating, amenities and location
        outweigh the overrun. That is the intended behaviour: capping at the allowance would
        turn a budgeting heuristic into a rule the domain never actually stated. The
        breakdown makes the trade-off visible instead of hiding it.
        """
        request = make_request()
        ranked = rank_accommodations(generate_candidates(retrieved, request).records, request)
        allowance = DEFAULT_BUDGET_POLICY.allowance(
            BudgetCategory.ACCOMMODATION, request.budget_total
        )
        best = ranked[0]
        assert best.total_cost > allowance
        assert best.total_cost <= request.budget_total
        assert 0.0 < best.scores.budget_fit < 1.0  # penalised, not disqualified
        assert best.scores.rating is not None and best.scores.rating > 0.85
