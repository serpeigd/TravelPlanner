"""The price model, its baselines, and the value factor it feeds.

The headline assertion — that the model beats a district median — is a regression test, not
a claim about hotels in general. The dataset is frozen, so the number is reproducible; if a
refactor quietly breaks the model, this fails.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from travel_intel.config import Settings
from travel_intel.domain.enums import Preference
from travel_intel.domain.models import TripRequest
from travel_intel.ml.price_model import (
    PRICE_FEATURES,
    DistrictMedianBaseline,
    GlobalMedianBaseline,
    HedonicPriceModel,
    cross_validate,
    prepare,
)
from travel_intel.ml.report import training_frame
from travel_intel.ranking import generate_candidates, rank_accommodations
from travel_intel.ranking.scoring import value_for_money
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


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    return training_frame()


class TestDataset:
    def test_trains_on_the_whole_market_not_the_affordable_subset(
        self, frame: pd.DataFrame
    ) -> None:
        """Training on post-filter candidates would truncate the target.

        Candidate generation drops everything above the traveller's budget. A model that
        never sees the expensive end would systematically under-predict what a good hotel
        costs, and every luxury property would then look like a bargain.
        """
        request = make_request()
        retrieved = SnapshotProvider(Settings().fixtures_dir).search_accommodations(request)
        affordable = generate_candidates(retrieved.records, request).records
        assert len(frame) == 30
        assert len(frame) > len(affordable)
        assert frame["price_per_night"].max() > 600

    def test_review_count_is_log_transformed(self, frame: pd.DataFrame) -> None:
        prepared = prepare(frame)
        assert "log_review_count" in prepared.columns
        assert set(PRICE_FEATURES) <= set(prepared.columns)

    def test_preparation_does_not_mutate_the_input(self, frame: pd.DataFrame) -> None:
        before = set(frame.columns)
        prepare(frame)
        assert set(frame.columns) == before


class TestBaselines:
    def test_global_median_predicts_one_number(self, frame: pd.DataFrame) -> None:
        baseline = GlobalMedianBaseline()
        baseline.fit(frame)
        predictions = baseline.predict(frame)
        assert len(set(predictions)) == 1
        assert predictions[0] == pytest.approx(frame["price_per_night"].median())

    def test_district_median_predicts_per_district(self, frame: pd.DataFrame) -> None:
        baseline = DistrictMedianBaseline()
        baseline.fit(frame)
        predictions = pd.Series(baseline.predict(frame), index=frame.index)
        taito = frame[frame["district"] == "Taito"]
        expected = taito["price_per_night"].median()
        assert predictions[taito.index].eq(expected).all()

    def test_unseen_district_falls_back_to_the_global_median(self, frame: pd.DataFrame) -> None:
        baseline = DistrictMedianBaseline()
        baseline.fit(frame)
        elsewhere = frame.head(1).copy()
        elsewhere["district"] = "Nowhere"
        assert baseline.predict(elsewhere)[0] == pytest.approx(frame["price_per_night"].median())


class TestHedonicModel:
    def test_predicts_positive_prices(self, frame: pd.DataFrame) -> None:
        model = HedonicPriceModel()
        predictions = model.fit_predict(frame)
        assert len(predictions) == len(frame)
        assert (predictions > 0).all()

    def test_beats_both_baselines_out_of_sample(self, frame: pd.DataFrame) -> None:
        results = cross_validate(
            frame, (GlobalMedianBaseline(), DistrictMedianBaseline(), HedonicPriceModel())
        )
        model = results["hedonic ridge (log price)"]
        district = results["baseline: district median"]
        global_median = results["baseline: global median"]
        assert model.mae < district.mae < global_median.mae
        assert model.mape < district.mape
        # Comfortably better, not a coin flip: roughly a 40 % cut in mean absolute error.
        assert model.mae < district.mae * 0.7

    def test_cross_validation_is_deterministic(self, frame: pd.DataFrame) -> None:
        first = cross_validate(frame, (HedonicPriceModel(),))
        second = cross_validate(frame, (HedonicPriceModel(),))
        assert first["hedonic ridge (log price)"] == second["hedonic ridge (log price)"]

    def test_every_estimator_sees_the_same_folds(self, frame: pd.DataFrame) -> None:
        results = cross_validate(frame, (GlobalMedianBaseline(), HedonicPriceModel()))
        counts = {metrics.n_observations for metrics in results.values()}
        assert len(counts) == 1  # paired comparison, not approximately comparable

    def test_star_rating_is_the_strongest_price_driver(self, frame: pd.DataFrame) -> None:
        model = HedonicPriceModel()
        model.fit(prepare(frame))
        coefficients = model.coefficients()
        assert coefficients["stars"] > 0
        assert coefficients["stars"] == max(coefficients.values())

    def test_distance_from_the_centre_lowers_price(self, frame: pd.DataFrame) -> None:
        model = HedonicPriceModel()
        model.fit(prepare(frame))
        assert model.coefficients()["distance_km"] < 0

    def test_guest_rating_barely_explains_price(self, frame: pd.DataFrame) -> None:
        """A finding, not a bug: what a hotel *is* prices it, not how guests scored it.

        Once stars and amenities are accounted for, the guest rating coefficient collapses
        to near zero. That is why rating and value are separate ranking factors — they carry
        different information.
        """
        model = HedonicPriceModel()
        model.fit(prepare(frame))
        coefficients = model.coefficients()
        assert abs(coefficients["rating"]) < abs(coefficients["stars"]) / 10


class TestValueFactor:
    def test_priced_as_predicted_is_neutral(self) -> None:
        assert value_for_money(100.0, 100.0) == pytest.approx(0.5)

    def test_cheaper_than_predicted_scores_above_neutral(self) -> None:
        assert value_for_money(80.0, 100.0) > 0.5

    def test_dearer_than_predicted_scores_below_neutral(self) -> None:
        assert value_for_money(130.0, 100.0) < 0.5

    def test_the_scale_is_proportional_not_absolute(self) -> None:
        """EUR 20 off a EUR 70 room is a discount; EUR 20 off a EUR 600 suite is noise."""
        cheap_discount = value_for_money(50.0, 70.0)
        luxury_discount = value_for_money(580.0, 600.0)
        assert cheap_discount > luxury_discount

    def test_extremes_are_clipped_to_the_unit_interval(self) -> None:
        assert value_for_money(1.0, 1000.0) == 1.0
        assert value_for_money(1000.0, 1.0) == 0.0

    def test_non_positive_prices_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            value_for_money(0.0, 100.0)


class TestRankingWithThePriceModel:
    @pytest.fixture
    def candidates(self) -> object:
        request = make_request()
        records = SnapshotProvider(Settings().fixtures_dir).search_accommodations(request)
        return generate_candidates(records.records, request).records

    def test_the_factor_activates_and_takes_its_weight(self, candidates: object) -> None:
        request = make_request()
        ranked = rank_accommodations(candidates, request, price_model=HedonicPriceModel())  # type: ignore[arg-type]
        assert all(item.scores.value_for_money is not None for item in ranked)
        assert all("value_for_money" in item.weights for item in ranked)
        assert all(sum(item.weights.values()) == pytest.approx(1.0) for item in ranked)

    def test_an_overpriced_property_is_demoted(self, candidates: object) -> None:
        """Bespoke Hotel Shinjuku: EUR 153/night for a 3-star with eight amenities.

        Its 8.7 guest rating carries it into the upper half on rating alone. The hedonic
        model prices it well under what it charges, and the value factor moves it down.
        """
        request = make_request()
        without = rank_accommodations(candidates, request)  # type: ignore[arg-type]
        with_model = rank_accommodations(candidates, request, price_model=HedonicPriceModel())  # type: ignore[arg-type]
        before = next(i for i in without if i.accommodation.id == "4007031")
        after = next(i for i in with_model if i.accommodation.id == "4007031")
        assert after.scores.value_for_money == 0.0
        assert after.rank > before.rank

    def test_ranking_with_the_model_is_deterministic(self, candidates: object) -> None:
        request = make_request()
        first = rank_accommodations(candidates, request, price_model=HedonicPriceModel())  # type: ignore[arg-type]
        second = rank_accommodations(  # type: ignore[arg-type]
            tuple(reversed(candidates)),  # type: ignore[call-overload]
            request,
            price_model=HedonicPriceModel(),
        )
        assert [i.accommodation.id for i in first] == [i.accommodation.id for i in second]

    def test_value_scores_span_a_useful_range(self, candidates: object) -> None:
        """A factor that returns the same number for everything ranks nothing."""
        ranked = rank_accommodations(  # type: ignore[arg-type]
            candidates, make_request(), price_model=HedonicPriceModel()
        )
        scores = np.array([i.scores.value_for_money for i in ranked], dtype="float64")
        assert scores.max() - scores.min() > 0.4
