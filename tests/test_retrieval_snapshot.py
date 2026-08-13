"""Retrieval tests.

These run against the committed Tokyo fixture — the same data the demo and the evaluation
harness use. That is the point of a snapshot: the test suite exercises the real dataset.
"""

from datetime import date
from pathlib import Path

import pytest

import travel_intel
from travel_intel.config import DataMode, LLMProvider, Settings
from travel_intel.domain.enums import Preference, Provenance
from travel_intel.domain.errors import NoCandidatesError, ProviderError
from travel_intel.domain.models import GeoPoint, TripRequest
from travel_intel.retrieval.base import destination_key
from travel_intel.retrieval.factory import build_providers
from travel_intel.retrieval.geo import haversine_km
from travel_intel.retrieval.snapshot import SnapshotProvider

FIXTURES = Settings().fixtures_dir


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
def provider() -> SnapshotProvider:
    return SnapshotProvider(FIXTURES)


class TestPackaging:
    def test_the_snapshot_ships_inside_the_package(self) -> None:
        """Regression: the fixture path used to be resolved relative to the repository root.

        That works under an editable install, where `config.py` really does sit in the source
        tree. Installed normally — a wheel, a container, Streamlit Cloud — the same file lives
        in `site-packages/travel_intel/` and walking up two levels lands in the Python library
        directory. The deployed app reported `Available: none` while every local test passed,
        because every local test ran against an editable install.
        """
        package_dir = Path(travel_intel.__file__).resolve().parent
        fixtures = Settings().fixtures_dir.resolve()
        assert fixtures.is_relative_to(package_dir), (
            f"{fixtures} is outside {package_dir}: it will not survive a non-editable install"
        )
        assert fixtures.is_dir()


class TestDestinationKey:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("Tokyo", "tokyo"),
            ("  TOKYO  ", "tokyo"),
            ("Tokyo, Japan", "tokyo"),
            ("Tokio, Japón", "tokyo"),  # Spanish exonym plus accent
            ("New York, NY", "new-york"),
        ],
    )
    def test_normalisation(self, raw: str, expected: str) -> None:
        assert destination_key(raw) == expected


class TestGeo:
    def test_haversine_matches_known_distance(self) -> None:
        # Tokyo Station -> Shinjuku Station is ~6.4 km in a straight line.
        tokyo_station = GeoPoint(lat=35.681236, lon=139.767125)
        shinjuku_station = GeoPoint(lat=35.690921, lon=139.700258)
        assert haversine_km(tokyo_station, shinjuku_station) == pytest.approx(6.13, abs=0.3)

    def test_distance_to_self_is_zero(self) -> None:
        point = GeoPoint(lat=35.681236, lon=139.767125)
        assert haversine_km(point, point) == 0.0


class TestAccommodationRetrieval:
    def test_returns_the_whole_captured_dataset(self, provider: SnapshotProvider) -> None:
        result = provider.search_accommodations(make_request())
        assert len(result.records) == 30
        assert result.source.provenance is Provenance.SNAPSHOT
        assert result.source.record_count == 30

    def test_every_record_is_tagged_as_snapshot(self, provider: SnapshotProvider) -> None:
        result = provider.search_accommodations(make_request())
        assert {r.provenance for r in result.records} == {Provenance.SNAPSHOT}

    def test_stay_total_is_converted_to_a_nightly_rate(self, provider: SnapshotProvider) -> None:
        result = provider.search_accommodations(make_request())
        hotel = next(r for r in result.records if r.id == "1118391")
        # Captured total: EUR 643.81 for 7 nights.
        assert hotel.price_per_night == pytest.approx(91.97, abs=0.01)
        assert hotel.total_cost(7) == pytest.approx(643.81, abs=0.05)

    def test_distance_to_center_is_derived_from_coordinates(
        self, provider: SnapshotProvider
    ) -> None:
        result = provider.search_accommodations(make_request())
        hotel = next(r for r in result.records if r.id == "1118391")
        assert hotel.distance_to_center_km is not None
        assert 5.0 < hotel.distance_to_center_km < 9.0

    def test_missing_rating_stays_missing(self, provider: SnapshotProvider) -> None:
        result = provider.search_accommodations(make_request())
        unrated = next(r for r in result.records if r.id == "14451771")
        assert unrated.rating is None
        assert unrated.review_count is None
        # And it is penalised on completeness rather than silently scored as zero.
        assert unrated.data_completeness < 1.0

    def test_ratings_stay_on_the_domain_scale(self, provider: SnapshotProvider) -> None:
        result = provider.search_accommodations(make_request())
        ratings = [r.rating for r in result.records if r.rating is not None]
        assert ratings and all(0.0 <= value <= 10.0 for value in ratings)

    def test_matching_request_produces_no_warnings(self, provider: SnapshotProvider) -> None:
        assert provider.search_accommodations(make_request()).warnings == ()

    def test_different_dates_are_flagged(self, provider: SnapshotProvider) -> None:
        request = make_request(start_date=date(2026, 10, 1), end_date=date(2026, 10, 8))
        warnings = provider.search_accommodations(request).warnings
        assert any("captured for" in w for w in warnings)

    def test_different_stay_length_is_flagged_as_extrapolation(
        self, provider: SnapshotProvider
    ) -> None:
        request = make_request(end_date=date(2026, 9, 13))
        warnings = provider.search_accommodations(request).warnings
        assert any("extrapolated" in w for w in warnings)

    def test_larger_party_than_captured_is_flagged(self, provider: SnapshotProvider) -> None:
        warnings = provider.search_accommodations(make_request(travelers=4)).warnings
        assert any("capacity" in w for w in warnings)

    def test_unknown_destination_is_a_domain_error(self, provider: SnapshotProvider) -> None:
        with pytest.raises(NoCandidatesError, match="no snapshot captured"):
            provider.search_accommodations(make_request(destination="Kyoto"))

    def test_retrieval_is_deterministic(self, provider: SnapshotProvider) -> None:
        first = provider.search_accommodations(make_request())
        second = SnapshotProvider(FIXTURES).search_accommodations(make_request())
        assert first.records == second.records


class TestActivityRetrieval:
    def test_returns_the_whole_captured_dataset(self, provider: SnapshotProvider) -> None:
        result = provider.search_activities(make_request())
        assert len(result.records) == 27
        assert result.source.provenance is Provenance.SNAPSHOT

    def test_provider_categories_map_to_the_domain_vocabulary(
        self, provider: SnapshotProvider
    ) -> None:
        result = provider.search_activities(make_request())
        skytree = next(r for r in result.records if r.id == "PRjr2HDdI2qd")
        assert skytree.categories == (Preference.CULTURE, Preference.ART)

    def test_multi_category_activities_keep_every_tag(self, provider: SnapshotProvider) -> None:
        result = provider.search_activities(make_request())
        # Returned by both the food_drinks and nature_outdoor searches.
        golden_gai = next(r for r in result.records if r.id == "PRGECJsLF2qd")
        assert set(golden_gai.categories) == {Preference.FOOD, Preference.NATURE}

    def test_rating_is_rescaled_from_five_to_ten(self, provider: SnapshotProvider) -> None:
        result = provider.search_activities(make_request())
        skytree = next(r for r in result.records if r.id == "PRjr2HDdI2qd")
        assert skytree.rating == pytest.approx(8.72)  # 4.36 on the provider's 0-5 scale

    def test_zero_reviews_means_unknown_not_bad(self, provider: SnapshotProvider) -> None:
        result = provider.search_activities(make_request())
        unrated = next(r for r in result.records if r.id == "PRAtySGWfdFZ")
        assert unrated.rating is None

    def test_free_and_paid_activities_price_per_person(self, provider: SnapshotProvider) -> None:
        result = provider.search_activities(make_request())
        skytree = next(r for r in result.records if r.id == "PRjr2HDdI2qd")
        assert skytree.cost_for(2) == pytest.approx(24.12)

    def test_every_preference_in_the_demo_request_is_covered(
        self, provider: SnapshotProvider
    ) -> None:
        result = provider.search_activities(make_request())
        covered = {c for record in result.records for c in record.categories}
        assert {Preference.FOOD, Preference.CULTURE, Preference.NATURE} <= covered


class TestProviderFactory:
    def test_snapshot_mode_is_the_default(self) -> None:
        accommodations, activities = build_providers(Settings())
        assert isinstance(accommodations, SnapshotProvider)
        assert isinstance(activities, SnapshotProvider)

    def test_live_mode_fails_loudly_instead_of_degrading(self) -> None:
        settings = Settings(data_mode=DataMode.LIVE, llm_provider=LLMProvider.FAKE)
        accommodations, _ = build_providers(settings)
        with pytest.raises(ProviderError, match="not implemented"):
            accommodations.search_accommodations(make_request())
