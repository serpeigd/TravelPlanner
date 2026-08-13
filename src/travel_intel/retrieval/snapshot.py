"""Snapshot provider: real provider data, frozen as a versioned fixture.

This is the default source. It is offline, byte-for-byte reproducible, and every record it
returns is tagged `Provenance.SNAPSHOT` so the UI can say exactly what the user is looking
at. The fixtures themselves are validated on load — a malformed capture fails loudly here
rather than producing silently wrong recommendations downstream.
"""

from __future__ import annotations

import json
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from travel_intel.domain.enums import Preference, Provenance
from travel_intel.domain.errors import NoCandidatesError, ProviderError
from travel_intel.domain.models import (
    Accommodation,
    Activity,
    DataSourceInfo,
    GeoPoint,
    TripRequest,
)
from travel_intel.retrieval.base import RetrievalResult, destination_key
from travel_intel.retrieval.geo import city_center, haversine_km

PROVIDER_NAME = "booking.com"

ACTIVITY_RATING_SCALE_FACTOR = 2.0
"""The attractions endpoint scores 0-5 while accommodations score 0-10. One domain scale."""

DEFAULT_ACTIVITY_HOURS = 3.0
"""The provider exposes no duration. A single documented default beats a fabricated one."""

CATEGORY_TO_PREFERENCES: dict[str, tuple[Preference, ...]] = {
    "food_drinks": (Preference.FOOD,),
    "museums_arts_culture": (Preference.CULTURE, Preference.ART),
    "nature_outdoor": (Preference.NATURE,),
    "tours": (Preference.CULTURE,),
    "entertainment_tickets": (Preference.NIGHTLIFE,),
    "workshop_classes": (Preference.CULTURE,),
    "travel_services_rental": (),
}
"""Provider search category -> our vocabulary.

Each activity is tagged with the categories whose *search* returned it, so the tag is the
provider's claim rather than our inference. The provider's own tagging is noisy (a bar tour
surfaces under `nature_outdoor`); that noise is left visible on purpose.
"""


class _Fixture(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: int
    provider: str
    captured_at: date
    currency: str


class _AccommodationRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    url: str | None = None
    price_total_stay: float = Field(gt=0)
    review_score: float | None = None
    review_count: int | None = None
    stars: int | None = None
    district: str | None = None
    lat: float | None = None
    lon: float | None = None
    amenities: tuple[str, ...] = ()


class _AccommodationFixture(_Fixture):
    checkin_date: date
    checkout_date: date
    nights: int = Field(gt=0)
    searched_adults: int = Field(gt=0)
    records: tuple[_AccommodationRecord, ...]


class _ActivityRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    url: str | None = None
    price_per_person: float = Field(ge=0)
    review_score: float | None = None
    review_count: int | None = None
    address: str | None = None
    source_queries: tuple[str, ...] = ()


class _ActivityFixture(_Fixture):
    start_date: date
    end_date: date
    records: tuple[_ActivityRecord, ...]


@lru_cache(maxsize=8)
def _read_json(path: Path) -> str:
    if not path.is_file():
        raise ProviderError(f"snapshot file not found: {path}")
    # Fixtures contain Japanese addresses; the encoding must never depend on the OS locale.
    return path.read_text(encoding="utf-8")


def _load_accommodations(path: Path) -> _AccommodationFixture:
    return _AccommodationFixture.model_validate(json.loads(_read_json(path)))


def _load_activities(path: Path) -> _ActivityFixture:
    return _ActivityFixture.model_validate(json.loads(_read_json(path)))


class SnapshotProvider:
    """Serves both accommodations and activities from one captured destination folder."""

    def __init__(self, fixtures_dir: Path) -> None:
        self._fixtures_dir = fixtures_dir

    # -- discovery ---------------------------------------------------------------------

    def available_destinations(self) -> dict[str, Path]:
        """Map destination key -> fixture folder, from the `<key>_<start>_<end>` convention."""
        if not self._fixtures_dir.is_dir():
            return {}
        found: dict[str, Path] = {}
        for folder in sorted(self._fixtures_dir.iterdir()):
            if not folder.is_dir():
                continue
            parts = folder.name.rsplit("_", 2)
            if len(parts) == 3:
                found.setdefault(parts[0], folder)
        return found

    def _folder_for(self, request: TripRequest) -> Path:
        destinations = self.available_destinations()
        key = destination_key(request.destination)
        folder = destinations.get(key)
        if folder is None:
            known = ", ".join(sorted(destinations)) or "none"
            raise NoCandidatesError(
                f"no snapshot captured for '{request.destination}' "
                f"(resolved to '{key}'). Available: {known}."
            )
        return folder

    # -- accommodations ----------------------------------------------------------------

    def search_accommodations(self, request: TripRequest) -> RetrievalResult[Accommodation]:
        folder = self._folder_for(request)
        fixture = _load_accommodations(folder / "accommodations.json")
        center = city_center(destination_key(request.destination))

        records = tuple(
            self._to_accommodation(record, fixture, center) for record in fixture.records
        )
        warnings = self._accommodation_warnings(request, fixture)
        source = DataSourceInfo(
            name=f"{fixture.provider} snapshot ({folder.name})",
            provenance=Provenance.SNAPSHOT,
            record_count=len(records),
            retrieved_at=datetime.combine(fixture.captured_at, time.min),
        )
        return RetrievalResult(records=records, source=source, warnings=warnings)

    def _to_accommodation(
        self,
        record: _AccommodationRecord,
        fixture: _AccommodationFixture,
        center: GeoPoint | None,
    ) -> Accommodation:
        location = (
            GeoPoint(lat=record.lat, lon=record.lon)
            if record.lat is not None and record.lon is not None
            else None
        )
        return Accommodation(
            id=record.id,
            name=record.name,
            provider=fixture.provider,
            provenance=Provenance.SNAPSHOT,
            # The provider quotes the whole stay for the searched party; the domain works
            # in nightly rates. Dividing here keeps that conversion in exactly one place.
            price_per_night=record.price_total_stay / fixture.nights,
            max_occupancy=fixture.searched_adults,
            rating=record.review_score,
            review_count=record.review_count,
            stars=record.stars,
            location=location,
            neighborhood=record.district,
            distance_to_center_km=(haversine_km(location, center) if location and center else None),
            amenities=record.amenities,
            source_url=record.url,
            retrieved_at=datetime.combine(fixture.captured_at, time.min),
        )

    @staticmethod
    def _accommodation_warnings(
        request: TripRequest, fixture: _AccommodationFixture
    ) -> tuple[str, ...]:
        """Flag every way the request diverges from what was actually captured.

        Silence here would be the dishonest option: prices captured for one stay length and
        party size do not automatically hold for another.
        """
        warnings: list[str] = []
        captured = (fixture.checkin_date, fixture.checkout_date)
        if (request.start_date, request.end_date) != captured:
            warnings.append(
                f"snapshot prices were captured for {captured[0]}..{captured[1]}, "
                f"not {request.start_date}..{request.end_date}"
            )
        if request.nights != fixture.nights:
            warnings.append(
                f"nightly rates derive from a {fixture.nights}-night stay and are extrapolated "
                f"to {request.nights} nights"
            )
        if request.travelers > fixture.searched_adults:
            warnings.append(
                f"snapshot was searched for {fixture.searched_adults} guests; capacity for "
                f"{request.travelers} is unverified"
            )
        return tuple(warnings)

    # -- activities --------------------------------------------------------------------

    def search_activities(self, request: TripRequest) -> RetrievalResult[Activity]:
        folder = self._folder_for(request)
        fixture = _load_activities(folder / "activities.json")

        records = tuple(self._to_activity(record, fixture) for record in fixture.records)
        source = DataSourceInfo(
            name=f"{fixture.provider} snapshot ({folder.name})",
            provenance=Provenance.SNAPSHOT,
            record_count=len(records),
            retrieved_at=datetime.combine(fixture.captured_at, time.min),
        )
        warnings: tuple[str, ...] = ()
        if (request.start_date, request.end_date) != (fixture.start_date, fixture.end_date):
            warnings = (
                f"activity availability was captured for {fixture.start_date}..{fixture.end_date}",
            )
        return RetrievalResult(records=records, source=source, warnings=warnings)

    @staticmethod
    def _to_activity(record: _ActivityRecord, fixture: _ActivityFixture) -> Activity:
        categories = tuple(
            dict.fromkeys(
                preference
                for query in record.source_queries
                for preference in CATEGORY_TO_PREFERENCES.get(query, ())
            )
        )
        return Activity(
            id=record.id,
            name=record.name,
            provider=fixture.provider,
            provenance=Provenance.SNAPSHOT,
            categories=categories,
            price_per_person=record.price_per_person,
            duration_hours=DEFAULT_ACTIVITY_HOURS,
            rating=(
                record.review_score * ACTIVITY_RATING_SCALE_FACTOR
                if record.review_score is not None
                else None
            ),
            review_count=record.review_count,
            neighborhood=None,
            source_url=record.url,
            retrieved_at=datetime.combine(fixture.captured_at, time.min),
        )
