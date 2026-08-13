"""Provider contracts.

The pipeline depends on these Protocols only, never on a concrete provider. Swapping the
frozen snapshot for a live API is a constructor change, not a rewrite.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Protocol

from travel_intel.domain.models import Accommodation, Activity, DataSourceInfo, TripRequest

DESTINATION_ALIASES = {
    "tokio": "tokyo",  # Spanish/Italian exonym; the user types the request in their language.
}


def destination_key(destination: str) -> str:
    """Normalise a free-text destination to a lookup key.

    'Tokio, Japón' and ' TOKYO ' both become 'tokyo'. Only the leading component is used:
    anything after a comma is a disambiguator (country, region), not the place itself.
    """
    head = destination.split(",")[0].strip().lower()
    stripped = unicodedata.normalize("NFKD", head).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", stripped).strip("-")
    return DESTINATION_ALIASES.get(slug, slug)


@dataclass(frozen=True)
class RetrievalResult[R: (Accommodation, Activity)]:
    """Records plus the provenance needed to show the user where they came from."""

    records: tuple[R, ...]
    source: DataSourceInfo
    warnings: tuple[str, ...] = field(default=())


class AccommodationProvider(Protocol):
    def search_accommodations(self, request: TripRequest) -> RetrievalResult[Accommodation]: ...


class ActivityProvider(Protocol):
    def search_activities(self, request: TripRequest) -> RetrievalResult[Activity]: ...
