"""Retrieval layer: one interface, interchangeable providers."""

from travel_intel.retrieval.base import (
    AccommodationProvider,
    ActivityProvider,
    RetrievalResult,
    destination_key,
)
from travel_intel.retrieval.factory import build_providers

__all__ = [
    "AccommodationProvider",
    "ActivityProvider",
    "RetrievalResult",
    "build_providers",
    "destination_key",
]
