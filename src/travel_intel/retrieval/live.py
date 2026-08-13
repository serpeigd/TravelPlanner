"""Live provider: the seam where a real API client goes.

Deliberately not implemented, and deliberately not faked.

The snapshot in `travel_intel/data/fixtures/` is real Booking.com data, but it was captured
through an
assistant-side MCP connector — a tool available to the developer, not an HTTP endpoint this
process can call, and not one that ships with credentials. Writing an untested client
against an API we cannot exercise would mean shipping code that claims to work.

What a production implementation would need is small and well-defined: an httpx client, an
API key from the environment, retry with backoff plus a circuit breaker, a response cache
keyed by (destination, dates, party), and a mapping into the same `Accommodation` /
`Activity` contracts this class already promises. Everything downstream stays untouched —
that is the point of the Protocol in `base.py`.
"""

from __future__ import annotations

from travel_intel.domain.errors import ProviderError
from travel_intel.domain.models import Accommodation, Activity, TripRequest
from travel_intel.retrieval.base import RetrievalResult

_MESSAGE = (
    "live data mode is not implemented: no provider credentials are configured. "
    "Run with TRAVEL_INTEL_DATA_MODE=snapshot (the default) to use the captured dataset."
)


class LiveProvider:
    """Satisfies the provider Protocols and fails loudly instead of silently degrading."""

    def search_accommodations(self, request: TripRequest) -> RetrievalResult[Accommodation]:
        raise ProviderError(_MESSAGE)

    def search_activities(self, request: TripRequest) -> RetrievalResult[Activity]:
        raise ProviderError(_MESSAGE)
