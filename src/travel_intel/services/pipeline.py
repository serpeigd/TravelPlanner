"""The pipeline, in one place.

The demo and the evaluation harness must exercise *the same* code path, or the numbers in
`docs/evaluation.md` describe something the demo does not do. This function is that path.

The explanation is deliberately not here. Everything up to and including constraint
validation is deterministic and offline; the explanation is the only stage that may call a
model, and keeping it outside means the evaluation harness can measure the decision-making
without a model server in the loop.
"""

from __future__ import annotations

from travel_intel.config import Settings, get_settings
from travel_intel.domain.models import TripRequest
from travel_intel.ml.price_model import HedonicPriceModel
from travel_intel.ranking import generate_candidates, rank_accommodations
from travel_intel.retrieval.factory import build_providers
from travel_intel.services.planner import PlannedTrip, plan_trip


def run_pipeline(request: TripRequest, settings: Settings | None = None) -> PlannedTrip:
    """Retrieve, filter, rank, plan, validate. Raises `NoCandidatesError` when it cannot."""
    resolved = settings or get_settings()
    accommodation_provider, activity_provider = build_providers(resolved)

    accommodations = accommodation_provider.search_accommodations(request)
    activities = activity_provider.search_activities(request)
    candidates = generate_candidates(accommodations.records, request)
    ranked = rank_accommodations(candidates.records, request, price_model=HedonicPriceModel())
    return plan_trip(
        ranked,
        activities.records,
        request,
        retrieval_warnings=accommodations.warnings + activities.warnings,
    )
