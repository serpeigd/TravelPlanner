"""The pipeline, in one place.

The demo, the evaluation harness and the API must exercise *the same* code path, or the
numbers in `docs/evaluation.md` describe something the API does not do. This function is that
path.

The explanation is deliberately not here. Everything up to and including constraint
validation is deterministic and offline; the explanation is the only stage that may call a
model, so keeping it outside lets the evaluation harness measure the decision-making with no
model server in the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from travel_intel.config import Settings, get_settings
from travel_intel.domain.models import DataSourceInfo, TripRequest
from travel_intel.ml.price_model import HedonicPriceModel
from travel_intel.ranking import generate_candidates, rank_accommodations
from travel_intel.ranking.candidates import CandidateSet
from travel_intel.retrieval.factory import build_providers
from travel_intel.services.planner import PlannedTrip, plan_trip


@dataclass(frozen=True)
class PipelineResult:
    """The plan, plus the record of how it was reached.

    The funnel and the sources are not decoration: they are what lets a response state how
    many options were considered and where every fact came from. Reconstructing them at the
    API layer would mean guessing at what the pipeline actually did.
    """

    trip: PlannedTrip
    candidates: CandidateSet
    sources: tuple[DataSourceInfo, ...]
    warnings: tuple[str, ...]


def run_pipeline(request: TripRequest, settings: Settings | None = None) -> PipelineResult:
    """Retrieve, filter, rank, plan, validate. Raises `NoCandidatesError` when it cannot."""
    resolved = settings or get_settings()
    accommodation_provider, activity_provider = build_providers(resolved)

    accommodations = accommodation_provider.search_accommodations(request)
    activities = activity_provider.search_activities(request)
    warnings = accommodations.warnings + activities.warnings

    candidates = generate_candidates(accommodations.records, request)
    ranked = rank_accommodations(candidates.records, request, price_model=HedonicPriceModel())
    trip = plan_trip(ranked, activities.records, request, retrieval_warnings=warnings)

    return PipelineResult(
        trip=trip,
        candidates=candidates,
        sources=(accommodations.source, activities.source, trip.assumptions.source()),
        warnings=warnings,
    )
