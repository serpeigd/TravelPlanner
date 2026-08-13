"""The top-level entry point: a request in, one validated response object out.

This module is where the deterministic pipeline and the LLM layer finally meet, and it sits
above both so neither has to know about the other. The ordering is the architecture: by the
time `build_explainer` is called, `run_pipeline` has already produced a plan that passed every
hard constraint, so there is nothing for the model to talk its way around.

`TripRecommendation` was defined in M1, before any of the stages existed. Assembling it here
without changing it is the payoff for writing the contracts first.
"""

from __future__ import annotations

from travel_intel.config import Settings, get_settings
from travel_intel.domain.models import QualitySignals, TripRecommendation, TripRequest
from travel_intel.llm.factory import build_explainer
from travel_intel.services.pipeline import PipelineResult, run_pipeline


def recommend(request: TripRequest, settings: Settings | None = None) -> TripRecommendation:
    """Plan a trip and explain it. Raises `NoCandidatesError` if no valid plan exists."""
    resolved = settings or get_settings()
    result = run_pipeline(request, resolved)
    explanation = build_explainer(resolved).explain(result.trip)

    trip = result.trip
    return TripRecommendation(
        request=request,
        recommended=trip.recommended,
        alternatives=trip.alternatives,
        itinerary=trip.itinerary.days,
        budget=trip.budget,
        constraints=trip.constraints,
        quality=quality_signals(result, explanation.grounded),
        explanation=explanation,
        data_sources=result.sources,
    )


def quality_signals(result: PipelineResult, explanation_grounded: bool | None) -> QualitySignals:
    """Per-request evaluation, shipped *with* the answer rather than in a separate report.

    A recommendation that arrives without any indication of how much was known, how many
    options were considered or whether anything was compromised asks the user to trust it
    blindly. These are the same measurements `docs/evaluation.md` aggregates over the golden
    set — here they describe the single request in front of you.
    """
    trip = result.trip
    stated = len(trip.request.preferences)
    covered = len(trip.itinerary.covered_preferences)
    completeness = [record.data_completeness for record in result.candidates.records]

    return QualitySignals(
        budget_compliant=trip.budget.total <= trip.request.budget_total,
        hard_violations=trip.constraints.hard_violation_count,
        preference_coverage=round(covered / stated, 4) if stated else 1.0,
        mean_data_completeness=round(sum(completeness) / len(completeness), 4)
        if completeness
        else 0.0,
        candidates_retrieved=result.candidates.retrieved,
        candidates_after_filters=result.candidates.kept,
        explanation_grounded=explanation_grounded,
    )
