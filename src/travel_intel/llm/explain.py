"""Explaining a decision that has already been made.

The LLM's second job, and the boundary is the whole point: by the time this module runs, the
accommodation is chosen, the itinerary is fixed, the budget is arithmetic and the constraints
have passed. Nothing the model writes can change any of it. It is turning a table into
sentences.

Two safeguards, in order:

1. **The payload is the contract.** The model is handed a compact JSON object and told to use
   nothing else. That same object builds the grounding context, so the check can never drift
   away from what was actually given.
2. **Failure falls back, visibly.** Malformed JSON, an unknown entity or an invented price
   discards the text and the deterministic template answers instead — with the reasons
   recorded on the `Explanation`, because a silent fallback is indistinguishable from a
   system that never tried.
"""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from travel_intel.domain.enums import Provenance
from travel_intel.domain.errors import LLMError
from travel_intel.domain.models import Explanation
from travel_intel.llm.client import LLMClient
from travel_intel.llm.grounding import build_context, check_grounding
from travel_intel.services.planner import PlannedTrip

MAX_ATTEMPTS = 2

SYSTEM_PROMPT = """You write a short, plain explanation of a travel recommendation.

You are given a JSON object describing a plan that has already been decided. Your only job
is to explain it in clear prose.

Reply with JSON only, in this exact shape:
{"summary": "...", "accommodation": "...", "itinerary": "...", "budget": "...",
 "referenced_ids": ["..."]}

Rules:
- Use ONLY facts present in the JSON you were given. Invent nothing.
- Every monetary figure you write must appear in that JSON. Never round a price to a
  friendlier number, and never estimate one.
- All prices are in EUR. Write EUR. Never write $, USD, yen or any other currency.
- Keep every figure's meaning: "total_cost" is for the whole stay, "price_per_night" is for
  one night. Never describe a total as a nightly rate.
- "rating" is a guest score out of 10. "stars" is a separate hotel classification out of 5.
  Do not mix them.
- Never name a hotel or activity that is not in the JSON.
- "referenced_ids" must list the id of every hotel and activity you mention.
- Two or three sentences per field. No markdown, no bullet points, no prose outside the JSON."""


class _Payload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    summary: str = ""
    accommodation: str = ""
    itinerary: str = ""
    budget: str = ""
    referenced_ids: list[str] = Field(default_factory=list)

    def as_text(self) -> str:
        parts = [self.summary, self.accommodation, self.itinerary, self.budget]
        return "\n\n".join(part.strip() for part in parts if part.strip())


class Explainer(Protocol):
    name: str

    def explain(self, trip: PlannedTrip) -> Explanation: ...


def build_payload(trip: PlannedTrip) -> dict[str, object]:
    """The complete set of facts the model may use — and nothing else.

    Deliberately compact. Every field here is either retrieved data or a number computed by
    deterministic code, and the grounding context is derived from this exact object.
    """
    selected = {activity.id: activity for activity in trip.itinerary.selected}
    recommended = trip.recommended
    return {
        "currency": trip.request.currency.value,
        "request": {
            "destination": trip.request.destination,
            "nights": trip.request.nights,
            "travelers": trip.request.travelers,
            "budget_total": trip.request.budget_total,
            "preferences": [p.value for p in trip.request.preferences],
        },
        "recommended": {
            "id": recommended.accommodation.id,
            "name": recommended.accommodation.name,
            "district": recommended.accommodation.neighborhood,
            "rating": recommended.accommodation.rating,
            "review_count": recommended.accommodation.review_count,
            "stars": recommended.accommodation.stars,
            "distance_to_center_km": recommended.accommodation.distance_to_center_km,
            "price_per_night": recommended.accommodation.price_per_night,
            "total_cost": recommended.total_cost,
            "score": recommended.overall,
            "score_breakdown": recommended.scores.as_dict(),
        },
        "budget": {
            "accommodation_cost": trip.budget.accommodation,
            "activities_cost": trip.budget.activities,
            "food_cost": trip.budget.food,
            "transport_cost": trip.budget.transport,
            "total_cost": trip.budget.total,
            "budget_total": trip.budget.budget_total,
            "remaining_amount": trip.budget.remaining,
            "food_and_transport_are_estimates": True,
        },
        "itinerary": [
            {
                "day": day.day_index,
                "date": day.day.isoformat(),
                "activity_id": activity_id,
                "name": selected[activity_id].name,
                "cost": day.estimated_cost,
            }
            for day in trip.itinerary.days
            for activity_id in day.activity_ids
        ],
        "alternatives": [
            {
                "id": option.accommodation.id,
                "name": option.accommodation.name,
                "total_cost": option.total_cost,
                "score": option.overall,
            }
            for option in trip.alternatives
        ],
        "warnings": [violation.message for violation in trip.constraints.violations],
    }


class TemplateExplainer:
    """Deterministic prose assembled from the payload.

    Grounded by construction: it can only format numbers it was handed. It is the fallback
    when the model is unavailable or its output fails the check, and it is what the test
    suite and CI use so neither depends on a model server.
    """

    name = "template"

    def explain(self, trip: PlannedTrip) -> Explanation:
        return self.render(trip, reasons=())

    def render(self, trip: PlannedTrip, reasons: tuple[str, ...]) -> Explanation:
        hotel = trip.recommended.accommodation
        budget = trip.budget
        request = trip.request
        activities = trip.itinerary.selected

        rating = f", rated {hotel.rating}/10" if hotel.rating is not None else ""
        district = f" in {hotel.neighborhood}" if hotel.neighborhood else ""
        covered = ", ".join(p.value for p in trip.itinerary.covered_preferences) or "none"

        text = (
            f"{hotel.name}{district}{rating} is the best fit for {request.travelers} "
            f"travellers in {request.destination} over {request.nights} nights: "
            f"{hotel.price_per_night:.2f} EUR per night, {trip.recommended.total_cost:.2f} EUR "
            f"for the stay.\n\n"
            f"The itinerary books {len(activities)} activities across {request.nights} days, "
            f"covering these interests: {covered}.\n\n"
            f"Estimated total {budget.total:.2f} EUR against a budget of "
            f"{budget.budget_total:.2f} EUR, leaving {budget.remaining:.2f} EUR. "
            f"Accommodation {budget.accommodation:.2f} EUR and activities "
            f"{budget.activities:.2f} EUR are retrieved prices; food {budget.food:.2f} EUR and "
            f"local transport {budget.transport:.2f} EUR are planning estimates."
        )
        return Explanation(
            text=text,
            provenance=Provenance.SYNTHETIC,
            grounded=True,
            model=None,
            referenced_ids=(hotel.id, *(a.id for a in activities)),
            rejection_reasons=reasons,
        )


class LLMExplainer:
    """Model-written prose, admitted only if it survives the grounding check."""

    def __init__(self, client: LLMClient, fallback: TemplateExplainer | None = None) -> None:
        self.name = client.name
        self._client = client
        self._fallback = fallback or TemplateExplainer()

    def explain(self, trip: PlannedTrip) -> Explanation:
        payload = build_payload(trip)
        context = build_context(payload)
        prompt = json.dumps(payload, ensure_ascii=False, indent=None)

        reasons: list[str] = []
        for _ in range(MAX_ATTEMPTS):
            try:
                raw = self._client.complete_json(SYSTEM_PROMPT, prompt)
                parsed = _Payload.model_validate(json.loads(raw))
            except (LLMError, json.JSONDecodeError, ValidationError) as error:
                reasons.append(f"invalid model output: {error}")
                continue

            text = parsed.as_text()
            if not text:
                reasons.append("model returned an empty explanation")
                continue

            violations = check_grounding(text, parsed.referenced_ids, context)
            if violations:
                reasons.extend(violations)
                continue

            return Explanation(
                text=text,
                provenance=Provenance.MODEL_GENERATED,
                grounded=True,
                model=self.name,
                referenced_ids=tuple(parsed.referenced_ids),
            )

        # The text is discarded, never repaired. A "corrected" explanation is one nobody
        # validated: fixing a hallucinated price still leaves prose built around it.
        return self._fallback.render(trip, reasons=tuple(reasons))
