"""End-to-end demo: `python -m travel_intel.demo`.

Runs the whole pipeline for the reference request and prints what each stage produced.

Everything up to the explanation is offline and deterministic — it reads the frozen snapshot.
The explanation follows `TRAVEL_INTEL_LLM_PROVIDER`: `ollama` (the default) calls the local
model and takes ~100 s on CPU; `fake` uses the deterministic template and returns instantly.
"""

from __future__ import annotations

import time
from datetime import date

from travel_intel.config import get_settings
from travel_intel.domain.enums import Preference
from travel_intel.domain.models import TripRequest
from travel_intel.llm.factory import build_explainer
from travel_intel.services.pipeline import run_pipeline

REFERENCE_REQUEST = TripRequest(
    destination="Tokyo",
    start_date=date(2026, 9, 10),
    end_date=date(2026, 9, 17),
    travelers=2,
    budget_total=2500,
    preferences=(Preference.FOOD, Preference.CULTURE, Preference.NATURE),
)


def main() -> None:
    settings = get_settings()
    request = REFERENCE_REQUEST
    print(
        f"{request.destination} | {request.start_date}..{request.end_date} | "
        f"{request.travelers} travellers | {request.budget_total:.0f} "
        f"{request.currency.value} | {', '.join(p.value for p in request.preferences)}\n"
    )

    trip = run_pipeline(request, settings)
    hotel = trip.recommended

    print("RECOMMENDED")
    print(f"  {hotel.accommodation.name}  ({hotel.accommodation.neighborhood})")
    print(f"  score {hotel.overall:.3f}  |  {hotel.total_cost:.2f} EUR for the stay")
    for factor, value in sorted(hotel.scores.as_dict().items()):
        print(f"    {factor:<20} {value:.3f}   (weight {hotel.weights.get(factor, 0):.3f})")

    print("\nALTERNATIVES")
    for option in trip.alternatives:
        print(
            f"  {option.rank:>2}. {option.accommodation.name[:44]:<44} "
            f"{option.overall:.3f}  {option.total_cost:8.2f} EUR"
        )

    if trip.rejected:
        print("\nREJECTED BY CONSTRAINTS (the ranking preferred these)")
        for refused in trip.rejected:
            print(f"  {refused.rank:>2}. {refused.name[:44]:<44} {', '.join(refused.codes)}")

    print("\nITINERARY")
    selected = {activity.id: activity for activity in trip.itinerary.selected}
    for day in trip.itinerary.days:
        names = ", ".join(selected[i].name for i in day.activity_ids) or "-"
        print(f"  {day.day}  {day.estimated_cost:7.2f} EUR  {names[:56]}")

    budget = trip.budget
    print("\nBUDGET")
    for label, amount, note in (
        ("accommodation", budget.accommodation, "retrieved"),
        ("activities", budget.activities, "retrieved"),
        ("food", budget.food, "estimate"),
        ("transport", budget.transport, "estimate"),
    ):
        print(f"  {label:<16} {amount:9.2f} EUR   ({note})")
    print(
        f"  {'total':<16} {budget.total:9.2f} EUR   of {budget.budget_total:.2f} "
        f"({budget.utilization:.1%}), {budget.remaining:.2f} left"
    )

    print(f"\nCONSTRAINTS  valid={trip.constraints.is_valid}")
    for violation in trip.constraints.violations:
        print(f"  [{violation.severity.value}] {violation.code}: {violation.message}")

    print(f"\nEXPLANATION  (provider: {settings.llm_provider.value})")
    started = time.time()
    explanation = build_explainer(settings).explain(trip)
    print(
        f"  provenance={explanation.provenance.value}  grounded={explanation.grounded}  "
        f"model={explanation.model}  {time.time() - started:.1f}s"
    )
    for reason in explanation.rejection_reasons:
        print(f"  REJECTED: {reason}")
    print()
    for line in explanation.text.splitlines():
        print(f"  {line}")


if __name__ == "__main__":
    main()
