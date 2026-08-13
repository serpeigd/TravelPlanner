"""Streamlit front end: `streamlit run src/travel_intel/ui/streamlit_app.py`.

Deliberately plain. The point of this project is the reasoning underneath, and a polished
interface would only hide it — so the screen shows the things a reviewer should be able to
interrogate: the score broken into its factors, which options the constraints refused,
which budget lines are retrieved prices and which are assumptions, and where every fact came
from.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from travel_intel.config import LLMProvider, Settings
from travel_intel.domain.enums import Preference
from travel_intel.domain.errors import NoCandidatesError, ProviderError
from travel_intel.domain.models import TripRequest
from travel_intel.llm.factory import build_explainer, build_interpreter
from travel_intel.recommend import quality_signals
from travel_intel.services.pipeline import run_pipeline

st.set_page_config(page_title="Travel Intelligence", layout="wide")

PROVENANCE_LABEL = {
    "snapshot": "retrieved (frozen snapshot of real provider data)",
    "real_api": "retrieved live",
    "synthetic": "our own estimate",
    "model_generated": "written by the language model",
}


def sidebar() -> tuple[TripRequest | None, Settings, str]:
    st.sidebar.header("Request")
    destination = st.sidebar.text_input("Destination", "Tokyo")
    start = st.sidebar.date_input("Check-in", date(2026, 9, 10))
    end = st.sidebar.date_input("Check-out", date(2026, 9, 17))
    travelers = st.sidebar.number_input("Travellers", min_value=1, max_value=12, value=2)
    budget = st.sidebar.number_input("Budget (EUR)", min_value=100, value=2500, step=100)

    st.sidebar.header("Preferences")
    notes = st.sidebar.text_area(
        "Describe the trip (optional)",
        placeholder="we love street food and quiet old temples",
        help="Interpreted by the LLM into the controlled vocabulary below.",
    )

    use_model = st.sidebar.toggle(
        "Use the local model (Ollama)",
        value=False,
        help="Off: deterministic keyword matching and a templated explanation. "
        "On: llama3.1:8b, roughly two minutes for the explanation.",
    )
    settings = Settings(
        llm_provider=LLMProvider.OLLAMA if use_model else LLMProvider.FAKE,
    )

    interpreted: list[Preference] = []
    if notes.strip():
        with st.spinner("Interpreting…"):
            result = build_interpreter(settings).interpret(notes)
        interpreted = list(result.preferences)
        st.sidebar.caption(f"Interpreted by: `{result.interpreter}`")
        if result.unmapped:
            st.sidebar.warning(
                "Could not be expressed in the vocabulary: " + ", ".join(result.unmapped)
            )

    preferences = st.sidebar.multiselect(
        "Vocabulary",
        options=list(Preference),
        default=interpreted or [Preference.FOOD, Preference.CULTURE, Preference.NATURE],
        format_func=lambda preference: preference.value,
    )

    try:
        request = TripRequest(
            destination=destination,
            start_date=start,
            end_date=end,
            travelers=int(travelers),
            budget_total=float(budget),
            preferences=tuple(preferences),
        )
    except ValueError as error:
        st.sidebar.error(str(error))
        return None, settings, notes
    return request, settings, notes


def main() -> None:
    st.title("Travel Intelligence")
    st.caption(
        "Deterministic code plans and validates the trip. The language model interprets "
        "preferences and writes the explanation — after the plan has already passed every "
        "hard constraint."
    )

    request, settings, _ = sidebar()
    if request is None:
        return

    try:
        result = run_pipeline(request, settings)
    except (NoCandidatesError, ProviderError) as refusal:
        st.error("No plan satisfies this request.")
        st.code(str(refusal), language=None)
        st.caption(
            "An explicit refusal, not an empty result: the system will not return a plan it "
            "has proven impossible."
        )
        return

    trip = result.trip
    budget = trip.budget

    columns = st.columns(4)
    columns[0].metric("Plan total", f"{budget.total:,.2f} EUR")
    columns[1].metric("Budget left", f"{budget.remaining:,.2f} EUR")
    columns[2].metric("Budget used", f"{budget.utilization:.1%}")
    columns[3].metric("Constraints", "valid" if trip.constraints.is_valid else "INVALID")

    st.caption(
        f"Funnel: {result.candidates.retrieved} retrieved → {result.candidates.kept} "
        f"candidates → {1 + len(trip.alternatives)} offered, {len(trip.rejected)} refused "
        "by the constraint check"
    )

    left, right = st.columns([3, 2])

    with left:
        hotel = trip.recommended
        st.subheader(hotel.accommodation.name)
        details = [hotel.accommodation.neighborhood, f"{hotel.total_cost:,.2f} EUR for the stay"]
        if hotel.accommodation.rating is not None:
            details.append(f"{hotel.accommodation.rating}/10")
        if hotel.accommodation.stars is not None:
            details.append(f"{hotel.accommodation.stars}★")
        st.write(" · ".join(str(detail) for detail in details if detail))

        st.markdown("**Why this one** — every factor, and the weight actually applied")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "factor": factor,
                        "score": round(value, 3),
                        "weight": round(hotel.weights.get(factor, 0.0), 3),
                        "contribution": round(value * hotel.weights.get(factor, 0.0), 3),
                    }
                    for factor, value in sorted(hotel.scores.as_dict().items())
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Overall {hotel.overall:.3f}. Factors that could not be computed are "
            "dropped and their weight redistributed, never scored as zero."
        )

        st.markdown("**Itinerary**")
        selected = {activity.id: activity for activity in trip.itinerary.selected}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "day": day.day.isoformat(),
                        "activity": ", ".join(selected[i].name for i in day.activity_ids) or "—",
                        "cost (EUR)": day.estimated_cost,
                    }
                    for day in trip.itinerary.days
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with right:
        st.markdown("**Budget**")
        st.dataframe(
            pd.DataFrame(
                [
                    {"line": "accommodation", "EUR": budget.accommodation, "source": "retrieved"},
                    {"line": "activities", "EUR": budget.activities, "source": "retrieved"},
                    {"line": "food", "EUR": budget.food, "source": "estimate"},
                    {"line": "transport", "EUR": budget.transport, "source": "estimate"},
                    {"line": "total", "EUR": budget.total, "source": ""},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**Alternatives**")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "rank": option.rank,
                        "name": option.accommodation.name,
                        "score": round(option.overall, 3),
                        "EUR": option.total_cost,
                    }
                    for option in trip.alternatives
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

        if trip.rejected:
            st.markdown("**Refused by the constraint check**")
            st.caption("The ranking preferred these; the full plan broke a hard constraint.")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"rank": r.rank, "name": r.name, "reason": ", ".join(r.codes)}
                        for r in trip.rejected
                    ]
                ),
                hide_index=True,
                use_container_width=True,
            )

    if trip.constraints.violations:
        st.markdown("**Constraint report**")
        for violation in trip.constraints.violations:
            renderer = st.error if violation.severity.value == "hard" else st.warning
            renderer(f"`{violation.code}` — {violation.message}")

    st.markdown("**Explanation**")
    with st.spinner("Writing the explanation…"):
        explanation = build_explainer(settings).explain(trip)
    st.caption(
        f"{PROVENANCE_LABEL.get(explanation.provenance.value, explanation.provenance.value)}"
        f"{f' · {explanation.model}' if explanation.model else ''}"
        f" · grounding check: {'passed' if explanation.grounded else 'FAILED'}"
    )
    for reason in explanation.rejection_reasons:
        st.warning(f"Model output discarded: {reason}")
    st.write(explanation.text)

    with st.expander("Quality signals and data sources"):
        signals = quality_signals(result, explanation.grounded)
        st.json(signals.model_dump())
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "source": source.name,
                        "provenance": PROVENANCE_LABEL.get(
                            source.provenance.value, source.provenance.value
                        ),
                        "records": source.record_count,
                    }
                    for source in result.sources
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )


main()
