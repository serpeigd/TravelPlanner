"""Streamlit front end: `streamlit run src/travel_intel/ui/streamlit_app.py`.

The interface exists to make the machinery visible, not to hide it. A traveller-facing product
would show one hotel and a price; this shows the funnel that produced it, every candidate with
its factors, what the price model predicted versus what each property charges, and which
options the constraint check refused and why.

Destination is a dropdown rather than a text box on purpose. The system is general but the
*data* is a single captured snapshot, and offering a free-text field invites a request the
system can only refuse — which reads as a broken app instead of an honest limitation.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from travel_intel.config import LLMProvider, Settings
from travel_intel.domain.enums import Preference
from travel_intel.domain.errors import NoCandidatesError, ProviderError
from travel_intel.domain.models import TripRequest
from travel_intel.features.accommodations import build_accommodation_features
from travel_intel.llm.factory import build_explainer, build_interpreter
from travel_intel.ml.price_model import (
    DistrictMedianBaseline,
    GlobalMedianBaseline,
    HedonicPriceModel,
    cross_validate,
)
from travel_intel.ranking.scoring import DEFAULT_WEIGHTS
from travel_intel.recommend import quality_signals
from travel_intel.retrieval.snapshot import SnapshotProvider
from travel_intel.services.pipeline import PipelineResult, run_pipeline

st.set_page_config(page_title="Travel Intelligence", layout="wide")

PROVENANCE_LABEL = {
    "snapshot": "retrieved · frozen snapshot of real provider data",
    "real_api": "retrieved live",
    "synthetic": "our own estimate, never presented as a fact",
    "model_generated": "written by the language model",
}

# Preset budgets, so the behaviour worth showing is one click away rather than one guess.
# Note: a bare string here would be *rendered on the page* — Streamlit prints any top-level
# expression, so attribute docstrings cannot be used in this file.
SCENARIOS: dict[str, int | None] = {
    "Reference — €2,500": 2500,
    "Tight — €2,100 (the constraint check starts refusing)": 2100,
    "Very tight — €1,600": 1600,
    "Impossible — €800 (explicit refusal)": 800,
    "Generous — €5,000": 5000,
    "Custom…": None,
}


# ---------------------------------------------------------------------------------------
# Cached lookups
# ---------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def available_destinations() -> dict[str, str]:
    """Destination key -> the label shown in the dropdown, read from the packaged snapshots."""
    provider = SnapshotProvider(Settings().fixtures_dir)
    found = provider.available_destinations()
    return {key: f"{key.title()} — {folder.name.split('_', 1)[1]}" for key, folder in found.items()}


@st.cache_data(show_spinner=False)
def price_model_metrics() -> pd.DataFrame:
    """Cross-validated error for the model and both baselines. Independent of the request."""
    request = TripRequest(
        destination="Tokyo",
        start_date=date(2026, 9, 10),
        end_date=date(2026, 9, 17),
        travelers=2,
        budget_total=2500,
    )
    provider = SnapshotProvider(Settings().fixtures_dir)
    frame = build_accommodation_features(provider.search_accommodations(request).records, request)
    results = cross_validate(
        frame, (GlobalMedianBaseline(), DistrictMedianBaseline(), HedonicPriceModel())
    )
    return pd.DataFrame(
        [
            {"estimator": name, "MAE (EUR)": m.mae, "RMSE (EUR)": m.rmse, "MAPE": m.mape}
            for name, m in results.items()
        ]
    )


# ---------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------


def sidebar() -> tuple[TripRequest | None, Settings]:
    destinations = available_destinations()

    st.sidebar.header("Request")
    if not destinations:
        st.sidebar.error("No snapshot is packaged with this build.")
        return None, Settings()

    key = st.sidebar.selectbox(
        "Destination",
        options=list(destinations),
        format_func=lambda k: destinations[k],
        help="One captured snapshot. The pipeline is destination-agnostic; the data is not.",
    )

    scenario = st.sidebar.selectbox("Budget scenario", options=list(SCENARIOS))
    preset = SCENARIOS[scenario]
    budget = (
        float(preset)
        if preset is not None
        else float(st.sidebar.number_input("Budget (EUR)", min_value=100, value=2500, step=100))
    )

    columns = st.sidebar.columns(2)
    start = columns[0].date_input("Check-in", date(2026, 9, 10))
    end = columns[1].date_input("Check-out", date(2026, 9, 17))
    travelers = st.sidebar.number_input("Travellers", min_value=1, max_value=12, value=2)

    st.sidebar.header("Preferences")
    notes = st.sidebar.text_area(
        "Describe the trip",
        placeholder="we love street food and quiet old temples",
        help="The LLM's first job: free text into the controlled vocabulary below.",
    )

    use_model = st.sidebar.toggle(
        "Use the local model (Ollama)",
        value=False,
        help="Off: keyword matching and a templated explanation, instant. "
        "On: llama3.1:8b, about two minutes for the explanation.",
    )
    settings = Settings(llm_provider=LLMProvider.OLLAMA if use_model else LLMProvider.FAKE)

    interpreted: list[Preference] = []
    if notes.strip():
        with st.spinner("Interpreting…"):
            result = build_interpreter(settings).interpret(notes)
        interpreted = list(result.preferences)
        st.sidebar.caption(f"Interpreted by `{result.interpreter}`")
        if result.unmapped:
            st.sidebar.warning("Not expressible in the vocabulary: " + ", ".join(result.unmapped))

    preferences = st.sidebar.multiselect(
        "Vocabulary",
        options=list(Preference),
        default=interpreted or [Preference.FOOD, Preference.CULTURE, Preference.NATURE],
        format_func=lambda preference: preference.value,
    )

    try:
        request = TripRequest(
            # The display name, not the internal key: it is echoed back in the explanation,
            # and `destination_key()` normalises it on the way into retrieval anyway.
            destination=key.title(),
            start_date=start,
            end_date=end,
            travelers=int(travelers),
            budget_total=budget,
            preferences=tuple(preferences),
        )
    except ValueError as error:
        st.sidebar.error(str(error))
        return None, settings
    return request, settings


# ---------------------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------------------


def tab_plan(result: PipelineResult, settings: Settings) -> None:
    trip = result.trip
    budget = trip.budget
    hotel = trip.recommended

    columns = st.columns(4)
    columns[0].metric("Plan total", f"{budget.total:,.2f} EUR")
    columns[1].metric("Budget left", f"{budget.remaining:,.2f} EUR")
    columns[2].metric("Budget used", f"{budget.utilization:.1%}")
    columns[3].metric("Constraints", "valid" if trip.constraints.is_valid else "INVALID")

    left, right = st.columns([3, 2])

    with left:
        st.subheader(hotel.accommodation.name)
        facts = [
            hotel.accommodation.neighborhood,
            f"{hotel.accommodation.price_per_night:,.2f} EUR/night",
            f"{hotel.total_cost:,.2f} EUR for the stay",
        ]
        if hotel.accommodation.rating is not None:
            facts.append(f"{hotel.accommodation.rating}/10 guest rating")
        if hotel.accommodation.stars is not None:
            facts.append(f"{hotel.accommodation.stars}-star")
        st.write(" · ".join(str(fact) for fact in facts if fact))

        st.markdown("**Why this one** — each factor times the weight actually applied")
        contributions = pd.DataFrame(
            [
                {
                    "factor": factor,
                    "score": round(value, 3),
                    "weight": round(hotel.weights.get(factor, 0.0), 3),
                    "contribution": round(value * hotel.weights.get(factor, 0.0), 4),
                }
                for factor, value in hotel.scores.as_dict().items()
            ]
        ).sort_values("contribution", ascending=False)
        st.bar_chart(contributions.set_index("factor")["contribution"], height=220)
        st.dataframe(contributions, hide_index=True, use_container_width=True)
        st.caption(
            f"Overall {hotel.overall:.3f}. Factors that cannot be computed are dropped and "
            "their weight redistributed over the rest — never scored as zero."
        )

        st.markdown("**Itinerary**")
        selected = {activity.id: activity for activity in trip.itinerary.selected}
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "day": day.day.isoformat(),
                        "activity": ", ".join(selected[i].name for i in day.activity_ids) or "—",
                        "EUR": day.estimated_cost,
                    }
                    for day in trip.itinerary.days
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        covered = ", ".join(p.value for p in trip.itinerary.covered_preferences) or "none"
        st.caption(f"Preferences covered: {covered}")

    with right:
        st.markdown("**Budget** — two lines are retrieved prices, two are declared estimates")
        st.dataframe(
            pd.DataFrame(
                [
                    {"line": "accommodation", "EUR": budget.accommodation, "source": "retrieved"},
                    {"line": "activities", "EUR": budget.activities, "source": "retrieved"},
                    {"line": "food", "EUR": budget.food, "source": "estimate"},
                    {"line": "transport", "EUR": budget.transport, "source": "estimate"},
                    {"line": "TOTAL", "EUR": budget.total, "source": ""},
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )

        st.markdown("**Constraint report**")
        if not trip.constraints.violations:
            st.success("No violations.")
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
        st.warning(f"Model output discarded — {reason}")
    st.write(explanation.text)
    st.session_state["explanation_grounded"] = explanation.grounded


def tab_ranking(result: PipelineResult, request: TripRequest) -> None:
    trip = result.trip
    st.markdown(
        f"**Funnel** — {result.candidates.retrieved} retrieved → "
        f"{result.candidates.kept} candidates → {1 + len(trip.alternatives)} offered, "
        f"{len(trip.rejected)} refused by the constraint check"
    )
    if result.candidates.dropped:
        st.caption(
            "Dropped before ranking: "
            + ", ".join(
                f"{count} x {reason}" for reason, count in result.candidates.dropped.items()
            )
        )

    st.markdown(f"**All {len(result.ranked)} candidates, every factor**")
    offered = {trip.recommended.accommodation.id, *(a.accommodation.id for a in trip.alternatives)}
    refused = {option.accommodation_id for option in trip.rejected}
    rows = [
        {
            "rank": item.rank,
            "outcome": (
                "offered"
                if item.accommodation.id in offered
                else "REFUSED"
                if item.accommodation.id in refused
                else "not offered"
            ),
            "name": item.accommodation.name,
            "overall": round(item.overall, 3),
            "EUR/night": item.accommodation.price_per_night,
            "EUR stay": item.total_cost,
            **{
                factor: (None if value is None else round(value, 3))
                for factor, value in item.scores.as_factors().items()
            },
        }
        for item in result.ranked
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption(
        "An empty cell means the factor could not be computed for that property — the "
        "unrated one in the snapshot shows this on `rating`."
    )

    st.markdown("**Weights in use**")
    st.dataframe(
        pd.DataFrame(
            [{"factor": name, "weight": weight} for name, weight in DEFAULT_WEIGHTS.items()]
        ),
        hide_index=True,
        use_container_width=True,
    )

    if trip.rejected:
        st.markdown("**Refused by the constraint check**")
        st.caption(
            "The ranking preferred these. Each one passed candidate generation — the room "
            "alone fits the budget — and only fails once food, transport and activities are "
            "added. That is exactly the arithmetic a fluent model gets confidently wrong."
        )
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
    else:
        st.info(
            f"Nothing refused at {request.budget_total:,.0f} EUR. Try the €2,100 scenario to "
            "see the constraint check overrule the ranking."
        )


def tab_model(result: PipelineResult, request: TripRequest) -> None:
    st.markdown(
        "**The prediction is not the product — the residual is.** The model predicts what a "
        "property *should* cost from its own attributes. A room priced well below that line is "
        "good value for what it is, and that becomes the `value_for_money` factor."
    )

    # Re-fitted here rather than threaded through the pipeline: it is a 30-row ridge, and the
    # chart needs the predictions the ranking consumed only as a number.
    features = build_accommodation_features(result.candidates.records, request)
    model = HedonicPriceModel()
    predicted = model.fit_predict(features)
    recommended_id = result.trip.recommended.accommodation.id

    frame = pd.DataFrame(
        {
            "name": features["name"],
            "actual": features["price_per_night"].round(2),
            "predicted": predicted.round(2),
        }
    )
    frame["residual %"] = ((frame["actual"] / frame["predicted"] - 1) * 100).round(1)
    frame["verdict"] = frame["residual %"].apply(
        lambda pct: "cheaper than predicted" if pct < 0 else "dearer than predicted"
    )
    frame.loc[features.index == recommended_id, "verdict"] = "recommended"

    left, right = st.columns(2)
    with left:
        st.markdown("**Charged vs. predicted price**")
        st.scatter_chart(frame, x="predicted", y="actual", color="verdict", height=340)
        st.caption("Below the diagonal is good value; above it, poor value.")
    with right:
        st.markdown("**How far each property sits from its predicted price**")
        st.bar_chart(
            frame.sort_values("residual %").set_index("name")["residual %"],
            height=340,
        )

    st.dataframe(
        frame.sort_values("residual %").reset_index(drop=True),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Out-of-sample error** — repeated 5-fold CV, 5 repeats, identical folds")
    st.dataframe(price_model_metrics(), hide_index=True, use_container_width=True)
    st.caption(
        "The district median is the honest competitor: location explains much of price and a "
        "grouped median needs no model at all. Had the regression not beaten it, that would "
        "have been the finding."
    )

    st.markdown("**What the model learned**")
    coefficients = pd.DataFrame(
        [{"feature": f, "coefficient": c} for f, c in model.coefficients().items()]
    ).sort_values("coefficient", key=abs, ascending=False)
    st.dataframe(coefficients, hide_index=True, use_container_width=True)
    st.caption(
        "Guest rating barely explains price once stars and amenities are known — which is why "
        "`rating` and `value_for_money` are separate ranking factors rather than one."
    )


def tab_data(result: PipelineResult) -> None:
    signals = quality_signals(result, st.session_state.get("explanation_grounded"))
    columns = st.columns(4)
    columns[0].metric("Retrieved", signals.candidates_retrieved)
    columns[1].metric("After filters", signals.candidates_after_filters)
    columns[2].metric("Preference coverage", f"{signals.preference_coverage:.0%}")
    columns[3].metric("Mean data completeness", f"{signals.mean_data_completeness:.0%}")

    st.markdown("**Where every fact came from**")
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

    if result.warnings:
        st.markdown("**Retrieval warnings**")
        for warning in result.warnings:
            st.warning(warning)
    else:
        st.success("The request matches the captured snapshot exactly — no extrapolation.")

    with st.expander("Quality signals (the object returned by the API)"):
        st.json(signals.model_dump())


# ---------------------------------------------------------------------------------------


def main() -> None:
    st.title("Travel Intelligence")
    st.caption(
        "Deterministic code retrieves, ranks, budgets and validates. The language model "
        "interprets preferences and writes the explanation — after the plan has already "
        "passed every hard constraint."
    )

    request, settings = sidebar()
    if request is None:
        return

    try:
        result = run_pipeline(request, settings)
    except (NoCandidatesError, ProviderError) as refusal:
        st.error("No plan satisfies this request.")
        st.code(str(refusal), language=None)
        st.caption(
            "An explicit refusal, not an empty result. The system will not return a plan it "
            "has just proven impossible — try a larger budget."
        )
        return

    plan, ranking, model, data = st.tabs(["Plan", "Ranking", "Price model", "Data & quality"])
    with plan:
        tab_plan(result, settings)
    with ranking:
        tab_ranking(result, request)
    with model:
        tab_model(result, request)
    with data:
        tab_data(result)


main()
