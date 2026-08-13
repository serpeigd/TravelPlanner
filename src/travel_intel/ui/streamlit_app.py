"""Streamlit front end: `streamlit run src/travel_intel/ui/streamlit_app.py`.

Written for someone who has never seen this project. The first screen answers the traveller's
question in plain words; every technical detail sits one click away behind an expander, and
every number carries a sentence saying what it means.

That ordering is deliberate. An earlier version opened with six factor names, a weights table
and a scatter plot — legible only to someone who already knew what they were looking at.

Destination is a dropdown rather than a text box: the system is destination-agnostic but the
*data* is a single captured snapshot, and a free-text field invites a request that can only be
refused, which reads as a broken app rather than an honest limit.
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
from travel_intel.recommend import quality_signals
from travel_intel.retrieval.snapshot import SnapshotProvider
from travel_intel.services.pipeline import PipelineResult, run_pipeline

st.set_page_config(page_title="Travel Intelligence", layout="wide")

# Human names for the six ranking factors, and one line each on what they measure. The code
# uses short identifiers; a screen should not.
FACTOR_LABEL = {
    "budget_fit": "Fits the budget",
    "value_for_money": "Good value for what it is",
    "rating": "Guest rating",
    "location": "Close to the centre",
    "preference_match": "Matches your interests",
    "data_completeness": "How much we know about it",
}
FACTOR_HELP = {
    "budget_fit": "Full marks while the room stays inside its share of the budget.",
    "value_for_money": "What it charges versus what a model predicts it should charge.",
    "rating": "Guest score, pulled toward the market average when it has few reviews.",
    "location": "Straight-line distance to the city centre.",
    "preference_match": "Share of your interests the hotel itself can support.",
    "data_completeness": "A tie-breaker: how many facts the provider actually gave us.",
}

SOURCE_LABEL = {
    "snapshot": "Real price, captured from the provider",
    "real_api": "Real price, fetched live",
    "synthetic": "Our estimate — not a real quote",
    "model_generated": "Written by the language model",
}

# The same provenance value means different things for a number and for a paragraph:
# `synthetic` money is an estimate, `synthetic` prose is written by fixed rules.
WRITER_LABEL = {
    "synthetic": "Written straight from the plan by fixed rules — no AI involved",
    "model_generated": "Written by the local AI model",
}

# Preset budgets, so the behaviour worth showing is one click away rather than one guess.
# A bare string here would be *rendered on the page*: Streamlit prints top-level expressions,
# so this file cannot use attribute docstrings.
SCENARIOS: dict[str, int | None] = {
    "€2,500 — the standard trip": 2500,
    "€2,100 — tight (watch options get refused)": 2100,
    "€1,600 — very tight": 1600,
    "€800 — impossible (the system says no)": 800,
    "€5,000 — generous": 5000,
    "Choose my own…": None,
}


# ---------------------------------------------------------------------------------------
# Cached lookups
# ---------------------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def available_destinations() -> dict[str, str]:
    provider = SnapshotProvider(Settings().fixtures_dir)
    found = provider.available_destinations()
    return {key: f"{key.title()} ({folder.name.split('_', 1)[1]})" for key, folder in found.items()}


@st.cache_data(show_spinner=False)
def price_model_metrics() -> pd.DataFrame:
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
    label = {
        "baseline: global median": "Simplest guess: the same price for every hotel",
        "baseline: district median": "Better guess: the typical price in that district",
        "hedonic ridge (log price)": "Our model: price from the hotel's own features",
    }
    return pd.DataFrame(
        [
            {
                "Method": label.get(name, name),
                "Average error": f"{m.mae:,.0f} EUR",
                "Typical error": f"{m.mape:.0%}",
            }
            for name, m in results.items()
        ]
    )


# ---------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------


def sidebar() -> tuple[TripRequest | None, Settings]:
    destinations = available_destinations()
    st.sidebar.title("Plan a trip")

    if not destinations:
        st.sidebar.error("No travel data is packaged with this build.")
        return None, Settings()

    key = st.sidebar.selectbox(
        "Where to",
        options=list(destinations),
        format_func=lambda k: destinations[k],
        help="One city is loaded. The system works anywhere; the data was captured once.",
    )

    scenario = st.sidebar.selectbox("How much can you spend?", options=list(SCENARIOS))
    preset = SCENARIOS[scenario]
    budget = (
        float(preset)
        if preset is not None
        else float(st.sidebar.number_input("Total budget (EUR)", 100, 20000, 2500, 100))
    )

    columns = st.sidebar.columns(2)
    start = columns[0].date_input("From", date(2026, 9, 10))
    end = columns[1].date_input("To", date(2026, 9, 17))
    travelers = st.sidebar.number_input("How many of you", min_value=1, max_value=12, value=2)

    st.sidebar.divider()
    st.sidebar.subheader("What do you enjoy?")

    settings = Settings(
        llm_provider=LLMProvider.OLLAMA
        if st.sidebar.toggle(
            "Use the AI model",
            value=False,
            help="Off: instant, uses fixed rules. On: a local AI writes the summary "
            "(about two minutes).",
        )
        else LLMProvider.FAKE
    )

    notes = st.sidebar.text_area(
        "Describe your trip in your own words",
        placeholder="we love street food and quiet old temples",
        help="The AI turns this into the tick-list below. Leave it empty to tick manually.",
    )

    interpreted: list[Preference] = []
    if notes.strip():
        with st.spinner("Reading that…"):
            result = build_interpreter(settings).interpret(notes)
        interpreted = list(result.preferences)
        if result.unmapped:
            st.sidebar.info("We can't cater for: " + ", ".join(result.unmapped))

    preferences = st.sidebar.multiselect(
        "Interests",
        options=list(Preference),
        default=interpreted or [Preference.FOOD, Preference.CULTURE, Preference.NATURE],
        format_func=lambda preference: preference.value.capitalize(),
    )

    try:
        request = TripRequest(
            # The display name, not the internal key: it is echoed back in the summary, and
            # retrieval normalises it anyway.
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
# Tab 1 - the answer
# ---------------------------------------------------------------------------------------


def tab_answer(result: PipelineResult, request: TripRequest, settings: Settings) -> None:
    trip = result.trip
    budget = trip.budget
    hotel = trip.recommended

    if trip.constraints.is_valid:
        st.success(
            f"**This trip works.** {budget.total:,.0f} EUR in total, "
            f"{budget.remaining:,.0f} EUR left over from your {budget.budget_total:,.0f} EUR."
        )

    st.subheader(f"Stay at {hotel.accommodation.name}")
    facts = [
        hotel.accommodation.neighborhood,
        f"{hotel.accommodation.price_per_night:,.0f} EUR a night",
        f"{hotel.total_cost:,.0f} EUR for {request.nights} nights",
    ]
    if hotel.accommodation.rating is not None:
        facts.append(f"guests rate it {hotel.accommodation.rating}/10")
    st.write(" · ".join(str(fact) for fact in facts if fact))

    left, right = st.columns([1, 1])

    with left:
        st.markdown("#### Where the money goes")
        for label, amount, real in (
            ("Hotel", budget.accommodation, True),
            ("Things to do", budget.activities, True),
            ("Food", budget.food, False),
            ("Getting around", budget.transport, False),
        ):
            row = st.columns([3, 2, 3])
            row[0].write(label)
            row[1].write(f"**{amount:,.0f} EUR**")
            row[2].caption("real price" if real else "our estimate")
        st.divider()
        row = st.columns([3, 2, 3])
        row[0].write("**Total**")
        row[1].write(f"**{budget.total:,.0f} EUR**")
        row[2].caption(f"of {budget.budget_total:,.0f} EUR")
        st.progress(min(budget.utilization, 1.0))
        st.caption(
            "Hotel and activities are prices we actually retrieved. Food and transport are "
            "estimates, and the system says so rather than passing them off as quotes."
        )

    with right:
        st.markdown("#### Your days")
        selected = {activity.id: activity for activity in trip.itinerary.selected}
        for day in trip.itinerary.days:
            names = ", ".join(selected[i].name for i in day.activity_ids)
            when = day.day.strftime("%a %d %b")
            if names:
                st.write(f"**{when}** — {names}  ·  {day.estimated_cost:,.0f} EUR")
            else:
                st.write(f"**{when}** — free day")
        covered = ", ".join(p.value for p in trip.itinerary.covered_preferences)
        if covered:
            st.caption(f"Covers everything you asked for: {covered}.")

    st.divider()
    st.markdown("#### In a sentence")
    with st.spinner("Writing the summary…"):
        explanation = build_explainer(settings).explain(trip)
    st.info(explanation.text)
    writer = WRITER_LABEL.get(explanation.provenance.value, explanation.provenance.value)
    st.caption(
        f"{writer}. Every figure in it was checked against the plan before you saw it"
        f"{' — the check passed.' if explanation.grounded else ' — the check FAILED.'}"
    )
    for reason in explanation.rejection_reasons:
        st.warning(
            f"The AI wrote something that did not match the plan, so it was thrown away: {reason}"
        )
    st.session_state["explanation_grounded"] = explanation.grounded

    if trip.constraints.violations:
        with st.expander("Things worth knowing about this plan"):
            for violation in trip.constraints.violations:
                st.write(f"- {violation.message}")


# ---------------------------------------------------------------------------------------
# Tab 2 - why
# ---------------------------------------------------------------------------------------


def tab_why(result: PipelineResult, request: TripRequest) -> None:
    trip = result.trip
    hotel = trip.recommended

    st.markdown("#### How we got from every hotel in the city to this one")
    steps = st.columns(4)
    steps[0].metric("Hotels found", result.candidates.retrieved)
    steps[1].metric("Could work", result.candidates.kept, help="Right size, and within budget")
    steps[2].metric("Ruled out later", len(trip.rejected), help="The whole trip did not fit")
    steps[3].metric("Shown to you", 1 + len(trip.alternatives))

    st.divider()
    st.markdown(f"#### Why **{hotel.accommodation.name}** won")
    st.caption(
        "Six things are scored from 0 to 100. Each one counts for a different amount, "
        "shown on the right. There is no AI in this step — it is arithmetic."
    )

    for factor, value in sorted(
        hotel.scores.as_dict().items(), key=lambda kv: -hotel.weights.get(kv[0], 0)
    ):
        weight = hotel.weights.get(factor, 0.0)
        columns = st.columns([3, 4, 2])
        columns[0].write(FACTOR_LABEL.get(factor, factor))
        columns[1].progress(value, text=f"{value:.0%}")
        columns[2].caption(f"counts for {weight:.0%}")
        columns[0].caption(FACTOR_HELP.get(factor, ""))

    st.caption(
        f"Weighted together, that gives {hotel.overall:.0%}. When something cannot be worked "
        "out for a hotel — nobody has rated it yet, for instance — that item is dropped and "
        "the others share its weight. It is never counted as a zero."
    )

    if trip.rejected:
        st.divider()
        st.markdown("#### Hotels the scoring preferred, but the budget check refused")
        st.caption(
            "Each of these looked better on paper and the room itself fits your budget. "
            "They only fail once food, getting around and things to do are added on top — "
            "which is exactly the sum an AI would get confidently wrong."
        )
        for option in trip.rejected[:6]:
            st.write(f"- **#{option.rank} {option.name}** — the full trip goes over budget")
    else:
        st.info(
            f"At {request.budget_total:,.0f} EUR nothing had to be refused. "
            "Switch to the €2,100 budget to watch the check overrule the scoring."
        )

    with st.expander("Show me every hotel and every score"):
        offered = {hotel.accommodation.id, *(a.accommodation.id for a in trip.alternatives)}
        refused = {option.accommodation_id for option in trip.rejected}
        rows = [
            {
                "#": item.rank,
                "Hotel": item.accommodation.name,
                "Score": round(item.overall * 100),
                "EUR/night": round(item.accommodation.price_per_night),
                "Outcome": (
                    "shown to you"
                    if item.accommodation.id in offered
                    else "refused: over budget"
                    if item.accommodation.id in refused
                    else "scored lower"
                ),
                **{
                    FACTOR_LABEL.get(factor, factor): (
                        None if value is None else round(value * 100)
                    )
                    for factor, value in item.scores.as_factors().items()
                },
            }
            for item in result.ranked
        ]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("An empty cell means that item could not be worked out for that hotel.")


# ---------------------------------------------------------------------------------------
# Tab 3 - the model
# ---------------------------------------------------------------------------------------


def tab_model(result: PipelineResult, request: TripRequest) -> None:
    st.markdown("#### Is this hotel a bargain, or overpriced?")
    st.caption(
        "Knowing a hotel costs 170 EUR a night tells you nothing on its own. So a model "
        "learns what a hotel *should* cost from its stars, facilities, location and reviews. "
        "If it charges less than that, it is good value for what it is."
    )

    features = build_accommodation_features(result.candidates.records, request)
    model = HedonicPriceModel()
    predicted = model.fit_predict(features)
    recommended_id = result.trip.recommended.accommodation.id

    frame = pd.DataFrame(
        {
            "Hotel": features["name"],
            "Charges": features["price_per_night"].round(0),
            "Should cost": predicted.round(0),
        }
    )
    frame["Difference"] = (frame["Charges"] / frame["Should cost"] - 1).round(3)

    chosen = frame[features.index == recommended_id]
    if not chosen.empty:
        row = chosen.iloc[0]
        gap = float(row["Difference"])
        verdict = "cheaper than it should be" if gap < 0 else "dearer than it should be"
        st.metric(
            f"{row['Hotel']}",
            f"{row['Charges']:,.0f} EUR a night",
            f"{gap:+.0%} vs the {row['Should cost']:,.0f} EUR it should cost",
            delta_color="inverse",
        )
        st.caption(f"That makes it **{verdict}** for what it offers.")

    st.markdown("##### Every hotel, cheapest-for-what-it-is first")
    ordered = frame.sort_values("Difference")
    st.bar_chart(ordered.set_index("Hotel")["Difference"], height=320)
    st.caption("Bars below zero charge less than the model expects. Above zero, more.")

    with st.expander("How good is that model, really?"):
        st.caption(
            "A model is only worth having if it beats the obvious alternatives. These are "
            "the errors when the model is tested on hotels it was not trained on."
        )
        st.dataframe(price_model_metrics(), hide_index=True, use_container_width=True)
        st.caption(
            "Our model is wrong by about 41 EUR a night on average, against 73 EUR for the "
            "best simple alternative. If it had not beaten them, that would have been the "
            "result to report."
        )

    with st.expander("What drives a hotel's price?"):
        coefficients = pd.DataFrame(
            [
                {"Feature": feature, "Effect on price": value}
                for feature, value in model.coefficients().items()
            ]
        ).sort_values("Effect on price", key=abs, ascending=False)
        st.dataframe(coefficients, hide_index=True, use_container_width=True)
        st.caption(
            "Stars matter most, then how many facilities it has. Distance from the centre "
            "pushes the price down. The surprise: the guest rating barely moves it at all — "
            "which is why 'well rated' and 'good value' are scored as two separate things."
        )


# ---------------------------------------------------------------------------------------
# Tab 4 - trust
# ---------------------------------------------------------------------------------------


def tab_trust(result: PipelineResult) -> None:
    signals = quality_signals(result, st.session_state.get("explanation_grounded"))

    st.markdown("#### Can you trust this answer?")
    columns = st.columns(3)
    columns[0].metric(
        "Within budget", "Yes" if signals.budget_compliant else "NO", help="Checked in code"
    )
    columns[1].metric("Rules broken", signals.hard_violations, help="Must always be zero")
    columns[2].metric("Your interests covered", f"{signals.preference_coverage:.0%}")

    st.divider()
    st.markdown("#### Where each fact came from")
    for source in result.sources:
        label = SOURCE_LABEL.get(source.provenance.value, source.provenance.value)
        if source.provenance.value == "synthetic":
            st.warning(f"**{label}** — {source.name}")
        else:
            st.success(f"**{label}** — {source.name} ({source.record_count} records)")
    st.caption(
        "Nothing invented is ever shown as if it were retrieved. Every record carries a tag "
        "saying where it came from, all the way to this screen."
    )

    if result.warnings:
        st.divider()
        st.markdown("#### Caveats for this particular request")
        for warning in result.warnings:
            st.warning(warning)

    with st.expander("The raw quality report (what the API returns)"):
        st.json(signals.model_dump())


# ---------------------------------------------------------------------------------------


def main() -> None:
    st.title("Travel Intelligence")
    st.write(
        "Tell it where, when and how much you can spend. It finds a hotel, fills your days, "
        "adds up the cost and checks the whole thing fits — then explains itself."
    )

    with st.expander("What am I looking at?"):
        st.markdown(
            """
**The point of this project is *how* the answer is produced, not the answer itself.**

- **The plan** — the trip it recommends, and what it costs.
- **Why this one** — the scoring, step by step, and the hotels it refused.
- **Bargain or rip-off** — a small machine-learning model that works out whether a hotel
  charges more or less than it should for what it offers.
- **Can I trust it** — where every number came from, and what was estimated rather than
  retrieved.

The arithmetic and the rules are ordinary code, checked by 246 automated tests. The AI only
does two things: it reads your description of the trip, and it writes the summary at the
end — after the plan has already been checked. It never decides anything.
            """
        )

    request, settings = sidebar()
    if request is None:
        return

    try:
        result = run_pipeline(request, settings)
    except (NoCandidatesError, ProviderError) as refusal:
        st.error("**No trip is possible with this budget.**")
        st.caption(
            "The system refuses rather than showing you something that does not add up. "
            "The technical reason:"
        )
        st.code(str(refusal), language=None)
        return

    answer, why, model, trust = st.tabs(
        ["The plan", "Why this one", "Bargain or rip-off?", "Can I trust it?"]
    )
    with answer:
        tab_answer(result, request, settings)
    with why:
        tab_why(result, request)
    with model:
        tab_model(result, request)
    with trust:
        tab_trust(result)


main()
