"""The LLM layer: interpretation, explanation, and the grounding tripwire.

No test here touches a model server. The interesting cases are model outputs that are
malformed or dishonest, and a `ScriptedClient` produces those on demand — which a real model
obligingly would not.
"""

import json
from datetime import date

import pytest

from travel_intel.config import LLMProvider, Settings
from travel_intel.domain.enums import Preference, Provenance
from travel_intel.domain.models import Activity, TripRequest
from travel_intel.llm.client import ScriptedClient
from travel_intel.llm.explain import LLMExplainer, TemplateExplainer, build_payload
from travel_intel.llm.factory import build_explainer, build_interpreter
from travel_intel.llm.grounding import build_context, check_grounding, money_figures, parse_money
from travel_intel.llm.interpret import (
    KeywordPreferenceInterpreter,
    LLMPreferenceInterpreter,
)
from travel_intel.ranking import generate_candidates, rank_accommodations
from travel_intel.retrieval.snapshot import SnapshotProvider
from travel_intel.services import PlannedTrip, plan_trip


def make_request(**overrides: object) -> TripRequest:
    payload: dict[str, object] = {
        "destination": "Tokyo",
        "start_date": date(2026, 9, 10),
        "end_date": date(2026, 9, 17),
        "travelers": 2,
        "budget_total": 2500,
        "preferences": [Preference.FOOD, Preference.CULTURE, Preference.NATURE],
    }
    payload.update(overrides)
    return TripRequest(**payload)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def trip() -> PlannedTrip:
    request = make_request()
    provider = SnapshotProvider(Settings().fixtures_dir)
    accommodations = provider.search_accommodations(request)
    activities: tuple[Activity, ...] = provider.search_activities(request).records
    ranked = rank_accommodations(
        generate_candidates(accommodations.records, request).records, request
    )
    return plan_trip(ranked, activities, request)


# ---------------------------------------------------------------------------------------
# Money parsing
# ---------------------------------------------------------------------------------------


class TestMoneyParsing:
    @pytest.mark.parametrize(
        ("token", "expected"),
        [
            ("1.201,06", 1201.06),  # European
            ("1,201.06", 1201.06),  # Anglo
            ("2500", 2500.0),
            ("2,500", 2500.0),  # three trailing digits: thousands
            ("45,50", 45.5),  # two trailing digits: decimal
            ("45.5", 45.5),
        ],
    )
    def test_both_conventions(self, token: str, expected: float) -> None:
        assert parse_money(token) == pytest.approx(expected)

    def test_garbage_is_rejected(self) -> None:
        assert parse_money("") is None
        assert parse_money("...") is None

    @pytest.mark.parametrize(
        "text",
        [
            "it costs 1.201,06 EUR for the stay",
            "it costs €1201.06 for the stay",
            "it costs 1201.06 euros for the stay",
            "it costs EUR 1201.06 for the stay",
        ],
    )
    def test_figures_are_found_in_prose(self, text: str) -> None:
        assert 1201.06 in money_figures(text)

    @pytest.mark.parametrize(
        "text",
        [
            "accommodation EUR 1201.06, activities EUR 427.46",
            "a total of EUR 1201.06.",
            "the stay (EUR 1201.06) is the largest line",
            "EUR 1201.06; the rest is food",
        ],
    )
    def test_trailing_punctuation_is_not_read_as_a_separator(self, text: str) -> None:
        """Regression: a greedy digit class swallowed the comma in "EUR 1201.06, activities".

        The trailing comma was then read as a decimal separator and €1,201.06 became
        €120,106 — the check rejecting an honest explanation for the parser's own mistake.
        Found against the real model, not in a unit test.
        """
        assert 1201.06 in money_figures(text)
        assert 120106.0 not in money_figures(text)


# ---------------------------------------------------------------------------------------
# Grounding
# ---------------------------------------------------------------------------------------


class TestGroundingContext:
    def test_only_monetary_keys_become_quotable_amounts(self) -> None:
        """A day index of 7 must not license the model to write "EUR 7".

        Collecting every number would let the rounding tolerance wave through any small
        figure, which is exactly the check quietly not checking.
        """
        context = build_context({"day": 7, "travelers": 2, "rating": 8.9, "total_cost": 2426.52})
        assert context.amounts == (2426.52,)
        assert not context.permits_amount(7.0)

    def test_ids_are_collected_from_id_keys(self) -> None:
        context = build_context({"recommended": {"id": "h1"}, "itinerary": [{"activity_id": "a1"}]})
        assert context.entity_ids == {"h1", "a1"}

    def test_booleans_are_not_amounts(self) -> None:
        context = build_context({"food_and_transport_are_estimates": True})
        assert context.amounts == ()

    def test_rounded_phrasing_is_permitted(self) -> None:
        context = build_context({"total_cost": 2426.52})
        assert context.permits_amount(2427.0)
        assert context.permits_amount(2426.52)

    def test_an_invented_round_number_is_not(self) -> None:
        context = build_context({"total_cost": 2426.52})
        assert not context.permits_amount(2400.0)


class TestGroundingCheck:
    def test_clean_text_passes(self) -> None:
        context = build_context({"recommended": {"id": "h1", "total_cost": 1201.06}})
        assert check_grounding("The stay costs 1201.06 EUR.", ["h1"], context) == ()

    def test_unknown_entity_is_caught(self) -> None:
        context = build_context({"recommended": {"id": "h1", "total_cost": 1201.06}})
        violations = check_grounding("A fine hotel.", ["h1", "ghost"], context)
        assert len(violations) == 1
        assert "ghost" in violations[0]

    def test_invented_price_is_caught(self) -> None:
        context = build_context({"recommended": {"id": "h1", "total_cost": 1201.06}})
        violations = check_grounding("A steal at just 950 EUR.", ["h1"], context)
        assert len(violations) == 1
        assert "950" in violations[0]

    def test_both_kinds_are_reported_together(self) -> None:
        context = build_context({"recommended": {"id": "h1", "total_cost": 1201.06}})
        violations = check_grounding("Hotel Nowhere, 950 EUR.", ["ghost"], context)
        assert len(violations) == 2


class TestObservedModelFailures:
    """Regression tests for what llama3.1:8b actually did on the first real run.

    Every case here is a mistake the model made unprompted, not one invented for the test.
    """

    @pytest.fixture
    def context(self) -> object:
        return build_context(
            {
                "recommended": {
                    "id": "h1",
                    "price_per_night": 171.58,
                    "total_cost": 1201.06,
                },
                "budget": {"budget_total": 2500.0, "total_cost": 2426.52},
            }
        )

    def test_a_stay_total_described_as_a_nightly_rate_is_caught(self, context: object) -> None:
        """The figure is real and in the payload; the claim attached to it is false."""
        text = "The accommodation is 1201.06 EUR per night."
        violations = check_grounding(text, ["h1"], context)  # type: ignore[arg-type]
        assert len(violations) == 1
        assert "nightly rate" in violations[0]

    def test_the_real_nightly_rate_passes(self, context: object) -> None:
        text = "The room is 171.58 EUR per night."
        assert check_grounding(text, ["h1"], context) == ()  # type: ignore[arg-type]

    @pytest.mark.parametrize("qualifier", ["per night", "a night", "/night", "nightly"])
    def test_every_nightly_phrasing_is_recognised(self, context: object, qualifier: str) -> None:
        text = f"It costs 1201.06 EUR {qualifier}."
        assert check_grounding(text, [], context)  # type: ignore[arg-type]

    def test_a_right_number_in_the_wrong_currency_is_caught(self, context: object) -> None:
        text = "The trip fits a budget of $2500."
        violations = check_grounding(text, [], context)  # type: ignore[arg-type]
        assert any("wrong currency" in violation for violation in violations)

    @pytest.mark.parametrize("wrong", ["$2500", "2500 USD", "2500 yen", "£2500"])
    def test_common_foreign_currencies_are_caught(self, context: object, wrong: str) -> None:
        assert check_grounding(f"It costs {wrong}.", [], context)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------
# Preference interpretation
# ---------------------------------------------------------------------------------------


class TestKeywordInterpreter:
    def test_matches_english_and_spanish(self) -> None:
        interpreter = KeywordPreferenceInterpreter()
        assert Preference.FOOD in interpreter.interpret("we love good food").preferences
        assert Preference.FOOD in interpreter.interpret("nos gusta la comida").preferences

    def test_reports_nothing_as_unmapped(self) -> None:
        """Its honest limitation: substring matching cannot know what it missed."""
        result = KeywordPreferenceInterpreter().interpret("we want to see sumo")
        assert result.unmapped == ()

    def test_is_labelled_as_not_model_generated(self) -> None:
        result = KeywordPreferenceInterpreter().interpret("food")
        assert result.provenance is Provenance.SYNTHETIC


class TestLLMInterpreter:
    def test_maps_a_clean_response(self) -> None:
        client = ScriptedClient(json.dumps({"preferences": ["food", "culture"]}))
        result = LLMPreferenceInterpreter(client).interpret("street food and old temples")
        assert result.preferences == (Preference.FOOD, Preference.CULTURE)
        assert result.provenance is Provenance.MODEL_GENERATED

    def test_terms_outside_the_vocabulary_surface_as_unmapped(self) -> None:
        """A wish we cannot express is information the user should have, not a silent drop."""
        client = ScriptedClient(
            json.dumps({"preferences": ["food", "onsen"], "unmapped": ["avoiding crowds"]})
        )
        result = LLMPreferenceInterpreter(client).interpret("food, onsen, no crowds")
        assert result.preferences == (Preference.FOOD,)
        assert set(result.unmapped) == {"onsen", "avoiding crowds"}

    def test_duplicates_are_collapsed(self) -> None:
        client = ScriptedClient(json.dumps({"preferences": ["food", "FOOD", " food "]}))
        assert LLMPreferenceInterpreter(client).interpret("food").preferences == (Preference.FOOD,)

    def test_malformed_json_is_retried(self) -> None:
        client = ScriptedClient("not json at all", json.dumps({"preferences": ["nature"]}))
        result = LLMPreferenceInterpreter(client).interpret("hiking")
        assert result.preferences == (Preference.NATURE,)
        assert len(client.calls) == 2

    def test_persistent_failure_degrades_visibly(self) -> None:
        client = ScriptedClient("nonsense", "still nonsense")
        result = LLMPreferenceInterpreter(client).interpret("we love food")
        assert result.preferences == (Preference.FOOD,)  # keyword fallback did the work
        assert "fallback" in result.interpreter  # and the response admits it

    def test_empty_notes_skip_the_model(self) -> None:
        client = ScriptedClient(json.dumps({"preferences": ["food"]}))
        result = LLMPreferenceInterpreter(client).interpret("   ")
        assert result.preferences == ()
        assert client.calls == []


# ---------------------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------------------


class TestPayload:
    def test_carries_the_decided_plan(self, trip: PlannedTrip) -> None:
        payload = build_payload(trip)
        assert payload["recommended"]["id"] == trip.recommended.accommodation.id  # type: ignore[index]
        assert payload["budget"]["total_cost"] == trip.budget.total  # type: ignore[index]
        assert len(payload["itinerary"]) == len(trip.itinerary.selected)  # type: ignore[arg-type]

    def test_the_context_is_derived_from_the_payload(self, trip: PlannedTrip) -> None:
        """The check and the prompt cannot drift apart, because they share a source."""
        context = build_context(build_payload(trip))
        assert trip.recommended.accommodation.id in context.entity_ids
        assert context.permits_amount(trip.budget.total)
        assert context.permits_amount(trip.recommended.total_cost)


class TestTemplateExplainer:
    def test_is_grounded_by_construction(self, trip: PlannedTrip) -> None:
        explanation = TemplateExplainer().explain(trip)
        context = build_context(build_payload(trip))
        assert check_grounding(explanation.text, explanation.referenced_ids, context) == ()
        assert explanation.grounded is True

    def test_is_not_labelled_as_model_output(self, trip: PlannedTrip) -> None:
        explanation = TemplateExplainer().explain(trip)
        assert explanation.provenance is Provenance.SYNTHETIC
        assert explanation.model is None

    def test_names_the_estimated_lines_as_estimates(self, trip: PlannedTrip) -> None:
        assert "estimates" in TemplateExplainer().explain(trip).text


class TestLLMExplainer:
    def _response(self, **overrides: object) -> str:
        payload: dict[str, object] = {
            "summary": "A solid week in Tokyo.",
            "accommodation": "The stay costs 1201.06 EUR in total.",
            "itinerary": "Seven days of food, culture and nature.",
            "budget": "The plan totals 2426.52 EUR against 2500.00 EUR.",
            "referenced_ids": ["179813"],
        }
        payload.update(overrides)
        return json.dumps(payload)

    def test_grounded_output_is_accepted(self, trip: PlannedTrip) -> None:
        explanation = LLMExplainer(ScriptedClient(self._response())).explain(trip)
        assert explanation.grounded is True
        assert explanation.provenance is Provenance.MODEL_GENERATED
        assert explanation.rejection_reasons == ()
        assert "2426.52" in explanation.text

    def test_an_invented_price_is_rejected_and_never_shown(self, trip: PlannedTrip) -> None:
        """The headline guarantee: a plausible, wrong number does not reach the user."""
        dishonest = self._response(budget="A bargain at only 1900 EUR all in.")
        explanation = LLMExplainer(ScriptedClient(dishonest)).explain(trip)
        assert "1900" not in explanation.text
        assert explanation.provenance is Provenance.SYNTHETIC  # the template answered
        assert any("1900" in reason for reason in explanation.rejection_reasons)

    def test_an_invented_hotel_is_rejected(self, trip: PlannedTrip) -> None:
        dishonest = self._response(referenced_ids=["179813", "hotel-that-does-not-exist"])
        explanation = LLMExplainer(ScriptedClient(dishonest)).explain(trip)
        assert explanation.provenance is Provenance.SYNTHETIC
        assert any("does-not-exist" in reason for reason in explanation.rejection_reasons)

    def test_malformed_output_is_retried_then_falls_back(self, trip: PlannedTrip) -> None:
        client = ScriptedClient("{ broken", self._response())
        explanation = LLMExplainer(client).explain(trip)
        assert explanation.grounded is True
        assert explanation.provenance is Provenance.MODEL_GENERATED
        assert len(client.calls) == 2

    def test_rejected_text_is_discarded_not_repaired(self, trip: PlannedTrip) -> None:
        """Fixing a hallucinated price still leaves prose built around it."""
        dishonest = self._response(budget="Only 1900 EUR all in.")
        explanation = LLMExplainer(ScriptedClient(dishonest)).explain(trip)
        assert "A solid week in Tokyo." not in explanation.text  # nothing salvaged
        assert explanation.rejection_reasons

    def test_empty_output_is_not_accepted(self, trip: PlannedTrip) -> None:
        empty = json.dumps({"summary": "", "referenced_ids": []})
        explanation = LLMExplainer(ScriptedClient(empty)).explain(trip)
        assert explanation.provenance is Provenance.SYNTHETIC
        assert any("empty" in reason for reason in explanation.rejection_reasons)


class TestFactory:
    def test_fake_provider_gives_the_deterministic_pair(self) -> None:
        settings = Settings(llm_provider=LLMProvider.FAKE)
        assert isinstance(build_interpreter(settings), KeywordPreferenceInterpreter)
        assert isinstance(build_explainer(settings), TemplateExplainer)

    def test_ollama_provider_gives_the_model_backed_pair(self) -> None:
        settings = Settings(llm_provider=LLMProvider.OLLAMA)
        assert isinstance(build_interpreter(settings), LLMPreferenceInterpreter)
        assert isinstance(build_explainer(settings), LLMExplainer)
