"""The HTTP surface.

These tests exist to prove the guarantees hold *through the API*, not just in the modules
beneath it. A budget check that works in `validate_plan` but is bypassed by the endpoint would
be no guarantee at all.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from travel_intel.api.app import create_app
from travel_intel.config import DataMode, LLMProvider, Settings

SETTINGS = Settings(data_mode=DataMode.SNAPSHOT, llm_provider=LLMProvider.FAKE)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(SETTINGS))


def payload(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "destination": "Tokyo",
        "start_date": "2026-09-10",
        "end_date": "2026-09-17",
        "travelers": 2,
        "budget_total": 2500,
        "preferences": ["food", "culture", "nature"],
    }
    body.update(overrides)
    return body


@pytest.fixture(scope="module")
def recommendation(client: TestClient) -> dict[str, Any]:
    response = client.post("/trip/recommend", json=payload())
    assert response.status_code == 200
    return dict(response.json())


class TestHealth:
    def test_reports_the_settings_that_change_what_answers_mean(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["data_mode"] == "snapshot"
        assert body["llm_provider"] == "fake"


class TestRecommendation:
    def test_returns_every_section_the_brief_asked_for(
        self, recommendation: dict[str, Any]
    ) -> None:
        assert set(recommendation) >= {
            "request",
            "recommended",
            "alternatives",
            "itinerary",
            "budget",
            "constraints",
            "quality",
            "explanation",
            "data_sources",
        }

    def test_the_hard_budget_guarantee_holds_through_http(
        self, recommendation: dict[str, Any]
    ) -> None:
        assert recommendation["budget"]["total"] <= recommendation["request"]["budget_total"]
        assert recommendation["constraints"]["is_valid"] is True
        assert recommendation["constraints"]["hard_violation_count"] == 0

    def test_the_score_breakdown_travels_with_its_weights(
        self, recommendation: dict[str, Any]
    ) -> None:
        recommended = recommendation["recommended"]
        assert recommended["scores"]["value_for_money"] is not None
        assert pytest.approx(sum(recommended["weights"].values())) == 1.0

    def test_budget_lines_add_up(self, recommendation: dict[str, Any]) -> None:
        budget = recommendation["budget"]
        lines = sum(budget[key] for key in ("accommodation", "activities", "food", "transport"))
        assert pytest.approx(lines) == budget["total"]

    def test_every_source_declares_its_provenance(self, recommendation: dict[str, Any]) -> None:
        provenances = {source["provenance"] for source in recommendation["data_sources"]}
        assert "snapshot" in provenances  # retrieved records
        assert "synthetic" in provenances  # the food and transport estimates

    def test_the_explanation_is_labelled_and_checked(self, recommendation: dict[str, Any]) -> None:
        explanation = recommendation["explanation"]
        assert explanation["provenance"] in {"model_generated", "synthetic"}
        assert explanation["grounded"] is True

    def test_quality_signals_ship_with_the_answer(self, recommendation: dict[str, Any]) -> None:
        """A recommendation with no indication of what was known asks for blind trust."""
        quality = recommendation["quality"]
        assert quality["budget_compliant"] is True
        assert quality["hard_violations"] == 0
        assert quality["candidates_retrieved"] == 30
        assert quality["candidates_after_filters"] < quality["candidates_retrieved"]
        assert quality["preference_coverage"] == 1.0

    def test_the_itinerary_covers_the_stay(self, recommendation: dict[str, Any]) -> None:
        assert len(recommendation["itinerary"]) == recommendation["request"]["nights"]


class TestRefusals:
    def test_an_impossible_budget_is_422_with_an_explanation(self, client: TestClient) -> None:
        """Not 200-with-nothing: a caller must not mistake 'impossible' for 'none today'."""
        response = client.post("/trip/recommend", json=payload(budget_total=800))
        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "no_valid_plan"
        assert "hard constraint" in body["detail"]

    def test_an_unknown_destination_says_which_are_available(self, client: TestClient) -> None:
        response = client.post("/trip/recommend", json=payload(destination="Kyoto"))
        assert response.status_code == 422
        assert "Available" in response.json()["detail"]

    def test_a_party_larger_than_the_snapshot_is_refused(self, client: TestClient) -> None:
        response = client.post("/trip/recommend", json=payload(travelers=4))
        assert response.status_code == 422
        assert response.json()["error"] == "no_valid_plan"


class TestRequestValidation:
    def test_reversed_dates_are_rejected_before_any_work_happens(self, client: TestClient) -> None:
        response = client.post(
            "/trip/recommend", json=payload(start_date="2026-09-17", end_date="2026-09-10")
        )
        assert response.status_code == 422

    def test_a_misspelled_field_is_not_silently_ignored(self, client: TestClient) -> None:
        body = payload()
        body["buget_total"] = body.pop("budget_total")
        assert client.post("/trip/recommend", json=body).status_code == 422

    def test_an_unknown_preference_is_rejected(self, client: TestClient) -> None:
        response = client.post("/trip/recommend", json=payload(preferences=["skiing"]))
        assert response.status_code == 422

    def test_a_negative_budget_is_rejected(self, client: TestClient) -> None:
        assert client.post("/trip/recommend", json=payload(budget_total=-5)).status_code == 422


class TestSchema:
    def test_openapi_is_generated_from_the_domain_contracts(self, client: TestClient) -> None:
        """No separate API models to keep in sync: the domain contract is the schema."""
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        assert {"TripRecommendation", "ScoredAccommodation", "BudgetBreakdown"} <= set(schemas)

    def test_the_request_schema_splits_input_from_output(self, client: TestClient) -> None:
        """`nights` and `budget_per_night` are derived, so they are returned but never sent.

        Pydantic and FastAPI encode that distinction for free because the computed fields are
        declared on the model. A hand-written pair of API DTOs would have had to remember.
        """
        schemas = client.get("/openapi.json").json()["components"]["schemas"]
        request_in = schemas["TripRequest-Input"]["properties"]
        request_out = schemas["TripRequest-Output"]["properties"]
        assert "nights" not in request_in
        assert "nights" in request_out
        assert request_in["budget_total"]["exclusiveMinimum"] == 0

    def test_the_endpoint_documents_its_refusals(self, client: TestClient) -> None:
        responses = client.get("/openapi.json").json()["paths"]["/trip/recommend"]["post"][
            "responses"
        ]
        assert {"200", "422", "503"} <= set(responses)
