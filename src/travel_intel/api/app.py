"""FastAPI application.

This file is short, and that is the point rather than an accident. `TripRequest` and
`TripRecommendation` were designed as domain contracts in M1; FastAPI validates the first,
serialises the second, and generates the OpenAPI schema from both. There is no separate set
of API models to keep in sync, because there is nothing for them to translate.

The endpoint is a plain `def`, not `async def`, on purpose. The pipeline is CPU-bound (pandas,
scikit-learn) and the explanation stage blocks on a local model for around two minutes.
Declaring it `async` would run all of that directly on the event loop and stall every other
request; as a sync endpoint, Starlette runs it in a worker thread instead.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from travel_intel import __version__
from travel_intel.config import Settings, get_settings
from travel_intel.domain.errors import LLMError, NoCandidatesError, ProviderError
from travel_intel.domain.models import TripRecommendation, TripRequest
from travel_intel.recommend import recommend

DESCRIPTION = """
Trip recommendation over a frozen snapshot of real Booking.com data.

Deterministic Python owns retrieval, ranking, budgeting and constraint validation. A local
LLM interprets free-text preferences and writes the explanation — after the plan has already
passed every hard constraint, and only if its output survives a grounding check.
"""


class HealthResponse(BaseModel):
    status: str
    version: str
    data_mode: str
    llm_provider: str


class ProblemResponse(BaseModel):
    """A refusal that explains itself.

    `detail` is the domain's own message, which names the constraint that could not be met
    rather than saying "no results". An unexplained empty answer is the failure mode this
    whole system is built to avoid.
    """

    error: str
    detail: str


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or get_settings()
    application = FastAPI(
        title="Travel Intelligence",
        description=DESCRIPTION,
        version=__version__,
    )
    application.dependency_overrides[get_settings] = lambda: resolved

    @application.exception_handler(NoCandidatesError)
    def _no_candidates(_: Request, error: NoCandidatesError) -> JSONResponse:
        # 422, not 404: the request was well-formed, it simply cannot be satisfied. And not
        # 200-with-an-empty-list, which would let a caller mistake "impossible" for "nothing
        # matched today".
        return JSONResponse(
            status_code=422,
            content=ProblemResponse(error="no_valid_plan", detail=str(error)).model_dump(),
        )

    @application.exception_handler(ProviderError)
    def _provider_down(_: Request, error: ProviderError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=ProblemResponse(error="provider_unavailable", detail=str(error)).model_dump(),
        )

    @application.exception_handler(LLMError)
    def _llm_down(_: Request, error: LLMError) -> JSONResponse:
        # Should be unreachable: the explainer falls back to the deterministic template rather
        # than propagating. Handled anyway, so a future caller of the LLM cannot turn a model
        # outage into a 500.
        return JSONResponse(
            status_code=503,
            content=ProblemResponse(error="llm_unavailable", detail=str(error)).model_dump(),
        )

    @application.get("/health", response_model=HealthResponse, tags=["ops"])
    def health(config: Annotated[Settings, Depends(get_settings)]) -> HealthResponse:
        """Liveness plus the two settings that change what the answers mean."""
        return HealthResponse(
            status="ok",
            version=__version__,
            data_mode=config.data_mode.value,
            llm_provider=config.llm_provider.value,
        )

    @application.post(
        "/trip/recommend",
        response_model=TripRecommendation,
        responses={422: {"model": ProblemResponse}, 503: {"model": ProblemResponse}},
        tags=["recommendation"],
    )
    def recommend_trip(
        trip_request: TripRequest,
        config: Annotated[Settings, Depends(get_settings)],
    ) -> TripRecommendation:
        """Plan a trip, or explain why it cannot be planned.

        The response carries the recommendation, ranked alternatives, the itinerary, the
        budget broken down by line, the constraint report, quality signals and the provenance
        of every source used.
        """
        return recommend(trip_request, config)

    return application


app = create_app()
