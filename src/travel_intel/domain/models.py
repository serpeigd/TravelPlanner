"""Contracts for the whole pipeline.

Two rules hold everywhere in this module:

1. Money is a rounded ``float`` in a single currency. These are trip *estimates*, not
   ledger entries; `Decimal` would buy accounting precision we do not need and would fight
   pandas/scikit-learn downstream. See docs/decisions.md (ADR-002).
2. `price_per_night` is the cost of the unit for the *whole party*, which is what
   accommodation providers quote for a search of N guests. Keeping that convention in one
   place removes an entire family of per-person/per-room budget bugs.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, computed_field, model_validator

from travel_intel.domain.enums import BudgetCategory, Currency, Preference, Provenance, Severity

MAX_NIGHTS = 30
MAX_TRAVELERS = 12


def _round_eur(value: float) -> float:
    return round(value, 2)


Amount = Annotated[float, Field(ge=0), AfterValidator(_round_eur)]
"""A non-negative money amount, rounded to cents at every boundary."""

UnitInterval = Annotated[float, Field(ge=0.0, le=1.0)]
"""A normalised score. Every ranking factor lives on this scale so weights are comparable."""


class Frozen(BaseModel):
    """Immutable, strict base for every value object in the domain."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------------------


class GeoPoint(Frozen):
    lat: float = Field(ge=-90.0, le=90.0)
    lon: float = Field(ge=-180.0, le=180.0)


class TripRequest(Frozen):
    """What the user asks for. The single source of truth for every hard constraint."""

    destination: str = Field(min_length=2, max_length=80)
    start_date: date
    end_date: date
    travelers: int = Field(ge=1, le=MAX_TRAVELERS)
    budget_total: Amount = Field(gt=0)
    """Total budget for the whole party and the whole trip."""
    currency: Currency = Currency.EUR
    preferences: tuple[Preference, ...] = ()
    notes: str | None = Field(default=None, max_length=1000)
    """Free-text wishes. The only field the LLM is allowed to interpret."""

    @model_validator(mode="after")
    def _check_dates(self) -> TripRequest:
        # Deliberately not compared against "today": the domain owns no clock, so golden
        # fixtures stay reproducible forever.
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        if (self.end_date - self.start_date).days > MAX_NIGHTS:
            raise ValueError(f"trip length must not exceed {MAX_NIGHTS} nights")
        return self

    @model_validator(mode="after")
    def _dedupe_preferences(self) -> TripRequest:
        unique = tuple(dict.fromkeys(self.preferences))
        if unique != self.preferences:
            object.__setattr__(self, "preferences", unique)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days

    @computed_field  # type: ignore[prop-decorator]
    @property
    def budget_per_night(self) -> float:
        return _round_eur(self.budget_total / self.nights)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def budget_per_person(self) -> float:
        return _round_eur(self.budget_total / self.travelers)


# --------------------------------------------------------------------------------------
# Retrieved entities
# --------------------------------------------------------------------------------------


class Accommodation(Frozen):
    """One bookable option, as returned by a provider adapter."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provenance: Provenance
    price_per_night: Amount = Field(gt=0)
    """Cost per night for the whole party (see module docstring)."""
    currency: Currency = Currency.EUR
    max_occupancy: int = Field(ge=1, le=MAX_TRAVELERS)
    rating: float | None = Field(default=None, ge=0.0, le=10.0)
    """Provider rating on a 0-10 scale. Adapters normalise 5-star scales on the way in."""
    review_count: int | None = Field(default=None, ge=0)
    location: GeoPoint | None = None
    neighborhood: str | None = None
    distance_to_center_km: float | None = Field(default=None, ge=0.0)
    amenities: tuple[str, ...] = ()
    source_url: str | None = None
    retrieved_at: datetime | None = None

    COMPLETENESS_FIELDS: ClassVar[tuple[str, ...]] = (
        "rating",
        "review_count",
        "location",
        "neighborhood",
        "distance_to_center_km",
        "source_url",
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def data_completeness(self) -> float:
        """Share of the optional-but-decision-relevant fields this record actually has.

        Feeds both a ranking factor (an option we know little about should not win by
        default) and a data-quality metric in the evaluation layer.
        """
        present = sum(getattr(self, f) is not None for f in self.COMPLETENESS_FIELDS)
        return round(present / len(self.COMPLETENESS_FIELDS), 4)

    def total_cost(self, nights: int) -> float:
        if nights < 1:
            raise ValueError("nights must be >= 1")
        return _round_eur(self.price_per_night * nights)


class Activity(Frozen):
    """Something to do at the destination, tagged with the same vocabulary as preferences."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    provenance: Provenance
    categories: tuple[Preference, ...] = ()
    price_per_person: Amount = 0.0
    duration_hours: float = Field(default=2.0, gt=0.0, le=24.0)
    rating: float | None = Field(default=None, ge=0.0, le=10.0)
    review_count: int | None = Field(default=None, ge=0)
    location: GeoPoint | None = None
    neighborhood: str | None = None
    source_url: str | None = None
    retrieved_at: datetime | None = None

    def cost_for(self, travelers: int) -> float:
        if travelers < 1:
            raise ValueError("travelers must be >= 1")
        return _round_eur(self.price_per_person * travelers)


# --------------------------------------------------------------------------------------
# Ranking output
# --------------------------------------------------------------------------------------


class ScoreBreakdown(Frozen):
    """Why a candidate got its score. Every factor is on [0, 1] and separately auditable."""

    budget_fit: UnitInterval
    location: UnitInterval
    rating: UnitInterval
    preference_match: UnitInterval
    data_completeness: UnitInterval
    value_for_money: UnitInterval | None = None
    """ML-derived: observed price vs. price predicted from the option's own features.

    None when the price model is unavailable, in which case its weight is redistributed
    over the remaining factors rather than silently scored as zero.
    """

    def as_dict(self) -> dict[str, float]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


class ScoredAccommodation(Frozen):
    accommodation: Accommodation
    scores: ScoreBreakdown
    overall: UnitInterval
    weights: dict[str, float]
    """The exact weights applied to produce `overall`, carried for auditability."""
    total_cost: Amount
    rank: int = Field(ge=1)


# --------------------------------------------------------------------------------------
# Plan output
# --------------------------------------------------------------------------------------


class ItineraryDay(Frozen):
    day_index: int = Field(ge=1)
    day: date
    activity_ids: tuple[str, ...] = ()
    estimated_cost: Amount = 0.0


class BudgetBreakdown(Frozen):
    """Deterministically computed. The LLM never touches these numbers."""

    budget_total: Amount
    accommodation: Amount
    activities: Amount
    food: Amount
    transport: Amount

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> float:
        return _round_eur(self.accommodation + self.activities + self.food + self.transport)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def remaining(self) -> float:
        return _round_eur(self.budget_total - self.total)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def utilization(self) -> float:
        return round(self.total / self.budget_total, 4)

    def by_category(self) -> dict[BudgetCategory, float]:
        return {
            BudgetCategory.ACCOMMODATION: self.accommodation,
            BudgetCategory.ACTIVITIES: self.activities,
            BudgetCategory.FOOD: self.food,
            BudgetCategory.TRANSPORT: self.transport,
        }


class ConstraintViolation(Frozen):
    code: str = Field(min_length=1)
    severity: Severity
    message: str = Field(min_length=1)
    observed: float | None = None
    limit: float | None = None


class ConstraintReport(Frozen):
    violations: tuple[ConstraintViolation, ...] = ()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_valid(self) -> bool:
        """A plan is valid iff it breaks no hard constraint. Soft ones only warn."""
        return not any(v.severity is Severity.HARD for v in self.violations)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def hard_violation_count(self) -> int:
        return sum(v.severity is Severity.HARD for v in self.violations)


class Explanation(Frozen):
    """LLM prose, permanently labelled as such and only ever about grounded records."""

    text: str
    provenance: Provenance = Provenance.MODEL_GENERATED
    grounded: bool
    """False if the grounding check failed; such text is dropped before it reaches a user."""
    model: str | None = None
    referenced_ids: tuple[str, ...] = ()


class DataSourceInfo(Frozen):
    name: str
    provenance: Provenance
    record_count: int = Field(ge=0)
    retrieved_at: datetime | None = None
    url: str | None = None


class QualitySignals(Frozen):
    """Per-request evaluation, surfaced next to the recommendation itself."""

    budget_compliant: bool
    hard_violations: int = Field(ge=0)
    preference_coverage: UnitInterval
    mean_data_completeness: UnitInterval
    candidates_retrieved: int = Field(ge=0)
    candidates_after_filters: int = Field(ge=0)
    explanation_grounded: bool | None = None


class TripRecommendation(Frozen):
    """The single response object of the system."""

    request: TripRequest
    recommended: ScoredAccommodation
    alternatives: tuple[ScoredAccommodation, ...] = ()
    itinerary: tuple[ItineraryDay, ...] = ()
    budget: BudgetBreakdown
    constraints: ConstraintReport
    quality: QualitySignals
    explanation: Explanation | None = None
    data_sources: tuple[DataSourceInfo, ...] = ()
