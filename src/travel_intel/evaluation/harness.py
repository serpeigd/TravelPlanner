"""Running the golden set and scoring what came back.

Every metric here answers a question someone would actually ask. None is included because it
is standard.

| Metric | Question |
|---|---|
| `passed` | Did it do the right *kind* of thing — plan when it can, refuse when it cannot? |
| `budget_compliant` | Does it ever overspend? Must be 100 %. |
| `hard_violations` | Does it ever return a plan it has proven invalid? Must be 0. |
| `preference_coverage` | Did the itinerary serve what the traveller asked for? |
| `budget_utilisation` | Is it leaving money unused, or scraping the ceiling? |
| `itinerary_fill` | How much of the stay actually has something booked? |
| `data_completeness` | How much do we know about what we recommended? |
| `candidates_*` | How many options survived each stage — is the funnel doing anything? |

`budget_compliant` and `hard_violations` are the two that must hold absolutely. The rest are
descriptive: there is no target value for budget utilisation, and pretending otherwise would
turn a diagnostic into a scoreboard.
"""

from __future__ import annotations

from dataclasses import dataclass

from travel_intel.config import Settings
from travel_intel.domain.errors import NoCandidatesError, ProviderError
from travel_intel.evaluation.scenarios import GOLDEN_SET, Expected, Scenario
from travel_intel.services.pipeline import run_pipeline


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    expected: Expected
    outcome: Expected
    probes: str

    refusal_reason: str | None = None
    budget_compliant: bool | None = None
    hard_violations: int | None = None
    soft_violations: int | None = None
    preference_coverage: float | None = None
    budget_utilisation: float | None = None
    itinerary_fill: float | None = None
    data_completeness: float | None = None
    candidates_ranked: int | None = None
    candidates_refused: int | None = None

    @property
    def passed(self) -> bool:
        """A refusal is a pass when refusing was correct — and a failure when it was not."""
        if self.outcome is not self.expected:
            return False
        if self.outcome is Expected.REFUSAL:
            return True
        return bool(self.budget_compliant) and self.hard_violations == 0


def evaluate_scenario(scenario: Scenario, settings: Settings | None = None) -> ScenarioResult:
    request = scenario.request
    try:
        trip = run_pipeline(request, settings)
    except (NoCandidatesError, ProviderError) as refusal:
        # Refusing is a legitimate outcome, not an error to be swallowed. What matters is
        # whether the refusal was the right answer and whether it explains itself.
        return ScenarioResult(
            scenario_id=scenario.id,
            expected=scenario.expected,
            outcome=Expected.REFUSAL,
            probes=scenario.probes,
            refusal_reason=str(refusal),
        )

    stated = len(request.preferences)
    covered = len(trip.itinerary.covered_preferences)
    days_with_activity = sum(1 for day in trip.itinerary.days if day.activity_ids)

    return ScenarioResult(
        scenario_id=scenario.id,
        expected=scenario.expected,
        outcome=Expected.PLAN,
        probes=scenario.probes,
        budget_compliant=trip.budget.total <= request.budget_total,
        hard_violations=trip.constraints.hard_violation_count,
        soft_violations=len(trip.constraints.violations) - trip.constraints.hard_violation_count,
        preference_coverage=round(covered / stated, 4) if stated else 1.0,
        budget_utilisation=trip.budget.utilization,
        itinerary_fill=round(days_with_activity / request.nights, 4),
        data_completeness=trip.recommended.accommodation.data_completeness,
        candidates_ranked=1 + len(trip.alternatives) + len(trip.rejected),
        candidates_refused=len(trip.rejected),
    )


def evaluate_scenarios(
    scenarios: tuple[Scenario, ...] = GOLDEN_SET,
    settings: Settings | None = None,
) -> tuple[ScenarioResult, ...]:
    return tuple(evaluate_scenario(scenario, settings) for scenario in scenarios)


def aggregate(results: tuple[ScenarioResult, ...]) -> dict[str, float]:
    """Headline numbers over the whole golden set."""
    planned = [r for r in results if r.outcome is Expected.PLAN]

    def mean(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "scenarios": float(len(results)),
        "passed": float(sum(r.passed for r in results)),
        "plans_produced": float(len(planned)),
        "refusals": float(len(results) - len(planned)),
        "budget_compliance_rate": mean([float(bool(r.budget_compliant)) for r in planned]),
        "hard_violations_total": float(sum(r.hard_violations or 0 for r in planned)),
        "mean_preference_coverage": mean([r.preference_coverage or 0.0 for r in planned]),
        "mean_budget_utilisation": mean([r.budget_utilisation or 0.0 for r in planned]),
        "mean_itinerary_fill": mean([r.itinerary_fill or 0.0 for r in planned]),
        "mean_data_completeness": mean([r.data_completeness or 0.0 for r in planned]),
    }
