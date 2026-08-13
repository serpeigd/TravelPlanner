"""The evaluation layer, and the whole system through it.

`test_every_scenario_passes` is the broadest regression test in the suite: it runs the entire
pipeline over eleven requests, including three where the correct answer is a refusal. If any
stage breaks, it fails here.
"""

import pytest

from travel_intel.config import Settings
from travel_intel.evaluation.grounding_suite import (
    Verdict,
    adversarial_cases,
    grounding_detection,
)
from travel_intel.evaluation.harness import aggregate, evaluate_scenarios
from travel_intel.evaluation.metrics import (
    factor_ablation,
    jaccard_at_k,
    kendall_tau,
    rank_stability,
)
from travel_intel.evaluation.scenarios import GOLDEN_SET, Expected
from travel_intel.llm.explain import build_payload
from travel_intel.llm.grounding import build_context
from travel_intel.ranking import generate_candidates
from travel_intel.retrieval.factory import build_providers
from travel_intel.services.pipeline import run_pipeline

SETTINGS = Settings()


@pytest.fixture(scope="module")
def results() -> object:
    return evaluate_scenarios(settings=SETTINGS)


@pytest.fixture(scope="module")
def candidates() -> object:
    request = GOLDEN_SET[0].request
    provider, _ = build_providers(SETTINGS)
    return generate_candidates(provider.search_accommodations(request).records, request).records


class TestGoldenSet:
    def test_every_scenario_passes(self, results: object) -> None:
        failures = [r.scenario_id for r in results if not r.passed]  # type: ignore[attr-defined]
        assert failures == []

    def test_the_suite_contains_cases_where_refusing_is_correct(self) -> None:
        """A suite of only happy paths measures nothing."""
        refusals = [s for s in GOLDEN_SET if s.expected is Expected.REFUSAL]
        assert len(refusals) >= 3

    def test_no_plan_ever_exceeds_its_budget(self, results: object) -> None:
        totals = aggregate(results)  # type: ignore[arg-type]
        assert totals["budget_compliance_rate"] == 1.0

    def test_no_plan_carries_a_hard_violation(self, results: object) -> None:
        totals = aggregate(results)  # type: ignore[arg-type]
        assert totals["hard_violations_total"] == 0.0

    def test_every_refusal_explains_itself(self, results: object) -> None:
        for result in results:  # type: ignore[attr-defined]
            if result.outcome is Expected.REFUSAL:
                assert result.refusal_reason
                assert len(result.refusal_reason) > 20

    def test_an_unmatchable_preference_still_yields_a_stay(self, results: object) -> None:
        """Nothing in the snapshot is tagged shopping: warn, but do not refuse a hotel."""
        result = next(r for r in results if r.scenario_id == "unmatchable-preference")  # type: ignore[attr-defined]
        assert result.outcome is Expected.PLAN
        assert result.preference_coverage == 0.0
        assert result.soft_violations >= 1


class TestReproducibility:
    def test_the_same_request_gives_the_same_answer(self) -> None:
        request = GOLDEN_SET[0].request
        first = run_pipeline(request, SETTINGS).trip
        second = run_pipeline(request, SETTINGS).trip
        assert first.recommended.accommodation.id == second.recommended.accommodation.id
        assert first.budget.total == second.budget.total
        assert [a.id for a in first.itinerary.selected] == [a.id for a in second.itinerary.selected]


class TestRankCorrelation:
    def test_identical_orders_correlate_perfectly(self) -> None:
        assert kendall_tau(["a", "b", "c"], ["a", "b", "c"]) == 1.0

    def test_reversed_orders_anticorrelate(self) -> None:
        assert kendall_tau(["a", "b", "c"], ["c", "b", "a"]) == -1.0

    def test_a_single_swap_is_between(self) -> None:
        tau = kendall_tau(["a", "b", "c"], ["b", "a", "c"])
        assert -1.0 < tau < 1.0

    def test_jaccard_measures_shortlist_membership(self) -> None:
        assert jaccard_at_k(["a", "b", "c"], ["a", "b", "c"], k=2) == 1.0
        assert jaccard_at_k(["a", "b"], ["c", "d"], k=2) == 0.0
        assert jaccard_at_k(["a", "b"], ["b", "a"], k=2) == 1.0  # order-blind by design


class TestRankStability:
    def test_the_ranking_survives_jittered_weights(self, candidates: object) -> None:
        """The weights are the most arbitrary part of the system.

        If a +/-20 % jitter reshuffled the shortlist, the recommendation would be an artefact
        of my choices rather than of the data, and no amount of documentation would fix that.
        """
        report = rank_stability(candidates, GOLDEN_SET[0].request)  # type: ignore[arg-type]
        assert report.mean_kendall_tau > 0.85
        assert report.mean_jaccard_at_k > 0.9
        assert report.top1_unchanged_rate > 0.8

    def test_stability_is_seeded(self, candidates: object) -> None:
        request = GOLDEN_SET[0].request
        first = rank_stability(candidates, request)  # type: ignore[arg-type]
        second = rank_stability(candidates, request)  # type: ignore[arg-type]
        assert first == second


class TestFactorAblation:
    def test_one_entry_per_factor_sorted_by_disruption(self, candidates: object) -> None:
        entries = factor_ablation(candidates, GOLDEN_SET[0].request)  # type: ignore[arg-type]
        assert len(entries) == 6
        taus = [entry.kendall_tau for entry in entries]
        assert taus == sorted(taus)

    def test_completeness_is_the_tie_breaker_it_was_designed_to_be(
        self, candidates: object
    ) -> None:
        """Documented in docs/ranking.md as a tie-breaker; the ablation confirms it."""
        entries = {e.factor: e for e in factor_ablation(candidates, GOLDEN_SET[0].request)}  # type: ignore[arg-type]
        assert entries["data_completeness"].kendall_tau > 0.95
        assert entries["data_completeness"].mean_rank_shift < 0.5

    def test_the_price_model_earns_its_weight(self, candidates: object) -> None:
        """`value_for_money` is the most disruptive factor: the ML is not decoration."""
        entries = factor_ablation(candidates, GOLDEN_SET[0].request)  # type: ignore[arg-type]
        assert entries[0].factor == "value_for_money"


@pytest.fixture(scope="module")
def context() -> object:
    """The grounding context of the reference plan, with the figures the cases need."""
    trip = run_pipeline(GOLDEN_SET[0].request, SETTINGS).trip
    return (
        build_context(build_payload(trip)),
        trip.recommended.accommodation.id,
        trip.recommended.total_cost,
        trip.budget.total,
    )


class TestGroundingSuite:
    def test_every_dishonest_case_is_caught(self, context: object) -> None:
        ctx, hotel_id, stay_total, plan_total = context  # type: ignore[misc]
        report = grounding_detection(ctx, adversarial_cases(hotel_id, stay_total, plan_total))
        assert report.detection_rate == 1.0

    def test_no_honest_case_is_flagged(self, context: object) -> None:
        """A check that fires on truthful output trains everyone to ignore it."""
        ctx, hotel_id, stay_total, plan_total = context  # type: ignore[misc]
        report = grounding_detection(ctx, adversarial_cases(hotel_id, stay_total, plan_total))
        assert report.false_positive_rate == 0.0

    def test_the_known_blind_spot_is_reported_not_hidden(self, context: object) -> None:
        ctx, hotel_id, stay_total, plan_total = context  # type: ignore[misc]
        report = grounding_detection(ctx, adversarial_cases(hotel_id, stay_total, plan_total))
        assert len(report.blind_spots) == 1
        assert "entailment" in report.blind_spots[0]

    def test_the_suite_covers_both_directions(self) -> None:
        cases = adversarial_cases("h1", 1000.0, 2000.0)
        verdicts = {case.verdict for case in cases}
        assert verdicts == {Verdict.CLEAN, Verdict.DETECT, Verdict.BLIND_SPOT}
