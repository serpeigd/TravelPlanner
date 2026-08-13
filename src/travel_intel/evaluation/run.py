"""The evaluation report: `python -m travel_intel.evaluation.run`.

Offline, seeded, and reproducible — same snapshot, same numbers, every time. It does not need
a model server: the grounding section evaluates the check itself against scripted adversarial
text, which is a sharper test of the guard than watching one model behave.
"""

from __future__ import annotations

from travel_intel.config import Settings
from travel_intel.evaluation.grounding_suite import adversarial_cases, grounding_detection
from travel_intel.evaluation.harness import aggregate, evaluate_scenarios
from travel_intel.evaluation.metrics import factor_ablation, rank_stability
from travel_intel.evaluation.scenarios import GOLDEN_SET, Expected
from travel_intel.llm.explain import build_payload
from travel_intel.llm.grounding import build_context
from travel_intel.ml.price_model import (
    DistrictMedianBaseline,
    GlobalMedianBaseline,
    HedonicPriceModel,
    cross_validate,
)
from travel_intel.ml.report import training_frame
from travel_intel.ranking import generate_candidates
from travel_intel.retrieval.factory import build_providers
from travel_intel.services.pipeline import run_pipeline

RULE = "-" * 78


def _section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> None:
    settings = Settings()
    reference = GOLDEN_SET[0].request

    print("Travel Intelligence - evaluation report")
    print(f"data mode: {settings.data_mode.value}  |  offline and seeded")

    # -- 1. the golden set -------------------------------------------------------------
    _section("1. GOLDEN SET  -  did the system do the right kind of thing?")
    results = evaluate_scenarios(settings=settings)
    print(f"  {'scenario':<24} {'expected':<9} {'got':<9} {'pass':<5} detail")
    for result in results:
        detail = (
            f"budget {result.budget_utilisation:.1%}, prefs {result.preference_coverage:.0%}, "
            f"fill {result.itinerary_fill:.0%}, {result.candidates_refused} refused"
            if result.outcome is Expected.PLAN
            else (result.refusal_reason or "")[:44]
        )
        print(
            f"  {result.scenario_id:<24} {result.expected.value:<9} {result.outcome.value:<9} "
            f"{'OK' if result.passed else 'FAIL':<5} {detail}"
        )

    totals = aggregate(results)
    print(
        f"\n  {int(totals['passed'])}/{int(totals['scenarios'])} passed  |  "
        f"{int(totals['plans_produced'])} plans, {int(totals['refusals'])} refusals"
    )
    print(f"  budget compliance      {totals['budget_compliance_rate']:.0%}   (must be 100%)")
    print(f"  hard violations        {int(totals['hard_violations_total'])}      (must be 0)")
    print(f"  preference coverage    {totals['mean_preference_coverage']:.0%}")
    print(f"  budget utilisation     {totals['mean_budget_utilisation']:.0%}")
    print(f"  itinerary fill         {totals['mean_itinerary_fill']:.0%}")
    print(f"  data completeness      {totals['mean_data_completeness']:.0%}")

    # -- 2. reproducibility ------------------------------------------------------------
    _section("2. REPRODUCIBILITY  -  does the same request give the same answer?")
    first = run_pipeline(reference, settings)
    second = run_pipeline(reference, settings)
    identical = (
        first.recommended.accommodation.id == second.recommended.accommodation.id
        and first.budget.total == second.budget.total
        and [a.id for a in first.itinerary.selected] == [a.id for a in second.itinerary.selected]
    )
    print(f"  two runs identical: {identical}")

    # -- 3. ranking robustness ---------------------------------------------------------
    _section("3. RANKING ROBUSTNESS  -  conclusion, or artefact of my weights?")
    provider, _ = build_providers(settings)
    candidates = generate_candidates(
        provider.search_accommodations(reference).records, reference
    ).records

    stability = rank_stability(candidates, reference)
    print(f"  {stability.trials} trials, weights jittered +/-{stability.jitter:.0%}")
    print(f"  mean Kendall tau vs baseline order   {stability.mean_kendall_tau:+.3f}")
    print(f"  mean Jaccard over the top 5          {stability.mean_jaccard_at_k:.3f}")
    print(f"  top-1 unchanged                      {stability.top1_unchanged_rate:.0%}")

    print("\n  Factor ablation (drop one, renormalise the rest; most disruptive first):")
    print(f"  {'factor':<20} {'tau':>7} {'jaccard@5':>10} {'mean shift':>11}  top-1")
    for entry in factor_ablation(candidates, reference):
        print(
            f"  {entry.factor:<20} {entry.kendall_tau:>+7.3f} {entry.jaccard_at_k:>10.3f} "
            f"{entry.mean_rank_shift:>11.2f}  {'CHANGED' if entry.top1_changed else 'same'}"
        )

    # -- 4. grounding ------------------------------------------------------------------
    _section("4. GROUNDING  -  does the tripwire fire, and only when it should?")
    context = build_context(build_payload(first))
    cases = adversarial_cases(
        hotel_id=first.recommended.accommodation.id,
        stay_total=first.recommended.total_cost,
        plan_total=first.budget.total,
    )
    report = grounding_detection(context, cases)
    caught = f"{report.detected}/{report.detectable} dishonest cases caught"
    flagged = f"{report.false_positives}/{report.clean_cases} honest cases wrongly flagged"
    print(f"  detection rate         {report.detection_rate:.0%}  ({caught})")
    print(f"  false positive rate    {report.false_positive_rate:.0%}  ({flagged})")
    if report.blind_spots:
        print("  known blind spots (reported, not hidden):")
        for blind_spot in report.blind_spots:
            print(f"    - {blind_spot}")

    # -- 5. price model ----------------------------------------------------------------
    _section("5. PRICE MODEL  -  is the ML component earning its place?")
    frame = training_frame(settings)
    metrics = cross_validate(
        frame, (GlobalMedianBaseline(), DistrictMedianBaseline(), HedonicPriceModel())
    )
    print(f"  {len(frame)} properties, repeated 5-fold CV, 5 repeats, identical folds")
    for name, price_metrics in metrics.items():
        print("  " + price_metrics.as_row(name))

    print()


if __name__ == "__main__":
    main()
