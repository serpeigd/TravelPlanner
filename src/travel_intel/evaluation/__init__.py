"""Evaluation: does the system actually work, or does the demo just look impressive?"""

from travel_intel.evaluation.grounding_suite import (
    AdversarialCase,
    GroundingReport,
    Verdict,
    adversarial_cases,
    grounding_detection,
)
from travel_intel.evaluation.harness import (
    ScenarioResult,
    aggregate,
    evaluate_scenario,
    evaluate_scenarios,
)
from travel_intel.evaluation.metrics import (
    factor_ablation,
    jaccard_at_k,
    kendall_tau,
    rank_stability,
)
from travel_intel.evaluation.scenarios import GOLDEN_SET, Expected, Scenario

__all__ = [
    "GOLDEN_SET",
    "AdversarialCase",
    "Expected",
    "GroundingReport",
    "Scenario",
    "ScenarioResult",
    "Verdict",
    "adversarial_cases",
    "aggregate",
    "evaluate_scenario",
    "evaluate_scenarios",
    "factor_ablation",
    "grounding_detection",
    "jaccard_at_k",
    "kendall_tau",
    "rank_stability",
]
