"""Evaluating the anti-hallucination check itself.

Most evaluation suites measure the happy path. This one measures the guard: given text that
is deliberately dishonest, does the tripwire fire — and given text that is honest, does it
stay quiet? Both directions matter equally. A check that never fires is decoration; a check
that fires on truthful output is worse than nothing, because it teaches everyone to ignore
it. That second failure is not hypothetical here: it happened, in M6, when a greedy regex
read a sentence comma as a decimal separator and rejected a correct explanation.

The suite also carries a case it is **expected to miss**, and reports it as a known blind
spot rather than quietly leaving it out. Claiming full coverage of a problem this system has
not solved would be the same dishonesty the check exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from travel_intel.llm.grounding import GroundingContext, check_grounding


class Verdict(StrEnum):
    CLEAN = "clean"
    """Honest text. Any violation is a false positive."""
    DETECT = "detect"
    """Dishonest text. Failing to flag it is a miss."""
    BLIND_SPOT = "blind_spot"
    """Known to slip through. Counted and reported, never hidden."""


@dataclass(frozen=True)
class AdversarialCase:
    id: str
    text: str
    referenced_ids: tuple[str, ...]
    verdict: Verdict
    note: str


@dataclass(frozen=True)
class GroundingReport:
    detected: int
    detectable: int
    false_positives: int
    clean_cases: int
    blind_spots: tuple[str, ...]

    @property
    def detection_rate(self) -> float:
        return round(self.detected / self.detectable, 4) if self.detectable else 1.0

    @property
    def false_positive_rate(self) -> float:
        return round(self.false_positives / self.clean_cases, 4) if self.clean_cases else 0.0


def adversarial_cases(
    hotel_id: str, stay_total: float, plan_total: float
) -> tuple[AdversarialCase, ...]:
    """Cases built from the plan under test, so the figures are the real ones."""
    return (
        AdversarialCase(
            id="honest-restatement",
            text=f"The stay costs {stay_total:.2f} EUR and the trip totals {plan_total:.2f} EUR.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.CLEAN,
            note="exact figures, exact ids",
        ),
        AdversarialCase(
            id="honest-rounded",
            text=f"The trip comes to about {plan_total:.0f} EUR.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.CLEAN,
            note="natural rounding must not be punished",
        ),
        AdversarialCase(
            id="honest-punctuated",
            text=f"Accommodation EUR {stay_total:.2f}, then activities, then food.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.CLEAN,
            note="regression: a trailing comma was once read as a decimal separator",
        ),
        AdversarialCase(
            id="invented-total",
            text="The whole trip comes to just 1899 EUR.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.DETECT,
            note="a plausible figure that appears nowhere in the plan",
        ),
        AdversarialCase(
            id="invented-hotel",
            text="We recommend the Imperial Garden Palace.",
            referenced_ids=(hotel_id, "hotel-9999"),
            verdict=Verdict.DETECT,
            note="an entity id that is not in the plan",
        ),
        AdversarialCase(
            id="wrong-currency",
            text=f"The trip totals ${plan_total:.2f}.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.DETECT,
            note="right number, wrong currency: observed from llama3.1:8b",
        ),
        AdversarialCase(
            id="stay-total-as-nightly",
            text=f"The room is {stay_total:.2f} EUR per night.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.DETECT,
            note="the figure is real, the claim attached to it is false",
        ),
        AdversarialCase(
            id="suspiciously-round",
            text="Everything in, it works out at 2400 EUR.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.DETECT,
            note="close to the true total but not in the plan; tolerance must not wave it through",
        ),
        AdversarialCase(
            id="rating-scale-conflation",
            text="The hotel scores 8.9 out of 5 stars.",
            referenced_ids=(hotel_id,),
            verdict=Verdict.BLIND_SPOT,
            note=(
                "both values are real and neither is money, so figure-matching cannot see it. "
                "Catching this needs claim-level entailment, not a grounding check"
            ),
        ),
    )


def grounding_detection(
    context: GroundingContext,
    cases: tuple[AdversarialCase, ...],
) -> GroundingReport:
    """Run every case through the real check and score both directions."""
    detected = 0
    detectable = 0
    false_positives = 0
    clean_cases = 0
    blind_spots: list[str] = []

    for case in cases:
        violations = check_grounding(case.text, case.referenced_ids, context)
        if case.verdict is Verdict.DETECT:
            detectable += 1
            detected += int(bool(violations))
        elif case.verdict is Verdict.CLEAN:
            clean_cases += 1
            false_positives += int(bool(violations))
        elif not violations:
            blind_spots.append(f"{case.id}: {case.note}")

    return GroundingReport(
        detected=detected,
        detectable=detectable,
        false_positives=false_positives,
        clean_cases=clean_cases,
        blind_spots=tuple(blind_spots),
    )
