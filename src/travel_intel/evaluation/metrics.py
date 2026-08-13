"""Ranking quality without relevance labels.

The textbook metrics — nDCG, Precision@K, MRR — all need to know which result *should* have
won. That means clicks or bookings, and this project has neither. Hand-labelling thirty
hotels would not fix it: the labels would encode my own opinion of the ranking, and the
metric would then confirm it. Fabricated ground truth measures the fabricator.

So this module asks two questions that *can* be answered without labels, and that a reviewer
can check for themselves:

1. **Is the ranking a conclusion or a knife-edge?** Perturb the weights and see whether the
   answer survives. Weights are the most arbitrary part of the system; if a ±20 % jitter
   reshuffles the top five, the recommendation was an artefact of my choices rather than of
   the data.
2. **Which factors actually drive it?** Remove one at a time and measure how far the ranking
   moves. A factor that changes nothing is dead weight in the formula; one that changes
   everything deserves more scrutiny than a number in a config.

Both are reproducible: the jitter is seeded.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from travel_intel.domain.models import Accommodation, TripRequest
from travel_intel.ml.price_model import HedonicPriceModel
from travel_intel.ranking.scoring import DEFAULT_WEIGHTS, rank_accommodations

STABILITY_TRIALS = 25
STABILITY_JITTER = 0.20
STABILITY_SEED = 42
TOP_K = 5


def kendall_tau(first: Sequence[str], second: Sequence[str]) -> float:
    """Rank correlation over the items both orderings contain.

    +1 is identical order, -1 is exactly reversed, 0 is unrelated. Implemented directly
    rather than pulling in scipy for one function over a few dozen items.
    """
    position_a = {item: index for index, item in enumerate(first)}
    position_b = {item: index for index, item in enumerate(second)}
    shared = [item for item in first if item in position_b]

    concordant = 0
    discordant = 0
    for i in range(len(shared)):
        for j in range(i + 1, len(shared)):
            left, right = shared[i], shared[j]
            direction = (position_a[left] - position_a[right]) * (
                position_b[left] - position_b[right]
            )
            if direction > 0:
                concordant += 1
            elif direction < 0:
                discordant += 1

    total = concordant + discordant
    return 1.0 if total == 0 else round((concordant - discordant) / total, 4)


def jaccard_at_k(first: Sequence[str], second: Sequence[str], k: int = TOP_K) -> float:
    """Overlap of the two top-k sets, ignoring order.

    Complements Kendall tau: a traveller looks at the shortlist, not at position 19 versus
    position 20. Order within the shortlist matters less than membership of it.
    """
    top_a = set(first[:k])
    top_b = set(second[:k])
    union = top_a | top_b
    return 1.0 if not union else round(len(top_a & top_b) / len(union), 4)


@dataclass(frozen=True)
class StabilityReport:
    mean_kendall_tau: float
    mean_jaccard_at_k: float
    top1_unchanged_rate: float
    trials: int
    jitter: float


def rank_stability(
    candidates: tuple[Accommodation, ...],
    request: TripRequest,
    *,
    trials: int = STABILITY_TRIALS,
    jitter: float = STABILITY_JITTER,
    seed: int = STABILITY_SEED,
) -> StabilityReport:
    """How much does the ranking depend on the exact weights I chose?

    Each trial multiplies every weight by a factor drawn uniformly from
    `[1 - jitter, 1 + jitter]` and re-ranks. High agreement means the data is doing the work;
    low agreement would mean the recommendation is an artefact of the weighting.
    """
    baseline = _order(candidates, request, DEFAULT_WEIGHTS)
    rng = random.Random(seed)

    taus: list[float] = []
    jaccards: list[float] = []
    top1_matches = 0

    for _ in range(trials):
        perturbed = {
            name: weight * rng.uniform(1.0 - jitter, 1.0 + jitter)
            for name, weight in DEFAULT_WEIGHTS.items()
        }
        order = _order(candidates, request, perturbed)
        taus.append(kendall_tau(baseline, order))
        jaccards.append(jaccard_at_k(baseline, order))
        top1_matches += int(bool(order) and order[0] == baseline[0])

    return StabilityReport(
        mean_kendall_tau=round(sum(taus) / trials, 4),
        mean_jaccard_at_k=round(sum(jaccards) / trials, 4),
        top1_unchanged_rate=round(top1_matches / trials, 4),
        trials=trials,
        jitter=jitter,
    )


@dataclass(frozen=True)
class AblationEntry:
    factor: str
    top1_changed: bool
    kendall_tau: float
    jaccard_at_k: float
    mean_rank_shift: float


def factor_ablation(
    candidates: tuple[Accommodation, ...],
    request: TripRequest,
) -> tuple[AblationEntry, ...]:
    """Drop each factor in turn and measure how far the ranking moves.

    Dropping a factor is not the same as zeroing it: the weights renormalise over what
    remains, exactly as they do when a factor is genuinely uncomputable. So this measures the
    factor's *contribution*, not the effect of scoring everything zero on it.

    Sorted by disruption, most disruptive first — which doubles as an explanation of what the
    ranking is really made of.
    """
    baseline = _order(candidates, request, DEFAULT_WEIGHTS)
    baseline_positions = {item: index for index, item in enumerate(baseline)}

    entries: list[AblationEntry] = []
    for factor in DEFAULT_WEIGHTS:
        reduced = {name: w for name, w in DEFAULT_WEIGHTS.items() if name != factor}
        if not reduced:
            continue
        order = _order(candidates, request, reduced)
        shifts = [
            abs(index - baseline_positions[item])
            for index, item in enumerate(order)
            if item in baseline_positions
        ]
        entries.append(
            AblationEntry(
                factor=factor,
                top1_changed=bool(order) and order[0] != baseline[0],
                kendall_tau=kendall_tau(baseline, order),
                jaccard_at_k=jaccard_at_k(baseline, order),
                mean_rank_shift=round(sum(shifts) / len(shifts), 3) if shifts else 0.0,
            )
        )

    return tuple(sorted(entries, key=lambda entry: entry.kendall_tau))


def _order(
    candidates: tuple[Accommodation, ...],
    request: TripRequest,
    weights: Mapping[str, float],
) -> list[str]:
    ranked = rank_accommodations(
        candidates, request, weights=weights, price_model=HedonicPriceModel()
    )
    return [item.accommodation.id for item in ranked]
