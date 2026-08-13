"""Budget allocation policy.

A trip budget is not one number, it is four. Splitting it up front is what lets the ranking
ask a well-posed question — "does this room fit the *accommodation* share?" — instead of the
meaningless "is this room cheaper than the whole trip?".

The shares below encode domain judgement, not a fitted model, and they are stated as data so
they can be argued with, overridden per request, and tested.
"""

from __future__ import annotations

from dataclasses import dataclass

from travel_intel.domain.enums import BudgetCategory

SHARE_TOLERANCE = 1e-6


@dataclass(frozen=True)
class BudgetPolicy:
    """How a total budget is split across categories, as fractions summing to 1.

    Defaults are tuned for a multi-night city trip where the traveller flies in and moves
    around by public transport:

    - **accommodation 45 %** — the single largest and least compressible line. Booked first,
      it is also the one decision the rest of the trip has to live with.
    - **food 25 %** — two or three meals a day for the whole party across the whole stay.
      Steady and predictable, which is why it is a share rather than an itemised list.
    - **activities 20 %** — paid attractions and tours. Genuinely discretionary: this is the
      line a traveller flexes when accommodation overruns.
    - **transport 10 %** — local transit and airport transfers only. International flights
      are explicitly *out of scope*: they are usually booked separately and would dominate
      the total, making every other share meaningless.
    """

    accommodation: float = 0.45
    food: float = 0.25
    activities: float = 0.20
    transport: float = 0.10

    def __post_init__(self) -> None:
        shares = self.as_dict()
        if any(share <= 0 for share in shares.values()):
            raise ValueError("every budget share must be positive")
        total = sum(shares.values())
        if abs(total - 1.0) > SHARE_TOLERANCE:
            raise ValueError(f"budget shares must sum to 1.0, got {total}")

    def as_dict(self) -> dict[BudgetCategory, float]:
        return {
            BudgetCategory.ACCOMMODATION: self.accommodation,
            BudgetCategory.FOOD: self.food,
            BudgetCategory.ACTIVITIES: self.activities,
            BudgetCategory.TRANSPORT: self.transport,
        }

    def allowance(self, category: BudgetCategory, budget_total: float) -> float:
        """The slice of `budget_total` available for `category`, rounded to cents."""
        return round(budget_total * self.as_dict()[category], 2)


DEFAULT_BUDGET_POLICY = BudgetPolicy()
