# Ranking

## The formula

Each candidate gets factors on `[0, 1]` and a weighted average:

```
overall = Σ wᵢ · fᵢ  /  Σ wᵢ      over the factors that could actually be computed
```

| Factor | Weight | What it measures |
|---|---:|---|
| `budget_fit` | 0.25 | Total room cost against the accommodation share of the budget |
| `value_for_money` | 0.20 | Observed price vs. price predicted from the property's own features *(M5)* |
| `rating` | 0.20 | Guest rating, shrunk toward the market mean by review volume |
| `location` | 0.15 | Distance to the city centre |
| `preference_match` | 0.12 | Share of the user's evidence-bearing preferences the property supports |
| `data_completeness` | 0.08 | How much we actually know about the property |

Nothing here is fitted. These are **stated judgements**, deliberately visible and arguable,
because there is no relevance data to fit them against — see [ADR-003](decisions.md). The
effective weights actually applied to each candidate travel with the result in
`ScoredAccommodation.weights`.

## Why these weights

**Budget and value together carry 0.45.** Price is the dominant axis of a budget-constrained
trip, but splitting it in two is the point: `budget_fit` asks *can the traveller afford it*,
`value_for_money` asks *is it worth what it costs*. A cheap bad room scores well on the first
and badly on the second, which is precisely the distinction a traveller cares about and a
single price factor cannot express.

**Rating is 0.20 rather than higher** because it is the noisiest input. Every property is
rated on its own guests' expectations, so the scale is not truly comparable across segments —
a hostel and a five-star hotel both cluster around 8.5.

**Location is 0.15.** It matters, but distance to a single point is a crude proxy for what
travellers actually want, which is time-to-things-they-will-do. Weighting it higher would
overstate the quality of the underlying signal.

**Preference match is 0.12** because the property-level evidence is thin by nature: a
restaurant in the building is weak evidence about a food-focused trip.

**Completeness is 0.08 — a tie-breaker, not a driver.** Between two otherwise equal options,
prefer the one we actually know something about. Any higher and the ranking would start
recommending well-documented mediocrity.

## The factor curves

**`budget_fit`** is flat at 1.0 up to the allowance, then decays linearly to 0 at +50 %.

Flat, not increasing-as-cheaper. Money left over is a benefit to the traveller, but it is not
a reason to rank one room above another — otherwise the cheapest listing always wins and the
ranking degenerates into a price sort. Cheapness is a *value* question, and `value_for_money`
owns it. Conflating the two would double-count price.

The allowance is 45 % of the budget ([`budget.py`](../src/travel_intel/budget.py)), a
policy, not a constraint. Exceeding it is penalised, never disqualifying: **the only hard
budget constraint is the total**, and it is enforced in code before ranking (see
[`candidates.py`](../src/travel_intel/ranking/candidates.py)) and again on the finished plan.

For the demo request this matters visibly. The top result (Hotel The Celestine Tokyo Shiba,
€1,201 for the stay) is 6.8 % over the €1,125 allowance and still wins, because its rating,
amenities and location outweigh a `budget_fit` of 0.86. A hard cap would have hidden that
trade-off instead of showing it. There is a test asserting exactly this behaviour.

**`rating`** is the guest rating shrunk toward the market mean, then divided by 10:

```
shrunk = (rating · n_reviews + market_mean · m) / (n_reviews + m),   m = 200
```

A 9.2 from 136 reviews is a strong claim on weak evidence; an 8.2 from 23,178 is a weak claim
on strong evidence. Raw ratings systematically favour the first. Shrinkage pulls thin evidence
toward what the market as a whole looks like, in proportion to how thin it is: at 136 reviews
the 9.2 lands near 8.8, while at 23,178 reviews the 8.2 barely moves. `m = 200` sits at the
knee of this dataset's review-count distribution — established properties clear it, brand-new
listings do not.

The market mean is computed over the candidate set, so the comparison is always against the
alternatives actually on the table.

**`location`** decays linearly from 1.0 at the centre to 0 at 10 km. Straight-line distance
to Tokyo Station, computed from the provider's real coordinates. Beyond ~10 km every outing
becomes a planned commute rather than a walk or a short hop.

This factor is **not** distance to the user's preferred activities, which would be the better
feature. The attractions endpoint returns no coordinates, so that number cannot be computed
honestly — see [docs/data.md](data.md).

**`preference_match`** is the share of the user's preferences that the property supports,
counting only preferences for which a property carries evidence at all.

`FOOD`, `NATURE`, `RELAXATION` and `NIGHTLIFE` map to amenity sets. `CULTURE`, `ART`,
`HISTORY` and `SHOPPING` map to nothing, and are excluded from the denominator rather than
counted as unmatched — a hotel is not more or less cultural, and pretending otherwise would
manufacture signal out of nothing. Those preferences are served by the **itinerary**, which
is where they are genuinely matched.

For the demo request (`food, culture, nature`), the denominator is 2, not 3. If a request
carries *only* evidence-free preferences, the factor is missing entirely and its weight is
redistributed.

## Missing factors are dropped, never zeroed

If a factor cannot be computed, it is removed and the remaining weights are renormalised.

The alternative — imputing 0 — would say "we don't know" and "it's terrible" in the same
number. An unrated property would be buried not for being bad but for being new. It would
also be a *double* penalty, since not knowing is already priced in through
`data_completeness`.

One property in the Tokyo snapshot has no rating at all. It keeps its budget, location,
preference and completeness factors, its `rating` weight is spread across those, and its
`weights` map shows `rating` simply absent. This same mechanism is what lets
`value_for_money` be declared today and start contributing in M5 with no re-tuning.

## What this ranking does not do

- **It is not learned.** No click logs exist, so there is nothing to learn from. Fabricating
  relevance labels would produce a model that recovers the heuristic used to fabricate them.
- **It does not consult an LLM.** The ordering is deterministic, reproducible and testable;
  an LLM adds non-determinism and no information the factors do not already carry.
- **It does not enforce constraints.** Scoring ranks; validation rejects. Keeping those
  separate is what makes "the LLM cannot violate the budget" a property of the architecture
  rather than a hope.
