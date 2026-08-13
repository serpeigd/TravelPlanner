# Evaluation

> Evaluation should measure whether the system works, not whether the demo looks impressive.

```bash
python -m travel_intel.evaluation.run
```

Offline, seeded, reproducible: same snapshot, same folds, same jitter, same numbers. It needs
no model server — the grounding section tests the *check* against scripted adversarial text,
which is a sharper measurement than watching one model behave on one afternoon.

Every metric below answers a question someone would actually ask. None is here because it is
standard.

---

## 1. The golden set — did the system do the right *kind* of thing?

Eleven requests, **three of which should be refused**. A suite of only happy paths measures
nothing: the interesting question is not whether a comfortable budget produces a plan, it is
whether an impossible one produces an honest refusal instead of a confident fiction.

| Scenario | Expected | Got | What it probes |
|---|---|---|---|
| `reference` | plan | plan ✅ | the demo case |
| `tight-budget` (€2,100) | plan | plan ✅ | 7 preferred options become undeliverable once fully costed |
| `very-tight-budget` (€1,600) | plan | plan ✅ | 24 refused; the fixed food/transport floor bites |
| `impossible-budget` (€800) | refusal | refusal ✅ | below the floor alone — refuse, don't improvise |
| `generous-budget` (€5,000) | plan | plan ✅ | does more money buy worse value? |
| `single-preference` | plan | plan ✅ | coverage with one interest |
| `no-preferences` | plan | plan ✅ | empty list means "no opinion", not "nothing qualifies" |
| `unmatchable-preference` | plan | plan ✅ | nothing is tagged shopping: warn, still deliver a stay |
| `short-stay` (3 nights) | plan | plan ✅ | extrapolation warnings against a 7-night snapshot |
| `party-too-large` (4) | refusal | refusal ✅ | capacity unverified above the searched party size |
| `unknown-destination` | refusal | refusal ✅ | no snapshot: explicit error, not an empty plan |

**11/11 passed** — 8 plans, 3 refusals.

| Metric | Result | Target |
|---|---:|---|
| Budget compliance | **100 %** | must be 100 % |
| Hard violations | **0** | must be 0 |
| Preference coverage | 88 % | descriptive |
| Budget utilisation | 71 % | descriptive |
| Itinerary fill | 79 % | descriptive |
| Data completeness | 100 % | descriptive |

The first two are absolute: a plan that overspends or carries a hard violation is a bug, not
a low score. The rest are **descriptive on purpose**. There is no correct budget utilisation —
71 % across a set that includes a deliberately generous scenario is information, not a grade.
Turning a diagnostic into a scoreboard invites optimising the number instead of the system.

Preference coverage is 88 % rather than 100 % because of `unmatchable-preference`, where the
snapshot contains nothing tagged shopping. Scoring that as a failure would punish the system
for the dataset's contents; it reports the gap as a soft violation and still returns a stay.

---

## 2. Reproducibility — same request, same answer?

Two runs of the reference request produce the identical accommodation, the identical budget
total and the identical itinerary. **True.**

Trivial-looking, and load-bearing: a ranking that reshuffles between runs cannot be evaluated
at all, and every number on this page would be noise.

---

## 3. Ranking robustness — a conclusion, or an artefact of my weights?

This is the honest answer to *"how do you evaluate ranking quality without relevance
labels?"*

nDCG, Precision@K and MRR all need to know which result should have won — clicks or bookings.
This project has neither. Hand-labelling thirty hotels would not fix it: the labels would
encode my own opinion of the ranking and the metric would then confirm it. **Fabricated
ground truth measures the fabricator.** So instead:

### Stability under weight perturbation

The weights are the most arbitrary part of the system. 25 trials, every weight multiplied by
a factor drawn uniformly from ±20 %, seeded:

| | |
|---|---:|
| Mean Kendall τ vs. the baseline order | **+0.961** |
| Mean Jaccard over the top 5 | **1.000** |
| Top-1 unchanged | **92 %** |

The shortlist never changed. The recommendation is being driven by the data, not by my choice
of 0.25 versus 0.20. Had the top five reshuffled, the right conclusion would have been that
the ranking was an artefact of the weighting — and no amount of documentation would have
fixed it.

### Factor ablation

Drop one factor and renormalise the rest — the same mechanism that runs when a factor is
genuinely uncomputable, so this measures each factor's *contribution* rather than the effect
of scoring everything zero.

| Factor dropped | Kendall τ | Jaccard@5 | Mean rank shift | Top-1 |
|---|---:|---:|---:|---|
| `value_for_money` | +0.557 | 1.000 | 4.15 | same |
| `budget_fit` | +0.631 | 1.000 | 4.54 | same |
| `location` | +0.748 | 1.000 | 2.62 | **changed** |
| `preference_match` | +0.791 | 0.250 | 2.23 | **changed** |
| `rating` | +0.920 | 1.000 | 1.00 | **changed** |
| `data_completeness` | +0.994 | 1.000 | 0.08 | same |

Three things worth saying out loud:

- **The ML component is doing the most work.** Removing `value_for_money` disturbs the order
  more than removing anything else. The price model is not decoration bolted onto a heuristic.
- **`data_completeness` is a tie-breaker, exactly as designed.** τ = 0.994 and a mean shift of
  0.08 positions. That was the stated intent in `docs/ranking.md` before this was measured;
  the ablation confirms it rather than discovering it.
- **`preference_match` reshuffles the shortlist without reshuffling the ranking.** τ stays
  high at 0.791 but Jaccard@5 collapses to 0.250 — it barely moves the overall order while
  changing three of the five options a traveller would actually look at. A single correlation
  number would have hidden that completely, which is why both are reported.

---

## 4. Grounding — does the tripwire fire, and only when it should?

Nine cases run through the real check: five dishonest, three honest, one known blind spot.

| | Result |
|---|---:|
| Detection rate | **100 %** (5/5 dishonest caught) |
| False positive rate | **0 %** (0/3 honest flagged) |

Both directions matter equally. A check that never fires is decoration; a check that fires on
truthful output is *worse than nothing*, because it teaches everyone to ignore it. That is not
hypothetical — it happened in M6, when a greedy regex read a sentence comma as a decimal
separator and rejected a correct explanation. One of the three honest cases is the regression
test for it.

Three of the five dishonest cases are things `llama3.1:8b` actually did, unprompted: quoting
the budget in dollars, and calling a seven-night total a nightly rate.

### The blind spot, reported rather than omitted

```
rating-scale-conflation: "The hotel scores 8.9 out of 5 stars."
```

Both values are real and in the payload, and neither is money — figure-matching cannot see
it. Catching this needs **claim-level entailment**: an NLI model, or a second structured pass
where the explainer emits assertions that code can check one at a time. This project has not
solved it, and the suite counts and prints the miss rather than quietly leaving the case out.

---

## 5. The price model — is the ML earning its place?

30 properties, repeated 5-fold cross-validation, 5 repeats, every estimator on identical
folds (paired, not approximate):

| Estimator | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| baseline: global median | €80.98 | €154.23 | 31.4 % |
| baseline: district median | €73.32 | €142.43 | 26.7 % |
| **hedonic ridge (log price)** | **€40.85** | **€77.14** | **20.9 %** |

A 44 % cut in mean absolute error against the district median — the honest competitor, since
location explains much of price in any city and a grouped median needs no model at all. Full
detail in [`docs/ml.md`](ml.md).

---

## What is deliberately not measured

- **nDCG / Precision@K / MRR.** No relevance labels exist. See section 3.
- **Explanation quality.** Whether the prose is *good* is a human judgement; whether it is
  *true* is section 4, and truth is the part that can break a user's trip.
- **Latency.** Measured (~100–128 s for an explanation on CPU, `docs/llm.md`) but not
  optimised. It is a property of running an 8B model on a laptop, not of the architecture.
- **A/B or online metrics.** No users, no traffic. Naming the metrics a production system
  would need — booking conversion, plan-abandonment, price-freshness alarms — is honest;
  reporting numbers for them would not be.
