# Travel Intelligence

A trip-recommendation system built to be **defended, not demoed**. Given a destination, dates,
party size, budget and preferences, it retrieves accommodation and activity candidates, ranks
them with an explainable score, enforces hard constraints in code, and explains the result
with a local LLM that is structurally unable to invent a fact.

Three principles drive every decision in this repository:

> **An LLM is a component of the system, not the system.**
>
> **Business logic belongs in deterministic, testable code** wherever that is possible.
>
> **Evaluation must measure whether the system works**, not whether the demo looks impressive.

| | |
|---|---|
| Tests | 245 (`pytest`), `mypy --strict` clean, `ruff` clean |
| Data | 30 real Tokyo properties, 27 activities, captured 2026-08-13 and frozen |
| Price model | MAE €40.85 vs €73.32 for the best baseline — a 44 % cut |
| Golden set | 11/11 scenarios pass, 100 % budget compliance, 0 hard violations |
| Grounding | 100 % detection, 0 % false positives, 1 blind spot reported |

## Quickstart

Requires Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,ui]"
pytest
```

```bash
python -m travel_intel.demo               # the whole pipeline, printed stage by stage
python -m travel_intel.evaluation.run      # the evaluation report
python -m travel_intel.ml.report           # the price model against its baselines
uvicorn travel_intel.api.app:app --reload  # API + docs at /docs
streamlit run src/travel_intel/ui/streamlit_app.py
```

Everything except the explanation is offline and deterministic: it reads a frozen snapshot of
real Booking.com data. The explanation follows `TRAVEL_INTEL_LLM_PROVIDER` — `fake` (the
default in CI) uses a deterministic template; `ollama` calls a local `llama3.1:8b`. See
`.env.example`.

### Hosted and containerized runs

`requirements.txt` exists only for hosts that insist on one (Streamlit Community Cloud,
Hugging Face Spaces). It contains a single line — `.[ui]` — so the real dependency list stays
in `pyproject.toml` and cannot drift from it. Local development should still use
`pip install -e ".[dev,ui]"`.

`.devcontainer/devcontainer.json` opens the repository in Codespaces and starts the Streamlit
UI on port 8501 automatically. **Known mismatch:** that container pins the
`python:1-3.11-bookworm` image while this package declares `requires-python = ">=3.12"`
(matching CI, `ruff`, and `mypy`), so the install step fails there until the image is bumped
to 3.12 — see [Limitations](#14-limitations).

---

## 1. The problem

Plan a trip end to end under a hard budget: pick where to stay, decide what to do, cost the
whole thing, and explain the choice — without ever quoting a price that is not real or
promising a trip that does not fit.

The reference request: **Tokyo, 10–17 September, 2 travellers, €2,500, food / culture /
nature.**

## 2. Why this is interesting from a Data Science perspective

It is a problem where the *obvious* ML move is the wrong one, and saying so is the work.

- **The recommendation looks like learning-to-rank** — but relevance labels (clicks, bookings)
  do not exist, and fabricating them trains a model to recover the heuristic used to fabricate
  them. There is exactly one real target in this data, and it is price.
- **Hard constraints and probabilistic models mix badly.** A budget is not a soft preference
  to be traded off; it either holds or the answer is wrong. That forces a clear split between
  what is scored and what is enforced.
- **The data is genuinely incomplete**, in ways that matter: one property has no rating,
  activities have no coordinates, provider category tags are noisy. Every one of those forces
  a decision about what the system may honestly claim.
- **Small-sample discipline.** Thirty rows is not a dataset. Knowing what that supports — five
  features, repeated cross-validation, two baselines — is more of the job than fitting.

## 3. Architecture

A linear pipeline of pure functions. No agent framework, no LangChain: the control flow is
fixed and knowable, so a model-driven loop would add non-determinism and latency for nothing.

```
TripRequest → retrieval → candidates → features → ml → ranking
           → itinerary → budget → [CONSTRAINT BARRIER] → llm → TripRecommendation
```

The barrier is load-bearing. By the time any text is generated, a plan that breaks a hard
constraint has already been discarded.

Full detail, including failure modes and the scaling path: **[docs/architecture.md](docs/architecture.md)**.
Decisions and their rejected alternatives: **[docs/decisions.md](docs/decisions.md)**.

## 4. Data sources

Real Booking.com data for Tokyo, captured once through an MCP connector on 2026-08-13 and
committed as a versioned fixture: **30 properties** (€70–€647 per night) and **27 activities**
across three categories.

Three accommodation searches rather than one, because a single query returns ten
near-identical mid-range hotels and a price model needs variance. One search *per category*
for activities, because the provider's category ids are opaque — so each activity is tagged
with the categories whose search returned it, which makes the tag the provider's claim rather
than my guess.

Every record carries a `Provenance`: `snapshot`, `real_api`, `synthetic`, or
`model_generated`. Nothing synthetic is ever presented as retrieved.

Capture decisions and what the data cannot support: **[docs/data.md](docs/data.md)**.

## 5. Data flow

`price.book` from the provider is the **total for the stay**, not a nightly rate — reading it
wrong put a Setagaya apartment at €333/night and corrupted every budget calculation without
raising a single error. That conversion now lives in exactly one place, with a test.

The feature table is built once and has two consumers: the ranking and the price model.
`MODEL_FEATURES` marks the request-independent subset, so the query cannot leak into a model
that is supposed to describe the property.

Missing stays missing through the whole pipeline. Nothing is imputed at the feature layer;
`NaN` reaches the scorer, which drops the factor and redistributes its weight.

## 6. Ranking

Six factors on `[0, 1]`, weighted-averaged, with the effective weights carried on every
result:

| Factor | Weight | Measures |
|---|---:|---|
| `budget_fit` | 0.25 | Room cost against its share of the budget |
| `value_for_money` | 0.20 | Observed price vs. price predicted from the property's own features |
| `rating` | 0.20 | Guest rating, shrunk toward the market mean by review volume |
| `location` | 0.15 | Distance to the city centre |
| `preference_match` | 0.12 | Share of evidence-bearing preferences supported |
| `data_completeness` | 0.08 | How much we actually know |

Two rules worth arguing about:

**Ratings are shrunk, not used raw.** A 9.2 from 136 reviews and an 8.2 from 23,178 are not
the same claim. Each is pulled toward the market mean in proportion to how thin its evidence
is, with a prior of 200 reviews for hotels and 25 for activities.

**Uncomputable factors are dropped, not zeroed.** Imputing 0 would say "we don't know" and
"it's terrible" with the same number, and would penalise twice — not knowing is already priced
in through `data_completeness`.

The formula, every weight's justification, and the activity-selection algorithm:
**[docs/ranking.md](docs/ranking.md)**.

## 7. Machine Learning

A **hedonic price regression** — ridge on log price over five property attributes. The
prediction is worthless on its own; nobody needs an estimate of a price that is printed on the
page. The **residual** is the product: a room priced well below what its stars, amenities,
location and reviews predict is good value *for what it is*.

Repeated 5-fold cross-validation, 5 repeats, identical folds for every estimator:

| Estimator | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| baseline: global median | €80.98 | €154.23 | 31.4 % |
| baseline: district median | €73.32 | €142.43 | 26.7 % |
| **hedonic ridge (log price)** | **€40.85** | **€77.14** | **20.9 %** |

Had it not beaten the district median, that would have been the finding and it would be
reported here as such.

One coefficient is worth knowing: **guest rating barely explains price** (−0.0141) once stars
(+0.2914) and amenities (+0.2251) are known. That is why `rating` and `value_for_money` are
separate factors — they carry different information.

Modelling choices, limitations, and what would make this a real model:
**[docs/ml.md](docs/ml.md)**.

## 8. Evaluation

```bash
python -m travel_intel.evaluation.run
```

Offline, seeded, reproducible, and wired into CI. Five sections, each answering a question
someone would actually ask.

**Golden set** — 11 scenarios, three of which *must be refused*. A suite of only happy paths
measures nothing. 11/11 pass, **100 % budget compliance**, **0 hard violations**.

**Ranking quality without relevance labels.** nDCG and Precision@K need to know which result
should have won. Instead: perturb the weights ±20 % over 25 seeded trials and see whether the
answer survives — mean Kendall τ **+0.961**, Jaccard@5 **1.000**, top-1 unchanged **92 %**.
Then ablate each factor. `value_for_money` turns out to be the most disruptive, so the ML is
not decoration; `data_completeness` moves the order by 0.08 positions, confirming the
tie-breaker role documented before it was measured.

**Grounding** — nine adversarial cases, both directions: **100 % detection, 0 % false
positives**, plus one blind spot counted and printed rather than omitted.

Metrics, targets, and what is deliberately *not* measured:
**[docs/evaluation.md](docs/evaluation.md)**.

## 9. Reliability and hallucination prevention

Four layers, in order:

1. **Ordering.** `validate_plan` runs on the complete costed plan, before any text exists. The
   planner walks the ranking until a plan passes every hard constraint. A budget-violating
   plan is never constructed.
2. **Schema validation.** Every LLM output is parsed into a Pydantic model. Terms outside the
   controlled vocabulary are surfaced as `unmapped`, never silently dropped.
3. **Grounding.** The allowed set is derived by walking the exact payload handed to the model,
   so the check cannot drift from the prompt. Four hard checks: entity ids, money figures,
   currency, and per-night claims.
4. **Discard, never repair.** Failing text is thrown away and the deterministic template
   answers, with the reasons recorded on the `Explanation`. Fixing a hallucinated price still
   leaves prose built around it.

Three of those checks exist because `llama3.1:8b` actually made those mistakes — quoting the
budget in dollars, and calling a seven-night total a nightly rate. Details, and what the check
still **cannot** catch: **[docs/llm.md](docs/llm.md)**.

## 10. Testing

245 tests, `mypy --strict` over 48 modules, `ruff` for lint and format, all four in CI with no
network and no model server.

The tests are written as claims about behaviour, not coverage padding. The interesting ones
are the adversarial cases — a `ScriptedClient` produces malformed and dishonest model output
on demand, which a real model obligingly would not — and the golden-set regression, which
drives the entire pipeline over eleven requests including three refusals.

## 11. Scalability

Prototype → service → distributed, in [docs/architecture.md](docs/architecture.md#scaling-prototype--service--distributed).
The short version: the retrieval `Protocol` is already the seam for a real client; the price
model moves from per-request fitting to an offline-trained versioned artefact; feature
building becomes a Spark job over a price history; and the ranking stays in-process, because
it is milliseconds of arithmetic over a few dozen candidates.

## 12. Azure → AWS

Nothing in this system is cloud-specific, and the deployment target is the least interesting
decision on the list. [docs/architecture.md](docs/architecture.md#azure--aws) carries a 1:1
mapping so the same architecture can be described in either vocabulary — Databricks ↔
EMR/Glue, Azure ML ↔ SageMaker, Data Factory ↔ Glue/Step Functions, Blob ↔ S3, Monitor ↔
CloudWatch, Key Vault ↔ Secrets Manager.

What transfers is not the service names but the design: separate compute from storage,
partition by date, version artefacts instead of mutating them, keep the inference path
stateless.

## 13. What would change in production

Offline-trained and versioned price model; chronological rather than random validation splits;
asynchronous explanation generation; weights *derived* from booking data rather than chosen;
monitoring on factor distributions rather than only outputs; refusal rate treated as a product
metric. Expanded in [docs/architecture.md](docs/architecture.md#what-would-change-in-production).

## 14. Limitations

Stated plainly, because a limitation you have to be told about is a limitation you hid.

- **Thirty properties, one city, one date range, one party size.** The price model is
  demonstrative with honest error bars, not a production estimator.
- **No temporal dimension.** Seasonality, day-of-week and lead time are the obvious missing
  variables, and none of them exists in a single snapshot.
- **Activities have no coordinates**, so hotel-to-activity distance cannot be computed and the
  ranking does not pretend to use it. `location` means distance to Tokyo Station.
- **Provider category tags are noisy** — a bar tour is returned under `nature_outdoor`. The
  noise is preserved rather than hand-corrected, because hiding it would hide a real data
  problem.
- **Grounding cannot catch claim-level misattribution.** "8.9 out of 5 stars" uses two real
  values and no money; catching it needs entailment checking, not figure matching.
- **Live retrieval is an unimplemented seam.** No credentials exist, and an untested HTTP
  client would be code that claims to work.
- **Food and transport are stated assumptions**, deliberately independent of the budget so the
  compliance check stays meaningful.
- **~100–128 s per explanation** on CPU with an 8B model.
- **The devcontainer's Python version is behind the package's.**
  `.devcontainer/devcontainer.json` pins `python:1-3.11-bookworm`, while `pyproject.toml`
  declares `requires-python = ">=3.12"` — the same version CI, `ruff` and `mypy` all target.
  A Codespace built from that file therefore fails on install. Fixing it means either bumping
  the image or lowering the floor, and the second would mean re-checking every 3.12-only
  construct in `src/`, so it is left as a flagged mismatch rather than a silent one.

## 15. Future improvements

1. **A price history**, which turns a cross-sectional hedonic regression into a demand-and-price
   model and "is this good value today" into "book now or wait".
2. **Claim-level verification** of the explanation — an NLI pass, or a structured assertion
   format the code can check one item at a time.
3. **Booking or click data**, at which point learning-to-rank stops being fabrication.
4. **Routing distances** for a `location` factor that measures time-to-things rather than
   kilometres-to-a-point.
5. **More destinations**, which would immediately test how much of the ranking is Tokyo-shaped.
6. **A text classifier over activity titles** to repair the provider's noisy category tags.

---

## Documentation

| | |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | Pipeline, provenance, failure modes, scaling, Azure→AWS |
| [`docs/decisions.md`](docs/decisions.md) | ADRs, each with the alternative that was rejected |
| [`docs/data.md`](docs/data.md) | The dataset, capture-time decisions, what it cannot support |
| [`docs/ranking.md`](docs/ranking.md) | The formula, every weight, the activity algorithm |
| [`docs/ml.md`](docs/ml.md) | The hedonic model, results against two baselines |
| [`docs/llm.md`](docs/llm.md) | The LLM's two jobs, the grounding check, what it misses |
| [`docs/evaluation.md`](docs/evaluation.md) | Golden set, ranking robustness, what is not measured |

## License and legal notice

Copyright © 2026 Sergio Peigneux d'Egmont ([@serpeigd](https://github.com/serpeigd)).

The source code and documentation in this repository are released under the
[MIT License](LICENSE) — use it, fork it, learn from it, including commercially, as long as
the copyright notice and licence text travel with it. It is provided **as is, without warranty
of any kind**; the full disclaimer is in the LICENSE file.

That grant covers this project's own code and prose. It does **not** cover, and cannot
relicense, the following:

- **The dataset under `data/fixtures/`.** A small sample of publicly listed Booking.com data
  for one city and one date range, captured once on 2026-08-13 through the Booking.com MCP
  connector and attributed with source URLs. It is committed so the pipeline is reproducible
  and its numbers auditable — **not** as a redistributable dataset, and not under the MIT
  grant above. Property names, descriptions, imagery references, review scores and prices
  remain the property of Booking.com and the individual properties. If you fork this
  repository for anything beyond study, review Booking.com's terms before reusing that
  fixture, and re-capture your own data rather than treating this one as a source.
- **Prices, availability and ratings** in that snapshot were true at one instant in 2026 and
  are now stale by construction. Nothing here is a booking service, a price quote, or travel
  advice; the itineraries and budgets it produces are illustrative output of a portfolio
  system, and `docs/data.md` states exactly what the data cannot support.
- **Third-party components** keep their own licences: [Ollama](https://ollama.com) and the
  `llama3.1:8b` weights it serves (only used when `TRAVEL_INTEL_LLM_PROVIDER=ollama`), plus
  FastAPI, Streamlit, scikit-learn and the rest of the dependency tree in `pyproject.toml`.

Nothing in this notice is a legal opinion.
