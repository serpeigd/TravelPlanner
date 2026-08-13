# Architecture Decision Records

Short records of the decisions that are worth defending out loud. Each one states the
alternative that was rejected, because that is usually the more interesting half.

---

## ADR-001 — No agent framework; an explicit function pipeline

**Status:** accepted

**Context.** The problem is a fixed sequence: normalise the request, retrieve options,
generate candidates, build features, rank, validate constraints, explain, evaluate. There
is no branching that the system cannot decide in code, and no step where the *next* step
depends on a model's judgement.

**Decision.** Plain Python functions with explicit inputs and outputs. No LangChain, no
LangGraph, no agent loop.

**Consequences.** Every stage is unit-testable in isolation and the data flow is readable
end to end. The cost is that adding genuinely open-ended behaviour later (e.g. multi-turn
negotiation with the user) would need real orchestration — at which point a framework
becomes justified.

**Rejected.** An agent loop with tools. It would add non-determinism and latency to a
pipeline whose control flow is already known, and would make the evaluation layer measure
the agent's planning rather than the recommendation quality.

---

## ADR-002 — Money as rounded `float`, single currency

**Status:** accepted

**Context.** The system produces *estimates* (a nightly rate times seven nights, a food
allowance), not accounting entries. Values flow into pandas and scikit-learn.

**Decision.** A single `Amount` type: non-negative `float`, rounded to cents at every model
boundary. One currency (EUR) enforced by an enum.

**Consequences.** No `Decimal`/`float` conversions in the feature pipeline. Rounding is
applied consistently rather than only at display time, so budget arithmetic is stable.

**Rejected.** `Decimal`. It buys precision this domain does not need and creates friction
with the numeric stack. If the system ever issued real quotes or bookings, this would flip.

---

## ADR-003 — The ML component is a price model, not a learning-to-rank model

**Status:** accepted

**Context.** The obvious "ML in a recommender" move is learning-to-rank. It needs relevance
labels — clicks, bookings, dwell time. We have none, and inventing them would produce a
model that only learns the heuristic used to fabricate the labels.

**Decision.** Model the one target that genuinely exists in the data: **price per night**,
regressed on the option's own features (location, rating, review volume, capacity,
amenities, season). The prediction is not the product; the **residual** is. An option
priced well below its predicted price is good value for what it is, and that becomes a
ranking factor (`value_for_money`) with an honest error metric (MAE / RMSE / MAPE against a
median-by-neighborhood baseline).

**Consequences.** A real supervised model with a real target, real metrics and a real
baseline, feeding a business question the domain actually asks. The ranking itself stays
deterministic and explainable.

**Rejected.** LambdaMART/XGBoost-ranker on synthetic relevance labels. Rejecting it *is*
the point: the project should show it knows when not to use ML.

---

## ADR-004 — Snapshot-first data adapter, live mode behind the same interface

**Status:** accepted

**Context.** Real data makes the demo credible; reproducibility makes the evaluation
meaningful. Live API calls give the first and destroy the second.

**Decision.** One `AccommodationProvider` / `ActivityProvider` Protocol with two
implementations. `snapshot` reads real provider data captured once and committed as a
versioned fixture, and is the default and the only mode used by tests and evaluation.
`live` is the seam for a real API client and is **deliberately unimplemented**: the snapshot
was captured through an assistant-side MCP connector, which is not an endpoint this process
can call, and no provider credentials exist. It raises `ProviderError` with an explanatory
message rather than silently falling back to the snapshot — a fallback would make the mode
flag a lie.

**Consequences.** The evaluation harness is deterministic and runs offline in CI. Every
record carries a `Provenance` value, so retrieved facts, frozen snapshots, synthetic rows
and model prose are never confused with each other in the UI or in the answer. Any request
that diverges from the captured query (different dates, longer stay, larger party) returns
an explicit warning instead of a confidently wrong price. What live mode would need is
listed in `retrieval/live.py`; nothing downstream changes.

**Rejected.** Calling the live API everywhere and mocking it in tests. That leaves the
demo's own numbers unreproducible and the fixture drifting from the real payload shape.

---

## ADR-005 — Local LLM (Ollama), with a deterministic fake for tests

**Status:** accepted

**Context.** The LLM has exactly two jobs: turning free-text wishes into structured
constraints, and writing the explanation over decisions already made in code. Neither
requires a frontier model, and neither should make the test suite depend on a network.

**Decision.** `llama3.1:8b` via Ollama, behind an `LLMClient` interface with a `fake`
implementation selected by configuration. Every LLM output is parsed into a Pydantic schema;
anything that fails validation or references an entity absent from the retrieved data is
discarded (`GroundingError`), never repaired and shown.

**Consequences.** Zero cost, runs offline, tests are deterministic. Explanation quality is
capped by an 8B model — acceptable, since the explanation only reformulates numbers that
deterministic code already computed.

**Rejected.** A hosted API as the default. It would add cost and a network dependency to a
project that must run offline on a laptop.

---

## ADR-006 — Constraints are checked on the finished plan, and the planner walks the ranking

**Status:** accepted

**Context.** The ranking answers "which accommodation is best". It does not answer "is the
resulting trip deliverable". A property can top the ranking and still push the total over
budget once food, transport and activities are added — and it is exactly this arithmetic
that a fluent language model gets confidently wrong.

**Decision.** Two separate stages. `validate_plan` checks a *complete costed plan* and
returns a report; `plan_trip` walks the ranking in order and returns the first option whose
plan passes every hard constraint, keeping the refused ones for explanation. Violations are
typed: **hard** (over budget, too small for the party, wrong currency) means the plan is
never returned; **soft** (over the accommodation allowance, an uncovered preference, stale
snapshot dates) means it is returned with a warning attached.

**Consequences.** "The LLM cannot violate a hard constraint" becomes a property of the
architecture rather than a hope about prompting: by the time any text is generated, an
invalid plan has already been discarded. Rejections are visible instead of silent, which is
what makes the funnel explainable. When nothing validates, the system raises rather than
answering.

**Rejected.** Enforcing the budget as a filter during candidate generation. It cannot work —
the room is only one of four cost lines, so affordability is not knowable until the plan
exists. Candidate generation keeps only the filters that are unarguable in isolation.

**Rejected.** Returning the best invalid plan with a warning. A plan the system has just
proven impossible is worse than an explicit refusal.

---

## ADR-007 — Food and transport are stated assumptions, not budget shares

**Status:** accepted

**Context.** Accommodation and activities have retrieved prices. Food and local transport do
not — no provider quotes what a traveller will spend on dinner. The lines still have to be
in the budget, or "does this trip fit?" is answered against a fiction.

**Decision.** Per-person, per-day estimates per destination (Tokyo: €45 food, €12 transport),
declared `Provenance.SYNTHETIC`, carried into the response as their own `DataSourceInfo`, and
charged for `nights` days rather than the calendar span.

**Consequences.** The budget check is meaningful: because the estimates are independent of
the budget, utilisation varies and a plan can genuinely fail. Retrieved prices and assumed
ones are never confused in the output.

**Rejected.** Setting food and transport equal to their budget shares. It is circular — the
plan would consume exactly 100 % of any budget by construction, utilisation would always be
1.0, and the constraint layer would have nothing left to detect. There is a test asserting
that doubling the budget leaves both estimated lines unchanged.
