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

**Decision.** One `AccommodationProvider` / `ActivityProvider` interface with two
implementations. `snapshot` reads real provider data captured once and committed as a
versioned fixture; `live` calls the provider. `snapshot` is the default and the only mode
used by tests and the evaluation harness.

**Consequences.** The evaluation harness is deterministic and runs offline in CI. Every
record carries a `Provenance` value, so retrieved facts, frozen snapshots, synthetic rows
and model prose are never confused with each other in the UI or in the answer.

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
portfolio demo that must run on a laptop during an interview.
