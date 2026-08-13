# Architecture

## The shape

A linear pipeline of pure functions. No agent framework, no orchestration library — and that
is an argued position, not an omission ([ADR-001](decisions.md)): the control flow is fixed
and knowable, so a model-driven loop would add non-determinism and latency to a problem that
does not need either.

```
TripRequest  (Pydantic, immutable, validated at the boundary)
      │
      ├─► retrieval ────────── Protocol + two implementations. Snapshot is default and offline.
      │
      ├─► candidates ───────── Only unarguable filters. Every drop counted by reason.
      │
      ├─► features ─────────── One pandas table. Two consumers: the ranking and the model.
      │
      ├─► ml ───────────────── Hedonic price regression. The residual, not the prediction.
      │
      ├─► ranking ──────────── Weighted average of six [0,1] factors. Weights travel with it.
      │
      ├─► itinerary ────────── Coverage first, then greedy by score-per-euro.
      │
      ├─► budget ───────────── Four lines: two retrieved, two declared as estimates.
      │
      ├─► constraints ══════╗  THE BARRIER. Hard violations are never returned.
      │                     ║  The planner walks the ranking until a plan passes.
      ├─► llm ──────────────╨  Runs *after* the barrier. Interprets and explains. Never decides.
      │
      └─► TripRecommendation  (one validated response object)
```

The barrier is the load-bearing part. By the time any text is generated, a plan that breaks a
hard constraint has already been discarded — so "the model cannot overspend your budget" is a
property of the ordering, not a hope about prompting.

## Stage by stage

| Stage | Module | Deterministic? | Why it is where it is |
|---|---|---|---|
| Request validation | `domain/models.py` | yes | Dates, capacity and budget checked at the boundary. The domain owns no clock, so fixtures stay reproducible forever. |
| Retrieval | `retrieval/` | yes | One `Protocol`, two implementations. Every record carries a `Provenance`. |
| Candidate generation | `ranking/candidates.py` | yes | Only facts disqualify here — cannot host the party, costs more than the entire budget. Judgement belongs in scoring. |
| Feature engineering | `features/` | yes | One table; `MODEL_FEATURES` marks the request-independent subset so the query cannot leak into the model. |
| Price model | `ml/` | yes (seeded) | Fitted on the candidate set; the residual becomes `value_for_money`. |
| Ranking | `ranking/scoring.py` | yes | Six factors, effective weights carried on every result. |
| Itinerary | `planning/itinerary.py` | yes | Where `culture` and `nature` are actually served. |
| Budget | `planning/costs.py` | yes | Plain arithmetic. No model computes a number that reaches a budget. |
| **Constraints** | `constraints.py` | yes | Hard vs soft. Hard means the plan is never returned. |
| Planner walk | `services/planner.py` | yes | First option whose *complete plan* passes. Refusals kept for explanation. |
| Interpretation | `llm/interpret.py` | no | Free text → controlled vocabulary, schema-validated, with a keyword fallback. |
| Explanation | `llm/explain.py` | no | Prose over decisions already made, discarded if it fails the grounding check. |
| Assembly | `recommend.py` | yes | Builds `TripRecommendation` and the per-request quality signals. |
| HTTP | `api/app.py` | yes | ~120 lines: the domain contracts *are* the API contract. |

## Data flow and provenance

Every fact that reaches a user carries where it came from:

| `Provenance` | Meaning | Example |
|---|---|---|
| `snapshot` | Real provider data, frozen as a versioned fixture | hotel prices, ratings, activities |
| `real_api` | Retrieved live during the request | (reserved; live mode is an unimplemented seam) |
| `synthetic` | Our own estimate, never presented as a retrieved fact | food €45/person/day, transport €12 |
| `model_generated` | Written by the LLM, grounded in the records above | the explanation |

This is a field on the model, not a note in a README, which is why the API response and the
UI can both separate retrieved prices from planning assumptions without any extra plumbing.

## Failure modes

Nothing degrades silently. Every fallback is recorded on the object that carries it.

| Failure | Behaviour |
|---|---|
| Unknown destination | `NoCandidatesError` → HTTP 422 naming the destinations that do exist |
| Budget cannot be met | `NoCandidatesError` → HTTP 422 with the dominant violation and the count refused |
| Request diverges from the snapshot | Soft `stale_data` violations, plan still returned |
| Live provider requested | `ProviderError` → HTTP 503. Never a silent fall back to the snapshot |
| Model unreachable / times out | Deterministic template answers; reason recorded on the `Explanation` |
| Model returns malformed JSON | One retry, then fallback |
| Model output fails grounding | Text discarded, violations recorded, template answers |

## Scaling: prototype → service → distributed

Nothing below is implemented. The point is that the seams already exist, not that the system
is secretly production-grade.

**Prototype (today).** One process, a frozen snapshot, a model on localhost. Everything
seeded and reproducible.

**Service.** Swap `SnapshotProvider` for an HTTP client behind the same `Protocol` — retries
with backoff, a circuit breaker, and a response cache keyed by `(destination, dates, party)`.
Nothing downstream changes; that is what the Protocol is for. The price model stops being
fitted per request and becomes an artefact trained offline, versioned in a registry and
loaded at startup. The explanation moves behind a queue, because a two-minute synchronous
call is not an HTTP request.

**Distributed.** Provider snapshots land in object storage partitioned by
`destination/capture_date`, and feature building becomes a Spark job over that history rather
than a pandas frame over thirty rows. The ranking stays per-request and in-process — it is
milliseconds of arithmetic over a few dozen candidates, and distributing it would add latency
to buy nothing.

**Real-time.** The parts that genuinely need it are price freshness and availability, not the
ranking. That argues for a caching and invalidation strategy at the retrieval layer, plus
event-driven refresh for high-demand destinations — not for rewriting the pipeline.

## Azure → AWS

Nothing in this system is cloud-specific. This mapping exists so the same architecture can be
described in either vocabulary, and to make the point that the provider is the least
interesting decision in the table below.

| Concern | What I have used (Azure) | AWS equivalent | Where it fits here |
|---|---|---|---|
| Distributed processing | Azure Databricks | EMR, or Glue for ETL | Feature building over a price history |
| Object storage | Azure Blob Storage / ADLS Gen2 | S3 | Partitioned provider snapshots |
| ML platform | Azure ML | SageMaker | Training, model registry, endpoints for the price model |
| Experiment tracking | Azure ML / MLflow | SageMaker Experiments, or MLflow on either | Versioning the hedonic model and its CV metrics |
| Orchestration | Azure Data Factory | Glue workflows, Step Functions, or MWAA | Nightly capture → validate → retrain → publish |
| Containers | Azure Container Apps / AKS | ECS Fargate, EKS | The FastAPI service |
| Serverless | Azure Functions | Lambda | Snapshot capture and cache invalidation |
| Managed relational | Azure SQL / PostgreSQL | RDS, Aurora | Request logs, plan history |
| Secrets | Azure Key Vault | Secrets Manager, Parameter Store | Provider API credentials |
| Monitoring | Azure Monitor / App Insights | CloudWatch | Latency, error rate, refusal rate |
| Data quality | Great Expectations on Databricks | Deequ, or Glue Data Quality | The checks `docs/data.md` describes doing by hand |

What actually transfers is not the service names. It is the design: separate compute from
storage, partition by date, version artefacts rather than mutating them, and keep the
inference path stateless.

## What would change in production

1. **The price model would be trained offline** on a real price history, versioned, and
   loaded — not fitted per request. Per-request fitting is right for a hedonic index over
   thirty candidates and wrong for anything larger.
2. **Temporal validation.** With a time axis, k-fold leaks the future into the past. Splits
   would be chronological, and the model would be re-validated on each capture.
3. **The explanation would be asynchronous.** Two minutes is a job, not a request.
4. **Weights would be re-derived, not chosen.** With booking data, learning-to-rank stops
   being fabrication and becomes the right tool — the objection in [ADR-003](decisions.md) is
   about *this* dataset, not about the technique.
5. **Monitoring would watch factor distributions, not just outputs.** Because the ranking is
   deterministic, a shift in `budget_fit` or `value_for_money` distributions is a clean drift
   signal — something a black-box ranker cannot offer.
6. **Refusal rate becomes a product metric.** A rising share of `NoCandidatesError` means the
   catalogue no longer covers what people ask for, which is a business signal before it is a
   technical one.

## Monitoring and drift

| Signal | Why it matters |
|---|---|
| Refusal rate by reason | Distinguishes "budgets too low" from "snapshot too stale" |
| Factor distributions | Deterministic scoring makes drift legible per factor |
| Price-model residual spread | Widening residuals mean the market moved and the model has not |
| Grounding rejection rate | A rising rate means the prompt, the model or the payload changed |
| Data completeness | A provider quietly dropping a field shows up here first |
| Snapshot age | Prices have a shelf life; the system already warns per request |
