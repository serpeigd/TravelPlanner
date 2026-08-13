# Travel Intelligence

A small trip-recommendation system built to be **defended, not demoed**: given a
destination, dates, party size, budget and preferences, it retrieves accommodation and
activity candidates, ranks them with an explainable score, enforces hard constraints in
code, and explains the result with a local LLM that is never allowed to invent a fact.

Three principles drive every design decision here:

- **An LLM is a component of the system, not the system.**
- **Business logic belongs in deterministic, testable code** wherever that is possible.
- **Evaluation must measure whether the system works**, not whether the demo looks
  impressive.

> ⚠️ Work in progress. The status table below is the source of truth — nothing is claimed
> here that is not implemented and tested.

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | Scaffolding: packaging, ruff, mypy (strict), pytest, CI | ✅ done |
| M1 | Domain contracts (Pydantic), provenance model, config | ✅ done |
| M2 | Retrieval adapters + real Tokyo snapshot (30 hotels, 27 activities) | ✅ done |
| M3 | Budget policy, feature engineering, explainable ranking | ✅ done |
| M4 | Itinerary, budget composition, hard-constraint validation | ✅ done |
| M5 | Hedonic price model → `value_for_money` factor | ✅ done |
| M6 | LLM: preference interpretation + grounded explanation | ✅ done |
| M7 | Reproducible evaluation harness | ⬜ next |
| M8 | FastAPI endpoint, Streamlit UI, full docs, Azure→AWS map | ⬜ |

## Quickstart

Requires Python 3.12+.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Then run the whole pipeline on the reference request:

```bash
python -m travel_intel.demo
```

Retrieval, ranking, the price model, the itinerary and the constraint check are offline and
deterministic — they read a frozen snapshot of real Booking.com data. Only the explanation
calls a model, and only if you ask it to: `TRAVEL_INTEL_LLM_PROVIDER=fake` uses the
deterministic template and is the default in CI. See `.env.example` for the rest.

## Documentation

- [`docs/decisions.md`](docs/decisions.md) — architecture decision records, including why
  there is no agent framework and why the ML component is a price model rather than
  learning-to-rank.
- [`docs/data.md`](docs/data.md) — what the dataset is, what was decided at capture time,
  and what the data honestly cannot support.
- [`docs/ranking.md`](docs/ranking.md) — the scoring formula, why each weight is what it is,
  and why missing factors are dropped rather than scored as zero.
- [`docs/ml.md`](docs/ml.md) — the hedonic price model, its cross-validated results against
  two baselines, and what the data cannot support.
- [`docs/llm.md`](docs/llm.md) — the LLM's two jobs, the grounding check, what the model got
  wrong on the first real run, and what the check still cannot catch.
- `docs/architecture.md` — pipeline, data flow, scalability, Azure→AWS mapping *(M8)*.
- `docs/evaluation.md` — metrics, golden set, results *(M7)*.
