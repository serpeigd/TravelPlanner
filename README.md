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
| M2 | Retrieval adapters: frozen snapshot + live mode | ⬜ next |
| M3 | Feature engineering + explainable ranking | ⬜ |
| M4 | Hard-constraint validation + budget breakdown | ⬜ |
| M5 | ML price model → `value_for_money` factor | ⬜ |
| M6 | LLM: preference normalisation + grounded explanation | ⬜ |
| M7 | Reproducible evaluation harness | ⬜ |
| M8 | FastAPI endpoint, Streamlit UI, full docs, Azure→AWS map | ⬜ |

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

Nothing needs network access or an API key: the default configuration reads frozen
snapshot data and a deterministic fake LLM client. See `.env.example` to switch to live
data or to the local Ollama model.

## Documentation

- [`docs/decisions.md`](docs/decisions.md) — architecture decision records, including why
  there is no agent framework and why the ML component is a price model rather than
  learning-to-rank.
- `docs/architecture.md` — pipeline, data flow, scalability, Azure→AWS mapping *(M8)*.
- `docs/evaluation.md` — metrics, golden set, results *(M7)*.
