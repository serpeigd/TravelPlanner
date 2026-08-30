# CLAUDE.md — TravelPlanner (Travel Intelligence)

Project memory for Claude Code.

A trip-recommendation system built to be **defended, not demoed**: it retrieves
real accommodation and activity candidates, ranks them with an explainable
score, enforces hard constraints in deterministic code, and only then lets an
LLM describe the result.

## Conventions

- **All repo content is English** — code, comments, docs, commit messages. Chat
  with Sergio stays in Spanish.
- **Python 3.12.** `pyproject.toml`, CI, `ruff`'s `target-version` and `mypy`'s
  `python_version` all agree on it. (`.devcontainer/` does not — see below.)
- `mypy --strict` and `ruff` are clean and expected to stay that way; CI runs
  lint, format check, types and tests on every push and PR.
- The docs argue the project rather than describing it. `docs/decisions.md`
  records each ADR **with the alternative that was rejected** — keep that shape.
- **Keep it short (2026-08-13, explicit request in chat).** Too much text, too much
  explaining. Lead with the answer or the change; give reasoning only where it would
  change a decision. Don't recap work already visible in the diff, don't restate the
  question before answering it, and don't close with a summing-up line. Applies to
  chat, commit messages and PR bodies. Reference docs (README, this file) can be
  longer, but only where the length earns it.

## The three principles, and what they actually forbid

> An LLM is a component of the system, not the system.
> Business logic belongs in deterministic, testable code wherever possible.
> Evaluation must measure whether the system works, not whether the demo looks impressive.

Concretely, in this codebase:

- **The constraint barrier is load-bearing.** By the time any text is generated,
  a plan that breaks a hard constraint has already been discarded. A budget is
  not a soft preference to trade off — don't let scoring absorb it.
- **No agent framework, deliberately.** The pipeline is a linear chain of pure
  functions; control flow is fixed and knowable, so a model-driven loop would
  add non-determinism and latency for nothing. Argue from a concrete need before
  reaching for one.
- **Uncomputable factors are dropped, not zeroed.** Imputing 0 would say "we
  don't know" and "it's terrible" with the same number. Missing stays missing
  through the whole pipeline.
- **Ratings are shrunk toward the market mean by review volume.** A 9.2 from 136
  reviews and an 8.2 from 23,178 are not the same claim.
- **Relevance labels don't exist and must not be fabricated.** Inventing them
  trains a model to recover the heuristic that invented them. The one real
  target in this data is price — hence the hedonic model, and nothing more.

## Resolved — `zz_HOWTOLOCAL.txt` leak (was flagged three times, fixed 2026-08-30)

`zz_HOWTOLOCAL.txt` at the repo root contained a real local Windows path and
username since `608c530`. Three documentation-sync passes flagged it without
fixing it (deleting/editing content was outside that recurring task's scope);
the weekly engineering-health review deleted the file directly and removed
the matching README Limitations entry. History still contains the path in
old commits — a non-issue for a public repo with no secret in it, just a
stale username, not worth a history rewrite.

## Known mismatch — the devcontainer can't build

⚠️ `.devcontainer/devcontainer.json` pins `python:1-3.11-bookworm`, while
`pyproject.toml` declares `requires-python = ">=3.12"`. The container's
`updateContentCommand` installs `requirements.txt`, which resolves to `.[ui]`,
so **a Codespace built from that file fails at install.**

Not fixed on purpose: bumping the image is the easy direction, but lowering
`requires-python` instead would mean re-auditing every 3.12-only construct in
`src/`. That's Sergio's call. Documented in the README's Limitations section
rather than left for someone to discover.

## Two traps for anyone editing the docs

- **Don't "correct" the test count.** The README says 245 tests. Counting `def
  test_` gives 213 — the difference is parametrized cases, and `pytest` is not
  installed in the scheduled-run environment to confirm the real number.
  Changing it on a guess replaces a probably-correct figure with a definitely
  wrong one. Same rule for any number you can't actually run.
- **The fixture path moved.** It's `src/travel_intel/data/fixtures/`, not
  `data/fixtures/` — the snapshot was shipped inside the package so it survives
  a non-editable install. Older docs and any stale branch still say the old path.

## Legal note that matters here

The Booking.com fixture is committed so the pipeline is reproducible and its
numbers auditable — **not** as a redistributable dataset, and not under the MIT
grant. Its prices and ratings were true at one instant in 2026 and are stale by
construction. Nothing this project outputs is a booking service, a price quote,
or travel advice. Keep that distinction in the README.

## Scheduled documentation-sync runs

- **Standing authorization to merge doc-only PRs from that task (2026-08-14,
  explicit decision in chat), once CI is green.** Same bar as any other merge,
  just no confirmation step, and only for this narrow case: README/`docs/`
  changes, never product code. If a check fails, don't merge — report why.
- **This repo moves while you work on it.** That same run hit a merge conflict
  because `main` gained two commits mid-session. Fetch and rebase or merge
  before assuming a docs branch still applies cleanly.

## Key files

| Area | File |
|---|---|
| Ranking formula and factor weights | `src/travel_intel/ranking/scoring.py` |
| Hedonic price model | `src/travel_intel/ml/price_model.py` |
| Hard-constraint enforcement (the barrier) | `src/travel_intel/constraints.py` |
| LLM interpretation + grounded explanation | `src/travel_intel/llm/` |
| Grounding check | `src/travel_intel/llm/grounding.py` |
| Evaluation harness and golden set | `src/travel_intel/evaluation/` |
| End-to-end demo | `src/travel_intel/demo.py` |
| Frozen Booking.com snapshot | `src/travel_intel/data/fixtures/` |
| ADRs, each with its rejected alternative | `docs/decisions.md` |
