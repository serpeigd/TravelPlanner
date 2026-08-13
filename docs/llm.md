# The LLM layer

## Two jobs, and the boundary around them

The model does exactly two things:

1. **Interpret free text.** "We love street food and quiet old temples, and we want to avoid
   crowds" is not something a controlled vocabulary matches by substring. This is what
   language models are genuinely good at.
2. **Explain a finished decision.** By the time the explainer runs, the accommodation is
   chosen, the itinerary is fixed, the budget is arithmetic and the constraints have passed.
   The model is turning a table into sentences. Nothing it writes can change any of it.

It does **not** rank, compute a budget, validate a constraint, or decide which hotel wins.
Those all have deterministic implementations upstream, and the pipeline order is what makes
that a structural guarantee rather than a promise: `validate_plan` runs *before* the model is
called, so a budget-violating plan has already been discarded.

Every output is validated against a Pydantic schema, checked for grounding, and backed by a
deterministic fallback. Running with `TRAVEL_INTEL_LLM_PROVIDER=fake` — the default in tests
and CI — uses the fallbacks throughout, so the suite exercises the real code paths with no
model server and no network.

## Setup

`llama3.1:8b` on local Ollama. Free, offline, runs on a laptop during an interview.

```bash
ollama pull llama3.1:8b
TRAVEL_INTEL_LLM_PROVIDER=ollama python -m travel_intel.demo
```

Ollama's `format: "json"` constrains decoding to valid JSON, which removes the most common
failure mode — prose wrapped around the object — without a parsing heuristic.

**Measured latency: ~100–115 s** for the explanation on CPU. The 60 s timeout this started
with expired on *every* real run and silently exercised the fallback, which looked like a
working system until the logs were read. The default is now 180 s.

## The grounding check

The rule is deliberately narrow and mechanical: **the model may only restate figures and
entities it was given.** The allowed set is not hand-curated — it is derived by walking the
exact JSON payload handed to the model, so the check and the prompt cannot drift apart. A
hand-maintained allow-list would fall out of date on the first change and quietly stop
checking.

Four hard checks:

| Check | What it catches |
|---|---|
| **Entities** | The model returns the ids it discusses; each must exist in the payload |
| **Money** | Every currency figure must match a payload amount (±€1 or ±0.5 %) |
| **Currency** | A right number in dollars or yen is still a wrong price |
| **Per-night** | A figure the model calls a nightly rate must be one |

Only values under monetary keys count as quotable amounts. Collecting *every* number would
have defanged the check: a day index of 7 and a party size of 2 would put small integers into
the allowed set, and with the rounding tolerance any small figure would then pass.

Failure discards the text and the template answers instead, with the reasons recorded on the
`Explanation`. Nothing is repaired and shown — fixing a hallucinated price still leaves prose
built around it, and a silent fallback is indistinguishable from a system that never tried.

## What the model actually did

Checks 3 and 4 exist because of the first real run against `llama3.1:8b`, not because they
seemed prudent. Unprompted, the model produced:

| Output | Problem |
|---|---|
| "a budget of **$2500**" | Right number, wrong currency |
| "**8.9 out of 5 stars**" | Conflated a 0-10 guest rating with a 5-star classification |
| "accommodation is **$1201.06 per night**" | That is the total for seven nights |

The third is the instructive one. The figure was in the payload, so an existence check waved
it through. Tightening the prompt (state the currency, keep each figure's meaning, keep
rating and stars apart) fixed all three; the per-night and currency checks make two of them
mechanically detectable if they come back.

## What this cannot catch

**Claim-level misattribution in general.** The per-night check is a targeted fix for one
observed error, not a solution to the class. "8.9 out of 5 stars" uses two real values from
the payload and involves no money at all — no amount of figure-matching sees it. Verifying
that the *claim attached to* a figure is true, rather than that the figure exists, is an
entailment problem: it needs an NLI model or a second structured pass where the explainer
emits assertions that code can check one by one. This project does not pretend to have solved
it, and the limitation is stated here rather than discovered by an interviewer.

## A bug the tests did not find

The first hardened run rejected a perfectly honest explanation, reporting €120,106 where the
model had written €1,201.06.

The model was right; the parser was wrong. `\d[\d.,]*` is greedy enough to swallow sentence
punctuation: in `"EUR 1201.06, activities EUR 427.46"` it captured `"1201.06,"`, and the
trailing comma was then read as a decimal separator. Requiring a figure to *end* in a digit
fixes it. There is now a regression test for trailing commas, periods, semicolons and
parentheses.

Worth stating plainly: a grounding check that fires on honest output is worse than no check,
because it trains everyone to ignore it. The failure was only visible by running the real
model and reading what came back.

## Degradation is graceful, never silent

| Situation | Behaviour |
|---|---|
| Model unreachable or times out | Deterministic fallback answers; the reason is recorded |
| Malformed JSON | One retry, then fallback |
| Term outside the vocabulary | Surfaced in `unmapped`, not dropped |
| Grounding violation | Text discarded, reasons kept on the `Explanation` |

The `Explanation` always carries its `provenance` — `model_generated` or `synthetic` — so a
response written without the model says so, and a UI can label it.

The keyword interpreter has an honest limitation of its own: substring matching cannot report
what it failed to understand, so its `unmapped` is always empty. That gap is precisely what
the LLM fills.
