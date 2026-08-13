"""The anti-hallucination tripwire.

The rule this module enforces is deliberately narrow and mechanical: **the model may only
restate figures and entities it was given**. The allowed set is not curated by hand — it is
derived from the exact payload handed to the model, so the check and the prompt cannot drift
apart. If the payload says the trip costs €2,426.52, the model may write €2,426.52, "about
€2,427", or nothing at all. It may not write €2,400 as a price, and it may not name a hotel
that is not in the plan.

Four things are checked, all hard:

1. **Entities.** The model returns the ids it is talking about; every one must exist in the
   payload. Asking for ids rather than parsing names out of prose keeps this exact instead
   of a fuzzy string-matching problem.
2. **Money.** Every currency figure in the prose must match an amount from the payload,
   within a small tolerance for rounded phrasing.
3. **Currency.** A figure written in dollars or yen is wrong even if the number is right.
4. **Per-night claims.** A figure the model calls a nightly rate must be one. Checks 3 and 4
   exist because `llama3.1:8b` made exactly those two errors on the first real run: it wrote
   "a budget of $2500", and it called the €1,201.06 total for a seven-night stay a nightly
   rate. The number was in the payload, so a pure existence check waved it through.

**What this cannot catch.** Check 4 is a targeted fix for one observed misattribution, not a
general solution. The model also wrote "8.9 out of 5 stars", conflating a 0-10 guest rating
with a star classification — both values were in the payload and neither is money. Verifying
that the *claim attached to* a figure is true, rather than that the figure exists, is a
claim-level entailment problem: it needs an NLI model or a second structured pass, and this
project does not pretend to have solved it. `docs/llm.md` says so plainly.

Failure discards the text. It is never repaired and shown — a "fixed" explanation is an
explanation nobody validated.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

ID_KEYS = frozenset({"id", "activity_id", "accommodation_id"})

MONEY_KEY_HINTS = ("cost", "price", "budget", "total", "remaining", "allowance", "amount")
"""Only values under a monetary key count as quotable amounts.

Collecting *every* number in the payload would silently defang the check: day indices and a
party size of 2 would put 1-7 into the allowed set, and with the rounding tolerance any small
figure would then pass. Restricting to monetary keys keeps the question sharp — did the model
invent a price?"""

PER_NIGHT_KEY_HINT = "per_night"

NUMBER = r"\d(?:[\d.,]*\d)?"
"""A figure must *end* in a digit.

The obvious `\\d[\\d.,]*` is greedy enough to swallow sentence punctuation: in
"EUR 1201.06, activities EUR 427.46" it captures "1201.06," and the trailing comma is then
read as a decimal separator, turning €1,201.06 into €120,106. That produced false rejections
of a perfectly honest explanation — the check accusing the model of the parser's mistake.
"""

MONEY_SUFFIX = re.compile(rf"({NUMBER})\s*(?:€|EUR\b|euros?\b)", re.IGNORECASE)
MONEY_PREFIX = re.compile(rf"(?:€|EUR)\s*({NUMBER})", re.IGNORECASE)

FOREIGN_CURRENCY = re.compile(
    rf"(?:[$£¥₩]\s*{NUMBER})|(?:{NUMBER}\s*(?:USD|GBP|JPY|CHF|dollars?|pounds?|yen)\b)",
    re.IGNORECASE,
)
"""A right number in the wrong currency is still a wrong price."""

PER_NIGHT_QUALIFIER = re.compile(
    r"^\W{0,4}(?:per|a|each|/)\s*-?\s*night|^\W{0,4}nightly", re.IGNORECASE
)
"""Matches immediately after a figure: "€171.58 per night", "€171.58/night", "€171.58 a night"."""

ABSOLUTE_TOLERANCE = 1.0
RELATIVE_TOLERANCE = 0.005
"""A figure matches if it is within €1 or 0.5 % of an allowed amount, whichever is larger.

Tight enough that €2,400 fails against €2,426.52 — an invented round number is exactly the
kind of plausible-sounding error worth catching — and loose enough that "about €2,427" and
"€1,201" for €1,201.06 both pass, so natural phrasing is not punished.
"""


@dataclass(frozen=True)
class MoneyClaim:
    """A currency figure in the prose, together with what the model called it."""

    value: float
    per_night: bool


@dataclass(frozen=True)
class GroundingContext:
    """Everything the model is allowed to refer to."""

    entity_ids: frozenset[str]
    amounts: tuple[float, ...]
    per_night_amounts: tuple[float, ...]
    """The subset that genuinely is a nightly rate."""

    def permits_amount(self, value: float) -> bool:
        return _matches(value, self.amounts)

    def permits_per_night(self, value: float) -> bool:
        return _matches(value, self.per_night_amounts)


def _matches(value: float, allowed_values: tuple[float, ...]) -> bool:
    return any(
        abs(value - allowed) <= max(ABSOLUTE_TOLERANCE, abs(allowed) * RELATIVE_TOLERANCE)
        for allowed in allowed_values
    )


def build_context(payload: Mapping[str, object]) -> GroundingContext:
    """Derive the allowed set from the payload the model will actually be given.

    Deriving rather than declaring is the point: a hand-maintained allow-list would drift
    away from the prompt on the first change, and the check would quietly stop checking.
    """
    ids: set[str] = set()
    amounts: set[float] = set()
    per_night: set[float] = set()
    _walk(payload, ids, amounts, per_night)
    return GroundingContext(
        entity_ids=frozenset(ids),
        amounts=tuple(sorted(amounts)),
        per_night_amounts=tuple(sorted(per_night)),
    )


def _walk(
    node: object,
    ids: set[str],
    amounts: set[float],
    per_night: set[float],
    key: str | None = None,
) -> None:
    if isinstance(node, Mapping):
        for child_key, child in node.items():
            _walk(child, ids, amounts, per_night, str(child_key))
    elif isinstance(node, str):
        if key in ID_KEYS:
            ids.add(node)
    elif isinstance(node, bool):
        return  # bool is an int subclass; it is never a quotable figure
    elif isinstance(node, int | float):
        if key and any(hint in key for hint in MONEY_KEY_HINTS):
            amounts.add(float(node))
            if PER_NIGHT_KEY_HINT in key:
                per_night.add(float(node))
    elif isinstance(node, Sequence):
        for child in node:
            _walk(child, ids, amounts, per_night, key)


def parse_money(token: str) -> float | None:
    """Read a currency figure written in either European or Anglo convention.

    `1.201,06` and `1,201.06` are the same amount. When both separators appear the rightmost
    is the decimal point. When only one appears, three trailing digits mean a thousands
    separator (`1,201`) and anything else means a decimal (`45,50`).
    """
    cleaned = token.strip().replace(" ", "")
    if not cleaned:
        return None

    has_dot = "." in cleaned
    has_comma = "," in cleaned
    if has_dot and has_comma:
        decimal_sep = "." if cleaned.rfind(".") > cleaned.rfind(",") else ","
        thousands_sep = "," if decimal_sep == "." else "."
        cleaned = cleaned.replace(thousands_sep, "").replace(decimal_sep, ".")
    elif has_dot or has_comma:
        separator = "." if has_dot else ","
        tail = cleaned.rsplit(separator, 1)[1]
        cleaned = (
            cleaned.replace(separator, "") if len(tail) == 3 else cleaned.replace(separator, ".")
        )

    try:
        return float(cleaned)
    except ValueError:
        return None


def money_claims(text: str) -> tuple[MoneyClaim, ...]:
    """Every currency amount in a piece of prose, and whether it is called a nightly rate."""
    found: list[MoneyClaim] = []
    for pattern in (MONEY_SUFFIX, MONEY_PREFIX):
        for match in pattern.finditer(text):
            value = parse_money(match.group(1))
            if value is None:
                continue
            trailing = text[match.end() : match.end() + 16]
            found.append(
                MoneyClaim(value=value, per_night=bool(PER_NIGHT_QUALIFIER.match(trailing)))
            )
    return tuple(found)


def money_figures(text: str) -> tuple[float, ...]:
    """Just the amounts, for callers that do not care how they were qualified."""
    return tuple(claim.value for claim in money_claims(text))


def check_grounding(
    text: str,
    referenced_ids: Iterable[str],
    context: GroundingContext,
) -> tuple[str, ...]:
    """Return every way this text departs from what the model was given. Empty means clean."""
    violations = [
        f"references an entity that is not in the plan: {identifier!r}"
        for identifier in referenced_ids
        if identifier not in context.entity_ids
    ]

    for claim in money_claims(text):
        if not context.permits_amount(claim.value):
            violations.append(f"cites a figure that is not in the plan: {claim.value:.2f}")
        elif claim.per_night and not context.permits_per_night(claim.value):
            violations.append(
                f"calls {claim.value:.2f} a nightly rate, but it is not one in the plan"
            )

    violations.extend(
        f"quotes a price in the wrong currency: {found.strip()!r}"
        for found in FOREIGN_CURRENCY.findall(text) or []
    )
    return tuple(violations)
