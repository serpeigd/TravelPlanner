"""Free text to structured preferences.

This is the LLM's first legitimate job. "We love street food and quiet old temples, and we
hate crowds" is not something a controlled vocabulary matches by substring, and it is exactly
what language models are good at.

What the model produces is never trusted as-is. It comes back as JSON, is validated against
a schema, and every term is mapped onto the `Preference` enum — anything outside the
vocabulary is *reported* as unmapped rather than silently discarded, because "we could not
express what you asked for" is information the user should have.
"""

from __future__ import annotations

import json
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from travel_intel.domain.enums import Preference, Provenance
from travel_intel.domain.errors import LLMError
from travel_intel.llm.client import LLMClient

MAX_ATTEMPTS = 2
"""One retry. A model that returns malformed JSON twice is not going to on the third try,
and the fallback interpreter is right there."""

SYSTEM_PROMPT = """You classify a traveller's free-text wishes into a fixed vocabulary.

Reply with JSON only, in this exact shape:
{"preferences": ["food", "culture"], "unmapped": ["onsen"]}

Rules:
- "preferences" may only contain these values: food, culture, nature, history, art,
  nightlife, shopping, relaxation.
- Put a value in "preferences" only if the traveller clearly wants that kind of thing.
- If the traveller mentions something real that none of those values expresses, put a short
  phrase for it in "unmapped". Do not force it into a preference.
- Do not invent preferences that the traveller did not express.
- No prose, no explanation, no markdown. JSON only."""


KEYWORDS: dict[Preference, tuple[str, ...]] = {
    Preference.FOOD: ("food", "eat", "restaurant", "cuisine", "comida", "gastronom", "comer"),
    Preference.CULTURE: ("culture", "museum", "temple", "shrine", "cultura", "museo", "templo"),
    Preference.NATURE: ("nature", "park", "hike", "outdoor", "naturaleza", "parque", "montaña"),
    Preference.HISTORY: ("history", "historic", "heritage", "historia", "histórico"),
    Preference.ART: ("art", "gallery", "arte", "galería"),
    Preference.NIGHTLIFE: ("nightlife", "bar", "club", "noche", "fiesta"),
    Preference.SHOPPING: ("shopping", "shop", "market", "compras", "tienda", "mercado"),
    Preference.RELAXATION: ("relax", "spa", "onsen", "wellness", "descans", "tranquil"),
}
"""Substrings for the deterministic fallback, in English and Spanish.

Deliberately crude. Its job is to keep the pipeline running when the model is unavailable,
not to compete with it.
"""


class PreferenceInterpretation(BaseModel):
    """What the traveller asked for, expressed in the system's vocabulary."""

    model_config = ConfigDict(frozen=True)

    preferences: tuple[Preference, ...]
    unmapped: tuple[str, ...] = ()
    """Real wishes the vocabulary cannot express. Surfaced, not swallowed."""
    interpreter: str
    provenance: Provenance


class _Payload(BaseModel):
    """The shape the model must produce. Anything else is a failed attempt."""

    model_config = ConfigDict(extra="ignore")

    preferences: list[str] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)


class PreferenceInterpreter(Protocol):
    name: str

    def interpret(self, notes: str) -> PreferenceInterpretation: ...


class KeywordPreferenceInterpreter:
    """Substring matching over the controlled vocabulary.

    The deterministic fallback, and the default in tests and CI so the suite never depends on
    a model server. Its honest limitation: it cannot report what it failed to understand, so
    `unmapped` is always empty. A keyword matcher does not know what it missed — which is
    precisely the gap the LLM fills.
    """

    name = "keyword"

    def interpret(self, notes: str) -> PreferenceInterpretation:
        lowered = notes.lower()
        matched = tuple(
            preference
            for preference, needles in KEYWORDS.items()
            if any(needle in lowered for needle in needles)
        )
        return PreferenceInterpretation(
            preferences=matched,
            unmapped=(),
            interpreter=self.name,
            provenance=Provenance.SYNTHETIC,
        )


class LLMPreferenceInterpreter:
    """Model-backed interpretation with schema validation and a deterministic fallback.

    Degradation is graceful but never silent: the result records which interpreter produced
    it, so a response built without the model says so.
    """

    def __init__(self, client: LLMClient, fallback: PreferenceInterpreter | None = None) -> None:
        self.name = client.name
        self._client = client
        self._fallback = fallback or KeywordPreferenceInterpreter()

    def interpret(self, notes: str) -> PreferenceInterpretation:
        if not notes.strip():
            return PreferenceInterpretation(
                preferences=(),
                unmapped=(),
                interpreter=self.name,
                provenance=Provenance.MODEL_GENERATED,
            )

        last_error: Exception | None = None
        for _ in range(MAX_ATTEMPTS):
            try:
                raw = self._client.complete_json(SYSTEM_PROMPT, notes)
                payload = _Payload.model_validate(json.loads(raw))
            except (LLMError, json.JSONDecodeError, ValidationError) as error:
                last_error = error
                continue
            return self._to_interpretation(payload)

        # Every attempt failed. Answering with a cruder interpretation beats not answering,
        # as long as the response admits which one it used.
        degraded = self._fallback.interpret(notes)
        return degraded.model_copy(
            update={"interpreter": f"{self._fallback.name} (fallback after {last_error})"}
        )

    @staticmethod
    def _to_interpretation(payload: _Payload) -> PreferenceInterpretation:
        """Map model terms onto the enum, moving anything unrecognised to `unmapped`.

        A term the model invented is not an error worth failing the request over, but it is
        also not something to quietly drop: it lands in `unmapped` where the user sees it.
        """
        recognised: list[Preference] = []
        unmapped: list[str] = list(payload.unmapped)
        for term in payload.preferences:
            try:
                recognised.append(Preference(term.strip().lower()))
            except ValueError:
                unmapped.append(term)
        return PreferenceInterpretation(
            preferences=tuple(dict.fromkeys(recognised)),
            unmapped=tuple(dict.fromkeys(unmapped)),
            interpreter="llm",
            provenance=Provenance.MODEL_GENERATED,
        )
