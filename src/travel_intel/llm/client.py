"""Transport to a language model.

The client's only responsibility is getting a JSON string back. It knows nothing about
travel, preferences or explanations — parsing, validation and grounding all happen a layer
up, which is what keeps those rules testable without a model server.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from travel_intel.domain.errors import LLMError


class LLMClient(Protocol):
    """Anything that can answer a prompt with a JSON document."""

    name: str

    def complete_json(self, system: str, prompt: str) -> str: ...


class OllamaClient:
    """Local model over Ollama's HTTP API.

    `format: "json"` constrains decoding to valid JSON, which removes the most common
    failure mode (prose wrapped around the object) without a parsing heuristic. Temperature
    is low by default: neither of this system's two LLM jobs benefits from creativity.
    """

    def __init__(
        self,
        model: str,
        host: str,
        *,
        timeout_s: float = 60.0,
        temperature: float = 0.2,
    ) -> None:
        self.name = f"ollama/{model}"
        self._model = model
        self._host = host.rstrip("/")
        self._timeout_s = timeout_s
        self._temperature = temperature

    def complete_json(self, system: str, prompt: str) -> str:
        payload = {
            "model": self._model,
            "system": system,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": self._temperature},
        }
        try:
            response = httpx.post(
                f"{self._host}/api/generate", json=payload, timeout=self._timeout_s
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as error:
            raise LLMError(f"{self.name} unreachable at {self._host}: {error}") from error
        except ValueError as error:
            raise LLMError(f"{self.name} returned a non-JSON envelope: {error}") from error

        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise LLMError(f"{self.name} returned an empty response")
        return text


class ScriptedClient:
    """Returns canned responses in order. For testing the wrapper, not the model.

    It exists so the schema validation, the retry and the grounding check can be tested
    against malformed and dishonest model output — including output no real model would
    conveniently produce on demand.
    """

    name = "scripted"

    def __init__(self, *responses: str) -> None:
        if not responses:
            raise ValueError("ScriptedClient needs at least one response")
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete_json(self, system: str, prompt: str) -> str:
        self.calls.append((system, prompt))
        if len(self.calls) > len(self._responses):
            return self._responses[-1]
        return self._responses[len(self.calls) - 1]
