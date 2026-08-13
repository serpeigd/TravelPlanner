"""The LLM layer.

Two jobs, and only two: turn free text into structured constraints, and write prose about
decisions that deterministic code has already made. Everything here is wrapped in schema
validation and a grounding check, and everything here has a deterministic fallback.
"""

from travel_intel.llm.client import LLMClient, OllamaClient, ScriptedClient
from travel_intel.llm.explain import LLMExplainer, TemplateExplainer
from travel_intel.llm.factory import build_explainer, build_interpreter
from travel_intel.llm.grounding import GroundingContext, build_context, check_grounding
from travel_intel.llm.interpret import (
    KeywordPreferenceInterpreter,
    LLMPreferenceInterpreter,
    PreferenceInterpretation,
)

__all__ = [
    "GroundingContext",
    "KeywordPreferenceInterpreter",
    "LLMClient",
    "LLMExplainer",
    "LLMPreferenceInterpreter",
    "OllamaClient",
    "PreferenceInterpretation",
    "ScriptedClient",
    "TemplateExplainer",
    "build_context",
    "build_explainer",
    "build_interpreter",
    "check_grounding",
]
