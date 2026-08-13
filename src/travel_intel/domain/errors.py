"""Domain-level errors.

These are raised by business logic and translated to HTTP responses at the API boundary,
so the domain never imports FastAPI.
"""


class TravelIntelError(Exception):
    """Base class for every error this system raises on purpose."""


class NoCandidatesError(TravelIntelError):
    """Retrieval or filtering left nothing to rank.

    Not a bug: an unreachable budget or an unknown destination lands here, and the API
    turns it into an explanatory 422 rather than an empty recommendation.
    """


class ProviderError(TravelIntelError):
    """A data provider failed or returned an unusable payload."""


class LLMError(TravelIntelError):
    """The LLM was unreachable, timed out, or returned unparseable output."""


class GroundingError(LLMError):
    """LLM output referenced an entity that is not in the retrieved data.

    This is the anti-hallucination tripwire: the text is discarded, never shown.
    """
