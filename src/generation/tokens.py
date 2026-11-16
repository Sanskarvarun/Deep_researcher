"""Token counting and cost estimation.

Uses tiktoken when available for accurate counts, falling back to a
character-based heuristic (≈4 chars per token) otherwise.
"""

from dataclasses import dataclass

try:  # tiktoken is optional; falls back to a heuristic when missing
    import tiktoken

    _TIKTOKEN_AVAILABLE = True
except ImportError:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]
    _TIKTOKEN_AVAILABLE = False


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @property
    def total_cost_usd(self) -> float:
        """Total estimated cost in USD (delegates to :func:`estimate_cost`)."""
        return estimate_cost(self.prompt_tokens, self.completion_tokens)


# Rough cost per 1K tokens (input, output) in USD for common free-tier models.
# These are estimates; actual billing may differ. All listed models are free
# tier so costs default to 0.0.
MODEL_COSTS: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite": (0.0, 0.0),
    "gemini-2.5-flash": (0.0, 0.0),
    "gemini-1.5-flash": (0.0, 0.0),
    "meta-llama/llama-3.3-70b-instruct:free": (0.0, 0.0),
    "llama-3.1-70b-versatile": (0.0, 0.0),
    "default": (0.0, 0.0),
}

CHARS_PER_TOKEN = 4


def count_tokens(text: str, model: str | None = None) -> int:
    """Estimate the number of tokens in ``text``.

    Uses tiktoken (``cl100k_base``) when installed; otherwise falls back to
    ``len(text) / CHARS_PER_TOKEN``. The ``model`` argument is accepted for
    API compatibility but the heuristic is model-agnostic.
    """
    if not text:
        return 0
    if _TIKTOKEN_AVAILABLE and tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except Exception:
            pass
    return max(1, len(text) // CHARS_PER_TOKEN)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """Count tokens across a list of chat messages."""
    total = 0
    for message in messages:
        total += count_tokens(message.get("content", ""))
        total += count_tokens(message.get("role", ""))
    return total


def estimate_cost(prompt_tokens: int, completion_tokens: int, model: str | None = None) -> float:
    """Estimate the USD cost for a generation."""
    model_name = model or "default"
    input_cost_per_1k, output_cost_per_1k = MODEL_COSTS.get(model_name, MODEL_COSTS["default"])
    return (
        prompt_tokens / 1000.0 * input_cost_per_1k + completion_tokens / 1000.0 * output_cost_per_1k
    )
