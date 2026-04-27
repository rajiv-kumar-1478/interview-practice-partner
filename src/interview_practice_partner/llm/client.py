"""LLMClient abstract base class — decouples service layer from LLM provider."""

from abc import ABC, abstractmethod


class LLMClient(ABC):
    """Abstract base class for LLM provider clients.

    Implementations must provide an async ``complete`` method that sends a
    list of chat messages to the underlying model and returns the generated
    text response.  The interface is intentionally minimal so that any
    OpenAI-compatible (or other) provider can be swapped in without touching
    the service layer.
    """

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Send *messages* to the LLM and return the generated text.

        Args:
            messages: A list of chat message dicts, each with at minimum a
                ``"role"`` key (``"system"``, ``"user"``, or ``"assistant"``)
                and a ``"content"`` key containing the message text.
            temperature: Sampling temperature in the range ``[0, 2]``.
                Lower values produce more deterministic output; higher values
                produce more varied output.  Defaults to ``0.7``.
            max_tokens: Maximum number of tokens to generate in the response.
                Defaults to ``1024``.

        Returns:
            The model's text response as a plain string.
        """
        ...
