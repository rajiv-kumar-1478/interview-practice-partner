"""OpenAIClient — AsyncOpenAI implementation of LLMClient with tenacity retry logic."""

import time

import openai
import structlog
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from interview_practice_partner.config import Settings
from interview_practice_partner.domain.exceptions import LLMError
from interview_practice_partner.llm.client import LLMClient

logger = structlog.get_logger()


def _is_retryable(exc: BaseException) -> bool:
    """Return True for APITimeoutError and APIStatusError with 5xx status codes."""
    if isinstance(exc, openai.APITimeoutError):
        return True
    if isinstance(exc, openai.APIStatusError) and exc.status_code >= 500:
        return True
    return False


class OpenAIClient(LLMClient):
    """AsyncOpenAI-backed implementation of LLMClient.

    Wraps every call with tenacity retry logic (2 retries, exponential backoff
    starting at 1 second) on transient errors (timeouts and 5xx responses).
    Logs model, token usage, and latency via structlog after each successful call.
    """

    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_api_base_url,
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable),
        reraise=True,
    )
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        use_json_mode: bool,
    ) -> openai.types.chat.ChatCompletion:
        """Make the raw API call with retry logic applied."""
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return await self._client.chat.completions.create(**kwargs)

    async def complete(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        use_json_mode: bool = False,
    ) -> str:
        """Send *messages* to the LLM and return the generated text.

        Args:
            messages: Chat message dicts with ``"role"`` and ``"content"`` keys.
            temperature: Sampling temperature in ``[0, 2]``. Defaults to ``0.7``.
            max_tokens: Maximum tokens to generate. Defaults to ``1024``.
            use_json_mode: When ``True``, sets ``response_format={"type": "json_object"}``
                to request structured JSON output. Defaults to ``False``.

        Returns:
            The model's text response as a plain string.

        Raises:
            LLMError: When the API call fails after all retries are exhausted.
        """
        start_time = time.monotonic()
        try:
            response = await self._call_api(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                use_json_mode=use_json_mode,
            )
        except (openai.APITimeoutError, openai.APIStatusError) as exc:
            raise LLMError(f"LLM API call failed: {exc}") from exc

        latency_ms = (time.monotonic() - start_time) * 1000
        usage = response.usage

        logger.info(
            "llm_api_call",
            model=self._model,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            latency_ms=round(latency_ms, 2),
        )

        content = response.choices[0].message.content
        return content or ""
