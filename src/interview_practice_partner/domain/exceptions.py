"""Domain-specific exceptions for the Interview Practice Partner application."""


class SessionStoreUnavailableError(Exception):
    """Raised when the external session store (Redis) is unreachable or unavailable."""

    def __init__(self, message: str = "Session store is currently unavailable") -> None:
        super().__init__(message)


class InvalidTwilioSignatureError(Exception):
    """Raised when an inbound request fails Twilio HMAC-SHA1 signature validation."""

    def __init__(self, message: str = "Invalid or missing Twilio webhook signature") -> None:
        super().__init__(message)


class LLMError(Exception):
    """Raised when the LLM API call fails after all retries are exhausted."""

    def __init__(self, message: str = "LLM API call failed") -> None:
        super().__init__(message)


class InvalidSessionStateError(Exception):
    """Raised when a session state transition is attempted that violates the state machine rules."""

    def __init__(self, message: str = "Invalid session state transition") -> None:
        super().__init__(message)


class UnsupportedRoleError(Exception):
    """Raised when a user specifies a role that is outside the Agent's supported scope."""

    def __init__(self, role: str) -> None:
        super().__init__(f"Role '{role}' is not supported by this agent")
        self.role = role


class TranscriptionError(Exception):
    """Raised when speech-to-text transcription fails.

    Covers both download failures (AudioDownloadClient) and API errors
    (GroqWhisperClient). InterviewService catches this and sends a text
    fallback message without advancing session state.
    """

    def __init__(self, message: str = "Voice note transcription failed") -> None:
        super().__init__(message)


class TTSError(Exception):
    """Raised when text-to-speech synthesis fails.

    ElevenLabsClient raises this on any API error. InterviewService catches
    this and falls back to sending the reply as a plain text message.
    """

    def __init__(self, message: str = "Text-to-speech synthesis failed") -> None:
        super().__init__(message)
