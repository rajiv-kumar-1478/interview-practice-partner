"""Application configuration loaded from environment variables via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Twilio
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_whatsapp_number: str

    # LLM
    llm_api_key: str
    llm_api_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o"
    llm_max_context_tokens: int = 8000
    llm_context_window_size: int = 6

    # Redis
    redis_url: str = "redis://localhost:6379"  # Default for local dev, override in production
    redis_session_ttl_seconds: int = 86400
    redis_idempotency_ttl_seconds: int = 600
    redis_pool_size: int = 10

    # Groq Whisper (speech-to-text)
    groq_api_key: str
    groq_whisper_model: str = "whisper-large-v3-turbo"

    # ElevenLabs (text-to-speech)
    elevenlabs_api_key: str
    elevenlabs_voice_id: str
    elevenlabs_model_id: str = "eleven_turbo_v2_5"

    # Media serving
    media_base_url: str = ""  # e.g. "https://your-domain.com" — set in production
    media_ttl_seconds: int = 600  # 10 minutes

    # Application
    app_version: str = "1.0.0"
    log_level: str = "INFO"
    environment: str = "production"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def get_settings() -> Settings:
    return Settings()
