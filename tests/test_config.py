"""Unit tests for Settings configuration fields added in the voice-note-support feature.

Covers:
- Settings raises ValidationError when GROQ_API_KEY is absent (Requirement 8.4)
- Settings raises ValidationError when ELEVENLABS_API_KEY is absent (Requirement 9.5)
- Settings raises ValidationError when ELEVENLABS_VOICE_ID is absent (Requirement 9.5)
- groq_whisper_model defaults to "whisper-large-v3-turbo" (Requirement 8.2)
- elevenlabs_model_id defaults to "eleven_turbo_v2_5" (Requirement 9.3)
- media_ttl_seconds defaults to 600 (Requirement 4.2)
"""

import pytest
from pydantic import ValidationError

from interview_practice_partner.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal set of required fields needed to construct a valid Settings object.
# pydantic-settings reads from environment variables by default, so we pass
# _env_file=None to prevent it from loading the .env file, and supply all
# required fields explicitly as keyword arguments.
REQUIRED_FIELDS = dict(
    twilio_account_sid="ACtest",
    twilio_auth_token="authtoken",
    twilio_whatsapp_number="whatsapp:+15550000000",
    llm_api_key="llm-key",
    redis_url="redis://localhost:6379/0",
    groq_api_key="groq-key",
    elevenlabs_api_key="el-key",
    elevenlabs_voice_id="voice-id",
)


def make_settings(**overrides) -> Settings:
    """Construct a Settings instance with all required fields, optionally overriding some."""
    kwargs = {**REQUIRED_FIELDS, **overrides}
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


# ===========================================================================
# Missing required field tests  (Requirements 8.4, 9.5)
# ===========================================================================


class TestSettingsMissingRequiredFields:
    """Settings must raise ValidationError at construction when required API keys are absent."""

    def test_missing_groq_api_key_raises_validation_error(self):
        """Requirement 8.4: application must fail to start when GROQ_API_KEY is not set."""
        fields = {k: v for k, v in REQUIRED_FIELDS.items() if k != "groq_api_key"}
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, **fields)  # type: ignore[call-arg]
        # Confirm the error mentions the missing field
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "groq_api_key" in field_names

    def test_missing_elevenlabs_api_key_raises_validation_error(self):
        """Requirement 9.5: application must fail to start when ELEVENLABS_API_KEY is not set."""
        fields = {k: v for k, v in REQUIRED_FIELDS.items() if k != "elevenlabs_api_key"}
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, **fields)  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "elevenlabs_api_key" in field_names

    def test_missing_elevenlabs_voice_id_raises_validation_error(self):
        """Requirement 9.5: application must fail to start when ELEVENLABS_VOICE_ID is not set."""
        fields = {k: v for k, v in REQUIRED_FIELDS.items() if k != "elevenlabs_voice_id"}
        with pytest.raises(ValidationError) as exc_info:
            Settings(_env_file=None, **fields)  # type: ignore[call-arg]
        errors = exc_info.value.errors()
        field_names = [e["loc"][0] for e in errors]
        assert "elevenlabs_voice_id" in field_names


# ===========================================================================
# Default value tests  (Requirements 8.2, 9.3)
# ===========================================================================


class TestSettingsDefaults:
    """Optional Settings fields must use their documented default values when not supplied."""

    def test_groq_whisper_model_defaults_to_whisper_large_v3_turbo(self):
        """Requirement 8.2: groq_whisper_model defaults to 'whisper-large-v3-turbo'."""
        settings = make_settings()
        assert settings.groq_whisper_model == "whisper-large-v3-turbo"

    def test_elevenlabs_model_id_defaults_to_eleven_turbo_v2_5(self):
        """Requirement 9.3: elevenlabs_model_id defaults to 'eleven_turbo_v2_5'."""
        settings = make_settings()
        assert settings.elevenlabs_model_id == "eleven_turbo_v2_5"

    def test_media_ttl_seconds_defaults_to_600(self):
        """media_ttl_seconds defaults to 600 (10 minutes)."""
        settings = make_settings()
        assert settings.media_ttl_seconds == 600

    def test_groq_whisper_model_can_be_overridden(self):
        """Requirement 8.2: groq_whisper_model is overridable."""
        settings = make_settings(groq_whisper_model="whisper-large-v3")
        assert settings.groq_whisper_model == "whisper-large-v3"

    def test_elevenlabs_model_id_can_be_overridden(self):
        """Requirement 9.3: elevenlabs_model_id is overridable."""
        settings = make_settings(elevenlabs_model_id="eleven_multilingual_v2")
        assert settings.elevenlabs_model_id == "eleven_multilingual_v2"

    def test_media_ttl_seconds_can_be_overridden(self):
        """media_ttl_seconds is overridable."""
        settings = make_settings(media_ttl_seconds=300)
        assert settings.media_ttl_seconds == 300
