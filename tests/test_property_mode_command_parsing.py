# Feature: voice-note-support, Property 2: Mode Command Parsing is Normalisation-Invariant
"""Property-based tests for InterviewService.is_mode_command.

**Validates: Requirements 6.1, 6.2**
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.services.interview import InterviewService

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Whitespace characters strategy (spaces, tabs, newlines, etc.)
_whitespace_strategy = st.text(
    alphabet=st.characters(whitelist_categories=("Zs",), max_codepoint=0x1000),
    max_size=10,
)

# Core command text with various case combinations
_voice_mode_variants = st.sampled_from([
    "voice mode",
    "VOICE MODE",
    "Voice Mode",
    "VoIcE MoDe",
    "VOICE mode",
    "voice MODE",
])

_text_mode_variants = st.sampled_from([
    "text mode",
    "TEXT MODE",
    "Text Mode",
    "TeXt MoDe",
    "TEXT mode",
    "text MODE",
])

# Non-command strings that should return None
_non_command_strings = st.one_of(
    st.just(""),
    st.just("voice"),
    st.just("text"),
    st.just("mode"),
    st.just("voicemode"),
    st.just("textmode"),
    st.just("voice  mode"),  # double space
    st.just("voice\tmode"),  # tab in middle
    st.just("voice\nmode"),  # newline in middle
    st.just("voice mode please"),
    st.just("switch to voice mode"),
    st.just("I want text mode"),
    st.just("voice mode now"),
    st.text(min_size=1, max_size=50).filter(
        lambda s: s.strip().lower() not in ("voice mode", "text mode")
    ),
)


# ---------------------------------------------------------------------------
# Property 2a: "voice mode" with any whitespace padding → "voice"
# ---------------------------------------------------------------------------


@given(
    prefix=_whitespace_strategy,
    suffix=_whitespace_strategy,
    command=_voice_mode_variants,
)
@settings(max_examples=200)
def test_property_2a_voice_mode_with_whitespace_returns_voice(
    prefix: str, suffix: str, command: str
) -> None:
    """Property 2a: any string normalising to "voice mode" returns "voice".

    **Validates: Requirement 6.1**
    """
    padded = prefix + command + suffix
    result = InterviewService.is_mode_command(padded)
    
    assert result == "voice", (
        f"Expected is_mode_command({padded!r}) to return 'voice', got {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 2b: "text mode" with any whitespace padding → "text"
# ---------------------------------------------------------------------------


@given(
    prefix=_whitespace_strategy,
    suffix=_whitespace_strategy,
    command=_text_mode_variants,
)
@settings(max_examples=200)
def test_property_2b_text_mode_with_whitespace_returns_text(
    prefix: str, suffix: str, command: str
) -> None:
    """Property 2b: any string normalising to "text mode" returns "text".

    **Validates: Requirement 6.2**
    """
    padded = prefix + command + suffix
    result = InterviewService.is_mode_command(padded)
    
    assert result == "text", (
        f"Expected is_mode_command({padded!r}) to return 'text', got {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 2c: non-command strings → None
# ---------------------------------------------------------------------------


@given(body=_non_command_strings)
@settings(max_examples=200)
def test_property_2c_non_command_strings_return_none(body: str) -> None:
    """Property 2c: strings that don't normalise to mode commands return None.

    **Validates: Requirements 6.1, 6.2**
    """
    result = InterviewService.is_mode_command(body)
    
    # Verify the string doesn't normalise to a command
    normalized = body.strip().lower()
    assert normalized not in ("voice mode", "text mode"), (
        f"Test precondition violated: {body!r} normalises to a command"
    )
    
    assert result is None, (
        f"Expected is_mode_command({body!r}) to return None, got {result!r}"
    )


# ---------------------------------------------------------------------------
# Property 2d: normalisation is idempotent
# ---------------------------------------------------------------------------


@given(
    body=st.one_of(
        _voice_mode_variants,
        _text_mode_variants,
        _non_command_strings,
    ),
    prefix=_whitespace_strategy,
    suffix=_whitespace_strategy,
)
@settings(max_examples=200)
def test_property_2d_normalisation_is_idempotent(
    body: str, prefix: str, suffix: str
) -> None:
    """Property 2d: applying whitespace padding multiple times doesn't change result.

    **Validates: Requirements 6.1, 6.2**
    """
    padded_once = prefix + body + suffix
    padded_twice = prefix + padded_once + suffix
    
    result_once = InterviewService.is_mode_command(padded_once)
    result_twice = InterviewService.is_mode_command(padded_twice)
    
    # The results should be the same (both "voice", both "text", or both None)
    assert result_once == result_twice, (
        f"Idempotency violated: is_mode_command({padded_once!r}) = {result_once!r}, "
        f"but is_mode_command({padded_twice!r}) = {result_twice!r}"
    )
