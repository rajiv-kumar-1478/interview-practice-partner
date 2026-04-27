# Feature: voice-note-support, Property 1: Voice Note Classification is Total and Correct
"""Property-based tests for InterviewService.is_voice_note.

Validates: Requirements 1.1, 1.2
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from interview_practice_partner.services.interview import InterviewService

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_positive_num_media = st.integers(min_value=1, max_value=10)
_zero_num_media = st.just(0)
_negative_num_media = st.integers(max_value=-1)

_audio_content_type = st.from_regex(r"audio/[a-z0-9]+", fullmatch=True)
_ogg_content_type = st.just("application/ogg")
_valid_voice_note_content_type = st.one_of(_audio_content_type, _ogg_content_type)

_non_audio_content_type = st.text(min_size=1).filter(
    lambda s: not s.lower().startswith("audio/") and s.lower() != "application/ogg"
)


# ---------------------------------------------------------------------------
# Property 1a: audio/* content type with num_media > 0 → True
# ---------------------------------------------------------------------------


@given(num_media=_positive_num_media, content_type=_audio_content_type)
@settings(max_examples=200)
def test_property_1a_audio_prefix_with_media_is_voice_note(
    num_media: int, content_type: str
) -> None:
    """Property 1a: any audio/* content type with num_media > 0 is classified as a voice note.

    Validates: Requirement 1.1
    """
    assert InterviewService.is_voice_note(num_media, content_type) is True, (
        f"Expected is_voice_note({num_media!r}, {content_type!r}) to be True"
    )


# ---------------------------------------------------------------------------
# Property 1b: application/ogg with num_media > 0 → True
# ---------------------------------------------------------------------------


@given(num_media=_positive_num_media)
@settings(max_examples=200)
def test_property_1b_application_ogg_with_media_is_voice_note(num_media: int) -> None:
    """Property 1b: application/ogg with num_media > 0 is classified as a voice note.

    Validates: Requirement 1.1
    """
    assert InterviewService.is_voice_note(num_media, "application/ogg") is True, (
        f"Expected is_voice_note({num_media!r}, 'application/ogg') to be True"
    )


# ---------------------------------------------------------------------------
# Property 1c: num_media == 0 → always False
# ---------------------------------------------------------------------------


@given(content_type=st.one_of(st.none(), st.text()))
@settings(max_examples=200)
def test_property_1c_zero_num_media_is_never_voice_note(
    content_type: str | None,
) -> None:
    """Property 1c: num_media == 0 is never classified as a voice note.

    Validates: Requirement 1.2
    """
    assert InterviewService.is_voice_note(0, content_type) is False, (
        f"Expected is_voice_note(0, {content_type!r}) to be False"
    )


# ---------------------------------------------------------------------------
# Property 1d: media_content_type is None → always False
# ---------------------------------------------------------------------------


@given(num_media=st.integers())
@settings(max_examples=200)
def test_property_1d_none_content_type_is_never_voice_note(num_media: int) -> None:
    """Property 1d: None content type is never classified as a voice note.

    Validates: Requirement 1.2
    """
    assert InterviewService.is_voice_note(num_media, None) is False, (
        f"Expected is_voice_note({num_media!r}, None) to be False"
    )


# ---------------------------------------------------------------------------
# Property 1e: non-audio content type with num_media > 0 → False
# ---------------------------------------------------------------------------


@given(num_media=_positive_num_media, content_type=_non_audio_content_type)
@settings(max_examples=200)
def test_property_1e_non_audio_content_type_is_not_voice_note(
    num_media: int, content_type: str
) -> None:
    """Property 1e: non-audio content types are never classified as voice notes.

    Validates: Requirement 1.2
    """
    assert InterviewService.is_voice_note(num_media, content_type) is False, (
        f"Expected is_voice_note({num_media!r}, {content_type!r}) to be False"
    )


# ---------------------------------------------------------------------------
# Property 1f: classification is case-insensitive for content type
# ---------------------------------------------------------------------------


@given(
    num_media=_positive_num_media,
    content_type=_valid_voice_note_content_type,
)
@settings(max_examples=200)
def test_property_1f_classification_is_case_insensitive(
    num_media: int, content_type: str
) -> None:
    """Property 1f: voice note classification is case-insensitive.

    Validates: Requirements 1.1, 1.2
    """
    upper = content_type.upper()
    lower = content_type.lower()
    mixed = content_type.swapcase()

    result_upper = InterviewService.is_voice_note(num_media, upper)
    result_lower = InterviewService.is_voice_note(num_media, lower)
    result_mixed = InterviewService.is_voice_note(num_media, mixed)

    assert result_upper == result_lower == result_mixed, (
        f"Case-insensitivity violated: upper={result_upper}, lower={result_lower}, "
        f"mixed={result_mixed} for content_type={content_type!r}"
    )
