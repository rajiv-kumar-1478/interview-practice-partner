"""Unit tests for InterviewService.

Covers:
- handle_short_response returns elaboration prompt for responses under 15 words
- evaluate_response correctly parses LLM JSON output
- handle_skip sets skipped=True on the question
- handle_off_topic increments off_topic_count
- handle_voice_note returns unsupported message
- handle_response orchestration (skip, voice note, short, off-topic, normal)
- generate_question enforces question type variety

Requirements: 4.1, 4.2, 4.3, 4.4, 4.6, 6.4, 7.1, 7.2, 7.3, 8.1, 8.3, 10.5
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from interview_practice_partner.domain.enums import QuestionType, Role, Stage
from interview_practice_partner.domain.exceptions import TranscriptionError
from interview_practice_partner.domain.models import Question, SessionState, UserResponse
from interview_practice_partner.services.interview import (
    InterviewService,
    _count_words,
    _determine_next_question_type,
    _is_audio_media,
    _is_skip_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
PHONE = "+15550001234"


def make_session(**overrides) -> SessionState:
    defaults = dict(
        session_id=str(uuid.uuid4()),
        phone_number=PHONE,
        stage=Stage.INTERVIEW,
        role=Role.SOFTWARE_ENGINEER,
        created_at=NOW,
        updated_at=NOW,
    )
    defaults.update(overrides)
    return SessionState(**defaults)


def make_question(
    question_id: str | None = None,
    text: str = "Tell me about yourself.",
    question_type: QuestionType = QuestionType.BEHAVIOURAL,
    skipped: bool = False,
) -> Question:
    return Question(
        question_id=question_id or str(uuid.uuid4()),
        text=text,
        question_type=question_type,
        asked_at=NOW,
        skipped=skipped,
    )


def make_response(
    question_id: str,
    text: str = "I have five years of experience working on distributed systems.",
) -> UserResponse:
    return UserResponse(
        response_id=str(uuid.uuid4()),
        question_id=question_id,
        text=text,
        word_count=len(text.split()),
        received_at=NOW,
    )


def make_service(
    llm_response: str = "What is your experience with Python?",
) -> tuple[InterviewService, AsyncMock, MagicMock]:
    """Build an InterviewService with mocked dependencies.

    Returns (service, mock_llm, mock_prompt_builder).
    """
    mock_llm = AsyncMock()
    mock_llm.complete.return_value = llm_response

    mock_prompt_builder = MagicMock()
    mock_prompt_builder.build_question_generation_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]
    mock_prompt_builder.build_response_evaluation_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    # Mock audio clients (added for voice note support)
    mock_whisper = AsyncMock()
    mock_tts = AsyncMock()
    mock_audio_download = AsyncMock()

    service = InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
    )
    return service, mock_llm, mock_prompt_builder


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestHelperFunctions:
    def test_count_words_normal(self):
        assert _count_words("Hello world") == 2

    def test_count_words_empty(self):
        assert _count_words("") == 0

    def test_count_words_single(self):
        assert _count_words("Hello") == 1

    def test_count_words_many(self):
        text = " ".join(["word"] * 20)
        assert _count_words(text) == 20

    def test_is_skip_request_skip(self):
        assert _is_skip_request("skip") is True

    def test_is_skip_request_next_question(self):
        assert _is_skip_request("next question") is True

    def test_is_skip_request_pass(self):
        assert _is_skip_request("pass") is True

    def test_is_skip_request_normal_answer(self):
        assert _is_skip_request("I have five years of experience.") is False

    def test_is_skip_request_case_insensitive(self):
        assert _is_skip_request("SKIP") is True

    def test_is_audio_media_audio_ogg(self):
        assert _is_audio_media("audio/ogg") is True

    def test_is_audio_media_audio_mpeg(self):
        assert _is_audio_media("audio/mpeg") is True

    def test_is_audio_media_image(self):
        assert _is_audio_media("image/jpeg") is False

    def test_is_audio_media_none(self):
        assert _is_audio_media(None) is False

    def test_is_audio_media_empty(self):
        assert _is_audio_media("") is False


# ===========================================================================
# Question type variety
# ===========================================================================


class TestDetermineNextQuestionType:
    def test_empty_session_returns_behavioural(self):
        """Empty session → first question is BEHAVIOURAL."""
        session = make_session()
        assert _determine_next_question_type(session) == QuestionType.BEHAVIOURAL

    def test_after_behavioural_returns_situational(self):
        """After one BEHAVIOURAL → next is SITUATIONAL."""
        session = make_session()
        session.questions.append(make_question(question_type=QuestionType.BEHAVIOURAL))
        assert _determine_next_question_type(session) == QuestionType.SITUATIONAL

    def test_after_behavioural_and_situational_returns_technical(self):
        """After BEHAVIOURAL + SITUATIONAL → next is TECHNICAL."""
        session = make_session()
        session.questions.append(make_question(question_type=QuestionType.BEHAVIOURAL))
        session.questions.append(make_question(question_type=QuestionType.SITUATIONAL))
        assert _determine_next_question_type(session) == QuestionType.TECHNICAL

    def test_after_all_three_cycles_back(self):
        """After all three types → cycles back to BEHAVIOURAL (fewest count)."""
        session = make_session()
        session.questions.append(make_question(question_type=QuestionType.BEHAVIOURAL))
        session.questions.append(make_question(question_type=QuestionType.SITUATIONAL))
        session.questions.append(make_question(question_type=QuestionType.TECHNICAL))
        # All have count=1, so BEHAVIOURAL (first in cycle) is returned
        assert _determine_next_question_type(session) == QuestionType.BEHAVIOURAL

    def test_follow_up_questions_not_counted(self):
        """FOLLOW_UP questions don't affect the type cycle."""
        session = make_session()
        session.questions.append(make_question(question_type=QuestionType.BEHAVIOURAL))
        session.questions.append(make_question(question_type=QuestionType.FOLLOW_UP))
        # Only BEHAVIOURAL counted → next should be SITUATIONAL
        assert _determine_next_question_type(session) == QuestionType.SITUATIONAL


# ===========================================================================
# generate_question
# ===========================================================================


class TestGenerateQuestion:
    @pytest.mark.asyncio
    async def test_generate_question_calls_llm(self):
        """generate_question calls LLM and returns question text."""
        service, mock_llm, mock_pb = make_service(
            llm_response="Describe a time you led a project."
        )
        session = make_session()

        result = await service.generate_question(session)

        assert result == "Describe a time you led a project."
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_question_appends_to_session(self):
        """generate_question appends the new Question to session.questions."""
        service, _, _ = make_service(llm_response="What is your experience with Python?")
        session = make_session()

        await service.generate_question(session)

        assert len(session.questions) == 1
        assert session.questions[0].text == "What is your experience with Python?"

    @pytest.mark.asyncio
    async def test_generate_question_uses_correct_type(self):
        """generate_question records the correct question_type."""
        service, _, _ = make_service(llm_response="Tell me about a challenge.")
        session = make_session()

        await service.generate_question(session, question_type=QuestionType.SITUATIONAL)

        assert session.questions[0].question_type == QuestionType.SITUATIONAL

    @pytest.mark.asyncio
    async def test_generate_question_passes_difficulty_signal(self):
        """generate_question passes difficulty_signal to PromptBuilder."""
        service, _, mock_pb = make_service(llm_response="A harder question.")
        session = make_session()

        await service.generate_question(session, difficulty_signal="increase")

        mock_pb.build_question_generation_prompt.assert_called_once_with(
            session=session,
            question_type=QuestionType.BEHAVIOURAL,
            difficulty_signal="increase",
        )

    @pytest.mark.asyncio
    async def test_generate_question_strips_whitespace(self):
        """generate_question strips leading/trailing whitespace from LLM output."""
        service, _, _ = make_service(llm_response="  What is your experience?  \n")
        session = make_session()

        result = await service.generate_question(session)

        assert result == "What is your experience?"

    @pytest.mark.asyncio
    async def test_generate_question_enforces_variety(self):
        """generate_question cycles through question types for variety."""
        service, _, _ = make_service(llm_response="A question.")
        session = make_session()

        # First question: BEHAVIOURAL
        await service.generate_question(session)
        assert session.questions[0].question_type == QuestionType.BEHAVIOURAL

        # Second question: SITUATIONAL
        await service.generate_question(session)
        assert session.questions[1].question_type == QuestionType.SITUATIONAL

        # Third question: TECHNICAL
        await service.generate_question(session)
        assert session.questions[2].question_type == QuestionType.TECHNICAL


# ===========================================================================
# evaluate_response
# ===========================================================================


class TestEvaluateResponse:
    @pytest.mark.asyncio
    async def test_evaluate_response_parses_json(self):
        """evaluate_response correctly parses LLM JSON output."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": True,
            "follow_up_text": "Can you give a specific example?",
            "difficulty_signal": "increase",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session()
        q = make_question()
        session.questions.append(q)
        response = make_response(question_id=q.question_id)

        result = await service.evaluate_response(session, response)

        assert result["is_off_topic"] is False
        assert result["is_short"] is False
        assert result["follow_up_warranted"] is True
        assert result["follow_up_text"] == "Can you give a specific example?"
        assert result["difficulty_signal"] == "increase"

    @pytest.mark.asyncio
    async def test_evaluate_response_sets_is_off_topic_on_response(self):
        """evaluate_response sets is_off_topic on the UserResponse object."""
        eval_json = json.dumps({
            "is_off_topic": True,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session()
        q = make_question()
        session.questions.append(q)
        response = make_response(question_id=q.question_id)

        await service.evaluate_response(session, response)

        assert response.is_off_topic is True

    @pytest.mark.asyncio
    async def test_evaluate_response_fallback_on_invalid_json(self):
        """evaluate_response falls back to safe defaults on invalid JSON."""
        service, mock_llm, _ = make_service(llm_response="Not valid JSON at all!")
        session = make_session()
        q = make_question()
        session.questions.append(q)
        # Response with 5 words (short)
        response = UserResponse(
            response_id=str(uuid.uuid4()),
            question_id=q.question_id,
            text="I don't know.",
            word_count=4,
            received_at=NOW,
        )

        result = await service.evaluate_response(session, response)

        assert result["is_off_topic"] is False
        assert result["is_short"] is True  # word_count=4 < 15
        assert result["follow_up_warranted"] is False
        assert result["follow_up_text"] is None
        assert result["difficulty_signal"] == "maintain"

    @pytest.mark.asyncio
    async def test_evaluate_response_question_not_found_returns_defaults(self):
        """evaluate_response returns safe defaults when question not found."""
        service, _, _ = make_service()
        session = make_session()
        # Response with a question_id not in session.questions
        response = UserResponse(
            response_id=str(uuid.uuid4()),
            question_id="nonexistent-id",
            text="Some answer here.",
            word_count=3,
            received_at=NOW,
        )

        result = await service.evaluate_response(session, response)

        assert result["is_off_topic"] is False
        assert result["difficulty_signal"] == "maintain"


# ===========================================================================
# handle_short_response
# ===========================================================================


class TestHandleShortResponse:
    def test_handle_short_response_returns_elaboration_prompt(self):
        """handle_short_response returns an elaboration prompt."""
        service, _, _ = make_service()
        session = make_session()

        reply = service.handle_short_response(session)

        assert isinstance(reply, str)
        assert len(reply) > 0
        # Should ask for elaboration
        assert any(
            word in reply.lower()
            for word in ["elaborate", "more", "detail", "example"]
        )

    def test_handle_short_response_does_not_advance_session(self):
        """handle_short_response does not modify session questions or responses."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question()
        session.questions.append(q)
        original_question_count = len(session.questions)
        original_response_count = len(session.responses)

        service.handle_short_response(session)

        assert len(session.questions) == original_question_count
        assert len(session.responses) == original_response_count


# ===========================================================================
# handle_off_topic
# ===========================================================================


class TestHandleOffTopic:
    def test_handle_off_topic_increments_off_topic_count(self):
        """handle_off_topic increments off_topic_count."""
        service, _, _ = make_service()
        session = make_session(off_topic_count=0)

        service.handle_off_topic(session)

        assert session.off_topic_count == 1

    def test_handle_off_topic_increments_consecutive_count(self):
        """handle_off_topic increments consecutive_out_of_scope_count."""
        service, _, _ = make_service()
        session = make_session(consecutive_out_of_scope_count=0)

        service.handle_off_topic(session)

        assert session.consecutive_out_of_scope_count == 1

    def test_handle_off_topic_returns_redirect_message(self):
        """handle_off_topic returns a redirect message."""
        service, _, _ = make_service()
        session = make_session()

        reply = service.handle_off_topic(session)

        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_handle_off_topic_three_consecutive_offers_end_or_restart(self):
        """After 3+ consecutive off-topic inputs, offers to end or restart."""
        service, _, _ = make_service()
        session = make_session(consecutive_out_of_scope_count=2)

        reply = service.handle_off_topic(session)

        # consecutive_out_of_scope_count is now 3
        assert session.consecutive_out_of_scope_count == 3
        # Reply should offer to end or return to role selection
        assert any(
            phrase in reply.lower()
            for phrase in ["end", "role selection", "1", "2"]
        )

    def test_handle_off_topic_accumulates_total_count(self):
        """handle_off_topic accumulates off_topic_count across multiple calls."""
        service, _, _ = make_service()
        session = make_session(off_topic_count=2)

        service.handle_off_topic(session)
        service.handle_off_topic(session)

        assert session.off_topic_count == 4


# ===========================================================================
# handle_skip
# ===========================================================================


class TestHandleSkip:
    @pytest.mark.asyncio
    async def test_handle_skip_marks_question_skipped(self):
        """handle_skip sets skipped=True on the current question."""
        service, _, _ = make_service(llm_response="Next question text.")
        session = make_session()
        q = make_question(text="Tell me about yourself.")
        session.questions.append(q)

        await service.handle_skip(session)

        assert q.skipped is True

    @pytest.mark.asyncio
    async def test_handle_skip_generates_next_question(self):
        """handle_skip generates and returns the next question."""
        service, mock_llm, _ = make_service(llm_response="What is your greatest strength?")
        session = make_session()
        q = make_question()
        session.questions.append(q)

        reply, updated = await service.handle_skip(session)

        assert "What is your greatest strength?" in reply
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_skip_acknowledges_skip(self):
        """handle_skip reply acknowledges the skip."""
        service, _, _ = make_service(llm_response="Next question.")
        session = make_session()
        q = make_question()
        session.questions.append(q)

        reply, _ = await service.handle_skip(session)

        assert any(
            word in reply.lower()
            for word in ["skip", "no problem", "next"]
        )

    @pytest.mark.asyncio
    async def test_handle_skip_no_current_question_still_generates_next(self):
        """handle_skip with no unanswered question still generates next question."""
        service, mock_llm, _ = make_service(llm_response="A new question.")
        session = make_session()
        # No questions in session

        reply, updated = await service.handle_skip(session)

        assert "A new question." in reply
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_skip_skips_most_recent_unanswered(self):
        """handle_skip marks the most recent unanswered question as skipped."""
        service, _, _ = make_service(llm_response="Next question.")
        session = make_session()
        q1 = make_question(text="Question 1?")
        q2 = make_question(text="Question 2?")
        session.questions.extend([q1, q2])
        # q1 is answered
        session.responses.append(make_response(question_id=q1.question_id))

        await service.handle_skip(session)

        # q2 should be skipped (most recent unanswered)
        assert q2.skipped is True
        assert q1.skipped is False


# ===========================================================================
# handle_voice_note
# ===========================================================================


class TestHandleModeCommand:
    @pytest.mark.asyncio
    async def test_handle_mode_command_voice_sets_preferred_mode(self):
        """handle_mode_command('voice') sets preferred_mode='voice'."""
        service, _, _ = make_service()
        session = make_session()
        assert session.preferred_mode == "text"  # default

        confirmation, updated = await service.handle_mode_command(session, "voice")

        assert updated.preferred_mode == "voice"

    @pytest.mark.asyncio
    async def test_handle_mode_command_text_sets_preferred_mode(self):
        """handle_mode_command('text') sets preferred_mode='text'."""
        service, _, _ = make_service()
        session = make_session(preferred_mode="voice")

        confirmation, updated = await service.handle_mode_command(session, "text")

        assert updated.preferred_mode == "text"

    @pytest.mark.asyncio
    async def test_handle_mode_command_voice_returns_confirmation(self):
        """handle_mode_command('voice') returns a non-empty confirmation string."""
        service, _, _ = make_service()
        session = make_session()

        confirmation, _ = await service.handle_mode_command(session, "voice")

        assert isinstance(confirmation, str)
        assert len(confirmation) > 0
        assert any(
            phrase in confirmation.lower()
            for phrase in ["voice mode", "active", "audio"]
        )

    @pytest.mark.asyncio
    async def test_handle_mode_command_text_returns_confirmation(self):
        """handle_mode_command('text') returns a non-empty confirmation string."""
        service, _, _ = make_service()
        session = make_session()

        confirmation, _ = await service.handle_mode_command(session, "text")

        assert isinstance(confirmation, str)
        assert len(confirmation) > 0
        assert any(
            phrase in confirmation.lower()
            for phrase in ["text mode", "active", "text"]
        )

    @pytest.mark.asyncio
    async def test_handle_mode_command_does_not_mutate_questions(self):
        """handle_mode_command does not mutate session.questions."""
        service, _, _ = make_service()
        session = make_session()
        q1 = make_question(text="Question 1?")
        q2 = make_question(text="Question 2?")
        session.questions.extend([q1, q2])
        original_questions = list(session.questions)

        _, updated = await service.handle_mode_command(session, "voice")

        assert updated.questions == original_questions

    @pytest.mark.asyncio
    async def test_handle_mode_command_does_not_mutate_responses(self):
        """handle_mode_command does not mutate session.responses."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question()
        session.questions.append(q)
        r = make_response(question_id=q.question_id)
        session.responses.append(r)
        original_responses = list(session.responses)

        _, updated = await service.handle_mode_command(session, "text")

        assert updated.responses == original_responses


class TestHandleVoiceNote:
    def test_handle_voice_note_returns_unsupported_message(self):
        """handle_voice_note returns a message saying voice notes are not supported."""
        service, _, _ = make_service()
        session = make_session()

        # This test is now obsolete since handle_voice_note has been replaced
        # with full implementation. The old behavior (unsupported message)
        # is now only returned on TranscriptionError or empty transcription.
        # We'll test the new behavior instead.
        
        # Test case: TranscriptionError during download should return fallback message
        with patch.object(service._audio_download, 'download', side_effect=TranscriptionError("Download failed")):
            reply, updated_session = asyncio.run(
                service.handle_voice_note(session, "https://example.com/media", "audio/ogg")
            )

        assert isinstance(reply, str)
        assert len(reply) > 0
        assert "couldn't process your voice note" in reply.lower()
        assert updated_session == session  # session should be unchanged

    def test_handle_voice_note_transcription_error_returns_fallback(self):
        """handle_voice_note returns fallback message when transcription fails."""
        service, _, _ = make_service()
        session = make_session()

        # Mock successful download but failed transcription
        with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
            with patch.object(service._whisper, 'transcribe', side_effect=TranscriptionError("API error")):
                reply, updated_session = asyncio.run(
                    service.handle_voice_note(session, "https://example.com/media", "audio/mpeg")
                )

        assert "couldn't process your voice note" in reply.lower()
        assert updated_session == session  # session should be unchanged

    def test_handle_voice_note_empty_transcription_returns_fallback(self):
        """handle_voice_note returns specific message when transcription is empty."""
        service, _, _ = make_service()
        session = make_session()

        # Mock successful download and transcription but empty result
        with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
            with patch.object(service._whisper, 'transcribe', return_value="   "):  # empty/whitespace
                reply, updated_session = asyncio.run(
                    service.handle_voice_note(session, "https://example.com/media", "audio/ogg")
                )

        assert "couldn't make out what you said" in reply.lower()
        assert updated_session == session  # session should be unchanged

    def test_handle_voice_note_successful_transcription_delegates_to_handle_response(self):
        """handle_voice_note sets preferred_mode to voice and delegates to handle_response."""
        service, mock_llm, _ = make_service()
        session = make_session()
        session.questions.append(make_question())  # Need a question for handle_response

        # Mock successful download and transcription (use longer text to avoid short response handling)
        transcribed_text = "This is my detailed answer to the interview question with sufficient length to avoid being flagged as too short"
        
        # Mock LLM response for evaluation
        eval_json = json.dumps({
            "is_off_topic": False,
            "difficulty_signal": "maintain",
            "follow_up_warranted": False,
        })
        mock_llm.complete.return_value = eval_json

        with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
            with patch.object(service._whisper, 'transcribe', return_value=transcribed_text):
                with patch.object(service, 'generate_question', return_value="Next question?"):
                    reply, updated_session = asyncio.run(
                        service.handle_voice_note(session, "https://example.com/media", "audio/ogg")
                    )

        # Should set preferred_mode to voice
        assert updated_session.preferred_mode == "voice"
        
        # Should have processed the transcribed text as a response
        assert len(updated_session.responses) == 1
        assert updated_session.responses[0].text == transcribed_text
        
        # Should return a reply (from handle_response)
        assert isinstance(reply, str)
        assert len(reply) > 0

    def test_handle_voice_note_derives_filename_from_content_type(self):
        """handle_voice_note derives correct filename from media_content_type."""
        service, _, _ = make_service()
        session = make_session()

        # Test different content types including case-insensitive handling
        test_cases = [
            ("audio/mpeg", "voice_note.mp3"),
            ("AUDIO/MPEG", "voice_note.mp3"),  # case-insensitive
            ("audio/ogg", "voice_note.ogg"),
            ("AUDIO/OGG", "voice_note.ogg"),  # case-insensitive
            ("audio/wav", "voice_note.ogg"),  # default
            ("", "voice_note.ogg"),  # default
            (None, "voice_note.ogg"),  # None content type
        ]

        for content_type, expected_filename in test_cases:
            with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
                with patch.object(service._whisper, 'transcribe', side_effect=TranscriptionError("test")) as mock_transcribe:
                    asyncio.run(
                        service.handle_voice_note(session, "https://example.com/media", content_type or "")
                    )
                    
                    # Verify the correct filename was passed to transcribe
                    mock_transcribe.assert_called_once_with(b"audio_bytes", expected_filename)


# ===========================================================================
# handle_response — main orchestration
# ===========================================================================


class TestHandleResponse:
    @pytest.mark.asyncio
    async def test_handle_response_skip_keyword_calls_handle_skip(self):
        """handle_response with 'skip' keyword calls handle_skip."""
        service, mock_llm, _ = make_service(llm_response="Next question after skip.")
        session = make_session()
        q = make_question()
        session.questions.append(q)

        # classify_intent is called first, then generate_question for the skip path
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            "Next question after skip.",
        ]

        reply, updated = await service.handle_response(session, "skip")

        assert q.skipped is True
        assert "Next question after skip." in reply

    @pytest.mark.asyncio
    async def test_handle_response_routes_to_handle_voice_note_with_audio(self):
        """handle_response routes to handle_voice_note when num_media > 0 and audio content type."""
        service, _, _ = make_service()
        session = make_session()
        
        # Mock the handle_voice_note method to verify it's called
        with patch.object(service, 'handle_voice_note', new_callable=AsyncMock) as mock_handle_voice_note:
            mock_handle_voice_note.return_value = ("Voice note processed", session)
            
            reply, updated = await service.handle_response(
                session,
                user_message="",
                num_media=1,
                media_content_type="audio/ogg",
                media_url="https://api.twilio.com/media/123",
            )
            
            # Verify handle_voice_note was called with correct parameters
            mock_handle_voice_note.assert_called_once_with(session, "https://api.twilio.com/media/123", "audio/ogg")
            assert reply == "Voice note processed"

    @pytest.mark.asyncio
    async def test_handle_response_routes_to_handle_voice_note_with_audio_mpeg(self):
        """handle_response routes to handle_voice_note for audio/mpeg content type."""
        service, _, _ = make_service()
        session = make_session()
        
        with patch.object(service, 'handle_voice_note', new_callable=AsyncMock) as mock_handle_voice_note:
            mock_handle_voice_note.return_value = ("Voice note processed", session)
            
            reply, updated = await service.handle_response(
                session,
                user_message="",
                num_media=1,
                media_content_type="audio/mpeg",
                media_url="https://api.twilio.com/media/456",
            )
            
            mock_handle_voice_note.assert_called_once_with(session, "https://api.twilio.com/media/456", "audio/mpeg")

    @pytest.mark.asyncio
    async def test_handle_response_routes_to_handle_voice_note_with_application_ogg(self):
        """handle_response routes to handle_voice_note for application/ogg content type."""
        service, _, _ = make_service()
        session = make_session()
        
        with patch.object(service, 'handle_voice_note', new_callable=AsyncMock) as mock_handle_voice_note:
            mock_handle_voice_note.return_value = ("Voice note processed", session)
            
            reply, updated = await service.handle_response(
                session,
                user_message="",
                num_media=1,
                media_content_type="application/ogg",
                media_url="https://api.twilio.com/media/789",
            )
            
            mock_handle_voice_note.assert_called_once_with(session, "https://api.twilio.com/media/789", "application/ogg")

    @pytest.mark.asyncio
    async def test_handle_response_voice_note_without_media_url_returns_error(self):
        """handle_response with voice note but no media_url returns error message."""
        service, _, _ = make_service()
        session = make_session()

        reply, updated = await service.handle_response(
            session,
            user_message="",
            num_media=1,
            media_content_type="audio/ogg",
            media_url=None,  # Missing media URL
        )

        assert "couldn't process your voice note" in reply.lower()
        assert "resend it or type your answer" in reply.lower()
        assert updated == session  # Session should be unchanged

    @pytest.mark.asyncio
    async def test_handle_response_short_response_returns_elaboration(self):
        """handle_response with < 15 words returns elaboration prompt."""
        service, mock_llm, _ = make_service()
        session = make_session()
        q = make_question()
        session.questions.append(q)

        # classify_intent is called first (falls back to "answer" on non-JSON),
        # then short-response check fires before evaluation
        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"})]

        # 5 words — short
        reply, updated = await service.handle_response(session, "I don't know much.")

        assert any(
            word in reply.lower()
            for word in ["elaborate", "more", "detail", "example"]
        )
        # Session should NOT have a new response recorded
        assert len(updated.responses) == 0
        # LLM should have been called once for classify_intent only (not for evaluation)
        mock_llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_response_off_topic_increments_counter(self):
        """handle_response with off-topic response increments off_topic_count."""
        eval_json = json.dumps({
            "is_off_topic": True,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session()
        q = make_question()
        session.questions.append(q)

        # 20-word response that is off-topic (must be >= 15 words to bypass short-response check)
        long_off_topic = "I really enjoy cooking pasta and making homemade sauces on weekends with my whole family together."
        reply, updated = await service.handle_response(session, long_off_topic)

        assert updated.off_topic_count == 1
        assert updated.consecutive_out_of_scope_count == 1
        # Response should NOT be recorded in session
        assert len(updated.responses) == 0

    @pytest.mark.asyncio
    async def test_handle_response_normal_records_response(self):
        """handle_response with normal response records it in session."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session()
        q = make_question()
        session.questions.append(q)

        # Override the second LLM call (question generation) to return a question
        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "What is your greatest strength?"]

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        reply, updated = await service.handle_response(session, long_answer)

        assert len(updated.responses) == 1
        assert updated.responses[0].text == long_answer

    @pytest.mark.asyncio
    async def test_handle_response_normal_resets_consecutive_count(self):
        """handle_response with normal response resets consecutive_out_of_scope_count."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session(consecutive_out_of_scope_count=2)
        q = make_question()
        session.questions.append(q)

        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "Next question?"]

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        _, updated = await service.handle_response(session, long_answer)

        assert updated.consecutive_out_of_scope_count == 0

    @pytest.mark.asyncio
    async def test_handle_response_follow_up_warranted_generates_follow_up(self):
        """handle_response with follow_up_warranted=True generates a follow-up question."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": True,
            "follow_up_text": "Can you give a specific example of that?",
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session()
        q = make_question()
        session.questions.append(q)

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        reply, updated = await service.handle_response(session, long_answer)

        # A follow-up question should be added to session.questions
        follow_up_questions = [
            q for q in updated.questions if q.question_type == QuestionType.FOLLOW_UP
        ]
        assert len(follow_up_questions) == 1
        assert follow_up_questions[0].text == "Can you give a specific example of that?"
        assert "Can you give a specific example of that?" in reply

    @pytest.mark.asyncio
    async def test_handle_response_next_question_keyword_triggers_skip(self):
        """handle_response with 'next question' triggers skip."""
        service, mock_llm, _ = make_service(llm_response="Here is the next question.")
        session = make_session()
        q = make_question()
        session.questions.append(q)

        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            "Here is the next question.",
        ]

        reply, updated = await service.handle_response(session, "next question")

        assert q.skipped is True

    @pytest.mark.asyncio
    async def test_handle_response_non_audio_media_not_treated_as_voice_note(self):
        """handle_response with image media is not treated as a voice note."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session()
        q = make_question()
        session.questions.append(q)

        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "Next question?"]

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        reply, updated = await service.handle_response(
            session,
            long_answer,
            num_media=1,
            media_content_type="image/jpeg",
        )

        # Should NOT return voice note unsupported message
        assert "voice note" not in reply.lower()
        # Response should be recorded
        assert len(updated.responses) == 1

    @pytest.mark.asyncio
    async def test_handle_response_routes_mode_commands_first(self):
        """handle_response routes mode commands before other processing."""
        service, _, _ = make_service()
        session = make_session()
        
        # Test voice mode command
        reply, updated = await service.handle_response(session, "voice mode")
        
        assert updated.preferred_mode == "voice"
        assert "voice mode is now active" in reply.lower()
        # Should not process as regular text (no LLM calls for intent classification)
        
        # Test text mode command
        reply, updated = await service.handle_response(updated, "text mode")
        
        assert updated.preferred_mode == "text"
        assert "text mode is now active" in reply.lower()

    @pytest.mark.asyncio
    async def test_handle_response_routes_mode_commands_case_insensitive(self):
        """handle_response routes mode commands case-insensitively."""
        service, _, _ = make_service()
        session = make_session()
        
        # Test uppercase voice mode command
        reply, updated = await service.handle_response(session, "VOICE MODE")
        
        assert updated.preferred_mode == "voice"
        assert "voice mode is now active" in reply.lower()
        
        # Test mixed case text mode command
        reply, updated = await service.handle_response(updated, "Text Mode")
        
        assert updated.preferred_mode == "text"
        assert "text mode is now active" in reply.lower()

    @pytest.mark.asyncio
    async def test_handle_response_routes_mode_commands_with_whitespace(self):
        """handle_response routes mode commands with extra whitespace."""
        service, _, _ = make_service()
        session = make_session()
        
        # Test voice mode command with leading/trailing whitespace
        reply, updated = await service.handle_response(session, "  voice mode  ")
        
        assert updated.preferred_mode == "voice"
        assert "voice mode is now active" in reply.lower()
        
        # Test text mode command with whitespace
        reply, updated = await service.handle_response(updated, "\t text mode \n")
        
        assert updated.preferred_mode == "text"
        assert "text mode is now active" in reply.lower()

    @pytest.mark.asyncio
    async def test_handle_response_sets_text_mode_for_text_messages(self):
        """handle_response sets preferred_mode='text' silently for text messages."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _ = make_service(llm_response=eval_json)
        session = make_session(preferred_mode="voice")  # Start in voice mode
        q = make_question()
        session.questions.append(q)

        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "Next question?"]

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale, focusing on performance optimization and reliability."
        reply, updated = await service.handle_response(session, long_answer)

        # Should silently switch to text mode
        assert updated.preferred_mode == "text"
        # Should process normally (response recorded)
        assert len(updated.responses) == 1
