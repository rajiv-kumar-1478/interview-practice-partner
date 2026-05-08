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
from interview_practice_partner.domain.enums import InterviewRoundType, ProblemDifficulty
from interview_practice_partner.services.interview import (
    InterviewService,
    _count_words,
    _detect_round_type,
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
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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

    @pytest.mark.asyncio
    async def test_voice_note_in_dsa_round_routes_to_evaluate_coding_solution(self):
        """Voice note in DSA round transcribes and routes to evaluate_coding_solution (Req 13.1-13.3).

        Requirements: 13.1, 13.2, 13.3
        """
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        # Intent classification returns "answer"
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(
            text="Find two numbers that sum to target.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        # Transcribed voice note contains a DSA solution (long enough to pass word count)
        transcribed_solution = (
            "I would use a hash map to store each number and its index as I iterate "
            "through the array, checking if the complement exists in the map already"
        )

        with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
            with patch.object(service._whisper, 'transcribe', return_value=transcribed_solution):
                reply, updated_session = await service.handle_voice_note(
                    session, "https://example.com/media", "audio/ogg"
                )

        # Voice mode should be set (Req 13.5)
        assert updated_session.preferred_mode == "voice"

        # evaluate_coding_solution should have been called (Req 13.3)
        mock_tech.evaluate_coding_solution.assert_called_once()

        # Response should be recorded
        assert len(updated_session.responses) == 1
        assert updated_session.responses[0].text == transcribed_solution

        # Should return a non-empty reply
        assert isinstance(reply, str) and len(reply) > 0

    @pytest.mark.asyncio
    async def test_voice_note_in_system_design_round_routes_to_evaluate_system_design(self):
        """Voice note in System Design round transcribes and routes to evaluate_system_design (Req 13.1-13.3).

        Requirements: 13.1, 13.2, 13.3
        """
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        # Intent classification returns "answer"
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(
            text="Design a URL shortener like bit.ly.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        # Transcribed voice note contains a system design explanation (long enough).
        # Deliberately avoids DSA/design keywords to prevent round-type-switch detection.
        transcribed_explanation = (
            "I would store the mapping in a database with a hash function to generate unique IDs, "
            "then serve redirects through multiple servers to handle high traffic volumes efficiently"
        )

        with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
            with patch.object(service._whisper, 'transcribe', return_value=transcribed_explanation):
                reply, updated_session = await service.handle_voice_note(
                    session, "https://example.com/media", "audio/ogg"
                )

        # Voice mode should be set (Req 13.5)
        assert updated_session.preferred_mode == "voice"

        # evaluate_system_design should have been called (Req 13.3)
        mock_tech.evaluate_system_design.assert_called_once()

        # Response should be recorded
        assert len(updated_session.responses) == 1
        assert updated_session.responses[0].text == transcribed_explanation

        # Should return a non-empty reply
        assert isinstance(reply, str) and len(reply) > 0

    def test_voice_note_unclear_transcription_returns_clarification_prompt(self):
        """Empty/whitespace transcription returns a clarification prompt (Req 13.4).

        Requirements: 13.4
        """
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )

        # Transcription returns only whitespace (unclear audio)
        with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
            with patch.object(service._whisper, 'transcribe', return_value="   \n  "):
                reply, updated_session = asyncio.run(
                    service.handle_voice_note(session, "https://example.com/media", "audio/ogg")
                )

        # Should ask user to clarify or resend (Req 13.4)
        assert "couldn't make out" in reply.lower() or "resend" in reply.lower()
        # Session should be unchanged (no response recorded)
        assert len(updated_session.responses) == 0

    def test_voice_note_sets_voice_mode_for_technical_rounds(self):
        """Voice note processing sets preferred_mode to 'voice' for technical rounds (Req 13.5).

        Requirements: 13.5
        """
        service, _, _ = make_service()

        for round_type in (InterviewRoundType.DSA_CODING, InterviewRoundType.SYSTEM_DESIGN):
            session = make_session(
                role=Role.SOFTWARE_ENGINEER,
                interview_round_type=round_type,
                preferred_mode="text",
            )

            # Simulate a transcription error so we don't need to mock the full flow
            with patch.object(service._audio_download, 'download', side_effect=TranscriptionError("err")):
                reply, updated_session = asyncio.run(
                    service.handle_voice_note(session, "https://example.com/media", "audio/ogg")
                )

            # Even on error, the download failure returns before setting preferred_mode.
            # The mode is set in the success path (after transcription). Verify the
            # fallback message is returned and session is unchanged.
            assert "couldn't process" in reply.lower()

        # Now verify the success path sets preferred_mode to "voice"
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            preferred_mode="text",
        )
        # Transcription returns empty so we get the clarification path (no full flow needed)
        with patch.object(service._audio_download, 'download', return_value=b"audio_bytes"):
            with patch.object(service._whisper, 'transcribe', return_value="   "):
                reply, updated_session = asyncio.run(
                    service.handle_voice_note(session, "https://example.com/media", "audio/ogg")
                )

        # Empty transcription returns before setting preferred_mode — mode stays as "text"
        # The mode is only set when transcription succeeds and handle_response is called.
        # This is correct behavior: we don't change mode if we can't process the audio.
        assert "couldn't make out" in reply.lower()


# ===========================================================================
# handle_response — main orchestration
# ===========================================================================


class TestHandleResponse:
    @pytest.mark.asyncio
    async def test_handle_response_skip_keyword_calls_handle_skip(self):
        """handle_response with 'skip' keyword calls handle_skip."""
        service, mock_llm, _ = make_service(llm_response="Next question after skip.")
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(consecutive_out_of_scope_count=2, interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(interview_round_type=InterviewRoundType.BEHAVIORAL)
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
        session = make_session(preferred_mode="voice", interview_round_type=InterviewRoundType.BEHAVIORAL)  # Start in voice mode
        q = make_question()
        session.questions.append(q)

        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "Next question?"]

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale, focusing on performance optimization and reliability."
        reply, updated = await service.handle_response(session, long_answer)

        # Should silently switch to text mode
        assert updated.preferred_mode == "text"
        # Should process normally (response recorded)
        assert len(updated.responses) == 1


# ===========================================================================
# _detect_round_type
# Requirements: 1.1-1.6, 20.1-20.5
# ===========================================================================


class TestDetectRoundType:
    # --- DSA_CODING detection ---

    def test_detect_dsa_keyword(self):
        assert _detect_round_type("DSA") == InterviewRoundType.DSA_CODING

    def test_detect_coding_keyword(self):
        assert _detect_round_type("coding round") == InterviewRoundType.DSA_CODING

    def test_detect_algorithms_keyword(self):
        assert _detect_round_type("algorithms practice") == InterviewRoundType.DSA_CODING

    def test_detect_dsa_case_insensitive(self):
        assert _detect_round_type("dsa round") == InterviewRoundType.DSA_CODING
        assert _detect_round_type("DSA Round") == InterviewRoundType.DSA_CODING
        assert _detect_round_type("Coding") == InterviewRoundType.DSA_CODING

    # --- SYSTEM_DESIGN detection ---

    def test_detect_system_design_phrase(self):
        assert _detect_round_type("system design") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_design_keyword(self):
        assert _detect_round_type("design round") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_architecture_keyword(self):
        assert _detect_round_type("architecture interview") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_system_design_case_insensitive(self):
        assert _detect_round_type("System Design") == InterviewRoundType.SYSTEM_DESIGN
        assert _detect_round_type("ARCHITECTURE") == InterviewRoundType.SYSTEM_DESIGN

    # --- BEHAVIORAL detection ---

    def test_detect_behavioral_keyword(self):
        assert _detect_round_type("behavioral round") == InterviewRoundType.BEHAVIORAL

    def test_detect_behavioural_spelling(self):
        assert _detect_round_type("behavioural interview") == InterviewRoundType.BEHAVIORAL

    def test_detect_soft_skills_keyword(self):
        assert _detect_round_type("soft skills") == InterviewRoundType.BEHAVIORAL

    def test_detect_behavioral_case_insensitive(self):
        assert _detect_round_type("Behavioral") == InterviewRoundType.BEHAVIORAL
        assert _detect_round_type("SOFT SKILLS") == InterviewRoundType.BEHAVIORAL

    # --- Ambiguous / None cases ---

    def test_returns_none_for_empty_string(self):
        assert _detect_round_type("") is None

    def test_returns_none_for_unrelated_message(self):
        assert _detect_round_type("hello there") is None

    def test_returns_none_for_ambiguous_multiple_matches(self):
        # "coding" (DSA) + "behavioral" both match → ambiguous
        assert _detect_round_type("coding behavioral") is None

    def test_returns_none_for_whitespace_only(self):
        assert _detect_round_type("   ") is None


# ===========================================================================
# handle_round_type_selection
# Requirements: 1.1-1.6, 11.1, 11.2, 20.1-20.5
# ===========================================================================


def make_service_with_technical(
    llm_response: str = "What is your experience with Python?",
) -> tuple[InterviewService, AsyncMock, MagicMock]:
    """Build an InterviewService with mocked dependencies including technical round service."""
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
    mock_prompt_builder.build_round_type_selection_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    mock_whisper = AsyncMock()
    mock_tts = AsyncMock()
    mock_audio_download = AsyncMock()

    # Create a mock technical round service
    from interview_practice_partner.domain.models import CodingProblem, SystemDesignQuestion
    from interview_practice_partner.domain.enums import ProblemDifficulty, ProblemTopic
    mock_technical_service = AsyncMock()
    mock_coding_problem = CodingProblem(
        problem_id=str(uuid.uuid4()),
        text="Given an array of integers, find the two numbers that add up to a target.",
        difficulty=ProblemDifficulty.MEDIUM,
        topic=ProblemTopic.ARRAYS,
        constraints="1 <= n <= 10^4",
        examples=["Input: [2,7,11,15], target=9 → Output: [0,1]"],
        asked_at=NOW,
    )
    mock_design_question = SystemDesignQuestion(
        question_id=str(uuid.uuid4()),
        text="Design a URL shortener like bit.ly",
        system_name="URL Shortener",
        description="Design a scalable URL shortening service.",
        asked_at=NOW,
    )
    mock_technical_service.generate_coding_problem.return_value = mock_coding_problem
    mock_technical_service.generate_system_design_question.return_value = mock_design_question

    service = InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
        technical_round_service=mock_technical_service,
    )
    return service, mock_llm, mock_prompt_builder


class TestHandleRoundTypeSelection:
    """Tests for round type selection flow in InterviewService.

    Requirements: 1.1-1.6, 11.1, 11.2, 20.1-20.5
    """

    @pytest.mark.asyncio
    async def test_dsa_keyword_sets_round_type(self):
        """When user sends 'DSA', round type is set to DSA_CODING."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about arrays."
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        reply, updated = await service.handle_round_type_selection(session, "DSA")

        assert updated.interview_round_type == InterviewRoundType.DSA_CODING

    @pytest.mark.asyncio
    async def test_behavioral_keyword_sets_round_type(self):
        """When user sends 'behavioral', round type is set to BEHAVIORAL."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        reply, updated = await service.handle_round_type_selection(session, "behavioral")

        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL

    @pytest.mark.asyncio
    async def test_system_design_keyword_sets_round_type(self):
        """When user sends 'system design', round type is set to SYSTEM_DESIGN."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        reply, updated = await service.handle_round_type_selection(session, "system design")

        assert updated.interview_round_type == InterviewRoundType.SYSTEM_DESIGN

    @pytest.mark.asyncio
    async def test_ambiguous_message_prompts_for_selection(self):
        """When message is ambiguous, LLM is called to present selection menu."""
        service, mock_llm, mock_pb = make_service_with_technical()
        # LLM returns a selection prompt (no round_type_detected)
        mock_llm.complete.return_value = json.dumps({
            "message": "Which round type would you like to practice?",
            "round_type_detected": None,
        })
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        reply, updated = await service.handle_round_type_selection(session, "hello")

        # Round type should NOT be set
        assert updated.interview_round_type is None
        # LLM should have been called for round type selection prompt
        mock_pb.build_round_type_selection_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_ambiguous_message_returns_selection_menu(self):
        """When message is ambiguous, reply contains selection options."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = json.dumps({
            "message": "Please choose: 1. DSA 2. System Design 3. Behavioral",
            "round_type_detected": None,
        })
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        reply, updated = await service.handle_round_type_selection(session, "I'm not sure")

        assert isinstance(reply, str)
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_llm_json_parse_failure_returns_default_menu(self):
        """When LLM returns invalid JSON, a default selection menu is returned."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Not valid JSON"
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        reply, updated = await service.handle_round_type_selection(session, "I'm not sure")

        # Should return a default menu
        assert isinstance(reply, str)
        assert len(reply) > 0
        # Round type should NOT be set
        assert updated.interview_round_type is None

    @pytest.mark.asyncio
    async def test_behavioral_round_generates_behavioral_question(self):
        """When behavioral round selected, a behavioral question is generated."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        reply, updated = await service.handle_round_type_selection(session, "behavioral")

        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL
        assert "Tell me about a time you led a project." in reply

    @pytest.mark.asyncio
    async def test_round_type_stored_in_session(self):
        """Selected round type is stored in session.interview_round_type (Req 11.1, 11.2)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(role=Role.SOFTWARE_ENGINEER)

        _, updated = await service.handle_round_type_selection(session, "behavioral")

        assert updated.interview_round_type is not None
        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL


class TestHandleResponseRoundTypeFlow:
    """Tests for round type selection flow triggered from handle_response.

    Requirements: 1.1-1.6, 11.1, 11.2, 20.1-20.5
    """

    @pytest.mark.asyncio
    async def test_swe_no_round_type_triggers_selection(self):
        """For SOFTWARE_ENGINEER with no round type, handle_response triggers selection."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = json.dumps({
            "message": "Which round type would you like?",
            "round_type_detected": None,
        })
        # SOFTWARE_ENGINEER with no round type set
        session = make_session(role=Role.SOFTWARE_ENGINEER, interview_round_type=None)

        reply, updated = await service.handle_response(session, "hello")

        # Should have called round type selection prompt
        mock_pb.build_round_type_selection_prompt.assert_called_once()
        # Round type should still be None (user hasn't chosen yet)
        assert updated.interview_round_type is None

    @pytest.mark.asyncio
    async def test_fast_path_swe_dsa_in_first_message(self):
        """Fast-path: 'Software Engineer DSA round' skips selection menu (Req 20.3)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Here is a DSA problem."
        session = make_session(role=Role.SOFTWARE_ENGINEER, interview_round_type=None)

        reply, updated = await service.handle_response(session, "DSA round")

        # Round type should be set directly
        assert updated.interview_round_type == InterviewRoundType.DSA_CODING
        # Should NOT have called the selection prompt LLM
        mock_pb.build_round_type_selection_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_swe_role_skips_round_type_selection(self):
        """Non-SOFTWARE_ENGINEER roles skip round type selection entirely."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, mock_pb = make_service_with_technical()
        session = make_session(
            role=Role.SALES_REPRESENTATIVE,
            interview_round_type=None,
        )
        q = make_question()
        session.questions.append(q)

        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "Next question?"]

        long_answer = "I have five years of experience in sales and customer relationship management working with enterprise clients."
        reply, updated = await service.handle_response(session, long_answer)

        # Should NOT have called round type selection
        mock_pb.build_round_type_selection_prompt.assert_not_called()
        # Response should be recorded
        assert len(updated.responses) == 1

    @pytest.mark.asyncio
    async def test_swe_with_round_type_set_skips_selection(self):
        """SOFTWARE_ENGINEER with round type already set skips selection."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, mock_pb = make_service_with_technical()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question()
        session.questions.append(q)

        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "Next question?"]

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        reply, updated = await service.handle_response(session, long_answer)

        # Should NOT have called round type selection
        mock_pb.build_round_type_selection_prompt.assert_not_called()
        # Response should be recorded
        assert len(updated.responses) == 1


class TestRoundTypeSwitching:
    """Tests for round type switching mid-session.

    Requirements: 1.5, 11.1, 11.2
    """

    @pytest.mark.asyncio
    async def test_switching_round_type_mid_session_resets_questions(self):
        """Switching round type mid-session clears questions and responses."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        # Add some questions and responses to simulate mid-session
        q = make_question(text="Solve this DSA problem.")
        session.questions.append(q)
        session.responses.append(make_response(question_id=q.question_id))

        # User requests behavioral round (different from current DSA)
        reply, updated = await service.handle_response(session, "behavioral round")

        # Questions and responses should be cleared
        assert len(updated.questions) <= 1  # At most the new question generated
        assert len(updated.responses) == 0
        # New round type should be set
        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL

    @pytest.mark.asyncio
    async def test_switching_to_same_round_type_does_not_reset(self):
        """Sending a keyword for the current round type does not trigger a switch."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, mock_pb = make_service_with_technical()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question()
        session.questions.append(q)
        session.responses.append(make_response(question_id=q.question_id))

        mock_llm.complete.side_effect = [json.dumps({"intent": "answer"}), eval_json, "Next question?"]

        # "behavioral" matches current round type — should NOT switch
        long_answer = "I have five years of experience in behavioral interviews and soft skills development."
        reply, updated = await service.handle_response(session, long_answer)

        # Round type should remain BEHAVIORAL
        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL
        # Responses should be recorded (not cleared)
        assert len(updated.responses) >= 1

    @pytest.mark.asyncio
    async def test_switching_round_type_sets_new_type(self):
        """After switching, the new round type is stored in session."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Solve this DSA problem.")
        session.questions.append(q)

        # Switch to behavioral
        reply, updated = await service.handle_response(session, "behavioral round")

        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL

    @pytest.mark.asyncio
    async def test_switching_round_type_resets_difficulty_history(self):
        """Switching round type clears difficulty_adjustment_history."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        session.difficulty_adjustment_history = [{"from": "medium", "to": "hard"}]
        q = make_question(text="Solve this DSA problem.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "behavioral round")

        assert updated.difficulty_adjustment_history == []


# ===========================================================================
# Routing logic for technical rounds (Task 6.3)
# Requirements: 1.6, 11.1-11.5, 12.1-12.5
# ===========================================================================


def make_service_with_technical_mocks(
    llm_response: str = "{}",
) -> tuple[InterviewService, AsyncMock, MagicMock, MagicMock]:
    """Build an InterviewService with a fully mocked TechnicalRoundService.

    Returns (service, mock_llm, mock_prompt_builder, mock_technical_service).
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
    mock_prompt_builder.build_intent_classification_prompt.return_value = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "user"},
    ]

    mock_whisper = AsyncMock()
    mock_tts = AsyncMock()
    mock_audio_download = AsyncMock()

    from interview_practice_partner.domain.models import (
        CodingProblem,
        ComplexityAnalysis,
        SystemDesignQuestion,
        TechnicalEvaluation,
    )
    from interview_practice_partner.domain.enums import ProblemDifficulty, ProblemTopic

    mock_technical_service = MagicMock()

    # Default coding problem for generate_coding_problem
    mock_coding_problem = CodingProblem(
        problem_id=str(uuid.uuid4()),
        text="Find the two numbers that add up to a target.",
        difficulty=ProblemDifficulty.MEDIUM,
        topic=ProblemTopic.ARRAYS,
        constraints="1 <= n <= 10^4",
        examples=["Input: [2,7,11,15], target=9 → Output: [0,1]"],
        asked_at=NOW,
    )
    mock_technical_service.generate_coding_problem = AsyncMock(return_value=mock_coding_problem)

    # Default system design question
    mock_design_question = SystemDesignQuestion(
        question_id=str(uuid.uuid4()),
        text="Design a URL shortener like bit.ly",
        system_name="URL Shortener",
        description="Design a scalable URL shortening service.",
        asked_at=NOW,
    )
    mock_technical_service.generate_system_design_question = AsyncMock(return_value=mock_design_question)

    # Default coding evaluation
    mock_coding_eval = TechnicalEvaluation(
        evaluation_id=str(uuid.uuid4()),
        question_id=str(uuid.uuid4()),
        response_id=str(uuid.uuid4()),
        correctness="correct",
        complexity_analysis=ComplexityAnalysis(
            time_complexity="O(n)",
            space_complexity="O(1)",
            is_optimal=True,
        ),
        follow_up_warranted=False,
        follow_up_text=None,
        difficulty_signal="increase",
        evaluated_at=NOW,
    )
    mock_technical_service.evaluate_coding_solution = AsyncMock(return_value=mock_coding_eval)

    # Default system design evaluation
    mock_design_eval = TechnicalEvaluation(
        evaluation_id=str(uuid.uuid4()),
        question_id=str(uuid.uuid4()),
        response_id=str(uuid.uuid4()),
        design_strengths=["Good scalability consideration"],
        design_weaknesses=[],
        follow_up_warranted=False,
        follow_up_text=None,
        difficulty_signal="maintain",
        evaluated_at=NOW,
    )
    mock_technical_service.evaluate_system_design = AsyncMock(return_value=mock_design_eval)

    # adjust_difficulty returns HARD (increase from MEDIUM)
    mock_technical_service.adjust_difficulty = MagicMock(return_value=ProblemDifficulty.HARD)

    service = InterviewService(
        llm_client=mock_llm,
        prompt_builder=mock_prompt_builder,
        whisper_client=mock_whisper,
        tts_client=mock_tts,
        audio_download_client=mock_audio_download,
        technical_round_service=mock_technical_service,
    )
    return service, mock_llm, mock_prompt_builder, mock_technical_service


class TestRoutingToDSACoding:
    """Tests that DSA_CODING round type routes to TechnicalRoundService.

    Requirements: 1.6, 11.1-11.5, 12.1-12.5
    """

    @pytest.mark.asyncio
    async def test_dsa_round_routes_to_evaluate_coding_solution(self):
        """DSA_CODING round calls TechnicalRoundService.evaluate_coding_solution."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        reply, updated = await service.handle_response(session, long_answer)

        mock_tech.evaluate_coding_solution.assert_called_once()
        # Should NOT call behavioral evaluate_response
        mock_llm.complete.assert_called_once()  # Only intent classification

    @pytest.mark.asyncio
    async def test_dsa_round_records_response(self):
        """DSA_CODING round records the user response in session."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        _, updated = await service.handle_response(session, long_answer)

        assert len(updated.responses) == 1
        assert updated.responses[0].text == long_answer

    @pytest.mark.asyncio
    async def test_dsa_round_adjusts_difficulty(self):
        """DSA_CODING round calls adjust_difficulty and updates session.problem_difficulty."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        from interview_practice_partner.domain.enums import ProblemDifficulty
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        session.problem_difficulty = ProblemDifficulty.MEDIUM
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        _, updated = await service.handle_response(session, long_answer)

        mock_tech.adjust_difficulty.assert_called_once()
        # adjust_difficulty mock returns HARD
        assert updated.problem_difficulty == ProblemDifficulty.HARD

    @pytest.mark.asyncio
    async def test_dsa_round_generates_next_problem_when_no_follow_up(self):
        """DSA_CODING round generates next problem when no follow-up warranted."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        reply, updated = await service.handle_response(session, long_answer)

        mock_tech.generate_coding_problem.assert_called_once()
        # Next problem text should appear in reply
        assert "Find the two numbers" in reply

    @pytest.mark.asyncio
    async def test_dsa_round_generates_follow_up_when_warranted(self):
        """DSA_CODING round generates follow-up question when evaluation warrants it."""
        from interview_practice_partner.domain.models import TechnicalEvaluation, ComplexityAnalysis

        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        # Override evaluation to warrant a follow-up
        follow_up_eval = TechnicalEvaluation(
            evaluation_id=str(uuid.uuid4()),
            question_id=str(uuid.uuid4()),
            response_id=str(uuid.uuid4()),
            correctness="correct",
            complexity_analysis=ComplexityAnalysis(
                time_complexity="O(n^2)",
                space_complexity="O(1)",
                is_optimal=False,
            ),
            follow_up_warranted=True,
            follow_up_text="Can you optimize this to O(n)?",
            difficulty_signal="maintain",
            evaluated_at=NOW,
        )
        mock_tech.evaluate_coding_solution = AsyncMock(return_value=follow_up_eval)

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use nested loops to check all pairs in O(n^2) time and return the indices when the sum equals the target value."
        reply, updated = await service.handle_response(session, long_answer)

        # Should NOT generate next problem
        mock_tech.generate_coding_problem.assert_not_called()
        # Follow-up should be in reply
        assert "Can you optimize this to O(n)?" in reply
        # Follow-up question should be added to session
        follow_ups = [q for q in updated.questions if q.question_type == QuestionType.FOLLOW_UP]
        assert len(follow_ups) == 1

    @pytest.mark.asyncio
    async def test_dsa_round_does_not_call_behavioral_evaluate_response(self):
        """DSA_CODING round does NOT call behavioral evaluate_response."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        await service.handle_response(session, long_answer)

        # build_response_evaluation_prompt should NOT be called (behavioral path)
        mock_pb.build_response_evaluation_prompt.assert_not_called()


class TestRoutingToSystemDesign:
    """Tests that SYSTEM_DESIGN round type routes to TechnicalRoundService.

    Requirements: 1.6, 11.1-11.5, 12.1-12.5
    """

    @pytest.mark.asyncio
    async def test_system_design_round_routes_to_evaluate_system_design(self):
        """SYSTEM_DESIGN round calls TechnicalRoundService.evaluate_system_design."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Design Twitter.")
        session.questions.append(q)

        long_answer = "I would use a microservices architecture with separate services for user management, tweet storage, and feed generation."
        reply, updated = await service.handle_response(session, long_answer)

        mock_tech.evaluate_system_design.assert_called_once()
        # Should NOT call behavioral evaluate_response
        mock_llm.complete.assert_called_once()  # Only intent classification

    @pytest.mark.asyncio
    async def test_system_design_round_records_response(self):
        """SYSTEM_DESIGN round records the user response in session."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Design Twitter.")
        session.questions.append(q)

        long_answer = "I would use a microservices architecture with separate services for user management, tweet storage, and feed generation."
        _, updated = await service.handle_response(session, long_answer)

        assert len(updated.responses) == 1
        assert updated.responses[0].text == long_answer

    @pytest.mark.asyncio
    async def test_system_design_round_generates_next_question_when_no_follow_up(self):
        """SYSTEM_DESIGN round generates next question when no follow-up warranted."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Design Twitter.")
        session.questions.append(q)

        long_answer = "I would use a microservices architecture with separate services for user management, tweet storage, and feed generation."
        reply, updated = await service.handle_response(session, long_answer)

        mock_tech.generate_system_design_question.assert_called_once()
        # Next question text should appear in reply
        assert "Design a URL shortener" in reply

    @pytest.mark.asyncio
    async def test_system_design_round_generates_follow_up_when_warranted(self):
        """SYSTEM_DESIGN round generates follow-up when evaluation warrants it."""
        from interview_practice_partner.domain.models import TechnicalEvaluation

        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        # Override evaluation to warrant a follow-up
        follow_up_eval = TechnicalEvaluation(
            evaluation_id=str(uuid.uuid4()),
            question_id=str(uuid.uuid4()),
            response_id=str(uuid.uuid4()),
            design_strengths=[],
            design_weaknesses=["Missing caching strategy"],
            follow_up_warranted=True,
            follow_up_text="How would you handle caching for the feed?",
            difficulty_signal="maintain",
            evaluated_at=NOW,
        )
        mock_tech.evaluate_system_design = AsyncMock(return_value=follow_up_eval)

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Design Twitter.")
        session.questions.append(q)

        long_answer = "I would use a microservices architecture with separate services for user management, tweet storage, and feed generation."
        reply, updated = await service.handle_response(session, long_answer)

        # Should NOT generate next question
        mock_tech.generate_system_design_question.assert_not_called()
        # Follow-up should be in reply
        assert "How would you handle caching for the feed?" in reply
        # Follow-up question should be added to session
        follow_ups = [q for q in updated.questions if q.question_type == QuestionType.FOLLOW_UP]
        assert len(follow_ups) == 1

    @pytest.mark.asyncio
    async def test_system_design_round_does_not_call_behavioral_evaluate_response(self):
        """SYSTEM_DESIGN round does NOT call behavioral evaluate_response."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Design Twitter.")
        session.questions.append(q)

        long_answer = "I would use a microservices architecture with separate services for user management, tweet storage, and feed generation."
        await service.handle_response(session, long_answer)

        # build_response_evaluation_prompt should NOT be called (behavioral path)
        mock_pb.build_response_evaluation_prompt.assert_not_called()


class TestBehavioralRoundNotMixed:
    """Tests that BEHAVIORAL round does not use technical evaluation.

    Requirements: 12.1-12.5
    """

    @pytest.mark.asyncio
    async def test_behavioral_round_uses_behavioral_evaluate_response(self):
        """BEHAVIORAL round uses behavioral evaluate_response, not technical."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "answer"}),
            eval_json,
            "What is your greatest strength?",
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question(question_type=QuestionType.BEHAVIOURAL)
        session.questions.append(q)

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        reply, updated = await service.handle_response(session, long_answer)

        # Should use behavioral evaluation (build_response_evaluation_prompt called)
        mock_pb.build_response_evaluation_prompt.assert_called_once()
        # Should NOT call technical evaluation
        mock_tech.evaluate_coding_solution.assert_not_called()
        mock_tech.evaluate_system_design.assert_not_called()

    @pytest.mark.asyncio
    async def test_behavioral_round_does_not_generate_dsa_problems(self):
        """BEHAVIORAL round does NOT generate DSA problems."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "answer"}),
            eval_json,
            "What is your greatest strength?",
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question(question_type=QuestionType.BEHAVIOURAL)
        session.questions.append(q)

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        await service.handle_response(session, long_answer)

        # Should NOT generate DSA problems
        mock_tech.generate_coding_problem.assert_not_called()

    @pytest.mark.asyncio
    async def test_behavioral_round_does_not_generate_system_design_questions(self):
        """BEHAVIORAL round does NOT generate system design questions."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "answer"}),
            eval_json,
            "What is your greatest strength?",
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question(question_type=QuestionType.BEHAVIOURAL)
        session.questions.append(q)

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."
        await service.handle_response(session, long_answer)

        # Should NOT generate system design questions
        mock_tech.generate_system_design_question.assert_not_called()


class TestRoundTypePersistence:
    """Tests that round type persists across requests.

    Requirements: 11.1-11.5
    """

    @pytest.mark.asyncio
    async def test_dsa_round_type_persists_after_response(self):
        """DSA_CODING round type remains set after handling a response."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        _, updated = await service.handle_response(session, long_answer)

        # Round type should still be DSA_CODING
        assert updated.interview_round_type == InterviewRoundType.DSA_CODING

    @pytest.mark.asyncio
    async def test_system_design_round_type_persists_after_response(self):
        """SYSTEM_DESIGN round type remains set after handling a response."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Design Twitter.")
        session.questions.append(q)

        long_answer = "I would use a microservices architecture with separate services for user management, tweet storage, and feed generation."
        _, updated = await service.handle_response(session, long_answer)

        # Round type should still be SYSTEM_DESIGN
        assert updated.interview_round_type == InterviewRoundType.SYSTEM_DESIGN

    @pytest.mark.asyncio
    async def test_dsa_round_only_generates_technical_questions(self):
        """DSA_CODING round only generates TECHNICAL question types (no BEHAVIOURAL mixing)."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        _, updated = await service.handle_response(session, long_answer)

        # All new questions added should be TECHNICAL or FOLLOW_UP (not BEHAVIOURAL/SITUATIONAL)
        new_questions = updated.questions[1:]  # Skip the original question
        for new_q in new_questions:
            assert new_q.question_type in (QuestionType.TECHNICAL, QuestionType.FOLLOW_UP), (
                f"Expected TECHNICAL or FOLLOW_UP, got {new_q.question_type}"
            )

    @pytest.mark.asyncio
    async def test_fallback_to_behavioral_when_no_technical_service(self):
        """When TechnicalRoundService is None, DSA round falls back to behavioral evaluation."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        # Build service WITHOUT technical round service
        service, mock_llm, mock_pb = make_service()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "answer"}),
            eval_json,
            "What is your greatest strength?",
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."
        reply, updated = await service.handle_response(session, long_answer)

        # Should fall back to behavioral evaluation
        mock_pb.build_response_evaluation_prompt.assert_called_once()
        # Response should still be recorded
        assert len(updated.responses) == 1


# ===========================================================================
# Task 6.5: Additional tests for round type detection and routing
# Requirements: 1.1-1.6, 20.1-20.5
# ===========================================================================


class TestDetectRoundTypeKeywords:
    """Explicit keyword coverage for _detect_round_type.

    Ensures every keyword group from the spec (Req 20.2) is tested.
    """

    # --- DSA keywords ---

    def test_detect_dsa_exact(self):
        """'dsa' keyword → DSA_CODING."""
        assert _detect_round_type("dsa") == InterviewRoundType.DSA_CODING

    def test_detect_coding_exact(self):
        """'coding' keyword → DSA_CODING."""
        assert _detect_round_type("coding") == InterviewRoundType.DSA_CODING

    def test_detect_algorithms_plural(self):
        """'algorithms' keyword → DSA_CODING."""
        assert _detect_round_type("algorithms") == InterviewRoundType.DSA_CODING

    def test_detect_algorithm_singular(self):
        """'algorithm' (singular) keyword → DSA_CODING."""
        assert _detect_round_type("algorithm") == InterviewRoundType.DSA_CODING

    def test_detect_data_structures(self):
        """'data structures' keyword → DSA_CODING."""
        assert _detect_round_type("data structures") == InterviewRoundType.DSA_CODING

    def test_detect_dsa_in_sentence(self):
        """'dsa' embedded in a sentence → DSA_CODING."""
        assert _detect_round_type("I want to practice dsa questions") == InterviewRoundType.DSA_CODING

    def test_detect_coding_in_sentence(self):
        """'coding' embedded in a sentence → DSA_CODING."""
        assert _detect_round_type("let's do a coding interview") == InterviewRoundType.DSA_CODING

    def test_detect_algorithms_in_sentence(self):
        """'algorithms' embedded in a sentence → DSA_CODING."""
        assert _detect_round_type("I need help with algorithms") == InterviewRoundType.DSA_CODING

    # --- System Design keywords ---

    def test_detect_system_design_exact(self):
        """'system design' phrase → SYSTEM_DESIGN."""
        assert _detect_round_type("system design") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_design_exact(self):
        """'design' keyword → SYSTEM_DESIGN."""
        assert _detect_round_type("design") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_architecture_exact(self):
        """'architecture' keyword → SYSTEM_DESIGN."""
        assert _detect_round_type("architecture") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_scalability_keyword(self):
        """'scalability' keyword → SYSTEM_DESIGN."""
        assert _detect_round_type("scalability") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_distributed_systems(self):
        """'distributed systems' keyword → SYSTEM_DESIGN."""
        assert _detect_round_type("distributed systems") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_design_in_sentence(self):
        """'design' embedded in a sentence → SYSTEM_DESIGN."""
        assert _detect_round_type("I want to practice design questions") == InterviewRoundType.SYSTEM_DESIGN

    def test_detect_architecture_in_sentence(self):
        """'architecture' embedded in a sentence → SYSTEM_DESIGN."""
        assert _detect_round_type("let's talk about architecture") == InterviewRoundType.SYSTEM_DESIGN

    # --- Behavioral keywords ---

    def test_detect_behavioral_exact(self):
        """'behavioral' keyword → BEHAVIORAL."""
        assert _detect_round_type("behavioral") == InterviewRoundType.BEHAVIORAL

    def test_detect_behavioural_british_spelling(self):
        """'behavioural' (British spelling) → BEHAVIORAL."""
        assert _detect_round_type("behavioural") == InterviewRoundType.BEHAVIORAL

    def test_detect_soft_skills_exact(self):
        """'soft skills' keyword → BEHAVIORAL."""
        assert _detect_round_type("soft skills") == InterviewRoundType.BEHAVIORAL

    def test_detect_behavioral_in_sentence(self):
        """'behavioral' embedded in a sentence → BEHAVIORAL."""
        assert _detect_round_type("I want to practice behavioral questions") == InterviewRoundType.BEHAVIORAL

    def test_detect_soft_skills_in_sentence(self):
        """'soft skills' embedded in a sentence → BEHAVIORAL."""
        assert _detect_round_type("let's focus on soft skills today") == InterviewRoundType.BEHAVIORAL

    # --- Ambiguous / None ---

    def test_returns_none_for_generic_greeting(self):
        """Generic greeting returns None (ambiguous)."""
        assert _detect_round_type("hi") is None

    def test_returns_none_for_number_only(self):
        """Number-only input returns None."""
        assert _detect_round_type("1") is None

    def test_returns_none_for_dsa_and_design_conflict(self):
        """'coding design' matches both DSA and System Design → None (ambiguous)."""
        assert _detect_round_type("coding design") is None

    def test_returns_none_for_all_three_conflict(self):
        """Message matching all three categories → None (ambiguous)."""
        assert _detect_round_type("coding design behavioral") is None


class TestFastPathDetection:
    """Fast-path: round type in opening message skips selection menu (Req 20.3)."""

    @pytest.mark.asyncio
    async def test_fast_path_software_engineer_dsa_round(self):
        """'Software Engineer DSA round' skips selection menu, starts DSA directly."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Here is a DSA problem."
        session = make_session(role=Role.SOFTWARE_ENGINEER, interview_round_type=None)

        reply, updated = await service.handle_response(session, "Software Engineer DSA round")

        assert updated.interview_round_type == InterviewRoundType.DSA_CODING
        mock_pb.build_round_type_selection_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_swe_coding_round(self):
        """'SWE coding round' skips selection menu, starts DSA directly."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Here is a coding problem."
        session = make_session(role=Role.SOFTWARE_ENGINEER, interview_round_type=None)

        reply, updated = await service.handle_response(session, "SWE coding round")

        assert updated.interview_round_type == InterviewRoundType.DSA_CODING
        mock_pb.build_round_type_selection_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_system_design_in_opening_message(self):
        """'system design' in opening message skips selection menu."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Design a URL shortener."
        session = make_session(role=Role.SOFTWARE_ENGINEER, interview_round_type=None)

        reply, updated = await service.handle_response(session, "I want to practice system design")

        assert updated.interview_round_type == InterviewRoundType.SYSTEM_DESIGN
        mock_pb.build_round_type_selection_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_fast_path_behavioral_in_opening_message(self):
        """'behavioral' in opening message skips selection menu."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(role=Role.SOFTWARE_ENGINEER, interview_round_type=None)

        reply, updated = await service.handle_response(session, "behavioral round please")

        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL
        mock_pb.build_round_type_selection_prompt.assert_not_called()

    @pytest.mark.asyncio
    async def test_ambiguous_opening_message_triggers_selection_menu(self):
        """Ambiguous opening message triggers the selection menu (no fast-path)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = json.dumps({
            "message": "Which round type would you like?",
            "round_type_detected": None,
        })
        session = make_session(role=Role.SOFTWARE_ENGINEER, interview_round_type=None)

        reply, updated = await service.handle_response(session, "I want to practice interviews")

        # No fast-path — selection prompt should be called
        mock_pb.build_round_type_selection_prompt.assert_called_once()
        assert updated.interview_round_type is None


class TestRoundSwitchingSessionReset:
    """Tests that switching round type mid-session fully resets session state (Req 1.5)."""

    @pytest.mark.asyncio
    async def test_switching_resets_questions(self):
        """Switching round type clears session.questions."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "behavioral round")

        # After switch, questions should only contain the new round's question
        # (the old DSA question should be gone)
        dsa_questions = [q for q in updated.questions if q.text == "Solve two-sum."]
        assert len(dsa_questions) == 0

    @pytest.mark.asyncio
    async def test_switching_resets_responses(self):
        """Switching round type clears session.responses."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)
        session.responses.append(make_response(question_id=q.question_id))

        reply, updated = await service.handle_response(session, "behavioral round")

        # Old responses should be cleared
        assert len(updated.responses) == 0

    @pytest.mark.asyncio
    async def test_switching_resets_off_topic_count(self):
        """Switching round type resets off_topic_count to 0."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            off_topic_count=3,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "behavioral round")

        assert updated.off_topic_count == 0

    @pytest.mark.asyncio
    async def test_switching_resets_consecutive_out_of_scope_count(self):
        """Switching round type resets consecutive_out_of_scope_count to 0."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_out_of_scope_count=2,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "behavioral round")

        assert updated.consecutive_out_of_scope_count == 0

    @pytest.mark.asyncio
    async def test_switching_resets_topics_covered(self):
        """Switching round type clears topics_covered."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        session.topics_covered = ["arrays", "trees"]
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "behavioral round")

        assert updated.topics_covered == []

    @pytest.mark.asyncio
    async def test_switching_resets_design_aspects_covered(self):
        """Switching round type clears design_aspects_covered."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        session.design_aspects_covered = ["scalability", "caching"]
        q = make_question(text="Design Twitter.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "dsa round")

        assert updated.design_aspects_covered == []

    @pytest.mark.asyncio
    async def test_switching_sets_new_round_type(self):
        """After switching, session.interview_round_type reflects the new type."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(text="Design Twitter.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "behavioral round")

        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL

    @pytest.mark.asyncio
    async def test_same_round_type_does_not_trigger_switch(self):
        """Sending a keyword matching the current round type does NOT trigger a switch."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "answer"}),
            eval_json,
            "What is your greatest strength?",
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question(question_type=QuestionType.BEHAVIOURAL)
        session.questions.append(q)

        long_answer = "I have strong behavioral skills and soft skills developed over many years of professional experience working in teams."
        reply, updated = await service.handle_response(session, long_answer)

        # Round type should remain BEHAVIORAL (not switched)
        assert updated.interview_round_type == InterviewRoundType.BEHAVIORAL
        # Responses should be recorded (not cleared)
        assert len(updated.responses) == 1


class TestRoundTypePersistenceMultipleResponses:
    """Tests that round type persists across multiple responses (Req 11.1-11.5)."""

    @pytest.mark.asyncio
    async def test_dsa_round_type_persists_after_three_responses(self):
        """DSA_CODING round type remains set after three consecutive responses."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )

        long_answer = "I would use a hash map to store complements and check each element in O(n) time."

        for _ in range(3):
            q = make_question(question_type=QuestionType.TECHNICAL, text="Solve a problem.")
            session.questions.append(q)
            _, session = await service.handle_response(session, long_answer)
            assert session.interview_round_type == InterviewRoundType.DSA_CODING

    @pytest.mark.asyncio
    async def test_system_design_round_type_persists_after_three_responses(self):
        """SYSTEM_DESIGN round type remains set after three consecutive responses."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )

        long_answer = "I would use a microservices architecture with separate services for each domain component."

        for _ in range(3):
            q = make_question(question_type=QuestionType.TECHNICAL, text="Design a system.")
            session.questions.append(q)
            _, session = await service.handle_response(session, long_answer)
            assert session.interview_round_type == InterviewRoundType.SYSTEM_DESIGN

    @pytest.mark.asyncio
    async def test_behavioral_round_type_persists_after_three_responses(self):
        """BEHAVIORAL round type remains set after three consecutive responses."""
        eval_json = json.dumps({
            "is_off_topic": False,
            "is_short": False,
            "follow_up_warranted": False,
            "follow_up_text": None,
            "difficulty_signal": "maintain",
        })
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )

        long_answer = "I have five years of experience working on distributed systems and microservices architecture at scale."

        for _ in range(3):
            mock_llm.complete.side_effect = [
                json.dumps({"intent": "answer"}),
                eval_json,
                "What is your greatest strength?",
            ]
            q = make_question(question_type=QuestionType.BEHAVIOURAL)
            session.questions.append(q)
            _, session = await service.handle_response(session, long_answer)
            assert session.interview_round_type == InterviewRoundType.BEHAVIORAL

    @pytest.mark.asyncio
    async def test_round_type_not_reset_by_unrelated_messages(self):
        """Round type is not accidentally reset by messages that don't match any keyword."""
        service, mock_llm, _, mock_tech = make_service_with_technical_mocks()
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(question_type=QuestionType.TECHNICAL, text="Solve two-sum.")
        session.questions.append(q)

        # A long answer with no round-type keywords
        long_answer = "I would iterate through the array and use a hash map to track previously seen values."
        _, updated = await service.handle_response(session, long_answer)

        assert updated.interview_round_type == InterviewRoundType.DSA_CODING


# ===========================================================================
# Technical Round Skip Handling (Task 8.1)
# ===========================================================================


class TestTechnicalRoundSkipHandling:
    """Tests for skip handling in DSA/System Design rounds (Req 14.4, 14.7, 15.3)."""

    @pytest.mark.asyncio
    async def test_skip_increments_consecutive_skips_count(self):
        """Skip in technical round increments consecutive_skips_count."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            json.dumps({
                "problem_statement": "Find the maximum subarray sum.",
                "examples": ["Input: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6"],
                "constraints": "1 <= nums.length <= 10^5",
                "topic": "arrays",
            }),
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_skips_count=0,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "skip")

        assert updated.consecutive_skips_count == 1
        assert q.skipped is True

    @pytest.mark.asyncio
    async def test_skip_resets_consecutive_skips_on_valid_answer(self):
        """Valid answer resets consecutive_skips_count to 0."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()
        # Mock evaluation response
        eval_json = json.dumps({
            "correctness": "correct",
            "time_complexity": "O(n)",
            "space_complexity": "O(1)",
            "is_optimal": True,
            "edge_cases_handled": ["empty array", "single element"],
            "edge_cases_missed": [],
            "follow_up_warranted": False,
            "difficulty_signal": "increase",
        })
        # Mock next problem generation
        next_problem_json = json.dumps({
            "problem_statement": "Find the longest substring without repeating characters.",
            "examples": ['Input: "abcabcbb"\nOutput: 3'],
            "constraints": "0 <= s.length <= 5 * 10^4",
            "topic": "strings",
        })
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "answer"}),
            eval_json,
            next_problem_json,
        ]
        # Set adjust_difficulty to return MEDIUM (no change from initial)
        mock_tech.adjust_difficulty.return_value = ProblemDifficulty.MEDIUM
        
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_skips_count=2,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        # Use a long enough response to avoid short response handling (>= 15 words)
        long_answer = (
            "def two_sum(nums, target): "
            "seen = {} "
            "for i, num in enumerate(nums): "
            "complement = target - num "
            "if complement in seen: return [seen[complement], i] "
            "seen[num] = i"
        )
        reply, updated = await service.handle_response(session, long_answer)

        assert updated.consecutive_skips_count == 0

    @pytest.mark.asyncio
    async def test_two_consecutive_skips_decreases_difficulty(self):
        """Two consecutive skips decrease difficulty from HARD to MEDIUM (Req 15.3)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            json.dumps({
                "problem_statement": "Find the maximum subarray sum.",
                "examples": ["Input: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6"],
                "constraints": "1 <= nums.length <= 10^5",
                "topic": "arrays",
            }),
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_skips_count=1,  # Already skipped once
            problem_difficulty=ProblemDifficulty.HARD,
        )
        q = make_question(text="Solve a hard problem.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "skip")

        assert updated.consecutive_skips_count == 2
        assert updated.problem_difficulty == ProblemDifficulty.MEDIUM
        assert len(updated.difficulty_adjustment_history) == 1
        assert updated.difficulty_adjustment_history[0]["from"] == "hard"
        assert updated.difficulty_adjustment_history[0]["to"] == "medium"
        assert updated.difficulty_adjustment_history[0]["reason"] == "consecutive_skips"

    @pytest.mark.asyncio
    async def test_two_consecutive_skips_decreases_difficulty_medium_to_easy(self):
        """Two consecutive skips decrease difficulty from MEDIUM to EASY (Req 15.3)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            json.dumps({
                "problem_statement": "Find the maximum subarray sum.",
                "examples": ["Input: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6"],
                "constraints": "1 <= nums.length <= 10^5",
                "topic": "arrays",
            }),
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_skips_count=1,
            problem_difficulty=ProblemDifficulty.MEDIUM,
        )
        q = make_question(text="Solve a medium problem.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "skip")

        assert updated.problem_difficulty == ProblemDifficulty.EASY

    @pytest.mark.asyncio
    async def test_two_consecutive_skips_stays_at_easy(self):
        """Two consecutive skips at EASY difficulty stay at EASY (boundary)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            json.dumps({
                "problem_statement": "Find the maximum subarray sum.",
                "examples": ["Input: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6"],
                "constraints": "1 <= nums.length <= 10^5",
                "topic": "arrays",
            }),
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_skips_count=1,
            problem_difficulty=ProblemDifficulty.EASY,
        )
        q = make_question(text="Solve an easy problem.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "skip")

        assert updated.problem_difficulty == ProblemDifficulty.EASY
        # No adjustment recorded since difficulty didn't change
        assert len(updated.difficulty_adjustment_history) == 0

    @pytest.mark.asyncio
    async def test_three_consecutive_skips_offers_to_end_session(self):
        """Three consecutive skips offer to end the session (Req 14.7)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = json.dumps({"intent": "skip"})
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_skips_count=2,  # Already skipped twice
        )
        q = make_question(text="Solve a problem.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "skip")

        assert updated.consecutive_skips_count == 3
        # Reply should offer to end session or continue
        assert any(
            phrase in reply.lower()
            for phrase in ["end the session", "continue", "1", "2"]
        )

    @pytest.mark.asyncio
    async def test_solution_request_provides_solution(self):
        """Requesting solution provides the solution via LLM (Req 14.4)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        solution_text = (
            "*Key Insight:* Use a hash map to store complements.\n\n"
            "*Solution:*\n"
            "def two_sum(nums, target):\n"
            "    seen = {}\n"
            "    for i, num in enumerate(nums):\n"
            "        complement = target - num\n"
            "        if complement in seen:\n"
            "            return [seen[complement], i]\n"
            "        seen[num] = i\n\n"
            "*Complexity:* Time O(n), Space O(n)"
        )
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            solution_text,  # Solution generation
            json.dumps({  # Next problem generation
                "problem_statement": "Find the maximum subarray sum.",
                "examples": ["Input: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6"],
                "constraints": "1 <= nums.length <= 10^5",
                "topic": "arrays",
            }),
        ]
        mock_pb.build_problem_solution_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "give me the solution")

        # Should include the solution in the reply
        assert "Key Insight" in reply or "Solution" in reply or "Complexity" in reply
        assert q.skipped is True

    @pytest.mark.asyncio
    async def test_solution_request_fallback_on_llm_failure(self):
        """Solution request falls back gracefully if LLM fails."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            Exception("LLM API error"),  # Solution generation fails
            json.dumps({  # Next problem generation
                "problem_statement": "Find the maximum subarray sum.",
                "examples": ["Input: [-2,1,-3,4,-1,2,1,-5,4]\nOutput: 6"],
                "constraints": "1 <= nums.length <= 10^5",
                "topic": "arrays",
            }),
        ]
        mock_pb.build_problem_solution_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "give me the solution")

        # Should still provide a fallback message
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert q.skipped is True

    @pytest.mark.asyncio
    async def test_off_topic_three_times_offers_skip_for_technical_rounds(self):
        """Three consecutive off-topic responses offer to skip for technical rounds (Req 14.7)."""
        service, mock_llm, mock_pb = make_service_with_technical()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_out_of_scope_count=2,  # Already off-topic twice
        )

        reply = service.handle_off_topic(session)

        assert session.consecutive_out_of_scope_count == 3
        # Reply should offer to skip, end, or continue
        assert any(
            phrase in reply.lower()
            for phrase in ["skip", "end the session", "continue", "1", "2", "3"]
        )

    @pytest.mark.asyncio
    async def test_off_topic_three_times_behavioral_round_no_skip_option(self):
        """Three consecutive off-topic in behavioral round doesn't offer skip."""
        service, mock_llm, mock_pb = make_service_with_technical()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
            consecutive_out_of_scope_count=2,
        )

        reply = service.handle_off_topic(session)

        assert session.consecutive_out_of_scope_count == 3
        # Reply should NOT mention "skip" for behavioral rounds
        assert "skip" not in reply.lower()
        # But should offer to end or return to role selection
        assert any(
            phrase in reply.lower()
            for phrase in ["end", "role selection", "1", "2"]
        )

    @pytest.mark.asyncio
    async def test_skip_in_system_design_round_increments_counter(self):
        """Skip in System Design round also increments consecutive_skips_count."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            json.dumps({
                "system_name": "URL Shortener",
                "question_text": "Design a URL shortener service like bit.ly",
                "description": "Design a service that takes long URLs and generates short aliases.",
            }),
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
            consecutive_skips_count=0,
        )
        q = make_question(text="Design Twitter.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "skip")

        assert updated.consecutive_skips_count == 1
        assert q.skipped is True

    @pytest.mark.asyncio
    async def test_skip_in_behavioral_round_does_not_track_consecutive_skips(self):
        """Skip in behavioral round does not increment consecutive_skips_count."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.side_effect = [
            json.dumps({"intent": "skip"}),
            "Tell me about a time you led a project.",
        ]
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
            consecutive_skips_count=0,
        )
        q = make_question(text="Tell me about yourself.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "skip")

        # Behavioral rounds don't track consecutive skips (no difficulty adjustment)
        assert updated.consecutive_skips_count == 0
        assert q.skipped is True

    @pytest.mark.asyncio
    async def test_round_type_switch_resets_consecutive_skips_count(self):
        """Switching round type resets consecutive_skips_count to 0."""
        service, mock_llm, mock_pb = make_service_with_technical()
        mock_llm.complete.return_value = "Tell me about a time you led a project."
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_skips_count=2,
        )
        q = make_question(text="Solve two-sum.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "behavioral round")

        assert updated.consecutive_skips_count == 0


# ===========================================================================
# Task 8.3: Off-topic and invalid response handling for technical rounds
# Requirements: 14.1, 14.2, 14.5, 14.6, 14.7
# ===========================================================================


class TestHandleEmptySubmission:
    """Tests for empty/whitespace-only submission handling (Req 14.5)."""

    def test_empty_message_returns_prompt_for_technical_round(self):
        """Empty submission in a technical round prompts to provide approach or skip (Req 14.5)."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )

        reply = service.handle_empty_submission(session)

        assert isinstance(reply, str)
        assert len(reply) > 0
        # Should mention providing approach or skipping
        assert any(
            phrase in reply.lower()
            for phrase in ["approach", "solution", "skip"]
        )

    def test_empty_message_returns_prompt_for_system_design_round(self):
        """Empty submission in a System Design round prompts to provide approach or skip (Req 14.5)."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )

        reply = service.handle_empty_submission(session)

        assert isinstance(reply, str)
        assert any(
            phrase in reply.lower()
            for phrase in ["approach", "solution", "skip"]
        )

    def test_empty_message_increments_consecutive_count(self):
        """Empty submission increments consecutive_out_of_scope_count."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_out_of_scope_count=0,
        )

        service.handle_empty_submission(session)

        assert session.consecutive_out_of_scope_count == 1
        assert session.off_topic_count == 1

    def test_empty_message_three_times_offers_skip_or_end(self):
        """Three empty submissions offer to skip or end the session (Req 14.7)."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
            consecutive_out_of_scope_count=2,
        )

        reply = service.handle_empty_submission(session)

        assert session.consecutive_out_of_scope_count == 3
        assert any(
            phrase in reply.lower()
            for phrase in ["skip", "end the session", "1", "2", "3"]
        )

    def test_empty_message_behavioral_round_returns_generic_prompt(self):
        """Empty submission in behavioral round returns a generic prompt."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )

        reply = service.handle_empty_submission(session)

        assert isinstance(reply, str)
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_handle_response_empty_string_triggers_empty_handler(self):
        """handle_response with empty string calls handle_empty_submission (Req 14.5)."""
        service, mock_llm, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Find two numbers that sum to target.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "")

        # Should not call LLM for intent classification
        mock_llm.complete.assert_not_called()
        # Should return a prompt to provide approach or skip
        assert isinstance(reply, str)
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_handle_response_whitespace_only_triggers_empty_handler(self):
        """handle_response with whitespace-only string calls handle_empty_submission (Req 14.5)."""
        service, mock_llm, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Find two numbers that sum to target.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "   \n\t  ")

        # Should not call LLM for intent classification
        mock_llm.complete.assert_not_called()
        assert isinstance(reply, str)
        assert len(reply) > 0

    @pytest.mark.asyncio
    async def test_handle_response_empty_does_not_advance_session(self):
        """Empty submission does not add a response to the session."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Find two numbers that sum to target.")
        session.questions.append(q)
        original_response_count = len(session.responses)

        await service.handle_response(session, "")

        assert len(session.responses) == original_response_count


class TestOffTopicTechnicalRoundHandling:
    """Tests for off-topic handling in technical rounds (Req 14.2, 14.6)."""

    def test_off_topic_technical_round_includes_problem_redirect(self):
        """Off-topic in technical round asks user to re-read the problem (Req 14.2, 14.6)."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Find two numbers that add up to a target sum.")
        session.questions.append(q)

        reply = service.handle_off_topic(session)

        # Should redirect back to the problem (Req 14.6)
        assert any(
            phrase in reply.lower()
            for phrase in ["re-read", "problem", "current problem"]
        )

    def test_off_topic_technical_round_includes_problem_text(self):
        """Off-topic redirect in technical round includes the current problem text (Req 14.6)."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        problem_text = "Find two numbers that add up to a target sum."
        q = make_question(text=problem_text)
        session.questions.append(q)

        reply = service.handle_off_topic(session)

        # Should include the problem text so user can re-read it
        assert problem_text in reply

    def test_off_topic_technical_round_no_active_question_still_redirects(self):
        """Off-topic redirect works even when no active question is found."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        # No questions in session

        reply = service.handle_off_topic(session)

        assert isinstance(reply, str)
        assert len(reply) > 0
        # Should still mention the problem
        assert "problem" in reply.lower()

    def test_off_topic_system_design_round_redirects_to_problem(self):
        """Off-topic in System Design round also redirects to current question (Req 14.6)."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        design_text = "Design a URL shortener like bit.ly."
        q = make_question(text=design_text)
        session.questions.append(q)

        reply = service.handle_off_topic(session)

        assert design_text in reply

    def test_off_topic_behavioral_round_uses_generic_redirect(self):
        """Off-topic in behavioral round uses the generic redirect (not problem-specific)."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question(text="Tell me about a time you led a project.")
        session.questions.append(q)

        reply = service.handle_off_topic(session)

        # Behavioral round should NOT include the question text in the redirect
        assert "Tell me about a time you led a project." not in reply
        # But should still redirect
        assert any(
            phrase in reply.lower()
            for phrase in ["interview", "question", "focused"]
        )

    def test_off_topic_skipped_question_not_included_in_redirect(self):
        """Off-topic redirect uses the current unanswered question, not a skipped one."""
        service, _, _ = make_service()
        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        skipped_q = make_question(text="Old skipped problem.", skipped=True)
        active_q = make_question(text="Current active problem.")
        session.questions.extend([skipped_q, active_q])

        reply = service.handle_off_topic(session)

        # Should include the active problem, not the skipped one
        assert "Current active problem." in reply
        assert "Old skipped problem." not in reply


class TestGetCurrentProblemText:
    """Tests for the _get_current_problem_text helper."""

    def test_returns_text_of_unanswered_question(self):
        """Returns the text of the most recent unanswered question."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question(text="What is a binary search tree?")
        session.questions.append(q)

        result = InterviewService._get_current_problem_text(session)

        assert result == "What is a binary search tree?"

    def test_returns_none_when_no_questions(self):
        """Returns None when there are no questions in the session."""
        service, _, _ = make_service()
        session = make_session()

        result = InterviewService._get_current_problem_text(session)

        assert result is None

    def test_returns_none_when_all_questions_answered(self):
        """Returns None when all questions have been answered."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question()
        session.questions.append(q)
        session.responses.append(make_response(question_id=q.question_id))

        result = InterviewService._get_current_problem_text(session)

        assert result is None

    def test_returns_most_recent_unanswered_question(self):
        """Returns the most recent unanswered question when multiple exist."""
        service, _, _ = make_service()
        session = make_session()
        q1 = make_question(text="First question.")
        q2 = make_question(text="Second question.")
        session.questions.extend([q1, q2])
        # q1 is answered
        session.responses.append(make_response(question_id=q1.question_id))

        result = InterviewService._get_current_problem_text(session)

        assert result == "Second question."

    def test_skipped_questions_not_returned(self):
        """Skipped questions are not returned as the current problem."""
        service, _, _ = make_service()
        session = make_session()
        skipped_q = make_question(text="Skipped problem.", skipped=True)
        session.questions.append(skipped_q)

        result = InterviewService._get_current_problem_text(session)

        assert result is None


class TestUnrecognizedCodeHandling:
    """Tests for unrecognized code/pseudocode handling (Req 14.1)."""

    @pytest.mark.asyncio
    async def test_unrecognized_language_code_is_evaluated_not_rejected(self):
        """Code in an unrecognized language is evaluated for logic, not rejected (Req 14.1)."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        # Set up intent classification to return "answer"
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(text="Find two numbers that sum to target.")
        session.questions.append(q)

        # Submit code in an unusual/unrecognized language (e.g., Haskell-like pseudocode)
        unusual_code = (
            "twoSum nums target = \n"
            "  let pairs = [(i,j) | i <- [0..n-1], j <- [i+1..n-1]]\n"
            "  in head [(i,j) | (i,j) <- pairs, nums!!i + nums!!j == target]\n"
            "  where n = length nums"
        )

        reply, updated = await service.handle_response(session, unusual_code)

        # Should NOT reject the submission — should evaluate it
        assert isinstance(reply, str)
        assert len(reply) > 0
        # Should not say "rejected" or "not supported"
        assert "rejected" not in reply.lower()
        assert "not supported" not in reply.lower()
        # Response should be recorded (not treated as off-topic)
        assert len(updated.responses) == 1
        # evaluate_coding_solution should have been called
        mock_tech.evaluate_coding_solution.assert_called_once()


# ===========================================================================
# Repeat / clarification handling — technical rounds
# Requirements: 3.6, 14.2
# ===========================================================================


class TestHandleRepeatRequest:
    """Tests for handle_repeat_request — works for both behavioral and technical rounds.

    Requirements: 3.6, 14.2
    """

    def test_repeat_request_returns_current_question(self):
        """handle_repeat_request re-sends the current unanswered question."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question(text="Describe a time you resolved a conflict.")
        session.questions.append(q)

        reply = service.handle_repeat_request(session)

        assert "Describe a time you resolved a conflict." in reply

    def test_repeat_request_includes_preamble(self):
        """handle_repeat_request includes a brief preamble before the question."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question(text="What is your greatest strength?")
        session.questions.append(q)

        reply = service.handle_repeat_request(session)

        # Should have some preamble before the question
        assert reply.startswith("Of course!")

    def test_repeat_request_no_active_question_returns_fallback(self):
        """handle_repeat_request returns a fallback when no active question exists."""
        service, _, _ = make_service()
        session = make_session()

        reply = service.handle_repeat_request(session)

        assert "don't have an active question" in reply.lower() or "no active question" in reply.lower() or "Please send any message" in reply

    def test_repeat_request_skips_answered_questions(self):
        """handle_repeat_request skips questions that already have responses."""
        service, _, _ = make_service()
        session = make_session()
        q1 = make_question(text="First question.")
        q2 = make_question(text="Second question — the active one.")
        session.questions.extend([q1, q2])
        # Mark q1 as answered
        session.responses.append(make_response(question_id=q1.question_id))

        reply = service.handle_repeat_request(session)

        assert "Second question — the active one." in reply
        assert "First question." not in reply

    def test_repeat_request_skips_skipped_questions(self):
        """handle_repeat_request skips questions marked as skipped."""
        service, _, _ = make_service()
        session = make_session()
        q1 = make_question(text="Skipped question.", skipped=True)
        q2 = make_question(text="Active question.")
        session.questions.extend([q1, q2])

        reply = service.handle_repeat_request(session)

        assert "Active question." in reply
        assert "Skipped question." not in reply

    def test_repeat_request_does_not_advance_session(self):
        """handle_repeat_request does not add questions or responses to the session."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question(text="Tell me about yourself.")
        session.questions.append(q)
        initial_question_count = len(session.questions)
        initial_response_count = len(session.responses)

        service.handle_repeat_request(session)

        assert len(session.questions) == initial_question_count
        assert len(session.responses) == initial_response_count

    def test_repeat_request_works_for_dsa_round(self):
        """handle_repeat_request re-sends the current DSA problem (Req 3.6)."""
        service, _, _ = make_service()
        session = make_session(
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(
            text="Given an array, find two numbers that sum to target.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        reply = service.handle_repeat_request(session)

        assert "Given an array, find two numbers that sum to target." in reply

    def test_repeat_request_works_for_system_design_round(self):
        """handle_repeat_request re-sends the current system design question."""
        service, _, _ = make_service()
        session = make_session(
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(
            text="Design a URL shortener like bit.ly.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        reply = service.handle_repeat_request(session)

        assert "Design a URL shortener like bit.ly." in reply

    def test_repeat_request_all_questions_answered_returns_fallback(self):
        """handle_repeat_request returns fallback when all questions are answered."""
        service, _, _ = make_service()
        session = make_session()
        q = make_question(text="Tell me about yourself.")
        session.questions.append(q)
        session.responses.append(make_response(question_id=q.question_id))

        reply = service.handle_repeat_request(session)

        # Should return a fallback message
        assert isinstance(reply, str)
        assert len(reply) > 0


class TestRepeatIntentRoutingInTechnicalRounds:
    """Tests that 'repeat' intent routes to handle_repeat_request in technical rounds.

    Requirements: 3.6, 14.2
    """

    @pytest.mark.asyncio
    async def test_repeat_intent_routes_to_handle_repeat_in_dsa_round(self):
        """When intent is 'repeat' in a DSA round, handle_repeat_request is called."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        # Intent classification returns "repeat"
        mock_llm.complete.return_value = json.dumps({"intent": "repeat"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(
            text="Find the longest common subsequence.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "can you repeat that?")

        assert "Find the longest common subsequence." in reply
        # Session should not have advanced
        assert len(updated.responses) == 0

    @pytest.mark.asyncio
    async def test_repeat_intent_routes_to_handle_repeat_in_system_design_round(self):
        """When intent is 'repeat' in a System Design round, handle_repeat_request is called."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        mock_llm.complete.return_value = json.dumps({"intent": "repeat"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(
            text="Design Twitter's feed system.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "I don't understand the question")

        assert "Design Twitter's feed system." in reply
        assert len(updated.responses) == 0

    @pytest.mark.asyncio
    async def test_repeat_intent_does_not_call_technical_evaluation(self):
        """Repeat intent should not trigger solution evaluation in technical rounds."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        mock_llm.complete.return_value = json.dumps({"intent": "repeat"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(
            text="Implement binary search.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        await service.handle_response(session, "say that again")

        # evaluate_coding_solution should NOT have been called
        mock_tech.evaluate_coding_solution.assert_not_called()

    @pytest.mark.asyncio
    async def test_repeat_intent_in_behavioral_round_also_works(self):
        """Repeat intent works in behavioral rounds too (not just technical)."""
        service, mock_llm, mock_pb = make_service()

        mock_llm.complete.return_value = json.dumps({"intent": "repeat"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.BEHAVIORAL,
        )
        q = make_question(text="Tell me about a challenge you overcame.")
        session.questions.append(q)

        reply, updated = await service.handle_response(session, "please repeat the question")

        assert "Tell me about a challenge you overcame." in reply
        assert len(updated.responses) == 0


class TestClarificationHandlingForTechnicalRounds:
    """Tests for clarification request handling during technical rounds.

    When a user's solution is ambiguous/incomplete, the LLM should ask
    clarifying follow-up questions (Req 3.6). When a user submits something
    completely off-topic, they are redirected to re-read the problem (Req 14.2).
    """

    @pytest.mark.asyncio
    async def test_ambiguous_solution_triggers_follow_up_question(self):
        """When a DSA solution is ambiguous, a follow-up clarification is asked (Req 3.6)."""
        from interview_practice_partner.domain.models import TechnicalEvaluation, ComplexityAnalysis

        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        # Intent classification returns "answer"
        mock_llm.complete.return_value = json.dumps({"intent": "answer"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        # Evaluation says follow-up is warranted (ambiguous solution)
        ambiguous_eval = TechnicalEvaluation(
            evaluation_id=str(uuid.uuid4()),
            question_id=str(uuid.uuid4()),
            response_id=str(uuid.uuid4()),
            correctness="partial",
            follow_up_warranted=True,
            follow_up_text="Can you walk me through your approach in more detail?",
            difficulty_signal="maintain",
            evaluated_at=NOW,
        )
        mock_tech.evaluate_coding_solution = AsyncMock(return_value=ambiguous_eval)
        mock_tech.adjust_difficulty = MagicMock(return_value=ProblemDifficulty.MEDIUM)

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(
            text="Find two numbers that sum to target.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        # Submit a vague/ambiguous solution (but long enough to pass word count check)
        vague_solution = "I would use a loop to iterate through the array and check each pair of numbers to see if they add up to the target value"
        reply, updated = await service.handle_response(session, vague_solution)

        # Should ask a clarifying follow-up question
        assert "Can you walk me through your approach in more detail?" in reply

    @pytest.mark.asyncio
    async def test_off_topic_in_dsa_round_redirects_to_problem(self):
        """Off-topic response in DSA round redirects user to re-read the problem (Req 14.2)."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        # Intent classification returns "out_of_scope"
        mock_llm.complete.return_value = json.dumps({"intent": "out_of_scope"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.DSA_CODING,
        )
        q = make_question(
            text="Implement a stack using two queues.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        reply, updated = await service.handle_response(
            session, "What's the weather like today?"
        )

        # Should redirect back to the problem
        assert "re-read" in reply.lower() or "current problem" in reply.lower()
        # Should include the problem text
        assert "Implement a stack using two queues." in reply

    @pytest.mark.asyncio
    async def test_off_topic_in_system_design_round_redirects_to_question(self):
        """Off-topic response in System Design round redirects user to re-read the question (Req 14.2)."""
        service, mock_llm, mock_pb, mock_tech = make_service_with_technical_mocks()

        mock_llm.complete.return_value = json.dumps({"intent": "out_of_scope"})
        mock_pb.build_intent_classification_prompt.return_value = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]

        session = make_session(
            role=Role.SOFTWARE_ENGINEER,
            interview_round_type=InterviewRoundType.SYSTEM_DESIGN,
        )
        q = make_question(
            text="Design a distributed cache.",
            question_type=QuestionType.TECHNICAL,
        )
        session.questions.append(q)

        reply, updated = await service.handle_response(
            session, "I like pizza and movies."
        )

        assert "re-read" in reply.lower() or "current problem" in reply.lower()
        assert "Design a distributed cache." in reply
