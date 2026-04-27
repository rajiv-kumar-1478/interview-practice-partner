"""PromptBuilder — constructs stage-appropriate system and user prompts for the LLM.

Each stage of the interview (role selection, question generation, response evaluation,
feedback) has its own prompt template.  All prompts instruct the LLM to use plain text
and WhatsApp-supported formatting only (bold via asterisks, line breaks) — no HTML,
no markdown headers, no code blocks.
"""

from __future__ import annotations

from typing import Optional

from interview_practice_partner.domain.enums import (
    DesignPhase,
    ProblemDifficulty,
    ProblemTopic,
    QuestionType,
    Role,
    SolutionFormat,
)
from interview_practice_partner.domain.models import (
    CodingProblem,
    Question,
    SessionState,
    SystemDesignQuestion,
    UserResponse,
)


# ---------------------------------------------------------------------------
# Role display names
# ---------------------------------------------------------------------------

_ROLE_DISPLAY: dict[Role, str] = {
    Role.SOFTWARE_ENGINEER: "Software Engineer",
    Role.SALES_REPRESENTATIVE: "Sales Representative",
    Role.RETAIL_ASSOCIATE: "Retail Associate",
    Role.UNKNOWN: "General",
}

_QUESTION_TYPE_DISPLAY: dict[QuestionType, str] = {
    QuestionType.BEHAVIOURAL: "behavioural",
    QuestionType.SITUATIONAL: "situational",
    QuestionType.TECHNICAL: "technical",
    QuestionType.FOLLOW_UP: "follow-up",
}

# ---------------------------------------------------------------------------
# Formatting rules injected into every system prompt
# ---------------------------------------------------------------------------

_FORMATTING_RULES = (
    "FORMATTING RULES (mandatory):\n"
    "- Use plain text only.\n"
    "- You may use *asterisks* for bold emphasis where helpful.\n"
    "- Use line breaks to separate paragraphs.\n"
    "- Do NOT use HTML tags of any kind (no angle-bracket tags such as bold, break, or paragraph tags).\n"
    "- Do NOT use markdown headers (lines starting with the hash character).\n"
    "- Do NOT use fenced code blocks (triple-backtick syntax).\n"
    "- Keep each message within 4096 characters."
)


class PromptBuilder:
    """Constructs stage-appropriate message lists for the LLM.

    Every public method returns a ``list[dict[str, str]]`` — a list of chat
    messages in the format expected by the ``LLMClient.complete`` interface
    (each dict has ``"role"`` and ``"content"`` keys).
    """

    # ------------------------------------------------------------------
    # Stage: ROLE_SELECTION
    # ------------------------------------------------------------------

    def build_role_selection_prompt(
        self,
        user_message: str,
        clarification_turn_count: int = 0,
    ) -> list[dict[str, str]]:
        """Build a prompt that extracts or confirms a job role from the user's message.

        Args:
            user_message: The raw text sent by the user.
            clarification_turn_count: How many clarification turns have already
                occurred.  When this reaches 2 the LLM should default to a
                general interview format.

        Returns:
            A messages list with a system prompt and the user message.
        """
        supported_roles = ", ".join(
            _ROLE_DISPLAY[r] for r in Role if r != Role.UNKNOWN
        )

        if clarification_turn_count >= 2:
            fallback_instruction = (
                "The user has not specified a clear role after two attempts. "
                "Politely inform them that you will proceed with a general interview format "
                "and ask them to confirm they are ready to begin."
            )
        else:
            fallback_instruction = (
                "If the role is ambiguous or not in the supported list, ask the user to "
                "clarify or confirm the closest supported role. "
                "If the role is completely unrecognised, inform the user of the limitation "
                "and suggest the closest supported role."
            )

        system_content = (
            "You are a professional interview coach helping a user prepare for a job interview "
            "via WhatsApp.\n\n"
            "Your task is to identify the job role the user wants to practise for.\n\n"
            f"Supported roles: {supported_roles}.\n\n"
            f"{fallback_instruction}\n\n"
            "When you have identified the role with confidence, confirm it with the user "
            "and let them know you are about to begin the mock interview.\n\n"
            "Respond with a JSON object containing:\n"
            '- "role": one of ["software_engineer", "sales_representative", '
            '"retail_associate", "unknown"]\n'
            '- "confidence": "high" | "low"\n'
            '- "message": the conversational reply to send to the user\n\n'
            f"{_FORMATTING_RULES}"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

    # ------------------------------------------------------------------
    # Stage: INTERVIEW — Question Generation
    # ------------------------------------------------------------------

    def build_question_generation_prompt(
        self,
        session: SessionState,
        question_type: QuestionType,
        difficulty_signal: str | None = None,
    ) -> list[dict[str, str]]:
        """Build a prompt that generates the next interview question.

        The prompt includes:
        - Role context so the LLM generates role-appropriate questions.
        - The question type to generate (behavioural / situational / technical).
        - All questions already asked in this session (for deduplication).
        - A difficulty signal when the user's recent performance warrants it.

        Args:
            session: The current ``SessionState``.
            question_type: The type of question to generate next.
            difficulty_signal: One of ``"increase"``, ``"maintain"``, or
                ``"decrease"`` (or ``None`` to omit the signal).

        Returns:
            A messages list with a system prompt and a user-turn trigger.
        """
        role_name = _ROLE_DISPLAY.get(session.role, "General")
        q_type_label = _QUESTION_TYPE_DISPLAY.get(question_type, question_type.value)

        # Build the list of already-asked questions for deduplication
        asked_questions = [q.text for q in session.questions if not q.skipped]
        if asked_questions:
            asked_block = "Questions already asked in this session (do NOT repeat these):\n" + "\n".join(
                f"- {q}" for q in asked_questions
            )
        else:
            asked_block = "No questions have been asked yet in this session."

        # Difficulty adjustment instruction
        if difficulty_signal == "increase":
            difficulty_instruction = (
                "The candidate has demonstrated strong knowledge. "
                "*Increase* the difficulty and depth of this question — "
                "probe for deeper understanding, edge cases, or advanced concepts."
            )
        elif difficulty_signal == "decrease":
            difficulty_instruction = (
                "The candidate has demonstrated weaker knowledge. "
                "*Decrease* the difficulty of this question — "
                "ask a more foundational, supportive question to build confidence."
            )
        else:
            difficulty_instruction = (
                "Maintain the current difficulty level for this question."
            )

        system_content = (
            f"You are a professional interviewer conducting a mock interview for a "
            f"*{role_name}* position.\n\n"
            "Your task is to generate ONE interview question of the specified type. "
            "The question must be relevant to the role, realistic, and distinct from "
            "any questions already asked.\n\n"
            f"Question type to generate: *{q_type_label}*\n\n"
            f"{asked_block}\n\n"
            f"Difficulty guidance: {difficulty_instruction}\n\n"
            "Return ONLY the question text — no preamble, no numbering, no explanation. "
            "The question should be a single, clear sentence or short paragraph as a "
            "real interviewer would ask it.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = (
            f"Generate the next {q_type_label} interview question for the {role_name} role."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Stage: INTERVIEW — Response Evaluation
    # ------------------------------------------------------------------

    def build_response_evaluation_prompt(
        self,
        question: Question,
        response: UserResponse,
        session: SessionState,
    ) -> list[dict[str, str]]:
        """Build a prompt that evaluates a candidate's response to an interview question.

        The LLM must return a JSON object with the following fields:
        - ``is_off_topic`` (bool): True if the response is not relevant to the question.
        - ``is_short`` (bool): True if the response is fewer than 15 words.
        - ``follow_up_warranted`` (bool): True if a follow-up question would add value.
        - ``follow_up_text`` (str | null): The follow-up question text, or null.
        - ``difficulty_signal`` ("increase" | "maintain" | "decrease"): Adjustment for
          the next question's difficulty based on the quality of this response.

        Args:
            question: The ``Question`` that was asked.
            response: The ``UserResponse`` to evaluate.
            session: The current ``SessionState`` (provides role context).

        Returns:
            A messages list with a system prompt and the evaluation request.
        """
        role_name = _ROLE_DISPLAY.get(session.role, "General")

        system_content = (
            f"You are an expert interview evaluator assessing a candidate's response "
            f"during a mock interview for a *{role_name}* position.\n\n"
            "Evaluate the candidate's response against the interview question and return "
            "a JSON object with EXACTLY these fields:\n\n"
            '- "is_off_topic": boolean — true if the response does not address the '
            "question or is unrelated to the interview context.\n"
            '- "is_short": boolean — true if the response is fewer than 15 words.\n'
            '- "follow_up_warranted": boolean — true if a follow-up question would '
            "meaningfully probe the candidate's answer further.\n"
            '- "follow_up_text": string or null — if follow_up_warranted is true, '
            "provide a concise follow-up question that directly references the "
            "candidate's answer; otherwise null.\n"
            '- "difficulty_signal": "increase" | "maintain" | "decrease" — '
            "set to \"increase\" if the candidate demonstrated strong knowledge "
            "(probe deeper next time), \"decrease\" if the candidate struggled "
            "(ask a more foundational question next time), or \"maintain\" otherwise.\n\n"
            "Return ONLY valid JSON. No explanation, no preamble.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = (
            f"Interview question:\n{question.text}\n\n"
            f"Candidate's response:\n{response.text}\n\n"
            "Evaluate this response and return the JSON object."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Stage: FEEDBACK — Report Generation
    # ------------------------------------------------------------------

    def build_intent_classification_prompt(
        self,
        user_message: str,
        current_question: str,
    ) -> list[dict[str, str]]:
        """Build a prompt that classifies the user's intent during an interview.

        Used to route the message before any evaluation happens. The LLM
        classifies the message into one of four intents:

        - ``answer``: The user is attempting to answer the current question.
        - ``skip``: The user wants to skip the question, doesn't know the
          answer, or wants a different question.
        - ``repeat``: The user wants the question repeated, rephrased, or
          explained more clearly.
        - ``out_of_scope``: The message is completely unrelated to the
          interview (e.g. casual chat, requests for unrelated help).

        Args:
            user_message: The raw text sent by the user.
            current_question: The current interview question text.

        Returns:
            A messages list with a system prompt and the user message.
        """
        system_content = (
            "You are an assistant helping to route messages during a mock job interview "
            "conducted over WhatsApp.\n\n"
            "The candidate was asked the following interview question:\n"
            f"{current_question}\n\n"
            "Classify the candidate's message into EXACTLY ONE of these intents:\n\n"
            '- "answer": The candidate is attempting to answer the interview question, '
            "even if the answer is short, vague, or poor quality.\n"
            '- "skip": The candidate wants to skip this question, move to a different '
            "question, says they don't know the answer, asks for the answer/solution, "
            "or explicitly requests a different type of question.\n"
            '- "repeat": The candidate wants the question repeated, rephrased, '
            "clarified, or explained in more detail. Includes messages like "
            "'explain again', 'I don't understand', 'what do you mean', 'say that again'.\n"
            '- "out_of_scope": The message is completely unrelated to the interview '
            "or interview preparation (e.g. casual greetings mid-session, requests "
            "for unrelated help, random statements).\n\n"
            "Return ONLY a JSON object with a single field:\n"
            '{"intent": "answer" | "skip" | "repeat" | "out_of_scope"}\n\n'
            "No explanation. No preamble. Just the JSON."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]


    def build_feedback_prompt(
        self,
        session: SessionState,
    ) -> list[dict[str, str]]:
        """Build a prompt that generates a structured ``FeedbackReport`` for the session.

        The LLM must return a JSON object matching the ``FeedbackReport`` schema:
        - ``dimension_scores``: list of ``DimensionScore`` objects, one per
          ``EvaluationDimension`` (communication_clarity, relevance,
          technical_knowledge, confidence).
        - ``strengths``: list of strings (at least one).
        - ``improvements``: list of strings (at least one).
        - ``actionable_recommendations``: list of strings (at least one).
        - ``off_topic_references``: list of strings referencing specific off-topic
          responses (may be empty if none occurred).

        Args:
            session: The completed ``SessionState`` containing all questions,
                responses, and evaluation metadata.

        Returns:
            A messages list with a system prompt and the transcript.
        """
        role_name = _ROLE_DISPLAY.get(session.role, "General")

        # Build the session transcript
        transcript_lines: list[str] = []
        response_map = {r.question_id: r for r in session.responses}

        for i, question in enumerate(session.questions, start=1):
            transcript_lines.append(f"Question {i} ({question.question_type.value}):")
            transcript_lines.append(question.text)

            if question.skipped:
                transcript_lines.append("Candidate response: [SKIPPED]")
            else:
                resp = response_map.get(question.question_id)
                if resp:
                    off_topic_flag = " [OFF-TOPIC]" if resp.is_off_topic else ""
                    transcript_lines.append(
                        f"Candidate response{off_topic_flag}: {resp.text}"
                    )
                else:
                    transcript_lines.append("Candidate response: [NO RESPONSE RECORDED]")

            transcript_lines.append("")  # blank line between Q&A pairs

        transcript = "\n".join(transcript_lines).strip()

        # Off-topic summary
        off_topic_count = session.off_topic_count
        if off_topic_count > 0:
            off_topic_note = (
                f"Note: The candidate went off-topic {off_topic_count} time(s) during "
                "this session. Reference the specific off-topic responses in "
                "\"off_topic_references\"."
            )
            if off_topic_count > 2:
                off_topic_note += (
                    " Since the candidate went off-topic more than twice, ensure that "
                    "\"improvements\" or \"actionable_recommendations\" includes a comment "
                    "on focus and relevance."
                )
        else:
            off_topic_note = "The candidate did not go off-topic during this session."

        system_content = (
            f"You are an expert interview coach generating a structured feedback report "
            f"for a candidate who just completed a mock interview for a *{role_name}* position.\n\n"
            "Analyse the full session transcript below and return a JSON object with "
            "EXACTLY these fields:\n\n"
            '"dimension_scores": array of objects, one for EACH of the following '
            "evaluation dimensions (all four are required):\n"
            '  - "communication_clarity"\n'
            '  - "relevance"\n'
            '  - "technical_knowledge"\n'
            '  - "confidence"\n'
            "  Each object must have:\n"
            '    - "dimension": the dimension name (string, from the list above)\n'
            '    - "qualitative_assessment": a 1–3 sentence qualitative assessment '
            "(plain text only, no angle-bracket tags, no markdown headers)\n"
            '    - "score": integer from 1 to 5 (1 = very poor, 5 = excellent; '
            "for internal use only, not shown to the candidate)\n\n"
            '"strengths": array of strings — at least one specific strength observed '
            "in the candidate's responses.\n\n"
            '"improvements": array of strings — at least one specific area for '
            "improvement.\n\n"
            '"actionable_recommendations": array of strings — at least one concrete, '
            "actionable recommendation the candidate can apply in a real interview.\n\n"
            '"off_topic_references": array of strings — quote or paraphrase each '
            "response that was flagged as off-topic; empty array if none.\n\n"
            f"{off_topic_note}\n\n"
            "Return ONLY valid JSON. No explanation, no preamble.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = (
            f"Session transcript for {role_name} mock interview:\n\n"
            f"{transcript}\n\n"
            "Generate the feedback report JSON."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Technical Rounds: Round Type Selection
    # ------------------------------------------------------------------

    def build_round_type_selection_prompt(
        self,
        user_message: str,
    ) -> list[dict[str, str]]:
        """Build a prompt for round type selection after role is confirmed.

        Presents three options: DSA/Coding, System Design, or Behavioral.

        Args:
            user_message: The user's message (may contain round type keywords).

        Returns:
            A messages list with a system prompt and the user message.
        """
        system_content = (
            "You are a professional interview coach helping a user prepare for a "
            "Software Engineer interview via WhatsApp.\n\n"
            "The user has selected the Software Engineer role. Your task is to help "
            "them choose an interview round type to practice.\n\n"
            "Present three options:\n"
            "1. *DSA/Coding Round* - Practice algorithmic problem-solving\n"
            "2. *System Design Round* - Practice architectural design\n"
            "3. *Behavioral Round* - Practice soft skills and experience questions\n\n"
            "Ask the user which round type they would like to practice.\n\n"
            "Return a JSON object with:\n"
            '- "message": the conversational message to send to the user\n'
            '- "round_type_detected": "dsa_coding" | "system_design" | "behavioral" | null\n\n'
            "If the user's message contains clear keywords (e.g., 'DSA', 'coding', "
            "'algorithms' for DSA; 'system design', 'design', 'architecture' for System Design; "
            "'behavioral', 'soft skills' for Behavioral), set round_type_detected to the "
            "appropriate value. Otherwise, set it to null and ask them to choose.\n\n"
            f"{_FORMATTING_RULES}"
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_message},
        ]

    # ------------------------------------------------------------------
    # Technical Rounds: DSA Problem Generation
    # ------------------------------------------------------------------

    def build_coding_problem_generation_prompt(
        self,
        session: SessionState,
        difficulty: ProblemDifficulty,
        topic: Optional[ProblemTopic],
        problems_asked: list[CodingProblem],
    ) -> list[dict[str, str]]:
        """Build a prompt for generating a coding problem.

        Args:
            session: The current SessionState.
            difficulty: The desired problem difficulty.
            topic: Optional specific topic to focus on.
            problems_asked: List of problems already asked in this session.

        Returns:
            A messages list with a system prompt and generation request.
        """
        topic_instruction = (
            f"Focus on the topic: *{topic.value}*.\n"
            if topic
            else "Choose an appropriate algorithmic topic.\n"
        )

        # Build list of already-asked problems
        if problems_asked:
            asked_block = (
                "Problems already asked in this session (do NOT repeat these):\n"
                + "\n".join(f"- {p.text[:100]}..." for p in problems_asked)
            )
        else:
            asked_block = "No problems have been asked yet in this session."

        system_content = (
            "You are an expert technical interviewer generating a coding problem for a "
            "Software Engineer interview.\n\n"
            f"Generate ONE coding problem with difficulty: *{difficulty.value}*\n"
            f"{topic_instruction}\n"
            "The problem must include:\n"
            "1. Clear problem statement\n"
            "2. Input/output examples (at least 2)\n"
            "3. Constraints (input size, value ranges)\n"
            "4. The primary algorithmic topic\n\n"
            f"{asked_block}\n\n"
            "Return a JSON object with:\n"
            '- "problem_statement": the main problem description\n'
            '- "examples": array of input/output example strings\n'
            '- "constraints": string describing constraints\n'
            '- "topic": the primary algorithmic topic (arrays, strings, trees, graphs, '
            "dynamic_programming, sorting, searching, hash_tables, stacks_queues, linked_lists)\n\n"
            "Return ONLY valid JSON. No explanation, no preamble.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = (
            f"Generate a {difficulty.value} difficulty coding problem"
            + (f" on {topic.value}" if topic else "")
            + "."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Technical Rounds: DSA Solution Evaluation
    # ------------------------------------------------------------------

    def build_coding_solution_evaluation_prompt(
        self,
        problem: CodingProblem,
        response: UserResponse,
        solution_format: SolutionFormat,
    ) -> list[dict[str, str]]:
        """Build a prompt for evaluating a coding solution.

        Args:
            problem: The CodingProblem that was asked.
            response: The UserResponse containing the solution.
            solution_format: The detected format (code/pseudocode/explanation).

        Returns:
            A messages list with a system prompt and evaluation request.
        """
        format_note = {
            SolutionFormat.CODE: "The candidate submitted actual code.",
            SolutionFormat.PSEUDOCODE: "The candidate submitted pseudocode.",
            SolutionFormat.EXPLANATION: "The candidate provided a plain explanation.",
        }[solution_format]

        system_content = (
            "You are an expert technical interviewer evaluating a candidate's solution "
            "to a coding problem.\n\n"
            f"Problem:\n{problem.text}\n\n"
            f"Constraints: {problem.constraints}\n\n"
            f"Examples:\n" + "\n".join(problem.examples) + "\n\n"
            f"Candidate's solution:\n{response.text}\n\n"
            f"{format_note}\n\n"
            "Evaluate the solution across these dimensions:\n"
            "1. Correctness: Does it solve the problem correctly?\n"
            "2. Time Complexity: What is the Big-O time complexity?\n"
            "3. Space Complexity: What is the Big-O space complexity?\n"
            "4. Edge Cases: Which edge cases are handled? Which are missed?\n"
            "5. Code Quality: Is the code readable and well-structured? (if applicable)\n\n"
            "Return a JSON object with:\n"
            '- "correctness": "correct" | "incorrect" | "partial"\n'
            '- "time_complexity": Big-O notation string (e.g., "O(n)", "O(n log n)")\n'
            '- "space_complexity": Big-O notation string\n'
            '- "is_optimal": boolean (is this the optimal solution?)\n'
            '- "optimization_suggestions": string or null\n'
            '- "edge_cases_handled": array of strings\n'
            '- "edge_cases_missed": array of strings\n'
            '- "code_quality_notes": string or null\n'
            '- "follow_up_warranted": boolean\n'
            '- "follow_up_text": string or null\n'
            '- "difficulty_signal": "increase" | "maintain" | "decrease"\n\n'
            "Return ONLY valid JSON. No explanation, no preamble.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = "Evaluate this solution and return the JSON object."

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Technical Rounds: System Design Question Generation
    # ------------------------------------------------------------------

    def build_system_design_question_generation_prompt(
        self,
        session: SessionState,
        questions_asked: list[SystemDesignQuestion],
    ) -> list[dict[str, str]]:
        """Build a prompt for generating a system design question.

        Args:
            session: The current SessionState.
            questions_asked: List of questions already asked in this session.

        Returns:
            A messages list with a system prompt and generation request.
        """
        # Build list of already-asked questions
        if questions_asked:
            asked_block = (
                "Questions already asked in this session (do NOT repeat these):\n"
                + "\n".join(f"- {q.system_name}: {q.text}" for q in questions_asked)
            )
        else:
            asked_block = "No questions have been asked yet in this session."

        system_content = (
            "You are an expert system design interviewer generating a design question "
            "for a Software Engineer interview.\n\n"
            "Generate ONE system design question appropriate for a mid-level to senior "
            "Software Engineer.\n\n"
            "Common systems to design:\n"
            "- Twitter/social media feed\n"
            "- URL shortener\n"
            "- Instagram/photo sharing\n"
            "- Netflix/video streaming\n"
            "- Uber/ride-sharing\n"
            "- WhatsApp/messaging\n"
            "- E-commerce platform\n"
            "- Search engine\n\n"
            "The question should be open-ended and allow for discussion of:\n"
            "- Functional requirements\n"
            "- Non-functional requirements (scale, latency, availability)\n"
            "- High-level architecture\n"
            "- Database design\n"
            "- API design\n"
            "- Caching and load balancing\n\n"
            f"{asked_block}\n\n"
            "Return a JSON object with:\n"
            '- "system_name": short name (e.g., "Twitter")\n'
            '- "question_text": the question to ask (e.g., "Design a social media feed like Twitter")\n'
            '- "description": brief context about the system\n\n'
            "Return ONLY valid JSON. No explanation, no preamble.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = "Generate a system design question."

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Technical Rounds: System Design Evaluation
    # ------------------------------------------------------------------

    def build_system_design_evaluation_prompt(
        self,
        question: SystemDesignQuestion,
        response: UserResponse,
        current_phase: DesignPhase,
    ) -> list[dict[str, str]]:
        """Build a prompt for evaluating a system design response.

        Args:
            question: The SystemDesignQuestion that was asked.
            response: The UserResponse containing the design explanation.
            current_phase: The current design phase.

        Returns:
            A messages list with a system prompt and evaluation request.
        """
        phase_guidance = {
            DesignPhase.REQUIREMENTS_GATHERING: (
                "Focus on whether functional and non-functional requirements are clarified."
            ),
            DesignPhase.HIGH_LEVEL_DESIGN: (
                "Focus on whether major components and their interactions are described."
            ),
            DesignPhase.DEEP_DIVE: (
                "Focus on whether specific components are elaborated with sufficient detail."
            ),
            DesignPhase.BOTTLENECK_ANALYSIS: (
                "Focus on whether scalability bottlenecks are identified and addressed."
            ),
        }[current_phase]

        system_content = (
            "You are an expert system design interviewer evaluating a candidate's design response.\n\n"
            f"System Design Question:\n{question.text}\n\n"
            f"Description: {question.description}\n\n"
            f"Current Design Phase: *{current_phase.value}*\n\n"
            f"Candidate's response:\n{response.text}\n\n"
            f"Evaluation guidance: {phase_guidance}\n\n"
            "Evaluate the response and return a JSON object with:\n"
            '- "design_aspects_evaluated": object mapping aspect names to assessment strings:\n'
            '  - "scalability": assessment string or null\n'
            '  - "database_design": assessment string or null\n'
            '  - "api_design": assessment string or null\n'
            '  - "caching_strategy": assessment string or null\n'
            '  - "load_balancing": assessment string or null\n'
            '- "design_strengths": array of specific strengths\n'
            '- "design_weaknesses": array of specific weaknesses\n'
            '- "follow_up_warranted": boolean\n'
            '- "follow_up_text": string or null\n'
            '- "next_phase_suggestion": "requirements_gathering" | "high_level_design" | '
            '"deep_dive" | "bottleneck_analysis" | null (null if design is complete)\n\n'
            "Return ONLY valid JSON. No explanation, no preamble.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = "Evaluate this design response and return the JSON object."

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Technical Rounds: Technical Feedback Generation
    # ------------------------------------------------------------------

    def build_technical_feedback_prompt(
        self,
        session: SessionState,
    ) -> list[dict[str, str]]:
        """Build a prompt for generating technical feedback.

        Args:
            session: The completed SessionState with technical round data.

        Returns:
            A messages list with a system prompt and feedback request.
        """
        round_type = session.interview_round_type
        if not round_type:
            # Fallback to behavioral feedback
            return self.build_feedback_prompt(session)

        # Build transcript based on round type
        transcript_lines: list[str] = []

        if round_type.value == "dsa_coding":
            transcript_lines.append("DSA/Coding Round Summary:")
            transcript_lines.append(f"Initial Difficulty: {session.problem_difficulty.value}")
            transcript_lines.append(f"Topics Covered: {', '.join(t.value for t in session.topics_covered)}")
            transcript_lines.append("")

            # Add difficulty adjustment history
            if session.difficulty_adjustment_history:
                transcript_lines.append("Difficulty Adjustments:")
                for adj in session.difficulty_adjustment_history:
                    transcript_lines.append(
                        f"- {adj.get('from', 'unknown')} → {adj.get('to', 'unknown')}: {adj.get('reason', 'no reason')}"
                    )
                transcript_lines.append("")

            # Add problems and responses
            for i, question in enumerate(session.questions, start=1):
                transcript_lines.append(f"Problem {i}:")
                transcript_lines.append(question.text)
                if question.skipped:
                    transcript_lines.append("Candidate response: [SKIPPED]")
                else:
                    # Find corresponding response
                    resp = next((r for r in session.responses if r.question_id == question.question_id), None)
                    if resp:
                        transcript_lines.append(f"Candidate response:\n{resp.text}")
                transcript_lines.append("")

        elif round_type.value == "system_design":
            transcript_lines.append("System Design Round Summary:")
            transcript_lines.append(f"Design Aspects Covered: {', '.join(a.value for a in session.design_aspects_covered)}")
            transcript_lines.append("")

            # Add questions and responses
            for i, question in enumerate(session.questions, start=1):
                transcript_lines.append(f"Design Question {i}:")
                transcript_lines.append(question.text)
                if question.skipped:
                    transcript_lines.append("Candidate response: [SKIPPED]")
                else:
                    # Find corresponding response
                    resp = next((r for r in session.responses if r.question_id == question.question_id), None)
                    if resp:
                        transcript_lines.append(f"Candidate response:\n{resp.text}")
                transcript_lines.append("")

        transcript = "\n".join(transcript_lines).strip()

        system_content = (
            "You are an expert technical interview coach generating a structured feedback "
            f"report for a candidate who just completed a *{round_type.value}* round.\n\n"
            "Analyse the session transcript below and return a JSON object with:\n"
            '- "strengths": array of specific strengths demonstrated\n'
            '- "improvements": array of specific areas for improvement\n'
            '- "actionable_recommendations": array of concrete, actionable recommendations\n'
        )

        if round_type.value == "dsa_coding":
            system_content += (
                '- "complexity_summary": string summarizing the candidate\'s understanding '
                "of time/space complexity\n"
                '- "problem_solving_approach": string assessing the candidate\'s approach '
                "to breaking down problems\n"
            )
        elif round_type.value == "system_design":
            system_content += (
                '- "design_thinking": string assessing the candidate\'s architectural thinking\n'
                '- "scalability_awareness": string assessing awareness of scalability concerns\n'
            )

        system_content += (
            "\nReturn ONLY valid JSON. No explanation, no preamble.\n\n"
            f"{_FORMATTING_RULES}"
        )

        user_content = (
            f"Session transcript for {round_type.value} round:\n\n"
            f"{transcript}\n\n"
            "Generate the technical feedback report JSON."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
