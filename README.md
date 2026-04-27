# Interview Practice Partner 🎯

An AI-powered WhatsApp bot that helps users prepare for job interviews through realistic mock interviews with voice note support.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-616%20passing-brightgreen.svg)](tests/)

---

## 🌟 Key Features

- **🎤 Voice Note Support** - Conduct interviews using voice notes with automatic transcription (Groq Whisper) and text-to-speech responses (ElevenLabs)
- **🤖 Intelligent Conversation** - Contextual follow-up questions that adapt to your responses
- **👥 Multiple Roles** - Software Engineer, Sales Representative, Retail Associate
- **📊 Structured Feedback** - Detailed post-interview analysis covering communication, technical knowledge, relevance, and confidence
- **🔄 Adaptive Difficulty** - Questions adjust based on your performance
- **💬 WhatsApp Native** - Seamless integration with WhatsApp via Twilio

---

## 📋 Table of Contents

- [Why Voice Notes?](#why-voice-notes)
- [Architecture & Design Decisions](#architecture--design-decisions)
- [How We Handle Different User Types](#how-we-handle-different-user-types)
- [Quick Start](#quick-start)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Design Choices](#design-choices)
- [Testing](#testing)

---

## 🎤 Why Voice Notes?

Voice notes are the **unique differentiator** of this interview practice bot. Here's why they matter:

### 1. **Realistic Interview Simulation**
Real interviews are spoken conversations. Practicing with voice notes:
- Helps you articulate thoughts verbally, not just in writing
- Reveals verbal tics, filler words, and pacing issues
- Builds confidence in speaking under pressure
- Simulates the actual interview experience more accurately

### 2. **Accessibility & Convenience**
- Practice while commuting, walking, or doing chores
- No need to type long responses on a phone keyboard
- Natural for users who think better when speaking
- Reduces friction - just press record and talk

### 3. **Richer Feedback Opportunities**
- Transcription reveals how clearly you communicate verbally
- Audio responses create a more engaging, human-like interaction
- Mode switching (voice ↔ text) adapts to your environment

### 4. **Technical Innovation**
- Groq Whisper API for fast, accurate transcription
- ElevenLabs for natural-sounding text-to-speech
- Automatic mode detection and switching
- Seamless integration with WhatsApp media messages

**Design Decision:** Voice notes are optional, not required. Users can switch between voice and text at any time, making the bot flexible for different environments (quiet room vs. public space).

---

## 🏗️ Architecture & Design Decisions

### Clean Architecture Principles

The system follows **clean architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                     WhatsApp (Twilio)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Webhook Layer (FastAPI)                    │
│  - Request validation & signature verification              │
│  - Correlation ID injection for tracing                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                            │
│  - InterviewService: Question generation & evaluation       │
│  - FeedbackService: Post-interview analysis                 │
│  - TwilioMessagingService: Outbound message handling        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Domain Layer                             │
│  - SessionState: Conversation state model                   │
│  - Question, Response, FeedbackReport: Core entities        │
│  - Enums: Role, Stage, QuestionType                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    Integration Layer                        │
│  - LLMClient: DeepSeek API integration                      │
│  - GroqWhisperClient: Voice transcription                   │
│  - ElevenLabsClient: Text-to-speech                         │
│  - AudioDownloadClient: Twilio media retrieval              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                 Data Access Layer                           │
│  - SessionRepository: Redis persistence                     │
│  - MediaStorage: Temporary audio file storage               │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

#### 1. **Stateless HTTP with Redis Session Persistence**
**Problem:** WhatsApp sends each message as a separate HTTP request. How do we maintain conversation context?

**Solution:** Store `SessionState` in Redis, keyed by phone number.

**Why Redis?**
- Fast in-memory lookups (< 1ms)
- Built-in TTL for automatic cleanup
- Simple key-value model fits our use case
- Easy to scale horizontally

**Alternative Considered:** PostgreSQL with JSONB column
- **Rejected because:** Overkill for simple key-value storage, slower than Redis, requires schema migrations

#### 2. **Abstract Base Classes for External Integrations**
**Problem:** How do we make external API clients testable and swappable?

**Solution:** Define abstract base classes (`LLMClient`, `WhisperClient`, `TTSClient`) with concrete implementations.

**Benefits:**
- Easy to mock in tests (616 tests, all passing)
- Can swap providers without changing service layer
- Clear contracts for what each integration must provide

**Example:**
```python
class LLMClient(ABC):
    @abstractmethod
    async def complete(self, messages: list[dict]) -> str:
        pass

class DeepSeekLLMClient(LLMClient):
    async def complete(self, messages: list[dict]) -> str:
        # Actual API call
```

#### 3. **Pydantic for Configuration & Validation**
**Problem:** How do we manage environment variables and ensure they're valid at startup?

**Solution:** `pydantic-settings` with strict validation.

**Benefits:**
- Fail fast at startup if config is invalid
- Type-safe access to settings throughout the app
- Automatic parsing (e.g., `REDIS_URL` → `RedisDsn`)
- Clear documentation of required environment variables

#### 4. **Structured Logging with Correlation IDs**
**Problem:** How do we trace a single user interaction across multiple log entries?

**Solution:** Inject a `correlation_id` (Twilio message SID) into every log entry.

**Example:**
```json
{"event": "webhook_request", "correlation_id": "SM123...", "from": "whatsapp:+1..."}
{"event": "groq_whisper.transcribe_start", "correlation_id": "SM123...", "audio_size": 45678}
{"event": "llm_call", "correlation_id": "SM123...", "tokens": 312, "latency_ms": 1240}
{"event": "session_saved", "correlation_id": "SM123...", "stage": "INTERVIEW"}
```

**Benefits:**
- Easy to trace a single request through the entire system
- Enables debugging in production
- Works with log aggregation tools (Datadog, CloudWatch, etc.)

#### 5. **Graceful Degradation for Voice Features**
**Problem:** What happens if Groq Whisper or ElevenLabs APIs fail?

**Solution:** Fall back to text mode with informative messages.

**Behavior:**
- Transcription fails → Ask user to resend or type their answer
- TTS fails → Send reply as text instead of audio
- Both failures logged with full context for debugging

**Why:** Voice is a premium feature, but the core interview functionality must always work.

#### 6. **Automatic Mode Switching**
**Problem:** Should users explicitly set voice/text mode, or should the bot detect it?

**Solution:** Both. Auto-detect input type and switch modes automatically, but also support explicit commands (`"voice mode"`, `"text mode"`).

**Why:**
- Auto-switching reduces friction (just send a voice note and the bot adapts)
- Explicit commands give power users control
- Best of both worlds

#### 7. **Twilio Signature Verification**
**Problem:** How do we prevent unauthorized requests to our webhook?

**Solution:** Validate Twilio's HMAC-SHA1 signature on every request.

**Implementation:**
```python
def verify_twilio_signature(request: Request, body: bytes) -> bool:
    signature = request.headers.get("X-Twilio-Signature")
    url = str(request.url)
    expected = compute_signature(auth_token, url, body)
    return hmac.compare_digest(signature, expected)
```

**Why:** Security best practice. Prevents replay attacks and unauthorized access.

---

## 👥 How We Handle Different User Types

The bot is designed to handle diverse user behaviors gracefully. Here's how:

### 1. **The Confused User** 🤔
**Behavior:** Unsure what they want, provides vague input, changes their mind.

**How We Handle:**
- **Clarification prompts** when role is ambiguous
- **Suggest closest match** for vague input (e.g., "something with computers" → Software Engineer)
- **Confirm before proceeding** to avoid misunderstandings
- **Allow role changes** mid-session with confirmation

**Example:**
```
User: "I need help"
Bot: "Which role are you preparing for? Software Engineer, Sales Rep, or Retail Associate?"

User: "Maybe something with computers?"
Bot: "It sounds like Software Engineer. Is that correct?"

User: "Actually, sales"
Bot: "Perfect! Let's prepare for Sales. Here's your first question..."
```

**Design Decision:** Never leave the user stuck. Always provide a path forward, even if it means making an educated guess and confirming.

### 2. **The Efficient User** ⚡
**Behavior:** Knows exactly what they want, wants to start immediately.

**How We Handle:**
- **Fast-path initialization** - detect role in first message and start immediately
- **No unnecessary setup questions** - if role is clear, begin interview in one turn
- **Support skip commands** - `"skip"` moves to next question without penalty
- **Concise feedback** - structured but not verbose

**Example:**
```
User: "Software Engineer"
Bot: "Great! Here's your first question: Tell me about a time when..."
```

**Design Decision:** Respect the user's time. If they're ready to go, don't slow them down with setup.

### 3. **The Chatty User** 💬
**Behavior:** Goes off-topic, shares personal stories, asks unrelated questions.

**How We Handle:**
- **Gentle redirection** - acknowledge their message, then refocus
- **Track off-topic count** in session state
- **Escalate firmness** after 2+ off-topic responses
- **Include in feedback** - mention focus issues in final report

**Example:**
```
User: "Oh, speaking of sales, I just bought a new Tesla! It's amazing..."
Bot: "That's great! Congratulations. However, let's focus on your interview prep. 
      Please answer: Describe a time when you closed a difficult sale."

[After 2nd off-topic response]
Bot: "I appreciate your enthusiasm, but let's stay focused on the interview. 
      Please answer the question."
```

**Design Decision:** Balance empathy with productivity. Acknowledge the user, but keep the session on track.

### 4. **The Edge Case User** 🔧
**Behavior:** Empty messages, very short responses, requests beyond scope, invalid input.

**How We Handle:**

| Edge Case | Bot Response | Design Rationale |
|-----------|--------------|------------------|
| Empty message | "I didn't receive a response. Could you please share your answer?" | Prompt for input without being judgmental |
| Very short response (< 15 words) | "Your response seems brief. Could you elaborate?" | Encourage complete answers like in real interviews |
| `"skip"` | "No problem, I'll skip this question. Let's move on..." | Support user autonomy, don't penalize |
| `"repeat"` | Restate question verbatim | Match real interviewer behavior |
| Out-of-scope request (e.g., "write my resume") | "I specialize in interview practice and can't help with that. Would you like to continue?" | Clear boundaries, redirect to core functionality |
| Offensive content | Decline to engage, issue polite warning | Safety and professionalism |
| 3+ consecutive invalid inputs | "Would you like to end the session or return to role selection?" | Offer exit path to avoid frustration |

**Design Decision:** Never crash, never leave the user without a response. Every edge case has a graceful fallback.

### 5. **The Voice Note User** 🎤
**Behavior:** Prefers speaking over typing, uses voice notes exclusively.

**How We Handle:**
- **Automatic transcription** via Groq Whisper
- **Auto-switch to voice mode** when first voice note is detected
- **Audio responses** via ElevenLabs TTS
- **Fallback to text** if transcription/TTS fails

**Example:**
```
User: [Sends voice note]
Bot: [Transcribes] → [Generates response] → [Synthesizes audio] → [Sends audio message]

[If transcription fails]
Bot: "I couldn't transcribe your voice note. Could you resend it or type your answer?"
```

**Design Decision:** Voice should feel seamless. The user shouldn't need to think about modes - just talk and the bot adapts.

---

## 🚀 Quick Start

### Deployment Options

#### Option 1: Deploy to Render (Recommended for Production)

Deploy to Render with one click using the Blueprint:

1. Push your code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click **New +** → **Blueprint**
4. Select your repository
5. Add your API keys in environment variables
6. Deploy!

See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for detailed instructions.

**Benefits:**
- ✅ Free tier available
- ✅ Automatic HTTPS (no ngrok needed)
- ✅ Managed Redis included
- ✅ Auto-deploy on git push

#### Option 2: Run Locally (For Development)

### Prerequisites
- Python 3.11+
- Docker Desktop (for Redis)
- ngrok account
- Twilio account (free trial works)
- DeepSeek API key
- Groq API key (for voice transcription)
- ElevenLabs API key (for text-to-speech)

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd interview-practice-partner

# Install dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (see SETUP.md for details)
```

### Running the Application

```bash
# Start Redis
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Start the FastAPI app
uvicorn src.interview_practice_partner.main:app --reload

# In another terminal, start ngrok
ngrok http 8000

# Configure Twilio webhook with your ngrok URL
# See SETUP.md for detailed instructions
```

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Expected: 616 tests passing
```

For detailed local setup instructions, see [SETUP.md](SETUP.md).  
For Render deployment instructions, see [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md).

---

## 🛠️ Technical Stack

| Component | Technology | Why We Chose It |
|-----------|-----------|-----------------|
| **Backend Framework** | FastAPI | Async support, automatic OpenAPI docs, Pydantic integration |
| **LLM Provider** | DeepSeek | Cost-effective, good quality, OpenAI-compatible API |
| **Voice Transcription** | Groq Whisper | Fast (< 1s), accurate, free tier, OpenAI-compatible SDK |
| **Text-to-Speech** | ElevenLabs | Natural voices, low latency, good free tier |
| **Messaging Platform** | Twilio WhatsApp API | Reliable, well-documented, sandbox for testing |
| **Session Store** | Redis | Fast, simple, built-in TTL, perfect for session data |
| **Configuration** | pydantic-settings | Type-safe, validates at startup, clear error messages |
| **Logging** | structlog | Structured JSON logs, correlation IDs, production-ready |
| **Testing** | pytest + hypothesis | 616 tests, property-based testing for robustness |

---

## 📁 Project Structure

```
interview-practice-partner/
├── src/interview_practice_partner/
│   ├── api/                      # Webhook layer
│   │   ├── routers/
│   │   │   ├── twilio_webhook.py # WhatsApp message handling
│   │   │   └── health.py         # Health check endpoint
│   │   ├── middleware/
│   │   │   ├── twilio_signature.py  # Request verification
│   │   │   └── request_logging.py   # Correlation ID injection
│   │   ├── dependencies.py       # Dependency injection
│   │   └── schemas.py            # Request/response DTOs
│   │
│   ├── services/                 # Service layer
│   │   ├── interview.py          # Core interview logic
│   │   ├── feedback.py           # Feedback generation
│   │   ├── twilio_messaging.py   # Outbound messages
│   │   └── code_parser.py        # Solution format detection (for tech rounds)
│   │
│   ├── domain/                   # Domain layer
│   │   ├── models.py             # SessionState, Question, Response, etc.
│   │   └── enums.py              # Role, Stage, QuestionType, etc.
│   │
│   ├── llm/                      # LLM integration layer
│   │   ├── llm_client.py         # Abstract base class
│   │   ├── deepseek_client.py    # DeepSeek implementation
│   │   └── prompt_builder.py     # Prompt templates
│   │
│   ├── audio/                    # Audio processing layer
│   │   ├── whisper_client.py     # Groq Whisper transcription
│   │   ├── tts_client.py         # ElevenLabs text-to-speech
│   │   └── download_client.py    # Twilio media download
│   │
│   ├── data/                     # Data access layer
│   │   └── session_repository.py # Redis persistence
│   │
│   ├── config.py                 # Configuration (pydantic-settings)
│   └── main.py                   # FastAPI application
│
├── tests/                        # Test suite (616 tests)
│   ├── test_domain_models.py
│   ├── test_interview_service.py
│   ├── test_feedback_service.py
│   ├── test_prompt_builder.py
│   ├── test_code_parser.py
│   └── ...
│
├── .env.example                  # Environment template
├── pyproject.toml                # Dependencies & project metadata
├── docker-compose.yml            # Redis + app containers
├── render.yaml                   # Render deployment blueprint
├── SETUP.md                      # Detailed setup guide
└── README.md                     # This file
```

---

## 🎨 Design Choices

### 1. **Why WhatsApp?**
- **Ubiquity:** 2+ billion users worldwide
- **Familiarity:** Users already know how to use it
- **Rich media:** Supports voice notes, audio messages, images
- **No app install:** Works through existing WhatsApp app
- **Twilio integration:** Reliable, well-documented API

**Alternative Considered:** Telegram
- **Rejected because:** Smaller user base, less familiar to most users

### 2. **Why DeepSeek for LLM?**
- **Cost-effective:** ~10x cheaper than GPT-4
- **Good quality:** Comparable to GPT-3.5 for conversational tasks
- **OpenAI-compatible:** Easy to swap if needed
- **Fast:** Low latency for real-time chat

**Alternative Considered:** OpenAI GPT-4
- **Rejected because:** Too expensive for a demo/prototype

### 3. **Why Groq Whisper?**
- **Speed:** < 1 second transcription for typical voice notes
- **Accuracy:** Whisper-large-v3-turbo is highly accurate
- **Free tier:** Generous limits for testing
- **OpenAI-compatible SDK:** Easy integration

**Alternative Considered:** OpenAI Whisper API
- **Rejected because:** Slower, more expensive

### 4. **Why ElevenLabs for TTS?**
- **Natural voices:** Best-in-class voice quality
- **Low latency:** Fast synthesis (< 2 seconds)
- **Voice library:** Many professional voices to choose from
- **Free tier:** Sufficient for testing

**Alternative Considered:** Google Cloud TTS
- **Rejected because:** Less natural-sounding voices

### 5. **Why Redis for Session Storage?**
- **Speed:** In-memory, < 1ms lookups
- **Simplicity:** Key-value model fits our use case perfectly
- **TTL:** Automatic cleanup of old sessions
- **Scalability:** Easy to scale horizontally

**Alternative Considered:** PostgreSQL
- **Rejected because:** Overkill for simple key-value storage

### 6. **Why Clean Architecture?**
- **Testability:** Easy to mock external dependencies
- **Maintainability:** Clear separation of concerns
- **Flexibility:** Can swap implementations without changing business logic
- **Scalability:** Each layer can be scaled independently

**Alternative Considered:** Monolithic structure
- **Rejected because:** Hard to test, hard to maintain, hard to scale

---

## 🧪 Testing

### Test Coverage

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Expected output:
# 616 tests passing
# Coverage: 92%
```

### Test Categories

| Category | Count | Purpose |
|----------|-------|---------|
| **Domain Models** | 90 | Validate Pydantic models, serialization, defaults |
| **Service Layer** | 180 | Test interview logic, feedback generation, mode switching |
| **LLM Integration** | 76 | Test prompt building, response parsing, error handling |
| **Audio Processing** | 51 | Test transcription, TTS, code parsing |
| **Repository** | 19 | Test Redis persistence, session lifecycle |
| **API Layer** | 120 | Test webhook handling, signature verification, error responses |
| **Integration** | 80 | End-to-end flows, multi-turn conversations |

### Property-Based Testing

We use Hypothesis for property-based testing to catch edge cases:

```python
@given(st.text(min_size=1, max_size=1000))
def test_session_state_serialization(text):
    """Any valid text should serialize/deserialize correctly."""
    session = SessionState(session_id="test", phone_number="+1234567890")
    session.questions.append(Question(text=text, ...))
    
    # Should round-trip through JSON
    json_str = session.model_dump_json()
    restored = SessionState.model_validate_json(json_str)
    assert restored.questions[0].text == text
```

---

## 📊 Evaluation Criteria Coverage

| Criterion | How We Address It | Evidence |
|-----------|-------------------|----------|
| **Conversational Quality** | Natural follow-ups, professional tone, contextual responses, adaptive difficulty | See `InterviewService.handle_response()`, `PromptBuilder` templates |
| **Agentic Behaviour** | Role detection, mode switching, graceful error handling, redirection, skip support | See "How We Handle Different User Types" section |
| **Technical Implementation** | Clean architecture, Redis persistence, structured logging, 616 passing tests | See "Architecture & Design Decisions" section |
| **Intelligence & Adaptability** | Difficulty adjustment, personalized feedback, pattern tracking, multi-persona support | See `FeedbackService`, session state tracking |
| **Voice Interaction** | Groq Whisper transcription, ElevenLabs TTS, automatic mode switching | See "Why Voice Notes?" section |

---

## 🤝 Contributing

This is a demo project for evaluation purposes. However, if you'd like to extend it:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📝 License

This project is created for evaluation purposes. All rights reserved.

---

## 🙏 Acknowledgments

- **Twilio** for WhatsApp API
- **DeepSeek** for cost-effective LLM
- **Groq** for fast Whisper transcription
- **ElevenLabs** for natural TTS voices
- **FastAPI** for excellent async framework
- **Pydantic** for data validation
- **Redis** for session storage

---

## 📧 Contact

For questions or feedback, please reach out via the evaluation platform.

---

**Built with ❤️ for realistic interview practice**
