# Interview Practice Partner — Setup Guide

This guide walks you through everything needed to run the Interview Practice Partner locally and connect it to WhatsApp via Twilio.

---

## Prerequisites

Before starting, make sure you have the following installed:

| Tool | Version | Download |
|---|---|---|
| Python | 3.11 or higher | https://www.python.org/downloads/ |
| Docker Desktop | Latest | https://www.docker.com/products/docker-desktop/ |
| ngrok | Latest | https://ngrok.com/download |
| Git | Any | https://git-scm.com/ |

You also need accounts at:
- **Twilio** — https://www.twilio.com/try-twilio (free trial is fine)
- **ngrok** — https://dashboard.ngrok.com/signup (free account is fine)
- **DeepSeek** — https://platform.deepseek.com (for the LLM API key)
- **Groq** — https://console.groq.com (for voice note transcription - free tier available)
- **ElevenLabs** — https://elevenlabs.io (for text-to-speech - free tier available)

---

## Step 1 — Clone and Install

```bash
# Clone the repository
git clone <your-repo-url>
cd interview-practice-partner

# Install all dependencies (including dev tools)
pip install -e ".[dev]"
```

Verify the install worked:

```bash
python -m pytest tests/ --tb=short -q
```

You should see `616 passed` with no failures.

---

## Step 2 — Configure Environment Variables

Copy the example env file and fill in your real credentials:

```bash
cp .env.example .env
```

Now open `.env` in a text editor and fill in every value:

```bash
# ── Twilio ──────────────────────────────────────────────────────────────────
# Get these from: https://console.twilio.com → Account Info
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_WHATSAPP_NUMBER=whatsapp:+14155238886

# ── LLM Provider (DeepSeek) ─────────────────────────────────────────────────
# Get your key from: https://platform.deepseek.com/api_keys
LLM_API_KEY=sk-your-deepseek-key-here
LLM_API_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-chat

# ── Groq Whisper (Voice Note Transcription) ─────────────────────────────────
# Get your key from: https://console.groq.com/keys
GROQ_API_KEY=gsk_your_groq_key_here
GROQ_WHISPER_MODEL=whisper-large-v3-turbo

# ── ElevenLabs (Text-to-Speech) ─────────────────────────────────────────────
# Get your key from: https://elevenlabs.io/app/settings/api-keys
# Browse voices at: https://elevenlabs.io/voice-library
ELEVENLABS_API_KEY=your_elevenlabs_key_here
# Recommended voices:
# - Neha (Nj17Z4VDrfZaOdsqTaPL) - Indian English, professional
# - Sarah (EXAVITQu4vr4xnSDxMaL) - American English, professional
ELEVENLABS_VOICE_ID=Nj17Z4VDrfZaOdsqTaPL
ELEVENLABS_MODEL_ID=eleven_turbo_v2_5

# ── Media Serving (for voice note audio files) ──────────────────────────────
# Set to your ngrok URL in development (e.g., https://abc123.ngrok-free.app)
# Set to your public hostname in production
MEDIA_BASE_URL=
MEDIA_TTL_SECONDS=600

# ── Redis ────────────────────────────────────────────────────────────────────
# Leave this as-is if running Redis locally via Docker
REDIS_URL=redis://localhost:6379/0
```

### Where to find your Twilio credentials

1. Log in at https://console.twilio.com
2. On the dashboard homepage, scroll down to **Account Info**
3. You will see:
   - **Account SID** — starts with `AC`
   - **Auth Token** — click the eye icon to reveal it
4. Copy both into your `.env` file

### Where to find your Twilio WhatsApp number

1. In the Twilio console, go to **Messaging → Try it out → Send a WhatsApp message**
2. The sandbox number is shown at the top of that page (e.g. `+1 415 523 8886`)
3. Format it as `whatsapp:+14155238886` in your `.env`

### Where to find your Groq API key

1. Sign up at https://console.groq.com
2. Go to **API Keys** in the left sidebar
3. Click **Create API Key**
4. Copy the key (starts with `gsk_`) into your `.env`

### Where to find your ElevenLabs API key and voice ID

1. Sign up at https://elevenlabs.io
2. Go to **Profile → API Keys** (https://elevenlabs.io/app/settings/api-keys)
3. Copy your API key into your `.env`
4. Browse voices at https://elevenlabs.io/voice-library or use the recommended voices in `.env.example`
5. For Indian English accent, use **Neha** (`Nj17Z4VDrfZaOdsqTaPL`)
6. For American English accent, use **Sarah** (`EXAVITQu4vr4xnSDxMaL`)

### Setting MEDIA_BASE_URL

The `MEDIA_BASE_URL` is required for voice note replies to work. Set it to your ngrok URL:

```bash
# In development (update this every time you restart ngrok)
MEDIA_BASE_URL=https://abc123def456.ngrok-free.app

# In production (your actual domain)
MEDIA_BASE_URL=https://yourdomain.com
```

**Important:** Update this value every time you restart ngrok, or use a static ngrok domain (see troubleshooting section).

---

## Step 3 — Start Redis

Redis stores all session state. Start it with Docker:

```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

Or if you prefer docker-compose (already configured in the project):

```bash
docker-compose up redis -d
```

Verify Redis is running:

```bash
docker ps
```

You should see a container named `redis` with status `Up`.

---

## Step 4 — Start the Application

```bash
uvicorn src.interview_practice_partner.main:app --host 0.0.0.0 --port 8000 --reload
```

Expected output:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

Verify the app is healthy by opening a second terminal and running:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "redis_connected": true,
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

If `redis_connected` is `false`, Redis is not running — go back to Step 3.

---

## Step 5 — Start ngrok

ngrok creates a public HTTPS tunnel to your local app so Twilio can reach it.

**First time only — authenticate ngrok:**

1. Log in at https://dashboard.ngrok.com
2. Go to **Your Authtoken** in the left sidebar
3. Copy your authtoken and run:

```bash
ngrok config add-authtoken YOUR_AUTHTOKEN_HERE
```

**Start the tunnel** (in a separate terminal, keep it running):

```bash
ngrok http 8000
```

You will see output like this:

```
Session Status                online
Account                       your@email.com
Version                       3.x.x
Region                        United States (us)
Forwarding                    https://abc123def456.ngrok-free.app -> http://localhost:8000

Connections                   ttl     opn     rt1     rt5     p50     p90
                              0       0       0.00    0.00    0.00    0.00
```

**Copy the `https://` URL** — in this example it is `https://abc123def456.ngrok-free.app`.

You will need this URL in the next step. Keep this terminal open — closing it stops the tunnel.

> **Note:** The free ngrok plan gives you a new random URL every time you restart ngrok. If you want a permanent URL, ngrok offers one free static domain per account. Go to **Domains** in the ngrok dashboard to claim yours.

---

## Step 6 — Configure Twilio Webhook

Now you need to tell Twilio to send incoming WhatsApp messages to your app.

1. Log in at https://console.twilio.com
2. In the left sidebar, click **Messaging**
3. Click **Try it out**
4. Click **Send a WhatsApp message**
5. You are now on the **WhatsApp Sandbox** page
6. Scroll down to the **Sandbox Settings** section
7. Fill in the two fields:

**"When a message comes in"** field:
```
https://abc123def456.ngrok-free.app/webhook/whatsapp
```
Set the dropdown next to it to **HTTP POST**

**"Status callback URL"** field:
```
https://abc123def456.ngrok-free.app/webhook/status
```

8. Click **Save** at the bottom

It should look like this:

```
┌─────────────────────────────────────────────────────────────────┐
│ Sandbox Settings                                                │
│                                                                 │
│ When a message comes in:                                        │
│ [https://abc123def456.ngrok-free.app/webhook/whatsapp] [POST▼] │
│                                                                 │
│ Status callback URL:                                            │
│ [https://abc123def456.ngrok-free.app/webhook/status]           │
│                                                                 │
│                                              [Save]            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step 7 — Join the WhatsApp Sandbox

Before you can receive messages, your phone number must join the sandbox.

1. On the same WhatsApp Sandbox page, look for the **"Join the sandbox"** section at the top
2. You will see a join code like: `join <word>-<word>` (e.g. `join apple-mango`)
3. Open **WhatsApp** on your phone
4. Send that exact message to the Twilio sandbox number shown on the page (e.g. `+1 415 523 8886`)
5. You will receive a confirmation reply from Twilio

You are now connected to the sandbox.

---

## Step 8 — Test the Full Flow

Send a WhatsApp message from your phone to the Twilio sandbox number. Try these in order:

| You send | Expected reply |
|---|---|
| `Hello` | Agent asks which role you want to practise for |
| `Software Engineer` | Agent starts the interview with a question |
| Any answer (15+ words) | Agent evaluates and asks next question |
| `skip` | Agent skips to the next question |
| Any short answer (under 15 words) | Agent asks you to elaborate |
| **🎤 Voice note** | Agent transcribes it and responds with audio (if in voice mode) |
| `voice mode` | Agent switches to voice mode (replies with audio) |
| `text mode` | Agent switches to text mode (replies with text) |
| After 5 questions answered | Agent delivers structured feedback report |

### Voice Note Features

The bot now supports **voice notes** and **text-to-speech replies**:

- **Send a voice note** → Bot transcribes it via Groq Whisper and automatically switches to voice mode
- **Type "voice mode"** → Bot replies with audio messages (via ElevenLabs TTS)
- **Type "text mode"** → Bot replies with text messages
- **Auto mode switching** → Bot automatically detects your input type and switches modes accordingly

Voice notes are transcribed using Groq's Whisper API, and audio replies are generated using ElevenLabs TTS with the voice you configured in `.env`.

---

## Step 9 — Monitor Logs

Watch the uvicorn terminal while testing. Every event is logged as structured JSON:

```json
{"event": "webhook_request", "method": "POST", "path": "/webhook/whatsapp", "from_number": "whatsapp:+447700900000", "correlation_id": "SM123..."}
{"event": "audio_download.start", "media_url": "https://api.twilio.com/...", "correlation_id": "SM123..."}
{"event": "groq_whisper.transcribe_complete", "latency_ms": 850, "transcript_length": 42, "correlation_id": "SM123..."}
{"event": "llm_call", "model": "deepseek-chat", "tokens_used": 312, "latency_ms": 1240, "correlation_id": "SM123..."}
{"event": "elevenlabs.synthesise_complete", "latency_ms": 1100, "audio_size_bytes": 45678, "correlation_id": "SM123..."}
{"event": "session_saved", "phone_number": "whatsapp:+447700900000", "stage": "INTERVIEW", "preferred_mode": "voice", "correlation_id": "SM123..."}
{"event": "twilio_send", "to": "whatsapp:+447700900000", "media_url": "https://abc123.ngrok-free.app/media/uuid.mp3", "correlation_id": "SM123..."}
```

You can also watch the ngrok web interface at http://localhost:4040 to inspect every HTTP request and response in detail.

---

## Troubleshooting

### Getting HTTP 403 from the webhook

The Twilio signature validation is failing. Check:
- Your `TWILIO_AUTH_TOKEN` in `.env` exactly matches the one in the Twilio console
- You are using the `https://` ngrok URL in Twilio (not `http://`)
- The URL in Twilio sandbox settings exactly matches what ngrok shows (no trailing slash)

### Redis connection error in health check

```json
{"status": "degraded", "redis_connected": false}
```

Redis is not running. Run:

```bash
docker start redis
```

Or start it fresh:

```bash
docker run -d -p 6379:6379 --name redis redis:7-alpine
```

### LLM not responding / timeout errors

- Verify your DeepSeek API key is correct in `.env`
- Check your DeepSeek account has credits at https://platform.deepseek.com
- The app will retry twice automatically before sending a fallback message to the user

### Voice note transcription failing

- Verify your `GROQ_API_KEY` is correct in `.env`
- Check your Groq account at https://console.groq.com
- The app will send a fallback message asking the user to resend or type their answer

### Audio replies not working / TTS errors

- Verify your `ELEVENLABS_API_KEY` is correct in `.env`
- Check your ElevenLabs account has credits at https://elevenlabs.io
- Verify `MEDIA_BASE_URL` is set to your ngrok URL (e.g., `https://abc123.ngrok-free.app`)
- The app will fall back to text replies if TTS fails

### Audio files not accessible (404 errors)

- Make sure `MEDIA_BASE_URL` in `.env` matches your current ngrok URL exactly
- Check that the `/media` endpoint is accessible: `curl https://your-ngrok-url.ngrok-free.app/health`
- Audio files are automatically cleaned up after 10 minutes (configurable via `MEDIA_TTL_SECONDS`)

### ngrok URL changed after restart

Every time you restart ngrok, you get a new URL. You must update the Twilio sandbox settings (Step 6) with the new URL each time.

To avoid this, claim your free static ngrok domain:
1. Go to https://dashboard.ngrok.com/domains
2. Click **New Domain** — you get one free static domain
3. Start ngrok with: `ngrok http --domain=your-static-domain.ngrok-free.app 8000`
4. Set that permanent URL in Twilio once and never change it again

### Messages not arriving / Twilio not calling the webhook

- Make sure your phone has joined the sandbox (Step 7) — the join code expires after a while, you may need to rejoin
- Check the ngrok terminal — you should see `POST /webhook/whatsapp 200 OK` for each message
- Check the Twilio console under **Monitor → Logs → Messaging** for delivery errors

---

## Running with Docker Compose (Optional)

If you want to run everything (app + Redis) together:

```bash
docker-compose up --build
```

The app will be available at `http://localhost:8000`. Then run ngrok pointing at port 8000 as described in Step 5.

---

## Quick Reference

| Command | Purpose |
|---|---|
| `uvicorn src.interview_practice_partner.main:app --reload` | Start the app |
| `ngrok http 8000` | Start the public tunnel |
| `docker run -d -p 6379:6379 --name redis redis:7-alpine` | Start Redis |
| `curl http://localhost:8000/health` | Check app + Redis health |
| `python -m pytest tests/ -q` | Run all 616 tests |
| `docker-compose up -d` | Start everything with Docker Compose |
| `docker-compose down` | Stop everything |
