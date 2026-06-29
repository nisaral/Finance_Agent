# Voice-First Financial Assistant

Production-grade voice agent for portfolio analysis, news, and real-time streaming responses.

## Features

- **WebSocket streaming** — sub-500ms TTFB via sentence-boundary LLM→TTS pipelining
- **Barge-in** — VAD-triggered interrupt cancels LLM stream and flushes TTS queue
- **LangGraph state machine** — portfolio verification, news analysis, visualization
- **Agent Debugger Console** — live state visualizer, latency breakdown, trace IDs
- **Safe mode** — graceful degradation when NewsAPI fails

## Tech Stack

| Layer | Stack |
|-------|-------|
| Backend | Python, FastAPI, LangGraph |
| STT/TTS | Cartesia (Ink + Sonic) |
| LLM | Google Gemini 2.0 Flash |
| Market/News | yfinance, NewsAPI |
| Observability | OpenTelemetry + in-memory trace store |

## Quick Start

```bash
git clone https://github.com/nisaral/Finance_Agent.git
cd Finance_Agent
python -m venv venv
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env      # add your API keys
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000` → **Connect WebSocket** → send a voice or text query.

## Environment Variables

```
CARTESIA_API_KEY=
GEMINI_API_KEY=
GEMINI_API_KEY_FALLBACK=
NEWSAPI_KEY=
SESSION_STORE=memory
```

Never commit `.env` — it is gitignored.

## Tests

```powershell
.\scripts\run_tests.ps1
```

## Deployment

Demo: Render/Railway with WebSocket keep-alive. Production voice: VPS (DigitalOcean/AWS) to avoid PaaS proxy timeouts on long-lived WebSocket connections.