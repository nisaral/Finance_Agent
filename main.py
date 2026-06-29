import asyncio
import logging
import os

import aiohttp
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.health import router as health_router
from api.rest import router as rest_router
from api.traces import router as traces_router
from api.websocket import router as ws_router
from core.config_loader import load_agent_config
from graph.nodes.shared import init_db
from observability.tracing import init_telemetry

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REQUIRED_VARS = ["CARTESIA_API_KEY", "GEMINI_API_KEY", "NEWSAPI_KEY"]
for var in REQUIRED_VARS:
    if not os.getenv(var):
        logger.warning(f"{var} not set — some features will be degraded")

init_db()

app = FastAPI(title="Finance Agent — Voice Debugger Console")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)
app.include_router(rest_router)
app.include_router(traces_router)
app.include_router(ws_router)

app.mount("/static", StaticFiles(directory="."), name="static")


@app.on_event("startup")
async def startup():
    init_telemetry(os.getenv("OTEL_SERVICE_NAME", "finance-agent"))
    load_agent_config()
    asyncio.create_task(_keepalive_ping())
    logger.info("Finance Agent started — WebSocket: /ws/voice")


async def _keepalive_ping():
    """Render keep-alive: prevent cold-start on idle services."""
    url = os.getenv("KEEPALIVE_URL", "http://127.0.0.1:8000/health")
    while True:
        await asyncio.sleep(240)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    logger.debug(f"Keepalive ping: {resp.status}")
        except Exception:
            pass


@app.get("/")
async def serve_index():
    return FileResponse("index.html")


@app.get("/api/config")
async def get_config():
    cfg = load_agent_config()
    return {
        "name": cfg.name,
        "compliance_mode": cfg.compliance_mode,
        "safe_mode": cfg.safe_mode.enabled,
        "latency": {
            "jitter_buffer_ms": cfg.latency.jitter_buffer_ms,
            "stt_endpointing_ms": cfg.latency.stt_endpointing_ms,
        },
    }