import base64
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from graph.agent_graph import run_agent
from graph.nodes.market import api_node
from graph.state import AgentState
from providers.factory import build_tts_chain

logger = logging.getLogger(__name__)
router = APIRouter()


class APIDataRequest(BaseModel):
    ticker: str


@router.post("/api/run")
async def api_run(request: APIDataRequest):
    state: AgentState = {"portfolio": request.ticker, "errors": []}
    state = await api_node(state)
    if state.get("error"):
        raise HTTPException(status_code=400, detail=state["error"])
    return state.get("market_data", {})


@router.post("/run")
async def run_legacy(
    query: Optional[str] = Form(None),
    portfolio: str = Form(...),
    user_id: str = Form("default"),
    audio: Optional[UploadFile] = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
):
    """Legacy REST endpoint — WebSocket /ws/voice is the primary path."""
    try:
        final_query = query or ""
        if audio:
            audio_bytes = await audio.read()
            if audio_bytes:
                from providers.factory import build_stt_chain
                stt = build_stt_chain()
                final_query, _ = await stt.execute("transcribe", audio_bytes)

        if not final_query:
            final_query = "Provide an update on my portfolio."

        state: AgentState = {
            "query": final_query,
            "portfolio": portfolio,
            "user_id": user_id,
            "errors": [],
            "charts": [],
        }
        result = await run_agent(state, thread_id=f"rest_{user_id}")

        audio_output = ""
        narrative = result.get("narrative_for_audio") or result.get("narrative", "")
        if narrative:
            try:
                tts = build_tts_chain()
                chunks = []
                stream, _ = tts.get_stream("synthesize_stream", narrative)
                async for c in stream:
                    chunks.append(c)
                if chunks:
                    audio_output = f"data:audio/wav;base64,{base64.b64encode(b''.join(chunks)).decode()}"
            except Exception as e:
                logger.error(f"TTS failed: {e}")

        return JSONResponse({
            "narrative": result.get("narrative", ""),
            "audio_file": audio_output,
            "audio_transcript": result.get("narrative", ""),
            "analysis": result.get("analysis_result", {}),
            "query": final_query,
            "news": result.get("news", []),
            "charts": result.get("charts", []),
            "error": result.get("error", ""),
            "safe_mode": result.get("safe_mode", False),
        })
    except Exception as e:
        logger.exception("Legacy pipeline error")
        raise HTTPException(status_code=500, detail=str(e))