from fastapi import APIRouter, HTTPException

from observability.log_store import trace_store

router = APIRouter()


@router.get("/api/traces")
async def list_traces(limit: int = 20):
    return {"traces": trace_store.list_recent(limit)}


@router.get("/api/traces/{trace_id}")
async def get_trace(trace_id: str):
    record = trace_store.get(trace_id)
    if not record:
        raise HTTPException(status_code=404, detail="Trace not found")
    return record