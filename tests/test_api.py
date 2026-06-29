import pytest
from fastapi.testclient import TestClient

from main import app
from observability.log_store import trace_store


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_config_endpoint(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "name" in data
    assert data["safe_mode"] is True


def test_traces_list(client):
    trace_id = trace_store.start_trace("call-test", "gen-test")
    trace_store.set_latency(trace_id, stt_ms=100, llm_ms=200, tts_ms=150)
    trace_store.finish_trace(trace_id, success=True)
    r = client.get("/api/traces")
    assert r.status_code == 200
    traces = r.json()["traces"]
    assert any(t["trace_id"] == trace_id for t in traces)


def test_trace_detail(client):
    trace_id = trace_store.start_trace("call-detail", "gen-detail")
    trace_store.add_event(trace_id, "stt.finalize", duration_ms=120)
    trace_store.finish_trace(trace_id)
    r = client.get(f"/api/traces/{trace_id}")
    assert r.status_code == 200
    assert r.json()["trace_id"] == trace_id
    assert len(r.json()["events"]) >= 1


def test_api_run_market(client):
    r = client.post("/api/run", json={"ticker": "AAPL:10"})
    assert r.status_code == 200
    data = r.json()
    assert "stocks" in data


def test_legacy_run_text(client, all_live_keys):
    if not all_live_keys:
        pytest.skip("Live API keys required for /run integration test")
    r = client.post(
        "/run",
        data={"portfolio": "AAPL:10", "query": "brief portfolio update", "user_id": "test"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("narrative") or body.get("query")