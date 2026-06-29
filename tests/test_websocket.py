import json

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_websocket_session_lifecycle(client, has_gemini):
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({
            "type": "session.start",
            "portfolio": "AAPL:10",
            "user_id": "ws-test",
        })
        msg = ws.receive_json()
        assert msg["type"] == "session.ready"
        assert msg.get("agent_name")

        if not has_gemini:
            return

        ws.send_json({
            "type": "text.query",
            "text": "hello",
            "generation_id": msg["generation_id"],
        })

        types_seen = set()
        for _ in range(30):
            try:
                m = ws.receive_json()
                types_seen.add(m["type"])
                if m["type"] in ("response.complete", "error"):
                    break
            except Exception:
                break
        assert len(types_seen) >= 1


def test_websocket_interrupt_ack(client):
    with client.websocket_connect("/ws/voice") as ws:
        ws.send_json({"type": "session.start", "portfolio": "AAPL:10"})
        ready = ws.receive_json()
        gen = ready["generation_id"]
        ws.send_json({"type": "interrupt", "generation_id": gen})
        ack = ws.receive_json()
        assert ack["type"] == "interrupt.ack"
        assert ack["generation_id"] != gen