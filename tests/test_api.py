"""API tests: validation, auth, rate limiting, headers, and error envelopes."""

import pytest
from fastapi.testclient import TestClient

from src.core.config import settings
from src.generation.chain import GenerationResult
from src.generation.citations import Citation
from src.generation.tokens import TokenUsage

API_KEY = "test-key-" + "a" * 24  # placeholder, not a real credential
AUTH = {"X-API-Key": API_KEY}


def _fake_generation() -> GenerationResult:
    return GenerationResult(
        answer="Answer text [doc-1].",
        citations=[
            Citation(
                key="doc-1",
                chunk_id="chunk-1",
                document_id="doc-1",
                content="chunk content",
                source="notes.txt",
                section_header="Intro",
                page_number=1,
                score=0.9,
            )
        ],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost_usd=0.0,
        provider="gemini",
        model="gemini-2.5-flash-lite",
        sources_markdown="Sources:\n1. notes.txt",
    )


@pytest.fixture
def client(monkeypatch):
    from src.api.app import create_app

    monkeypatch.setattr(settings, "api_key", API_KEY)
    monkeypatch.setattr(settings, "require_auth", True)
    monkeypatch.setattr(settings, "enable_rate_limit", True)
    monkeypatch.setattr(settings, "rate_limit_requests", 100)
    monkeypatch.setattr(settings, "rate_limit_window", 60)
    monkeypatch.setattr(settings, "max_request_bytes", 8192)
    monkeypatch.setattr(settings, "enable_cache", False)
    monkeypatch.setattr("src.api.routes._ask_sync", lambda payload: (_fake_generation(), 12.5))
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client


class TestHealth:
    def test_health_is_public_and_sets_headers(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-Request-ID"]

    def test_request_id_is_echoed(self, client):
        response = client.get("/health", headers={"X-Request-ID": "trace-abc-123"})
        assert response.headers["X-Request-ID"] == "trace-abc-123"

    def test_invalid_request_id_is_replaced(self, client):
        response = client.get("/health", headers={"X-Request-ID": "bad id with spaces"})
        assert response.headers["X-Request-ID"] != "bad id with spaces"


class TestAuth:
    def test_missing_key_is_rejected(self, client):
        response = client.post("/api/v1/ask", json={"question": "hello"})
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == "AUTHENTICATION_ERROR"
        assert body["error"]["request_id"]

    def test_wrong_key_is_rejected(self, client):
        response = client.post(
            "/api/v1/ask", json={"question": "hello"}, headers={"X-API-Key": "nope"}
        )
        assert response.status_code == 401

    def test_bearer_token_is_accepted(self, client):
        response = client.post(
            "/api/v1/ask",
            json={"question": "hello"},
            headers={"Authorization": f"Bearer {API_KEY}"},
        )
        assert response.status_code == 200


class TestAskEndpoint:
    def test_returns_answer_with_citations(self, client):
        response = client.post("/api/v1/ask", json={"question": "hello"}, headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Answer text [doc-1]."
        assert body["citations"][0]["source"] == "notes.txt"
        assert body["provider"] == "gemini"
        assert body["usage"]["total_tokens"] == 15
        assert body["latency_ms"] == pytest.approx(12.5)
        assert body["request_id"]
        assert response.headers["X-RateLimit-Limit"] == "100"

    @pytest.mark.parametrize(
        "payload",
        [
            {"question": ""},
            {"question": "   "},
            {"question": "ok", "top_k": 0},
            {"question": "ok", "top_k": 51},
            {"question": "ok", "score_threshold": 1.5},
            {"question": "ok", "search_type": "magic"},
            {"question": "ok", "tenant_id": "bad tenant!"},
            {"question": "x" * 2001},
            {"question": "ok", "unexpected": True},
            {"question": "ok", "filters": {"created_after": "not-a-date"}},
        ],
    )
    def test_rejects_invalid_input(self, client, payload):
        response = client.post("/api/v1/ask", json=payload, headers=AUTH)
        assert response.status_code == 422
        body = response.json()
        assert body["error"]["code"] == "VALIDATION_ERROR"
        errors = body["error"]["details"]["errors"]
        assert errors
        # Validation details must not echo raw input back to the client: only
        # location, message, and type are forwarded (Pydantic also emits
        # `input` and `ctx`, which can contain the raw payload).
        for error in errors:
            assert set(error) == {"loc", "msg", "type"}
        assert "bad tenant" not in response.text

    def test_rejects_control_characters(self, client):
        response = client.post("/api/v1/ask", json={"question": "hi\x00there"}, headers=AUTH)
        assert response.status_code == 422

    def test_payload_too_large(self, client, monkeypatch):
        monkeypatch.setattr(settings, "max_request_bytes", 1024)
        response = client.post("/api/v1/ask", json={"question": "x" * 2000}, headers=AUTH)
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


class TestRateLimit:
    def test_second_request_is_limited(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        first = client.post("/api/v1/ask", json={"question": "one"}, headers=AUTH)
        second = client.post("/api/v1/ask", json={"question": "two"}, headers=AUTH)

        assert first.status_code == 200
        assert second.status_code == 429
        body = second.json()
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert body["error"]["details"]["limit"] == 1
        assert second.headers["Retry-After"]
        assert second.headers["X-RateLimit-Remaining"] == "0"

    def test_health_is_not_rate_limited(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_requests", 1)
        assert client.get("/health").status_code == 200
        assert client.get("/health").status_code == 200


class TestIngestValidation:
    @pytest.mark.parametrize(
        "payload",
        [
            {"path": "../secrets.txt"},
            {"path": "C:\\temp\\payload.exe"},
            {"path": ""},
            {"path": "notes.txt", "tenant_id": "bad tenant!"},
        ],
    )
    def test_rejects_unsafe_paths(self, client, payload):
        response = client.post("/api/v1/ingest", json=payload, headers=AUTH)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "VALIDATION_ERROR"


class TestErrorHandling:
    def test_unknown_route_uses_error_envelope(self, client):
        response = client.get("/v1/models")
        assert response.status_code == 404
        body = response.json()
        assert body["error"]["code"] == "NOT_FOUND"
        assert body["error"]["request_id"]
        assert "detail" not in body

    def test_wrong_method_uses_error_envelope(self, client):
        response = client.get("/api/v1/ask")
        assert response.status_code == 405
        assert response.json()["error"]["code"] == "METHOD_NOT_ALLOWED"

    def test_unhandled_error_returns_generic_envelope(self, monkeypatch):
        from src.api.app import create_app

        monkeypatch.setattr(settings, "api_key", API_KEY)
        monkeypatch.setattr(settings, "require_auth", True)
        monkeypatch.setattr(settings, "enable_rate_limit", False)

        def boom(payload):
            raise RuntimeError("internal detail should not leak")

        monkeypatch.setattr("src.api.routes._ask_sync", boom)
        app = create_app()
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/api/v1/ask", json={"question": "hello"}, headers=AUTH)

        assert response.status_code == 500
        body = response.json()
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert "internal detail" not in response.text
