"""Tests for circuit breakers, retry policy, and graceful LLM degradation."""

import pytest

from src.core.exceptions import CircuitOpenError, GenerationError
from src.core.resilience import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitState,
    call_with_resilience,
    is_transient_error,
    retry_policy,
)
from src.generation.llm import LLMFactory, invoke_with_fallback


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestCircuitBreaker:
    def test_starts_closed(self):
        breaker = CircuitBreaker("svc")
        assert breaker.state is CircuitState.CLOSED

    def test_opens_after_threshold(self):
        breaker = CircuitBreaker("svc", failure_threshold=2)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert breaker.state is CircuitState.OPEN

    def test_open_circuit_rejects_calls(self):
        breaker = CircuitBreaker("svc", failure_threshold=1)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

        with pytest.raises(CircuitOpenError) as exc_info:
            breaker.call(lambda: "should not run")
        assert exc_info.value.status_code == 503
        assert exc_info.value.details["retry_after"] >= 0

    def test_successes_do_not_open(self):
        breaker = CircuitBreaker("svc", failure_threshold=2)
        for _ in range(10):
            assert breaker.call(lambda: "ok") == "ok"
        assert breaker.state is CircuitState.CLOSED

    def test_half_open_after_recovery_timeout(self):
        clock = FakeClock()
        breaker = CircuitBreaker("svc", failure_threshold=1, recovery_timeout=10, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        assert breaker.state is CircuitState.OPEN

        clock.advance(9.9)
        assert breaker.state is CircuitState.OPEN
        clock.advance(0.2)
        assert breaker.state is CircuitState.HALF_OPEN

    def test_half_open_success_closes(self):
        clock = FakeClock()
        breaker = CircuitBreaker("svc", failure_threshold=1, recovery_timeout=5, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        clock.advance(6)

        assert breaker.call(lambda: "recovered") == "recovered"
        assert breaker.state is CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        clock = FakeClock()
        breaker = CircuitBreaker("svc", failure_threshold=1, recovery_timeout=5, clock=clock)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        clock.advance(6)
        assert breaker.state is CircuitState.HALF_OPEN

        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("still down")))
        assert breaker.state is CircuitState.OPEN

    def test_half_open_limits_concurrent_probes(self):
        clock = FakeClock()
        breaker = CircuitBreaker(
            "svc",
            failure_threshold=1,
            recovery_timeout=1,
            half_open_max_calls=1,
            clock=clock,
        )
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        clock.advance(2)

        breaker.before_call()  # first probe admitted
        with pytest.raises(CircuitOpenError):
            breaker.before_call()  # second probe rejected

    def test_reset_closes(self):
        breaker = CircuitBreaker("svc", failure_threshold=1)
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        breaker.reset()
        assert breaker.state is CircuitState.CLOSED
        assert breaker.snapshot()["failures"] == 0

    def test_registry_isolates_names(self):
        registry = CircuitBreakerRegistry()
        first = registry.get("a")
        second = registry.get("b")
        assert first is registry.get("a")
        assert first is not second

    def test_abandoned_half_open_probe_is_released(self):
        """A probe that never reports back (client disconnect) must not wedge
        the breaker in HALF_OPEN forever."""
        clock = FakeClock()
        breaker = CircuitBreaker(
            "svc",
            failure_threshold=1,
            recovery_timeout=5,
            half_open_max_calls=1,
            clock=clock,
        )
        with pytest.raises(RuntimeError):
            breaker.call(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        clock.advance(6)
        assert breaker.state is CircuitState.HALF_OPEN

        breaker.before_call()  # admitted, then abandoned (no success/failure)
        with pytest.raises(CircuitOpenError):
            breaker.before_call()

        clock.advance(6)  # past the probe timeout -> slot reclaimed
        breaker.before_call()  # admitted again instead of rejecting forever


class TestRetryPolicy:
    def test_retries_transient_errors(self):
        attempts = {"n": 0}

        @retry_policy(max_attempts=3, base_delay=0.0, max_delay=0.0)
        def flaky():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TimeoutError("timed out")
            return "ok"

        assert flaky() == "ok"
        assert attempts["n"] == 3

    def test_gives_up_after_max_attempts(self):
        attempts = {"n": 0}

        @retry_policy(max_attempts=2, base_delay=0.0, max_delay=0.0)
        def always_down():
            attempts["n"] += 1
            raise ConnectionError("connection refused")

        with pytest.raises(ConnectionError):
            always_down()
        assert attempts["n"] == 2

    def test_does_not_retry_permanent_errors(self):
        attempts = {"n": 0}

        @retry_policy(max_attempts=5, base_delay=0.0, max_delay=0.0)
        def invalid():
            attempts["n"] += 1
            raise ValueError("invalid api key")

        with pytest.raises(ValueError, match="invalid api key"):
            invalid()
        assert attempts["n"] == 1

    def test_is_transient_error(self):
        assert is_transient_error(TimeoutError("timed out"))
        assert is_transient_error(ConnectionError("connection reset"))
        assert is_transient_error(RuntimeError("HTTP 503 service unavailable"))
        assert is_transient_error(RuntimeError("429 too many requests"))
        assert not is_transient_error(ValueError("bad input"))
        assert not is_transient_error(CircuitOpenError("llm:gemini"))


class TestCallWithResilience:
    def test_breaker_sees_one_failure_after_retries_exhausted(self):
        calls = {"n": 0}

        def flaky() -> str:
            calls["n"] += 1
            raise TimeoutError("timed out")

        with pytest.raises(TimeoutError):
            call_with_resilience("svc", flaky, retry_on=is_transient_error)

        assert calls["n"] == 3  # retry budget exhausted inside one breaker call
        snapshot = breakers_snapshot_for("svc")
        assert snapshot["failures"] == 1
        assert snapshot["state"] == "closed"

    def test_rejected_when_open(self):
        calls = {"n": 0}

        def down() -> str:
            calls["n"] += 1
            raise ConnectionError("connection refused")

        # Default threshold is 5 failures; each call exhausts its retry budget
        # but still counts as a single breaker failure.
        for _ in range(5):
            with pytest.raises(ConnectionError):
                call_with_resilience("svc", down)
        before = calls["n"]
        with pytest.raises(CircuitOpenError):
            call_with_resilience("svc", down)
        assert calls["n"] == before  # rejected without invoking the function


def breakers_snapshot_for(name: str) -> dict:
    from src.core.resilience import breakers

    return breakers.snapshot()[name]


# ---------------------------------------------------------------------------
# Graceful degradation in the LLM chain
# ---------------------------------------------------------------------------


class FakeChatModel:
    def __init__(self, text: str) -> None:
        self._text = text

    def invoke(self, messages, **kwargs):
        return self._text


class FailingModel:
    def invoke(self, messages, **kwargs):
        raise RuntimeError("provider down")


class TestGracefulDegradation:
    def test_falls_back_to_degraded_model(self, monkeypatch):
        monkeypatch.setattr("src.generation.llm.settings.google_degraded_model", "small-model")

        factory = LLMFactory()
        factory._clients["gemini::"] = FailingModel()
        factory._clients["gemini:small-model:"] = FakeChatModel("degraded answer")

        outcome: dict = {}
        response = invoke_with_fallback(
            factory,
            [{"role": "user", "content": "hi"}],
            provider="gemini",
            outcome=outcome,
        )

        assert response == "degraded answer"
        assert outcome["provider"] == "gemini"
        assert outcome["model"] == "small-model"
        assert outcome["degraded"] is True

    def test_explicit_model_is_not_labelled_degraded(self):
        """A caller-requested model is full quality, not a degraded fallback."""
        factory = LLMFactory()
        factory._clients["gemini:custom-model:"] = FakeChatModel("answer")

        outcome: dict = {}
        response = invoke_with_fallback(
            factory,
            [{"role": "user", "content": "hi"}],
            provider="gemini",
            model="custom-model",
            outcome=outcome,
        )

        assert response == "answer"
        assert outcome == {
            "provider": "gemini",
            "model": "custom-model",
            "degraded": False,
        }

    def test_skips_provider_with_open_circuit(self):
        from src.core.resilience import breakers

        factory = LLMFactory()
        factory._clients["gemini::"] = FakeChatModel("should not be used")
        factory._clients["openrouter::"] = FakeChatModel("fallback answer")

        breaker = breakers.get("llm:gemini")
        breaker.before_call()
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()
        breaker.record_failure()

        response = invoke_with_fallback(
            factory,
            [{"role": "user", "content": "hi"}],
            provider="gemini",
        )
        assert response == "fallback answer"

    def test_all_providers_fail(self):
        factory = LLMFactory()
        for name in ("gemini::", "openrouter::", "groq::"):
            factory._clients[name] = FailingModel()

        with pytest.raises(GenerationError):
            invoke_with_fallback(
                factory,
                [{"role": "user", "content": "hi"}],
                provider="gemini",
            )
