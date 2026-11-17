"""Circuit breakers and retry helpers for external service calls.

Two primitives:

- :class:`CircuitBreaker` — stops hammering a failing dependency. It opens
  after N consecutive failures, rejects calls while open, then allows a single
  probe (half-open) after the recovery timeout before closing again.
- :func:`retry_policy` — tenacity retry with exponential backoff **plus
  jitter**, applied only to transient failures (timeouts, 429/5xx, connection
  errors). Permanent failures (auth, validation) are not retried.

:func:`call_with_resilience` combines both: retries happen *inside* the
breaker, so the breaker sees one failure after the retry budget is exhausted
rather than opening on a single blip.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum
from time import monotonic
from typing import Any, TypeVar

import httpx
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from src.core.config import settings
from src.core.exceptions import CircuitOpenError
from src.core.logging import get_logger
from src.core.metrics import metrics

log = get_logger(__name__)

T = TypeVar("T")

_TRANSIENT_MARKERS = (
    "timeout",
    "timed out",
    "connection",
    "temporarily",
    "temporary failure",
    "rate limit",
    "too many requests",
    "overloaded",
    "unavailable",
    "429",
    "500",
    "502",
    "503",
    "504",
)
_TRANSIENT_TYPES = {
    "timeouterror",
    "connecterror",
    "connectionerror",
    "readtimeout",
    "connecttimeout",
    "networkerror",
    "remoteprotocolerror",
    "apiconnectionerror",
    "internalservererror",
    "ratelimiterror",
    "serviceunavailableerror",
}


def is_transient_error(exc: BaseException) -> bool:
    """Return True for errors that are worth retrying."""
    if isinstance(exc, CircuitOpenError):
        return False
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        return True
    if type(exc).__name__.lower() in _TRANSIENT_TYPES:
        return True
    message = str(exc).lower()
    return any(marker in message for marker in _TRANSIENT_MARKERS)


def retry_policy(
    *,
    max_attempts: int | None = None,
    base_delay: float | None = None,
    max_delay: float | None = None,
    retry_on: Callable[[BaseException], bool] | None = None,
    service: str = "external",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Build a tenacity retry decorator with exponential backoff + jitter.

    Args:
        max_attempts: Total attempts (defaults to ``RETRY_MAX_ATTEMPTS``).
        base_delay: Initial backoff in seconds (``RETRY_BASE_DELAY``).
        max_delay: Backoff ceiling in seconds (``RETRY_MAX_DELAY``).
        retry_on: Predicate deciding whether an exception is retryable.
            Defaults to :func:`is_transient_error`.
        service: Label used in logs and metrics.
    """
    attempts = max(1, max_attempts if max_attempts is not None else settings.retry_max_attempts)
    base = base_delay if base_delay is not None else settings.retry_base_delay
    cap = max_delay if max_delay is not None else settings.retry_max_delay
    predicate = retry_on or is_transient_error

    def _before_sleep(state: RetryCallState) -> None:
        outcome = state.outcome
        error = str(outcome.exception()) if outcome is not None else "unknown"
        metrics.increment("resilience.retries", labels={"service": service})
        log.warning(
            "retry_scheduled",
            service=service,
            attempt=state.attempt_number,
            next_attempt=state.attempt_number + 1,
            error=error,
        )

    return retry(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=max(base, 0.01), min=base, max=cap)
        + wait_random(0, max(base / 2, 0.01)),
        retry=retry_if_exception(predicate),
        reraise=True,
        before_sleep=_before_sleep,
    )


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Thread-safe circuit breaker for one external dependency."""

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int | None = None,
        recovery_timeout: float | None = None,
        half_open_max_calls: int | None = None,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.name = name
        self._failure_threshold = (
            failure_threshold
            if failure_threshold is not None
            else settings.circuit_failure_threshold
        )
        self._recovery_timeout = (
            recovery_timeout if recovery_timeout is not None else settings.circuit_recovery_timeout
        )
        self._half_open_max_calls = (
            half_open_max_calls
            if half_open_max_calls is not None
            else settings.circuit_half_open_max_calls
        )
        self._clock = clock
        self._lock = threading.RLock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_calls = 0
        self._half_open_started_at = 0.0

    # -- state ---------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._refresh_locked()
            return self._state

    def _transition(self, new_state: CircuitState) -> None:
        if new_state == self._state:
            return
        previous = self._state
        self._state = new_state
        metrics.increment(
            "circuit.state_changes",
            labels={"breaker": self.name, "state": new_state.value},
        )
        log.warning(
            "circuit_state_changed",
            breaker=self.name,
            previous=previous.value,
            current=new_state.value,
        )

    def _refresh_locked(self) -> None:
        """Advance time-based state transitions.

        * OPEN -> HALF_OPEN once the recovery timeout has elapsed.
        * HALF_OPEN -> HALF_OPEN with released slots when a probe was
          abandoned (e.g. the caller disconnected mid-stream and neither
          `record_success` nor `record_failure` ran). Without this the breaker
          would reject every request forever.
        """
        if self._state is CircuitState.OPEN:
            elapsed = self._clock() - self._opened_at
            if elapsed >= self._recovery_timeout:
                self._half_open_calls = 0
                self._half_open_started_at = 0.0
                self._transition(CircuitState.HALF_OPEN)
            return

        if (
            self._state is CircuitState.HALF_OPEN
            and self._half_open_calls
            and self._clock() - self._half_open_started_at >= self._recovery_timeout
        ):
            log.warning("circuit_probe_abandoned", breaker=self.name)
            self._half_open_calls = 0
            self._half_open_started_at = 0.0

    # -- lifecycle -----------------------------------------------------
    def before_call(self) -> None:
        """Admit a call or raise :class:`CircuitOpenError`."""
        with self._lock:
            self._refresh_locked()
            if self._state is CircuitState.OPEN:
                retry_after = max(0.0, self._recovery_timeout - (self._clock() - self._opened_at))
                metrics.increment("circuit.rejected", labels={"breaker": self.name})
                raise CircuitOpenError(self.name, retry_after=retry_after)
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._half_open_max_calls:
                    metrics.increment("circuit.rejected", labels={"breaker": self.name})
                    raise CircuitOpenError(self.name, retry_after=self._recovery_timeout)
                if self._half_open_calls == 0:
                    self._half_open_started_at = self._clock()
                self._half_open_calls += 1

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_calls = 0
            self._half_open_started_at = 0.0
            if self._state is not CircuitState.CLOSED:
                self._transition(CircuitState.CLOSED)

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN or self._failures >= self._failure_threshold:
                self._opened_at = self._clock()
                self._half_open_calls = 0
                self._half_open_started_at = 0.0
                self._transition(CircuitState.OPEN)

    def reset(self) -> None:
        """Force the breaker closed and clear counters (tests/admin)."""
        with self._lock:
            self._failures = 0
            self._half_open_calls = 0
            self._half_open_started_at = 0.0
            self._opened_at = 0.0
            self._transition(CircuitState.CLOSED)

    # -- execution -----------------------------------------------------
    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Execute ``func`` through the breaker."""
        self.before_call()
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            self._refresh_locked()
            retry_after = 0.0
            if self._state is CircuitState.OPEN:
                retry_after = max(0.0, self._recovery_timeout - (self._clock() - self._opened_at))
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "threshold": self._failure_threshold,
                "retry_after": round(retry_after, 3),
            }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"CircuitBreaker(name={self.name!r}, state={self.state.value})"


class CircuitBreakerRegistry:
    """Lazily-created, thread-safe collection of named breakers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        breaker = self._breakers.get(name)
        if breaker is None:
            with self._lock:
                breaker = self._breakers.get(name)
                if breaker is None:
                    breaker = CircuitBreaker(name)
                    self._breakers[name] = breaker
        return breaker

    def call(self, name: str, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        return self.get(name).call(func, *args, **kwargs)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {name: breaker.snapshot() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        for breaker in self._breakers.values():
            breaker.reset()

    def clear(self) -> None:
        """Drop all breaker instances (tests)."""
        with self._lock:
            self._breakers.clear()


breakers = CircuitBreakerRegistry()


def call_with_resilience(
    service: str,
    func: Callable[..., T],
    *args: Any,
    retry_on: Callable[[BaseException], bool] | None = None,
    **kwargs: Any,
) -> T:
    """Run ``func`` with retries inside :class:`CircuitBreaker` protection."""
    resilient = retry_policy(retry_on=retry_on, service=service)(func)
    return breakers.get(service).call(resilient, *args, **kwargs)
