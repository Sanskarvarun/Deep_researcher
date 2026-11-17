"""In-memory metrics registry.

Tracks counters (requests, errors, ...) and latency histograms so the
service can derive P50/P95/P99 latency, throughput, and error rates without
depending on an external metrics backend. Thread-safe and dependency-free;
a Prometheus / OTLP exporter can be layered on top later.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Generator
from contextlib import contextmanager
from functools import wraps
from threading import Lock
from time import perf_counter
from typing import Any, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class _Histogram:
    """Rolling latency histogram with percentile estimation."""

    def __init__(self) -> None:
        self._samples: list[float] = []
        self._sum = 0.0

    def observe(self, value: float) -> None:
        self._samples.append(value)
        self._sum += value

    def snapshot(self) -> dict[str, float]:
        if not self._samples:
            return {
                "count": 0,
                "sum": 0.0,
                "avg": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
            }
        ordered = sorted(self._samples)
        n = len(ordered)

        def percentile(q: float) -> float:
            k = (q / 100.0) * (n - 1)
            f = int(k)
            c = min(f + 1, n - 1)
            if f == c:
                return ordered[f]
            return ordered[f] + (ordered[c] - ordered[f]) * (k - f)

        return {
            "count": float(n),
            "sum": round(self._sum, 6),
            "avg": round(self._sum / n, 6),
            "p50": round(percentile(50), 6),
            "p95": round(percentile(95), 6),
            "p99": round(percentile(99), 6),
        }


class MetricsRegistry:
    """Thread-safe registry of named counters and histograms."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, _Histogram] = defaultdict(_Histogram)

    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            self._histograms[key].observe(value)

    @contextmanager
    def timer(self, name: str, labels: dict[str, str] | None = None) -> Generator[None, None, None]:
        """Context manager that records the elapsed time as a latency sample."""
        start = perf_counter()
        try:
            yield
        finally:
            self.observe(name, perf_counter() - start, labels=labels)

    def counter(self, name: str, labels: dict[str, str] | None = None) -> _Counter:
        return _Counter(self, name, labels)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            histograms = {key: hist.snapshot() for key, hist in self._histograms.items()}
        return {"counters": counters, "histograms": histograms}

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}|{label_str}"


class _Counter:
    def __init__(self, registry: MetricsRegistry, name: str, labels: dict[str, str] | None) -> None:
        self._registry = registry
        self._name = name
        self._labels = labels

    def inc(self, value: float = 1.0) -> None:
        self._registry.increment(self._name, value, labels=self._labels)


metrics = MetricsRegistry()


def track_latency(
    name: str, labels: dict[str, str] | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Decorator that observes a function's latency in a histogram."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            with metrics.timer(name, labels=labels):
                return func(*args, **kwargs)

        return wrapper

    return decorator
