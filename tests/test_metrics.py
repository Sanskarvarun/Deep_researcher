"""Tests for the in-memory metrics registry (Phase 2.1 Observability)."""

import pytest

from src.core.metrics import MetricsRegistry, metrics, track_latency


class TestMetricsRegistry:
    def test_counter_increment(self):
        reg = MetricsRegistry()
        reg.increment("requests", labels={"endpoint": "/query"})
        reg.increment("requests", labels={"endpoint": "/query"})
        reg.increment("requests", labels={"endpoint": "/ingest"})
        snap = reg.snapshot()
        assert int(snap["counters"]["requests|endpoint=/query"]) == 2
        assert int(snap["counters"]["requests|endpoint=/ingest"]) == 1

    def test_counter_default_value(self):
        reg = MetricsRegistry()
        reg.increment("errors")
        assert int(reg.snapshot()["counters"]["errors"]) == 1

    def test_histogram_percentiles(self):
        reg = MetricsRegistry()
        for i in range(1, 101):  # 1..100 ms
            reg.observe("latency", float(i))
        snap = reg.snapshot()["histograms"]["latency"]
        assert snap["count"] == 100
        # p50 of 1..100 == 50.5, p95 == 95.05, p99 == 99.01 (linear interpolation)
        assert abs(snap["p50"] - 50.5) < 1e-6
        assert abs(snap["p95"] - 95.05) < 1e-6
        assert abs(snap["p99"] - 99.01) < 1e-6

    def test_histogram_empty(self):
        reg = MetricsRegistry()
        snap = reg.snapshot()["histograms"]
        assert snap == {}

    def test_timer_records_latency(self):
        reg = MetricsRegistry()
        with reg.timer("op.latency"):
            pass
        snap = reg.snapshot()["histograms"]["op.latency"]
        assert snap["count"] == 1
        assert snap["sum"] >= 0

    def test_labels_isolate_histograms(self):
        reg = MetricsRegistry()
        reg.observe("latency", 10.0, labels={"route": "a"})
        reg.observe("latency", 100.0, labels={"route": "b"})
        h = reg.snapshot()["histograms"]
        assert h["latency|route=a"]["p50"] == pytest.approx(10.0)
        assert h["latency|route=b"]["p50"] == pytest.approx(100.0)


class TestGlobalMetrics:
    def test_module_singleton(self):
        # `metrics` is a shared singleton; increments are visible globally.
        before = metrics.snapshot()["counters"]
        metrics.increment("smoke.requests")
        after = metrics.snapshot()["counters"]
        assert set(after) - set(before) == {"smoke.requests"}


class TestTrackLatencyDecorator:
    def test_decorator_records_latency(self):
        from src.core.metrics import metrics as m

        @track_latency("decorated.op", labels={"lang": "py"})
        def run2():
            return 42

        assert run2() == 42
        snap = m.snapshot()["histograms"]["decorated.op|lang=py"]
        assert snap["count"] == 1
        assert snap["p50"] >= 0
