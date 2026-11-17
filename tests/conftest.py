"""Shared pytest fixtures."""

import os

# Keep the test suite offline/deterministic: without this, every test that
# touches the cache or rate limiter would try to reach Redis and wait for the
# connect timeout.
os.environ.setdefault("ENABLE_CACHE", "false")

import pytest

from src.api.rate_limit import close_rate_limiter
from src.core import breakers, close_cache


@pytest.fixture(autouse=True)
def _isolate_shared_state():
    """Give every test fresh process-wide state (cache, breakers, limiter)."""
    close_cache()
    breakers.clear()
    close_rate_limiter()
    yield
    close_cache()
    breakers.clear()
    close_rate_limiter()
