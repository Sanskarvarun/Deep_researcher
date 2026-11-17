"""Shared pytest fixtures."""

import os

# Keep the test suite offline/deterministic: without this, every test that
# touches the cache would try to reach Redis and wait for the connect timeout.
os.environ.setdefault("ENABLE_CACHE", "false")

import pytest

from src.core import close_cache


@pytest.fixture(autouse=True)
def _isolate_cache():
    """Give every test a fresh cache.

    The process-wide cache would otherwise leak values between tests (the same
    query against separate temporary collections would hit a stale entry).
    """
    close_cache()
    yield
    close_cache()
