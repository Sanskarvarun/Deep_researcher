"""Tests for secret handling, config validation, and path safety."""

import pytest

from src.core.config import Settings
from src.core.exceptions import ConfigurationError, ValidationError
from src.core.security import (
    REDACTED,
    constant_time_equals,
    mask_secret,
    redact,
    redact_mapping,
    resolve_safe_path,
    secret_warnings,
    validate_security_config,
)

SECURE_KEY = "prod-key-" + "b" * 32  # placeholder, not a real credential


def _settings(**overrides) -> Settings:
    """Build a Settings instance with alias-style overrides (ignores .env noise)."""
    return Settings(**overrides)


class TestRedaction:
    def test_redact(self):
        assert redact("super-secret") == REDACTED
        assert redact("") == ""
        assert redact(None) == ""

    def test_mask_secret(self):
        masked = mask_secret("abcdefghijklmnop")
        assert masked.startswith("abcd")
        assert masked.endswith("mnop")
        assert "efghij" not in masked
        assert mask_secret("short") == REDACTED
        assert mask_secret("") == ""

    def test_redact_mapping_recurses(self):
        payload = {
            "api_key": "secret-value",
            "nested": {"authorization": "Bearer abc", "keep": 1},
            "list": [{"token": "xyz"}, {"keep": 2}],
            "plain": "visible",
        }
        cleaned = redact_mapping(payload)
        assert cleaned["api_key"] == REDACTED
        assert cleaned["nested"]["authorization"] == REDACTED
        assert cleaned["nested"]["keep"] == 1
        assert cleaned["list"][0]["token"] == REDACTED
        assert cleaned["list"][1]["keep"] == 2
        assert cleaned["plain"] == "visible"

    def test_constant_time_equals(self):
        assert constant_time_equals("abc", "abc")
        assert not constant_time_equals("abc", "abd")
        assert not constant_time_equals("abc", "abcd")


class TestSecurityConfig:
    def test_weak_placeholder_key_warns_in_development(self):
        config = _settings(APP_ENV="development", API_KEY="changeme", REQUIRE_AUTH="true")
        warnings = validate_security_config(config)
        assert any("placeholder" in warning for warning in warnings)

    def test_production_weak_key_fails(self):
        config = _settings(
            APP_ENV="production",
            API_KEY="changeme",
            DEBUG="false",
            CORS_ORIGINS=["https://app.example.com"],
            REQUIRE_AUTH="true",
        )
        with pytest.raises(ConfigurationError):
            validate_security_config(config)

    def test_production_requires_api_key(self):
        config = _settings(
            APP_ENV="production",
            API_KEY="",
            DEBUG="false",
            CORS_ORIGINS=["https://app.example.com"],
            REQUIRE_AUTH="true",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_security_config(config)
        assert "API_KEY" in exc_info.value.message

    def test_production_rejects_debug_and_wildcard_cors(self):
        config = _settings(
            APP_ENV="production",
            API_KEY=SECURE_KEY,
            DEBUG="true",
            CORS_ORIGINS=["*"],
            REQUIRE_AUTH="true",
        )
        with pytest.raises(ConfigurationError) as exc_info:
            validate_security_config(config)
        errors = exc_info.value.details["errors"]
        assert any("DEBUG" in error for error in errors)
        assert any("CORS" in error for error in errors)

    def test_safe_production_config_passes(self):
        config = _settings(
            APP_ENV="production",
            API_KEY=SECURE_KEY,
            DEBUG="false",
            CORS_ORIGINS=["https://app.example.com"],
            REQUIRE_AUTH="true",
        )
        warnings = validate_security_config(config)
        assert isinstance(warnings, list)

    def test_secret_warnings_flags_short_key(self):
        config = _settings(APP_ENV="development", API_KEY="short", REQUIRE_AUTH="true")
        warnings = secret_warnings(config)
        assert any("shorter" in warning for warning in warnings)


class TestResolveSafePath:
    def test_accepts_existing_file(self, tmp_path):
        doc = tmp_path / "notes.txt"
        doc.write_text("hello", encoding="utf-8")
        resolved = resolve_safe_path(str(doc), allowed_extensions=[".txt"], base_dir=tmp_path)
        assert resolved == doc.resolve()

    def test_rejects_traversal(self, tmp_path):
        with pytest.raises(ValidationError, match="traversal"):
            resolve_safe_path(str(tmp_path / ".." / "secret.txt"))

    def test_rejects_null_byte(self):
        with pytest.raises(ValidationError):
            resolve_safe_path("notes\x00.txt")

    def test_rejects_unsupported_extension(self, tmp_path):
        script = tmp_path / "payload.exe"
        script.write_text("MZ", encoding="utf-8")
        with pytest.raises(ValidationError, match="extension"):
            resolve_safe_path(str(script), allowed_extensions=[".txt", ".pdf"])

    def test_rejects_path_outside_base_dir(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("nope", encoding="utf-8")
        with pytest.raises(ValidationError, match="outside"):
            resolve_safe_path(str(outside), base_dir=root)

    def test_rejects_missing_file(self, tmp_path):
        with pytest.raises(ValidationError, match="does not exist"):
            resolve_safe_path(str(tmp_path / "missing.txt"))

    def test_rejects_oversized_file(self, tmp_path):
        doc = tmp_path / "big.txt"
        doc.write_text("x" * 2048, encoding="utf-8")
        with pytest.raises(ValidationError, match="maximum size"):
            resolve_safe_path(str(doc), max_size_mb=0)
