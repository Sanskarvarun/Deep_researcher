"""Tests for metadata utilities: hashing, source detection, validation."""

from pathlib import Path

import pytest

from src.core.exceptions import DocumentProcessingError
from src.ingestion.metadata import (
    build_document_metadata,
    compute_content_hash,
    detect_source_type,
    extract_section_header,
    extract_title,
    validate_source,
)
from src.ingestion.models import Document, SourceType


class TestContentHash:
    def test_deterministic(self):
        assert compute_content_hash("hello world") == compute_content_hash("hello world")

    def test_normalizes_whitespace(self):
        """Same text with different whitespace → same hash."""
        assert compute_content_hash("hello  world") == compute_content_hash("hello world")
        assert compute_content_hash("hello\n\nworld") == compute_content_hash("hello world")

    def test_case_insensitive(self):
        assert compute_content_hash("Hello World") == compute_content_hash("hello world")

    def test_different_content_different_hash(self):
        assert compute_content_hash("abc") != compute_content_hash("xyz")

    def test_is_sha256_length(self):
        assert len(compute_content_hash("test")) == 64


class TestSourceTypeDetection:
    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("file.pdf", SourceType.PDF),
            ("file.PDF", SourceType.PDF),
            ("docs/file.md", SourceType.MARKDOWN),
            ("readme.markdown", SourceType.MARKDOWN),
            ("notes.txt", SourceType.TXT),
            ("page.html", SourceType.HTML),
            ("page.htm", SourceType.HTML),
        ],
    )
    def test_extensions(self, source, expected):
        assert detect_source_type(source) == expected

    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
        ],
    )
    def test_youtube(self, url):
        assert detect_source_type(url) == SourceType.YOUTUBE

    def test_unknown(self):
        assert detect_source_type("file.xyz") == SourceType.UNKNOWN
        assert detect_source_type("noextension") == SourceType.UNKNOWN


class TestExtractTitle:
    def test_markdown_h1(self):
        content = "# My Great Paper\n\nSome body text."
        assert extract_title(content, "x.md", SourceType.MARKDOWN) == "My Great Paper"

    def test_html_title_tag(self):
        content = "<html><head><title>Page Title</title></head><body>text</body></html>"
        assert extract_title(content, "x.html", SourceType.HTML) == "Page Title"

    def test_html_h1(self):
        content = "<html><body><h1>Big Heading</h1><p>text</p></body></html>"
        assert extract_title(content, "x.html", SourceType.HTML) == "Big Heading"

    def test_first_meaningful_line(self):
        content = "\n\n\nReal Title Here\n\nmore content"
        assert extract_title(content, "x.txt", SourceType.TXT) == "Real Title Here"

    def test_falls_back_to_filename(self):
        content = "\n\n---\n\n"  # no good lines
        title = extract_title(content, "my-report.txt", SourceType.TXT)
        assert "my report" in title.lower()

    def test_truncates_long_titles(self):
        content = "x" * 500
        title = extract_title(content, "x.txt", SourceType.TXT)
        assert len(title) <= 200


class TestExtractSectionHeader:
    def test_markdown_heading(self):
        chunk = "## Installation\n\nRun pip install..."
        assert extract_section_header(chunk) == "Installation"

    def test_deepest_heading_wins(self):
        chunk = "# Title\n## Section\n### Implementation\n\ntext"
        assert extract_section_header(chunk) == "Implementation"

    def test_caps_heading(self):
        chunk = "INTRODUCTION\n\nSome text"
        assert extract_section_header(chunk) == "Introduction"

    def test_none_when_absent(self):
        chunk = "Just plain text with no heading."
        assert extract_section_header(chunk) is None


class TestValidateSource:
    def test_missing_file_raises(self):
        with pytest.raises(DocumentProcessingError, match=r"Unsupported|not found"):
            validate_source("/nonexistent/file.xyz")

    def test_unknown_type_raises(self, tmp_path: Path):
        f = tmp_path / "file.xyz"
        f.write_text("data")
        with pytest.raises(DocumentProcessingError, match="Unsupported"):
            validate_source(str(f))

    def test_valid_file(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("hello world content")
        validate_source(str(f))  # should not raise

    def test_too_large_raises(self, tmp_path: Path):
        f = tmp_path / "big.txt"
        f.write_text("x" * (2 * 1024 * 1024))  # 2MB
        with pytest.raises(DocumentProcessingError, match="too large"):
            validate_source(str(f), max_size_mb=1)

    def test_youtube_url_skips_file_check(self):
        validate_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ")


class TestDocumentMetadata:
    def test_build_metadata(self):
        doc = Document(
            content="# Title\ncontent",
            source="test.md",
            source_type=SourceType.MARKDOWN,
            title="Title",
        )
        meta = build_document_metadata(doc)
        assert meta["document_id"] == str(doc.id)
        assert meta["source"] == "test.md"
        assert meta["source_type"] == "markdown"
        assert meta["content_hash"] == doc.content_hash
