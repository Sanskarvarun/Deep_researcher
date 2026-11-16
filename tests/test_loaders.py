"""Tests for document loaders using real temp files."""

from pathlib import Path

import pytest

from src.core.exceptions import DocumentProcessingError, IngestionError
from src.ingestion.loaders import (
    _extract_youtube_id,
    load_batch,
    load_document,
    load_html,
    load_markdown,
    load_pdf,
    load_txt,
)
from src.ingestion.models import SourceType


@pytest.fixture
def sample_txt(tmp_path: Path) -> Path:
    f = tmp_path / "sample.txt"
    f.write_text(
        "Introduction to Machine Learning\n\n"
        "Machine learning is a subset of artificial intelligence. "
        "It focuses on building systems that learn from data. "
        "Supervised learning uses labeled examples. Unsupervised learning "
        "finds patterns without labels.",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_md(tmp_path: Path) -> Path:
    f = tmp_path / "sample.md"
    f.write_text(
        "# My Research Paper\n\n"
        "## Abstract\n\nThis paper presents findings.\n\n"
        "## Introduction\n\nBackground material here.",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_html(tmp_path: Path) -> Path:
    f = tmp_path / "page.html"
    f.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<script>console.log("should be stripped");</script>
<nav>Navigation junk</nav>
<main>
  <h1>Main Heading</h1>
  <p>First paragraph with content.</p>
  <p>Second paragraph with more content.</p>
</main>
<footer>Footer junk</footer>
</body>
</html>""",
        encoding="utf-8",
    )
    return f


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a simple one-page PDF with pypdf."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    f = tmp_path / "sample.pdf"
    with Path(f).open("wb") as fh:
        writer.write(fh)
    return f


class TestLoadTxt:
    def test_basic(self, sample_txt):
        doc = load_txt(str(sample_txt))
        assert doc.source_type == SourceType.TXT
        assert "Machine learning" in doc.content
        assert doc.title  # extracted
        assert doc.metadata["file_size_bytes"] > 0

    def test_missing_file(self):
        with pytest.raises(IngestionError):
            load_txt("/nonexistent/file.txt")


class TestLoadMarkdown:
    def test_basic(self, sample_md):
        doc = load_markdown(str(sample_md))
        assert doc.source_type == SourceType.MARKDOWN
        assert "My Research Paper" in doc.content
        assert doc.title == "My Research Paper"  # from H1

    def test_preserves_headers(self, sample_md):
        doc = load_markdown(str(sample_md))
        assert "## Abstract" in doc.content
        assert "## Introduction" in doc.content


class TestLoadHtml:
    def test_strips_scripts_and_nav(self, sample_html):
        doc = load_html(str(sample_html))
        assert "console.log" not in doc.content
        assert "Navigation junk" not in doc.content
        assert "Footer junk" not in doc.content

    def test_keeps_main_content(self, sample_html):
        doc = load_html(str(sample_html))
        assert "Main Heading" in doc.content
        assert "First paragraph" in doc.content

    def test_title_from_title_tag(self, sample_html):
        doc = load_html(str(sample_html))
        assert doc.title == "Test Page"


class TestLoadPdf:
    def test_blank_pdf_empty_content(self, sample_pdf):
        """Blank PDFs yield empty text; the empty check lives in load_document."""
        doc = load_pdf(str(sample_pdf))
        assert doc.source_type == SourceType.PDF
        assert not doc.content.strip()

    def test_blank_pdf_raises_via_dispatch(self, sample_pdf):
        """load_document rejects documents with no extractable content."""
        with pytest.raises(IngestionError, match="empty"):
            load_document(str(sample_pdf))

    def test_pdf_with_text(self, tmp_path: Path):
        """Write a real text PDF if reportlib available; otherwise skip."""
        pytest.importorskip("reportlab")
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        f = tmp_path / "text.pdf"
        c = canvas.Canvas(str(f), pagesize=letter)
        c.drawString(72, 720, "Hello from PDF. This is test content.")
        c.save()

        doc = load_pdf(str(f))
        assert "Hello from PDF" in doc.content
        assert doc.page_count == 1


class TestLoadDocumentDispatch:
    def test_dispatches_by_extension(self, sample_txt):
        doc = load_document(str(sample_txt))
        assert doc.source_type == SourceType.TXT

    def test_unknown_extension_raises(self, tmp_path: Path):
        f = tmp_path / "data.xyz"
        f.write_text("content")
        with pytest.raises(DocumentProcessingError):
            load_document(str(f))


class TestLoadBatch:
    def test_continues_on_failure(self, sample_txt, sample_md, tmp_path: Path):
        bad = tmp_path / "bad.txt"  # doesn't exist
        docs = load_batch([str(sample_txt), str(bad), str(sample_md)])
        # 2 good loaded, 1 failure logged and skipped
        assert len(docs) == 2

    def test_all_valid(self, sample_txt, sample_md):
        docs = load_batch([str(sample_txt), str(sample_md)])
        assert len(docs) == 2


class TestYouTubeIdExtraction:
    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ],
    )
    def test_extracts_id(self, url, expected):
        assert _extract_youtube_id(url) == expected

    def test_returns_none_for_garbage(self):
        assert _extract_youtube_id("not a url") is None
