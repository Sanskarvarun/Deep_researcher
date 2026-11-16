"""Document loaders: extract raw text from various source formats.

Each loader returns a `Document` with plain text content. Errors are
wrapped in `DocumentProcessingError` with the source context attached
so ingestion jobs can fail gracefully per-document.
"""

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.core import get_logger
from src.core.exceptions import DocumentProcessingError, IngestionError
from src.ingestion.metadata import (
    detect_source_type,
    extract_file_metadata,
    extract_title,
)
from src.ingestion.models import Document, SourceType

log = get_logger(__name__)

# Truncation guard: skip pathological files that would blow up chunking
MAX_CONTENT_CHARS = 10_000_000  # 10M chars ≈ 2.5M tokens


def load_document(source: str) -> Document:
    """Load a document from a file path or YouTube URL.

    Dispatches by detected source type.

    Args:
        source: File path or YouTube URL.

    Returns:
        Document with extracted text content.

    Raises:
        DocumentProcessingError: If loading fails.
    """
    source_type = detect_source_type(source)
    loaders = {
        SourceType.PDF: load_pdf,
        SourceType.MARKDOWN: load_markdown,
        SourceType.TXT: load_txt,
        SourceType.HTML: load_html,
        SourceType.YOUTUBE: load_youtube,
    }

    loader = loaders.get(source_type)
    if loader is None:
        raise DocumentProcessingError(f"No loader for source type: {source_type}")

    try:
        document = loader(source)
    except DocumentProcessingError:
        raise
    except Exception as e:
        raise IngestionError(f"Failed to load {source}: {e}", details={"source": source}) from e

    if not document.content or not document.content.strip():
        raise IngestionError("Document is empty after extraction", details={"source": source})

    if len(document.content) > MAX_CONTENT_CHARS:
        raise IngestionError(
            f"Document too large: {len(document.content)} chars",
            details={"source": source, "max_chars": MAX_CONTENT_CHARS},
        )

    log.info(
        "document_loaded",
        source=source,
        source_type=source_type.value,
        chars=len(document.content),
    )
    return document


def load_pdf(source: str) -> Document:
    """Extract text from a PDF using pypdf."""
    path = Path(source)
    try:
        reader = PdfReader(source)
    except OSError as e:
        raise IngestionError(f"Cannot read PDF: {e}", details={"source": source}) from e
    except Exception as e:
        raise IngestionError(f"Invalid PDF: {e}", details={"source": source}) from e

    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    content = "\n\n".join(pages)
    metadata = extract_file_metadata(path)
    metadata["page_count"] = len(reader.pages)

    # Try PDF document info for author/title
    pdf_meta = reader.metadata
    author: str | None = getattr(pdf_meta, "author", None) if pdf_meta else None

    return Document(
        content=content,
        source=source,
        source_type=SourceType.PDF,
        title=extract_title(content, source, SourceType.PDF),
        author=author,
        page_count=len(reader.pages),
        metadata=metadata,
    )


def load_txt(source: str) -> Document:
    """Load a plain text file."""
    path = Path(source)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise IngestionError(f"Cannot read file: {e}", details={"source": source}) from e
    metadata = extract_file_metadata(path)

    return Document(
        content=content,
        source=source,
        source_type=SourceType.TXT,
        title=extract_title(content, source, SourceType.TXT),
        metadata=metadata,
    )


def load_markdown(source: str) -> Document:
    """Load a markdown file as-is (already plain text-ish)."""
    path = Path(source)
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise IngestionError(f"Cannot read file: {e}", details={"source": source}) from e
    metadata = extract_file_metadata(path)

    return Document(
        content=content,
        source=source,
        source_type=SourceType.MARKDOWN,
        title=extract_title(content, source, SourceType.MARKDOWN),
        metadata=metadata,
    )


def load_html(source: str) -> Document:
    """Load an HTML file, stripping scripts/styles/nav boilerplate."""
    path = Path(source)
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise IngestionError(f"Cannot read file: {e}", details={"source": source}) from e

    soup = BeautifulSoup(raw, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "noscript", "iframe", "svg", "nav", "footer"]):
        tag.decompose()

    # Prefer <main> or <article> if present
    main = soup.find("main") or soup.find("article") or soup.body or soup

    text = main.get_text(separator="\n")
    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    metadata = extract_file_metadata(path)

    return Document(
        content=text,
        source=source,
        source_type=SourceType.HTML,
        title=title or extract_title(text, source, SourceType.HTML),
        metadata=metadata,
    )


def load_youtube(source: str) -> Document:
    """Fetch a YouTube video transcript via youtube-transcript-api."""
    from youtube_transcript_api import YouTubeTranscriptApi

    video_id = _extract_youtube_id(source)
    if not video_id:
        raise IngestionError(f"Could not parse YouTube video ID: {source}")

    try:
        # API v1.x: list(video_id) → TranscriptList, then pick a transcript
        transcript_list = YouTubeTranscriptApi().list(video_id)
        # Prefer manually created English, then generated English, then anything
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except Exception:
                # Iterate all available transcripts, take first translatable one
                for t in transcript_list:
                    transcript = t
                    break

        segments = transcript.fetch()
        lines = [seg.text for seg in segments]
    except Exception as e:
        raise IngestionError(
            f"Could not fetch transcript for {video_id}: {e}",
            details={"source": source, "video_id": video_id},
        ) from e

    content = "\n".join(lines)
    title = f"YouTube Transcript {video_id}"

    return Document(
        content=content,
        source=source,
        source_type=SourceType.YOUTUBE,
        title=title,
        metadata={"video_id": video_id, "segment_count": len(segments)},
    )


def _extract_youtube_id(url: str) -> str | None:
    """Extract video ID from various YouTube URL formats."""
    patterns = [
        r"(?:v=|\/)([0-9A-Za-z_-]{11})",  # watch?v=ID, youtu.be/ID, /embed/ID
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def load_batch(sources: list[str]) -> list[Document]:
    """Load multiple documents, skipping (and logging) failures.

    Returns:
        List of successfully loaded documents. Failures are logged, not raised,
        so one bad file doesn't abort a batch.
    """
    documents: list[Document] = []
    for source in sources:
        try:
            documents.append(load_document(source))
        except (DocumentProcessingError, IngestionError) as e:
            log.error("batch_load_failed", source=source, error=str(e))
    return documents
