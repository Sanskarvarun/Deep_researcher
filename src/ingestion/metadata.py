"""Metadata extraction and content hashing utilities.

Content hashing is the backbone of idempotent ingestion: a document whose
hash already exists in the store is skipped rather than re-indexed.
"""

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

from src.core import get_logger
from src.core.exceptions import DocumentProcessingError
from src.ingestion.models import Document, SourceType

log = get_logger(__name__)

# Map file extensions to source types
_EXTENSION_MAP: dict[str, SourceType] = {
    ".pdf": SourceType.PDF,
    ".md": SourceType.MARKDOWN,
    ".markdown": SourceType.MARKDOWN,
    ".txt": SourceType.TXT,
    ".text": SourceType.TXT,
    ".html": SourceType.HTML,
    ".htm": SourceType.HTML,
}


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of *normalized* text.

    Normalization (collapse whitespace, lowercase) makes the hash robust
    to trivial formatting differences so re-uploads are detected.

    Args:
        text: Raw content.

    Returns:
        Hex digest string (64 chars).
    """
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def detect_source_type(source: str) -> SourceType:
    """Detect source type from a filename, path, or URL.

    Args:
        source: File path or URL.

    Returns:
        SourceType enum. Returns UNKNOWN for unrecognized sources.
    """
    # YouTube URLs
    parsed = urlparse(source)
    if parsed.netloc and ("youtube.com" in parsed.netloc or "youtu.be" in parsed.netloc):
        return SourceType.YOUTUBE

    # By extension
    ext = Path(urlparse(source).path).suffix.lower()
    return _EXTENSION_MAP.get(ext, SourceType.UNKNOWN)


def extract_title(content: str, source: str, source_type: SourceType) -> str | None:
    """Best-effort title extraction.

    Strategy by source type:
    - Markdown/HTML: first `# H1` / `<h1>` / <title>
    - TXT/PDF/YouTube: first non-empty line (truncated)
    - Fallback: filename stem
    """
    candidates: list[str] = []

    if source_type in (SourceType.MARKDOWN, SourceType.HTML):
        # Markdown H1
        md_match = re.search(r"^\s*#\s+(.+)$", content, re.MULTILINE)
        if md_match:
            candidates.append(md_match.group(1).strip())
        # HTML <title> or <h1>
        title_match = re.search(r"<title[^>]*>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        if title_match:
            candidates.append(re.sub(r"\s+", " ", title_match.group(1)).strip())
        h1_match = re.search(r"<h1[^>]*>(.*?)</h1>", content, re.IGNORECASE | re.DOTALL)
        if h1_match:
            candidates.append(re.sub(r"<[^>]+>", "", h1_match.group(1)).strip())

    # First meaningful non-empty line
    for line in content.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if len(stripped) >= 3 and not stripped.startswith(("---", "===", "```")):
            candidates.append(stripped)
            break

    # Filename stem as last resort
    candidates.append(Path(urlparse(source).path).stem.replace("-", " ").replace("_", " "))

    for candidate in candidates:
        cleaned = candidate.strip()
        if len(cleaned) >= 2:
            return cleaned[:200]
    return None


def extract_section_header(chunk_text: str) -> str | None:
    """Extract a section header from a markdown/text chunk, if present.

    Looks for the nearest preceding heading line in the chunk.
    """
    md_matches = re.findall(r"^(#{1,4})\s+(.+)$", chunk_text, re.MULTILINE)
    if md_matches:
        # Return the deepest (most specific) heading found
        md_matches.sort(key=lambda m: len(m[0]), reverse=True)
        heading: str = md_matches[0][1].strip()
        return heading[:200]

    # ALL-CAPS line (common in reports)
    caps_match = re.search(r"^([A-Z][A-Z\s]{4,50})$", chunk_text, re.MULTILINE)
    if caps_match:
        caps: str = caps_match.group(1).strip()
        return caps.title()[:200]

    return None


def build_document_metadata(document: Document) -> dict:
    """Build the full metadata payload stored alongside each chunk."""
    return {
        "document_id": str(document.id),
        "document_name": document.title or Path(document.source).name,
        "source": document.source,
        "source_type": document.source_type.value,
        "author": document.author,
        "created_at": document.created_at.isoformat(),
        "content_hash": document.content_hash,
        **document.metadata,
    }


def extract_file_metadata(path: Path) -> dict:
    """Extract filesystem-level metadata for a file."""
    try:
        stat = path.stat()
        return {
            "file_size_bytes": stat.st_size,
            "file_modified_at": datetime.fromtimestamp(stat.st_mtime, tz=UTC).isoformat(),
        }
    except OSError as e:
        log.warning("file_metadata_failed", path=str(path), error=str(e))
        return {}


def validate_source(source: str, max_size_mb: int = 50) -> None:
    """Validate a document source before ingestion.

    Raises:
        DocumentProcessingError: If the source is invalid (unknown type,
            too large, missing file).
    """
    source_type = detect_source_type(source)
    if source_type == SourceType.UNKNOWN:
        raise DocumentProcessingError(
            f"Unsupported source: {source}",
            details={"supported": [t.value for t in SourceType if t != SourceType.UNKNOWN]},
        )

    # YouTube URLs need no file checks
    if source_type == SourceType.YOUTUBE:
        return

    path = Path(source)
    if not path.exists():
        raise DocumentProcessingError(f"File not found: {source}")
    if not path.is_file():
        raise DocumentProcessingError(f"Not a file: {source}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        raise DocumentProcessingError(
            f"File too large: {size_mb:.1f}MB (max {max_size_mb}MB)",
            details={"size_mb": round(size_mb, 2), "max_mb": max_size_mb},
        )
