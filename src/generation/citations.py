"""Citation handling for RAG responses.

Assigns stable citation keys to retrieved chunks, formats inline markers like
``[doc_id]`` in the answer text, and builds a clickable reference list.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.retrieval.models import SearchResult


class Citation(BaseModel):
    """A single citation pointing to a retrieved chunk."""

    key: str  # short stable key, e.g. "doc-a" or "c1"
    chunk_id: str
    document_id: str | None = None
    content: str
    source: str | None = None
    section_header: str | None = None
    page_number: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitationTracker:
    """Maps retrieved chunks to citations and formats inline references."""

    def __init__(self) -> None:
        self._citations: dict[str, Citation] = {}
        self._key_for_chunk: dict[str, str] = {}

    def add(self, result: SearchResult) -> str:
        """Register a search result and return its citation key."""
        chunk_id = result.id
        if chunk_id in self._key_for_chunk:
            return self._key_for_chunk[chunk_id]

        key = str(result.document_id or chunk_id)
        base_key = key
        counter = 1
        while key in self._citations:
            key = f"{base_key}-{counter}"
            counter += 1

        citation = Citation(
            key=key,
            chunk_id=chunk_id,
            document_id=str(result.document_id) if result.document_id is not None else None,
            content=result.content,
            source=result.source,
            section_header=result.section_header,
            page_number=result.page_number,
            score=result.score,
            metadata=dict(result.payload),
        )
        self._citations[key] = citation
        self._key_for_chunk[chunk_id] = key
        return key

    def cite(self, chunk_id: str) -> str:
        """Return the citation key for a chunk, or the raw id if untracked."""
        return self._key_for_chunk.get(chunk_id, chunk_id)

    @staticmethod
    def inline_marker(key: str) -> str:
        """Return the inline citation marker for a key: ``[key]``."""
        return f"[{key}]"

    def reference_list(self) -> list[Citation]:
        """Return all registered citations in insertion order."""
        return list(self._citations.values())

    def format_references(self) -> str:
        """Format a human-readable markdown reference list."""
        if not self._citations:
            return ""
        lines = ["## Sources", ""]
        for citation in self.reference_list():
            parts = [f"- **{citation.key}**: {citation.content[:200]}"]
            if citation.source:
                parts.append(f"  (source: {citation.source})")
            if citation.section_header:
                parts.append(f"  (section: {citation.section_header})")
            if citation.page_number is not None:
                parts.append(f"  (page: {citation.page_number})")
            lines.append(" ".join(parts))
        return "\n".join(lines)
