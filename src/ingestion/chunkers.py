"""Text chunking strategies.

Chunking is one of the highest-leverage decisions in RAG quality:
- **Fixed-size + overlap**: baseline; predictable token usage
- **Recursive character**: respects structure (paragraphs → sentences)
- **Markdown-aware**: splits on headers so chunks align with sections
- **Semantic**: embeds sentence windows and splits where topic coherence
  drops (requires an embedder — the only strategy that depends on a model)

All strategies produce chunks as `(content, context)` pairs where `content`
is embedded/retrieved and `context` (if set) is extra text for the LLM.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from src.core import get_logger
from src.core.exceptions import ChunkingError
from src.ingestion.metadata import extract_section_header
from src.ingestion.models import Chunk, Document

log = get_logger(__name__)

# Approximate chars-per-token ratio for budgeting (conservative for English)
CHARS_PER_TOKEN = 4

# Chunk pairs returned by strategies: (embedded content, LLM context or None)
ChunkEntry = tuple[str, "str | None"]
ChunkFunction = Callable[..., list[ChunkEntry]]


class ChunkingStrategy(StrEnum):
    """Available chunking strategies."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    MARKDOWN = "markdown"
    SEMANTIC = "semantic"


@dataclass(frozen=True)
class ChunkConfig:
    """Configuration for all chunking strategies."""

    chunk_size: int = 512  # target size in characters
    chunk_overlap: int = 50  # overlap between consecutive chunks
    min_chunk_size: int = 100  # chunks below this get merged into neighbors
    separators: tuple[str, ...] = (
        "\n\n",  # paragraph
        "\n",  # line
        ". ",  # sentence
        " ",
        "",
    )


class EmbedderProtocol(Protocol):
    """Structural type for embedders used by semantic chunking."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...


def chunk_document(
    document: Document,
    config: ChunkConfig | None = None,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    embedder: EmbedderProtocol | None = None,
) -> list[Chunk]:
    """Chunk a document using the given strategy.

    Args:
        document: Parsed document.
        config: Chunking configuration (defaults applied if None).
        strategy: Which strategy to use.
        embedder: Required only for SEMANTIC strategy.

    Returns:
        List of Chunk objects with chunk_index and chunk_count set.

    Raises:
        ChunkingError: On failure or invalid config.
    """
    cfg = config or ChunkConfig()

    if cfg.chunk_overlap >= cfg.chunk_size:
        raise ChunkingError(
            f"chunk_overlap ({cfg.chunk_overlap}) must be < chunk_size ({cfg.chunk_size})"
        )

    strategies: dict[ChunkingStrategy, ChunkFunction] = {
        ChunkingStrategy.FIXED: chunk_fixed,
        ChunkingStrategy.RECURSIVE: chunk_recursive,
        ChunkingStrategy.MARKDOWN: chunk_markdown,
        ChunkingStrategy.SEMANTIC: chunk_semantic,
    }

    chunk_fn = strategies[strategy]
    try:
        entries = chunk_fn(document.content, cfg, embedder=embedder)
    except ChunkingError:
        raise
    except Exception as e:
        raise ChunkingError(f"Chunking failed: {e}", details={"strategy": strategy.value}) from e

    if not entries:
        log.warning("no_chunks_produced", source=document.source, strategy=strategy.value)
        return []

    total = len(entries)
    chunks: list[Chunk] = []
    for idx, (content, context) in enumerate(entries):
        chunks.append(
            Chunk(
                document_id=document.id,
                content=content,
                context=context,
                chunk_index=idx,
                chunk_count=total,
                section_header=extract_section_header(content),
                metadata={
                    "strategy": strategy.value,
                    "char_count": len(content),
                    "approx_tokens": len(content) // CHARS_PER_TOKEN,
                    **document.metadata,
                },
            )
        )

    log.info("document_chunked", source=document.source, strategy=strategy.value, chunks=total)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-size with overlap
# ---------------------------------------------------------------------------


def chunk_fixed(text: str, cfg: ChunkConfig, **_: object) -> list[ChunkEntry]:
    """Split text into fixed-size chunks with sliding-window overlap.

    Simplest baseline: predictable chunk sizes, uniform token usage.
    Downside: splits mid-sentence and ignores document structure.
    """
    if len(text) <= cfg.chunk_size:
        return [(text, None)] if text.strip() else []

    chunks: list[ChunkEntry] = []
    step = max(cfg.chunk_size - cfg.chunk_overlap, 1)

    for start in range(0, len(text), step):
        chunk_text = text[start : start + cfg.chunk_size]
        if not chunk_text.strip():
            continue
        chunks.append((chunk_text.strip(), None))
        if start + cfg.chunk_size >= len(text):
            break

    return _merge_tiny_chunks(chunks, cfg.min_chunk_size)


# ---------------------------------------------------------------------------
# Strategy 2: Recursive character splitting
# ---------------------------------------------------------------------------


def chunk_recursive(text: str, cfg: ChunkConfig, **_: object) -> list[ChunkEntry]:
    """Recursively split on separators (paragraph → line → sentence → word).

    Prefers natural boundaries: tries the largest semantic unit first,
    falls back to finer separators only when a chunk is too big.
    """
    if len(text) <= cfg.chunk_size:
        return [(text, None)] if text.strip() else []

    parts = _split_recursive(text, cfg.separators, cfg.chunk_size)
    chunks: list[ChunkEntry] = [(part.strip(), None) for part in parts]

    return _merge_tiny_chunks(chunks, cfg.min_chunk_size)


def _split_recursive(text: str, separators: tuple[str, ...], chunk_size: int) -> list[str]:
    """Inner recursive splitter (LangChain-style)."""
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    separator = separators[0]
    remaining = separators[1:]

    # No separators left — hard-split by size
    if not separator:
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    splits = text.split(separator)
    if len(splits) == 1:
        # Separator not present — try the next one
        return _split_recursive(text, remaining, chunk_size)

    chunks: list[str] = []
    current = ""
    for part in splits:
        candidate = f"{current}{separator}{part}" if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # A single part may still exceed chunk_size — recurse deeper
            if len(part) > chunk_size:
                chunks.extend(_split_recursive(part, remaining, chunk_size))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Markdown-aware chunking
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_markdown(text: str, cfg: ChunkConfig, **_: object) -> list[ChunkEntry]:
    """Split markdown on headers first, then recursively within sections.

    Ideal for docs/manuals where sections are the semantic unit.
    Each oversized section is split and tagged with its heading as context.
    """
    if len(text) <= cfg.chunk_size and not _HEADER_RE.search(text):
        return [(text, None)] if text.strip() else []

    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        # Not markdown — fall back to recursive
        return [(part, None) for part in _split_recursive(text, cfg.separators, cfg.chunk_size)]

    chunks: list[ChunkEntry] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        section_text = text[start:end].strip()

        if len(section_text) <= cfg.chunk_size:
            chunks.append((section_text, None))
        else:
            sub_parts = _split_recursive(section_text, cfg.separators, cfg.chunk_size)
            for sub in sub_parts:
                chunks.append((sub.strip(), f"Section: {heading}"))

    return _merge_tiny_chunks(chunks, cfg.min_chunk_size)


# ---------------------------------------------------------------------------
# Strategy 4: Semantic chunking (embedding-based)
# ---------------------------------------------------------------------------


def chunk_semantic(
    text: str, cfg: ChunkConfig, embedder: EmbedderProtocol | None = None, **_: object
) -> list[ChunkEntry]:
    """Split text where semantic similarity between adjacent windows drops.

    Algorithm:
    1. Split into sentences.
    2. Group sentences into windows of ~chunk_size chars.
    3. Embed consecutive windows; merge while cosine similarity stays high.
    4. Split when similarity drops below a percentile-based threshold.

    Requires an embedder and produces the highest-quality chunks,
    at 1-2 extra embedding passes per document.
    """
    if embedder is None:
        raise ChunkingError("Semantic chunking requires an embedder")

    sentences = _split_sentences(text)
    if len(sentences) <= 1:
        return [(text, None)] if text.strip() else []

    # Group sentences into windows
    windows: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > cfg.chunk_size and current:
            windows.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(sentence)
    if current:
        windows.append(" ".join(current))

    if len(windows) <= 1:
        return [(text, None)]

    # Embed windows and compute similarity between adjacent ones
    embeddings = embedder.embed_texts(windows)
    sims = [
        _cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
    ]

    # Threshold: split where similarity is in the bottom quartile
    threshold = _percentile(sims, 25)

    chunks: list[ChunkEntry] = []
    current_chunk: list[str] = [windows[0]]

    for i, sim in enumerate(sims):
        if sim < threshold:
            chunks.append((" ".join(current_chunk), None))
            current_chunk = [windows[i + 1]]
        else:
            current_chunk.append(windows[i + 1])

    if current_chunk:
        chunks.append((" ".join(current_chunk), None))

    return _merge_tiny_chunks(chunks, cfg.min_chunk_size)


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter that protects common abbreviations."""
    protected = re.sub(r"\b(Mr|Mrs|Dr|Prof|Sr|Jr|vs|etc|e\.g|i\.e)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+", protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)  # type: ignore[no-any-return]


def _percentile(values: list[float], pct: int) -> float:
    """Naive percentile (no numpy dependency)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_tiny_chunks(chunks: list[ChunkEntry], min_size: int) -> list[ChunkEntry]:
    """Merge chunks below min_size into the preceding chunk.

    Prevents tiny orphan fragments that hurt retrieval precision.
    """
    if not chunks:
        return []

    merged: list[ChunkEntry] = []
    carry = ""

    for content, context in chunks:
        content = carry + (" " if carry else "") + content
        carry = ""

        if len(content.strip()) < min_size:
            carry = content  # hold for next merge
            continue

        merged.append((content, context))

    # Append leftover to last chunk if possible
    if carry and merged:
        last_content, last_ctx = merged[-1]
        merged[-1] = (last_content + " " + carry, last_ctx)
    elif carry:
        merged.append((carry, None))

    return merged
