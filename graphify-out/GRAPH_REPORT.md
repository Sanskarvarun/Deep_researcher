# Graph Report - Deepresearcher  (2026-11-01)

## Corpus Check
- Corpus is ~22,373 words - fits in a single context window. You may not need a graph.

## Summary
- 511 nodes · 1112 edges · 40 communities (29 shown, 11 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 134 edges (avg confidence: 0.92)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Chunking Strategies
- Embedding Providers
- Architecture & Product Specs
- Vector Store & Dedup
- Exception Hierarchy
- HuggingFace Embedder
- Document Loaders
- Constructors & Init Hooks
- LLM Provider Fallback
- Ingest CLI & Logging
- Metadata & Content Hashing
- Embedding & Service Errors
- Configuration Settings
- Structured Logging
- Generation & Pipeline Tests
- Batch Loading Tests
- Vector Store Test Doubles
- Qdrant VectorStore Wrapper
- Ingestion Orchestration
- PDF Text Extraction
- Title Extraction
- Source Validation
- Ingestion Pipeline Config
- Markdown & File Metadata
- Source Type Detection
- Section Header Extraction
- Source Types & Fixtures
- Chunking Strategy Enum
- YouTube ID Parsing
- HTML Loading
- Ingest File Tests
- Agent Module Stub
- API Module Stub
- Evaluation Module Stub
- Ingestion Module Stub
- Package Root
- Retrieval Module Stub
- UI Module Stub
- Project Metadata

## God Nodes (most connected - your core abstractions)
1. `ChunkConfig` - 37 edges
2. `Document` - 36 edges
3. `chunk_document()` - 32 edges
4. `SourceType` - 29 edges
5. `IngestionPipeline` - 23 edges
6. `FakeEmbedder` - 22 edges
7. `ChunkingStrategy` - 21 edges
8. `DocumentProcessingError` - 20 edges
9. `load_document()` - 19 edges
10. `VectorStore` - 19 edges

## Surprising Connections (you probably didn't know these)
- `HyDE (Hypothetical Document Embeddings)` --semantically_similar_to--> `Hybrid Search with Reciprocal Rank Fusion`  [INFERRED] [semantically similar]
  prd.md → trd.md
- `Local-First Decision (local bge-reranker + local Phoenix)` --semantically_similar_to--> `Resilience Patterns (circuit breaker, retry+jitter, timeouts, bulkhead)`  [INFERRED] [semantically similar]
  PLAN.md → trd.md
- `Query State Machine (SUBMITTED→COMPLETE with fallbacks)` --semantically_similar_to--> `Resilience Patterns (circuit breaker, retry+jitter, timeouts, bulkhead)`  [INFERRED] [semantically similar]
  appflow.md → trd.md
- `PostgreSQL Entity Relationship Model` --conceptually_related_to--> `Local Dev Stack (Qdrant, Redis, Phoenix)`  [AMBIGUOUS]
  backendschema.md → docker-compose.yml
- `ingest()` --uses--> `ChunkConfig`  [INFERRED]
  scripts/ingest.py → src/ingestion/chunkers.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Hybrid Retrieval & Ranking Stack** — trd_hybrid_search_rrf, backendschema_qdrant_collection, trd_bge_reranker_local, prd_hyde, backendschema_multi_tenancy [INFERRED 0.85]
- **Document Ingestion Pipeline (parse → chunk → embed → dedup → upsert)** — appflow_ingestion_pipeline, backendschema_content_hash_dedup, backendschema_redis_cache_patterns, backendschema_qdrant_collection [EXTRACTED 1.00]
- **Evaluation & Observability Feedback Loop** — prd_ragas_evaluation, backendschema_golden_dataset, trd_phoenix_tracing, backendschema_postgres_erd [INFERRED 0.85]

## Communities (40 total, 11 thin omitted)

### Community 0 - "Chunking Strategies"
Cohesion: 0.07
Nodes (42): ChunkEntry, dataclasses, enum, ChunkingError, Raised when text chunking fails., chunk_document(), chunk_fixed(), chunk_markdown() (+34 more)

### Community 1 - "Embedding Providers"
Cohesion: 0.07
Nodes (23): create_embedder(), embed_chunks(), EmbeddingModel, GeminiEmbedder, LocalEmbedder, Protocol, Embeddings via local sentence-transformers model. No API needed., Embeddings via Gemini API (gemini-embedding-001). Free tier available. Flexible… (+15 more)

### Community 2 - "Architecture & Product Specs"
Cohesion: 0.11
Nodes (35): Pre-Commit Hooks (ruff, mypy, gitleaks), Application Flow & User Journey, Citation Click → Side Panel Flow, Celery Document Ingestion Pipeline, Query State Machine (SUBMITTED→COMPLETE with fallbacks), Backend Schema & Data Design, Content-Hash Idempotent Deduplication, Golden Dataset / QA Pairs / Eval Runs Tables (+27 more)

### Community 3 - "Vector Store & Dedup"
Cohesion: 0.09
Nodes (17): BaseModel, datetime, qdrant_client, qdrant_client_http, compute_content_hash(), Compute SHA-256 hash of *normalized* text. Normalization (collapse whitespace,…, QdrantPoint, SHA-256 hash of normalized content (for deduplication). (+9 more)

### Community 4 - "Exception Hierarchy"
Cohesion: 0.12
Nodes (22): Exception, AgentError, AuthenticationError, AuthorizationError, CacheError, ConfigurationError, DeepResearcherError, EvaluationError (+14 more)

### Community 5 - "HuggingFace Embedder"
Cohesion: 0.10
Nodes (8): HuggingFaceEmbedder, retry, Embed texts in batches of BATCH_SIZE., Embeddings via HuggingFace Inference API (sentence-transformers host). Free…, HF returns [[[tok1], [tok2], ...]] — mean-pooled to sentence vec., Model-loading 503s should raise EmbeddingError (tenacity retries)., Texts exceeding BATCH_SIZE produce multiple API calls., TestHFEmbedder

### Community 6 - "Document Loaders"
Cohesion: 0.16
Nodes (15): bs4, pypdf, DocumentProcessingError, IngestionError, Raised when document processing fails., Raised when document ingestion fails., load_document(), load_txt() (+7 more)

### Community 8 - "LLM Provider Fallback"
Cohesion: 0.19
Nodes (10): BaseChatModel, invoke_with_fallback(), LLMFactory, Any, retry, Return ordered provider list: primary first, then fallbacks., Invoke LLM with automatic provider fallback. Tries primary provider, then each…, Creates chat models for any provider with automatic fallback. (+2 more)

### Community 9 - "Ingest CLI & Logging"
Cohesion: 0.17
Nodes (15): command, rich_console, rich_table, _color(), ingest(), ingest_dir(), _print_result(), CLI tool for document ingestion. Usage: # Ingest a single file uv run python -m… (+7 more)

### Community 10 - "Metadata & Content Hashing"
Cohesion: 0.20
Nodes (12): hashlib, pathlib, pytest, re, build_document_metadata(), Metadata extraction and content hashing utilities. Content hashing is the…, Build the full metadata payload stored alongside each chunk., Document (+4 more)

### Community 11 - "Embedding & Service Errors"
Cohesion: 0.19
Nodes (13): EmbeddingError, ExternalServiceError, GenerationError, Custom exceptions for the application., Raised when an external service call fails., Raised when embedding generation fails., Raised when document retrieval fails., Raised when LLM generation fails. (+5 more)

### Community 12 - "Configuration Settings"
Cohesion: 0.16
Nodes (12): BaseSettings, functools, langchain_core_language_models, pydantic, pydantic_settings, get_settings(), Core configuration using Pydantic Settings., Application settings loaded from environment variables. (+4 more)

### Community 13 - "Structured Logging"
Cohesion: 0.15
Nodes (13): BoundLogger, collections_abc, pythonjsonlogger, _console_renderer(), get_logger(), _json_renderer(), Any, Structured logging configuration with structlog. (+5 more)

### Community 14 - "Generation & Pipeline Tests"
Cohesion: 0.18
Nodes (10): Generation module - LLM calls, prompt templates, streaming., DocumentStatus, StrEnum, Document processing status., pipeline(), fixture, Integration test for the ingestion pipeline (fake embedder + qdrant client)., sample_file() (+2 more)

### Community 15 - "Batch Loading Tests"
Cohesion: 0.24
Nodes (11): load_batch(), Load multiple documents, skipping (and logging) failures. Returns: List of…, fixture, Path, Tests for document loaders using real temp files., Create a simple one-page PDF with pypdf., sample_html(), sample_md() (+3 more)

### Community 16 - "Vector Store Test Doubles"
Cohesion: 0.15
Nodes (4): FakeVectorStore, In-memory stand-in for Qdrant: tracks ensures, upserts, dedup., Unit tests of dedup helpers without Qdrant., TestVectorStoreDedupLogic

### Community 17 - "Qdrant VectorStore Wrapper"
Cohesion: 0.17
Nodes (6): Check Qdrant connectivity., Check if a document (by content hash) is already ingested. Every chunk of a…, Qdrant wrapper: setup, dedup-check, upsert., Total points in the collection., Create the collection if missing (idempotent). Configures: - dense vectors…, VectorStore

### Community 18 - "Ingestion Orchestration"
Cohesion: 0.24
Nodes (6): IngestionResult, Result of ingesting a single document., Ingest multiple documents; continues on failure. Returns: One IngestionResult…, Ingest an already-parsed Document object (no file access)., Check if a document with this content hash is already ingested. Documents are…, Ingest a single document (file path or URL). Args: source: File path or YouTube…

### Community 19 - "PDF Text Extraction"
Cohesion: 0.25
Nodes (6): load_pdf(), Extract text from a PDF using pypdf., Blank PDFs yield empty text; the empty check lives in load_document., load_document rejects documents with no extractable content., Write a real text PDF if reportlib available; otherwise skip., TestLoadPdf

### Community 20 - "Title Extraction"
Cohesion: 0.36
Nodes (3): extract_title(), Best-effort title extraction. Strategy by source type: - Markdown/HTML: first…, TestExtractTitle

### Community 21 - "Source Validation"
Cohesion: 0.39
Nodes (4): Validate a document source before ingestion. Raises: DocumentProcessingError:…, validate_source(), Path, TestValidateSource

### Community 22 - "Ingestion Pipeline Config"
Cohesion: 0.36
Nodes (5): IngestionPipeline, PipelineConfig, Aggregate ingestion configuration., End-to-end document ingestion pipeline., TestPipelineConfig

### Community 23 - "Markdown & File Metadata"
Cohesion: 0.29
Nodes (6): load_markdown(), Load a markdown file as-is (already plain text-ish)., extract_file_metadata(), Path, Extract filesystem-level metadata for a file., TestLoadMarkdown

### Community 24 - "Source Type Detection"
Cohesion: 0.43
Nodes (4): detect_source_type(), Detect source type from a filename, path, or URL. Args: source: File path or…, parametrize, TestSourceTypeDetection

### Community 25 - "Section Header Extraction"
Cohesion: 0.43
Nodes (3): extract_section_header(), Extract a section header from a markdown/text chunk, if present. Looks for the…, TestExtractSectionHeader

### Community 26 - "Source Types & Fixtures"
Cohesion: 0.33
Nodes (7): Supported document source types., SourceType, document(), markdown_document(), fixture, A document long enough to require multiple chunks., Markdown doc with sections, sized to force splitting.

### Community 27 - "Chunking Strategy Enum"
Cohesion: 0.33
Nodes (5): ChunkingStrategy, StrEnum, Available chunking strategies., Ingestion pipeline: orchestrates load → chunk → embed → upsert. This is the…, time

### Community 28 - "YouTube ID Parsing"
Cohesion: 0.40
Nodes (4): _extract_youtube_id(), Extract video ID from various YouTube URL formats., parametrize, TestYouTubeIdExtraction

### Community 29 - "HTML Loading"
Cohesion: 0.47
Nodes (3): load_html(), Load an HTML file, stripping scripts/styles/nav boilerplate., TestLoadHtml

## Ambiguous Edges - Review These
- `Application Flow & User Journey` → `Provider Fallback Chain (Gemini → OpenRouter → Groq → local)`  [AMBIGUOUS]
  appflow.md · relation: references
- `PostgreSQL Entity Relationship Model` → `Local Dev Stack (Qdrant, Redis, Phoenix)`  [AMBIGUOUS]
  docker-compose.yml · relation: conceptually_related_to

## Knowledge Gaps
- **3 isolated node(s):** `deepresearcher`, `Six-Phase 12-Week Roadmap`, `Pre-Commit Hooks (ruff, mypy, gitleaks)`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 202 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Application Flow & User Journey` and `Provider Fallback Chain (Gemini → OpenRouter → Groq → local)`?**
  _Edge tagged AMBIGUOUS (relation: references) - confidence is low._
- **What is the exact relationship between `PostgreSQL Entity Relationship Model` and `Local Dev Stack (Qdrant, Redis, Phoenix)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `Document` connect `Metadata & Content Hashing` to `Chunking Strategies`, `Vector Store & Dedup`, `Document Loaders`, `Generation & Pipeline Tests`, `Batch Loading Tests`, `Ingestion Orchestration`, `PDF Text Extraction`, `Ingestion Pipeline Config`, `Markdown & File Metadata`, `Source Types & Fixtures`, `Chunking Strategy Enum`, `HTML Loading`?**
  _High betweenness centrality (0.080) - this node is a cross-community bridge._
- **Why does `HuggingFaceEmbedder` connect `HuggingFace Embedder` to `Embedding Providers`, `Embedding & Service Errors`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `VectorStore` connect `Qdrant VectorStore Wrapper` to `Embedding Providers`, `Vector Store & Dedup`, `Ingest CLI & Logging`, `Embedding & Service Errors`, `Ingestion Pipeline Config`, `Chunking Strategy Enum`?**
  _High betweenness centrality (0.050) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `ChunkConfig` (e.g. with `ingest()` and `smoke()`) actually correct?**
  _`ChunkConfig` has 12 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `Document` (e.g. with `chunk_document()` and `load_batch()`) actually correct?**
  _`Document` has 13 INFERRED edges - model-reasoned connections that need verification._
