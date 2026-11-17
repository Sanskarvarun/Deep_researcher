# DeepResearcher - Production-Grade RAG Agent Plan

> **Companion documents**: [PRD](prd.md) • [TRD](trd.md) • [App Flow](appflow.md) • [UI/UX](uiux.md) • [Backend Schema](backendschema.md)

## Project Overview

Build a production-grade RAG (Retrieval-Augmented Generation) agent with agentic capabilities using free APIs. Focus on learning production patterns: evaluation, observability, security, and agentic workflows.

**Learning Objective**: Understand what makes RAG "production-grade" — not just a demo that works once, but a system that is observable, evaluable, resilient, and safe.

---

## Tech Stack (All Free Tiers)

| Layer | Technology | Free Tier Limits | Fallback |
|-------|------------|------------------|----------|
| **LLM (primary)** | Google Gemini (`gemini-2.5-flash-lite`, 1M ctx, function calling) | Free in/out tokens, ~10-15 RPM, 250-1000 RPD | OpenRouter |
| **LLM (fallback)** | OpenRouter (`:free` models, 300+ models) | 20 RPM / 50 RPD (1000 RPD w/ $10 credit) | Groq |
| **LLM (optional)** | Groq (Llama 3.1 70B) | 14,400 req/day | Local Llama.cpp |
| **Embeddings** | HF Inference API (bge-large-en-v1.5) | 30,000 req/month | Local sentence-transformers |
| **Embeddings (alt)** | Gemini `gemini-embedding-001` | Free tier | — |
| **Vector DB** | Qdrant Cloud (1GB free) / local | 1GB / Unlimited | Chroma local |
| **Reranking** | Local `bge-reranker-large` | Unlimited | Cohere API (1K/mo) |
| **Document Parsing** | pypdf, markdown-it, bs4, YT API | Unlimited | Marker (local) |
| **Orchestration** | LangGraph | Unlimited | — |
| **Observability** | LangSmith (OTLP, 5K/mo free) | Unlimited (Dev plan) | Phoenix local / console |
| **Evaluation** | RAGAS + custom | Unlimited | — |
| **API Framework** | FastAPI | Unlimited | — |
| **Frontend** | Streamlit | Unlimited | — |
| **Caching** | Redis (local / Upstash free) | 10MB / 10K req/day | In-memory LRU |
| **Queue** | Celery + Redis | Unlimited | Dramatiq |
| **Relational DB** | PostgreSQL + SQLAlchemy + Alembic | Unlimited | SQLite (dev) |
| **Package Manager** | uv | Unlimited | pip |

> **Note on Google AI Pro**: The Google AI Pro subscription ($19.99/mo) only benefits the AI Studio web UI — it does **not** provide external API keys. However, a **separate free Gemini API key** can be created at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) with a free tier.

---

## Phase 1: Foundation (Week 1-2)

**Goal**: Core RAG pipeline working end-to-end.

### 1.1 Project Setup ✅
- [x] Initialize Python project with `uv`
- [x] Project structure (`src/{core,ingestion,retrieval,generation,agent,evaluation,api,ui}`)
- [x] `pyproject.toml` with ruff, mypy, pytest config
- [x] `src/core/`: config (Pydantic Settings), logging (structlog), exceptions
- [x] `.env.example`, `docker-compose.yml` (Qdrant, Redis)
- [x] Pre-commit hooks (ruff, mypy, gitleaks)
- [x] Documentation: PRD, TRD, AppFlow, UI/UX, Backend Schema, Plan

### 1.2 Document Ingestion Pipeline ✅
- [x] `src/ingestion/models.py` — Document, Chunk, EmbeddedChunk, IngestionResult
- [x] `src/ingestion/metadata.py` — Source detection, content hash, title/section extraction, validation
- [x] `src/ingestion/loaders.py` — PDF, MD, TXT, HTML, YouTube loaders (batch-safe)
- [x] `src/ingestion/chunkers.py` — Fixed-size, recursive, markdown-aware, semantic (4 strategies)
- [x] `src/ingestion/embedder.py` — HF/local/Gemini embedders, batching + retry
- [x] `src/ingestion/vectorstore.py` — Qdrant setup (dense + BM25 + int8 quant), doc- & chunk-level dedup
- [x] `src/ingestion/pipeline.py` — Full orchestration with status tracking
- [x] `scripts/ingest.py` — CLI (ingest, ingest-dir, smoke test)
- [x] Tests: 84 passing (chunkers, loaders, metadata, embedder, pipeline)

### 1.3 Basic Retrieval ✅
- [x] Dense vector search (Qdrant, cosine)
- [x] Sparse/keyword search (BM25 via Qdrant sparse vectors)
- [x] Hybrid search with RRF (Reciprocal Rank Fusion)
- [x] Metadata filtering (date, source, tags, tenant_id)
- [x] Top-k with score thresholds

### 1.4 Generation Layer ✅
- [x] Prompt templates (system, user, few-shot)
- [x] Streaming responses via Gemini (`langchain-google-genai`)
- [x] Citations: inline `[doc_id]` with clickable chunks
- [x] Response formatting: markdown, structured output
- [x] Token counting & cost estimation
- [x] Provider fallback chain: Gemini → OpenRouter → Groq

### 1.5 First Working End-to-End Test ✅
- [x] Ingest 3-5 sample documents
- [x] Query via CLI/script
- [x] Verify citations trace back to sources

---

## Phase 2: Production Hardening (Week 3-4)

**Goal**: Observable, evaluable, cached, secure.

### 2.1 Observability ✅
- [x] Structured JSON logs with trace_id / span_id propagation
  - `src/core/logging.py` — structlog processor auto-injects OTel `trace_id` / `span_id` into every log record
- [x] Phoenix integration: spans for embed → retrieve → rerank → generate
  - `src/core/tracing.py` — OpenTelemetry tracer; **LangSmith OTLP primary** (`LANGSMITH_ENDPOINT/otel/v1/traces` with `x-api-key`); Phoenix at `PHOENIX_URL` as fallback; console if neither reachable
  - Spans wired into `embedder.py`, `retriever.py`, `generation/chain.py`; `trace_rerank` helper ready for the rerank stage
  - Phoenix removed from `docker-compose.yml` (no local service required)
- [x] Metrics: P50/P95/P99 latency, throughput, error rates
  - `src/core/metrics.py` — thread-safe in-memory registry (counters + latency histograms) with `track_latency` decorator
- [x] Correlation IDs across all services
  - Every log inside a span carries the same `trace_id`/`span_id` as the exported span

### 2.2 Evaluation Framework ✅
- [x] RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
  - `src/evaluation/ragas_eval.py` — `RAGASEvaluator` using our LLM/embedding providers
- [x] Custom: citation accuracy, hallucination rate
  - `src/evaluation/custom_eval.py` — `CitationAccuracyEvaluator`, `HallucinationRateEvaluator` (LLM-as-judge)
- [x] Golden dataset (50-200 QA pairs) in `data/eval/`
  - `src/evaluation/golden_dataset.py` — `GoldenDataset`, `QAPair`, JSON persistence
  - Sample dataset created at `data/eval/golden_set.json`
- [x] CI integration: run evals on change
  - `src/evaluation/runner.py` — `EvaluationRunner`, `EvalConfig`, `run_evaluation()`
  - Produces `EvalRunResult` with aggregate + per-question metrics
- [x] `eval_runs` + `eval_results` tables (see [backendschema.md](backendschema.md))
  - Schema defined; ready for PostgreSQL + Alembic in Phase 5

### 2.3 Caching & Performance ✅
- [x] Redis: embeddings, retrieval results, LLM responses
  - `src/core/cache.py` — backend-agnostic `CacheBackend` ABC + `InMemoryCache` (LRU + TTL, thread-safe) + `RedisCache` (JSON, namespaced `dr:` keys, fail-fast ping, prefix-scoped `clear()`)
  - `create_cache()` auto-selects Redis when reachable and **falls back to in-memory** otherwise — same code locally (no Docker) and in production
  - Embeddings: per-text cache `emb:*` in all three providers (`huggingface`, `local`, `gemini`)
  - Retrieval: `ret:*` keyed on collection + embedding model + query + search type + top_k + score_threshold + filter + weights, TTL 1h
  - LLM: `llm:*` keyed on full prompt + provider + model + temperature, TTL 24h (structured output intentionally bypasses the cache)
- [x] TTL strategies, request dedup (in-flight coalescing)
  - TTLs: embeddings use backend default, retrieval 3600s, LLM 86400s (`CACHE_TTL` configurable)
  - `RAGGenerator.generate` coalesces concurrent identical requests with a leader/`Event` handshake (30s cap) instead of duplicate LLM calls; lock released in `finally`
  - Stable keys via `make_cache_key()` (BLAKE2b) — Python's `hash()` is per-process randomized and would break a shared Redis cache
  - Hit/miss metrics: `retrieve.cache_hits`, `generate.cache_hits`, `generate.coalesced_hits`; misses continue to be counted on the normal request metric
- [x] Connection pooling, async throughout
  - redis-py client pool reused through the module-level singleton (`get_cache()` / `close_cache()`), 1s connect/socket timeouts
  - Async callers: cache ops are microsecond-scale and non-blocking for the in-memory backend; the async API path is unaffected (see `src/api`)
  - 18 cache tests in `tests/test_cache.py` (TTL expiry, LRU eviction, key stability, factory fallback; Redis tests auto-skip when no server)
  - `tests/conftest.py` isolates the process-wide cache per test so cached results can't leak across temporary collections

### 2.4 Security & Reliability ✅
- [x] Secrets from env only (gitleaks pre-commit)
  - All credentials come from `Settings` (env / `.env`); `gitleaks` + `detect-private-key` run on every commit
  - `src/core/security.py` — `redact()`, `mask_secret()`, `redact_mapping()` for log-safe payloads; `SECRET_FIELDS` lists every credential so none can be logged by accident
  - `validate_security_config()` refuses to start in production with `DEBUG=true`, wildcard `CORS_ORIGINS`, an empty `API_KEY` when `REQUIRE_AUTH=true`, or a placeholder key
- [x] Rate limiting (per user/IP/endpoint)
  - `src/api/rate_limit.py` — fixed-window counters, `InMemoryRateLimiter` (local, no Docker) + `RedisRateLimiter` (shared across workers); `create_rate_limiter()` falls back to memory when Redis is unreachable — same code path in both environments
  - Identity = SHA-256 hash of a **valid** API key (`key:<16 hex>`) else client IP; `X-Forwarded-For` honored only from `TRUSTED_PROXIES`, so rotating fake keys can't bypass the limit
  - Per method + path bucket; `429` with `Retry-After`, `X-RateLimit-Limit/Remaining/Reset`; `/health`, `/metrics`, `/docs` exempt
- [x] Pydantic input validation, HTTP surface
  - `src/api/schemas.py` — `AskRequest` / `IngestRequest` with length, range, enum, tenant-pattern, control-character, and extra-field rejection; ingestion paths go through `resolve_safe_path()` (traversal, null bytes, extension allowlist, size cap, optional `INGEST_ROOT` jail)
  - `src/api/app.py`, `routes.py`, `middleware.py`, `auth.py`, `responses.py` — FastAPI app factory, `POST /api/v1/ask`, `POST /api/v1/ingest`, `GET /health`
  - Every response carries `X-Request-ID` (validated inbound or generated) and security headers; body-size limit returns `413`; errors use one `{"error": {code, message, details, request_id}}` envelope that never echoes raw input
  - API key auth via `X-API-Key` or `Authorization: Bearer`, constant-time comparison, identities hashed before they touch logs/metrics/Redis
- [x] Circuit breakers, retry w/ backoff + jitter
  - `src/core/resilience.py` — thread-safe `CircuitBreaker` (closed/open/half-open, injectable clock, abandoned-probe recovery) + `CircuitBreakerRegistry`; `retry_policy()` = exponential backoff with jitter, transient failures only; `call_with_resilience()` retries **inside** the breaker so one outage counts as one failure
  - Wired into LLM calls (`llm:<provider>`), embeddings (`embed:huggingface|local|gemini`), and Qdrant (`qdrant`)
  - Per-request timeout on every provider client (`LLM_TIMEOUT_SECONDS`, SDK `max_retries=0`) so a stalled provider fails instead of hanging forever — without a deadline neither retries nor breakers can fire
  - Quota/plan/auth errors are classified as permanent, so a daily-quota 429 ("retry in ~18h") fails immediately instead of being retried three times; only genuinely transient failures (timeouts, connection resets, short 429s) retry
  - Degraded tiers get their own breaker (`llm:<provider>:degraded`), so an overloaded large model can't disable its own small-model fallback
  - Unknown routes and wrong methods return the same error envelope (`NOT_FOUND` / `METHOD_NOT_ALLOWED`) as every other failure
- [x] Graceful degradation (70B → 8B fallback)
  - `invoke_with_fallback()` tries every full-size model first, then each provider's smaller model (`*_DEGRADED_MODEL`), skipping providers whose breaker is open; streaming and async streaming follow the same plan
  - `outcome` dict reports the winning provider/model and whether the degraded tier was used; `generate.degraded` / `generate.circuit_rejected` metrics track how often it happens
  - Empty retrieval is handled explicitly: the prompt says there is nothing to cite, so answers don't ship with invented markers (e.g. `[doc-a]`) alongside an empty sources list
  - Default Gemini models are the `-latest` aliases (`gemini-flash-latest` → `gemini-flash-lite-latest`) so pinned names can't silently rot; content blocks are normalised in `_content_text()` because Gemini 3.x returns them as dicts (answers were previously wrapped in a Python repr)
- Tests: `tests/test_resilience.py`, `test_security.py`, `test_rate_limit.py`, `test_api.py`

### 2.5 Data Pipeline Robustness ✅
- [x] Idempotent ingestion (content hash dedup)
  - Document-level dedup via SHA-256 content hash stored on every chunk payload
  - Chunk-level dedup skips re-embedding unchanged chunks
  - `ingest_file` returns `chunks_skipped=-1` for full-document duplicate
- [x] Incremental updates
  - File fingerprinting (size + mtime_ns) via `_get_file_fingerprint()`
  - `needs_update()` and `incremental_ingest()` methods
  - Skips re-processing with metadata `"skipped: unchanged"`
- [x] Dead letter handling for failed docs
  - `DeadLetterEntry` model with error type, stage, retry count, payload
  - `DeadLetterQueue` with max size, add/get/retry/remove/clear
  - Automatic enqueue on ingestion failure; `retry_dead_letter_task` for retry
- [x] Celery background jobs + WebSocket progress
  - Celery app in `src/jobs/__init__.py` with Redis broker, signal handlers, queue routing
  - Tasks: `ingest_file_task`, `ingest_batch_task`, `retry_dead_letter_task`
  - WebSocket `/ws/ingest/{task_id}` with real-time progress updates
  - REST endpoints: async ingest (single/batch/incremental), task status/revoke, dead letter retry
  - `IngestBatchRequest/Response`, `TaskStatusResponse` schemas
- Tests: `tests/test_pipeline.py` (17 tests), `tests/test_resilience.py`, `tests/test_security.py`, `tests/test_rate_limit.py`, `test_api.py`

---

*Plan Version 3.2 | Last Updated: 2026-11-17*

## Phase 3: Agentic Capabilities (Week 5-6)

**Goal**: Multi-step reasoning, tool use, planning.

### 3.1 LangGraph Agent ✅
- [x] State: messages, retrieved_docs, citations, plan
  - `src/agent/state.py` — `AgentState` TypedDict with all required fields
  - `create_initial_state()` factory function
- [x] Nodes: retrieve, rerank, generate, reflect, tool_use
  - `src/agent/nodes.py` — All five nodes implemented
  - `RetrieveTool`, `RerankTool`, `GenerateTool`, `ReflectTool`, `CalculatorTool`
- [x] Conditional routing
  - `route_to_tool()` and `should_continue()` for dynamic routing
  - `StateGraph` with conditional edges between all nodes
- [x] Checkpointing (LangGraph built-in, SqliteSaver)
  - `src/agent/nodes.py` — `create_agent_graph(checkpointer=SqliteSaver.from_conn_string("sqlite:///checkpoints.db"))`
- [x] Human-in-the-loop (interrupt-based)
  - `src/agent/nodes.py` — `human_review_node` with `interrupt()`, `route_after_human_review`

### 3.2 Tools ✅
- [x] `RetrieveTool`, `RerankTool`, `GenerateTool`, `ReflectTool`, `CalculatorTool`
  - `src/agent/tools.py` — All tools implemented as LangChain BaseTool subclasses
  - Lazy initialization to avoid Qdrant lock conflicts at import time
- [x] `SummarizeTool`
  - `src/agent/tools.py` — Extractive summarization with configurable length/focus
- [x] `WebSearchTool`
  - `src/agent/tools.py` — Placeholder for SerpAPI/Brave/Google CSE integration
- [x] `CodeExecutionTool`
  - `src/agent/tools.py` — Sandboxed Python execution with timeout, AST validation
- [x] Dynamic tool selection by query type
  - `src/agent/tools.py` — `select_tools_for_query()` with keyword-based heuristic

### 3.3 Agentic Patterns
- [x] Query decomposition (via `plan_node` creating execution plan)
- [x] Iterative retrieval (via `reflect` node feeding back to `rerank`/`generate`)
- [x] Self-correction (via `reflect` tool identifying issues)
- [ ] Planning before acting (partial - basic plan created, not yet LLM-driven)
- [ ] Multi-hop reasoning

### 3.4 Memory ✅
- [x] Short-term conversation memory (Redis)
  - `src/agent/memory.py` — `ConversationMemory` with Redis backend, TTL, session isolation
- [x] Long-term memory (vector store)
  - `src/agent/memory.py` — `LongTermMemory` using Qdrant, fact storage/retrieval
  - `MemoryManager` combining both with unified interface

---

*Plan Version 3.2 | Last Updated: 2026-11-17*

*Plan Version 3.1 | Last Updated: 2026-11-17*

## Phase 4: Advanced RAG Techniques (Week 7-8)

**Goal**: SOTA retrieval quality.

- [ ] HyDE (Hypothetical Document Embeddings)
- [ ] Query expansion via LLM
- [ ] Multi-query retrieval
- [ ] Parent Document Retriever (small chunks → parent context)
- [ ] Sentence Window Retrieval
- [ ] Context compression ("lost in the middle" mitigation)
- [ ] Dynamic top-k by query complexity
- [ ] Optional: knowledge graph retrieval, ColBERT

---

## Phase 5: API & Deployment (Week 9-10)

**Goal**: Production API, UI, deployment.

### 5.1 FastAPI
- [ ] Async endpoints with dependency injection
- [ ] Pydantic v2 request/response models
- [ ] Middleware: CORS, auth, logging, rate limit, tracing
- [ ] SSE + WebSocket streaming
- [ ] OpenAPI docs at `/docs`

### 5.2 Authentication & Multi-tenancy
- [ ] JWT auth (RS256) + API keys (SHA-256 hashed)
- [ ] Roles: admin, user, viewer
- [ ] Tenant isolation via Qdrant payload filtering

### 5.3 Frontend (Streamlit per [uiux.md](uiux.md))
- [ ] Chat interface with streaming
- [ ] Upload + ingestion progress
- [ ] Settings (model, top-k, temperature, hybrid)
- [ ] Conversation history sidebar
- [ ] Evaluation dashboard

### 5.4 Deployment
- [ ] Docker multi-stage build
- [ ] GitHub Actions CI/CD (lint → type → test → build)
- [ ] Health/readiness probes
- [ ] HPA config (optional)

---

## Phase 6: Evaluation & Optimization (Week 11-12)

**Goal**: Measure, iterate, optimize, document.

### 6.1 Comprehensive Evaluation
- [ ] Offline RAGAS + custom metrics on golden set
- [ ] Online user feedback (thumbs up/down)
- [ ] LLM-as-judge for subjective quality
- [ ] Cost/latency tracking per query type

### 6.2 Optimization Loop
- [ ] Prompt optimization
- [ ] Chunking strategy A/B tests
- [ ] Embedding model comparison
- [ ] Reranker ablation
- [ ] Token budget tuning

### 6.3 Documentation
- [ ] ADRs (Architecture Decision Records)
- [ ] Runbooks
- [ ] Lessons learned

---

## Free API Quota Summary (Monthly)

| Service | Free Limit | Est. Usage (1K queries/day) | Headroom |
|---------|------------|----------------------------|----------|
| Gemini API (primary) | ~10-15 RPM, 250-1000 RPD | ~15-20 RPM (batched) | ⚠️ Watch RPD during eval |
| OpenRouter (fallback) | 20 RPM / 50 RPD | ~2K req/month | ⚠️ Tight — only fallback traffic |
| HF Embeddings (bge-large) | 30K req/month | ~15K req/month | ✅ OK |
| Qdrant Cloud | 1GB storage | ~500MB | ✅ OK |
| Cohere Rerank | 1K req/month | — | ✅ Using local instead |
| Phoenix (local) | Unlimited | — | ✅ Instead of LangSmith |
| Upstash Redis | 10K req/day | ~5K req/day | ✅ OK |

**Local-first decision**: Use local `bge-reranker` + local Phoenix to avoid quota ceilings entirely.

---

## Learning Objectives Checklist

### RAG Fundamentals
- [ ] Chunking strategies & tradeoffs
- [ ] Embedding model selection & evaluation
- [ ] Vector vs keyword vs hybrid search
- [ ] Reranking impact on quality
- [ ] Context window management

### Production Engineering
- [x] Observability (logs, metrics, traces)
- [x] Evaluation-driven development
- [ ] Caching strategies
- [ ] Circuit breakers & resilience
- [ ] CI/CD for ML systems

### Agentic AI
- [ ] LangGraph state machines
- [ ] Tool calling / function calling
- [ ] Planning & decomposition
- [ ] Self-reflection & correction
- [ ] Multi-agent orchestration

### Advanced Topics
- [ ] Query rewriting/expansion
- [ ] Multi-tenancy & isolation
- [ ] Evaluation methodologies
- [ ] Security (auth, rate limits, secrets)

---

## Quick Start

```bash
# Dependencies
uv sync --dev

# Environment
cp .env.example .env   # Add GOOGLE_API_KEY, OPENROUTER_API_KEY, HF_API_TOKEN, QDRANT_URL

# Local stack
docker-compose up -d   # Qdrant, Redis, Phoenix

# Run API
uv run uvicorn src.api.main:app --reload

# Run UI
uv run streamlit run src/ui/app.py

# Tests, lint, types
uv run pytest
uv run ruff check src/
uv run mypy src/
```

---

## Next Step

**Phase 2.5**: Data Pipeline Robustness - idempotent ingestion (content hash dedup); incremental updates; dead letter handling for failed docs; Celery background jobs + WebSocket progress.

---

*Plan Version 2.4 | Last Updated: 2026-11-17*
