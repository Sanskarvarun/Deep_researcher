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
| **Observability** | Phoenix (local) | Unlimited | LangSmith (5K/mo) |
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
- [x] `.env.example`, `docker-compose.yml` (Qdrant, Redis, Phoenix)
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

### 2.1 Observability
- [ ] Structured JSON logs with trace_id / span_id propagation
- [ ] Phoenix integration: spans for embed → retrieve → rerank → generate
- [ ] Metrics: P50/P95/P99 latency, throughput, error rates
- [ ] Correlation IDs across all services

### 2.2 Evaluation Framework
- [ ] RAGAS: faithfulness, answer_relevancy, context_precision, context_recall
- [ ] Custom: citation accuracy, hallucination rate
- [ ] Golden dataset (50-200 QA pairs) in `data/eval/`
- [ ] CI integration: run evals on change
- [ ] `eval_runs` + `eval_results` tables (see [backendschema.md](backendschema.md))

### 2.3 Caching & Performance
- [ ] Redis: embeddings, retrieval results, LLM responses
- [ ] TTL strategies, request dedup (in-flight coalescing)
- [ ] Connection pooling, async throughout

### 2.4 Security & Reliability
- [ ] Secrets from env only (gitleaks pre-commit)
- [ ] Rate limiting (per user/IP/endpoint)
- [ ] Pydantic input validation
- [ ] Circuit breakers, retry w/ backoff + jitter
- [ ] Graceful degradation (70B → 8B fallback)

### 2.5 Data Pipeline Robustness
- [ ] Idempotent ingestion (content hash dedup)
- [ ] Incremental updates
- [ ] Dead letter handling for failed docs
- [ ] Celery background jobs + WebSocket progress

---

## Phase 3: Agentic Capabilities (Week 5-6)

**Goal**: Multi-step reasoning, tool use, planning.

### 3.1 LangGraph Agent
- [ ] State: messages, retrieved_docs, citations, plan
- [ ] Nodes: retrieve, rerank, generate, reflect, tool_use
- [ ] Conditional routing, checkpointing, human-in-the-loop

### 3.2 Tools
- [ ] `RetrieveTool`, `RerankTool`, `SummarizeTool`
- [ ] `CalculatorTool`, `WebSearchTool`, `CodeExecutionTool`
- [ ] Dynamic tool selection by query type

### 3.3 Agentic Patterns
- [ ] Query decomposition
- [ ] Iterative retrieval (retrieve → reflect → retrieve)
- [ ] Self-correction (verify citations, retry on hallucination)
- [ ] Planning before acting
- [ ] Multi-hop reasoning

### 3.4 Memory
- [ ] Short-term conversation memory (Redis)
- [ ] Long-term memory (vector store)

---

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
- [ ] Observability (logs, metrics, traces)
- [ ] Evaluation-driven development
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

**Phase 1.3**: Basic retrieval — dense search, BM25 sparse, hybrid RRF, metadata filters (retriever module).

---

*Plan Version 2.0 | Last Updated: 2026-10-11*
