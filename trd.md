# DeepResearcher - Technical Requirements Document (TRD)

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DEEPRESEARCHER ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐  │
│  │   CLIENT     │───▶│   FASTAPI    │───▶│  LANGGRAPH   │───▶│  GROQ    │  │
│  │  (Streamlit) │    │   GATEWAY    │    │   AGENT      │    │  (LLM)   │  │
│  └──────────────┘    └──────┬───────┘    └──────┬───────┘    └──────────┘  │
│                             │                   │                             │
│                    ┌────────▼────────┐  ┌───────▼───────┐                    │
│                    │  MIDDLEWARE     │  │    TOOLS      │                    │
│                    │ • Auth/JWT      │  │ • Retrieve    │                    │
│                    │ • Rate Limit    │  │ • Rerank      │                    │
│                    │ • Logging       │  │ • Summarize   │                    │
│                    │ • CORS          │  │ • Calculator  │                    │
│                    │ • Tracing       │  │ • Web Search  │                    │
│                    └────────┬────────┘  │ • Code Exec   │                    │
│                             │           └───────┬───────┘                    │
│                    ┌────────▼────────┐          │                             │
│                    │  DEPENDENCY     │          │                             │
│                    │  INJECTION      │          │                             │
│                    │ • Qdrant Client │◀─────────┘                             │
│                    │ • Redis Client  │                                        │
│                    │ • Embedding     │                                        │
│                    │ • Reranker      │                                        │
│                    └────────┬────────┘                                        │
│                             │                                                 │
│           ┌─────────────────┼─────────────────┐                              │
│           ▼                 ▼                 ▼                              │
│    ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                        │
│    │   QDRANT    │  │    REDIS    │  │  PHOENIX    │                        │
│    │ (Vector DB) │  │   (Cache)   │  │ (Tracing)   │                        │
│    └─────────────┘  └─────────────┘  └─────────────┘                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technology Decisions & Rationale

| Layer | Choice | Rationale | Alternative Considered |
|-------|--------|-----------|------------------------|
| **Language** | Python 3.11+ | ML ecosystem, async support, type hints | Go (less ML libs), TypeScript (no ML) |
| **Package Manager** | uv | 10-100x faster than pip, lockfile, virtual envs | pip, poetry |
| **Config** | Pydantic Settings | Type-safe, env-file, validation, aliases | python-dotenv + dataclasses |
| **Logging** | structlog + Rich | Structured JSON + pretty dev console | standard logging, loguru |
| **API Framework** | FastAPI | Async, OpenAPI, DI, Pydantic integration | Flask (sync), Starlette (low-level) |
| **Orchestration** | LangGraph | Stateful, cyclic graphs, checkpointing | LangChain (linear), custom |
| **Vector DB** | Qdrant | Hybrid search, filtering, gRPC, 1GB free | Chroma (no hybrid), Pinecone (paid) |
| **LLM (primary)** | Google Gemini 2.5 Flash-Lite | Free in/out tokens, 1M context window, native function calling (agent-critical), structured output | OpenRouter, Groq, local |
| **LLM (fallback)** | OpenRouter | One key → 300+ models, `:free` variants, provider fallback routing, BYOK support | None |
| **Embeddings** | HF bge-large-en-v1.5 | 1024-dim, SOTA open, free API/local | OpenAI (paid), instructor-xl, Gemini embeddings |
| **Reranking** | Local bge-reranker-large | Unlimited, no API quota, cross-encoder | Cohere (1K/mo free) |
| **Caching** | Redis + Upstash | Distributed, TTL, LRU, 10K req/day free | In-memory (no persistence) |
| **Tracing** | Phoenix (local) | Unlimited, open source, OTel compatible | LangSmith (5K/mo free) |
| **Evaluation** | RAGAS | Industry standard, open source | Custom, TruLens |
| **Frontend** | Streamlit | Rapid prototyping, chat components | Gradio, custom React |
| **Queue** | Celery + Redis | Mature, distributed, monitoring | Dramatiq (simpler), RQ |
| **Container** | Docker multi-stage | Reproducible, small prod images | Podman, buildpacks |
| **CI/CD** | GitHub Actions | Free for public, integrated | GitLab CI, CircleCI |

---

## 3. API Specification

### 3.1 REST Endpoints

#### Authentication
```
POST   /api/v1/auth/login           # Email/password → JWT
POST   /api/v1/auth/refresh         # Refresh token → new JWT
POST   /api/v1/auth/api-keys        # Create API key (admin)
GET    /api/v1/auth/api-keys        # List API keys
DELETE /api/v1/auth/api-keys/{id}   # Revoke API key
```

#### Documents
```
POST   /api/v1/documents            # Upload & ingest (multipart)
GET    /api/v1/documents            # List with pagination/filter
GET    /api/v1/documents/{id}       # Get document metadata
DELETE /api/v1/documents/{id}       # Delete document + vectors
GET    /api/v1/documents/{id}/chunks # List document chunks
POST   /api/v1/documents/batch      # Batch ingest from URLs/paths
GET    /api/v1/documents/stats      # Collection statistics
```

#### Query / Chat
```
POST   /api/v1/query                # Single query (sync)
POST   /api/v1/query/stream         # Streaming query (SSE)
POST   /api/v1/chat                 # Multi-turn conversation
POST   /api/v1/chat/stream          # Streaming chat (SSE)
GET    /api/v1/chat/{id}/history    # Conversation history
DELETE /api/v1/chat/{id}            # Delete conversation
```

#### Evaluation
```
POST   /api/v1/eval/run             # Run evaluation on golden set
GET    /api/v1/eval/results         # List evaluation runs
GET    /api/v1/eval/results/{id}    # Detailed eval results
GET    /api/v1/eval/golden-set      # Get golden dataset
POST   /api/v1/eval/golden-set      # Add QA pair to golden set
```

#### System
```
GET    /health                      # Liveness probe
GET    /ready                       # Readiness probe
GET    /api/v1/system/stats         # System metrics
GET    /api/v1/system/config        # Public config (models, limits)
```

### 3.2 WebSocket Endpoints
```
WS /api/v1/ws/chat/{conversation_id}  # Real-time chat streaming
WS /api/v1/ws/ingest/{job_id}         # Ingestion progress updates
```

### 3.3 Request/Response Models

#### Query Request
```json
{
  "query": "string (required, 1-2000 chars)",
  "top_k": "integer (optional, 1-50, default: 10)",
  "rerank_top_k": "integer (optional, 1-20, default: 5)",
  "score_threshold": "float (optional, 0.0-1.0, default: 0.5)",
  "use_hybrid": "boolean (optional, default: true)",
  "filters": {
    "source": "string[]",
    "date_from": "ISO8601",
    "date_to": "ISO8601",
    "tags": "string[]"
  },
  "model": "string (optional, provider-aware enum: [gemini-2.5-flash-lite, gemini-3.8-flash, meta-llama/llama-3.3-70b-instruct:free, llama-3.1-70b-versatile])",
  "temperature": "float (optional, 0.0-1.0, default: 0.1)",
  "stream": "boolean (optional, default: false)",
  "include_citations": "boolean (optional, default: true)"
}
```

#### Query Response (Sync)
```json
{
  "answer": "string",
  "citations": [
    {
      "doc_id": "string",
      "chunk_id": "string",
      "content": "string (truncated)",
      "score": "float",
      "metadata": {}
    }
  ],
  "metadata": {
    "model": "string",
    "tokens_used": {"prompt": 0, "completion": 0, "total": 0},
    "latency_ms": {"embedding": 0, "retrieval": 0, "rerank": 0, "generation": 0, "total": 0},
    "retrieved_count": 0,
    "reranked_count": 0
  }
}
```

#### Streaming Response (SSE)
```
event: token
data: {"token": "string", "done": false}

event: citation
data: {"citations": [...]}

event: done
data: {"metadata": {...}}

event: error
data: {"code": "string", "message": "string"}
```

#### Document Upload Response
```json
{
  "document_id": "string",
  "filename": "string",
  "status": "enum: [pending, processing, completed, failed]",
  "chunks_created": 0,
  "error": "string (if failed)"
}
```

---

## 4. Data Models

### 4.1 Document (Qdrant Payload)
```json
{
  "id": "uuid",
  "content": "string (chunk text)",
  "content_hash": "sha256",
  "document_id": "uuid",
  "document_name": "string",
  "source": "string (filename/url)",
  "source_type": "enum: [pdf, markdown, txt, html, youtube]",
  "chunk_index": "integer",
  "chunk_count": "integer",
  "section_header": "string (optional)",
  "page_number": "integer (optional)",
  "author": "string (optional)",
  "created_at": "ISO8601",
  "updated_at": "ISO8601",
  "tags": "string[]",
  "tenant_id": "string (for multi-tenancy)",
  "embedding_model": "string",
  "embedding_dim": 1024
}
```

### 4.2 Conversation (Redis + Optional Persistence)
```json
{
  "id": "uuid",
  "user_id": "string",
  "title": "string (auto-generated from first query)",
  "messages": [
    {
      "role": "user|assistant",
      "content": "string",
      "citations": [],
      "metadata": {},
      "timestamp": "ISO8601"
    }
  ],
  "created_at": "ISO8601",
  "updated_at": "ISO8601"
}
```

### 4.3 Evaluation Run
```json
{
  "id": "uuid",
  "dataset_name": "string",
  "metrics": {
    "faithfulness": 0.0,
    "answer_relevancy": 0.0,
    "context_precision": 0.0,
    "context_recall": 0.0,
    "citation_accuracy": 0.0,
    "hallucination_rate": 0.0
  },
  "config_snapshot": {},
  "created_at": "ISO8601",
  "duration_seconds": 0
}
```

---

## 5. Infrastructure Requirements

### 5.1 Local Development
```yaml
# docker-compose.yml services
services:
  qdrant:
    image: qdrant/qdrant:v1.10
    ports: ["6333:6333", "6334:6334"]
    volumes: ["qdrant_data:/qdrant/storage"]

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    command: redis-server --appendonly yes --maxmemory 256mb --maxmemory-policy allkeys-lru

  phoenix:
    image: arizephoenix/phoenix:latest
    ports: ["6006:6006"]
    volumes: ["phoenix_data:/app/data"]
```

### 5.2 Production (Container)
```dockerfile
# Multi-stage build
FROM python:3.11-slim AS builder
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /app/.venv .venv
COPY src ./src
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 5.3 Resource Requirements

| Component | CPU | Memory | Storage | Network |
|-----------|-----|--------|---------|---------|
| API (per replica) | 0.5-1 vCPU | 512MB-1GB | 100MB | 100Mbps |
| Qdrant | 1-2 vCPU | 2-4GB | 1-10GB | 1Gbps |
| Redis | 0.25 vCPU | 256-512MB | 100MB | 1Gbps |
| Phoenix | 0.5 vCPU | 1-2GB | 1GB | 100Mbps |
| Celery Worker | 0.5 vCPU | 1-2GB | 100MB | 100Mbps |

---

## 6. Security Requirements

### 6.1 Authentication
- **JWT RS256**: 15min access token, 7d refresh token
- **API Keys**: SHA-256 hashed, prefix `dr_` for identification
- **Token Storage**: HttpOnly Secure cookies (web) / Authorization header (API)

### 6.2 Authorization
| Role | Permissions |
|------|-------------|
| Admin | All + user management, API keys, system config |
| User | Own documents, queries, conversations |
| Viewer | Read-only queries, no ingestion |

### 6.3 Data Protection
- **Encryption at rest**: Qdrant/Redis managed (cloud) or LUKS (self-hosted)
- **Encryption in transit**: TLS 1.3 for all external connections
- **PII handling**: No PII in logs; content hashes only
- **Secrets**: Never in code; `.env` + GitHub Secrets + Doppler/1Password

### 6.4 Rate Limiting
```
Tier          | Requests/min | Burst
--------------|--------------|------
Anonymous     | 10           | 20
Authenticated | 60           | 120
API Key       | 300          | 600
Admin         | 1000         | 2000
```

---

## 7. Observability Requirements

### 7.1 Structured Logging
```json
{
  "timestamp": "2026-10-11T10:30:00.123Z",
  "level": "INFO",
  "logger": "src.retrieval.hybrid",
  "event": "hybrid_search_completed",
  "trace_id": "abc123",
  "span_id": "def456",
  "user_id": "user_789",
  "query": "what is RAG?",
  "retrieved_count": 10,
  "reranked_count": 5,
  "latency_ms": {"dense": 45, "sparse": 12, "rrf": 3, "rerank": 89, "total": 149}
}
```

### 7.2 Distributed Tracing (Phoenix/LangSmith)
Spans for each operation:
- `embedding.generate` - input texts, model, token count, latency
- `retrieval.dense` - query vector, top_k, filter, results
- `retrieval.sparse` - query text, top_k, filter, results
- `retrieval.hybrid_rrf` - dense_results, sparse_results, fused_results
- `rerank.cross_encoder` - query, candidates, reranked_results
- `generation.llm_call` - prompt_template, model, tokens, latency
- `agent.step` - node_name, state_before, state_after

### 7.3 Metrics (Prometheus Format)
```
# HELP deepresearcher_query_duration_seconds Query latency histogram
# TYPE deepresearcher_query_duration_seconds histogram
deepresearcher_query_duration_seconds_bucket{le="0.5"} 1247
deepresearcher_query_duration_seconds_bucket{le="1.0"} 2103
deepresearcher_query_duration_seconds_bucket{le="2.0"} 2891
deepresearcher_query_duration_seconds_bucket{le="5.0"} 3102

# HELP deepresearcher_query_total Total queries by status
# TYPE deepresearcher_query_total counter
deepresearcher_query_total{status="success"} 2987
deepresearcher_query_total{status="error"} 12

# HELP deepresearcher_tokens_total Total tokens by type
# TYPE deepresearcher_tokens_total counter
deepresearcher_tokens_total{type="prompt"} 1523400
deepresearcher_tokens_total{type="completion"} 487200
```

---

## 8. Error Handling Strategy

### 8.1 Error Codes
| Code | HTTP | Description | Retryable |
|------|------|-------------|-----------|
| `VALIDATION_ERROR` | 400 | Invalid request body | No |
| `AUTHENTICATION_ERROR` | 401 | Invalid/expired token | No |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions | No |
| `NOT_FOUND` | 404 | Resource doesn't exist | No |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Yes (Retry-After) |
| `EMBEDDING_ERROR` | 502 | Embedding service failed | Yes |
| `RETRIEVAL_ERROR` | 502 | Vector DB failed | Yes |
| `GENERATION_ERROR` | 502 | LLM API failed | Yes |
| `RERANK_ERROR` | 502 | Reranker failed | Yes |
| `INGESTION_ERROR` | 500 | Document processing failed | No |
| `INTERNAL_ERROR` | 500 | Unexpected error | No |

### 8.2 Resilience Patterns
- **Circuit Breaker**: 5 failures → open 30s → half-open
- **Retry**: Exponential backoff (1s, 2s, 4s, 8s) + jitter (±25%)
- **Timeouts**: Embedding 10s, Retrieval 5s, Rerank 10s, Generation 60s
- **Fallback chain**: Gemini 2.5 Flash-Lite → OpenRouter free model → Groq 70B → Local model (if available)
- **Bulkhead**: Separate connection pools per external service

---

## 9. Configuration Management

### 9.1 Environment Variables (All in `.env`)
```bash
# Required
GOOGLE_API_KEY=          # Gemini (primary LLM)
OPENROUTER_API_KEY=      # Fallback LLM
HF_API_TOKEN=            # Embeddings
QDRANT_URL=
QDRANT_API_KEY=

# Model config (with defaults)
GOOGLE_MODEL=gemini-2.5-flash-lite
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct:free
GROQ_API_KEY=            # Optional legacy provider
GROQ_MODEL=llama-3.1-70b-versatile
LLM_PRIMARY=gemini
LLM_FALLBACKS=["openrouter", "groq"]
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
TOP_K=10
RERANK_TOP_K=5
CHUNK_SIZE=512
REDIS_URL=redis://localhost:6379/0
LOG_LEVEL=INFO
```

### 9.2 Feature Flags (via Config)
```python
class FeatureFlags:
    ENABLE_HYBRID_SEARCH = True
    ENABLE_RERANKING = True
    ENABLE_CACHING = True
    ENABLE_STREAMING = True
    ENABLE_AGENT = False  # Phase 3
    ENABLE_WEB_SEARCH = False
    ENABLE_CODE_EXEC = False
```

---

## 10. Testing Strategy

| Layer | Tool | Coverage Target |
|-------|------|-----------------|
| Unit | pytest + pytest-asyncio | 80%+ |
| Integration | pytest + testcontainers | Key paths |
| Contract | schemathesis (OpenAPI) | All endpoints |
| Load | locust | 50 concurrent users |
| E2E | playwright (UI) | Critical flows |
| Evaluation | RAGAS + custom | Regression detection |

---

## 11. Deployment Checklist

- [ ] All secrets in GitHub Secrets / Doppler
- [ ] Docker image builds and runs locally
- [ ] Health checks pass (`/health`, `/ready`)
- [ ] OpenAPI docs accessible at `/docs`
- [ ] Metrics endpoint at `/metrics`
- [ ] Logs output JSON to stdout
- [ ] Traces appear in Phoenix/LangSmith
- [ ] Rate limiting enforced
- [ ] CORS configured for frontend origin
- [ ] Database migrations (if any) run on startup
- [ ] Graceful shutdown (SIGTERM handling)
- [ ] Horizontal pod autoscaling config (K8s)

---

*Document Version: 1.0 | Last Updated: 2026-10-11*
