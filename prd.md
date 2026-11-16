# DeepResearcher - Product Requirements Document (PRD)

## 1. Product Vision

**DeepResearcher** is a production-grade RAG (Retrieval-Augmented Generation) agent that enables users to query their private document collections with accurate, cited answers. It combines advanced retrieval techniques with agentic reasoning to handle complex, multi-hop questions while maintaining production reliability through observability, evaluation, and robust engineering practices.

**Core Value Proposition**: "Ask questions about your documents, get accurate answers with citations, and trust the system because it's built with production-grade practices."

---

## 2. Target Users

| Persona | Needs | Pain Points |
|---------|-------|-------------|
| **Knowledge Worker** | Query internal docs, reports, PDFs | Search is keyword-only; no synthesis across docs |
| **Researcher/Analyst** | Multi-hop reasoning, comparison | Manual cross-referencing; hallucination risk |
| **Developer** | Extensible RAG pipeline, eval framework | Black-box RAG tools; no observability |
| **Team Lead** | Cost control, quality metrics, audit trail | No visibility into RAG performance/costs |

---

## 3. Core Features (MVP - Phase 1-3)

### 3.1 Document Ingestion
- **Supported formats**: PDF, Markdown, TXT, HTML, YouTube transcripts
- **Chunking strategies**: Fixed-size, semantic, markdown-aware, recursive
- **Metadata extraction**: Source, date, author, section headers, content hash
- **Idempotent upsert**: Deduplication via content hash; incremental updates
- **Batch processing**: Async ingestion with progress tracking

### 3.2 Retrieval Engine
- **Dense search**: Vector similarity (Qdrant, cosine/euclidean)
- **Sparse search**: BM25 keyword matching
- **Hybrid search**: RRF (Reciprocal Rank Fusion) with configurable weights
- **Metadata filters**: Date range, source, tags, custom fields
- **Reranking**: Cross-encoder (local bge-reranker-large) + optional Cohere API

### 3.3 Generation & Answering
- **Streaming responses**: Token-by-token via WebSocket/SSE
- **Citations**: Inline `[doc_id]` references with clickable source context
- **Grounding**: Answer strictly from retrieved context; refuse if insufficient
- **Structured output**: JSON schema for programmatic consumption
- **Model selection**: Groq (Llama 3.1 70B/8B, Mixtral) with fallback

### 3.4 Agentic Capabilities (Phase 3)
- **Query decomposition**: Break complex questions into sub-queries
- **Iterative retrieval**: Retrieve → Reflect → Retrieve more
- **Self-correction**: Verify citations, retry on hallucination detection
- **Planning**: Generate execution plan before acting
- **Tool use**: Calculator, web search, code execution, summarization

### 3.5 Evaluation & Observability
- **Offline eval**: RAGAS (faithfulness, relevancy, precision, recall)
- **Custom evals**: Citation accuracy, hallucination rate, latency
- **Golden dataset**: Curated QA pairs for regression testing
- **Online signals**: Thumbs up/down, implicit feedback
- **Tracing**: Full request spans (embedding → retrieval → rerank → generation)
- **Metrics**: P50/P95/P99 latency, throughput, error rates, cost/query

---

## 4. Advanced Features (Phase 4-6)

### 4.1 Advanced Retrieval
- HyDE (Hypothetical Document Embeddings)
- Query expansion via LLM
- Multi-query retrieval
- Parent document retriever (small chunks → parent context)
- Sentence window retrieval
- Knowledge graph retrieval (optional)

### 4.2 Production Hardening
- Authentication (JWT) + API keys
- Multi-tenancy (namespace isolation in Qdrant)
- Rate limiting (per user/IP/endpoint)
- Circuit breakers, retry with backoff
- Request deduplication, connection pooling
- Redis caching (embeddings, retrieval, LLM responses)

### 4.3 Deployment & Operations
- FastAPI async API with OpenAPI docs
- Streamlit chat UI with settings panel
- Docker multi-stage build
- GitHub Actions CI/CD (lint, type, test, build, deploy)
- Health checks, readiness probes
- Horizontal scaling config

---

## 5. Non-Functional Requirements

| Category | Requirement | Target |
|----------|-------------|--------|
| **Performance** | P95 query latency | < 3s (cached < 500ms) |
| **Performance** | Throughput | 50+ concurrent queries |
| **Reliability** | Uptime | 99.9% |
| **Reliability** | Error rate | < 0.1% |
| **Quality** | Faithfulness (RAGAS) | > 0.85 |
| **Quality** | Answer relevancy | > 0.80 |
| **Security** | Data isolation | Tenant-level namespace |
| **Security** | API auth | JWT RS256 + API keys |
| **Cost** | Cost per query | < $0.01 (free APIs) |
| **Observability** | Trace coverage | 100% of requests |

---

## 6. Constraints & Assumptions

### Constraints
- **Free APIs only**: Groq, HuggingFace, Qdrant Cloud (1GB), Cohere (1K/mo), local models
- **No paid Cloud**: No Vertex AI, OpenAI, Pinecone, Weaviate Cloud
- **Python 3.11+**: Modern async stack
- **Single-region deployment**: No multi-region HA in MVP

### Assumptions
- User has Docker for local development
- Documents fit in 1GB Qdrant (≈1M chunks @ 512 tokens)
- Team size: 1-3 developers
- Evaluation dataset: 50-200 QA pairs initially

---

## 7. Success Metrics (KPIs)

| Metric | Baseline | Target (3 mo) |
|--------|----------|---------------|
| Query success rate | N/A | > 99% |
| Avg latency (p50) | N/A | < 1.5s |
| Faithfulness score | N/A | > 0.85 |
| User satisfaction (thumbs up) | N/A | > 80% |
| Cost per 1K queries | N/A | < $0.50 |
| Eval regression detection | Manual | Automated CI |

---

## 8. Out of Scope (Future)

- Fine-tuning embeddings/LLMs
- Multi-modal (images, audio, video)
- Real-time collaborative editing
- Advanced RBAC (beyond admin/user/viewer)
- On-premise air-gapped deployment
- Multi-language UI (English only for MVP)

---

## 9. Dependencies

### External Services (Free Tiers)
| Service | Purpose | Fallback |
|---------|---------|----------|
| Groq | LLM inference | Local Llama.cpp |
| HuggingFace Inference API | Embeddings | Local sentence-transformers |
| Qdrant Cloud | Vector DB | Local Qdrant/Chroma |
| Cohere | Reranking | Local bge-reranker |
| LangSmith / Phoenix | Tracing | Local Phoenix |

### Internal Dependencies
- Python 3.11+, uv package manager
- Docker, Docker Compose
- Git, GitHub (CI/CD)

---

## 10. Timeline Summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| 1. Foundation | 2 weeks | Working RAG pipeline |
| 2. Production Hardening | 2 weeks | Observable, evaluated, cached |
| 3. Agentic Capabilities | 2 weeks | Multi-step reasoning agent |
| 4. Advanced RAG | 2 weeks | SOTA retrieval quality |
| 5. API & Deployment | 2 weeks | Production API + UI |
| 6. Evaluation & Optimization | 2 weeks | Measured, optimized, documented |

**Total: 12 weeks (3 months)**

---

*Document Version: 1.0 | Last Updated: 2026-10-11*
