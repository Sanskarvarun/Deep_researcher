# DeepResearcher

Production-grade RAG agent with agentic capabilities using free APIs.

## Quick Start

```bash
# Install uv (fast Python package manager)
pip install uv

# Clone and setup
git clone <repo-url>
cd deepresearcher

# Install dependencies
uv sync --dev

# Copy environment template
cp .env.example .env
# Edit .env with your API keys

# Start local services (Qdrant, Redis, Phoenix)
docker-compose up -d

# Run the API
uv run uvicorn src.api.main:app --reload

# Run the UI (in another terminal)
uv run streamlit run src/ui/app.py
```

## Project Structure

```
deepresearcher/
├── src/
│   ├── core/           # Config, logging, exceptions
│   ├── ingestion/      # Document loading, chunking, embedding
│   ├── retrieval/      # Vector search, hybrid search, reranking
│   ├── generation/     # LLM calls, prompt templates, streaming
│   ├── agent/          # LangGraph agent, tools, state
│   ├── evaluation/     # RAGAS metrics, custom evals
│   ├── api/            # FastAPI routes, middleware
│   └── ui/             # Streamlit interface
├── tests/
├── data/               # Sample documents
└── scripts/            # Utility scripts
```

## Free APIs Used

| Service | Purpose | Free Tier |
|---------|---------|-----------|
| Google Gemini | Primary LLM (2.5 Flash-Lite, 1M ctx) | Free in/out tokens, ~10-15 RPM |
| OpenRouter | Fallback LLM (300+ models, `:free`) | 20 RPM / 50 RPD |
| HuggingFace | Embeddings (bge-large) | 30K req/month |
| Qdrant Cloud | Vector Database | 1GB storage |
| Local bge-reranker | Reranking | Unlimited |
| Phoenix (local) | Tracing | Unlimited |

**Provider fallback chain**: Gemini → OpenRouter → Groq (configurable via `LLM_PRIMARY` / `LLM_FALLBACKS`)

> Note: A Google AI Pro subscription covers the AI Studio web UI only — the [Gemini API free key](https://aistudio.google.com/apikey) is separate and always free.

## Development

```bash
# Run tests
uv run pytest

# Run linting
uv run ruff check .
uv run ruff format .

# Type checking
uv run mypy src/

# Install pre-commit hooks
uv run pre-commit install
```

## License

MIT
