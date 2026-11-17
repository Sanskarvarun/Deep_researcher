"""Memory management for the LangGraph agent.

Provides short-term conversation memory (Redis) and long-term memory (vector store).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import redis

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)


class ConversationMemory:
    """Short-term conversation memory stored in Redis.

    Stores message history per conversation/session with TTL.
    """

    def __init__(self, redis_url: str | None = None, ttl_seconds: int = 3600):
        self.redis_url = redis_url or settings.redis_url
        self.ttl = ttl_seconds
        self._client: redis.Redis | None = None

    @property
    def client(self) -> redis.Redis:
        if self._client is None:
            self._client = redis.from_url(self.redis_url, decode_responses=True)
        return self._client

    def _session_key(self, session_id: str) -> str:
        return f"conversation:{session_id}"

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        key = self._session_key(session_id)
        self.client.rpush(key, json.dumps(message))
        self.client.expire(key, self.ttl)

    def get_messages(self, session_id: str, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent messages from conversation history."""
        key = self._session_key(session_id)
        messages = self.client.lrange(key, -limit, -1)
        return [json.loads(msg) for msg in messages]

    def clear(self, session_id: str) -> None:
        """Clear conversation history for a session."""
        self.client.delete(self._session_key(session_id))

    def get_session_ids(self) -> list[str]:
        """Get all active session IDs."""
        return [key.split(":")[-1] for key in self.client.scan_iter("conversation:*")]


class LongTermMemory:
    """Long-term memory using the vector store.

    Stores important facts, preferences, and summaries from conversations
    for retrieval across sessions.
    """

    def __init__(self) -> None:
        # Import here to avoid circular imports
        from src.retrieval.retriever import Retriever

        self.retriever = Retriever()

    def store_fact(
        self,
        content: str,
        user_id: str | None = None,
        tags: list[str] | None = None,
        source: str = "conversation",
    ) -> str:
        """Store a fact in long-term memory.

        This creates a document in the vector store that can be retrieved later.
        """
        from src.ingestion.models import Document, SourceType
        from src.ingestion.pipeline import IngestionPipeline

        doc_id = uuid4()
        document = Document(
            id=doc_id,
            content=content,
            source="memory",
            source_type=SourceType.TXT,
            metadata={
                "user_id": user_id,
                "tags": tags or [],
                "source": source,
                "stored_at": datetime.now(UTC).isoformat(),
            },
        )

        pipeline = IngestionPipeline()
        pipeline.ingest_document(document)
        log.info("memory_fact_stored", doc_id=str(doc_id), user_id=user_id)
        return str(doc_id)

    def retrieve_facts(
        self,
        query: str,
        user_id: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant facts from long-term memory."""
        from src.retrieval.models import MetadataFilter

        filter_obj = MetadataFilter(tenant_id=user_id) if user_id else None
        results = self.retriever.search(query, top_k=top_k, filter=filter_obj)

        return [
            {
                "content": r.content,
                "score": r.score,
                "document_id": str(r.document_id) if r.document_id else None,
                "metadata": r.payload,
            }
            for r in results
        ]


class MemoryManager:
    """Combined memory manager for short and long-term memory."""

    def __init__(self) -> None:
        self.short_term = ConversationMemory()
        self.long_term = LongTermMemory()

    def add_user_message(self, session_id: str, content: str) -> None:
        """Add a user message to short-term memory."""
        self.short_term.add_message(session_id, "user", content)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        """Add an assistant message to short-term memory."""
        self.short_term.add_message(session_id, "assistant", content)

    def get_context(self, session_id: str, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent conversation context."""
        return self.short_term.get_messages(session_id, limit)

    def store_important_fact(
        self,
        content: str,
        user_id: str | None = None,
        tags: list[str] | None = None,
    ) -> str:
        """Store an important fact in long-term memory."""
        return self.long_term.store_fact(content, user_id=user_id, tags=tags)

    def retrieve_relevant_facts(
        self, query: str, user_id: str | None = None, top_k: int = 5
    ) -> list[dict[str, Any]]:
        """Retrieve relevant facts from long-term memory."""
        return self.long_term.retrieve_facts(query, user_id=user_id, top_k=top_k)
