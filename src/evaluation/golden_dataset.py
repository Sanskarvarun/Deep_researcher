"""Golden dataset for evaluation.

Loads and manages curated QA pairs for regression testing.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.core.config import settings
from src.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class QAPair:
    """A single question-answer pair in the golden dataset."""

    question: str
    ground_truth: str
    relevant_doc_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "ground_truth": self.ground_truth,
            "relevant_doc_ids": self.relevant_doc_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QAPair:
        return cls(
            question=data["question"],
            ground_truth=data.get("ground_truth", data.get("answer", "")),
            relevant_doc_ids=data.get("relevant_doc_ids", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class GoldenDataset:
    """Collection of QA pairs for evaluation."""

    name: str
    description: str
    qa_pairs: list[QAPair]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.qa_pairs)

    def __getitem__(self, idx: int) -> QAPair:
        return self.qa_pairs[idx]

    def __iter__(self) -> Iterator[QAPair]:
        return iter(self.qa_pairs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
            "metadata": self.metadata,
        }

    def save(self, path: Path | str) -> None:
        """Save dataset to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("golden_dataset_saved", path=str(path), qa_count=len(self))

    @classmethod
    def load(cls, path: Path | str) -> GoldenDataset:
        """Load dataset from JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            name=data.get("name", path.stem),
            description=data.get("description", ""),
            qa_pairs=[QAPair.from_dict(qa) for qa in data.get("qa_pairs", [])],
            metadata=data.get("metadata", {}),
        )


def load_golden_dataset(
    path: Path | str | None = None,
) -> GoldenDataset:
    """Load the default golden dataset from config path."""
    path = path or settings.evaluation_dataset_path
    path = Path(path)

    if path.exists():
        return GoldenDataset.load(path)

    # Create a minimal default dataset if file doesn't exist
    log.warning("golden_dataset_not_found", path=str(path))
    return GoldenDataset(
        name="default",
        description="Default placeholder dataset - replace with curated QA pairs",
        qa_pairs=[
            QAPair(
                question="What is RAG?",
                ground_truth=(
                    "RAG (Retrieval-Augmented Generation) combines retrieval with generation."
                ),
                relevant_doc_ids=[],
                metadata={"topic": "rag_basics"},
            ),
        ],
    )


def create_sample_golden_dataset(path: Path | str) -> GoldenDataset:
    """Create a sample golden dataset file for reference."""
    path = Path(path)
    dataset = GoldenDataset(
        name="sample",
        description="Sample golden dataset - replace with domain-specific QA pairs",
        qa_pairs=[
            QAPair(
                question="What is RAG (Retrieval-Augmented Generation)?",
                ground_truth=(
                    "RAG is an AI framework that combines information retrieval "
                    "with text generation. It retrieves relevant documents from a "
                    "knowledge base and uses them as context for a language model "
                    "to generate answers."
                ),
                relevant_doc_ids=["doc_0"],
                metadata={"topic": "rag_basics", "difficulty": "easy"},
            ),
            QAPair(
                question="How does hybrid search work in RAG?",
                ground_truth=(
                    "Hybrid search combines dense vector search (semantic "
                    "similarity) with sparse keyword search (BM25). Results are "
                    "fused using Reciprocal Rank Fusion (RRF) to leverage both "
                    "semantic and lexical matching."
                ),
                relevant_doc_ids=["doc_0", "doc_1"],
                metadata={"topic": "retrieval", "difficulty": "medium"},
            ),
            QAPair(
                question="What is the purpose of reranking in RAG?",
                ground_truth=(
                    "Reranking uses a cross-encoder model to re-score retrieved "
                    "candidates based on query-document relevance, improving "
                    "precision by filtering out irrelevant results that passed "
                    "initial vector search."
                ),
                relevant_doc_ids=["doc_1"],
                metadata={"topic": "reranking", "difficulty": "medium"},
            ),
        ],
        metadata={"version": "1.0", "created_for": "deepresearcher"},
    )
    dataset.save(path)
    return dataset
