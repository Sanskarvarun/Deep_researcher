"""RAGAS evaluation integration.

Provides faithfulness, answer_relevancy, context_precision, context_recall
metrics using the RAGAS framework with our LLM/embedding providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ragas import evaluate  # type: ignore[import-untyped]
from ragas.metrics import (  # type: ignore[import-untyped]
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # type: ignore[import-untyped]

from src.core.config import settings
from src.core.logging import get_logger
from src.generation.llm import llm_factory

log = get_logger(__name__)

_RAGAS_METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
]


@dataclass
class RAGASEvalResult:
    """Result of a RAGAS evaluation."""

    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }

    @property
    def aggregate_score(self) -> float | None:
        """Mean of all non-None metric scores."""
        vals = [v for v in self.to_dict().values() if v is not None]
        return sum(vals) / len(vals) if vals else None


class RAGASEvaluator:
    """RAGAS evaluator using our LLM and embedding providers."""

    def __init__(
        self,
        llm_provider: str | None = None,
        embedding_provider: str | None = None,
    ) -> None:
        self.llm_provider = llm_provider or settings.llm_primary
        self.embedding_provider = embedding_provider or settings.embedding_provider
        self._llm: Any | None = None
        self._embeddings: Any | None = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = llm_factory.get_model(self.llm_provider)
        return self._llm

    @property
    def embeddings(self) -> Any:
        if self._embeddings is None:
            from src.ingestion.embedder import create_embedder

            self._embeddings = create_embedder(self.embedding_provider)
        return self._embeddings

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: list[str],
    ) -> RAGASEvalResult:
        """Run RAGAS evaluation on a batch of QA pairs.

        Args:
            questions: List of question strings.
            answers: List of generated answer strings.
            contexts: List of list of context strings (retrieved docs per question).
            ground_truths: List of ground truth answer strings.

        Returns:
            RAGASEvalResult with the four metric scores.
        """
        if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
            raise ValueError("All input lists must have the same length")

        dataset = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }

        # Configure RAGAS to use our LLM and embeddings
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]
        from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]

        ragas_llm = LangchainLLMWrapper(self.llm)
        ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)

        run_config = RunConfig(
            max_workers=2,
            max_wait=180,
            max_retries=3,
        )

        log.info(
            "ragas_eval_start",
            num_samples=len(questions),
            llm_provider=self.llm_provider,
            embedding_provider=self.embedding_provider,
        )

        result = evaluate(
            dataset=dataset,
            metrics=_RAGAS_METRICS,
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=run_config,
        )

        scores = result.to_pandas()
        eval_result = RAGASEvalResult(
            faithfulness=float(scores["faithfulness"].mean()) if "faithfulness" in scores else None,
            answer_relevancy=float(scores["answer_relevancy"].mean())
            if "answer_relevancy" in scores
            else None,
            context_precision=float(scores["context_precision"].mean())
            if "context_precision" in scores
            else None,
            context_recall=float(scores["context_recall"].mean())
            if "context_recall" in scores
            else None,
        )

        log.info(
            "ragas_eval_complete",
            faithfulness=eval_result.faithfulness,
            answer_relevancy=eval_result.answer_relevancy,
            context_precision=eval_result.context_precision,
            context_recall=eval_result.context_recall,
            aggregate=eval_result.aggregate_score,
        )

        return eval_result


def evaluate_rag(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
    llm_provider: str | None = None,
    embedding_provider: str | None = None,
) -> RAGASEvalResult:
    """Convenience function to run RAGAS evaluation."""
    evaluator = RAGASEvaluator(llm_provider=llm_provider, embedding_provider=embedding_provider)
    return evaluator.evaluate(questions, answers, contexts, ground_truths)
