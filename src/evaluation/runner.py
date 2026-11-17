"""Evaluation runner - orchestrates RAGAS + custom evals on golden dataset."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.core.config import settings
from src.core.logging import get_logger
from src.core.metrics import metrics
from src.core.tracing import trace_span
from src.evaluation.custom_eval import CustomEvaluator
from src.evaluation.golden_dataset import GoldenDataset, load_golden_dataset
from src.evaluation.ragas_eval import RAGASEvalResult, RAGASEvaluator
from src.generation.chain import RAGGenerator
from src.generation.citations import Citation
from src.retrieval.retriever import Retriever

log = get_logger(__name__)


@dataclass
class EvalConfig:
    """Configuration for an evaluation run."""

    dataset_path: str | None = None
    llm_provider: str | None = None
    embedding_provider: str | None = None
    top_k: int | None = None
    rerank_top_k: int | None = None
    search_type: str = "hybrid"
    temperature: float | None = None
    save_results: bool = True
    output_dir: str = "data/eval/results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvalRunResult:
    """Result of a complete evaluation run."""

    run_id: str
    config: EvalConfig
    dataset_name: str
    total_questions: int
    completed: int
    failed: int
    aggregate_metrics: dict[str, float | None]
    per_question_results: list[dict[str, Any]]
    started_at: str
    completed_at: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["config"] = self.config.to_dict()
        return d


class EvaluationRunner:
    """Runs evaluation on a golden dataset using RAG pipeline."""

    def __init__(self, config: EvalConfig | None = None) -> None:
        self.config = config or EvalConfig()
        self.retriever = Retriever()
        self.generator = RAGGenerator()
        self.ragas_evaluator = RAGASEvaluator(
            llm_provider=self.config.llm_provider,
            embedding_provider=self.config.embedding_provider,
        )
        self.custom_evaluator = CustomEvaluator(llm_provider=self.config.llm_provider)

    def _retrieve_and_generate(
        self, question: str
    ) -> tuple[str, list[Citation], list[str], list[str]]:
        """Run retrieval + generation for a single question.

        Returns:
            Tuple of (answer, citations, retrieved_contents, retrieved_ids)
        """
        from src.retrieval.models import SearchType

        search_type = SearchType(self.config.search_type)
        top_k = self.config.top_k or settings.top_k

        results = self.retriever.search(
            question,
            search_type=search_type,
            top_k=top_k,
        )

        retrieved_contents = [r.content for r in results]
        retrieved_ids = [r.id for r in results]

        gen_result = self.generator.generate(
            question=question,
            results=results,
            provider=self.config.llm_provider,
            temperature=self.config.temperature,
        )

        return gen_result.answer, gen_result.citations, retrieved_contents, retrieved_ids

    def run(self, dataset: GoldenDataset | None = None) -> EvalRunResult:
        """Run evaluation on the dataset."""
        run_id = str(uuid4())
        started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        if dataset is None:
            dataset = load_golden_dataset(self.config.dataset_path)

        log.info(
            "eval_run_start",
            run_id=run_id,
            dataset=dataset.name,
            num_questions=len(dataset),
            config=self.config.to_dict(),
        )

        metrics.increment("eval.runs.started")

        all_questions = []
        all_answers = []
        all_contexts = []
        all_ground_truths = []

        per_question_results = []
        completed = 0
        failed = 0

        for i, qa in enumerate(dataset):
            with trace_span(
                "eval_question",
                run_id=run_id,
                question_index=i,
                question=qa.question[:80],
            ):
                try:
                    # Retrieve + Generate
                    with metrics.timer("eval.question.latency"):
                        (
                            answer,
                            citations,
                            retrieved_contents,
                            retrieved_ids,
                        ) = self._retrieve_and_generate(qa.question)

                    # Custom evaluations
                    custom_results = self.custom_evaluator.evaluate(
                        answer=answer,
                        citations=citations,
                        retrieved_contents=retrieved_contents,
                    )

                    # Collect for RAGAS batch evaluation
                    all_questions.append(qa.question)
                    all_answers.append(answer)
                    all_contexts.append(retrieved_contents)
                    all_ground_truths.append(qa.ground_truth)

                    per_question_results.append({
                        "question_index": i,
                        "question": qa.question,
                        "ground_truth": qa.ground_truth,
                        "generated_answer": answer,
                        "retrieved_ids": retrieved_ids,
                        "custom_metrics": custom_results,
                    })

                    completed += 1
                    metrics.increment("eval.questions.completed")

                except Exception as e:
                    log.error(
                        "eval_question_failed",
                        run_id=run_id,
                        question_index=i,
                        error=str(e),
                    )
                    failed += 1
                    metrics.increment("eval.questions.failed")
                    per_question_results.append({
                        "question_index": i,
                        "question": qa.question,
                        "error": str(e),
                    })

        # Run RAGAS evaluation on all collected results
        ragas_result: RAGASEvalResult | None = None
        if all_questions:
            with trace_span("ragas_evaluation", run_id=run_id, num_samples=len(all_questions)):
                ragas_result = self.ragas_evaluator.evaluate(
                    questions=all_questions,
                    answers=all_answers,
                    contexts=all_contexts,
                    ground_truths=all_ground_truths,
                )

        aggregate = {}
        if ragas_result:
            aggregate.update(ragas_result.to_dict())

        completed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        result = EvalRunResult(
            run_id=run_id,
            config=self.config,
            dataset_name=dataset.name,
            total_questions=len(dataset),
            completed=completed,
            failed=failed,
            aggregate_metrics=aggregate,
            per_question_results=per_question_results,
            started_at=started_at,
            completed_at=completed_at,
        )

        log.info(
            "eval_run_complete",
            run_id=run_id,
            completed=completed,
            failed=failed,
            aggregate=aggregate,
        )

        metrics.increment("eval.runs.completed")

        if self.config.save_results:
            self._save_results(result)

        return result

    def _save_results(self, result: EvalRunResult) -> None:
        """Save evaluation results to JSON file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = f"eval_{result.run_id[:8]}_{result.started_at.replace(':', '-')}.json"
        filepath = output_dir / filename

        with filepath.open("w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

        log.info("eval_results_saved", path=str(filepath))


def run_evaluation(
    config: EvalConfig | None = None,
    dataset: GoldenDataset | None = None,
) -> EvalRunResult:
    """Convenience function to run evaluation."""
    runner = EvaluationRunner(config)
    return runner.run(dataset)
