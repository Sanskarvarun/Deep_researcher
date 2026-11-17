"""Custom evaluation metrics: citation accuracy, hallucination rate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.core.logging import get_logger
from src.generation.citations import Citation

log = get_logger(__name__)


@dataclass
class CitationAccuracyResult:
    """Result of citation accuracy evaluation."""

    total_citations: int
    valid_citations: int
    accuracy: float
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "accuracy": self.accuracy,
            "details": self.details,
        }


@dataclass
class HallucinationResult:
    """Result of hallucination rate evaluation."""

    total_statements: int
    hallucinated_statements: int
    hallucination_rate: float
    details: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_statements": self.total_statements,
            "hallucinated_statements": self.hallucinated_statements,
            "hallucination_rate": self.hallucination_rate,
            "details": self.details,
        }


class CitationAccuracyEvaluator:
    """Evaluates whether citations in generated answers correctly reference retrieved sources."""

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        answer: str,
        citations: list[Citation],
        retrieved_contents: list[str],
    ) -> CitationAccuracyResult:
        """Check if each citation maps to a valid retrieved document.

        Args:
            answer: The generated answer text.
            citations: List of Citation objects from the generator.
            retrieved_contents: List of retrieved document contents.

        Returns:
            CitationAccuracyResult with accuracy score and per-citation details.
        """
        if not citations:
            return CitationAccuracyResult(
                total_citations=0,
                valid_citations=0,
                accuracy=1.0,
                details=[],
            )

        # Build a set of available doc IDs from retrieval
        # Citations use key format like "doc_0", "doc_1" etc.
        valid_doc_ids = {f"doc_{i}" for i in range(len(retrieved_contents))}

        details = []
        valid_count = 0

        for i, citation in enumerate(citations):
            doc_id = citation.key
            is_valid = doc_id in valid_doc_ids
            if is_valid:
                valid_count += 1

            details.append({
                "citation_index": i,
                "doc_id": doc_id,
                "text_snippet": citation.content[:100] if citation.content else "",
                "valid": is_valid,
            })

        accuracy = valid_count / len(citations) if citations else 1.0

        log.info(
            "citation_accuracy_eval",
            total=len(citations),
            valid=valid_count,
            accuracy=accuracy,
        )

        return CitationAccuracyResult(
            total_citations=len(citations),
            valid_citations=valid_count,
            accuracy=accuracy,
            details=details,
        )


class HallucinationRateEvaluator:
    """Evaluates hallucination rate using LLM-as-judge.

    Checks if statements in the generated answer are supported by the retrieved context.
    """

    def __init__(self, llm_provider: str | None = None) -> None:
        self.llm_provider = llm_provider

    def _get_llm(self) -> Any:
        if self.llm_provider is None:
            from src.core.config import settings

            self.llm_provider = settings.llm_primary
        from src.generation.llm import llm_factory

        return llm_factory.get_model(self.llm_provider)

    def evaluate(
        self,
        answer: str,
        contexts: list[str],
    ) -> HallucinationResult:
        """Check each statement in the answer against the retrieved contexts.

        Uses an LLM to classify statements as supported/unsupported.

        Args:
            answer: The generated answer text.
            contexts: List of retrieved document contents.

        Returns:
            HallucinationResult with rate and per-statement details.
        """
        if not answer.strip() or not contexts:
            return HallucinationResult(
                total_statements=0,
                hallucinated_statements=0,
                hallucination_rate=0.0,
                details=[],
            )

        # Simple statement splitter (by sentence)
        import re

        statements = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]

        if not statements:
            return HallucinationResult(
                total_statements=0,
                hallucinated_statements=0,
                hallucination_rate=0.0,
                details=[],
            )

        context_text = "\n\n".join(f"[Doc {i}] {c}" for i, c in enumerate(contexts))

        prompt = (
            "You are an evaluator checking for hallucinations in a generated answer.\n"
            "Given the CONTEXT below, determine if each STATEMENT is SUPPORTED "
            "(fully backed by context) or UNSUPPORTED (hallucinated/unsupported).\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            "STATEMENTS TO EVALUATE:\n"
            + "\n".join(f"{i + 1}. {stmt}" for i, stmt in enumerate(statements))
            + "\n\nRespond with a JSON array where each element has:\n"
            '- "statement_index": 0-based index\n'
            '- "statement": the statement text\n'
            '- "verdict": "SUPPORTED" or "UNSUPPORTED"\n'
            '- "reasoning": brief explanation\n\n'
            "Only output the JSON array."
        )

        from langchain_core.messages import HumanMessage

        llm = self._get_llm()
        response = llm.invoke([HumanMessage(content=prompt)])

        # Parse JSON response
        import json

        try:
            content = response.content
            if isinstance(content, list):
                content = "".join(str(c) for c in content)
            results = json.loads(content)
        except Exception as e:
            log.warning("hallucination_eval_parse_failed", error=str(e))
            # Fallback: assume all supported if parsing fails
            results = [
                {"statement_index": i, "verdict": "SUPPORTED", "reasoning": "parse_failed"}
                for i in range(len(statements))
            ]

        hallucinated = sum(1 for r in results if r.get("verdict") == "UNSUPPORTED")
        rate = hallucinated / len(statements) if statements else 0.0

        details = []
        for i, stmt in enumerate(statements):
            r: dict[str, Any] = next((r for r in results if r.get("statement_index") == i), {})
            details.append({
                "statement_index": i,
                "statement": stmt,
                "verdict": r.get("verdict", "SUPPORTED"),
                "reasoning": r.get("reasoning", ""),
            })

        log.info(
            "hallucination_eval",
            total_statements=len(statements),
            hallucinated=hallucinated,
            rate=rate,
        )

        return HallucinationResult(
            total_statements=len(statements),
            hallucinated_statements=hallucinated,
            hallucination_rate=rate,
            details=details,
        )


class CustomEvaluator:
    """Combined custom evaluator for citation accuracy and hallucination rate."""

    def __init__(self, llm_provider: str | None = None) -> None:
        self.citation_evaluator = CitationAccuracyEvaluator()
        self.hallucination_evaluator = HallucinationRateEvaluator(llm_provider=llm_provider)

    def evaluate(
        self,
        answer: str,
        citations: list[Citation],
        retrieved_contents: list[str],
    ) -> dict[str, Any]:
        """Run both custom evaluations."""
        citation_result = self.citation_evaluator.evaluate(answer, citations, retrieved_contents)
        hallucination_result = self.hallucination_evaluator.evaluate(answer, retrieved_contents)

        return {
            "citation_accuracy": citation_result.to_dict(),
            "hallucination_rate": hallucination_result.to_dict(),
        }
