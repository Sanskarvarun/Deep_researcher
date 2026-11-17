"""Evaluation module - RAGAS metrics, custom evals, golden datasets."""

import src.evaluation._ragas_patch
from src.evaluation.custom_eval import (
    CitationAccuracyEvaluator,
    CustomEvaluator,
    HallucinationRateEvaluator,
)
from src.evaluation.golden_dataset import GoldenDataset, QAPair, load_golden_dataset
from src.evaluation.ragas_eval import RAGASEvaluator, evaluate_rag
from src.evaluation.runner import EvalConfig, EvaluationRunner, run_evaluation

__all__ = [
    "CitationAccuracyEvaluator",
    "CustomEvaluator",
    "EvalConfig",
    "EvaluationRunner",
    "GoldenDataset",
    "HallucinationRateEvaluator",
    "QAPair",
    "RAGASEvaluator",
    "evaluate_rag",
    "load_golden_dataset",
    "run_evaluation",
    "src",
]
