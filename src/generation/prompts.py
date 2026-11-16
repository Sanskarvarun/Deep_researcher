"""Prompt templates for RAG generation.

Provides system, user, and few-shot prompt templates plus a builder that
assembles a complete RAG prompt from retrieved context and a user question.
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from src.core.exceptions import ValidationError


class PromptType(StrEnum):
    SYSTEM = "system"
    USER = "user"
    FEW_SHOT = "few_shot"


class PromptTemplate(BaseModel):
    """A reusable prompt template with named variables."""

    name: str
    template: str
    input_variables: list[str] = Field(default_factory=list)
    prompt_type: PromptType = PromptType.USER

    def format(self, **kwargs: Any) -> str:
        """Format the template with the given variables.

        Raises:
            ValidationError: If required variables are missing or unknown.
        """
        missing = set(self.input_variables) - set(kwargs)
        if missing:
            raise ValidationError(
                f"Missing template variables: {sorted(missing)}",
                details={"template": self.name, "missing": sorted(missing)},
            )
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValidationError(
                f"Template {self.name!r} references unknown variable: {e}",
                details={"template": self.name, "variable": str(e)},
            ) from e


# ---------------------------------------------------------------------------
# Built-in RAG prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a precise, helpful research assistant.

Answer the user's question using ONLY the provided context. Follow these rules:
- If the answer is supported by the context, state it concisely and cite sources
  using inline markers like [doc_id] (e.g. [doc-a], [doc-b]).
- If the context does not contain enough information, say so clearly rather than
  guessing.
- Do not fabricate citations or sources.
- Be accurate, neutral, and concise."""

DEFAULT_USER_TEMPLATE = """Question: {question}

Context (retrieved chunks):
{context}

Answer:"""

DEFAULT_FEW_SHOT: list[dict[str, str]] = [
    {
        "question": "What is RAG?",
        "context": (
            "[doc-a] RAG combines retrieval with generation. "
            "[doc-b] It grounds LLMs in external knowledge."
        ),
        "answer": (
            "RAG (Retrieval-Augmented Generation) combines retrieval with "
            "generation to ground LLMs in external knowledge [doc-a][doc-b]."
        ),
    },
]


class RAGPromptBuilder:
    """Assembles system + few-shot + user prompts for RAG generation."""

    def __init__(
        self,
        system_template: PromptTemplate | None = None,
        user_template: PromptTemplate | None = None,
        few_shots: list[dict[str, str]] | None = None,
    ) -> None:
        self.system_template = system_template or PromptTemplate(
            name="rag_system",
            template=DEFAULT_SYSTEM_PROMPT,
            input_variables=[],
            prompt_type=PromptType.SYSTEM,
        )
        self.user_template = user_template or PromptTemplate(
            name="rag_user",
            template=DEFAULT_USER_TEMPLATE,
            input_variables=["question", "context"],
            prompt_type=PromptType.USER,
        )
        self.few_shots = few_shots or []

    def build_context(self, results: list) -> str:
        """Format retrieval results into a context block with doc_id labels."""
        lines: list[str] = []
        for result in results:
            doc_id = result.document_id or result.id
            label = f"[{doc_id}]"
            content = result.content.strip()
            lines.append(f"{label} {content}")
        return "\n".join(lines)

    def build_messages(self, question: str, results: list) -> list[dict[str, str]]:
        """Build the message list for the chat model.

        Returns a list of ``{"role", "content"}`` dicts suitable for most chat
        providers (Gemini, OpenAI-compatible, Groq).
        """
        context = self.build_context(results)
        user_content = self.user_template.format(question=question, context=context)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_template.format()},
        ]

        # Few-shot examples as alternating user/assistant turns.
        for example in self.few_shots:
            messages.append({"role": "user", "content": example["question"]})
            messages.append({"role": "assistant", "content": example["answer"]})

        messages.append({"role": "user", "content": user_content})
        return messages
