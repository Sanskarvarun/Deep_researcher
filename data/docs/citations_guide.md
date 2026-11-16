# Citations in RAG

Citations let users verify where an answer came from. In this system, every
retrieved chunk is assigned a citation key (typically its document ID), and the
LLM is instructed to cite sources inline using markers like `[doc-a]`.

## Citation Format

- **Inline marker**: `[doc-a]` appears in the answer text at the relevant claim
- **Reference list**: A "Sources" section lists each citation with its source
  file, section header, and page number (when available)

## Why Citations Matter

Citations make RAG auditable. Users can click through to the source chunk and
confirm the answer is grounded in real evidence rather than hallucinated.
