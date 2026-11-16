# Vector Databases for RAG

Vector databases store dense embeddings and support fast approximate nearest
neighbor search. Popular options include Qdrant, Pinecone, Milvus, and Chroma.

## Qdrant

Qdrant is an open-source vector search engine written in Rust. It supports:

- **Dense vectors**: cosine, dot product, and euclidean distance
- **Sparse vectors**: BM25-style keyword search for hybrid retrieval
- **Payload filtering**: filter results by metadata (tenant_id, source, date)
- **Quantization**: int8 scalar quantization reduces memory usage ~4x

## Choosing a Vector DB

For local development, Chroma or Qdrant in-memory work well. For production,
Qdrant Cloud or a self-hosted Qdrant cluster offers better scalability and
multi-tenancy through payload-based isolation.
