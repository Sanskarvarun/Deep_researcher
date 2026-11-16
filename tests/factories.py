"""Fake embedder for tests — deterministic, no network."""


class FakeEmbedder:
    """Deterministic embedder: hashes text into a fixed-dim vector.

    Similar texts produce similar vectors (bag-of-chars) so semantic
    chunking tests can rely on coherent similarity behavior.
    """

    def __init__(self, dim: int = 8, model: str = "fake-embedder-v1") -> None:
        self.model_name = model
        self.dim = dim
        self.call_count = 0
        self.call_args: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.call_count += 1
        self.call_args.append(texts)
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        # Bag-of-characters hashed into dim buckets, normalized
        vec = [0.0] * self.dim
        for ch in text.lower():
            bucket = ord(ch) % self.dim
            vec[bucket] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
