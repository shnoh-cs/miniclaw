"""Embedding provider: calls vLLM /v1/embeddings endpoint."""

from __future__ import annotations

import hashlib

import numpy as np
from openai import AsyncOpenAI

from openclaw.config import AppConfig


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalize an embedding vector.

    Sanitizes non-finite values and normalizes to unit length so that
    cosine similarity works correctly (dot product == cosine for unit vectors).
    Matches the sanitizeAndNormalizeEmbedding pattern from the original OpenClaw.
    """
    # Replace any non-finite values with 0
    sanitized = np.where(np.isfinite(vec), vec, 0.0).astype(np.float32)
    magnitude = np.linalg.norm(sanitized)
    if magnitude < 1e-10:
        return sanitized
    return sanitized / magnitude


class EmbeddingProvider:
    """Generates embeddings via OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, config: AppConfig) -> None:
        self.model = config.models.embedding
        self.client = AsyncOpenAI(
            base_url=config.endpoints.embedding.base_url,
            api_key=config.endpoints.embedding.api_key,
        )

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts. Returns list of L2-normalized embedding vectors."""
        if not texts:
            return []

        response = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )

        return [
            _l2_normalize(np.array(item.embedding, dtype=np.float32))
            for item in sorted(response.data, key=lambda x: x.index)
        ]

    async def embed_single(self, text: str) -> np.ndarray:
        """Embed a single text. Returns an L2-normalized embedding vector."""
        results = await self.embed([text])
        return results[0]

    @staticmethod
    def text_hash(text: str) -> str:
        """Generate a hash for embedding cache lookup."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]
