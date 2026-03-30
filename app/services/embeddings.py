from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime when package missing
    OpenAI = None

TOKEN_PATTERN = re.compile(r'[\u4e00-\u9fff]+|[A-Za-z0-9_]+')


class BaseEmbedder:
    dimension: int | None = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - interface only
        raise NotImplementedError


@dataclass
class HashEmbedder(BaseEmbedder):
    dimension: int = 256

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = TOKEN_PATTERN.findall((text or '').lower())
            if not tokens:
                tokens = ['<empty>']
            for token in tokens:
                digest = hashlib.sha256(token.encode('utf-8')).digest()
                idx = int.from_bytes(digest[:4], 'little') % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row_idx, idx] += sign
        return matrix


class OpenAICompatibleEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 128,
    ) -> None:
        if OpenAI is None:
            raise ImportError('openai 包未安装，无法使用 OpenAI embedding。')
        api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('未找到 OPENAI_API_KEY，无法使用 OpenAI embedding。')

        client_kwargs: dict[str, str] = {'api_key': api_key}
        if base_url:
            client_kwargs['base_url'] = base_url

        self.client = OpenAI(**client_kwargs)
        self.model = model
        self.batch_size = batch_size
        self.dimension = None

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            response = self.client.embeddings.create(model=self.model, input=batch)
            vectors.extend(item.embedding for item in response.data)
        matrix = np.asarray(vectors, dtype=np.float32)
        self.dimension = int(matrix.shape[1]) if matrix.size else None
        return matrix


def build_embedder(
    provider: str,
    *,
    openai_model: str,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    hash_dimension: int = 256,
) -> BaseEmbedder:
    provider = provider.lower().strip()
    if provider == 'hash':
        return HashEmbedder(dimension=hash_dimension)
    if provider == 'openai':
        return OpenAICompatibleEmbedder(
            model=openai_model,
            api_key=openai_api_key,
            base_url=openai_base_url,
        )
    raise ValueError(f'不支持的 embedding provider: {provider}')
