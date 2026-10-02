import hashlib
import logging
import math
import re

from langchain_core.embeddings import Embeddings

from app.config import Settings

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have how i in is it its of on or that the this to "
    "was what when where which who why will with you your do does should can".split()
)


class HashingEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings (hashed unigrams + bigrams).

    Used for offline development, tests, and as a fallback when no embedding API is configured.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign

    def _embed(self, text: str) -> list[float]:
        tokens = [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        vec = [0.0] * self.dim
        for feat in features:
            idx, sign = self._bucket(feat)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def build_embeddings(settings: Settings) -> Embeddings:
    if settings.use_openai_embeddings:
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI embeddings", extra={"model": settings.embedding_model})
        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    logger.info("Using local hashing embeddings")
    return HashingEmbeddings()
