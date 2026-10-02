import os
import uuid

import pytest

from app.config import Settings
from app.embeddings import HashingEmbeddings
from app.vectorstore import build_and_index, search_with_scores

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set"),
]


async def test_pgvector_index_and_search(docs):
    settings = Settings(
        _env_file=None,
        vector_backend="pgvector",
        database_url=os.environ["DATABASE_URL"],
        collection_name=f"test_{uuid.uuid4().hex[:8]}",
        openai_api_key=None,
    )
    store, backend = await build_and_index(settings, HashingEmbeddings(), docs)
    assert backend == "pgvector"

    results = await search_with_scores(store, "What does the LAER method stand for?", 3)
    assert results[0][0].metadata["source"].endswith("objection-handling.md")
    assert all(0.0 <= score <= 1.0 for _, score in results)

    # Re-indexing with the same deterministic ids must upsert, not duplicate.
    await build_and_index(settings, HashingEmbeddings(), docs)
    results = await search_with_scores(store, "What does the LAER method stand for?", 3)
    assert len({d.metadata["id"] for d, _ in results}) == len(results)
    await store.adelete_collection()


async def test_fallback_to_memory_when_db_unreachable(docs):
    settings = Settings(
        _env_file=None,
        vector_backend="pgvector",
        database_url="postgresql+psycopg://bad:bad@127.0.0.1:1/none",
        openai_api_key=None,
    )
    _, backend = await build_and_index(settings, HashingEmbeddings(), docs)
    assert backend == "memory"
