import logging

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore

from app.config import Settings

logger = logging.getLogger(__name__)


def build_pgvector(settings: Settings, embeddings: Embeddings) -> VectorStore:
    from langchain_postgres import PGVector

    return PGVector(
        embeddings=embeddings,
        connection=settings.database_url,
        collection_name=settings.collection_name,
        use_jsonb=True,
        async_mode=True,
    )


def build_memory_store(embeddings: Embeddings) -> VectorStore:
    return InMemoryVectorStore(embeddings)


async def index_documents(store: VectorStore, docs: list[Document]) -> int:
    if not docs:
        return 0
    ids = [d.metadata["id"] for d in docs]
    await store.aadd_documents(docs, ids=ids)
    return len(docs)


async def build_and_index(
    settings: Settings, embeddings: Embeddings, docs: list[Document] | None
) -> tuple[VectorStore, str]:
    """Create the configured store and index docs, falling back to in-memory on failure."""
    if settings.vector_backend == "pgvector" and settings.database_url:
        try:
            store = build_pgvector(settings, embeddings)
            if docs:
                await index_documents(store, docs)
            else:
                await store.asimilarity_search("healthcheck", k=1)
            logger.info("pgvector store ready", extra={"collection": settings.collection_name})
            return store, "pgvector"
        except Exception:
            logger.exception("pgvector unavailable, falling back to in-memory vector store")

    store = build_memory_store(embeddings)
    if docs:
        await index_documents(store, docs)
    return store, "memory"


async def search_with_scores(
    store: VectorStore, query: str, k: int
) -> list[tuple[Document, float]]:
    """Return (doc, relevance) pairs where higher relevance is better, across backends."""
    if isinstance(store, InMemoryVectorStore):
        return await store.asimilarity_search_with_score(query, k=k)
    return await store.asimilarity_search_with_relevance_scores(query, k=k)
