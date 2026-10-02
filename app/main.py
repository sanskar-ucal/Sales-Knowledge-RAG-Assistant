import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request

from app import __version__
from app.agent import SalesAgent
from app.cache import AsyncTTLCache
from app.config import Settings, get_settings
from app.embeddings import build_embeddings
from app.ingest import load_markdown_documents
from app.llm import LLMProvider
from app.logging_config import configure_logging, request_id_var
from app.rag import RAGService, to_source
from app.schemas import (
    AnswerResponse,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    SearchResponse,
)
from app.vectorstore import build_and_index, index_documents

logger = logging.getLogger(__name__)


def _load_docs(settings: Settings):
    return load_markdown_documents(
        settings.docs_dir,
        settings.source_base_url,
        settings.chunk_size,
        settings.chunk_overlap,
    )


def create_app(settings: Settings | None = None, llm: LLMProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        embeddings = build_embeddings(settings)
        docs = _load_docs(settings) if settings.auto_ingest else None
        store, backend = await build_and_index(settings, embeddings, docs)
        cache = AsyncTTLCache(settings.cache_max_items, settings.cache_ttl_seconds)
        provider = llm or LLMProvider(settings)
        rag = RAGService(store, provider, settings, cache)

        app.state.settings = settings
        app.state.store = store
        app.state.backend = backend
        app.state.cache = cache
        app.state.rag = rag
        app.state.agent = SalesAgent(rag, settings.agent_max_steps)
        app.state.chunks_indexed = len(docs or [])
        logger.info(
            "startup complete",
            extra={"backend": backend, "llm": provider.available, "chunks": len(docs or [])},
        )
        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description="RAG assistant over sales documentation with source-linked answers.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = request_id_var.set(rid)
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = rid
            logger.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "latency_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            return response
        finally:
            request_id_var.reset(token)

    @app.get("/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        state = request.app.state
        return HealthResponse(
            status="ok",
            vector_backend=state.backend,
            llm_available=state.rag.llm.available,
            chunks_indexed=state.chunks_indexed,
        )

    @app.post("/query", response_model=AnswerResponse)
    async def query(body: QueryRequest, request: Request) -> AnswerResponse:
        """Single-shot RAG: retrieve top-k passages, then generate a cited answer."""
        return await request.app.state.rag.answer(body.question, body.top_k, body.use_cache)

    @app.post("/agent", response_model=AnswerResponse)
    async def agent(body: QueryRequest, request: Request) -> AnswerResponse:
        """Tool-calling agent that decides how to query the knowledge base."""
        return await request.app.state.agent.run(body.question, body.use_cache)

    @app.get("/search", response_model=SearchResponse)
    async def search(
        request: Request,
        q: str = Query(..., min_length=2),
        k: int = Query(4, ge=1, le=20),
    ) -> SearchResponse:
        results = await request.app.state.rag.retrieve(q, k)
        return SearchResponse(query=q, results=[to_source(d, s) for d, s in results])

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: Request) -> IngestResponse:
        state = request.app.state
        try:
            docs = _load_docs(state.settings)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        count = await index_documents(state.store, docs)
        state.chunks_indexed = count
        await state.cache.clear()
        return IngestResponse(chunks_indexed=count, backend=state.backend)

    @app.get("/cache/stats")
    async def cache_stats(request: Request) -> dict:
        return request.app.state.cache.stats()

    return app


app = create_app()
