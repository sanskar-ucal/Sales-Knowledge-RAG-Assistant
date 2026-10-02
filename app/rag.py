import logging
import re
import time

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore

from app.cache import AsyncTTLCache, make_key
from app.config import Settings
from app.llm import LLMProvider
from app.prompts import NO_ANSWER, RAG_PROMPT, format_context
from app.schemas import AnswerResponse, Source
from app.vectorstore import search_with_scores

logger = logging.getLogger(__name__)


def to_source(doc: Document, score: float, snippet_len: int = 280) -> Source:
    text = re.sub(r"\s+", " ", doc.page_content).strip()
    return Source(
        id=str(doc.metadata.get("id", "")),
        title=str(doc.metadata.get("title", "")),
        section=str(doc.metadata.get("section", "")),
        url=str(doc.metadata.get("url", "")),
        score=round(float(score), 4),
        snippet=text[:snippet_len] + ("..." if len(text) > snippet_len else ""),
    )


def extractive_answer(results: list[tuple[Document, float]], max_passages: int = 2) -> str:
    """Retrieval-only fallback: quote the top passages with citations."""
    if not results:
        return NO_ANSWER
    lines = ["LLM unavailable - most relevant documentation excerpts:"]
    for i, (doc, _) in enumerate(results[:max_passages], start=1):
        text = re.sub(r"\s+", " ", doc.page_content).strip()
        lines.append(f"- {text[:500]} [{i}]")
    return "\n".join(lines)


class RAGService:
    def __init__(
        self,
        store: VectorStore,
        llm: LLMProvider,
        settings: Settings,
        cache: AsyncTTLCache | None = None,
    ):
        self.store = store
        self.llm = llm
        self.settings = settings
        self.cache = cache

    async def retrieve(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        k = k or self.settings.top_k
        results = await search_with_scores(self.store, query, k)
        return [(d, s) for d, s in results if s >= self.settings.min_relevance]

    async def answer(
        self, question: str, k: int | None = None, use_cache: bool = True
    ) -> AnswerResponse:
        start = time.perf_counter()
        k = k or self.settings.top_k
        key = make_key("rag", question.strip().lower(), k)
        if use_cache and self.cache is not None:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached.model_copy(update={"cached": True, "latency_ms": _ms(start)})

        results = await self.retrieve(question, k)
        sources = [to_source(d, s) for d, s in results]
        mode = "llm"

        if not results:
            answer = NO_ANSWER
        elif self.llm.available:
            try:
                answer = await self._generate(question, results)
            except Exception:
                logger.exception("LLM generation failed, using extractive fallback")
                answer, mode = extractive_answer(results), "extractive"
        else:
            answer, mode = extractive_answer(results), "extractive"

        response = AnswerResponse(
            question=question,
            answer=answer,
            sources=sources,
            mode=mode,
            latency_ms=_ms(start),
        )
        if use_cache and self.cache is not None and mode == "llm":
            await self.cache.set(key, response)
        logger.info(
            "rag answer",
            extra={"mode": mode, "n_sources": len(sources), "latency_ms": response.latency_ms},
        )
        return response

    async def _generate(self, question: str, results: list[tuple[Document, float]]) -> str:
        context = format_context(
            [
                (str(i), f"{d.metadata.get('title')} > {d.metadata.get('section')}", d.page_content)
                for i, (d, _) in enumerate(results, start=1)
            ]
        )
        chain = RAG_PROMPT | self.llm.chat() | StrOutputParser()
        return (await chain.ainvoke({"context": context, "question": question})).strip()


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
