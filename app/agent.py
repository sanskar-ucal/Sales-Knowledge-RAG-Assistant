import logging
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool

from app.cache import make_key
from app.prompts import AGENT_SYSTEM_PROMPT, NO_ANSWER
from app.rag import RAGService, to_source
from app.schemas import AnswerResponse, Source

logger = logging.getLogger(__name__)


@dataclass
class _ToolSession:
    """Per-request state so concurrent agent runs never share citations."""

    rag: RAGService
    sources: dict[str, Source] = field(default_factory=dict)
    seen_ids: dict[str, str] = field(default_factory=dict)

    async def search_sales_docs(self, query: str, k: int = 4) -> str:
        """Search the sales documentation knowledge base. Returns passages with citation ids."""
        results = await self.rag.retrieve(query, min(max(k, 1), 8))
        if not results:
            return "No relevant passages found."
        blocks = []
        for doc, score in results:
            doc_id = str(doc.metadata.get("id"))
            label = self.seen_ids.get(doc_id)
            if label is None:
                label = f"S{len(self.seen_ids) + 1}"
                self.seen_ids[doc_id] = label
                self.sources[label] = to_source(doc, score)
            where = f"{doc.metadata.get('title')} > {doc.metadata.get('section')}"
            blocks.append(f"[{label}] ({where})\n{doc.page_content}")
        return "\n\n".join(blocks)

    async def list_sales_topics(self) -> str:
        """List the documents (topics) available in the sales knowledge base."""
        results = await self.rag.retrieve("sales process overview", k=50)
        titles = sorted({str(d.metadata.get("title")) for d, _ in results})
        return "\n".join(f"- {t}" for t in titles) or "No documents indexed."

    def tools(self) -> list[BaseTool]:
        return [
            StructuredTool.from_function(
                coroutine=self.search_sales_docs, name="search_sales_docs"
            ),
            StructuredTool.from_function(
                coroutine=self.list_sales_topics, name="list_sales_topics"
            ),
        ]


class SalesAgent:
    """Tool-calling agent: the LLM decides when/how to query the retriever."""

    def __init__(self, rag: RAGService, max_steps: int = 4):
        self.rag = rag
        self.max_steps = max_steps

    async def run(self, question: str, use_cache: bool = True) -> AnswerResponse:
        if not self.rag.llm.available:
            response = await self.rag.answer(question, use_cache=use_cache)
            return response

        start = time.perf_counter()
        cache = self.rag.cache
        key = make_key("agent", question.strip().lower())
        if use_cache and cache is not None:
            cached = await cache.get(key)
            if cached is not None:
                return cached.model_copy(update={"cached": True})

        session = _ToolSession(self.rag)
        tools = {t.name: t for t in session.tools()}
        try:
            llm = self.rag.llm.chat_with_tools(list(tools.values()))
            answer, tool_log = await self._loop(llm, tools, question)
        except Exception:
            logger.exception("Agent failed, falling back to single-shot RAG")
            return await self.rag.answer(question, use_cache=use_cache)

        response = AnswerResponse(
            question=question,
            answer=answer,
            sources=list(session.sources.values()),
            mode="agent",
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            tool_calls=tool_log,
        )
        if use_cache and cache is not None:
            await cache.set(key, response)
        logger.info(
            "agent answer",
            extra={"tool_calls": len(tool_log), "latency_ms": response.latency_ms},
        )
        return response

    async def _loop(self, llm, tools: dict[str, BaseTool], question: str):
        messages = [SystemMessage(AGENT_SYSTEM_PROMPT), HumanMessage(question)]
        tool_log: list[dict] = []
        for _ in range(self.max_steps):
            ai: AIMessage = await llm.ainvoke(messages)
            messages.append(ai)
            if not ai.tool_calls:
                return _text(ai) or NO_ANSWER, tool_log
            for call in ai.tool_calls:
                tool = tools.get(call["name"])
                if tool is None:
                    output = f"Unknown tool: {call['name']}"
                else:
                    output = await tool.ainvoke(call["args"])
                tool_log.append({"tool": call["name"], "args": call["args"]})
                messages.append(ToolMessage(content=str(output), tool_call_id=call["id"]))

        messages.append(HumanMessage("Step limit reached. Give your best final answer now."))
        final: AIMessage = await llm.ainvoke(messages)
        return _text(final) or NO_ANSWER, tool_log


def _text(msg: AIMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content).strip()
