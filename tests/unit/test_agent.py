from langchain_core.messages import AIMessage, ToolMessage

from app.agent import SalesAgent
from app.llm import LLMProvider
from app.rag import RAGService
from tests.conftest import ScriptedChatModel


def tool_then_answer(messages):
    if not any(isinstance(m, ToolMessage) for m in messages):
        return AIMessage(
            "",
            tool_calls=[
                {"name": "search_sales_docs", "args": {"query": "LAER method"}, "id": "c1"}
            ],
        )
    tool_output = next(m for m in messages if isinstance(m, ToolMessage)).content
    assert "[S1]" in tool_output
    return AIMessage("Listen, Acknowledge, Explore, Respond [S1].")


async def test_agent_calls_retrieval_tool_and_cites(store, settings):
    model = ScriptedChatModel(responder=tool_then_answer)
    agent = SalesAgent(RAGService(store, LLMProvider(settings, primary=model), settings))
    resp = await agent.run("What is LAER?")
    assert resp.mode == "agent"
    assert resp.tool_calls == [{"tool": "search_sales_docs", "args": {"query": "LAER method"}}]
    assert resp.sources and resp.sources[0].title == "Objection Handling Guide"
    assert "[S1]" in resp.answer


async def test_agent_step_limit_forces_final_answer(store, settings):
    def always_tool(messages):
        if "Step limit reached" in str(messages[-1].content):
            return AIMessage("Final.")
        return AIMessage(
            "", tool_calls=[{"name": "search_sales_docs", "args": {"query": "x"}, "id": "c"}]
        )

    model = ScriptedChatModel(responder=always_tool)
    agent = SalesAgent(
        RAGService(store, LLMProvider(settings, primary=model), settings), max_steps=2
    )
    resp = await agent.run("loop forever")
    assert resp.answer == "Final." and len(resp.tool_calls) == 2


async def test_agent_falls_back_to_rag_on_failure(store, settings):
    model = ScriptedChatModel(responder=lambda m: AIMessage("x"), fail=True)
    agent = SalesAgent(RAGService(store, LLMProvider(settings, primary=model), settings))
    resp = await agent.run("What is LAER?")
    assert resp.mode == "extractive"


async def test_agent_without_llm_uses_rag(store, settings):
    agent = SalesAgent(RAGService(store, LLMProvider(settings), settings))
    resp = await agent.run("What is LAER?")
    assert resp.mode == "extractive"
