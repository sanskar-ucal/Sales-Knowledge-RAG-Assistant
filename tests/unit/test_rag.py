from langchain_core.messages import AIMessage

from app.cache import AsyncTTLCache
from app.llm import LLMProvider
from app.prompts import NO_ANSWER
from app.rag import RAGService
from tests.conftest import ScriptedChatModel


async def test_retrieval_finds_relevant_doc(store, settings):
    rag = RAGService(store, LLMProvider(settings), settings)
    results = await rag.retrieve("What discount can an account executive approve?")
    assert results[0][0].metadata["source"].endswith("pricing-negotiation.md")


async def test_extractive_fallback_without_llm(store, settings):
    rag = RAGService(store, LLMProvider(settings), settings)
    resp = await rag.answer("What does MEDDICC stand for?")
    assert resp.mode == "extractive"
    assert resp.sources and "[1]" in resp.answer


async def test_llm_answer_uses_context_and_is_cached(store, settings):
    model = ScriptedChatModel(responder=lambda msgs: AIMessage("Use 3x to 4x coverage [1]."))
    rag = RAGService(store, LLMProvider(settings, primary=model), settings, AsyncTTLCache())
    first = await rag.answer("What is a healthy pipeline coverage ratio?")
    assert first.mode == "llm" and not first.cached
    system_prompt = model.calls[0][0].content
    assert "Pipeline coverage" in system_prompt and "[1]" in system_prompt

    second = await rag.answer("what is a healthy pipeline coverage ratio?")
    assert second.cached and second.answer == first.answer
    assert len(model.calls) == 1


async def test_fallback_model_used_when_primary_fails(store, settings):
    primary = ScriptedChatModel(responder=lambda m: AIMessage("x"), fail=True)
    backup = ScriptedChatModel(responder=lambda m: AIMessage("Backup answer [1]."))
    rag = RAGService(store, LLMProvider(settings, primary=primary, fallback=backup), settings)
    resp = await rag.answer("How long should a cold email be?")
    assert resp.mode == "llm" and resp.answer == "Backup answer [1]."


async def test_extractive_when_all_models_fail(store, settings):
    primary = ScriptedChatModel(responder=lambda m: AIMessage("x"), fail=True)
    rag = RAGService(store, LLMProvider(settings, primary=primary), settings, AsyncTTLCache())
    resp = await rag.answer("How long should a cold email be?")
    assert resp.mode == "extractive"
    assert rag.cache.stats()["size"] == 0, "degraded answers must not be cached"


async def test_no_results_returns_abstention(store, settings):
    settings.min_relevance = 0.99
    rag = RAGService(store, LLMProvider(settings), settings)
    resp = await rag.answer("Quantum chromodynamics lattice gauge")
    assert resp.answer == NO_ANSWER and resp.sources == []
