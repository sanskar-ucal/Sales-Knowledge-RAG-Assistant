from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from app.config import Settings
from app.embeddings import HashingEmbeddings
from app.ingest import load_markdown_documents
from app.vectorstore import build_memory_store, index_documents

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "data" / "docs"


class ScriptedChatModel(BaseChatModel):
    """Fake chat model that returns scripted responses and supports bind_tools."""

    responder: Callable[[list[BaseMessage]], AIMessage]
    calls: list[list[BaseMessage]] = []
    fail: bool = False

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.fail:
            raise RuntimeError("simulated provider outage")
        self.calls.append(list(messages))
        return ChatResult(generations=[ChatGeneration(message=self.responder(messages))])


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        vector_backend="memory",
        database_url=None,
        openai_api_key=None,
        embedding_provider="hashing",
        docs_dir=str(DOCS_DIR.relative_to(ROOT)),
        log_json=False,
        log_level="WARNING",
    )


@pytest.fixture
def docs(settings, monkeypatch):
    monkeypatch.chdir(ROOT)
    return load_markdown_documents(settings.docs_dir, settings.source_base_url)


@pytest.fixture
async def store(docs):
    s = build_memory_store(HashingEmbeddings())
    await index_documents(s, docs)
    return s
