import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from app.config import Settings

logger = logging.getLogger(__name__)


def _openai_model(settings: Settings, model: str) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
    )


class LLMProvider:
    """Builds chat models with a primary -> fallback model chain.

    When no API key is configured, `available` is False and callers degrade to
    extractive (retrieval-only) answers.
    """

    def __init__(
        self,
        settings: Settings,
        primary: BaseChatModel | None = None,
        fallback: BaseChatModel | None = None,
    ):
        self.settings = settings
        if primary is None and settings.openai_api_key:
            primary = _openai_model(settings, settings.chat_model)
            fallback = fallback or _openai_model(settings, settings.fallback_chat_model)
        self.primary = primary
        self.fallback = fallback

    @property
    def available(self) -> bool:
        return self.primary is not None

    def chat(self) -> Runnable:
        if self.primary is None:
            raise RuntimeError("No chat model configured")
        if self.fallback is None:
            return self.primary
        return self.primary.with_fallbacks([self.fallback])

    def chat_with_tools(self, tools: Sequence[BaseTool]) -> Runnable:
        if self.primary is None:
            raise RuntimeError("No chat model configured")
        bound = self.primary.bind_tools(tools)
        if self.fallback is None:
            return bound
        return bound.with_fallbacks([self.fallback.bind_tools(tools)])
