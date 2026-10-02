from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    use_cache: bool = True


class Source(BaseModel):
    id: str
    title: str
    section: str
    url: str
    score: float
    snippet: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    mode: Literal["llm", "agent", "extractive"]
    cached: bool = False
    latency_ms: float = 0.0
    tool_calls: list[dict] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    results: list[Source]


class IngestResponse(BaseModel):
    chunks_indexed: int
    backend: str


class HealthResponse(BaseModel):
    status: str
    vector_backend: str
    llm_available: bool
    chunks_indexed: int
