import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.llm import LLMProvider
from app.main import create_app
from tests.conftest import ROOT, ScriptedChatModel

pytestmark = pytest.mark.integration


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.chdir(ROOT)
    model = ScriptedChatModel(responder=lambda m: AIMessage("Under 100 words [1]."))
    app = create_app(settings, llm=LLMProvider(settings, primary=model))
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["vector_backend"] == "memory"
    assert body["llm_available"] is True
    assert body["chunks_indexed"] > 0


def test_query_returns_source_linked_answer(client):
    r = client.post("/query", json={"question": "How long should a cold email be?"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "llm" and "[1]" in body["answer"]
    assert body["sources"][0]["url"].startswith("https://github.com/")
    assert "x-request-id" in r.headers


def test_query_is_cached(client):
    payload = {"question": "How long should a cold email be?"}
    client.post("/query", json=payload)
    assert client.post("/query", json=payload).json()["cached"] is True
    assert client.get("/cache/stats").json()["hits"] >= 1


def test_search(client):
    body = client.get("/search", params={"q": "renewal timeline", "k": 2}).json()
    assert len(body["results"]) == 2
    assert body["results"][0]["title"] == "Renewals, Churn, and Expansion"


def test_agent_endpoint(client):
    r = client.post("/agent", json={"question": "How long should a cold email be?"})
    assert r.status_code == 200 and r.json()["mode"] == "agent"


def test_ingest_reindexes_and_clears_cache(client):
    client.post("/query", json={"question": "How long should a cold email be?"})
    body = client.post("/ingest").json()
    assert body["chunks_indexed"] > 0
    assert client.get("/cache/stats").json()["size"] == 0


def test_validation_error(client):
    assert client.post("/query", json={"question": "x"}).status_code == 422


def test_request_id_propagates(client):
    r = client.get("/health", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
