# Sales Knowledge RAG Assistant

[![CI](https://github.com/sanskar-ucal/Sales-Knowledge-RAG-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/sanskar-ucal/Sales-Knowledge-RAG-Assistant/actions/workflows/ci.yml)

A retrieval-augmented assistant that answers sales-enablement questions (qualification, discovery, objection handling, pricing, forecasting, renewals) from a documentation corpus, with **source-linked, cited answers**.

**Stack:** Python 3.12, LangChain, PostgreSQL + pgvector, FastAPI (async), Docker, GitHub Actions.

## Features

- **RAG over sales docs**: markdown ingestion with header-aware chunking; every chunk carries a deep link (`.../file.md#section`) that is returned with the answer.
- **pgvector search** via `langchain-postgres` in async mode, with idempotent upserts (deterministic chunk ids).
- **Prompt engineering for grounding**: context-only answers, mandatory `[n]` citations, fixed abstention phrase for out-of-scope questions.
- **Tool-based retrieval agent**: `/agent` endpoint where the LLM calls `search_sales_docs` / `list_sales_topics` tools, possibly several times, before answering.
- **Evaluation harness**: ground-truth questions; retrieval metrics (hit@k, recall@k, precision@k, MRR); answer correctness (key-fact coverage, LLM-as-judge); hallucination checks (citation validity, unsupported-sentence ratio, judge faithfulness); human-review CSV workflow.
- **Production concerns**: structured JSON logging with request ids, async TTL/LRU cache, model fallbacks, graceful degradation (pgvector to in-memory, LLM to extractive), Docker, CI/CD, unit and integration tests.
- **Runs fully offline**: with no API key it uses local hashing embeddings and extractive answers, so tests and CI need no secrets.

## Quickstart

### Docker (API + pgvector)

```bash
cp .env.example .env          # optionally add OPENAI_API_KEY
docker compose up --build
```

### Local

```bash
uv sync
cp .env.example .env
# Without Postgres:
VECTOR_BACKEND=memory uv run uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

## API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Backend, LLM availability, chunk count |
| `POST` | `/query` | Single-shot RAG answer with sources |
| `POST` | `/agent` | Tool-calling agent answer with sources and tool trace |
| `GET` | `/search?q=&k=` | Raw vector search results |
| `POST` | `/ingest` | Re-index `data/docs` and clear the cache |
| `GET` | `/cache/stats` | Cache size, hits, misses |

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "How should I handle a price objection?"}'
```

```json
{
  "question": "How should I handle a price objection?",
  "answer": "Treat it as a value objection: ask what they are comparing the price against and re-anchor on the cost of the problem [1]. Avoid discounting immediately [1].",
  "sources": [
    {
      "title": "Objection Handling Guide",
      "section": "\"It's too expensive\"",
      "url": "https://github.com/sanskar-ucal/Sales-Knowledge-RAG-Assistant/blob/main/data/docs/objection-handling.md#its-too-expensive",
      "score": 0.83,
      "snippet": "..."
    }
  ],
  "mode": "llm",
  "cached": false
}
```

## Evaluation

```bash
make eval                 # offline retrieval benchmark
make eval-agent           # agent + LLM-as-judge (needs OPENAI_API_KEY)
```

See [docs/EVALUATION.md](docs/EVALUATION.md) for metric definitions, the human review workflow, and baseline numbers.

## Testing

```bash
make test                 # unit + integration (pgvector tests run when DATABASE_URL is set)
make lint
```

CI (`.github/workflows/ci.yml`) runs lint, then unit and integration tests against a real `pgvector/pgvector:pg16` service, then the offline benchmark (uploaded as an artifact), then builds the Docker image and pushes it to GHCR on `main`.

## Project layout

```
app/            FastAPI app, RAG service, agent, vector store, LLM provider, cache, logging
evaluation/     ground truth, metrics, LLM judge, benchmark runner, human review summarizer
data/docs/      knowledge base (markdown with front matter)
tests/          unit/ and integration/
docs/           architecture and evaluation docs
```

## Knowledge base

`data/docs/` contains original write-ups of publicly documented sales methodologies (MEDDICC, BANT, SPIN, LAER, BATNA, pipeline and retention metrics). Each file links its public reference in the `reference` front matter field. Some playbook details, such as the discount approval tiers, are illustrative examples. To use your own corpus, drop markdown files into `data/docs/` and call `POST /ingest`.

## Configuration

All settings are environment variables (see `.env.example` and `app/config.py`). Key ones: `OPENAI_API_KEY`, `CHAT_MODEL`, `FALLBACK_CHAT_MODEL`, `EMBEDDING_PROVIDER`, `VECTOR_BACKEND`, `DATABASE_URL`, `TOP_K`, `MIN_RELEVANCE`, `CACHE_TTL_SECONDS`.

Architecture details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
