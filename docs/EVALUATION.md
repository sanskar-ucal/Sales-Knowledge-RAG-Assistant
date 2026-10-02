# Evaluation

The benchmark lives in `evaluation/` and runs the same services the API uses.

## Ground truth

`evaluation/ground_truth.jsonl`. Each line has:

| Field | Meaning |
| --- | --- |
| `question` | User question |
| `expected_answer` | Reference answer written by a human |
| `relevant_sources` | Doc paths that contain the answer (retrieval labels) |
| `key_facts` | Short strings a correct answer must mention |
| `answerable` | `false` for out-of-scope questions that should be refused |

## Metrics

**Retrieval quality** (doc-level, after de-duplicating chunks to their source file):
- `hit@k`: at least one relevant doc in the top k
- `recall@k`, `precision@k`
- `mrr`: mean reciprocal rank of the first relevant doc

**Answer correctness**
- `key_fact_coverage`: share of `key_facts` present in the answer
- `judge_correctness` (1-5): LLM-as-judge against the reference answer

**Grounding / hallucination**
- `citation_validity`: share of `[n]` citations that point to a passage that was actually provided
- `unsupported_ratio`: share of answer sentences whose content words are mostly absent from the retrieved context (a cheap, deterministic hallucination proxy)
- `judge_faithfulness` (1-5) and `judge_hallucination` (rate) from the LLM judge
- `correct_abstention`: on unanswerable questions, did the system refuse?

## Running

```bash
# Offline: hashing embeddings, extractive answers, retrieval metrics only
VECTOR_BACKEND=memory EMBEDDING_PROVIDER=hashing uv run python -m evaluation.run_eval --mode rag

# Full: OpenAI embeddings + generation + LLM-as-judge
OPENAI_API_KEY=sk-... uv run python -m evaluation.run_eval --mode rag --judge
OPENAI_API_KEY=sk-... uv run python -m evaluation.run_eval --mode agent --judge
```

Outputs go to `reports/`:
- `eval_<mode>.json`: summary plus per-question rows (answers, metrics, judge rationale)
- `human_review_<mode>.csv`: sheet for human reviewers

## Human review

1. Open `reports/human_review_<mode>.csv` in a spreadsheet.
2. For each row fill `human_correct` (y/n), `human_hallucination` (y/n) and optional notes.
3. Aggregate:

```bash
uv run python -m evaluation.summarize_review reports/human_review_rag.csv
```

Comparing human labels with `judge_correctness` / `judge_hallucination` on the same rows shows how much the LLM judge can be trusted before relying on it for regression testing.

## Baseline (offline, hashing embeddings, k=4, 18 questions)

| Metric | Value |
| --- | --- |
| hit@4 | 1.00 |
| recall@4 | 1.00 |
| MRR | 0.96 |
| key_fact_coverage (extractive) | 0.71 |
| citation_validity | 1.00 |

Extractive mode never abstains (it always quotes the top passages), so `correct_abstention` is only meaningful with an LLM. CI runs the offline benchmark on every push and uploads the report as an artifact.
