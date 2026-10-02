"""Benchmark retrieval and answers against the ground-truth question set.

Usage:
    uv run python -m evaluation.run_eval --mode rag
    uv run python -m evaluation.run_eval --mode agent --judge
"""

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

from app.agent import SalesAgent
from app.config import Settings, get_settings
from app.embeddings import build_embeddings
from app.ingest import load_markdown_documents
from app.llm import LLMProvider
from app.logging_config import configure_logging
from app.rag import RAGService
from app.vectorstore import build_and_index
from evaluation import metrics as m
from evaluation.judge import LLMJudge

ROOT = Path(__file__).resolve().parent.parent


def load_ground_truth(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def unique_in_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]


async def evaluate(
    settings: Settings, mode: str, use_judge: bool, k: int, gt_path: Path, out_dir: Path
) -> dict:
    embeddings = build_embeddings(settings)
    docs = load_markdown_documents(
        settings.docs_dir, settings.source_base_url, settings.chunk_size, settings.chunk_overlap
    )
    store, backend = await build_and_index(settings, embeddings, docs)
    llm = LLMProvider(settings)
    rag = RAGService(store, llm, settings, cache=None)
    agent = SalesAgent(rag, settings.agent_max_steps)
    judge = LLMJudge(llm.chat()) if use_judge and llm.available else None

    rows = []
    for item in load_ground_truth(gt_path):
        question, relevant = item["question"], item.get("relevant_sources", [])
        answerable = item.get("answerable", True)

        retrieved = await rag.retrieve(question, k)
        retrieved_sources = unique_in_order([d.metadata["source"] for d, _ in retrieved])
        contexts = [d.page_content for d, _ in retrieved]

        t0 = time.perf_counter()
        if mode == "agent":
            resp = await agent.run(question, use_cache=False)
        else:
            resp = await rag.answer(question, k, use_cache=False)
        latency = round((time.perf_counter() - t0) * 1000, 1)

        labels = (
            [f"S{i}" for i in range(1, len(resp.sources) + 1)]
            if resp.mode == "agent"
            else [str(i) for i in range(1, len(resp.sources) + 1)]
        )
        abstained = m.is_abstention(resp.answer)
        row = {
            "id": item["id"],
            "question": question,
            "expected_answer": item["expected_answer"],
            "answer": resp.answer,
            "mode": resp.mode,
            "retrieved_sources": retrieved_sources,
            "hit@k": m.hit_at_k(retrieved_sources, relevant, k),
            "recall@k": m.recall_at_k(retrieved_sources, relevant, k),
            "precision@k": m.precision_at_k(retrieved_sources, relevant, k),
            "mrr": m.reciprocal_rank(retrieved_sources, relevant),
            "key_fact_coverage": m.key_fact_coverage(resp.answer, item.get("key_facts", [])),
            "citation_validity": m.citation_validity(resp.answer, labels),
            "unsupported_ratio": m.unsupported_sentence_ratio(resp.answer, contexts),
            "correct_abstention": (1.0 if abstained else 0.0) if not answerable else float("nan"),
            "latency_ms": latency,
        }
        if judge is not None:
            score = await judge.score(
                question, item["expected_answer"], "\n\n".join(contexts), resp.answer
            )
            if score:
                row.update(
                    judge_correctness=score.correctness,
                    judge_faithfulness=score.faithfulness,
                    judge_hallucination=1.0 if score.hallucination else 0.0,
                    judge_rationale=score.rationale,
                )
        rows.append(row)

    metric_keys = [
        "hit@k", "recall@k", "precision@k", "mrr", "key_fact_coverage",
        "citation_validity", "unsupported_ratio", "correct_abstention", "latency_ms",
        "judge_correctness", "judge_faithfulness", "judge_hallucination",
    ]  # fmt: skip
    summary = {
        key: m.mean([r[key] for r in rows if key in r])
        for key in metric_keys
        if any(key in r for r in rows)
    }
    summary.update(
        {"n_questions": len(rows), "k": k, "mode": mode, "backend": backend,
         "llm_available": llm.available, "judge": judge is not None}
    )  # fmt: skip

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"eval_{mode}.json").write_text(
        json.dumps({"summary": summary, "results": rows}, indent=2, default=str)
    )
    write_human_review_sheet(rows, out_dir / f"human_review_{mode}.csv")
    return summary


def write_human_review_sheet(rows: list[dict], path: Path) -> None:
    """CSV for human reviewers: fill in the empty human_* columns, then run summarize_review."""
    fields = ["id", "question", "expected_answer", "answer", "retrieved_sources",
              "human_correct", "human_hallucination", "human_notes"]  # fmt: skip
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    **{k: r[k] for k in fields[:4]},
                    "retrieved_sources": "; ".join(r["retrieved_sources"]),
                    "human_correct": "",
                    "human_hallucination": "",
                    "human_notes": "",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["rag", "agent"], default="rag")
    parser.add_argument("--judge", action="store_true", help="enable LLM-as-judge scoring")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--ground-truth", type=Path, default=ROOT / "evaluation/ground_truth.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", settings.log_json)
    summary = asyncio.run(
        evaluate(settings, args.mode, args.judge, args.k, args.ground_truth, args.out)
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
