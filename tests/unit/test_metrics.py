import math

from evaluation import metrics as m
from evaluation.judge import parse_judge_output


def test_retrieval_metrics():
    retrieved = ["a", "b", "c"]
    assert m.hit_at_k(retrieved, ["c"], 3) == 1.0
    assert m.hit_at_k(retrieved, ["c"], 2) == 0.0
    assert m.recall_at_k(retrieved, ["a", "z"], 3) == 0.5
    assert m.precision_at_k(retrieved, ["a", "b"], 2) == 1.0
    assert m.reciprocal_rank(retrieved, ["b"]) == 0.5
    assert m.reciprocal_rank(retrieved, ["z"]) == 0.0
    assert math.isnan(m.hit_at_k(retrieved, [], 3))


def test_key_fact_coverage():
    assert m.key_fact_coverage("Approve up to 10% yourself", ["10%", "VP"]) == 0.5
    assert math.isnan(m.key_fact_coverage("x", []))


def test_citation_validity():
    assert m.citation_validity("Foo [1]. Bar [3].", ["1", "2"]) == 0.5
    assert m.citation_validity("Foo [S1][S2].", ["S1", "S2"]) == 1.0
    assert math.isnan(m.citation_validity("no citations", ["1"]))


def test_unsupported_sentence_ratio():
    ctx = ["Pipeline coverage should be 3x to 4x of remaining quota at quarter start."]
    grounded = "Pipeline coverage should be 3x to 4x of remaining quota [1]."
    invented = "Our CEO mandates quarterly hackathons for every engineering squad."
    assert m.unsupported_sentence_ratio(grounded, ctx) == 0.0
    assert m.unsupported_sentence_ratio(f"{grounded} {invented}", ctx) == 0.5


def test_abstention_detection():
    assert m.is_abstention("I don't know based on the available sales documentation.")
    assert not m.is_abstention("Use 3x coverage.")


def test_mean_ignores_nan():
    assert m.mean([1.0, float("nan"), 0.0]) == 0.5


def test_parse_judge_output():
    s = parse_judge_output(
        'Sure: {"correctness": 5, "faithfulness": 4, "hallucination": false, "rationale": "ok"}'
    )
    assert s.correctness == 5 and s.faithfulness == 4 and s.hallucination is False
