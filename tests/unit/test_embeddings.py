import math

from app.embeddings import HashingEmbeddings


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_embeddings_are_deterministic_and_normalized():
    emb = HashingEmbeddings(dim=128)
    v1 = emb.embed_query("pipeline coverage ratio")
    v2 = emb.embed_query("pipeline coverage ratio")
    assert v1 == v2
    assert len(v1) == 128
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, rel_tol=1e-6)


def test_similar_texts_score_higher_than_unrelated():
    emb = HashingEmbeddings()
    q = emb.embed_query("how do I handle a price objection")
    related = emb.embed_documents(["Price objections are usually value objections."])[0]
    unrelated = emb.embed_documents(["Start renewals 120 days before contract end."])[0]
    assert cosine(q, related) > cosine(q, unrelated)


def test_empty_text_does_not_crash():
    assert len(HashingEmbeddings(dim=16).embed_query("")) == 16
