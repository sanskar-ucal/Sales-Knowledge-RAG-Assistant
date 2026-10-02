import pytest

from app.ingest import load_markdown_documents, parse_front_matter, slugify


def test_parse_front_matter():
    meta, body = parse_front_matter("---\ntitle: Hello\nreference: http://x\n---\n# Body\n")
    assert meta == {"title": "Hello", "reference": "http://x"}
    assert body.startswith("# Body")


def test_parse_front_matter_absent():
    meta, body = parse_front_matter("# Just text")
    assert meta == {} and body == "# Just text"


def test_slugify():
    assert slugify("Give-get trading") == "give-get-trading"
    assert slugify("What's next?") == "whats-next"


def test_chunks_have_source_metadata(docs):
    assert len(docs) > 20
    ids = [d.metadata["id"] for d in docs]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    for d in docs:
        md = d.metadata
        assert md["source"].startswith("data/docs/")
        assert md["url"].startswith("https://github.com/")
        assert md["title"]
        assert len(d.page_content) <= 800


def test_section_anchor_in_url(docs):
    give_get = [d for d in docs if d.metadata["section"] == "Give-get trading"]
    assert give_get and give_get[0].metadata["url"].endswith("#give-get-trading")


def test_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_markdown_documents(tmp_path / "nope", "https://example.com")
