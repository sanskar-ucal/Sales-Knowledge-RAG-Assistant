import hashlib
import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end() :]


def slugify(text: str) -> str:
    slug = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s]+", "-", slug)


def chunk_id(source: str, index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}:{index}:{content}".encode()).hexdigest()
    return digest[:32]


def load_markdown_documents(
    docs_dir: str | Path,
    source_base_url: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Document]:
    """Load markdown files, split by headers then size, and attach source-link metadata."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_path}")

    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS, strip_headers=False)
    size_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks: list[Document] = []
    for path in sorted(docs_path.rglob("*.md")):
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        rel_path = path.as_posix()
        title = meta.get("title") or path.stem.replace("-", " ").title()
        sections = size_splitter.split_documents(header_splitter.split_text(body))
        for i, section in enumerate(sections):
            heading = section.metadata.get("h3") or section.metadata.get("h2") or ""
            anchor = f"#{slugify(heading)}" if heading else ""
            section.metadata = {
                "source": rel_path,
                "title": title,
                "section": heading or title,
                "url": f"{source_base_url.rstrip('/')}/{rel_path}{anchor}",
                "reference": meta.get("reference", ""),
                "chunk_index": i,
                "id": chunk_id(rel_path, i, section.page_content),
            }
            chunks.append(section)

    logger.info("Loaded documents", extra={"chunks": len(chunks), "docs_dir": str(docs_path)})
    return chunks
