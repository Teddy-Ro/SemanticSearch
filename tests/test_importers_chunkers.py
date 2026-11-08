import pytest

from semantic_search.chunkers import MarkdownChunker, TokenChunker
from semantic_search.importers import DirectoryImporter
from semantic_search.schemas import Document, canonical_path


def test_directory_ids_and_filters(tmp_path):
    folder = tmp_path / "DS" / "nested"
    folder.mkdir(parents=True)
    (folder / "Тест.md").write_text("# Привет\nТекст", encoding="utf-8-sig")
    (tmp_path / ".obsidian").mkdir()
    (tmp_path / ".obsidian" / "secret.md").write_text("excluded")
    (tmp_path / "skip.pdf").write_bytes(b"excluded")
    importer = DirectoryImporter(tmp_path)
    docs = list(importer.iter_documents())
    assert len(docs) == 1
    doc = docs[0]
    assert not doc.text.startswith("\ufeff")
    assert doc.source_id == DirectoryImporter(tmp_path / ".").source_id
    assert doc.id == list(importer.iter_documents())[0].id
    assert canonical_path(tmp_path / "DS") in doc.metadata["directories"]
    assert canonical_path(tmp_path / "DS-old") not in doc.metadata["directories"]


def test_missing_directory_is_not_empty_snapshot(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(DirectoryImporter(tmp_path / "missing").iter_documents())


def test_unsupported_formats_fail_explicitly(tmp_path):
    with pytest.raises(ValueError):
        DirectoryImporter(tmp_path, (".pdf",))


@pytest.mark.parametrize("max_tokens,overlap", [(0, 0), (100, 100), (513, 0), (100, -1)])
def test_invalid_token_settings(embedder, max_tokens, overlap):
    with pytest.raises(ValueError):
        TokenChunker(embedder, max_tokens, overlap)


@pytest.mark.parametrize(
    "text", ["", "   ", "абвгд" * 100, "Текст. " * 200, "a\n\nb\n\n" * 100, "👋🌍" * 150]
)
def test_token_limits_offsets_and_coverage(embedder, text):
    doc = Document("d", "s", text)
    chunks = TokenChunker(embedder, max_tokens=80, overlap_tokens=12).split(doc)
    covered = set()
    for chunk in chunks:
        assert embedder.document_token_count(chunk.embedding_text) <= 80
        start, stop = chunk.metadata["char_start"], chunk.metadata["char_end"]
        assert text[start:stop] == chunk.text
        covered.update(range(start, stop))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_markdown_ignores_frontmatter_and_code_headings(embedder):
    text = "---\r\ntags: [test]\r\n---\r\n# Main\r\n```python\r\n# comment\r\nx=1\r\n```\r\n## Child\r\nBody"
    doc = Document("d", "s", text, {"title": "Note"})
    chunks = MarkdownChunker(embedder, max_tokens=300, overlap_tokens=10).split(doc)
    assert len(chunks) == 2
    assert chunks[0].context == "Note > Main"
    assert chunks[1].context == "Note > Main > Child"
    assert "tags:" not in "".join(chunk.text for chunk in chunks)
    assert "```python\r\n# comment" in chunks[0].text
    assert chunks[0].metadata["line_start"] == 4


def test_setext_heading_and_long_title(embedder):
    doc = Document("d", "s", "Heading\n=======\n" + "Body " * 100, {"title": "long " * 100})
    chunks = MarkdownChunker(embedder, max_tokens=100, overlap_tokens=10).split(doc)
    assert chunks
    assert all(embedder.document_token_count(chunk.embedding_text) <= 100 for chunk in chunks)
