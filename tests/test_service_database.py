from dataclasses import replace
from threading import Barrier

import pytest
from conftest import FakeEmbedder
from fastapi.testclient import TestClient

from semantic_search.api import create_app
from semantic_search.chunkers import MarkdownChunker, TokenChunker
from semantic_search.importers import DirectoryImporter
from semantic_search.schemas import Chunk, Document, IndexMismatchError, canonical_path, stable_id
from semantic_search.service import SearchService
from semantic_search.source import Source


def make_service(tmp_path, embedder, database, size=180):
    source = Source(DirectoryImporter(tmp_path), MarkdownChunker(embedder, size, 10))
    return SearchService([source], embedder, database), source


def test_incremental_update_delete_and_empty_document(tmp_path, embedder, database):
    path = tmp_path / "note.md"
    path.write_text("# Title\nOld content")
    service, source = make_service(tmp_path, embedder, database)
    assert service.index()[0].indexed == 1
    assert service.index()[0].skipped == 1
    assert embedder.document_calls == 1
    path.write_text("# Title\nNew content")
    assert service.index()[0].indexed == 1
    results = service.search("New")
    assert results and all("Old content" not in result.chunk.text for result in results)
    path.write_text("")
    service.index()
    assert service.search("New") == []
    assert len(database.documents(source.id)) == 1
    path.unlink()
    assert service.index()[0].deleted == 1
    assert database.documents(source.id) == {}


def test_directory_filter_respects_boundaries(tmp_path, embedder, database):
    for folder in ("DS/nested", "DS-old"):
        path = tmp_path / folder
        path.mkdir(parents=True)
        (path / "note.md").write_text("Some content")
    service, _ = make_service(tmp_path, embedder, database)
    service.index()
    results = service.search("content", directory=str(tmp_path / "DS"))
    assert len(results) == 1
    assert results[0].chunk.metadata["path"] == canonical_path(tmp_path / "DS/nested/note.md")


def test_model_mismatch_even_with_same_dimensions(tmp_path, embedder, database):
    make_service(tmp_path, embedder, database)
    with pytest.raises(IndexMismatchError):
        make_service(tmp_path, FakeEmbedder("another-model"), database)


def test_changed_chunker_requires_source_rebuild(tmp_path, embedder, database):
    (tmp_path / "note.md").write_text("Content " * 40)
    service, _ = make_service(tmp_path, embedder, database)
    service.index()
    changed, _ = make_service(tmp_path, embedder, database, size=100)
    with pytest.raises(IndexMismatchError):
        changed.search("Content")
    with pytest.raises(IndexMismatchError):
        changed.index()
    assert changed.index(rebuild=True)[0].indexed == 1
    assert changed.search("Content")


def test_missing_source_does_not_delete_old_index(tmp_path, embedder, database):
    root = tmp_path / "source"
    root.mkdir()
    (root / "note.md").write_text("Content")
    service, source = make_service(root, embedder, database)
    service.index()
    root.rename(tmp_path / "temporarily_unavailable")
    with pytest.raises(FileNotFoundError):
        service.index()
    assert len(database.documents(source.id)) == 1


def test_sources_import_concurrently_and_keep_distinct_chunkers(tmp_path, embedder, database):
    barrier = Barrier(2)

    class SynchronizedImporter(DirectoryImporter):
        def iter_documents(self):
            barrier.wait(timeout=5)
            yield from super().iter_documents()

    sources = []
    for index, chunker in enumerate(
        (MarkdownChunker(embedder, 180, 10), TokenChunker(embedder, 100, 5))
    ):
        root = tmp_path / str(index)
        root.mkdir()
        (root / "same.md").write_text("# Heading\nText " * 30)
        sources.append(Source(SynchronizedImporter(root), chunker))
    service = SearchService(sources, embedder, database, workers=2)
    assert len(service.index()) == 2
    assert sources[0].signature != sources[1].signature
    results = service.search("Text", source_ids=[sources[0].id])
    assert results and all(result.chunk.source_id == sources[0].id for result in results)


def test_interrupted_publication_keeps_previous_revision_searchable(
    embedder, database, monkeypatch
):
    database.ensure_index(embedder.signature, embedder.dimension)
    doc = Document("document", "source", "old")
    old = Chunk(stable_id("old"), doc.id, doc.source_id, "old")
    database.replace_document(doc, [old], [[0.1, 1.0, 0.0, 0.0]])
    original_put = database._put

    def interrupted(key, payload):
        if key.startswith("document:"):
            raise OSError("Simulated interrupted manifest write")
        original_put(key, payload)

    monkeypatch.setattr(database, "_put", interrupted)
    # More orphaned high-scoring points than one result page: search must keep paging.
    new_chunks = [Chunk(stable_id(str(i)), doc.id, doc.source_id, "new") for i in range(70)]
    with pytest.raises(OSError):
        database.replace_document(replace(doc, text="new"), new_chunks, [[1.0, 0.0, 0.0, 0.0]] * 70)
    results = database.search([1.0, 0.0, 0.0, 0.0], 1, [doc.source_id])
    assert [result.chunk.text for result in results] == ["old"]
    monkeypatch.setattr(database, "_put", original_put)
    database.replace_document(replace(doc, text="new"), new_chunks[:1], [[1.0, 0.0, 0.0, 0.0]])
    assert database.client.count(database.collection, exact=True).count == 1


def test_bad_vectors_never_replace_working_document(embedder, database):
    database.ensure_index(embedder.signature, embedder.dimension)
    doc = Document("doc", "source", "text")
    chunk = Chunk(stable_id("chunk"), doc.id, doc.source_id, doc.text)
    database.replace_document(doc, [chunk], [[1.0, 0.0, 0.0, 0.0]])
    for vectors in ([], [[1.0]], [[float("nan"), 0.0, 0.0, 0.0]]):
        with pytest.raises(ValueError):
            database.replace_document(doc, [chunk], vectors)
    assert database.search([1.0, 0.0, 0.0, 0.0], 1, [doc.source_id])


def test_api_index_search_and_validation(tmp_path, embedder, database):
    (tmp_path / "note.md").write_text("# Search\nExample content")
    service, source = make_service(tmp_path, embedder, database)
    with TestClient(create_app(service)) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/sources").json()[0]["id"] == source.id
        assert client.post("/index", json={}).status_code == 200
        response = client.post("/search", json={"query": "content", "directory": str(tmp_path)})
        assert response.status_code == 200 and response.json()
        assert client.post("/search", json={"query": "   "}).status_code == 422
        assert client.post("/search", json={"query": "x", "limit": 101}).status_code == 422
        assert client.post("/index", json={"source_ids": ["unknown"]}).status_code == 422
