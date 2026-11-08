from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from .databases.base import BaseDatabase
from .embedders.base import BaseEmbedder
from .schemas import IndexBusyError, IndexMismatchError, IndexReport, SearchResult, canonical_path
from .source import Source


class SearchService:
    """Orchestration only: no Qdrant, tokenizer or HTTP-library dependencies."""

    def __init__(
        self,
        sources: list[Source],
        embedder: BaseEmbedder,
        database: BaseDatabase,
        workers: int = 2,
    ):
        self.sources = {source.id: source for source in sources}
        if len(self.sources) != len(sources):
            raise ValueError("A source is configured more than once")
        if workers < 1:
            raise ValueError("workers must be positive")
        self.embedder, self.database, self.workers = embedder, database, workers
        self._locks = {source_id: Lock() for source_id in self.sources}
        self.database.ensure_index(embedder.signature, embedder.dimension)

    def _select(self, source_ids: list[str] | None) -> list[Source]:
        if source_ids is None:
            return list(self.sources.values())
        missing = set(source_ids) - self.sources.keys()
        if missing:
            raise ValueError(f"Unknown source IDs: {', '.join(sorted(missing))}")
        return [self.sources[source_id] for source_id in dict.fromkeys(source_ids)]

    def list_sources(self) -> list[dict]:
        result = []
        for source in self.sources.values():
            state = self.database.get_source(source.id)
            result.append(
                {
                    "id": source.id,
                    "importer": source.importer.signature,
                    "chunker": source.chunker.signature,
                    "indexing": self._locks[source.id].locked(),
                    "ready": bool(
                        state and state.get("ready") and state.get("signature") == source.signature
                    ),
                }
            )
        return result

    def index(
        self, source_ids: list[str] | None = None, *, rebuild: bool = False
    ) -> list[IndexReport]:
        sources = self._select(source_ids)
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = [executor.submit(self._index_source, source, rebuild) for source in sources]
            # All tasks finish before this call returns, even if one source fails.
            return [future.result() for future in futures]

    def _index_source(self, source: Source, rebuild: bool) -> IndexReport:
        lock = self._locks[source.id]
        if not lock.acquire(blocking=False):
            raise IndexBusyError(f"Source is already being indexed: {source.id}")
        try:
            return self._index_locked(source, rebuild)
        finally:
            lock.release()

    def _index_locked(self, source: Source, rebuild: bool) -> IndexReport:
        state = self.database.get_source(source.id)
        changed = bool(state and state.get("signature") != source.signature)
        if changed and not rebuild:
            raise IndexMismatchError(
                f"Source {source.id} processing changed. Reindex with rebuild."
            )
        force = rebuild or bool(state and not state.get("ready"))
        new_state = {
            "signature": source.signature,
            "importer": source.importer.signature,
            "chunker": source.chunker.signature,
        }
        if state is None or force:
            self.database.set_source(source.id, {**new_state, "ready": False})
        existing = self.database.documents(source.id)
        seen = set()
        report = IndexReport(source.id)
        for document in source.importer.iter_documents():
            if document.source_id != source.id or document.id in seen:
                raise ValueError("Importer returned a wrong source ID or duplicate document ID")
            seen.add(document.id)
            previous = existing.get(document.id)
            if not force and previous and previous["content_hash"] == document.content_hash:
                report.skipped += 1
                continue
            chunks = source.chunker.split(document)
            vectors = self.embedder.embed_documents([chunk.embedding_text for chunk in chunks])
            self.database.replace_document(document, chunks, vectors)
            report.indexed += 1
            report.chunks += len(chunks)
        # Only a completed source snapshot authorizes removal of missing documents.
        for document_id in existing.keys() - seen:
            self.database.delete_document(document_id)
            report.deleted += 1
        self.database.set_source(source.id, {**new_state, "ready": True})
        return report

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        source_ids: list[str] | None = None,
        directory: str | None = None,
    ) -> list[SearchResult]:
        query = query.strip()
        if not query or not 1 <= limit <= 100:
            raise ValueError("Query must not be empty; limit must be between 1 and 100")
        selected = []
        for source in self._select(source_ids):
            state = self.database.get_source(source.id)
            if state is None:
                continue
            if not state.get("ready") or state.get("signature") != source.signature:
                raise IndexMismatchError(f"Source {source.id} needs a completed reindex")
            selected.append(source.id)
        if not selected:
            return []
        vector = self.embedder.embed_query(query)
        return self.database.search(
            vector, limit, selected, canonical_path(directory) if directory else None
        )

    def close(self) -> None:
        self.database.close()
