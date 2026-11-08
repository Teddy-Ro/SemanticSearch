import math
from dataclasses import asdict
from threading import RLock
from uuid import uuid4

from qdrant_client import QdrantClient, models

from ..schemas import Chunk, Document, IndexMismatchError, SearchResult, fingerprint, stable_id
from .base import BaseDatabase


def match(key: str, value: str) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


class QdrantDatabase(BaseDatabase):
    """Qdrant payloads hold text and manifests. No second database is required.

    Each update writes an immutable document revision, publishes its manifest, then
    removes obsolete points. Search checks manifests, including after interrupted writes.
    One service process owns writes; the lock also supports embedded Qdrant safely.
    """

    def __init__(
        self,
        collection: str = "documents",
        *,
        url: str | None = None,
        path: str | None = None,
        client: QdrantClient | None = None,
    ):
        if client is None and bool(url) == bool(path):
            raise ValueError("Specify exactly one Qdrant url or path")
        self.client = client or (QdrantClient(url=url) if url else QdrantClient(path=path))
        self.collection = collection
        self.metadata_collection = f"{collection}__metadata"
        self._lock = RLock()
        self._dimension = 0

    def _get(self, key: str) -> dict | None:
        records = self.client.retrieve(
            self.metadata_collection, [stable_id(key)], with_payload=True, with_vectors=False
        )
        return records[0].payload if records else None

    def _put(self, key: str, payload: dict) -> None:
        self.client.upsert(
            self.metadata_collection,
            [models.PointStruct(id=stable_id(key), vector={}, payload=payload)],
            wait=True,
        )

    def ensure_index(self, embedding_signature: dict, dimension: int) -> None:
        with self._lock:
            if not self.client.collection_exists(self.metadata_collection):
                self.client.create_collection(self.metadata_collection, vectors_config={})
            stored = self._get("index")
            signature = fingerprint(embedding_signature)
            if stored and (stored["signature"] != signature or stored["dimension"] != dimension):
                raise IndexMismatchError(
                    "Embedding model changed. Use a new collection and reindex."
                )
            if not self.client.collection_exists(self.collection):
                if stored:
                    raise IndexMismatchError("Vector collection is missing. Use a new collection.")
                self.client.create_collection(
                    self.collection,
                    vectors_config=models.VectorParams(
                        size=dimension, distance=models.Distance.COSINE
                    ),
                )
            elif stored is None:
                raise IndexMismatchError("Collection has no model manifest. Use a new collection.")
            if stored is None:
                self._put(
                    "index",
                    {
                        "kind": "index",
                        "signature": signature,
                        "embedding": embedding_signature,
                        "dimension": dimension,
                    },
                )
            self._dimension = dimension

    def get_source(self, source_id: str) -> dict | None:
        with self._lock:
            return self._get(f"source:{source_id}")

    def set_source(self, source_id: str, state: dict) -> None:
        with self._lock:
            self._put(f"source:{source_id}", {**state, "kind": "source", "source_id": source_id})

    def documents(self, source_id: str) -> dict[str, dict]:
        with self._lock:
            result = {}
            offset = None
            while True:
                points, offset = self.client.scroll(
                    self.metadata_collection,
                    scroll_filter=models.Filter(
                        must=[match("kind", "document"), match("source_id", source_id)]
                    ),
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                result.update({point.payload["document_id"]: point.payload for point in points})
                if offset is None:
                    return result

    def _validate_vectors(self, vectors: list[list[float]]) -> None:
        for vector in vectors:
            if len(vector) != self._dimension or any(not math.isfinite(v) for v in vector):
                raise ValueError("Invalid embedding dimension or non-finite vector")

    def replace_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk/vector counts differ")
        self._validate_vectors(vectors)
        version = str(uuid4())
        points = [
            models.PointStruct(
                id=stable_id(f"{chunk.id}/{version}"),
                vector=vector,
                payload={**asdict(chunk), "version": version},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        with self._lock:
            for start in range(0, len(points), 128):
                self.client.upsert(self.collection, points[start : start + 128], wait=True)
            self._put(
                f"document:{document.id}",
                {
                    "kind": "document",
                    "document_id": document.id,
                    "source_id": document.source_id,
                    "content_hash": document.content_hash,
                    "version": version,
                    "chunks": len(chunks),
                    "metadata": document.metadata,
                },
            )
            self.client.delete(
                self.collection,
                models.FilterSelector(
                    filter=models.Filter(
                        must=[match("document_id", document.id)],
                        must_not=[match("version", version)],
                    )
                ),
                wait=True,
            )

    def delete_document(self, document_id: str) -> None:
        with self._lock:
            # Removing the manifest first hides old points even if cleanup is interrupted.
            self.client.delete(
                self.metadata_collection,
                models.PointIdsList(points=[stable_id(f"document:{document_id}")]),
                wait=True,
            )
            self.client.delete(
                self.collection,
                models.FilterSelector(
                    filter=models.Filter(must=[match("document_id", document_id)])
                ),
                wait=True,
            )

    def search(
        self, vector: list[float], limit: int, source_ids: list[str], directory: str | None = None
    ) -> list[SearchResult]:
        self._validate_vectors([vector])
        if not source_ids:
            return []
        filters = [models.FieldCondition(key="source_id", match=models.MatchAny(any=source_ids))]
        if directory:
            filters.append(match("metadata.directories", directory))
        results = []
        offset, page_size = 0, max(32, limit * 2)
        with self._lock:
            while len(results) < limit:
                points = self.client.query_points(
                    self.collection,
                    query=vector,
                    query_filter=models.Filter(must=filters),
                    limit=page_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                ).points
                if not points:
                    break
                document_ids = {point.payload["document_id"] for point in points}
                manifests = self.client.retrieve(
                    self.metadata_collection,
                    [stable_id(f"document:{doc_id}") for doc_id in document_ids],
                    with_payload=True,
                    with_vectors=False,
                )
                active = {
                    point.payload["document_id"]: point.payload["version"] for point in manifests
                }
                for point in points:
                    payload = dict(point.payload)
                    if active.get(payload["document_id"]) != payload.pop("version"):
                        continue
                    results.append(SearchResult(chunk=Chunk(**payload), score=point.score))
                    if len(results) == limit:
                        break
                offset += len(points)
                if len(points) < page_size:
                    break
        return results

    def close(self) -> None:
        with self._lock:
            self.client.close()
