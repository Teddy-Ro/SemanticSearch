from abc import ABC, abstractmethod

from ..schemas import Chunk, Document, SearchResult


class BaseDatabase(ABC):
    @abstractmethod
    def ensure_index(self, embedding_signature: dict, dimension: int) -> None: ...

    @abstractmethod
    def get_source(self, source_id: str) -> dict | None: ...

    @abstractmethod
    def set_source(self, source_id: str, state: dict) -> None: ...

    @abstractmethod
    def documents(self, source_id: str) -> dict[str, dict]: ...

    @abstractmethod
    def replace_document(
        self, document: Document, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        """Publish only after all new chunks are stored; old revisions must not leak into search."""
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...

    @abstractmethod
    def search(
        self, vector: list[float], limit: int, source_ids: list[str], directory: str | None = None
    ) -> list[SearchResult]: ...

    @abstractmethod
    def close(self) -> None: ...
