from abc import ABC, abstractmethod

from ..schemas import Chunk, Document


class BaseChunker(ABC):
    @property
    @abstractmethod
    def signature(self) -> dict: ...

    @abstractmethod
    def split(self, document: Document) -> list[Chunk]: ...
