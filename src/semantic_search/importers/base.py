from abc import ABC, abstractmethod
from collections.abc import Iterator

from ..schemas import Document


class BaseImporter(ABC):
    @property
    @abstractmethod
    def source_id(self) -> str: ...

    @property
    @abstractmethod
    def signature(self) -> dict: ...

    @abstractmethod
    def iter_documents(self) -> Iterator[Document]:
        """Yield a full snapshot. Raise on incomplete reads; never hide missing sources."""
        ...
