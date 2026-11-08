from dataclasses import dataclass

from .chunkers.base import BaseChunker
from .importers.base import BaseImporter
from .schemas import fingerprint


@dataclass(frozen=True)
class Source:
    importer: BaseImporter
    chunker: BaseChunker

    @property
    def id(self) -> str:
        return self.importer.source_id

    @property
    def signature(self) -> str:
        return fingerprint({"importer": self.importer.signature, "chunker": self.chunker.signature})
