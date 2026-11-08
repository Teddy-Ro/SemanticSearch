"""Shared data; no database or ML-library types cross component boundaries."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5


def canonical_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def stable_id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, value))


def fingerprint(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class IndexMismatchError(ValueError):
    """Stored vectors or chunks were produced with incompatible settings."""


class IndexBusyError(RuntimeError):
    """This source is already being indexed."""


@dataclass(frozen=True)
class Document:
    id: str
    source_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return fingerprint({"text": self.text, "metadata": self.metadata})


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    source_id: str
    text: str
    context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def embedding_text(self) -> str:
        return f"{self.context}\n{self.text}" if self.context else self.text


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass
class IndexReport:
    source_id: str
    indexed: int = 0
    skipped: int = 0
    deleted: int = 0
    chunks: int = 0
