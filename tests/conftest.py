import hashlib
import math

import pytest
from qdrant_client import QdrantClient

from semantic_search.databases import QdrantDatabase
from semantic_search.embedders.base import BaseEmbedder


class FakeEmbedder(BaseEmbedder):
    """Deterministic test double; tests do not download a model or claim semantic quality."""

    def __init__(self, name="test"):
        self.name = name
        self.document_calls = 0

    @property
    def dimension(self):
        return 4

    @property
    def max_tokens(self):
        return 512

    @property
    def signature(self):
        return {"model": self.name, "dimension": self.dimension}

    def count_tokens(self, text):
        return len(text)

    def document_token_count(self, text):
        return len(text) + 9

    def embed_query(self, text):
        vector = [float(value + 1) for value in hashlib.sha256(text.encode()).digest()[:4]]
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector]

    def embed_documents(self, texts):
        self.document_calls += 1
        return [self.embed_query(text) for text in texts]


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def database():
    db = QdrantDatabase(client=QdrantClient(":memory:"))
    yield db
    db.close()
