from contextlib import contextmanager
from pathlib import Path
from threading import Condition

from .base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """One CPU model; query work takes priority between document batches."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        query_prefix: str = "query: ",
        document_prefix: str = "passage: ",
        device: str = "cpu",
        batch_size: int = 8,
        threads: int = 4,
        revision: str | None = None,
        cache_folder: str | None = None,
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        if batch_size < 1 or threads < 1:
            raise ValueError("batch_size and threads must be positive")
        local_model = Path(model_name).is_dir()
        if local_model and not revision:
            raise ValueError("A local model directory requires an explicit revision identifier")
        torch.set_num_threads(threads)
        self.model = SentenceTransformer(
            model_name,
            device=device,
            revision=revision,
            cache_folder=cache_folder,
            trust_remote_code=False,
        )
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self._condition = Condition()
        self._busy = False
        self._waiting_queries = 0
        config = getattr(getattr(self.model[0], "auto_model", None), "config", None)
        resolved_revision = revision if local_model else getattr(config, "_commit_hash", None)
        if not resolved_revision:
            # New Transformers versions need not expose _commit_hash on the loaded model.
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(
                model_name, "config.json", cache_dir=cache_folder, revision=revision
            )
            if isinstance(cached, str) and Path(cached).parent.parent.name == "snapshots":
                resolved_revision = Path(cached).parent.name
        if not resolved_revision:
            raise ValueError(
                "Cannot determine the loaded model revision; use a versioned local model"
            )
        self._signature = {
            "type": "sentence-transformer",
            "model": model_name,
            "revision": resolved_revision,
            "dimension": self.dimension,
            "max_tokens": self.max_tokens,
            "query_prefix": query_prefix,
            "document_prefix": document_prefix,
            "normalize": True,
            "implementation_version": 1,
        }

    @property
    def dimension(self) -> int:
        getter = getattr(self.model, "get_embedding_dimension", None)
        return int(getter() if getter else self.model.get_embedding_dimension())

    @property
    def max_tokens(self) -> int:
        return int(self.model.max_seq_length)

    @property
    def signature(self) -> dict:
        return self._signature.copy()

    def count_tokens(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=False, truncation=False))

    def document_token_count(self, text: str) -> int:
        return self._input_count(self.document_prefix + text)

    def _input_count(self, text: str) -> int:
        return len(self.model.tokenizer.encode(text, add_special_tokens=True, truncation=False))

    @contextmanager
    def _turn(self, query: bool):
        with self._condition:
            if query:
                self._waiting_queries += 1
            try:
                self._condition.wait_for(
                    lambda: not self._busy and (query or self._waiting_queries == 0)
                )
                self._busy = True
            finally:
                if query:
                    self._waiting_queries -= 1
        try:
            yield
        finally:
            with self._condition:
                self._busy = False
                self._condition.notify_all()

    def _encode(self, texts: list[str], *, query: bool) -> list[list[float]]:
        if any(self._input_count(text) > self.max_tokens for text in texts):
            raise ValueError(f"Input exceeds model limit ({self.max_tokens} tokens)")
        with self._turn(query):
            # Explicitly suppress saved prompts: inputs already contain configured prefixes.
            vectors = self.model.encode(
                texts,
                prompt="",
                normalize_embeddings=True,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text], query=True)[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch = [self.document_prefix + text for text in texts[start : start + self.batch_size]]
            vectors.extend(self._encode(batch, query=False))
        return vectors
