import sys
from types import SimpleNamespace

import pytest

from semantic_search.embedders import SentenceTransformerEmbedder


@pytest.mark.parametrize("exposes_revision", [True, False])
def test_prefixes_batching_revision_and_truncation_guard(monkeypatch, exposes_revision):
    calls = []

    class Model:
        max_seq_length = 50
        tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text))

        def __init__(self, *args, **kwargs):
            pass

        def __getitem__(self, index):
            return SimpleNamespace(
                auto_model=SimpleNamespace(
                    config=SimpleNamespace(
                        _commit_hash="resolved-revision" if exposes_revision else None
                    )
                )
            )

        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return SimpleNamespace(tolist=lambda: [[1.0, 0.0] for _ in texts])

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda value: None))
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(
            try_to_load_from_cache=lambda *args, **kwargs: (
                "C:/models/snapshots/resolved-revision/config.json"
            )
        ),
    )
    embedder = SentenceTransformerEmbedder(batch_size=2)
    assert embedder.signature["revision"] == "resolved-revision"
    assert embedder.embed_query("вопрос") == [1.0, 0.0]
    assert calls[0][0] == ["query: вопрос"]
    assert len(embedder.embed_documents(["один", "два", "три"])) == 3
    assert calls[1][0] == ["passage: один", "passage: два"]
    assert calls[2][0] == ["passage: три"]
    assert calls[1][1]["prompt"] == ""
    with pytest.raises(ValueError):
        embedder.embed_query("x" * 50)
    assert len(calls) == 3
