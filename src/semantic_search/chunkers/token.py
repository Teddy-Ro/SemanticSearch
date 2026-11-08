import re

from ..embedders.base import BaseEmbedder
from ..schemas import Chunk, Document, stable_id
from .base import BaseChunker


class TokenChunker(BaseChunker):
    """Token-bounded windows with original offsets and optional token overlap."""

    def __init__(self, embedder: BaseEmbedder, max_tokens: int = 320, overlap_tokens: int = 32):
        if not 0 <= overlap_tokens < max_tokens <= embedder.max_tokens:
            raise ValueError("Require 0 <= overlap_tokens < max_tokens <= model limit")
        if embedder.document_token_count("x") > max_tokens:
            raise ValueError("Token budget is too small for the model's document prefix")
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    @property
    def signature(self) -> dict:
        return {
            "type": "token",
            "version": 1,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
        }

    def _sections(self, document: Document) -> list[tuple[int, int, str]]:
        return [(0, len(document.text), str(document.metadata.get("title", "")))]

    def _fits(self, context: str, text: str) -> bool:
        value = f"{context}\n{text}" if context else text
        return self.embedder.document_token_count(value) <= self.max_tokens

    def _end(self, text: str, start: int, stop: int, context: str) -> int:
        # Expand a bounded probe instead of repeatedly tokenizing the entire remaining book.
        low = start
        high = min(stop, start + self.max_tokens * 4)
        while self._fits(context, text[start:high]):
            if high == stop:
                return stop
            low = high
            high = min(stop, start + (high - start) * 2)
        while low < high:
            middle = (low + high + 1) // 2
            if self._fits(context, text[start:middle]):
                low = middle
            else:
                high = middle - 1
        if low == start:
            raise ValueError("Token budget cannot fit document content")
        # Prefer paragraph/sentence/word boundaries, without exceeding the verified window.
        window = text[start:low]
        for pattern in (r"\n\s*\n", r"[.!?]\s+", r"\s+"):
            ends = [m.end() for m in re.finditer(pattern, window) if m.end() >= len(window) // 2]
            if ends and self._fits(context, window[: ends[-1]]):
                return start + ends[-1]
        return low

    def _overlap_start(self, text: str, start: int, stop: int) -> int:
        if self.overlap_tokens == 0:
            return stop
        low, high = start + 1, stop
        while low < high:
            middle = (low + high) // 2
            if self.embedder.count_tokens(text[middle:stop]) <= self.overlap_tokens:
                high = middle
            else:
                low = middle + 1
        return low

    def split(self, document: Document) -> list[Chunk]:
        chunks = []
        for section_start, section_end, context in self._sections(document):
            # Extremely long titles must not consume the whole input budget.
            while context and not self._fits(context, "x" * 32):
                context = context[: len(context) // 2]
            start = section_start
            while start < section_end:
                stop = self._end(document.text, start, section_end, context)
                raw = document.text[start:stop]
                left = start + len(raw) - len(raw.lstrip())
                right = stop - (len(raw) - len(raw.rstrip()))
                text = document.text[left:right]
                if text:
                    # Trimming can change tokenization, so verify the actual final input.
                    if not self._fits(context, text):
                        text, left, right = raw, start, stop
                    index = len(chunks)
                    chunks.append(
                        Chunk(
                            id=stable_id(f"{document.id}/{index}"),
                            document_id=document.id,
                            source_id=document.source_id,
                            text=text,
                            context=context,
                            metadata={
                                **document.metadata,
                                "chunk_index": index,
                                "char_start": left,
                                "char_end": right,
                                "line_start": document.text.count("\n", 0, left) + 1,
                                "line_end": document.text.count("\n", 0, max(left, right - 1)) + 1,
                            },
                        )
                    )
                if stop >= section_end:
                    break
                start = self._overlap_start(document.text, start, stop)
        return chunks
