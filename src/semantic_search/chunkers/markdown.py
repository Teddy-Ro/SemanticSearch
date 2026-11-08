from markdown_it import MarkdownIt

from ..schemas import Document
from .token import TokenChunker


class MarkdownChunker(TokenChunker):
    """Actual Markdown headings (including setext); code-fence contents are not headings."""

    @property
    def signature(self) -> dict:
        return {**super().signature, "type": "markdown", "version": 1}

    def _sections(self, document: Document) -> list[tuple[int, int, str]]:
        text = document.text
        lines = text.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        first_line = 0
        if lines and lines[0].strip() == "---":
            for i in range(1, len(lines)):
                if lines[i].strip() in {"---", "..."}:
                    first_line = i + 1
                    break
        body = "".join(lines[first_line:])
        tokens = MarkdownIt("commonmark").parse(body)
        title = str(document.metadata.get("title", ""))
        sections = []
        headings: list[tuple[int, str]] = []
        start = offsets[first_line]
        context = title
        for i, token in enumerate(tokens):
            if token.type != "heading_open" or token.map is None:
                continue
            position = offsets[first_line + token.map[0]]
            if position > start:
                sections.append((start, position, context))
            level = int(token.tag[1:])
            heading = tokens[i + 1].content
            headings = [(depth, value) for depth, value in headings if depth < level]
            headings.append((level, heading))
            context = " > ".join(value for value in [title, *(h[1] for h in headings)] if value)
            start = position
        if start < len(text):
            sections.append((start, len(text), context))
        return sections
