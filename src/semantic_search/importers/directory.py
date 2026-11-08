import os
from collections.abc import Iterator
from pathlib import Path

from ..schemas import Document, canonical_path, stable_id
from .base import BaseImporter


class DirectoryImporter(BaseImporter):
    """UTF-8 Markdown/TXT directory. Hidden entries and symlinks are excluded."""

    def __init__(self, path: str | Path, extensions: tuple[str, ...] = (".md", ".txt")):
        self.root = Path(canonical_path(path))
        self.extensions = tuple(sorted({ext.lower() for ext in extensions}))
        if not self.extensions or set(self.extensions) - {".md", ".txt"}:
            raise ValueError("DirectoryImporter currently supports .md and .txt only")

    @property
    def source_id(self) -> str:
        return stable_id(self.root.as_uri())

    @property
    def signature(self) -> dict:
        return {
            "type": "directory",
            "version": 1,
            "root": str(self.root),
            "extensions": self.extensions,
        }

    def iter_documents(self) -> Iterator[Document]:
        if not self.root.is_dir():
            raise FileNotFoundError(f"Source directory is unavailable: {self.root}")

        def fail(error: OSError) -> None:
            raise error

        for parent, directories, files in os.walk(self.root, onerror=fail, followlinks=False):
            directories[:] = sorted(
                d for d in directories if not d.startswith(".") and not Path(parent, d).is_symlink()
            )
            for name in sorted(files):
                path = Path(parent, name)
                if name.startswith(".") or path.is_symlink():
                    continue
                if path.suffix.lower() not in self.extensions:
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(self.root):
                    continue
                relative = resolved.relative_to(self.root).as_posix()
                # Ancestors are exact filter values: DS never matches DS-old.
                ancestors = []
                current = resolved.parent
                while current.is_relative_to(self.root):
                    ancestors.append(canonical_path(current))
                    if current == self.root:
                        break
                    current = current.parent
                text = resolved.read_text(encoding="utf-8-sig")
                yield Document(
                    id=stable_id(f"{self.source_id}/{relative}"),
                    source_id=self.source_id,
                    text=text,
                    metadata={
                        "path": canonical_path(resolved),
                        "relative_path": relative,
                        "source_path": str(self.root),
                        "directories": ancestors,
                        "title": resolved.stem,
                        "format": resolved.suffix.lower(),
                    },
                )
