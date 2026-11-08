import argparse
import json
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path

from .chunkers import MarkdownChunker, TokenChunker
from .databases import QdrantDatabase
from .embedders import SentenceTransformerEmbedder
from .importers import DirectoryImporter
from .service import SearchService
from .source import Source


def read_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    with config_path.open("rb") as stream:
        return tomllib.load(stream), config_path.parent


def build_service(config: dict, root: Path) -> SearchService:
    embedding = dict(config.get("embedding", {}))
    embedding["model_name"] = embedding.pop("model", "intfloat/multilingual-e5-small")
    if embedding.get("cache_folder"):
        embedding["cache_folder"] = str(root / embedding["cache_folder"])
    embedder = SentenceTransformerEmbedder(**embedding)
    sources = []
    for settings in config.get("sources", []):
        importer = DirectoryImporter(
            root / settings["path"], tuple(settings.get("extensions", [".md", ".txt"]))
        )
        chunker_name = settings.get("chunker", "markdown")
        if chunker_name == "markdown":
            chunker_type = MarkdownChunker
        elif chunker_name == "token":
            chunker_type = TokenChunker
        else:
            raise ValueError(f"Unknown chunker: {chunker_name}")
        chunker = chunker_type(
            embedder,
            max_tokens=settings.get("max_tokens", 320),
            overlap_tokens=settings.get("overlap_tokens", 32),
        )
        sources.append(Source(importer, chunker))
    database_config = dict(config.get("database", {}))
    if database_config.get("path"):
        database_config["path"] = str(root / database_config["path"])
    database = QdrantDatabase(**database_config)
    try:
        return SearchService(
            sources, embedder, database, workers=config.get("indexing", {}).get("workers", 2)
        )
    except Exception:
        database.close()
        raise


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Local semantic document search")
    parser.add_argument("--config", default="config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    index = commands.add_parser("index", help="Index configured sources")
    index.add_argument("--source", action="append", dest="source_ids")
    index.add_argument("--rebuild", action="store_true")
    search = commands.add_parser("search", help="Search indexed documents")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--directory")
    search.add_argument("--source", action="append", dest="source_ids")
    commands.add_parser("sources", help="Show source IDs and status")
    commands.add_parser("serve", help="Run the local HTTP API")
    args = parser.parse_args()
    config, root = read_config(args.config)
    try:
        service = build_service(config, root)
        try:
            if args.command == "serve":
                import uvicorn

                from .api import create_app

                server = config.get("server", {})
                uvicorn.run(
                    create_app(service),
                    host=server.get("host", "127.0.0.1"),
                    port=server.get("port", 8000),
                )
                return
            if args.command == "index":
                result = [
                    asdict(item) for item in service.index(args.source_ids, rebuild=args.rebuild)
                ]
            elif args.command == "search":
                result = [
                    asdict(item)
                    for item in service.search(
                        args.query, args.limit, source_ids=args.source_ids, directory=args.directory
                    )
                ]
            else:
                result = service.list_sources()
            print(json.dumps(result, ensure_ascii=False, indent=2))
        finally:
            # API lifespan owns shutdown when running the HTTP server.
            if args.command != "serve":
                service.close()
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
