from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .schemas import IndexBusyError, IndexMismatchError
from .service import SearchService


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=10000)
    limit: int = Field(default=5, ge=1, le=100)
    source_ids: list[str] | None = None
    directory: str | None = None


class IndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_ids: list[str] | None = None
    rebuild: bool = False


def create_app(service: SearchService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.close()

    app = FastAPI(title="Semantic Search", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(IndexMismatchError)
    @app.exception_handler(IndexBusyError)
    async def conflict(request: Request, error: Exception):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def invalid(request: Request, error: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(OSError)
    async def unavailable(request: Request, error: OSError):
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/sources")
    def sources():
        return service.list_sources()

    @app.post("/index")
    def index(body: IndexRequest):
        """Waits for indexing to finish; sources are processed concurrently."""
        return [asdict(report) for report in service.index(body.source_ids, rebuild=body.rebuild)]

    @app.post("/search")
    def search(body: SearchRequest):
        return [
            asdict(result)
            for result in service.search(
                body.query, body.limit, source_ids=body.source_ids, directory=body.directory
            )
        ]

    return app
