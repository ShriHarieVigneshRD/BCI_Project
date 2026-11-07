"""FastAPI application factory and local entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import APP_VERSION, settings
from .core.preprocessing import PreprocessingError
from .core.splits import SplitError
from .lifecycle import TransitionError
from .data.importer import DatasetImportError
from .routers import datasets, experiments, exports, imports, models, system
from .services import repo
from .services.artifacts import ArtifactError
from .services.datasets import UploadRejected
from .services.experiments import ConfigError, WorkflowError
from .services.imports import ImportConflict, ImportRejected, recover_interrupted
from .services.jobs import manager, recover_orphans


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    repo.db()
    recover_orphans()
    recover_interrupted()
    manager.start()
    yield


def _err(status: int, kind: str, message: str, **extra) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": message, "kind": kind, **extra})


def create_app() -> FastAPI:
    app = FastAPI(title="EEGNet Architecture Playground API", version=APP_VERSION, lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False,
                       allow_methods=["*"], allow_headers=["*"])

    @app.exception_handler(WorkflowError)
    async def _workflow(_r: Request, exc: WorkflowError):
        return _err(409, "workflow", str(exc))

    @app.exception_handler(TransitionError)
    async def _transition(_r: Request, exc: TransitionError):
        return _err(409, "lifecycle", str(exc), current=exc.current.value, target=exc.target.value)

    @app.exception_handler(ArtifactError)
    async def _artifact(_r: Request, exc: ArtifactError):
        return _err(409, "artifacts", str(exc))

    @app.exception_handler(ConfigError)
    async def _config(_r: Request, exc: ConfigError):
        return _err(422, "config", str(exc), details=exc.details)

    @app.exception_handler(UploadRejected)
    async def _upload(_r: Request, exc: UploadRejected):
        return _err(422, "upload", str(exc) or "Upload rejected.", issues=exc.issues, format=exc.format)

    @app.exception_handler(ImportRejected)
    async def _imp_rej(_r: Request, exc: ImportRejected):
        return _err(422, "import", str(exc))

    @app.exception_handler(DatasetImportError)
    async def _imp_err(_r: Request, exc: DatasetImportError):
        return _err(422, "import", str(exc))

    @app.exception_handler(ImportConflict)
    async def _imp_conf(_r: Request, exc: ImportConflict):
        return _err(409, "import", str(exc))

    @app.exception_handler(SplitError)
    async def _split(_r: Request, exc: SplitError):
        return _err(422, "split", str(exc))

    @app.exception_handler(PreprocessingError)
    async def _prep(_r: Request, exc: PreprocessingError):
        return _err(422, "preprocessing", str(exc))

    @app.exception_handler(repo.NotFound)
    async def _nf(_r: Request, exc: repo.NotFound):
        return _err(404, "not_found", str(exc.args[0]) if exc.args else "Not found")

    @app.exception_handler(KeyError)
    async def _key(_r: Request, exc: KeyError):
        return _err(404, "not_found", str(exc.args[0]) if exc.args else "Not found")

    @app.exception_handler(ValueError)
    async def _val(_r: Request, exc: ValueError):
        return _err(422, "invalid", str(exc))

    for r in (system.router, datasets.router, imports.router, models.router, experiments.router,
              exports.router):
        app.include_router(r, prefix="/api")

    dist = settings.frontend_dist
    if (dist / "index.html").exists():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        async def spa(path: str):
            if path.startswith("api/"):
                return _err(404, "not_found", "Unknown API route")
            f = (dist / path).resolve()
            if path and f.is_file() and dist.resolve() in f.parents:
                return FileResponse(f)
            return FileResponse(dist / "index.html")

    return app


app = create_app()


def run() -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
