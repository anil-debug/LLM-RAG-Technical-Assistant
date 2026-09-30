"""FastAPI application."""

from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes.chat import router as chat_router
from api.routes.documents import router as documents_router
from api.routes.health import router as health_router
from core import __version__
from core.context import reset_request_id, set_request_id
from core.errors import (
    AppError,
    ClassifierNotReadyError,
    DatabaseUnavailableError,
    DocumentNotFoundError,
    EmptyDocumentError,
    ModelUnavailableError,
    UnsupportedMediaTypeError,
    UploadTooLargeError,
)
from core.logging_config import setup_logging
from api.deps import Container, build_container

_STATUS = {
    DocumentNotFoundError: (404, "not_found"),
    UnsupportedMediaTypeError: (415, "unsupported_media_type"),
    UploadTooLargeError: (413, "payload_too_large"),
    EmptyDocumentError: (422, "empty_document"),
    ModelUnavailableError: (503, "model_unavailable"),
    DatabaseUnavailableError: (503, "database_unavailable"),
    ClassifierNotReadyError: (409, "classifier_not_ready"),
}


def create_app(container: Container | None = None) -> FastAPI:
    """Build the API. Pass a container in tests to avoid real model servers."""
    app = FastAPI(title="Technical RAG Assistant", version=__version__)
    app.state.container = container or build_container()
    setup_logging(app.state.container.settings.log_level)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(chat_router)

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid4().hex
        token = set_request_id(request_id)
        try:
            response = await call_next(request)
        finally:
            reset_request_id(token)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(_request: Request, exc: AppError) -> JSONResponse:
        status, code = _STATUS.get(type(exc), (400, "bad_request"))
        return JSONResponse(status_code=status, content={"error": {"code": code, "message": str(exc)}})

    return app


app = create_app()
