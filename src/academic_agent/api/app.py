"""FastAPI application factory for Academic-Agent serving layer."""

import logging
from typing import Optional, Sequence

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from academic_agent.api.dependencies import set_default_agent_service
from academic_agent.api.routes import router
from academic_agent.service.protocols import AgentServiceProtocol

logger = logging.getLogger(__name__)


def create_app(
    service: Optional[AgentServiceProtocol] = None,
    allowed_origins: Optional[Sequence[str]] = None,
    title: str = "Academic-Agent Serving API",
    version: str = "0.1.0",
) -> FastAPI:
    """Constructs and configures the FastAPI serving application.

    Args:
        service: Optional AgentService instance to register as the default provider.
        allowed_origins: Optional list of authorized CORS origins. Defaults to unrestricted local/dev.
        title: API application title for OpenAPI documentation.
        version: Application semantic version.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=title,
        version=version,
        description="Serving layer and SSE event streaming API for Academic-Agent student research workstation.",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. Register service dependency if provided
    if service is not None:
        set_default_agent_service(service)

    # 2. CORS Middleware
    if allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_origins),
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    else:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 3. Custom Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        logger.warning("API validation error on %s: %s", request.url.path, exc.errors())
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "error": "Unprocessable Entity",
                "detail": exc.errors(),
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail if isinstance(exc.detail, str) else "HTTP Exception",
                "detail": exc.detail,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server exception on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
            },
        )

    # 4. Attach Routers
    app.include_router(router)

    return app


__all__ = ["create_app"]
