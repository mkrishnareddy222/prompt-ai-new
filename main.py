import logging

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config.settings import settings
from api.search import router as search_router

logger = logging.getLogger(__name__)

app = FastAPI(title="Emma AI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["POST"],
    allow_headers=["*"],
)
app.include_router(search_router)


def _redact_sensitive_fields(value):
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if key.lower() in {"api_key_override", "authorization", "api_key", "token"}
                else _redact_sensitive_fields(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_fields(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    body = exc.body
    if not isinstance(body, (dict, list)):
        body = "[unavailable: request body was invalid or not JSON]"

    logger.warning(
        "Request validation failed: method=%s path=%s errors=%s sanitized_body=%s",
        request.method,
        request.url.path,
        [
            {
                "loc": error.get("loc"),
                "msg": error.get("msg"),
                "type": error.get("type"),
            }
            for error in exc.errors()
        ],
        _redact_sensitive_fields(body),
    )
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors())},
    )
