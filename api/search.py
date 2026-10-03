import logging

from fastapi import APIRouter
from fastapi import HTTPException
from pydantic import BaseModel, Field

from application.chat_service import ChatService
from app.config.settings import settings
from domain.models import ChatRequest, Message
from infrastructure.provider_factory import create_llm_provider

router = APIRouter(prefix="/api", tags=["search"])
logger = logging.getLogger(__name__)


class SearchRequest(BaseModel):
    message: str
    remember: bool = True
    provider: str = settings.LLM_PROVIDER
    api_key_override: str | None = None
    temperature: float = settings.TEMPERATURE
    top_p: float = settings.TOP_P
    max_tokens: int = settings.MAX_TOKENS
    history: list[Message] = Field(default_factory=list)


class SearchResponse(BaseModel):
    role: str = "assistant"
    content: str
    message: str
    remember: bool


def get_chat_service(provider: str, api_key_override: str | None = None) -> ChatService:
    llm_client = create_llm_provider(
        provider,
        api_key_override=api_key_override,
    )
    return ChatService(llm_client=llm_client)


@router.post("/search")
async def search(request: SearchRequest):
    logged_request = request.model_dump(exclude={"api_key_override"})
    logger.info("Received POST /api/search request: %s", logged_request)

    try:
        chat_service = get_chat_service(
            provider=request.provider,
            api_key_override=request.api_key_override,
        )

        history = [
            Message(
                role=item.role,
                content=item.content,
            )
            for item in request.history
        ]

        reply = "".join(
            chat_service.stream(
                request=ChatRequest(
                    message=request.message,
                    remember=request.remember,
                ),
                history=history,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )
        )
    except Exception as exc:
        logger.exception(
            "POST /api/search failed; sanitized request: %s",
            logged_request,
        )
        raise HTTPException(
            status_code=502,
            detail="The chat provider rejected or could not process the request. "
            "Check the API server logs for details.",
        ) from exc

    return SearchResponse(
        role="assistant",
        content=reply,
        message=request.message,
        remember=request.remember,
    ).model_dump()
