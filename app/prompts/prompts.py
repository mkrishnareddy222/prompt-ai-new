from app.config.settings import settings
from domain.models import Message


class PromptService:
    """Responsible for constructing prompts."""

    @staticmethod
    def system_message() -> Message:
        return Message(
            role="system",
            content=settings.SYSTEM_PROMPT,
        )

    @staticmethod
    def user_message(prompt: str) -> Message:
        return Message(
            role="user",
            content=prompt,
        )
