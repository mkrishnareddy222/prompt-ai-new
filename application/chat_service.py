from collections.abc import Iterator

from app.config.settings import settings
from application.llm_provider import LLMProvider
from domain.models import ChatRequest, Message
from app.prompts.prompts import PromptService


class ChatService:
    """
    Application service responsible for chatbot behavior.

    Responsibilities:
    - Build the LLM message payload
    - Apply conversation history
    - Limit conversation history
    - Delegate LLM generation to the configured provider
    """

    def __init__(self, llm_client: LLMProvider) -> None:
        self.llm_client = llm_client

    def build_messages(
        self,
        request: ChatRequest,
        history: list[Message],
    ) -> list[Message]:
        """
        Build the messages that will be sent to the LLM.
        """

        messages = [PromptService.system_message()]

        # Add previous conversation only when
        # "Remember conversation" is enabled.
        if request.remember:
            messages.extend(
                self._trim_history(history)
            )

        # Always add the current user message.
        messages.append(
            PromptService.user_message(request.message)
        )

        return messages

    def stream(
        self,
        request: ChatRequest,
        history: list[Message],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """
        Generate a streaming response from the LLM.
        """

        messages = self.build_messages(
            request=request,
            history=history,
        )

        yield from self.llm_client.stream(
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )

    @staticmethod
    def _trim_history(
        history: list[Message],
    ) -> list[Message]:
        """
        Limit conversation history according to configuration.
        """

        return history[-settings.MAX_HISTORY:]
