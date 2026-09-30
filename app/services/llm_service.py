from collections.abc import Iterator

from groq import Groq

from app.config.settings import settings


class LLMService:
    """Service responsible for communicating with Groq."""

    def __init__(self) -> None:
        self.client = Groq(
            api_key=settings.GROQ_API_KEY
        )

    def stream_response(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Iterator[str]:
        """
        Stream the assistant response from Groq.
        """

        stream = self.client.chat.completions.create(
            model=settings.MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content

            if content:
                yield content
