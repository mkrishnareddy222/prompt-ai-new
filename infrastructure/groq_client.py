from collections.abc import Iterator

from groq import Groq

from domain.models import Message


class GroqClient:
    """Infrastructure adapter for the Groq API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = Groq(
            api_key=api_key
        )
        self.model = model

    def stream(
        self,
        messages: list[Message],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]:

        payload = [
            message.to_dict()
            for message in messages
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=payload,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stream=True,
        )

        for chunk in response:

            if not chunk.choices:
                continue

            content = chunk.choices[0].delta.content

            if content:
                yield content
