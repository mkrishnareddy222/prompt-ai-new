from collections.abc import Iterator

from openai import OpenAI

from domain.models import Message


class OpenAIClient:
    """Infrastructure adapter for the OpenAI chat completions API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def stream(
        self,
        messages: list[Message],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[message.to_dict() for message in messages],
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