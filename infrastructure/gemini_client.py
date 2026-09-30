from collections.abc import Iterator

from google import genai
from google.genai import types

from domain.models import Message


class GeminiClient:
    """Infrastructure adapter for Google's Gemini generate content API."""

    def __init__(self, api_key: str, model: str) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def stream(
        self,
        messages: list[Message],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]:
        system_instruction = "\n\n".join(
            message.content
            for message in messages
            if message.role == "system"
        )
        contents = [
            types.Content(
                role="model" if message.role == "assistant" else "user",
                parts=[types.Part.from_text(text=message.content)],
            )
            for message in messages
            if message.role != "system"
        ]
        response = self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction or None,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_tokens,
            ),
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text