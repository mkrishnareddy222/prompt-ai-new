from collections.abc import Iterator
from typing import Protocol

from domain.models import Message


class LLMProvider(Protocol):
    """Common streaming interface implemented by each LLM adapter."""

    def stream(
        self,
        messages: list[Message],
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> Iterator[str]: ...