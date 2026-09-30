from dataclasses import dataclass
from typing import Literal


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


@dataclass
class ChatRequest:
    message: str
    remember: bool = True


@dataclass
class ChatResponse:
    content: str
