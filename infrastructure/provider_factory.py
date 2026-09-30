from collections.abc import Callable

from application.llm_provider import LLMProvider
from app.config.settings import settings
from infrastructure.gemini_client import GeminiClient
from infrastructure.groq_client import GroqClient
from infrastructure.openai_client import OpenAIClient


PROVIDERS: dict[str, Callable[..., LLMProvider]] = {
    "groq": GroqClient,
    "openai": OpenAIClient,
    "gemini": GeminiClient,
}


def create_llm_provider(
    provider: str,
    api_key_override: str | None = None,
) -> LLMProvider:
    """Construct the configured infrastructure adapter for a provider."""
    provider = provider.strip().lower()

    try:
        client_factory = PROVIDERS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM provider: {provider}") from exc

    api_key, model = settings.provider_credentials(
        provider,
        api_key_override=api_key_override,
    )
    return client_factory(api_key=api_key, model=model)