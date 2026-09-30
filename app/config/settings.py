import os

from dotenv import load_dotenv


load_dotenv()


class Settings:
    """Application configuration."""

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "groq").lower()

    GROQ_API_KEY: str = os.getenv(
        "GROQ_API_KEY",
        "",
    )

    OPENAI_API_KEY: str = os.getenv(
        "OPENAI_API_KEY",
        "",
    )

    GEMINI_API_KEY: str = os.getenv(
        "GEMINI_API_KEY",
        "",
    )

    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-20b",
    )

    OPENAI_MODEL: str = os.getenv(
        "OPENAI_MODEL",
        "gpt-6-luna",
    )

    GEMINI_MODEL: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    )

    # =========================================================
    # LLM Configuration
    # =========================================================

    MAX_TOKENS: int = int(
        os.getenv(
            "MAX_TOKENS",
            "1024",
        )
    )

    TEMPERATURE: float = float(
        os.getenv(
            "TEMPERATURE",
            "0.3",
        )
    )

    TOP_P: float = float(
        os.getenv(
            "TOP_P",
            "1.0",
        )
    )

    # =========================================================
    # Chat Configuration
    # =========================================================

    MAX_HISTORY: int = int(
        os.getenv(
            "MAX_HISTORY",
            "20",
        )
    )

    SYSTEM_PROMPT: str = (
        "You are Emma, a concise AI assistant. "
        "If you are unsure about recent events, "
        "say so clearly."
    )

    def provider_credentials(
        self,
        provider: str,
        api_key_override: str | None = None,
    ) -> tuple[str, str]:
        provider = provider.lower()
        credentials = {
            "groq": (self.GROQ_API_KEY, self.GROQ_MODEL, "GROQ_API_KEY"),
            "openai": (self.OPENAI_API_KEY, self.OPENAI_MODEL, "OPENAI_API_KEY"),
            "gemini": (self.GEMINI_API_KEY, self.GEMINI_MODEL, "GEMINI_API_KEY"),
        }

        if provider not in credentials:
            raise ValueError(f"Unsupported LLM provider: {provider}")

        configured_key, model, key_name = credentials[provider]
        api_key = (api_key_override or "").strip() or configured_key
        if not api_key or api_key.startswith("paste_"):
            raise ValueError(
                f"Enter a token or configure {key_name} to use {provider.title()}."
            )

        return api_key, model

    def provider_model(self, provider: str) -> str:
        return {
            "groq": self.GROQ_MODEL,
            "openai": self.OPENAI_MODEL,
            "gemini": self.GEMINI_MODEL,
        }[provider.lower()]


settings = Settings()
