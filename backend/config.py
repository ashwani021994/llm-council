"""Server-side configuration. No secrets are passed to the browser."""

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path
import re

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
MAX_QUESTION_CHARS = 8_000
MAX_RESPONSE_CHARS = 24_000
MAX_HTTP_RESPONSE_BYTES = 2_000_000
REQUEST_TIMEOUT_SECONDS = 90
RUN_TIMEOUT_SECONDS = 300
MODEL_PATTERN = re.compile(r"[A-Za-z0-9._-]+/[A-Za-z0-9._:/-]+\Z")
DEFAULT_MODELS = (
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1-nano",
)


@dataclass(frozen=True)
class Settings:
    api_key: str = field(default="", repr=False)
    models: tuple[str, ...] = DEFAULT_MODELS
    chairman: str = DEFAULT_MODELS[0]
    max_output_tokens: int = 1_024
    port: int = 7860
    local_password: str = field(default="", repr=False)

    def __post_init__(self):
        if not 2 <= len(self.models) <= 6 or len(set(self.models)) != len(self.models):
            raise ValueError("Configure between 2 and 6 distinct council models.")
        if any(not MODEL_PATTERN.fullmatch(m) or len(m) > 150 for m in (*self.models, self.chairman)):
            raise ValueError("Model names must be OpenRouter identifiers such as openai/gpt-5.1.")
        if not 128 <= self.max_output_tokens <= 4_096:
            raise ValueError("MAX_OUTPUT_TOKENS must be between 128 and 4096.")
        if not 1024 <= self.port <= 65535:
            raise ValueError("LOCAL_PORT must be between 1024 and 65535.")
        if self.local_password and len(self.local_password) < 16:
            raise ValueError("LOCAL_APP_PASSWORD must contain at least 16 characters.")
        if self.api_key and self.local_password == self.api_key:
            raise ValueError("Use a local login password different from your OpenRouter API key.")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    # Read only this project's .env, never a parent directory's file.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    model_list = os.getenv("COUNCIL_MODELS")
    return Settings(
        api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        models=tuple(m.strip() for m in model_list.split(",")) if model_list else DEFAULT_MODELS,
        chairman=os.getenv("CHAIRMAN_MODEL", DEFAULT_MODELS[0]).strip(),
        max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", "1024")),
        port=int(os.getenv("LOCAL_PORT", "7860")),
        local_password=os.getenv("LOCAL_APP_PASSWORD", ""),
    )
