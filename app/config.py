"""Runtime configuration, read once from environment variables (see .env.example)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date


def _env(name: str, default: str = "") -> str:
    value = os.environ.get(name)
    return value if value not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    corpus_dir: str = field(default_factory=lambda: _env("CORPUS_DIR", "./data/corpus"))
    manifest_path: str = field(default_factory=lambda: _env("MANIFEST_PATH", "./data/manifest.json"))
    db_path: str = field(default_factory=lambda: _env("DB_PATH", "./data/veltrix.db"))
    index_path: str = field(default_factory=lambda: _env("INDEX_PATH", "./.index/index.json"))
    app_today: str = field(default_factory=lambda: _env("APP_TODAY", ""))

    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "openai").lower())
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", ""))
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_SECONDS", "20")))

    # Hard ceiling from the brief is 3,000 cl100k tokens of corpus text per LLM call.
    # We pack to a slightly lower target so headers/separators can never push us over.
    corpus_token_budget: int = field(default_factory=lambda: int(_env("CORPUS_TOKEN_BUDGET", "2800")))
    corpus_token_hard_limit: int = 3000

    ocr_lang: str = field(default_factory=lambda: _env("OCR_LANG", "eng"))
    log_level: str = field(default_factory=lambda: _env("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: _env("LOG_FILE", ""))

    def today(self) -> date:
        """APP_TODAY if set, otherwise the system date."""
        if self.app_today:
            return date.fromisoformat(self.app_today)
        return date.today()


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Used by tests after changing environment variables."""
    global _settings
    _settings = None
