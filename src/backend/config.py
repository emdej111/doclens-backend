"""Centralized configuration loaded from environment variables / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"

    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_url: str = "http://localhost:8000"

    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k_chunks: int = 4

    max_summary_chars: int = 60_000

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
