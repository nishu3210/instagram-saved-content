"""Centralized configuration management for Instagram Analyzer."""

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv  # type: ignore

# Load environment variables
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)


@dataclass(frozen=True)
class InstagramConfig:
    """Instagram API configuration."""

    sessionid: Optional[str] = None
    raw_cookie: Optional[str] = None
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    )
    browser: str = "none"

    @classmethod
    def from_env(cls) -> "InstagramConfig":
        """Create config from environment variables."""
        return cls(
            sessionid=os.getenv("INSTAGRAM_SESSIONID"),
            raw_cookie=os.getenv("RAW_COOKIE"),
            user_agent=os.getenv("USER_AGENT", cls.user_agent),
            browser=os.getenv("BROWSER", "none"),
        )


@dataclass(frozen=True)
class AzureConfig:
    """Azure OpenAI configuration."""

    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    api_version: str = "2024-08-01-preview"
    model: str = "DeepSeek-V3.2"

    @classmethod
    def from_env(cls) -> "AzureConfig":
        """Create config from environment variables."""
        return cls(
            api_key=os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_API_BASE") or os.getenv("AZURE_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", cls.api_version),
            model=os.getenv("MODEL", cls.model),
        )

    def is_configured(self) -> bool:
        """Check if Azure is properly configured."""
        return bool(self.api_key and self.endpoint)


@dataclass(frozen=True)
class EmbeddingConfig:
    """Embedding model configuration."""

    model: str = "text-embedding-3-large"
    dimension: int = 3072

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create config from environment variables."""
        return cls(
            model=os.getenv("EMBEDDING_MODEL", cls.model),
            dimension=int(os.getenv("EMBEDDING_DIM", cls.dimension)),
        )


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""

    url: str = "sqlite:///output/instagram_analyzer.db"

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        return cls(url=os.getenv("DATABASE_URL", cls.url))


@dataclass
class AppConfig:
    """Application configuration."""

    debug: bool = False
    port: int = 5001
    host: str = "127.0.0.1"
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", secrets.token_hex(32))
    )
    max_posts: int = 20
    output_dir: Path = field(default_factory=lambda: Path("output"))

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables."""
        return cls(
            debug=os.getenv("FLASK_DEBUG", "0") == "1",
            port=int(os.getenv("FLASK_PORT", "5001")),
            host=os.getenv("FLASK_HOST", "127.0.0.1"),
            max_posts=int(os.getenv("MAX_POSTS", "20")),
        )


class Config:
    """Main configuration container."""

    def __init__(self):
        self.instagram = InstagramConfig.from_env()
        self.azure = AzureConfig.from_env()
        self.embedding = EmbeddingConfig.from_env()
        self.database = DatabaseConfig.from_env()
        self.app = AppConfig.from_env()

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.app.output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
