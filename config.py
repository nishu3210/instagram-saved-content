"""Centralized configuration management for Instagram Analyzer."""

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv  # type: ignore

# Project root directory
project_root = Path(__file__).parent.absolute()

# Load environment variables
env_path = project_root / ".env"
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
class VerificationConfig:
    """Grounded verification provider configuration."""

    provider: str = "tavily_gemini"
    model: str = "gemini-3.1-flash-lite-preview"
    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com/v1"
    tavily_api_key: Optional[str] = None
    max_claims: int = 5
    max_sources: int = 5

    @classmethod
    def from_env(cls) -> "VerificationConfig":
        """Create config from environment variables."""
        provider = os.getenv("VERIFICATION_PROVIDER", cls.provider)
        default_model = (
            os.getenv("GEMINI_MODEL", GeminiConfig.model)
            if provider == "tavily_gemini"
            else "gpt-4.1"
        )
        api_key = os.getenv("VERIFICATION_API_KEY")
        if not api_key and provider == "tavily_gemini":
            api_key = os.getenv("GEMINI_API_KEY")
        elif not api_key:
            api_key = os.getenv("OPENAI_API_KEY")

        return cls(
            provider=provider,
            model=os.getenv("VERIFICATION_MODEL", default_model),
            api_key=api_key,
            base_url=os.getenv("VERIFICATION_BASE_URL", cls.base_url),
            tavily_api_key=os.getenv("TAVILY_API_KEY")
            or os.getenv("VERIFICATION_SEARCH_API_KEY"),
            max_claims=int(os.getenv("VERIFICATION_MAX_CLAIMS", str(cls.max_claims))),
            max_sources=int(
                os.getenv("VERIFICATION_MAX_SOURCES", str(cls.max_sources))
            ),
        )

    def is_configured(self) -> bool:
        """Check if verification config is usable."""
        if self.provider == "tavily_gemini":
            return bool(self.provider and self.api_key and self.tavily_api_key)
        return bool(self.provider and self.api_key)


@dataclass(frozen=True)
class GeminiConfig:
    """Gemini analysis configuration."""

    api_key: Optional[str] = None
    model: str = "gemini-3.1-flash-lite-preview"

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        """Create Gemini config from environment variables."""
        return cls(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=os.getenv("GEMINI_MODEL", cls.model),
        )

    def is_configured(self) -> bool:
        """Check if Gemini is configured."""
        return bool(self.api_key)


@dataclass(frozen=True)
class DatabaseConfig:
    """Database configuration."""

    url: str = field(
        default_factory=lambda: f"sqlite:///{project_root / 'output' / 'instagram_analyzer.db'}"
    )

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Create config from environment variables."""
        url = os.getenv("DATABASE_URL")
        if url:
            return cls(url=url)
        # Use absolute path
        return cls(url=f"sqlite:///{project_root / 'output' / 'instagram_analyzer.db'}")


@dataclass
class AppConfig:
    """Application configuration."""

    debug: bool = False
    port: int = 5001
    host: str = "127.0.0.1"
    secret_key: str = field(
        default_factory=lambda: os.getenv("SECRET_KEY", secrets.token_hex(32))
    )
    max_posts: int = 200
    output_dir: Path = field(default_factory=lambda: project_root / "output")
    rate_limit_storage_uri: str = "memory://"

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create config from environment variables."""
        return cls(
            debug=os.getenv("FLASK_DEBUG", "0") == "1",
            port=int(os.getenv("FLASK_PORT", "5001")),
            host=os.getenv("FLASK_HOST", "127.0.0.1"),
            max_posts=int(os.getenv("MAX_POSTS", "200")),
            rate_limit_storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
        )


class Config:
    """Main configuration container."""

    def __init__(self):
        self.instagram = InstagramConfig.from_env()
        self.azure = AzureConfig.from_env()
        self.gemini = GeminiConfig.from_env()
        self.embedding = EmbeddingConfig.from_env()
        self.verification = VerificationConfig.from_env()
        self.database = DatabaseConfig.from_env()
        self.app = AppConfig.from_env()

    def ensure_directories(self) -> None:
        """Ensure required directories exist."""
        self.app.output_dir.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
