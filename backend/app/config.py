"""Application configuration using Pydantic Settings."""

from pydantic_settings import BaseSettings
from typing import Optional
import os


def get_database_url_from_supabase(supabase_url: str) -> Optional[str]:
    """Convert Supabase URL to PostgreSQL connection string.

    Supabase URLs look like: https://[project-ref].supabase.co
    Database URLs look like: postgresql://postgres.[ref]:[password]@aws-0-[region].pooler.supabase.com:6543/postgres
    """
    if not supabase_url:
        return None

    # Extract project ref from URL
    # https://fahknapjoqocoprkptar.supabase.co -> fahknapjoqocoprkptar
    try:
        import re
        match = re.search(r'https://([a-z]+)\.supabase\.co', supabase_url)
        if match:
            project_ref = match.group(1)
            # Return template - user needs to add password
            return f"postgresql://postgres.{project_ref}:[YOUR-PASSWORD]@aws-0-us-east-1.pooler.supabase.com:6543/postgres"
    except Exception:
        pass
    return None


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Supabase / PostgreSQL
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    database_url: str = ""

    # Redis
    redis_url: str = "redis://localhost:6379"

    # Elasticsearch
    elasticsearch_url: str = "http://localhost:9200"

    # Groq LLM
    groq_api_key: Optional[str] = None

    # News APIs
    newsapi_key: Optional[str] = None

    # App settings
    environment: str = "development"
    log_level: str = "INFO"

    # Processing settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dims: int = 384
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Intelligence feed settings
    emergence_check_interval_minutes: int = 30
    narrative_check_interval_minutes: int = 60
    feed_generation_hours: list[int] = [6, 18]  # 6 AM and 6 PM

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra env vars

    def __post_init__(self):
        """Validate and derive settings after initialization."""
        # If database_url not set, try to derive from supabase_url
        if not self.database_url and self.supabase_url:
            derived = get_database_url_from_supabase(self.supabase_url)
            if derived:
                self.database_url = derived

    @property
    def is_configured(self) -> bool:
        """Check if minimum required settings are present."""
        return bool(self.supabase_url and self.supabase_anon_key)


# Global settings instance
settings = Settings()
