"""Utility modules for the News Analytics Platform."""

from app.utils.redis_client import redis_client
from app.utils.elasticsearch_client import es_client
from app.utils.supabase_client import db_client

__all__ = ["redis_client", "es_client", "db_client"]
