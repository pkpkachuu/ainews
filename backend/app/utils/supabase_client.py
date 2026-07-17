"""Supabase client for database operations.

Uses the Supabase REST API instead of direct PostgreSQL connections,
which works with just the URL and anon key.
"""

from supabase import create_client, Client
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import structlog

from app.config import settings

logger = structlog.get_logger()


class SupabaseDBClient:
    """Database client using Supabase REST API."""

    def __init__(self):
        self.client: Optional[Client] = None

    def connect(self) -> None:
        """Initialize Supabase client."""
        if not settings.supabase_url or not settings.supabase_anon_key:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")

        self.client = create_client(
            settings.supabase_url,
            settings.supabase_anon_key
        )
        logger.info("supabase_connected", url=settings.supabase_url)

    def disconnect(self) -> None:
        """Close connection (no-op for REST client)."""
        self.client = None
        logger.info("supabase_disconnected")

    @staticmethod
    def compute_url_hash(url: str) -> str:
        """Compute SHA256 hash of URL for deduplication."""
        return hashlib.sha256(url.encode()).hexdigest()

    async def article_exists(self, url: str) -> bool:
        """Check if article with URL already exists."""
        url_hash = self.compute_url_hash(url)
        result = self.client.table("articles").select("article_id").eq("url_hash", url_hash).execute()
        return len(result.data) > 0

    async def create_article(self, article_data: Dict[str, Any]) -> str:
        """Insert a new article into the database.

        Returns:
            Article UUID
        """
        url_hash = self.compute_url_hash(article_data["url"])

        insert_data = {
            "url": article_data["url"],
            "url_hash": url_hash,
            "title": article_data.get("title"),
            "body": article_data.get("body"),
            "summary": article_data.get("summary"),
            "source": article_data.get("source"),
            "published_at": article_data.get("published_at"),
            "language": article_data.get("language", "en"),
            "category": article_data.get("category"),
            "sentiment_score": article_data.get("sentiment_score"),
            "sentiment_label": article_data.get("sentiment_label"),
            "word_count": article_data.get("word_count"),
            "raw_storage_path": article_data.get("raw_storage_path"),
            "processing_status": article_data.get("processing_status", "pending"),
        }

        result = self.client.table("articles").insert(insert_data).execute()
        return result.data[0]["article_id"]

    async def update_article_processing(
        self,
        article_id: str,
        processing_data: Dict[str, Any]
    ) -> None:
        """Update article with NLP processing results."""
        update_data = {}

        if processing_data.get("body"):
            update_data["body"] = processing_data["body"]
        if processing_data.get("summary"):
            update_data["summary"] = processing_data["summary"]
        if processing_data.get("category"):
            update_data["category"] = processing_data["category"]
        if processing_data.get("sentiment_score") is not None:
            update_data["sentiment_score"] = processing_data["sentiment_score"]
        if processing_data.get("sentiment_label"):
            update_data["sentiment_label"] = processing_data["sentiment_label"]
        if processing_data.get("word_count"):
            update_data["word_count"] = processing_data["word_count"]
        if processing_data.get("processing_status"):
            update_data["processing_status"] = processing_data["processing_status"]
        if processing_data.get("indexed_at"):
            update_data["indexed_at"] = processing_data["indexed_at"]

        self.client.table("articles").update(update_data).eq("article_id", article_id).execute()

    async def add_article_entities(
        self,
        article_id: str,
        entities: List[Dict[str, Any]]
    ) -> None:
        """Insert entities for an article."""
        for entity in entities:
            self.client.table("article_entities").insert({
                "article_id": article_id,
                "entity_text": entity["text"],
                "entity_type": entity["type"],
                "confidence": entity.get("confidence", 1.0)
            }).execute()

    async def add_article_keyphrases(
        self,
        article_id: str,
        keyphrases: List[Dict[str, Any]]
    ) -> None:
        """Insert keyphrases for an article."""
        for kp in keyphrases:
            self.client.table("article_keyphrases").insert({
                "article_id": article_id,
                "keyphrase": kp["phrase"],
                "score": kp.get("score", 1.0)
            }).execute()

    async def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article by ID with entities and keyphrases."""
        result = self.client.table("articles").select("*").eq("article_id", article_id).execute()

        if not result.data:
            return None

        article = result.data[0]

        # Get entities
        entities_result = self.client.table("article_entities").select(
            "entity_text, entity_type, confidence"
        ).eq("article_id", article_id).execute()
        article["entities"] = entities_result.data

        # Get keyphrases
        keyphrases_result = self.client.table("article_keyphrases").select(
            "keyphrase, score"
        ).eq("article_id", article_id).execute()
        article["keyphrases"] = keyphrases_result.data

        return article

    async def get_pending_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles pending processing."""
        result = self.client.table("articles").select(
            "article_id, url, title, body, source, published_at, raw_storage_path, created_at"
        ).eq("processing_status", "pending").order("created_at").limit(limit).execute()

        return result.data

    async def get_tracked_entities(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all tracked entities."""
        query = self.client.table("tracked_entities").select("*")
        if active_only:
            query = query.eq("is_active", True)
        result = query.execute()
        return result.data

    async def add_tracked_entity(
        self,
        entity_name: str,
        entity_type: str
    ) -> str:
        """Add a new entity to track."""
        result = self.client.table("tracked_entities").upsert({
            "entity_name": entity_name,
            "entity_type": entity_type,
            "is_active": True
        }, on_conflict="entity_name").execute()
        return result.data[0]["id"]

    async def get_entity_volume_history(
        self,
        entity_name: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get article count per day for an entity."""
        # This requires a SQL function or view - simplified for now
        result = self.client.table("article_entities").select(
            "article_id, articles!inner(published_at)"
        ).eq("entity_text", entity_name).execute()
        return result.data

    async def get_articles_by_entity(
        self,
        entity_name: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get the most recent articles mentioning an entity.

        Uses a case-insensitive match: entity_name here often comes from an
        LLM re-extracting a name from a free-text query (e.g. "trump" or
        "Donald Trump"), not the exact casing/form spaCy's NER originally
        tagged and stored - an exact match would silently return nothing on
        any casing mismatch and fall through to a much noisier BM25 search.

        IMPORTANT: article_entities has no timestamp of its own, so we can't
        order/limit at that level and expect recency. The old version did
        `.limit(limit)` directly on article_entities with no order() at all,
        which returns whichever `limit` matching rows Postgres happens to
        return first (effectively insertion order) - meaning the same old
        article could stay "in the top 3" forever regardless of what's
        ingested afterward. Instead, pull a much larger candidate pool of
        article_ids, fetch their real published_at via the articles table,
        sort by that, and only then truncate to `limit`.
        """
        # Pull a generous candidate pool of matching article_ids - offset
        # only applies within this pool, so widen it enough that recent
        # mentions aren't excluded by an early, arbitrary cutoff.
        candidate_pool = max(limit * 20, 200)

        entities_result = self.client.table("article_entities").select(
            "article_id"
        ).ilike("entity_text", entity_name).limit(candidate_pool).execute()

        article_ids = list({e["article_id"] for e in entities_result.data})

        if not article_ids:
            return []

        # Get article details, sorted by actual recency, then apply the
        # real limit/offset here where sorting has already happened.
        articles_result = self.client.table("articles").select(
            "article_id, title, source, published_at, sentiment_score, category"
        ).in_("article_id", article_ids).order(
            "published_at", desc=True
        ).range(offset, offset + limit - 1).execute()

        return articles_result.data

    async def mark_article_failed(self, article_id: str, error: str) -> None:
        """Mark article as failed processing."""
        self.client.table("articles").update({
            "processing_status": "failed"
        }).eq("article_id", article_id).execute()

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        total_result = self.client.table("articles").select("article_id", count="exact").execute()
        total_articles = total_result.count if hasattr(total_result, 'count') else len(total_result.data)

        processed_result = self.client.table("articles").select("article_id", count="exact").eq("processing_status", "done").execute()
        processed_articles = processed_result.count if hasattr(processed_result, 'count') else len(processed_result.data)

        pending_result = self.client.table("articles").select("article_id", count="exact").eq("processing_status", "pending").execute()
        pending_articles = pending_result.count if hasattr(pending_result, 'count') else len(pending_result.data)

        entity_result = self.client.table("article_entities").select("id", count="exact").execute()
        entity_count = entity_result.count if hasattr(entity_result, 'count') else len(entity_result.data)

        tracked_result = self.client.table("tracked_entities").select("id", count="exact").eq("is_active", True).execute()
        tracked_count = tracked_result.count if hasattr(tracked_result, 'count') else len(tracked_result.data)

        return {
            "total_articles": total_articles,
            "processed_articles": processed_articles,
            "pending_articles": pending_articles,
            "entity_count": entity_count,
            "tracked_entities": tracked_count
        }


# Global database client instance
db_client = SupabaseDBClient()
