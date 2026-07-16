"""Database client for PostgreSQL/Supabase operations."""

import asyncpg
from typing import Optional, List, Dict, Any
from datetime import datetime
import structlog
import hashlib
import json

from app.config import settings

logger = structlog.get_logger()


class DatabaseClient:
    """Async PostgreSQL client wrapper."""

    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Initialize connection pool."""
        self.pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=5,
            max_size=20
        )
        logger.info("database_connected")

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            logger.info("database_disconnected")

    @staticmethod
    def compute_url_hash(url: str) -> str:
        """Compute SHA256 hash of URL for deduplication."""
        return hashlib.sha256(url.encode()).hexdigest()

    async def article_exists(self, url: str) -> bool:
        """Check if article with URL already exists."""
        url_hash = self.compute_url_hash(url)
        async with self.pool.acquire() as conn:
            result = await conn.fetchval(
                "SELECT 1 FROM articles WHERE url_hash = $1",
                url_hash
            )
            return result is not None

    async def create_article(self, article_data: Dict[str, Any]) -> str:
        """Insert a new article into the database.

        Returns:
            Article UUID
        """
        url_hash = self.compute_url_hash(article_data["url"])

        async with self.pool.acquire() as conn:
            article_id = await conn.fetchval(
                """
                INSERT INTO articles (
                    url, url_hash, title, body, summary, source,
                    published_at, language, category, sentiment_score,
                    sentiment_label, word_count, raw_storage_path,
                    processing_status, created_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                RETURNING article_id
                """,
                article_data["url"],
                url_hash,
                article_data.get("title"),
                article_data.get("body"),
                article_data.get("summary"),
                article_data.get("source"),
                article_data.get("published_at"),
                article_data.get("language", "en"),
                article_data.get("category"),
                article_data.get("sentiment_score"),
                article_data.get("sentiment_label"),
                article_data.get("word_count"),
                article_data.get("raw_storage_path"),
                article_data.get("processing_status", "pending"),
                article_data.get("created_at", datetime.utcnow())
            )

        return str(article_id)

    async def update_article_processing(
        self,
        article_id: str,
        processing_data: Dict[str, Any]
    ) -> None:
        """Update article with NLP processing results."""
        async with self.pool.acquire() as conn:
            # Convert embedding list to string format for pgvector
            embedding = processing_data.get("embedding")
            embedding_str = None
            if embedding:
                embedding_str = f"[{','.join(map(str, embedding))}]"

            await conn.execute(
                """
                UPDATE articles SET
                    body = COALESCE($2, body),
                    summary = COALESCE($3, summary),
                    category = COALESCE($4, category),
                    sentiment_score = COALESCE($5, sentiment_score),
                    sentiment_label = COALESCE($6, sentiment_label),
                    word_count = COALESCE($7, word_count),
                    embedding = COALESCE($8::vector, embedding),
                    processing_status = $9,
                    indexed_at = $10
                WHERE article_id = $1
                """,
                article_id,
                processing_data.get("body"),
                processing_data.get("summary"),
                processing_data.get("category"),
                processing_data.get("sentiment_score"),
                processing_data.get("sentiment_label"),
                processing_data.get("word_count"),
                embedding_str,
                processing_data.get("processing_status", "done"),
                processing_data.get("indexed_at", datetime.utcnow())
            )

    async def add_article_entities(
        self,
        article_id: str,
        entities: List[Dict[str, Any]]
    ) -> None:
        """Insert entities for an article."""
        async with self.pool.acquire() as conn:
            for entity in entities:
                await conn.execute(
                    """
                    INSERT INTO article_entities (article_id, entity_text, entity_type, confidence)
                    VALUES ($1, $2, $3, $4)
                    """,
                    article_id,
                    entity["text"],
                    entity["type"],
                    entity.get("confidence", 1.0)
                )

    async def add_article_keyphrases(
        self,
        article_id: str,
        keyphrases: List[Dict[str, Any]]
    ) -> None:
        """Insert keyphrases for an article."""
        async with self.pool.acquire() as conn:
            for kp in keyphrases:
                await conn.execute(
                    """
                    INSERT INTO article_keyphrases (article_id, keyphrase, score)
                    VALUES ($1, $2, $3)
                    """,
                    article_id,
                    kp["phrase"],
                    kp.get("score", 1.0)
                )

    async def get_article(self, article_id: str) -> Optional[Dict[str, Any]]:
        """Get article by ID with entities and keyphrases."""
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT * FROM articles WHERE article_id = $1
                """,
                article_id
            )

            if not row:
                return None

            article = dict(row)

            # Get entities
            entities = await conn.fetch(
                """
                SELECT entity_text, entity_type, confidence
                FROM article_entities WHERE article_id = $1
                """,
                article_id
            )
            article["entities"] = [dict(e) for e in entities]

            # Get keyphrases
            keyphrases = await conn.fetch(
                """
                SELECT keyphrase, score FROM article_keyphrases WHERE article_id = $1
                """,
                article_id
            )
            article["keyphrases"] = [dict(k) for k in keyphrases]

            return article

    async def get_pending_articles(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get articles pending processing."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT article_id, url, title, body, source, published_at,
                       raw_storage_path, created_at
                FROM articles
                WHERE processing_status = 'pending'
                ORDER BY created_at ASC
                LIMIT $1
                """,
                limit
            )
            return [dict(row) for row in rows]

    async def get_all_articles_for_reindex(
        self,
        batch_size: int = 1000
    ) -> Any:
        """Generator yielding all articles for reindexing."""
        async with self.pool.acquire() as conn:
            offset = 0
            while True:
                rows = await conn.fetch(
                    """
                    SELECT a.article_id, a.url, a.title, a.body, a.summary,
                           a.source, a.published_at, a.language, a.category,
                           a.sentiment_score, a.sentiment_label, a.word_count,
                           a.processing_status, a.created_at, a.indexed_at,
                           ARRAY_AGG(DISTINCT jsonb_build_object(
                               'text', ae.entity_text,
                               'type', ae.entity_type,
                               'confidence', ae.confidence
                           )) FILTER (WHERE ae.entity_text IS NOT NULL) as entities,
                           ARRAY_AGG(DISTINCT jsonb_build_object(
                               'phrase', ak.keyphrase,
                               'score', ak.score
                           )) FILTER (WHERE ak.keyphrase IS NOT NULL) as keyphrases
                    FROM articles a
                    LEFT JOIN article_entities ae ON a.article_id = ae.article_id
                    LEFT JOIN article_keyphrases ak ON a.article_id = ak.article_id
                    GROUP BY a.article_id
                    ORDER BY a.created_at
                    OFFSET $1 LIMIT $2
                    """,
                    offset, batch_size
                )

                if not rows:
                    break

                for row in rows:
                    article = dict(row)
                    # Parse entities and keyphrases from JSON
                    article["entities"] = article["entities"] or []
                    article["keyphrases"] = [kp["phrase"] for kp in (article["keyphrases"] or [])]
                    yield article

                offset += batch_size

    async def get_tracked_entities(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Get all tracked entities."""
        async with self.pool.acquire() as conn:
            query = "SELECT * FROM tracked_entities"
            if active_only:
                query += " WHERE is_active = true"
            rows = await conn.fetch(query)
            return [dict(row) for row in rows]

    async def add_tracked_entity(
        self,
        entity_name: str,
        entity_type: str
    ) -> str:
        """Add a new entity to track."""
        async with self.pool.acquire() as conn:
            entity_id = await conn.fetchval(
                """
                INSERT INTO tracked_entities (entity_name, entity_type)
                VALUES ($1, $2)
                ON CONFLICT (entity_name) DO UPDATE SET is_active = true
                RETURNING id
                """,
                entity_name, entity_type
            )
            return str(entity_id)

    async def get_entity_volume_history(
        self,
        entity_name: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        """Get article count per day for an entity."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DATE(a.published_at) as date, COUNT(*) as count,
                       AVG(a.sentiment_score) as avg_sentiment
                FROM articles a
                JOIN article_entities ae ON a.article_id = ae.article_id
                WHERE ae.entity_text = $1
                    AND a.published_at >= NOW() - INTERVAL '%s days'
                GROUP BY DATE(a.published_at)
                ORDER BY date
                """ % days,
                entity_name
            )
            return [dict(row) for row in rows]

    async def get_articles_by_entity(
        self,
        entity_name: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get articles mentioning an entity."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT a.article_id, a.title, a.source, a.published_at,
                       a.sentiment_score, a.category
                FROM articles a
                JOIN article_entities ae ON a.article_id = ae.article_id
                WHERE ae.entity_text = $1
                ORDER BY a.published_at DESC
                LIMIT $2 OFFSET $3
                """,
                entity_name, limit, offset
            )
            return [dict(row) for row in rows]

    async def mark_article_failed(self, article_id: str, error: str) -> None:
        """Mark article as failed processing."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE articles
                SET processing_status = 'failed'
                WHERE article_id = $1
                """,
                article_id
            )

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with self.pool.acquire() as conn:
            total_articles = await conn.fetchval("SELECT COUNT(*) FROM articles")
            processed_articles = await conn.fetchval(
                "SELECT COUNT(*) FROM articles WHERE processing_status = 'done'"
            )
            pending_articles = await conn.fetchval(
                "SELECT COUNT(*) FROM articles WHERE processing_status = 'pending'"
            )
            entity_count = await conn.fetchval("SELECT COUNT(*) FROM article_entities")
            tracked_count = await conn.fetchval(
                "SELECT COUNT(*) FROM tracked_entities WHERE is_active = true"
            )

            return {
                "total_articles": total_articles,
                "processed_articles": processed_articles,
                "pending_articles": pending_articles,
                "entity_count": entity_count,
                "tracked_entities": tracked_count
            }


# Global database client instance
db_client = DatabaseClient()
