"""Temporal Analytics Engine.

Computes trend metrics, spike detection, and daily digests.
Schedules regular computation and caches results in Redis.
"""

import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter
import structlog
import numpy as np

from app.config import settings
from app.utils.redis_client import redis_client
from app.utils.elasticsearch_client import es_client
from app.utils.supabase_client import db_client

logger = structlog.get_logger()

# Redis cache keys
TRENDING_KEY = "trending:topics"
DIGEST_KEY = "daily:digest"
ENTITY_TREND_KEY = "entity:trend:{}"
CACHE_TTL = 3600  # 1 hour


class TemporalAnalytics:
    """Computes temporal trends and analytics."""

    async def get_entity_trend(
        self,
        entity: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get trend data for an entity over time.

        Returns volume, sentiment trend, and spike detection.
        """
        cache_key = ENTITY_TREND_KEY.format(entity)

        # Check cache
        cached = await redis_client.get(cache_key)
        if cached:
            return cached

        # Get volume over time
        volume_data = await es_client.get_volume_over_time(entity=entity, days=days)

        if not volume_data:
            return {
                "entity": entity,
                "error": "No data available"
            }

        # Compute rolling averages and detect spikes
        processed = self._process_volume_data(volume_data)

        # Get recent articles
        recent = await db_client.get_articles_by_entity(entity, limit=20)

        result = {
            "entity": entity,
            "days": days,
            "volume": processed["volume"],
            "stats": processed["stats"],
            "spike_detected": processed["spike_detected"],
            "spike_ratio": processed["spike_ratio"],
            "trend_direction": processed["trend_direction"],
            "recent_articles": recent[:10]
        }

        # Cache result
        await redis_client.set(cache_key, result, ttl=CACHE_TTL)

        return result

    async def get_topic_trend(
        self,
        topic: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get trend data for a topic/keyword over time."""
        volume_data = await es_client.get_volume_over_time(topic=topic, days=days)

        if not volume_data:
            return {
                "topic": topic,
                "error": "No data available"
            }

        processed = self._process_volume_data(volume_data)

        return {
            "topic": topic,
            "days": days,
            "volume": processed["volume"],
            "stats": processed["stats"],
            "spike_detected": processed["spike_detected"],
            "spike_ratio": processed["spike_ratio"],
            "trend_direction": processed["trend_direction"]
        }

    def _process_volume_data(
        self,
        volume_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process volume data to compute trends and spikes."""
        counts = [d["count"] for d in volume_data]
        sentiments = [d.get("avg_sentiment", 0) for d in volume_data]

        # Compute rolling 7-day average
        window_size = min(7, len(counts))
        rolling_avg = np.convolve(
            counts,
            np.ones(window_size) / window_size,
            mode='valid'
        ) if len(counts) >= window_size else [np.mean(counts)]

        # Compute spike ratio (today vs rolling average)
        today_count = counts[-1] if counts else 0
        baseline = rolling_avg[-1] if len(rolling_avg) > 0 and rolling_avg[-1] > 0 else 1
        spike_ratio = today_count / baseline

        # Classify trend
        spike_detected = spike_ratio > 3.0
        if spike_ratio > 3.0:
            trend_direction = "SPIKE"
        elif spike_ratio > 1.5:
            trend_direction = "RISING"
        elif spike_ratio < 0.5:
            trend_direction = "FALLING"
        else:
            trend_direction = "STABLE"

        # Sentiment trend
        valid_sentiments = [s for s in sentiments if s is not None]
        if len(valid_sentiments) >= 7:
            recent_sentiment = np.mean(valid_sentiments[-3:])
            prior_sentiment = np.mean(valid_sentiments[:-3])
            sentiment_trend = "improving" if recent_sentiment > prior_sentiment else "declining" if recent_sentiment < prior_sentiment else "stable"
        else:
            sentiment_trend = "unknown"

        return {
            "volume": volume_data,
            "stats": {
                "total_articles": int(sum(counts)),
                "avg_daily": float(round(np.mean(counts), 1)),
                "max_daily": int(max(counts)),
                "avg_sentiment": float(round(np.mean([s for s in sentiments if s is not None]), 3)) if sentiments else 0.0
            },
            "spike_detected": bool(spike_detected),
            "spike_ratio": float(round(spike_ratio, 2)),
            "trend_direction": trend_direction,
            "sentiment_trend": sentiment_trend
        }

    async def get_trending_entities(
        self,
        top_n: int = 20,
        min_articles: int = 5
    ) -> List[Dict[str, Any]]:
        """Get entities with highest velocity (change rate).

        Uses Elasticsearch aggregation for efficiency.
        """
        # Check cache
        cached = await redis_client.get(TRENDING_KEY)
        if cached:
            return cached[:top_n]

        # Compute from scratch
        result = await es_client.client.search(
            index="articles",
            body={
                "query": {
                    "range": {
                        "published_at": {"gte": "now-7d"}
                    }
                },
                "aggs": {
                    "top_entities": {
                        "nested": {"path": "entities"},
                        "aggs": {
                            "entity_names": {
                                "terms": {
                                    "field": "entities.text",
                                    "size": 50,
                                    "min_doc_count": min_articles
                                },
                                "aggs": {
                                    "avg_sentiment": {"avg": {"field": "sentiment_score"}},
                                    "recent": {
                                        "filter": {
                                            "range": {
                                                "published_at": {"gte": "now-24h"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "size": 0
            }
        )

        entities = []
        buckets = result["aggregations"]["top_entities"]["entity_names"]["buckets"]

        ENTITY_BLACKLIST = {
            "us", "uk", "un", "eu", "america", "american", "americans",
            "bbc", "cnn", "reuters", "guardian", "the guardian", "ap",
            "associated press", "afp", "euronews", "wired", "techcrunch",
            "ars technica", "new york", "washington"
        }

        for bucket in buckets:
            entity_name = bucket["key"]

            if entity_name.strip().lower() in ENTITY_BLACKLIST:
                continue
            if len(entity_name.strip()) < 3:
                continue

            total_count = bucket["doc_count"]
            recent_count = bucket["recent"]["doc_count"]
            avg_sentiment = bucket["avg_sentiment"].get("value", 0)

            if total_count > 0:
                velocity = (recent_count / total_count) * 100
            else:
                velocity = 0

            entities.append({
                "entity": entity_name,
                "total_articles": total_count,
                "recent_articles": recent_count,
                "velocity": round(velocity, 1),
                "avg_sentiment": round(avg_sentiment, 3) if avg_sentiment else 0
            })

        entities.sort(key=lambda x: x["velocity"], reverse=True)

        # Cache
        await redis_client.set(TRENDING_KEY, entities, ttl=CACHE_TTL)

        return entities[:top_n]

    async def compute_daily_digest(self) -> Dict[str, Any]:
        """Compute daily digest of top stories.

        Runs on schedule (6 AM and 6 PM) and caches result.
        """
        # Get trending entities
        trending = await self.get_trending_entities(top_n=30)

        # Get articles from past 24h
        recent_articles = await es_client.get_recent_articles(hours=24, size=200)

        # Compute category breakdown
        categories = Counter(a.get("category", "unknown") for a in recent_articles)

        # Compute overall sentiment
        sentiments = [a.get("sentiment_score", 0) for a in recent_articles if a.get("sentiment_score") is not None]
        avg_sentiment = np.mean(sentiments) if sentiments else 0

        # Detect spikes
        spikes = []
        for entity_data in trending:
            if entity_data["velocity"] > 200:  # High velocity threshold
                # Get articles for this entity
                articles = await db_client.get_articles_by_entity(
                    entity_data["entity"],
                    limit=5
                )
                spikes.append({
                    "entity": entity_data["entity"],
                    "velocity": entity_data["velocity"],
                    "articles": articles
                })

        digest = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "period": "24h",
            "total_articles": len(recent_articles),
            "avg_sentiment": round(avg_sentiment, 3),
            "categories": dict(categories),
            "trending_entities": trending[:10],
            "spikes": spikes[:5]
        }

        # Cache digest
        await redis_client.set(DIGEST_KEY, digest, ttl=43200)  # 12 hours

        logger.info(
            "digest_computed",
            articles=len(recent_articles),
            trending=len(trending),
            spikes=len(spikes)
        )

        return digest

    async def get_daily_digest(self) -> Optional[Dict[str, Any]]:
        """Get cached daily digest."""
        return await redis_client.get(DIGEST_KEY)


# Global instance
temporal_analytics = TemporalAnalytics()
