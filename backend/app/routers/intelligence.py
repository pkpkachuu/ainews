"""Intelligence API Router.

Handles proactive intelligence feed, trends, and analytics endpoints.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

from app.services.proactive_intelligence import pie_engine
from app.services.temporal_analytics import temporal_analytics
from app.utils.redis_client import redis_client

router = APIRouter(prefix="/intelligence", tags=["intelligence"])


class FeedResponse(BaseModel):
    """Intelligence feed response model."""
    feed: List[Dict[str, Any]]
    generated_at: Optional[str]
    item_count: int
    message: str = ""


@router.get("/feed", response_model=FeedResponse)
async def get_intelligence_feed():
    """Get the latest proactively generated intelligence feed with duplicate protection."""
    items = await pie_engine.get_feed()

    if not items:
        return FeedResponse(
            feed=[],
            generated_at=None,
            item_count=0,
            message="Feed not yet generated. Run the feed generator first."
        )

    # Items are stored newest first (LPUSH), so reverse for chronological
    feed = list(reversed(items))

    # Defensive Deduplication: Filter out any duplicates by matching topic_label
    seen_labels = set()
    deduplicated_feed = []
    for item in feed:
        label = item.get("topic_label", "")
        if label not in seen_labels:
            seen_labels.add(label)
            deduplicated_feed.append(item)

    return FeedResponse(
        feed=deduplicated_feed,
        generated_at=deduplicated_feed[0].get("generated_at") if deduplicated_feed else None,
        item_count=len(deduplicated_feed)
    )


@router.post("/feed/generate")
async def generate_feed():
    """Manually trigger intelligence feed generation."""
    feed = await pie_engine.generate_feed()

    return {
        "message": "Feed generated successfully",
        "item_count": len(feed),
        "items": feed
    }


@router.post("/emergence/detect")
async def detect_emergence():
    """Manually trigger emergence detection."""
    emerging = await pie_engine.run_emergence_detection()

    return {
        "message": "Emergence detection completed",
        "emerging_topics": len(emerging),
        "topics": emerging
    }


@router.post("/narrative/monitor")
async def monitor_narratives():
    """Manually trigger narrative monitoring."""
    shifts = await pie_engine.run_narrative_monitoring()

    return {
        "message": "Narrative monitoring completed",
        "shifts_detected": len(shifts),
        "shifts": shifts
    }


# Trending Analytics Endpoints
@router.get("/trending")
async def get_trending_entities(
    top_n: int = Query(20, ge=5, le=50),
    min_articles: int = Query(5, ge=1)
):
    """Get trending entities by velocity (rate of change)."""
    trending = await temporal_analytics.get_trending_entities(
        top_n=top_n,
        min_articles=min_articles
    )

    return {
        "trending": trending,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/digest")
async def get_daily_digest():
    """Get the daily news digest."""
    digest = await temporal_analytics.get_daily_digest()

    if not digest:
        digest = await temporal_analytics.compute_daily_digest()

    return digest


@router.post("/digest/refresh")
async def refresh_daily_digest():
    """Manually refresh the daily digest."""
    digest = await temporal_analytics.compute_daily_digest()
    return digest


# Entity Analytics Endpoints
@router.get("/entity/{entity_name}/trend")
async def get_entity_trend(
    entity_name: str,
    days: int = Query(30, ge=7, le=90)
):
    """Get trend analysis for a specific entity."""
    trend = await temporal_analytics.get_entity_trend(entity_name, days=days)

    if "error" in trend:
        raise HTTPException(status_code=404, detail=trend["error"])

    return trend


@router.get("/entity/{entity_name}/articles")
async def get_entity_articles(
    entity_name: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """Get articles mentioning a specific entity."""
    from app.utils.supabase_client import db_client

    articles = await db_client.get_articles_by_entity(
        entity_name,
        limit=limit,
        offset=offset
    )

    return {
        "entity": entity_name,
        "articles": articles,
        "count": len(articles)
    }


@router.get("/topic/{topic}/trend")
async def get_topic_trend(
    topic: str,
    days: int = Query(30, ge=7, le=90)
):
    """Get trend analysis for a topic/keyword."""
    trend = await temporal_analytics.get_topic_trend(topic, days=days)

    if "error" in trend:
        raise HTTPException(status_code=404, detail=trend["error"])

    return trend


@router.get("/volume")
async def get_volume_over_time(
    entity: Optional[str] = None,
    topic: Optional[str] = None,
    days: int = Query(30, ge=7, le=90),
    interval: str = Query("day", regex="^(hour|day|week)$")
):
    """Get article volume over time."""
    from app.utils.elasticsearch_client import es_client

    if not entity and not topic:
        raise HTTPException(
            status_code=400,
            detail="Must provide either entity or topic parameter"
        )

    volume = await es_client.get_volume_over_time(
        entity=entity,
        topic=topic,
        days=days,
        interval=interval
    )

    return {
        "entity": entity,
        "topic": topic,
        "days": days,
        "interval": interval,
        "data": volume
    }
