"""Articles API Router.

Handles article CRUD operations and retrieval.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict, Any

from app.utils.supabase_client import db_client
from app.utils.elasticsearch_client import es_client

router = APIRouter(prefix="/articles", tags=["articles"])


@router.get("/")
async def list_articles(
    source: Optional[str] = None,
    category: Optional[str] = None,
    days: Optional[int] = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """List articles with optional filters.

    Supports filtering by source, category, and date range.
    """
    # Build ES query
    filters = {}
    if source:
        filters["source"] = source
    if category:
        filters["category"] = category
    if days:
        filters["date_range"] = {"gte": f"now-{days}d"}

    # Search using hybrid approach for better results
    results = await es_client.search_bm25(
        query="*",  # Match all
        filters=filters if filters else None,
        size=limit,
        from_=offset
    )

    return {
        "articles": results,
        "count": len(results),
        "filters": {
            "source": source,
            "category": category,
            "days": days
        }
    }


@router.get("/recent")
async def get_recent_articles(
    hours: int = Query(24, ge=1, le=168),
    size: int = Query(50, ge=1, le=200)
) -> Dict[str, Any]:
    """Get recent articles from the specified time window."""
    articles = await es_client.get_recent_articles(hours=hours, size=size)

    return {
        "articles": articles,
        "count": len(articles),
        "hours": hours
    }


@router.get("/stats")
async def get_article_stats() -> Dict[str, Any]:
    """Get database statistics about articles."""
    stats = await db_client.get_stats()

    # Get additional stats from ES
    try:
        # Get total count in ES
        es_count = await es_client.client.count(index="articles")
        stats["es_indexed"] = es_count["count"]

        # Get categories
        cat_result = await es_client.client.search(
            index="articles",
            body={
                "size": 0,
                "aggs": {
                    "categories": {
                        "terms": {
                            "field": "category",
                            "size": 10
                        }
                    }
                }
            }
        )
        stats["categories"] = [
            {"category": bucket["key"], "count": bucket["doc_count"]}
            for bucket in cat_result["aggregations"]["categories"]["buckets"]
        ]

        # Get sources
        src_result = await es_client.client.search(
            index="articles",
            body={
                "size": 0,
                "aggs": {
                    "sources": {
                        "terms": {
                            "field": "source",
                            "size": 20
                        }
                    }
                }
            }
        )
        stats["sources"] = [
            {"source": bucket["key"], "count": bucket["doc_count"]}
            for bucket in src_result["aggregations"]["sources"]["buckets"]
        ]

    except Exception as e:
        stats["es_error"] = str(e)

    return stats


@router.get("/{article_id}")
async def get_article(article_id: str) -> Dict[str, Any]:
    """Get a single article by ID.

    Returns article with entities and keyphrases.

    NOTE: this route must stay registered AFTER the static /, /recent, and
    /stats routes above - FastAPI matches routes in registration order, so
    a dynamic /{article_id} defined first will greedily swallow requests to
    /articles/stats and /articles/recent, treating "stats"/"recent" as an
    article_id and blowing up with an invalid-UUID error at the DB layer.
    """
    article = await db_client.get_article(article_id)

    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    return article


@router.delete("/{article_id}")
async def delete_article(article_id: str) -> Dict[str, str]:
    """Delete an article from database and index."""
    # Check if article exists
    article = await db_client.get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    # Delete from ES index
    await es_client.delete_article(article_id)

    # Note: For MVP, we don't actually delete from PostgreSQL
    # Just mark as deleted or remove from index only

    return {"message": "Article deleted from index", "article_id": article_id}
