"""Search API Router.

Handles native, lightning-fast BM25 keyword search with 0% external API dependencies.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import time
import structlog

from app.utils.elasticsearch_client import es_client

logger = structlog.get_logger()
router = APIRouter(prefix="/search", tags=["search"])


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    filters: Optional[Dict[str, Any]] = None
    size: int = 20
    use_reranking: bool = True


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[Dict[str, Any]]
    total: int
    latency_ms: float
    used_reranking: bool


@router.post("", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """Perform native BM25 keyword search directly on Elasticsearch."""
    start_time = time.time()

    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    # Native BM25 keyword search - 100% reliable, 0MB RAM, zero external APIs
    results = await es_client.search_bm25(
        query=request.query,
        filters=request.filters,
        size=request.size
    )

    latency_ms = (time.time() - start_time) * 1000

    return SearchResponse(
        query=request.query,
        results=results,
        total=len(results),
        latency_ms=round(latency_ms, 2),
        used_reranking=False
    )


@router.get("/keyword")
async def keyword_search(
    query: str = Query(..., min_length=2),
    source: Optional[str] = None,
    category: Optional[str] = None,
    days: Optional[int] = None,
    size: int = Query(20, ge=1, le=100),
    from_: int = Query(0, alias="from", ge=0)
):
    """Perform keyword-only search using BM25."""
    start_time = time.time()

    filters = {}
    if source:
        filters["source"] = source
    if category:
        filters["category"] = category
    if days:
        filters["date_range"] = {"gte": f"now-{days}d"}

    results = await es_client.search_bm25(
        query=query,
        filters=filters if filters else None,
        size=size,
        from_=from_
    )

    latency_ms = (time.time() - start_time) * 1000

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "latency_ms": round(latency_ms, 2)
    }


@router.get("/semantic")
async def semantic_search(
    query: str = Query(..., min_length=2),
    source: Optional[str] = None,
    category: Optional[str] = None,
    days: Optional[int] = None,
    size: int = Query(20, ge=1, le=100)
):
    """Fall back to BM25 search to bypass cloud DNS issues."""
    return await keyword_search(
        query=query,
        source=source,
        category=category,
        days=days,
        size=size
    )


@router.get("/similar/{article_id}")
async def find_similar(
    article_id: str,
    size: int = Query(10, ge=1, le=50)
):
    """Find similar articles by matching source and category."""
    article = await es_client.client.get(index="articles", id=article_id, _source=["category", "source", "title"])
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")

    category = article["_source"].get("category", "")
    title = article["_source"].get("title", "")

    # Query standard BM25 using the category as a filter
    results = await es_client.search_bm25(
        query="*",
        filters={"category": category} if category else None,
        size=size + 1
    )

    results = [r for r in results if r.get("article_id") != article_id][:size]

    return {
        "article_id": article_id,
        "title": title,
        "similar_articles": results
    }
