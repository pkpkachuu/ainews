"""Search API Router.

Handles hybrid search, semantic search, and keyword search endpoints using native DNS resolution.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import asyncio
import urllib.request
import json
import structlog

from app.utils.elasticsearch_client import es_client
from app.services.reranker import reranker

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


def get_query_embedding_native(query: str) -> List[float]:
    """Fetch query embedding using Python's native urllib (immune to httpx DNS bugs)."""
    # Try the main endpoint, fallback to CDN if needed
    urls = [
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        "https://api.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    last_error = None
    for url in urls:
        try:
            data = json.dumps({"inputs": query}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            # Use native OS-level blocking DNS resolution
            with urllib.request.urlopen(req, timeout=8.0) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    
                    # Auto-flatten nested lists (e.g. [[[...]]] or [[...]] -> [...])
                    if isinstance(result, list) and len(result) > 0:
                        while isinstance(result[0], list):
                            result = result[0]
                        return result
                    raise ValueError("Unexpected API response format")
                else:
                    last_error = f"Status {response.status}"
        except Exception as e:
            logger.warning("native_endpoint_failed", url=url, error=str(e))
            last_error = str(e)
            
    raise RuntimeError(f"All native endpoints failed: {last_error}")


@router.post("", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """Perform hybrid search combining BM25 and semantic retrieval."""
    import time
    start_time = time.time()

    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    try:
        # Run the native blocking DNS request on a background thread (bypasses anyio bug)
        query_embedding = await asyncio.to_thread(get_query_embedding_native, request.query)
    except Exception as e:
        logger.error("native_dns_resolution_failed", error=str(e))
        raise HTTPException(status_code=502, detail="Failed to generate search embedding via native resolver")

    # Perform hybrid search with RRF on Elastic Serverless
    results = await es_client.hybrid_search_rrf(
        query=request.query,
        query_vector=query_embedding,
        filters=request.filters,
        k=request.size * 2,
        num_candidates=100
    )

    # Apply lightweight reranking (recency boost)
    if request.use_reranking and len(results) > 5:
        results = await reranker.rerank(request.query, results, top_k=request.size)

    latency_ms = (time.time() - start_time) * 1000

    return SearchResponse(
        query=request.query,
        results=results[:request.size],
        total=len(results),
        latency_ms=round(latency_ms, 2),
        used_reranking=request.use_reranking
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
    import time
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
    """Perform semantic search using vector embeddings."""
    import time
    start_time = time.time()

    try:
        query_embedding = await asyncio.to_thread(get_query_embedding_native, query)
    except Exception as e:
        logger.error("native_dns_resolution_failed", error=str(e))
        raise HTTPException(status_code=502, detail="Failed to generate search embedding via native resolver")

    filters = {}
    if source:
        filters["source"] = source
    if category:
        filters["category"] = category
    if days:
        filters["date_range"] = {"gte": f"now-{days}d"}

    results = await es_client.search_knn(
        query_vector=query_embedding,
        k=size,
        num_candidates=100,
        filters=filters if filters else None
    )

    latency_ms = (time.time() - start_time) * 1000

    return {
        "query": query,
        "results": results,
        "total": len(results),
        "latency_ms": round(latency_ms, 2)
    }


@router.get("/similar/{article_id}")
async def find_similar(
    article_id: str,
    size: int = Query(10, ge=1, le=50)
):
    """Find articles similar to a given article using vector search."""
    article = await es_client.client.get(index="articles", id=article_id, _source=["embedding", "title"])
    if not article or "embedding" not in article["_source"]:
        raise HTTPException(status_code=404, detail="Article not found or no embedding")

    embedding = article["_source"]["embedding"]
    title = article["_source"].get("title", "")

    results = await es_client.search_knn(
        query_vector=embedding,
        k=size + 1,
        num_candidates=50
    )

    results = [r for r in results if r.get("article_id") != article_id][:size]

    return {
        "article_id": article_id,
        "title": title,
        "similar_articles": results
    }
