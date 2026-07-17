"""Search API Router.

Handles hybrid search, semantic search, and keyword search endpoints with DNS fallback routing.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
import httpx
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


async def get_query_embedding_from_api(query: str) -> List[float]:
    """Fetch query embedding from Hugging Face's serverless Inference API using DNS fallbacks."""
    # Alternative DNS routes for Hugging Face serverless inference
    endpoints = [
        "https://api.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    last_error = None
    async with httpx.AsyncClient() as client:
        for url in endpoints:
            try:
                response = await client.post(
                    url,
                    json={"inputs": query},
                    timeout=10.0
                )
                if response.status_code == 200:
                    embedding = response.json()
                    
                    # Auto-flatten nested lists (e.g. [[[...]]] or [[...]] -> [...])
                    if isinstance(embedding, list) and len(embedding) > 0:
                        while isinstance(embedding[0], list):
                            embedding = embedding[0]
                        return embedding
                    
                    raise ValueError("Unexpected API response format")
                else:
                    logger.warning("hf_endpoint_failed", url=url, status_code=response.status_code)
                    last_error = f"Status {response.status_code}: {response.text}"
            except Exception as e:
                logger.warning("hf_endpoint_exception", url=url, error=str(e))
                last_error = str(e)
                
    logger.error("all_hf_endpoints_failed", last_error=last_error)
    raise HTTPException(status_code=502, detail=f"Failed to generate embedding from Hugging Face: {last_error}")


@router.post("", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """Perform hybrid search combining BM25 and semantic retrieval."""
    import time
    start_time = time.time()

    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    # Generate query embedding using HF free API with automatic DNS fallback routing (No PyTorch loaded in memory!)
    query_embedding = await get_query_embedding_from_api(request.query)

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

    query_embedding = await get_query_embedding_from_api(query)

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
