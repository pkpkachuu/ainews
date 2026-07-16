"""Search API Router.

Handles hybrid search, semantic search, and keyword search endpoints.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel

from app.utils.elasticsearch_client import es_client
from app.services.reranker import reranker
from sentence_transformers import SentenceTransformer

# Lazy load embedding model
_embedding_model = None


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from app.config import settings
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


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
    """Perform hybrid search combining BM25 and semantic retrieval.

    Optionally reranks results using cross-encoder model.
    """
    import time
    start_time = time.time()

    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    # Generate query embedding
    model = get_embedding_model()
    query_embedding = model.encode(request.query, normalize_embeddings=True).tolist()

    # Perform hybrid search with RRF
    results = await es_client.hybrid_search_rrf(
        query=request.query,
        query_vector=query_embedding,
        filters=request.filters,
        k=request.size * 2,  # Get more candidates for reranking
        num_candidates=100
    )

    # Apply reranking if requested
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

    # Generate query embedding
    model = get_embedding_model()
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

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
    # Get the article's embedding
    article = await es_client.client.get(index="articles", id=article_id, _source=["embedding", "title"])
    if not article or "embedding" not in article["_source"]:
        raise HTTPException(status_code=404, detail="Article not found or no embedding")

    embedding = article["_source"]["embedding"]
    title = article["_source"].get("title", "")

    # Search for similar articles
    results = await es_client.search_knn(
        query_vector=embedding,
        k=size + 1,  # +1 because the article itself will be included
        num_candidates=50
    )

    # Filter out the original article
    results = [r for r in results if r.get("article_id") != article_id][:size]

    return {
        "article_id": article_id,
        "title": title,
        "similar_articles": results
    }
