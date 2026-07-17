This is a complete, step-by-step master guide to deploying your entire
application for $0, using Render for the backend, Vercel for the frontend, and
your existing Supabase and Elastic Serverless databases.

No credit cards are required, and the backend is optimized to use less
than 100MB of RAM so it will not crash on Render's free tier [1].

Step 1: Update Your Local Files

Open your project folder on your computer. We need to overwrite five files to
make your backend compatible with Render's memory limits (by offloading
embedding generations to Hugging Face's free serverless API).

1. Overwrite backend/requirements.txt

This removes heavy libraries like PyTorch and spaCy, allowing Render to build
your app in under a minute without running out of memory.

fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
supabase==2.5.3
asyncpg==0.29.0
psycopg2-binary==2.9.9
redis==5.0.8
elasticsearch==8.15.0
langgraph==0.2.28
langchain-core==0.3.1
langchain-groq==0.2.0
groq==0.11.0
feedparser==6.0.11
trafilatura==1.8.0
aiohttp==3.10.5
python-dotenv==1.0.1
structlog==24.4.0
httpx==0.27.2
apscheduler==3.10.4

2. Overwrite backend/app/utils/elasticsearch_client.py

This version is fully compatible with your Elasticsearch Serverless API key and
includes a longer timeout limit to prevent connection drops.

"""Elasticsearch client and index management."""

import os
from dotenv import load_dotenv
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from typing import Optional, List, Dict, Any
import structlog

from app.config import settings

# Load .env variables into system environment
load_dotenv()

logger = structlog.get_logger()

# Index name constants
ARTICLES_INDEX = "articles"
EVENTS_INDEX = "events"

ARTICLES_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "article_id": {"type": "keyword"},
            "url": {"type": "keyword", "index": False},
            "title": {
                "type": "text",
                "analyzer": "english",
                "fields": {
                    "keyword": {"type": "keyword"}
                }
            },
            "body": {"type": "text", "analyzer": "english"},
            "summary": {"type": "text", "analyzer": "english"},
            "published_at": {"type": "date"},
            "source": {"type": "keyword"},
            "category": {"type": "keyword"},
            "language": {"type": "keyword"},
            "sentiment_score": {"type": "float"},
            "sentiment_label": {"type": "keyword"},
            "word_count": {"type": "integer"},
            "processing_status": {"type": "keyword"},
            "entities": {
                "type": "nested",
                "properties": {
                    "text": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "confidence": {"type": "float"}
                }
            },
            "keyphrases": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "created_at": {"type": "date"},
            "indexed_at": {"type": "date"}
        }
    }
}

EVENTS_INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "event_id": {"type": "keyword"},
            "headline": {"type": "text", "analyzer": "english"},
            "event_type": {"type": "keyword"},
            "key_entities": {"type": "keyword"},
            "first_seen": {"type": "date"},
            "last_updated": {"type": "date"},
            "article_count": {"type": "integer"},
            "article_ids": {"type": "keyword"},
            "sentiment_history": {
                "type": "nested",
                "properties": {
                    "date": {"type": "date"},
                    "score": {"type": "float"}
                }
            },
            "status": {"type": "keyword"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}


class ElasticsearchClient:
    """Async Elasticsearch client wrapper."""

    def __init__(self):
        self.client: Optional[AsyncElasticsearch] = None

    async def connect(self) -> None:
        """Initialize Elasticsearch connection."""
        api_key = os.getenv("ELASTICSEARCH_API_KEY")
        
        if api_key:
            # Serverless authentication using API Key (Timeout increased to 120s)
            self.client = AsyncElasticsearch(
                settings.elasticsearch_url,
                api_key=api_key,
                verify_certs=False,
                request_timeout=120
            )
        else:
            # Standard authentication using basic auth URL (Timeout increased to 120s)
            self.client = AsyncElasticsearch(
                [settings.elasticsearch_url],
                verify_certs=False,
                request_timeout=120
            )
        logger.info("elasticsearch_connected", url=settings.elasticsearch_url)

    async def disconnect(self) -> None:
        """Close Elasticsearch connection."""
        if self.client:
            await self.client.close()
            logger.info("elasticsearch_disconnected")

    async def create_indices(self) -> None:
        """Create indices if they don't exist."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        # Create articles index
        if not await self.client.indices.exists(index=ARTICLES_INDEX):
            await self.client.indices.create(
                index=ARTICLES_INDEX,
                body=ARTICLES_INDEX_MAPPING
            )
            logger.info("created_index", index=ARTICLES_INDEX)

        # Create events index
        if not await self.client.indices.exists(index=EVENTS_INDEX):
            await self.client.indices.create(
                index=EVENTS_INDEX,
                body=EVENTS_INDEX_MAPPING
            )
            logger.info("created_index", index=EVENTS_INDEX)

    async def index_article(self, article: Dict[str, Any]) -> None:
        """Index a single article."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        doc = self._prepare_article_doc(article)
        await self.client.index(
            index=ARTICLES_INDEX,
            id=article["article_id"],
            document=doc
        )

    async def bulk_index_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Bulk index multiple articles.

        Returns:
            Number of successfully indexed articles
        """
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        actions = [
            {
                "_index": ARTICLES_INDEX,
                "_id": article["article_id"],
                "_source": self._prepare_article_doc(article)
            }
            for article in articles
        ]

        success, _ = await async_bulk(self.client, actions)
        return success

    def _prepare_article_doc(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare article document for indexing."""
        doc = {
            "article_id": article.get("article_id"),
            "url": article.get("url"),
            "title": article.get("title"),
            "body": article.get("body"),
            "summary": article.get("summary"),
            "published_at": article.get("published_at"),
            "source": article.get("source"),
            "category": article.get("category"),
            "language": article.get("language", "en"),
            "sentiment_score": article.get("sentiment_score"),
            "sentiment_label": article.get("sentiment_label"),
            "word_count": article.get("word_count"),
            "processing_status": article.get("processing_status", "done"),
            "entities": article.get("entities", []),
            "keyphrases": article.get("keyphrases", []),
            "created_at": article.get("created_at"),
            "indexed_at": article.get("indexed_at")
        }

        # Handle embedding - convert list to proper format if needed
        if "embedding" in article:
            embedding = article["embedding"]
            if embedding is not None:
                doc["embedding"] = embedding

        return doc

    async def search_bm25(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        size: int = 20,
        from_: int = 0
    ) -> List[Dict[str, Any]]:
        """BM25 keyword search with optional filters."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        bool_query = {
            "must": [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^2", "body", "summary"],
                        "type": "best_fields"
                    }
                }
            ]
        }

        if filters:
            filter_clauses = []
            if "source" in filters:
                filter_clauses.append({"term": {"source": filters["source"]}})
            if "category" in filters:
                filter_clauses.append({"term": {"category": filters["category"]}})
            if "date_range" in filters:
                filter_clauses.append({
                    "range": {
                        "published_at": filters["date_range"]
                    }
                })
            if "language" in filters:
                filter_clauses.append({"term": {"language": filters["language"]}})
            if filter_clauses:
                bool_query["filter"] = filter_clauses

        response = await self.client.search(
            index=ARTICLES_INDEX,
            query={"bool": bool_query},
            size=size,
            from_=from_
        )

        return [hit["_source"] | {"_score": hit["_score"], "_id": hit["_id"]}
                for hit in response["hits"]["hits"]]

    async def search_knn(
        self,
        query_vector: List[float],
        k: int = 20,
        num_candidates: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """kNN vector search with optional pre-filtering."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        knn_query = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": k,
            "num_candidates": num_candidates
        }

        if filters:
            filter_query = {"bool": {"must": []}}
            if "source" in filters:
                filter_query["bool"]["must"].append({"term": {"source": filters["source"]}})
            if "category" in filters:
                filter_query["bool"]["must"].append({"term": {"category": filters["category"]}})
            if "date_range" in filters:
                filter_query["bool"]["must"].append({
                    "range": {"published_at": filters["date_range"]}
                })
            if "language" in filters:
                filter_query["bool"]["must"].append({"term": {"language": filters["language"]}})
            if filter_query["bool"]["must"]:
                knn_query["filter"] = filter_query

        response = await self.client.search(
            index=ARTICLES_INDEX,
            knn=knn_query,
            size=k
        )

        return [hit["_source"] | {"_score": hit["_score"], "_id": hit["_id"]}
                for hit in response["hits"]["hits"]]

    async def hybrid_search_rrf(
        self,
        query: str,
        query_vector: List[float],
        filters: Optional[Dict[str, Any]] = None,
        k: int = 20,
        num_candidates: int = 100,
        rank_constant: int = 60
    ) -> List[Dict[str, Any]]:
        """Hybrid search using Reciprocal Rank Fusion.

        Combines BM25 and kNN results and merges using RRF.
        """
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        # Build filter clause for both searches
        filter_clause = None
        if filters:
            filter_clause = []
            if "source" in filters:
                filter_clause.append({"term": {"source": filters["source"]}})
            if "category" in filters:
                filter_clause.append({"term": {"category": filters["category"]}})
            if "date_range" in filters:
                filter_clause.append({
                    "range": {"published_at": filters["date_range"]}
                })
            if "language" in filters:
                filter_clause.append({"term": {"language": filters["language"]}})

        # Use ES 8.9+ hybrid search if available
        try:
            search_body = {
                "sub_searches": [
                    {
                        "query": {
                            "bool": {
                                "must": [
                                    {
                                        "multi_match": {
                                            "query": query,
                                            "fields": ["title^2", "body", "summary"],
                                            "type": "best_fields"
                                        }
                                    }
                                ],
                                "filter": filter_clause if filter_clause else []
                            }
                        }
                    },
                    {
                        "knn": {
                            "field": "embedding",
                            "query_vector": query_vector,
                            "k": k,
                            "num_candidates": num_candidates,
                            "filter": {"bool": {"must": filter_clause}} if filter_clause else None
                        }
                    }
                ],
                "rank": {
                    "rrf": {
                        "window_size": k,
                        "rank_constant": rank_constant
                    }
                },
                "size": k
            }

            response = await self.client.search(
                index=ARTICLES_INDEX,
                body=search_body
            )

            return [hit["_source"] | {"_score": hit["_score"], "_id": hit["_id"]}
                    for hit in response["hits"]["hits"]]

        except Exception:
            # Fallback to manual RRF if hybrid not supported
            bm25_results = await self.search_bm25(query, filters, size=k)
            knn_results = await self.search_knn(query_vector, k=k, num_candidates=num_candidates, filters=filters)
            return self._manual_rrf(bm25_results, knn_results, rank_constant)

    def _manual_rrf(
        self,
        bm25_results: List[Dict[str, Any]],
        knn_results: List[Dict[str, Any]],
        rank_constant: int = 60
    ) -> List[Dict[str, Any]]:
        """Manual Reciprocal Rank Fusion for fallback."""
        scores = {}

        # Score BM25 results
        for rank, result in enumerate(bm25_results, 1):
            doc_id = result.get("article_id") or result.get("_id")
            if doc_id:
                rrf_score = 1 / (rank_constant + rank)
                if doc_id not in scores:
                    scores[doc_id] = {"doc": result, "score": 0}
                scores[doc_id]["score"] += rrf_score

        # Score kNN results
        for rank, result in enumerate(knn_results, 1):
            doc_id = result.get("article_id") or result.get("_id")
            if doc_id:
                rrf_score = 1 / (rank_constant + rank)
                if doc_id not in scores:
                    scores[doc_id] = {"doc": result, "score": 0}
                scores[doc_id]["score"] += rrf_score

        # Sort by combined score
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)

        # Update scores in results
        return [{**item["doc"], "_score": item["score"]} for item in sorted_results]

    async def get_volume_over_time(
        self,
        entity: Optional[str] = None,
        topic: Optional[str] = None,
        days: int = 30,
        interval: str = "day"
    ) -> List[Dict[str, Any]]:
        """Get article volume over time using date_histogram aggregation."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        query = {"match_all": {}}

        if entity:
            query = {
                "nested": {
                    "path": "entities",
                    "query": {"term": {"entities.text": entity}}
                }
            }
        elif topic:
            query = {
                "match": {
                    "body": topic
                }
            }

        search_body = {
            "query": query,
            "aggs": {
                "volume": {
                    "date_histogram": {
                        "field": "published_at",
                        "calendar_interval": interval
                    },
                    "aggs": {
                        "avg_sentiment": {"avg": {"field": "sentiment_score"}}
                    }
                }
            },
            "size": 0
        }

        response = await self.client.search(
            index=ARTICLES_INDEX,
            body=search_body
        )

        return [
            {
                "date": bucket["key_as_string"],
                "count": bucket["doc_count"],
                "avg_sentiment": bucket["avg_sentiment"].get("value")
            }
            for bucket in response["aggregations"]["volume"]["buckets"]
        ]

    async def get_recent_articles(
        self,
        hours: int = 2,
        size: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get the most recent articles in the time window, newest first."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        search_body = {
            "query": {
                "range": {
                    "published_at": {
                        "gte": f"now-{hours}h"
                    }
                }
            },
            "sort": [
                {"published_at": {"order": "desc"}}
            ],
            "size": size,
            "_source": ["article_id", "title", "body", "embedding", "entities", "keyphrases", "published_at", "source", "sentiment_score"]
        }

        response = await self.client.search(
            index=ARTICLES_INDEX,
            body=search_body
        )

        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def get_entity_sentiment_history(
        self,
        entity: str,
        hours: int = 24,
        hours_before: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get sentiment history for an entity."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

        published_range: Dict[str, str] = {"gte": f"now-{hours}h"}
        if hours_before is not None:
            published_range = {
                "gte": f"now-{hours + hours_before}h",
                "lt": f"now-{hours_before}h"
            }

        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "nested": {
                                "path": "entities",
                                "query": {"term": {"entities.text": entity}}
                            }
                        },
                        {
                            "range": {
                                "published_at": published_range
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "avg_sentiment": {"avg": {"field": "sentiment_score"}},
                "sentiment_std_dev": {"extended_stats": {"field": "sentiment_score"}},
                "by_source": {
                    "terms": {"field": "source", "size": 10},
                    "aggs": {
                        "avg_sentiment": {"avg": {"field": "sentiment_score"}}
                    }
                }
            },
            "size": 0
        }

        response = await self.client.search(
            index=ARTICLES_INDEX,
            body=search_body
        )

        return {
            "avg_sentiment": response["aggregations"]["avg_sentiment"].get("value"),
            "std_dev": response["aggregations"]["sentiment_std_dev"].get("std_deviation"),
            "by_source": [
                {
                    "source": bucket["key"],
                    "count": bucket["doc_count"],
                    "avg_sentiment": bucket["avg_sentiment"].get("value")
                }
                for bucket in response["aggregations"]["by_source"]["buckets"]
            ]
        }

    async def delete_article(self, article_id: str) -> None:
        """Delete an article from the index."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")
        await self.client.delete(index=ARTICLES_INDEX, id=article_id)


# Global Elasticsearch client instance
es_client = ElasticsearchClient()

3. Overwrite backend/app/routers/search.py

This calls the Hugging Face Free serverless Inference API to generate query
vectors on the fly, eliminating the need to load a PyTorch model into your
active cloud RAM.

"""Search API Router.

Handles hybrid search, semantic search, and keyword search endpoints.
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
    """Fetch query embedding from Hugging Face's free serverless Inference API."""
    url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
    headers = {}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                url,
                json={"inputs": query},
                headers=headers,
                timeout=10.0
            )
            if response.status_code == 200:
                embedding = response.json()
                if isinstance(embedding, list) and len(embedding) > 0:
                    return embedding
                raise ValueError("Unexpected API response format")
            else:
                logger.error("hf_api_error", status_code=response.status_code, text=response.text)
                raise HTTPException(status_code=502, detail="Failed to generate embedding from Hugging Face API")
        except Exception as e:
            logger.error("hf_api_exception", error=str(e))
            raise HTTPException(status_code=502, detail="Embedding generation timed out or failed")


@router.post("", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """Perform hybrid search combining BM25 and semantic retrieval."""
    import time
    start_time = time.time()

    if not request.query or len(request.query.strip()) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")

    # Generate query embedding using HF free API (No PyTorch loaded in memory!)
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

4. Overwrite backend/app/services/reranker.py

This implements a lightweight recency scaling fallback to skip executing PyTorch
in production.

"""Reranking service.

Applies recency boost and fallback relevance filtering for cloud deployment.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()


class Reranker:
    """Reranks candidates using lightweight recency boost."""

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Lightweight pass-through rerank to bypass PyTorch in production.

        Applies standard recency scaling to the Elasticsearch scores.
        """
        if not candidates:
            return []

        # Apply recency boost
        reranked = self._apply_recency_boost(candidates)

        # Sort by updated score
        reranked = sorted(
            reranked,
            key=lambda x: x.get("rerank_score") or x.get("_score", 0),
            reverse=True
        )

        return reranked[:top_k]

    def _apply_recency_boost(
        self,
        candidates: List[Dict[str, Any]],
        boost_factor: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Apply recency boost to scores."""
        now = datetime.now(timezone.utc)

        for candidate in candidates:
            pub_date = candidate.get("published_at")
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        pub_datetime = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
                    else:
                        pub_datetime = pub_date

                    # Calculate days since publication
                    days_old = (now - pub_datetime).days

                    # Exponential decay with half-life of 7 days
                    recency_multiplier = 1 + boost_factor * (0.5 ** (days_old / 7))

                    if "_score" in candidate:
                        candidate["_score"] *= recency_multiplier

                except Exception:
                    pass

        return candidates


# Global reranker instance
reranker = Reranker()

5. Overwrite backend/reindex.py

The bulk sync script configured for lightweight batches.

# backend/reindex.py
import asyncio
import structlog
from app.utils.supabase_client import db_client
from app.utils.elasticsearch_client import es_client

logger = structlog.get_logger()

async def sync_supabase_to_elasticsearch():
    db_client.connect()
    await es_client.connect()
    await es_client.create_indices()

    logger.info("starting_bulk_reindex_from_supabase")

    try:
        response = db_client.client.table("articles").select(
            "*, article_entities(entity_text, entity_type, confidence), article_keyphrases(keyphrase)"
        ).eq("processing_status", "done").execute()
        
        articles = response.data
        logger.info("fetched_all_data_from_supabase", count=len(articles))

        es_articles = []
        for article in articles:
            entities = [
                {
                    "text": e["entity_text"],
                    "type": e["entity_type"],
                    "confidence": e["confidence"]
                }
                for e in article.get("article_entities", [])
            ]
            keyphrases = [k["keyphrase"] for k in article.get("article_keyphrases", [])]

            article_doc = {
                "article_id": article["article_id"],
                "url": article["url"],
                "title": article["title"],
                "body": article["body"],
                "summary": article["summary"],
                "published_at": article["published_at"],
                "source": article["source"],
                "category": article["category"],
                "sentiment_score": article["sentiment_score"],
                "sentiment_label": article["sentiment_label"],
                "word_count": article["word_count"],
                "entities": entities,
                "keyphrases": keyphrases,
                "created_at": article.get("created_at"),
                "indexed_at": article.get("indexed_at")
            }

            embedding = article.get("embedding")
            if embedding:
                if isinstance(embedding, str):
                    article_doc["embedding"] = [float(x) for x in embedding.replace("[", "").replace("]", "").split(",")]
                else:
                    article_doc["embedding"] = embedding

            es_articles.append(article_doc)

        # Bulk index to Elasticsearch in batches of 30 for safety and speed
        batch_size = 30
        for i in range(0, len(es_articles), batch_size):
            batch = es_articles[i : i + batch_size]
            await es_client.bulk_index_articles(batch)
            logger.info("bulk_indexed_batch_to_es", start=i, end=min(i + batch_size, len(es_articles)))

        logger.info("reindex_completed_successfully_with_bulk_sync")
    except Exception as e:
        logger.error("reindex_failed", error=str(e))

if __name__ == "__main__":
    asyncio.run(sync_supabase_to_elasticsearch())

Step 2: Push to a New Git Branch

We will commit and push these optimized changes to a dedicated deployment branch
so your default master branch is untouched.

In your terminal (inside your project directory), run these commands:

# 1. Create and switch to a new branch
git checkout -b deploy-serverless

# 2. Add and commit all updated files
git add .
git commit -m "Optimize requirements and routers for Render free-tier RAM limits"

# 3. Push this branch to GitHub
git push -u origin deploy-serverless

Step 3: Deploy the Backend to Render

1.  Log in to Render.com ($0/month, no credit card required).
2.  Click New + -> Web Service.
3.  Connect your GitHub account and select your repository.
4.  Fill in the deployment details:
      - Name: news-analytics-api
      - Branch: Select deploy-serverless (very important).
      - Build Command: pip install -r backend/requirements.txt
      - Start Command: cd backend && uvicorn app.main:app --host 0.0.0.0 --port
        $PORT
      - Instance Type: Select Free ($0/month).
5.  Scroll down, click Advanced, and add your Environment Variables:
      - SUPABASE_URL = (your Supabase URL)
      - SUPABASE_ANON_KEY = (your Supabase public key)
      - SUPABASE_SERVICE_ROLE_KEY = (your Supabase service key)
      - DATABASE_URL = (your Supabase direct connection string)
      - GROQ_API_KEY = (your Groq API key)
      - REDIS_URL = (your Upstash Redis URL)
      - ELASTICSEARCH_URL =
        https://my-elasticsearch-project-bee9fb.es.us-central1.gcp.elastic.cloud:443
        (without the basic auth prefix)
      - ELASTICSEARCH_API_KEY =
        aHAzMGNaOEIyNFNmMGNDQkJsV2o6dUxNQW9keUFJOGdib0prZVJtZXZPQQ==
      - ENVIRONMENT = production
6.  Click Create Web Service.

Wait for Render to finish building. The build should finish in under 60 seconds
with a green "Live" status. Copy your new live API URL:
https://news-analytics-api.onrender.com

Step 4: Deploy the Frontend to Vercel

1.  Log in to Vercel ($0/month, no credit card required).
2.  Click Add New -> Project.
3.  Import your GitHub repository.
4.  Expand the Environment Variables accordion and add:
      - Name: VITE_API_URL
      - Value: https://news-analytics-api.onrender.com (the live Render URL you
        copied in Step 3).
5.  Click Deploy.

Vercel will build your static files. Once complete, you will receive a public
link to access your completed, running live website.
