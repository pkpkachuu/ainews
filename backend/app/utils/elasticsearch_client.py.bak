"""Elasticsearch client and index management."""

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
from typing import Optional, List, Dict, Any
import structlog

from app.config import settings

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
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
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
    },
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    }
}


class ElasticsearchClient:
    """Async Elasticsearch client wrapper."""

    def __init__(self):
        self.client: Optional[AsyncElasticsearch] = None

    async def connect(self) -> None:
        """Initialize Elasticsearch connection."""
        self.client = AsyncElasticsearch(
            [settings.elasticsearch_url],
            verify_certs=False
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
        """Get articles from recent time window."""
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
        hours: int = 24
    ) -> Dict[str, Any]:
        """Get sentiment history for an entity."""
        if not self.client:
            raise RuntimeError("Elasticsearch not connected")

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
                                "published_at": {"gte": f"now-{hours}h"}
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
