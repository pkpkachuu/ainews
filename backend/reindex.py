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
        # Relational join: Fetch all articles, entities, and keyphrases in ONE single network request
        response = db_client.client.table("articles").select(
            "*, article_entities(entity_text, entity_type, confidence), article_keyphrases(keyphrase)"
        ).eq("processing_status", "done").execute()
        
        articles = response.data
        logger.info("fetched_all_data_from_supabase", count=len(articles))

        es_articles = []
        for article in articles:
            # Map Supabase relational join arrays to expected Elasticsearch format
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

            # Safely parse embeddings if they exist
            embedding = article.get("embedding")
            if embedding:
                if isinstance(embedding, str):
                    article_doc["embedding"] = [float(x) for x in embedding.replace("[", "").replace("]", "").split(",")]
                else:
                    article_doc["embedding"] = embedding

            es_articles.append(article_doc)

        # Bulk index to Elasticsearch in batches of 100 for safety and speed
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
