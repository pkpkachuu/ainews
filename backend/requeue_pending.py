"""One-off script: push all pending articles back onto the Redis stream."""
import asyncio
from app.utils.supabase_client import db_client
from app.utils.redis_client import redis_client

async def main():
    db_client.connect()
    await redis_client.connect()

    pending = await db_client.get_pending_articles(limit=500)
    print(f"Found {len(pending)} pending articles")

    for article in pending:
        await redis_client.add_to_stream(
            "articles:raw",
            {
                "article_id": article["article_id"],
                "url": article["url"],
                "title": article["title"],
                "body": article["body"],
                "source": article["source"],
                "published_at": article["published_at"],
                "category": "news"
            }
        )
        print(f"Requeued: {article['title'][:60]}")

    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
