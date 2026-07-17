"""News ingestion fetcher service.

Fetches articles from RSS feeds and news APIs, stores raw content,
and publishes to Redis stream for processing.
"""

import feedparser
import trafilatura
import aiohttp
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import hashlib
import re
import structlog

from app.config import settings
from app.utils.redis_client import redis_client
from app.utils.supabase_client import db_client

logger = structlog.get_logger()

# RSS feed sources (configured for MVP)
RSS_FEEDS = [
    {"name": "BBC World", "url": "https://feeds.bbci.co.uk/news/world/rss.xml", "category": "world"},
    {"name": "BBC Technology", "url": "https://feeds.bbci.co.uk/news/technology/rss.xml", "category": "technology"},
    {"name": "Reuters", "url": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best", "category": "news"},
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "technology"},
    {"name": "The Guardian", "url": "https://www.theguardian.com/world/rss", "category": "world"},
    {"name": "Ars Technica", "url": "https://feeds.arstechnica.com/arstechnica/index", "category": "technology"},
    {"name": "Wired", "url": "https://www.wired.com/feed/rss", "category": "technology"},
    {"name": "Euronews", "url": "https://www.euronews.com/rss", "category": "europe"},
]

# Titles matching these patterns aren't news articles - they're recurring
# template/index pages (TV bulletin landing pages, programme grids, live
# blogs re-published each edition). Their "content" is mostly boilerplate
# (show names, section headers), which pollutes NER/keyphrase extraction
# and makes near-identical editions falsely cluster as an "emerging topic"
# in DBSCAN since their embeddings are almost identical to each other.
NON_ARTICLE_TITLE_PATTERNS = [
    r"^latest news bulletin\b",
    r"\bmorning bulletin\b",
    r"\bevening bulletin\b",
    r"^live:?\s",
    r"\bliveblog\b",
]


def _looks_like_non_article(title: str) -> bool:
    """Heuristic filter for template/digest pages that aren't real articles."""
    if not title:
        return False
    title_lower = title.lower()
    return any(re.search(pattern, title_lower) for pattern in NON_ARTICLE_TITLE_PATTERNS)


class NewsFetcher:
    """Fetches news articles from RSS feeds and APIs."""

    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.stats = {
            "fetched": 0,
            "skipped": 0,
            "failed": 0
        }

    async def start(self) -> None:
        """Initialize HTTP session."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "NewsAnalyticsBot/1.0"}
        )

    async def stop(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()

    async def fetch_all_feeds(self) -> Dict[str, int]:
        """Fetch all configured RSS feeds.

        Returns:
            Statistics about fetched articles.
        """
        self.stats = {"fetched": 0, "skipped": 0, "failed": 0}

        for feed_config in RSS_FEEDS:
            try:
                await self._fetch_feed(feed_config)
            except Exception as e:
                logger.error(
                    "feed_fetch_failed",
                    feed=feed_config["name"],
                    error=str(e)
                )
                self.stats["failed"] += 1

        logger.info("fetch_complete", **self.stats)
        return self.stats

    async def _fetch_feed(self, feed_config: Dict[str, Any]) -> None:
        """Fetch and process a single RSS feed."""
        logger.info("fetching_feed", feed=feed_config["name"])

        try:
            async with self.session.get(feed_config["url"]) as response:
                if response.status != 200:
                    logger.warning(
                        "feed_response_error",
                        feed=feed_config["name"],
                        status=response.status
                    )
                    return

                content = await response.text()

        except Exception as e:
            logger.error("feed_request_failed", feed=feed_config["name"], error=str(e))
            return

        # Parse RSS feed
        feed = feedparser.parse(content)

        if feed.bozo and not feed.entries:
            logger.warning(
                "feed_parse_error",
                feed=feed_config["name"],
                error=str(feed.bozo_exception) if feed.bozo_exception else "Unknown"
            )
            return

        for entry in feed.entries:
            try:
                await self._process_entry(entry, feed_config)
            except Exception as e:
                logger.error(
                    "entry_process_failed",
                    url=entry.get("link", "unknown"),
                    error=str(e)
                )

    async def _process_entry(
        self,
        entry: Dict[str, Any],
        feed_config: Dict[str, Any]
    ) -> None:
        """Process a single RSS entry."""
        url = entry.get("link")
        if not url:
            return

        # Check for duplicate
        if await db_client.article_exists(url):
            self.stats["skipped"] += 1
            return

        # Extract content
        title = entry.get("title", "")

        if _looks_like_non_article(title):
            self.stats["skipped"] += 1
            logger.debug("skipped_non_article", title=title[:60])
            return

        published = entry.get("published_parsed") or entry.get("updated_parsed")

        if published:
            published_at = datetime(*published[:6], tzinfo=timezone.utc)
        else:
            published_at = datetime.now(timezone.utc)

        # Fetch full article content
        body = await self._extract_content(url)

        # trafilatura fails silently on paywalled, bot-blocked, or JS-rendered
        # pages (returns ""). RSS entries almost always carry a summary/
        # description too - falling back to that keeps the article alive
        # instead of it silently dying at the nlp_worker's short-text skip.
        if not body:
            summary = entry.get("summary") or entry.get("description") or ""
            if summary:
                # RSS summaries are sometimes HTML - strip tags for plain text
                import re as _re
                body = _re.sub(r"<[^>]+>", " ", summary).strip()

        # Generate unique ID
        article_id = hashlib.sha256(url.encode()).hexdigest()[:36]

        # Store raw article
        article_data = {
            "url": url,
            "title": title,
            "body": body,
            "source": feed_config["name"],
            "published_at": published_at.isoformat(),
            "category": feed_config.get("category", "news"),
            "raw_storage_path": None,  # Will be set if stored to S3
        }

        # Create article in database
        db_article_id = await db_client.create_article(article_data)

        # Publish to Redis stream for NLP processing
        await redis_client.add_to_stream(
            "articles:raw",
            {
                "article_id": db_article_id,
                "url": url,
                "title": title,
                "body": body,
                "source": feed_config["name"],
                "published_at": published_at.isoformat(),
                "category": feed_config.get("category", "news")
            }
        )

        self.stats["fetched"] += 1
        logger.debug(
            "article_queued",
            article_id=db_article_id,
            title=title[:50]
        )

    async def _extract_content(self, url: str) -> str:
        """Extract article content from URL using trafilatura."""
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    return ""

                html = await response.text()

            # Use trafilatura to extract main content
            content = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True
            )

            return content or ""

        except Exception as e:
            logger.warning("content_extraction_failed", url=url, error=str(e))
            return ""

    async def fetch_from_newsapi(
        self,
        query: str = None,
        category: str = None,
        page_size: int = 50
    ) -> List[Dict[str, Any]]:
        """Fetch articles from NewsAPI if configured."""
        if not settings.newsapi_key:
            logger.warning("newsapi_key_not_configured")
            return []

        url = "https://newsapi.org/v2/everything"
        params = {
            "apiKey": settings.newsapi_key,
            "pageSize": page_size,
            "language": "en",
            "sortBy": "publishedAt"
        }

        if query:
            params["q"] = query
        if category:
            params["category"] = category

        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()

                if data.get("status") != "ok":
                    logger.warning("newsapi_error", error=data.get("message"))
                    return []

                articles = []
                for article in data.get("articles", []):
                    url = article.get("url")
                    if not url or await db_client.article_exists(url):
                        continue

                    article_data = {
                        "url": url,
                        "title": article.get("title", ""),
                        "body": article.get("description") or article.get("content", ""),
                        "source": article.get("source", {}).get("name", "NewsAPI"),
                        "published_at": article.get("publishedAt"),
                        "category": category or "news"
                    }

                    db_id = await db_client.create_article(article_data)
                    article_data["article_id"] = db_id
                    articles.append(article_data)

                    # Queue for NLP
                    await redis_client.add_to_stream("articles:raw", article_data)

                return articles

        except Exception as e:
            logger.error("newsapi_fetch_failed", error=str(e))
            return []


# Global fetcher instance
news_fetcher = NewsFetcher()
