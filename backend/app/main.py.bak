"""Main FastAPI Application.

AI-Powered Intelligent News Analytics Platform.

API Endpoints:
- /search - Hybrid search with reranking
- /intelligence - Proactive intelligence feed, trends, analytics
- /agent - Agentic workflows (timeline, trend explanation)
- /articles - Article CRUD operations
- /ingest - Manual ingestion triggers
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from app.config import settings
from app.utils.redis_client import redis_client
from app.utils.elasticsearch_client import es_client
from app.utils.supabase_client import db_client

from app.routers import search, intelligence, agent, articles

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Handles startup and shutdown of external connections.
    """
    # Startup
    logger.info("application_starting", environment=settings.environment)

    # Connect to services
    await redis_client.connect()
    db_client.connect()  # Sync for Supabase REST client
    await es_client.connect()

    # Create ES indices if needed
    await es_client.create_indices()

    logger.info("application_started", services=["redis", "supabase", "elasticsearch"])

    yield

    # Shutdown
    logger.info("application_shutting_down")

    await redis_client.disconnect()
    db_client.disconnect()
    await es_client.disconnect()

    logger.info("application_stopped")


# Create FastAPI app with metadata
app = FastAPI(
    title="News Analytics Platform",
    description="AI-Powered Intelligent News Analytics & Agentic Search Platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search.router)
app.include_router(intelligence.router)
app.include_router(agent.router)
app.include_router(articles.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "environment": settings.environment,
        "services": {
            "redis": "connected" if redis_client.client else "disconnected",
            "elasticsearch": "connected" if es_client.client else "disconnected",
            "supabase": "connected" if db_client.client else "disconnected"
        }
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "News Analytics Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "search": "/search",
            "intelligence": "/intelligence/feed",
            "trending": "/intelligence/trending",
            "agent": "/agent/timeline",
            "articles": "/articles"
        }
    }


# Manual ingestion endpoints
@app.post("/ingest/fetch")
async def trigger_fetch():
    """Manually trigger news fetching from RSS feeds."""
    from app.services.fetcher import news_fetcher

    await news_fetcher.start()
    stats = await news_fetcher.fetch_all_feeds()
    await news_fetcher.stop()

    return {
        "message": "Fetch completed",
        "stats": stats
    }


@app.post("/ingest/process")
async def trigger_processing():
    """Manually trigger processing of pending articles.

    Processes pending articles through NLP pipeline.
    """
    pending = await db_client.get_pending_articles(limit=10)

    if not pending:
        return {"message": "No pending articles to process"}

    # Note: In production, this would spawn workers
    # For now, it just returns pending article count
    return {
        "message": "Found pending articles",
        "count": len(pending),
        "article_ids": [a["article_id"] for a in pending]
    }


@app.on_event("startup")
async def startup_event():
    """Log startup event."""
    logger.info(
        "api_server_started",
        elasticsearch_url=settings.elasticsearch_url,
        redis_url=settings.redis_url
    )
