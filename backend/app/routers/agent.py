"""Agent API Router.

Handles agentic workflows for timeline generation and trend explanation.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from app.agents.workflows import generate_timeline, explain_trend

router = APIRouter(prefix="/agent", tags=["agent"])


class TimelineRequest(BaseModel):
    """Timeline generation request."""
    query: str


class TrendRequest(BaseModel):
    """Trend explanation request."""
    query: str


@router.post("/timeline")
async def create_timeline(request: TimelineRequest) -> Dict[str, Any]:
    """Generate a timeline of events for an entity or topic.

    This agent workflow:
    1. Extracts the main entity/topic from the query
    2. Retrieves relevant articles chronologically
    3. Fetches trend data
    4. Synthesizes a narrative timeline using LLM

    Returns:
        - query: Original query
        - entity: Extracted entity name
        - narrative: Brief story of how events unfolded
        - timeline: Chronological list of events with articles
        - article_count: Total articles found
    """
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters"
        )

    result = await generate_timeline(request.query)

    return result


@router.post("/explain-trend")
async def explain_trend_reason(request: TrendRequest) -> Dict[str, Any]:
    """Explain why a topic is trending.

    This agent workflow:
    1. Extracts the topic from the query
    2. Fetches trend analytics data
    3. Retrieves supporting articles
    4. Generates an explanation using LLM

    Returns:
        - query: Original query
        - topic: Extracted topic name
        - explanation: Why this topic is trending
        - trend_data: Supporting analytics data
        - article_count: Total supporting articles
    """
    if not request.query or len(request.query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters"
        )

    result = await explain_trend(request.query)

    return result


@router.post("/ask")
async def ask_analyst(request: dict) -> Dict[str, Any]:
    """Ask the analyst a general question.

    Routes to appropriate workflow based on query intent.
    """
    query = request.get("query", "")

    if not query or len(query.strip()) < 3:
        raise HTTPException(
            status_code=400,
            detail="Query must be at least 3 characters"
        )

    query_lower = query.lower()

    # Route based on intent
    if any(kw in query_lower for kw in ["timeline", "chronology", "history", "what happened", "time"]):
        return await generate_timeline(query)
    elif any(kw in query_lower for kw in ["why", "trending", "popular", "spiking", "surge"]):
        return await explain_trend(query)
    else:
        # Default to timeline for general queries
        return await generate_timeline(query)
