"""LangGraph Agent Workflows.

Thin orchestration layer for:
- Timeline generation: chronological event extraction
- Trend explanation: synthesize why a topic is trending

Agents orchestrate retrieval and analytics services, then use Groq for synthesis.
"""

from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime, timezone
import structlog
from langgraph.graph import StateGraph, END

from app.config import settings
from app.services.temporal_analytics import temporal_analytics
from app.utils.elasticsearch_client import es_client
from app.utils.supabase_client import db_client

logger = structlog.get_logger()

# Groq client (lazy loaded)
_groq_client = None


def get_groq_client():
    """Lazy load Groq client."""
    global _groq_client
    if _groq_client is None and settings.groq_api_key:
        from groq import Groq
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


# State definitions for workflows
class TimelineState(TypedDict):
    """State for timeline generation workflow."""
    query: str
    entity_or_topic: Optional[str]
    retrieved_articles: List[Dict[str, Any]]
    trend_data: Optional[Dict[str, Any]]
    timeline: List[Dict[str, Any]]
    narrative: str
    turns: int


class TrendExplanationState(TypedDict):
    """State for trend explanation workflow."""
    query: str
    entity_or_topic: str
    trend_data: Optional[Dict[str, Any]]
    retrieved_articles: List[Dict[str, Any]]
    explanation: str
    turns: int


# Timeline Generation Workflow
async def extract_entity(state: TimelineState) -> TimelineState:
    """Extract main entity or topic from query."""
    query = state["query"]
    model = get_groq_client()

    if not model:
        # Fallback: use query as-is
        return {**state, "entity_or_topic": query, "turns": state["turns"] + 1}

    try:
        response = model.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Extract the main entity (person, organization, or topic) from the query. Return just the entity name, nothing else."
                },
                {"role": "user", "content": query}
            ],
            max_tokens=50,
            temperature=0
        )
        entity = response.choices[0].message.content.strip()
        return {**state, "entity_or_topic": entity, "turns": state["turns"] + 1}
    except Exception as e:
        logger.warning("entity_extraction_failed", error=str(e))
        return {**state, "entity_or_topic": query, "turns": state["turns"] + 1}


async def retrieve_timeline_articles(state: TimelineState) -> TimelineState:
    """Retrieve articles for timeline, ordered chronologically."""
    entity = state.get("entity_or_topic", state["query"])

    # Get articles from database
    articles = await db_client.get_articles_by_entity(entity, limit=50)

    if not articles:
        # Try keyword search
        results = await es_client.search_bm25(entity, size=50)
        articles = results

    # Sort by published_at
    articles.sort(key=lambda x: x.get("published_at", "") or "")

    return {**state, "retrieved_articles": articles, "turns": state["turns"] + 1}


async def fetch_trend(state: TimelineState) -> TimelineState:
    """Fetch trend data for the entity."""
    entity = state.get("entity_or_topic", state["query"])

    trend_data = await temporal_analytics.get_entity_trend(entity, days=30)

    return {**state, "trend_data": trend_data, "turns": state["turns"] + 1}


async def synthesize_timeline(state: TimelineState) -> TimelineState:
    """Synthesize timeline with LLM."""
    model = get_groq_client()
    articles = state["retrieved_articles"]
    entity = state.get("entity_or_topic", state["query"])

    if not articles:
        return {
            **state,
            "timeline": [],
            "narrative": f"No articles found for {entity}.",
            "turns": state["turns"] + 1
        }

    # Group articles by date
    from collections import defaultdict
    by_date = defaultdict(list)
    for article in articles:
        pub_date = article.get("published_at", "")
        if pub_date:
            date_str = pub_date[:10] if len(pub_date) >= 10 else pub_date
            by_date[date_str].append(article)

    # Build timeline with events
    timeline = []
    for date, date_articles in sorted(by_date.items()):
        # Summarize this date's events
        titles = [a.get("title", "") for a in date_articles[:3]]
        sources = [a.get("source", "") for a in date_articles]
        avg_sentiment = sum(a.get("sentiment_score", 0) for a in date_articles if a.get("sentiment_score")) / len(date_articles) if date_articles else 0

        timeline.append({
            "date": date,
            "event_count": len(date_articles),
            "headline_titles": titles,
            "sources": list(set(sources)),
            "sentiment": round(avg_sentiment, 3)
        })

    # Generate narrative with LLM
    if model and len(timeline) > 0:
        try:
            # Build context
            context = "\n".join([
                f"{t['date']}: {len(t['headline_titles'])} articles - {', '.join(t['headline_titles'][:2])}"
                for t in timeline[-10:]  # Last 10 events
            ])

            response = model.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a news analyst. Generate a 2-3 sentence narrative explaining the progression of events, focusing on how the story developed over time."
                    },
                    {
                        "role": "user",
                        "content": f"Write a brief narrative about {entity} based on this timeline:\n\n{context}"
                    }
                ],
                max_tokens=200,
                temperature=0.7
            )
            narrative = response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning("timeline_narrative_failed", error=str(e))
            narrative = f"Coverage of {entity} spans {len(timeline)} days with {len(articles)} total articles."
    else:
        narrative = f"Found {len(articles)} articles for {entity} spanning {len(timeline)} days."

    return {
        **state,
        "timeline": timeline,
        "narrative": narrative,
        "turns": state["turns"] + 1
    }


def should_fetch_trend(state: TimelineState) -> str:
    """Route based on article count."""
    if state["turns"] >= 4:
        return "synthesize"
    article_count = len(state.get("retrieved_articles", []))
    if article_count < 3:
        return "synthesize"  # No need for trend if few articles
    return "fetch_trend"


# Build timeline generation graph
def build_timeline_graph():
    """Build the timeline generation workflow graph."""
    graph = StateGraph(TimelineState)

    graph.add_node("extract_entity", extract_entity)
    graph.add_node("retrieve", retrieve_timeline_articles)
    graph.add_node("fetch_trend", fetch_trend)
    graph.add_node("synthesize", synthesize_timeline)

    graph.set_entry_point("extract_entity")
    graph.add_edge("extract_entity", "retrieve")
    graph.add_conditional_edges("retrieve", should_fetch_trend, {
        "fetch_trend": "fetch_trend",
        "synthesize": "synthesize"
    })
    graph.add_edge("fetch_trend", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Trend Explanation Workflow
async def extract_trend_topic(state: TrendExplanationState) -> TrendExplanationState:
    """Extract topic from trend query."""
    query = state["query"]

    # Simple extraction - look for quoted phrases or capitalized words
    import re
    quoted = re.search(r'"([^"]+)"', query)
    if quoted:
        topic = quoted.group(1)
    else:
        # Use the last meaningful word(s)
        words = query.split()
        if len(words) > 2:
            topic = " ".join(words[-3:])
        else:
            topic = query

    return {**state, "entity_or_topic": topic, "turns": state["turns"] + 1}


async def fetch_trend_data(state: TrendExplanationState) -> TrendExplanationState:
    """Fetch trend analytics data."""
    topic = state.get("entity_or_topic", state["query"])

    # Try as entity first, then as topic/keyword
    trend_data = await temporal_analytics.get_entity_trend(topic, days=7)

    if "error" in trend_data:
        trend_data = await temporal_analytics.get_topic_trend(topic, days=7)

    return {**state, "trend_data": trend_data, "turns": state["turns"] + 1}


async def retrieve_supporting_articles(state: TrendExplanationState) -> TrendExplanationState:
    """Retrieve articles to support explanation."""
    topic = state.get("entity_or_topic", "")
    trend = state.get("trend_data", {})

    # Get recent articles for the topic
    articles = await db_client.get_articles_by_entity(topic, limit=10)

    if not articles:
        # Fallback to keyword search
        articles = await es_client.search_bm25(topic, size=10)

    return {**state, "retrieved_articles": articles, "turns": state["turns"] + 1}


async def generate_trend_explanation(state: TrendExplanationState) -> TrendExplanationState:
    """Generate trend explanation with LLM."""
    model = get_groq_client()
    topic = state.get("entity_or_topic", state["query"])
    trend = state.get("trend_data", {})
    articles = state.get("retrieved_articles", [])

    if not model:
        # Fallback explanation
        direction = trend.get("trend_direction", "unknown")
        spike_ratio = trend.get("spike_ratio", 1.0)
        return {
            **state,
            "explanation": f"{topic} is {direction.lower()}. Recent coverage is {spike_ratio:.1f}x the baseline average.",
            "turns": state["turns"] + 1
        }

    try:
        # Build context
        article_context = "\n".join([
            f"- {a.get('title', 'Untitled')} ({a.get('source', 'Unknown')})"
            for a in articles[:5]
        ])

        trend_info = f"Direction: {trend.get('trend_direction', 'stable')}, Spike ratio: {trend.get('spike_ratio', 1.0):.1f}x, Recent articles: {trend.get('stats', {}).get('total_articles', 0)}"

        response = model.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a news analyst. Explain why a topic is trending based on the data. Be specific and concise (2-3 sentences)."
                },
                {
                    "role": "user",
                    "content": f"""Why is {topic} trending now?

Trend data: {trend_info}

Recent headlines:
{article_context}

Provide a brief explanation focusing on the recent developments."""
                }
            ],
            max_tokens=200,
            temperature=0.7
        )

        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("trend_explanation_failed", error=str(e))
        explanation = f"{topic} shows {trend.get('trend_direction', 'stable')} trend with recent increased coverage."

    return {**state, "explanation": explanation, "turns": state["turns"] + 1}


def should_retrieve_articles(state: TrendExplanationState) -> str:
    """Route based on trend data availability."""
    if state["turns"] >= 3:
        return "explain"
    trend = state.get("trend_data", {})
    if "error" in trend or not trend.get("stats", {}).get("total_articles", 0):
        return "explain"  # Skip retrieval if no trend data
    return "retrieve"


# Build trend explanation graph
def build_trend_graph():
    """Build the trend explanation workflow graph."""
    graph = StateGraph(TrendExplanationState)

    graph.add_node("extract_topic", extract_trend_topic)
    graph.add_node("fetch_trend", fetch_trend_data)
    graph.add_node("retrieve", retrieve_supporting_articles)
    graph.add_node("explain", generate_trend_explanation)

    graph.set_entry_point("extract_topic")
    graph.add_edge("extract_topic", "fetch_trend")
    graph.add_conditional_edges("fetch_trend", should_retrieve_articles, {
        "retrieve": "retrieve",
        "explain": "explain"
    })
    graph.add_edge("retrieve", "explain")
    graph.add_edge("explain", END)

    return graph.compile()


# Compiled workflow instances
timeline_workflow = None
trend_workflow = None


def get_timeline_workflow():
    """Get or create timeline workflow."""
    global timeline_workflow
    if timeline_workflow is None:
        timeline_workflow = build_timeline_graph()
    return timeline_workflow


def get_trend_workflow():
    """Get or create trend workflow."""
    global trend_workflow
    if trend_workflow is None:
        trend_workflow = build_trend_graph()
    return trend_workflow


# Convenience functions for API
async def generate_timeline(query: str) -> Dict[str, Any]:
    """Generate timeline for a query.

    Args:
        query: User query about an entity or topic

    Returns:
        Timeline with narrative and events
    """
    workflow = get_timeline_workflow()

    initial_state: TimelineState = {
        "query": query,
        "entity_or_topic": None,
        "retrieved_articles": [],
        "trend_data": None,
        "timeline": [],
        "narrative": "",
        "turns": 0
    }

    result = await workflow.ainvoke(initial_state)

    return {
        "query": query,
        "entity": result.get("entity_or_topic"),
        "narrative": result.get("narrative"),
        "timeline": result.get("timeline", []),
        "article_count": len(result.get("retrieved_articles", [])),
        "turns": result.get("turns", 0)
    }


async def explain_trend(query: str) -> Dict[str, Any]:
    """Explain why a topic is trending.

    Args:
        query: User query about trending topic

    Returns:
        Explanation with supporting data
    """
    workflow = get_trend_workflow()

    initial_state: TrendExplanationState = {
        "query": query,
        "entity_or_topic": "",
        "trend_data": None,
        "retrieved_articles": [],
        "explanation": "",
        "turns": 0
    }

    result = await workflow.ainvoke(initial_state)

    return {
        "query": query,
        "topic": result.get("entity_or_topic"),
        "explanation": result.get("explanation"),
        "trend_data": result.get("trend_data"),
        "article_count": len(result.get("retrieved_articles", [])),
        "turns": result.get("turns", 0)
    }
