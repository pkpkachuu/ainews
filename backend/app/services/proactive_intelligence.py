"""Proactive Intelligence Engine (PIE).

Core component that generates intelligence without user queries:
- Emergence Detector: Finds new topic clusters every 30 minutes
- Narrative Monitor: Tracks sentiment shifts for entities every hour
- Feed Generator: Creates intelligence feed twice daily
"""

import json
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from collections import Counter
import structlog
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.utils.redis_client import redis_client
from app.utils.elasticsearch_client import es_client
from app.utils.supabase_client import db_client

logger = structlog.get_logger()

# Redis key constants
INTELLIGENCE_FEED_KEY = "intelligence:feed"
INTELLIGENCE_CANDIDATES_KEY = "intelligence:candidates"
FEED_TTL = 50400  # 14 hours
CANDIDATES_TTL = 172800  # 48 hours

# Groq LLM client (will be initialized when needed)
_groq_client = None


def get_groq_client():
    """Lazy load Groq client."""
    global _groq_client
    if _groq_client is None and settings.groq_api_key:
        from groq import Groq
        _groq_client = Groq(api_key=settings.groq_api_key)
    return _groq_client


class EmergenceDetector:
    """Detects emerging topic clusters not seen in prior 48 hours."""

    def __init__(self):
        self.min_articles = 5
        self.similarity_threshold = 0.85

    async def detect(self) -> List[Dict[str, Any]]:
        """Run emergence detection on recent articles.

        Returns:
            List of emerging topic candidates
        """
        logger.info("emergence_detection_starting")

        # Get articles from last 2 hours
        recent_articles = await es_client.get_recent_articles(hours=720, size=500)  # widened for dev/testing backlog data

        if len(recent_articles) < self.min_articles:
            logger.info("emergence_insufficient_articles", count=len(recent_articles))
            return []

        # Extract embeddings
        embeddings = []
        valid_articles = []
        for article in recent_articles:
            if article.get("embedding"):
                embeddings.append(article["embedding"])
                valid_articles.append(article)

        if len(embeddings) < self.min_articles:
            logger.info("emergence_insufficient_embeddings", count=len(embeddings))
            return []

        embeddings_array = np.array(embeddings)

        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=0.15, min_samples=self.min_articles, metric="cosine")
        labels = db.fit_predict(embeddings_array)
        unique_labels = sorted(set(labels) - {-1})

        existing_centroids = await redis_client.get_all_cluster_centroids()

        emerging_topics = []

        MAX_CLUSTER_SIZE = 40

        def split_oversized(article_group, embedding_group):
            if len(article_group) <= MAX_CLUSTER_SIZE:
                return [(article_group, embedding_group)]

            tight_db = DBSCAN(eps=0.08, min_samples=self.min_articles, metric="cosine")
            tight_labels = tight_db.fit_predict(embedding_group)
            tight_unique = sorted(set(tight_labels) - {-1})

            if not tight_unique:
                return [(article_group[:MAX_CLUSTER_SIZE], embedding_group[:MAX_CLUSTER_SIZE])]

            sub_groups = []
            for t_id in tight_unique:
                t_indices = np.where(tight_labels == t_id)[0]
                sub_articles = [article_group[i] for i in t_indices]
                sub_embeddings = embedding_group[t_indices]
                if len(sub_articles) >= self.min_articles:
                    sub_groups.append((sub_articles[:MAX_CLUSTER_SIZE], sub_embeddings[:MAX_CLUSTER_SIZE]))
            return sub_groups or [(article_group[:MAX_CLUSTER_SIZE], embedding_group[:MAX_CLUSTER_SIZE])]

        for cluster_id in unique_labels:
            cluster_indices = np.where(labels == cluster_id)[0]
            cluster_articles = [valid_articles[i] for i in cluster_indices]
            cluster_embeddings = embeddings_array[cluster_indices]

            if len(cluster_articles) < self.min_articles:
                continue

            for sub_idx, (sub_articles, sub_embeddings) in enumerate(split_oversized(cluster_articles, cluster_embeddings)):
                if len(sub_articles) < self.min_articles:
                    continue

                centroid = np.mean(sub_embeddings, axis=0)

                is_emerging = True
                for existing_id, existing_centroid in existing_centroids.items():
                    existing_array = np.array(existing_centroid)
                    similarity = cosine_similarity([centroid], [existing_array])[0][0]
                    if similarity > self.similarity_threshold:
                        is_emerging = False
                        break

                if is_emerging:
                    topic_candidate = await self._create_topic_candidate(
                        sub_articles,
                        centroid.tolist(),
                        cluster_id
                    )
                    emerging_topics.append(topic_candidate)

                cluster_key = f"clust_{datetime.now().strftime('%Y%m%d_%H%M')}_{cluster_id}_{sub_idx}"
                await redis_client.set_embeddings_centroid(cluster_key, centroid.tolist())

        logger.info(
            "emergence_detection_complete",
            clusters=len(unique_labels),
            emerging=len(emerging_topics)
        )

        return emerging_topics

    async def _create_topic_candidate(
        self,
        articles: List[Dict[str, Any]],
        centroid: List[float],
        cluster_id: int
    ) -> Dict[str, Any]:
        """Create topic candidate from cluster."""
        # Extract common entities
        all_entities = []
        all_keyphrases = []

        for article in articles:
            entities = article.get("entities", [])
            if isinstance(entities, list):
                for e in entities:
                    if isinstance(e, dict) and e.get("text"):
                        all_entities.append(e["text"])
                    elif isinstance(e, str):
                        all_entities.append(e)

            keyphrases = article.get("keyphrases", [])
            if isinstance(keyphrases, list):
                for kp in keyphrases:
                    if isinstance(kp, str):
                        all_keyphrases.append(kp)
                    elif isinstance(kp, dict) and kp.get("phrase"):
                        all_keyphrases.append(kp["phrase"])

        entity_counts = Counter(all_entities)
        phrase_counts = Counter(all_keyphrases)

        top_entities = [e for e, _ in entity_counts.most_common(5)]

        LABEL_STOPWORDS = {
            "they", "that", "this", "it", "he", "she", "we", "you", "i",
            "them", "these", "those", "there", "here", "what", "who"
        }
        filtered_phrases = [
            p for p, _ in phrase_counts.most_common(10)
            if p.strip().lower() not in LABEL_STOPWORDS and len(p.strip()) > 2
        ]
        top_phrases = filtered_phrases[:3]

        # Generate candidate label - prefer entities over raw keyphrases
        candidate_label = top_entities[0] if top_entities else top_phrases[0] if top_phrases else f"Topic {cluster_id}"

        # Get earliest article time
        timestamps = []
        for a in articles:
            published = a.get("published_at")
            if published:
                timestamps.append(published)

        first_seen = min(timestamps) if timestamps else datetime.now(timezone.utc).isoformat()

        return {
            "type": "EMERGING_TOPIC",
            "cluster_id": f"clust_{datetime.now().strftime('%Y%m%d_%H%M')}_{cluster_id}",
            "candidate_label": candidate_label,
            "article_count": len(articles),
            "key_entities": top_entities,
            "keyphrases": top_phrases,
            "first_article_at": first_seen,
            "centroid_embedding": centroid,
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "confidence_score": round(len(articles) / 50, 2)  # Simple confidence
        }


class NarrativeMonitor:
    """Tracks sentiment evolution for tracked entities."""

    def __init__(self):
        self.sentiment_shift_threshold = 0.3
        self.sentiment_reversal_threshold = 0.5
        self.divergence_threshold = 0.4
        self.min_articles = 10

    async def monitor(self) -> List[Dict[str, Any]]:
        """Run narrative monitoring for tracked entities.

        Returns:
            List of narrative shift candidates
        """
        logger.info("narrative_monitor_starting")

        # Get tracked entities
        tracked = await db_client.get_tracked_entities(active_only=True)

        if not tracked:
            # Auto-track entities with recent activity
            await self._auto_track_entities()
            tracked = await db_client.get_tracked_entities(active_only=True)

        narrative_shifts = []

        for entity in tracked:
            shift = await self._check_entity_sentiment(entity)
            if shift:
                narrative_shifts.append(shift)

        logger.info(
            "narrative_monitor_complete",
            entities_checked=len(tracked),
            shifts=len(narrative_shifts)
        )

        return narrative_shifts

    async def _auto_track_entities(self) -> None:
        """Auto-track entities with high activity."""
        # Get entities with most articles in past 7 days
        from app.utils.elasticsearch_client import es_client

        # Query for top entities by count
        try:
            result = await es_client.client.search(
                index="articles",
                body={
                    "query": {
                        "range": {
                            "published_at": {"gte": "now-7d"}
                        }
                    },
                    "aggs": {
                        "top_entities": {
                            "nested": {"path": "entities"},
                            "aggs": {
                                "entity_names": {
                                    "terms": {
                                        "field": "entities.text",
                                        "size": 20
                                    }
                                }
                            }
                        }
                    },
                    "size": 0
                }
            )

            buckets = result["aggregations"]["top_entities"]["entity_names"]["buckets"]

            for bucket in buckets[:20]:
                entity_name = bucket["key"]
                entity_type = "ORG"  # Default, could be improved
                await db_client.add_tracked_entity(entity_name, entity_type)

        except Exception as e:
            logger.warning("auto_track_failed", error=str(e))

    async def _check_entity_sentiment(
        self,
        entity: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Check sentiment shift for a single entity."""
        entity_name = entity["entity_name"]

        # Get sentiment for current 24h window
        current = await es_client.get_entity_sentiment_history(entity_name, hours=24)
        prior = await es_client.get_entity_sentiment_history(entity_name, hours=48)

        # Calculate prior window sentiment (24-48h ago)
        # Note: This is simplified - proper implementation would query specific date ranges

        current_sentiment = current.get("avg_sentiment")
        if current_sentiment is None:
            return None

        # Get article count for threshold
        result = await es_client.client.search(
            index="articles",
            body={
                "query": {
                    "bool": {
                        "must": [
                            {
                                "nested": {
                                    "path": "entities",
                                    "query": {"term": {"entities.text": entity_name}}
                                }
                            },
                            {"range": {"published_at": {"gte": "now-7d"}}}
                        ]
                    }
                },
                "size": 0
            }
        )
        article_count = result["hits"]["total"]["value"]

        if article_count < self.min_articles:
            return None

        # Get prior period sentiment (approximate by extending window)
        # For MVP, compare current to a baseline of 0 (neutral)
        prior_sentiment = 0.0  # Would need proper date range queries

        delta = current_sentiment - prior_sentiment

        # Classify shift type
        shift_type = None
        if delta < -self.sentiment_reversal_threshold:
            shift_type = "SENTIMENT_REVERSAL_NEGATIVE"
        elif delta > self.sentiment_reversal_threshold:
            shift_type = "SENTIMENT_REVERSAL_POSITIVE"
        elif delta < -self.sentiment_shift_threshold:
            shift_type = "SENTIMENT_NEGATIVE_SHIFT"
        elif delta > self.sentiment_shift_threshold:
            shift_type = "SENTIMENT_POSITIVE_SHIFT"

        # Check for narrative divergence across sources
        std_dev = current.get("std_dev", 0)
        is_divergent = std_dev > self.divergence_threshold

        if not shift_type and not is_divergent:
            return None

        # Get supporting articles
        articles = await db_client.get_articles_by_entity(entity_name, limit=3)

        return {
            "type": "NARRATIVE_SHIFT",
            "entity": entity_name,
            "shift_type": shift_type or "NARRATIVE_DIVERGENCE",
            "sentiment_now": round(current_sentiment, 3),
            "sentiment_prior": round(prior_sentiment, 3),
            "delta": round(delta, 3),
            "std_dev": round(std_dev, 3),
            "is_divergent": is_divergent,
            "supporting_articles": [a["article_id"] for a in articles],
            "detected_at": datetime.now(timezone.utc).isoformat()
        }


class FeedGenerator:
    """Generates the intelligence feed from detected signals."""

    def __init__(self):
        self.max_feed_items = 15

    async def generate(self) -> List[Dict[str, Any]]:
        """Generate intelligence feed from candidates.

        Returns:
            List of feed items
        """
        logger.info("feed_generation_starting")

        # Get all candidates from Redis
        candidates_raw = await redis_client.lrange(INTELLIGENCE_CANDIDATES_KEY, 0, -1)

        if not candidates_raw:
            logger.info("feed_no_candidates")
            return []

        candidates = list(candidates_raw)

        # Consume the candidates now that we've read them - otherwise they
        # accumulate forever and get re-scored in every future generation cycle
        await redis_client.delete(INTELLIGENCE_CANDIDATES_KEY)

        # Consume the candidates now that we've read them - otherwise they
        # accumulate forever and get re-scored in every future generation cycle
        await redis_client.delete(INTELLIGENCE_CANDIDATES_KEY)

        # Deduplicate by entity/topic overlap
        candidates = self._deduplicate_candidates(candidates)

        # Score and rank
        scored_candidates = []
        for candidate in candidates:
            score = self._calculate_score(candidate)
            candidate["combined_score"] = score
            scored_candidates.append(candidate)

        # Sort by score
        scored_candidates.sort(key=lambda x: x["combined_score"], reverse=True)

        # Take top items
        top_items = scored_candidates[:self.max_feed_items]

        # Generate summaries with LLM
        feed_items = []
        for i, item in enumerate(top_items):
            feed_item = await self._create_feed_item(item, i)
            if feed_item:
                feed_items.append(feed_item)

        # Write to feed
        await self._write_feed(feed_items)

        logger.info("feed_generation_complete", items=len(feed_items))

        return feed_items

    def _deduplicate_candidates(
        self,
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge candidates with overlapping entities."""
        if len(candidates) <= 1:
            return candidates

        merged = []
        seen_entities = {}

        for candidate in candidates:
            # Key by main entity/topic
            if candidate["type"] == "NARRATIVE_SHIFT":
                key = candidate.get("entity", "")
            else:
                entities = candidate.get("key_entities", [])
                key = entities[0] if entities else candidate.get("candidate_label", "")

            if key in seen_entities:
                # Merge with existing
                existing_idx = seen_entities[key]
                merged[existing_idx]["article_count"] += candidate.get("article_count", 0)
            else:
                seen_entities[key] = len(merged)
                merged.append(candidate)

        return merged

    def _calculate_score(self, candidate: Dict[str, Any]) -> float:
        """Calculate combined score for ranking."""
        # Novelty score
        novelty = 1.0 if candidate["type"] == "EMERGING_TOPIC" else 0.6

        # Reach score
        article_count = candidate.get("article_count", 1)
        max_articles = 100
        reach = min(np.log(article_count + 1) / np.log(max_articles + 1), 1.0)

        # Urgency score (time-based decay)
        detected_at = candidate.get("detected_at")
        if detected_at:
            try:
                detected_time = datetime.fromisoformat(detected_at.replace("Z", "+00:00"))
                age_hours = (datetime.now(timezone.utc) - detected_time).total_seconds() / 3600
                urgency = max(0.2, 1.0 - (age_hours / 48))
            except Exception:
                urgency = 0.5
        else:
            urgency = 0.5

        # Combined score
        return 0.4 * novelty + 0.35 * reach + 0.25 * urgency

    async def _create_feed_item(
        self,
        candidate: Dict[str, Any],
        index: int
    ) -> Optional[Dict[str, Any]]:
        """Create a feed item with LLM-generated summary."""
        # Get article IDs for context
        article_ids = candidate.get("supporting_articles", [])

        # Fetch article snippets
        articles = []
        for aid in article_ids[:3]:
            article = await db_client.get_article(aid)
            if article:
                articles.append(article)

        # If no supporting articles, try to get recent ones for the topic
        if not articles and candidate["type"] == "EMERGING_TOPIC":
            # Use key entities to search
            entities = candidate.get("key_entities", [])
            if entities:
                # Simple search by entity
                for entity in entities[:1]:
                    found = await db_client.get_articles_by_entity(entity, limit=3)
                    articles.extend(found)

        # Generate title + summary via LLM, grounded in the actual articles
        intelligence = await self._generate_intelligence(candidate, articles)

        # Generate feed item ID
        now = datetime.now(timezone.utc)
        period = "AM" if now.hour < 12 else "PM"
        feed_id = f"feed_{now.strftime('%Y%m%d')}_{period}_{index:02d}"

        return {
            "feed_item_id": feed_id,
            "type": candidate["type"],
            "topic_label": intelligence["title"],
            "trigger_reason": self._get_trigger_reason(candidate),
            "summary": intelligence["summary"],
            "key_entities": intelligence["key_entities"] or candidate.get("key_entities", [candidate.get("entity", "")]),
            "supporting_article_ids": [a["article_id"] for a in articles if a.get("article_id")],
            "article_count": candidate.get("article_count", len(articles)),
            "combined_score": round(candidate.get("combined_score", 0), 2),
            "generated_at": now.isoformat()
        }

    async def _generate_intelligence(
        self,
        candidate,
        articles
    ):
        """Generate title + summary + entities together via Groq LLM, grounded in
        the actual articles rather than raw entity/phrase frequency. Falls back to
        the first article's title if the LLM is unavailable or the cluster looks
        incoherent."""
        client = get_groq_client()
        candidate_label = candidate.get("candidate_label") or candidate.get("entity", "this topic")

        if not client or not articles:
            return self._fallback_intelligence(candidate, articles)

        snippets = "\n".join(
            f"- [{a.get('source', 'Unknown')}] {a.get('title', '')}: {(a.get('body', '') or '')[:250]}\n"
            f"  (Entities: {a.get('entities', [])})"
            for a in articles[:5]
        )

        prompt = (
            f"You are a news analyst. Analyze this cluster of articles.\n\n"
            f"CONSTRAINT 1 (COHESION): Verify at least 70% of these articles discuss the "
            f"same underlying event or topic. If they are unrelated (e.g. mixed topics), "
            f"set \"valid\": false.\n\n"
            f"CONSTRAINT 2 (GROUNDING): The candidate label is '{candidate_label}'. Only use it "
            f"in your title or summary if it's actually supported by the articles below - "
            f"otherwise write a title that reflects what the articles actually share.\n\n"
            f"CONSTRAINT 3: Never speculate beyond what's stated in the articles.\n\n"
            f"Respond with ONLY a raw JSON object, no markdown, no commentary:\n"
            f"{{\n"
            f'  "valid": true,\n'
            f'  "reason": "1-sentence reason if valid is false, else empty string",\n'
            f'  "title": "A clear headline, 6-12 words, describing the actual core development",\n'
            f'  "summary": "2-3 factual sentences on what is happening",\n'
            f'  "key_entities": ["up to 5 people/orgs actually involved"]\n'
            f"}}\n\n"
            f"Articles:\n{snippets}"
        )

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a concise, factual news analyst. Respond only with raw JSON."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0.5
            )
            raw = response.choices[0].message.content.strip()

            import re
            match = re.search(r"(\{.*\})", raw, re.DOTALL)
            if match:
                parsed = json.loads(match.group(1))
                if parsed.get("valid") is False:
                    logger.info("cluster_rejected_by_llm", reason=parsed.get("reason"))
                    return self._fallback_intelligence(candidate, articles)
                return {
                    "title": str(parsed.get("title", "") or "").strip() or candidate_label,
                    "summary": str(parsed.get("summary", "") or "").strip(),
                    "key_entities": list(parsed.get("key_entities", []) or [])[:5] or candidate.get("key_entities", [])
                }
        except Exception as e:
            logger.warning("llm_intelligence_failed", error=str(e))

        return self._fallback_intelligence(candidate, articles)

    def _fallback_intelligence(
        self,
        candidate,
        articles
    ):
        """Fallback title/summary when the LLM is unavailable or fails."""
        topic = candidate.get("candidate_label") or candidate.get("entity", "this topic")
        title = articles[0]["title"] if articles else topic
        summary = self._fallback_summary(candidate, articles)
        return {
            "title": title,
            "summary": summary,
            "key_entities": candidate.get("key_entities", [])
        }

    async def _generate_summary(
        self,
        candidate: Dict[str, Any],
        articles: List[Dict[str, Any]]
    ) -> str:
        """Generate summary using Groq LLM."""
        client = get_groq_client()

        if not client:
            # Fallback summary
            return self._fallback_summary(candidate, articles)

        # Build context from articles
        context = ""
        for i, article in enumerate(articles[:3]):
            title = article.get("title", "")
            source = article.get("source", "Unknown")
            context += f"Article {i+1}: {title} (Source: {source})\n"
            if article.get("body"):
                context += f"{article['body'][:500]}...\n\n"

        # Build prompt
        topic = candidate.get("candidate_label") or candidate.get("entity", "this topic")
        entity_list = ", ".join(candidate.get("key_entities", [])[:3])

        prompt = f"""In 2-3 sentences, explain what is happening with {topic}.
Be factual and concise. Ground your explanation in these articles:

{context if context else "No specific articles available. Summarize based on the topic name."}

Key entities mentioned: {entity_list}

Start with the most important development. Do not speculate."""

        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a concise news analyst. Provide factual summaries in 2-3 sentences."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.7
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.warning("llm_summary_failed", error=str(e))
            return self._fallback_summary(candidate, articles)

    def _fallback_summary(
        self,
        candidate: Dict[str, Any],
        articles: List[Dict[str, Any]]
    ) -> str:
        """Generate fallback summary without LLM - uses the actual lead article's
        real content instead of a templated label sentence."""
        if articles:
            lead = articles[0]
            title = lead.get("title", "")
            body = (lead.get("body", "") or "").strip()
            if body:
                return f"{title}. {body[:280]}..."
            return title

        topic = candidate.get("candidate_label") or candidate.get("entity", "this topic")
        return f"Recent coverage of {topic} shows significant activity."

    def _get_trigger_reason(self, candidate: Dict[str, Any]) -> str:
        """Get human-readable trigger reason."""
        if candidate["type"] == "EMERGING_TOPIC":
            return f"{candidate.get('article_count', 0)} new articles in 2h; no prior cluster match"
        else:
            return f"Sentiment {candidate.get('shift_type', 'shift').replace('_', ' ').lower()} of {abs(candidate.get('delta', 0)):.2f}"

    async def _write_feed(self, feed_items: List[Dict[str, Any]]) -> None:
        """Write feed items to Redis."""
        # Clear existing feed
        await redis_client.delete(INTELLIGENCE_FEED_KEY)

        # Add items (most recent first)
        for item in reversed(feed_items):
            await redis_client.lpush(INTELLIGENCE_FEED_KEY, item)

        # Set TTL
        async with redis_client.client:
            await redis_client.client.expire(INTELLIGENCE_FEED_KEY, FEED_TTL)


class ProactiveIntelligenceEngine:
    """Main orchestrator for proactive intelligence."""

    def __init__(self):
        self.emergence_detector = EmergenceDetector()
        self.narrative_monitor = NarrativeMonitor()
        self.feed_generator = FeedGenerator()

    async def run_emergence_detection(self) -> List[Dict[str, Any]]:
        """Run emergence detection and store candidates."""
        emerging = await self.emergence_detector.detect()

        for topic in emerging:
            await redis_client.lpush(INTELLIGENCE_CANDIDATES_KEY, topic)

        return emerging

    async def run_narrative_monitoring(self) -> List[Dict[str, Any]]:
        """Run narrative monitoring and store candidates."""
        shifts = await self.narrative_monitor.monitor()

        for shift in shifts:
            await redis_client.lpush(INTELLIGENCE_CANDIDATES_KEY, shift)

        return shifts

    async def generate_feed(self) -> List[Dict[str, Any]]:
        """Generate and publish intelligence feed."""
        return await self.feed_generator.generate()

    async def get_feed(self) -> List[Dict[str, Any]]:
        """Get current intelligence feed."""
        items = await redis_client.lrange(INTELLIGENCE_FEED_KEY, 0, -1)
        return items


# Global PIE instance
pie_engine = ProactiveIntelligenceEngine()
