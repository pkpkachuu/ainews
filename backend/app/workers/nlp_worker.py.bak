"""NLP Processing Worker.

Consumes articles from Redis stream and performs:
- Named Entity Recognition (spaCy)
- Sentiment Analysis
- Topic Classification
- Keyphrase Extraction
- Embedding Generation
"""

import asyncio
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import structlog

from app.config import settings
from app.utils.redis_client import redis_client
from app.utils.elasticsearch_client import es_client
from app.utils.supabase_client import db_client

logger = structlog.get_logger()

# Global model instances (lazy loaded)
_spacy_nlp = None
_embedding_model = None
_sentiment_pipeline = None


def get_spacy_model():
    """Lazy load spaCy NER model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy
        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download if not present
            from spacy.cli import download
            download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def get_embedding_model():
    """Lazy load sentence transformer embedding model."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(settings.embedding_model)
    return _embedding_model


def get_sentiment_pipeline():
    """Lazy load sentiment analysis pipeline."""
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None
        )
    return _sentiment_pipeline


class NLPProcessor:
    """Processes articles through NLP pipeline."""

    def __init__(self, consumer_name: str = "worker-1"):
        self.consumer_name = consumer_name
        self.group_name = "nlp-workers"
        self.stream_key = "articles:raw"
        self.running = False
        self.stats = {
            "processed": 0,
            "failed": 0,
            "start_time": None
        }

    async def start(self) -> None:
        """Start processing loop."""
        self.running = True
        self.stats["start_time"] = datetime.now(timezone.utc)

        logger.info(
            "nlp_worker_starting",
            consumer=self.consumer_name,
            group=self.group_name
        )

        while self.running:
            try:
                # Read messages from stream
                messages = await redis_client.read_stream(
                    stream_key=self.stream_key,
                    group=self.group_name,
                    consumer=self.consumer_name,
                    count=5,
                    block=10000  # 10 second timeout
                )

                if not messages:
                    continue

                for msg_id, msg_data in messages:
                    await self._process_message(msg_id, msg_data)

            except Exception as e:
                logger.error("worker_loop_error", error=str(e))
                await asyncio.sleep(5)

    def stop(self) -> None:
        """Stop processing loop."""
        self.running = False

    async def _process_message(
        self,
        msg_id: str,
        msg_data: Dict[str, Any]
    ) -> None:
        """Process a single article message."""
        article_id = msg_data.get("article_id")

        try:
            logger.debug(
                "processing_article",
                article_id=article_id,
                title=str(msg_data.get("title", ""))[:50]
            )

            # Extract text for processing
            text = self._prepare_text(msg_data)

            if len(text) < 100:
                # Skip very short articles
                await db_client.mark_article_failed(article_id, "Article too short")
                self.stats["failed"] += 1
                return

            # Run NLP pipeline
            entities = self._extract_entities(text)
            sentiment = self._analyze_sentiment(text)
            keyphrases = self._extract_keyphrases(text)
            embedding = self._generate_embedding(text)
            category = msg_data.get("category") or self._classify_topic(text)

            # Update article in database
            processing_data = {
                "body": msg_data.get("body", text),
                "sentiment_score": sentiment["score"],
                "sentiment_label": sentiment["label"],
                "word_count": len(text.split()),
                "category": category,
                "processing_status": "done",
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }

            if embedding:
                processing_data["embedding"] = embedding

            await db_client.update_article_processing(article_id, processing_data)

            # Save entities
            if entities:
                await db_client.add_article_entities(article_id, entities)

            # Save keyphrases
            if keyphrases:
                await db_client.add_article_keyphrases(article_id, keyphrases)

            # Index to Elasticsearch
            article_doc = {
                "article_id": article_id,
                "url": msg_data.get("url"),
                "title": msg_data.get("title"),
                "body": msg_data.get("body", text),
                "source": msg_data.get("source"),
                "published_at": msg_data.get("published_at"),
                "category": category,
                "sentiment_score": sentiment["score"],
                "sentiment_label": sentiment["label"],
                "word_count": len(text.split()),
                "entities": entities,
                "keyphrases": [kp["phrase"] for kp in keyphrases],
                "embedding": embedding,
                "created_at": msg_data.get("created_at"),
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }

            await es_client.index_article(article_doc)

            # Acknowledge message
            await redis_client.ack_message(
                self.stream_key,
                self.group_name,
                msg_id
            )

            self.stats["processed"] += 1

            logger.info(
                "article_processed",
                article_id=article_id,
                entities=len(entities),
                sentiment=sentiment["label"]
            )

        except Exception as e:
            logger.error(
                "article_processing_failed",
                article_id=article_id,
                error=str(e)
            )

            # Add to dead letter queue
            await redis_client.add_to_dlq(
                self.stream_key,
                msg_id,
                msg_data,
                str(e)
            )

            # Acknowledge to avoid reprocessing
            await redis_client.ack_message(
                self.stream_key,
                self.group_name,
                msg_id
            )

            self.stats["failed"] += 1

    def _prepare_text(self, msg_data: Dict[str, Any]) -> str:
        """Prepare text for NLP processing."""
        title = msg_data.get("title", "")
        body = msg_data.get("body", "")

        if body:
            return f"{title}\n\n{body}"
        return title

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities using spaCy."""
        nlp = get_spacy_model()
        doc = nlp(text[:5000])  # Limit text length for NER

        entities = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "NORP", "EVENT", "PRODUCT"):
                text_clean = ent.text.strip()
                if text_clean and len(text_clean) > 1 and text_clean not in seen:
                    entities.append({
                        "text": text_clean,
                        "type": ent.label_,
                        "confidence": 0.9  # spaCy doesn't provide confidence
                    })
                    seen.add(text_clean)

        return entities

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        pipeline = get_sentiment_pipeline()

        # Truncate properly at the tokenizer level (character slicing is unreliable)
        text_chunk = text[:3000]  # rough pre-trim for speed only

        try:
            results = pipeline(text_chunk, truncation=True, max_length=512)[0]

            # Find the label with highest score
            sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
            top = sorted_results[0]

            # Map labels to standard names
            label_map = {
                "LABEL_0": "NEGATIVE",
                "LABEL_1": "NEUTRAL",
                "LABEL_2": "POSITIVE"
            }
            label = label_map.get(top["label"], top["label"])

            # Convert to score from -1 to +1
            if label == "POSITIVE":
                score = top["score"]
            elif label == "NEGATIVE":
                score = -top["score"]
            else:
                score = 0.0

            return {"label": label, "score": round(score, 3)}

        except Exception as e:
            logger.warning("sentiment_analysis_failed", error=str(e))
            return {"label": "NEUTRAL", "score": 0.0}

    def _extract_keyphrases(self, text: str) -> List[Dict[str, Any]]:
        """Extract key phrases from text."""
        # Simple approach: extract noun chunks and repeated phrases
        nlp = get_spacy_model()
        doc = nlp(text[:3000])

        phrase_counts = {}

        for chunk in doc.noun_chunks:
            phrase = chunk.text.strip().lower()
            if len(phrase) > 3 and len(phrase) < 50:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1

        # Filter and sort
        keyphrases = [
            {"phrase": phrase, "score": count / len(doc)}
            for phrase, count in phrase_counts.items()
            if count >= 2
        ]

        keyphrases.sort(key=lambda x: x["score"], reverse=True)

        return keyphrases[:10]

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding vector for text."""
        model = get_embedding_model()

        try:
            # Use title + first 512 tokens of body
            text_for_embed = text[:500]
            embedding = model.encode(text_for_embed, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning("embedding_generation_failed", error=str(e))
            return None

    def _classify_topic(self, text: str) -> str:
        """Classify article into topic category."""
        text_lower = text.lower()

        # Simple rule-based classification for MVP
        if any(kw in text_lower for kw in ["technology", "software", "ai", "tech", "digital", "cyber"]):
            return "technology"
        elif any(kw in text_lower for kw in ["politics", "election", "government", "congress", "parliament"]):
            return "politics"
        elif any(kw in text_lower for kw in ["economy", "market", "stock", "financial", "bank"]):
            return "economy"
        elif any(kw in text_lower for kw in ["health", "medical", "hospital", "disease", "vaccine"]):
            return "health"
        elif any(kw in text_lower for kw in ["sport", "game", "team", "player", "match"]):
            return "sports"
        elif any(kw in text_lower for kw in ["climate", "environment", "emissions", "sustainable"]):
            return "environment"
        else:
            return "news"


async def run_worker(consumer_name: str = "worker-1") -> None:
    """Run NLP worker as standalone process."""
    # Connect to services
    await redis_client.connect()
    db_client.connect()  # Sync for Supabase REST client
    await es_client.connect()
    await es_client.create_indices()

    processor = NLPProcessor(consumer_name)

    try:
        await processor.start()
    except KeyboardInterrupt:
        logger.info("worker_shutdown")
        processor.stop()
    finally:
        await redis_client.disconnect()
        await db_client.disconnect()
        await es_client.disconnect()


if __name__ == "__main__":
    import sys

    consumer = sys.argv[1] if len(sys.argv) > 1 else "worker-1"
    asyncio.run(run_worker(consumer))
