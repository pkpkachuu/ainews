"""Reranking service using cross-encoder models.

Reranks top-k candidates from hybrid retrieval for improved relevance.
"""

from typing import List, Dict, Any, Optional
import structlog

from app.config import settings

logger = structlog.get_logger()

# Global reranker model (lazy loaded)
_reranker_model = None


def get_reranker():
    """Lazy load cross-encoder reranker model."""
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        try:
            _reranker_model = CrossEncoder(settings.reranker_model)
        except Exception as e:
            logger.warning("reranker_load_failed", error=str(e))
            return None
    return _reranker_model


class Reranker:
    """Reranks candidates using cross-encoder model."""

    def __init__(self):
        self.model = None

    def _get_model(self):
        """Get or load reranker model."""
        if self.model is None:
            self.model = get_reranker()
        return self.model

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Original search query
            candidates: List of candidate documents with _score
            top_k: Number of candidates to return

        Returns:
            Reranked candidates with updated scores
        """
        if not candidates:
            return []

        model = self._get_model()

        if model is None:
            # Fallback: return original order
            return candidates[:top_k]

        # Prepare query-document pairs
        pairs = []
        for candidate in candidates:
            # Combine title and body for document text
            text = candidate.get("title", "")
            if candidate.get("body"):
                # Use first 200 chars of body for efficiency
                text += " " + candidate["body"][:200]

            pairs.append([query, text])

        try:
            # Score pairs with cross-encoder
            scores = model.predict(pairs)

            # Add rerank scores to candidates
            for i, score in enumerate(scores):
                candidates[i]["rerank_score"] = float(score)

            # Sort by rerank score
            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

            # Apply recency boost
            reranked = self._apply_recency_boost(reranked)

            return reranked[:top_k]

        except Exception as e:
            logger.warning("rerank_failed", error=str(e))
            return candidates[:top_k]

    def _apply_recency_boost(
        self,
        candidates: List[Dict[str, Any]],
        boost_factor: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Apply recency boost to scores."""
        from datetime import datetime, timezone

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

                    # Boost factor: newer articles get higher boost
                    # Exponential decay with half-life of 7 days
                    recency_multiplier = 1 + boost_factor * (0.5 ** (days_old / 7))

                    if "rerank_score" in candidate:
                        candidate["rerank_score"] *= recency_multiplier
                    elif "_score" in candidate:
                        candidate["_score"] *= recency_multiplier

                except Exception:
                    pass

        return candidates


# Global reranker instance
reranker = Reranker()
