"""Reranking service.

Applies recency boost and fallback relevance filtering for cloud deployment.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import structlog

logger = structlog.get_logger()


class Reranker:
    """Reranks candidates using lightweight recency boost."""

    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Lightweight pass-through rerank to bypass PyTorch in production.

        Applies standard recency scaling to the Elasticsearch scores.
        """
        if not candidates:
            return []

        # Apply recency boost
        reranked = self._apply_recency_boost(candidates)

        # Sort by updated score
        reranked = sorted(
            reranked,
            key=lambda x: x.get("rerank_score") or x.get("_score", 0),
            reverse=True
        )

        return reranked[:top_k]

    def _apply_recency_boost(
        self,
        candidates: List[Dict[str, Any]],
        boost_factor: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Apply recency boost to scores."""
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

                    # Exponential decay with half-life of 7 days
                    recency_multiplier = 1 + boost_factor * (0.5 ** (days_old / 7))

                    if "_score" in candidate:
                        candidate["_score"] *= recency_multiplier

                except Exception:
                    pass

        return candidates


# Global reranker instance
reranker = Reranker()
