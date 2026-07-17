"""Redis client and queue utilities for message passing."""

import redis.asyncio as aioredis
import json
from typing import Optional, List, Dict, Any
import structlog

from app.config import settings

logger = structlog.get_logger()


class RedisClient:
    """Async Redis client wrapper for queue operations and caching."""

    def __init__(self):
        self.client: Optional[aioredis.Redis] = None

    async def connect(self) -> None:
        """Initialize Redis connection."""
        self.client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("redis_connected", url=settings.redis_url)

    async def disconnect(self) -> None:
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            logger.info("redis_disconnected")

    # Stream operations for ingestion queue
    async def add_to_stream(
        self,
        stream_key: str,
        data: Dict[str, Any],
        maxlen: int = 10000
    ) -> str:
        """Add a message to a Redis stream.

        Args:
            stream_key: Stream name (e.g., "articles:raw")
            data: Message data to add
            maxlen: Maximum stream length (approximate)

        Returns:
            Message ID
        """
        if not self.client:
            raise RuntimeError("Redis not connected")

        # Serialize data for storage
        serialized = {k: json.dumps(v, default=str) if isinstance(v, dict) else str(v)
                      for k, v in data.items()}

        msg_id = await self.client.xadd(
            stream_key,
            serialized,
            maxlen=maxlen
        )
        return msg_id

    async def read_stream(
        self,
        stream_key: str,
        group: str,
        consumer: str,
        count: int = 10,
        block: int = 5000
    ) -> List[tuple[str, Dict[str, Any]]]:
        """Read messages from a stream using consumer group.

        Args:
            stream_key: Stream name
            group: Consumer group name
            consumer: Consumer name
            count: Max messages to read
            block: Block timeout in milliseconds

        Returns:
            List of (message_id, data) tuples
        """
        if not self.client:
            raise RuntimeError("Redis not connected")

        # Ensure consumer group exists
        try:
            await self.client.xgroup_create(
                stream_key,
                group,
                id="0",
                mkstream=True
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        # Read messages
        messages = await self.client.xreadgroup(
            group,
            consumer,
            {stream_key: ">"},
            count=count,
            block=block
        )

        results = []
        if messages:
            for stream_name, stream_messages in messages:
                for msg_id, msg_data in stream_messages:
                    # Deserialize data
                    deserialized = {}
                    for k, v in msg_data.items():
                        try:
                            deserialized[k] = json.loads(v)
                        except (json.JSONDecodeError, TypeError):
                            deserialized[k] = v
                    results.append((msg_id, deserialized))

        return results

    async def ack_message(self, stream_key: str, group: str, msg_id: str) -> None:
        """Acknowledge a processed message."""
        if not self.client:
            raise RuntimeError("Redis not connected")
        await self.client.xack(stream_key, group, msg_id)

    async def add_to_dlq(
        self,
        stream_key: str,
        msg_id: str,
        msg_data: Dict[str, Any],
        error: str
    ) -> None:
        """Add failed message to dead-letter queue."""
        if not self.client:
            raise RuntimeError("Redis not connected")

        dlq_key = f"{stream_key}:dlq"
        dlq_data = {
            "original_id": msg_id,
            "error": error,
            "data": json.dumps(msg_data, default=str)
        }
        await self.client.xadd(dlq_key, dlq_data)

    # Cache operations
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        if not self.client:
            raise RuntimeError("Redis not connected")

        value = await self.client.get(key)
        if value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with optional TTL."""
        if not self.client:
            raise RuntimeError("Redis not connected")

        serialized = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
        if ttl:
            await self.client.setex(key, ttl, serialized)
        else:
            await self.client.set(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete a cached value."""
        if not self.client:
            raise RuntimeError("Redis not connected")
        await self.client.delete(key)

    async def lpush(self, key: str, value: Any) -> None:
        """Push to left of list."""
        if not self.client:
            raise RuntimeError("Redis not connected")
        serialized = json.dumps(value, default=str) if isinstance(value, (dict, list)) else str(value)
        await self.client.lpush(key, serialized)

    async def lrange(self, key: str, start: int = 0, end: int = -1) -> List[Any]:
        """Get range from list."""
        if not self.client:
            raise RuntimeError("Redis not connected")
        values = await self.client.lrange(key, start, end)
        results = []
        for v in values:
            try:
                results.append(json.loads(v))
            except json.JSONDecodeError:
                results.append(v)
        return results

    async def llen(self, key: str) -> int:
        """Get list length."""
        if not self.client:
            raise RuntimeError("Redis not connected")
        return await self.client.llen(key)

    async def set_embeddings_centroid(
        self,
        cluster_id: str,
        centroid: List[float],
        ttl: int = 172800  # 48 hours
    ) -> None:
        """Store cluster centroid for emergence detection."""
        key = f"cluster:{cluster_id}:centroid"
        await self.set(key, centroid, ttl)

    async def get_embeddings_centroid(self, cluster_id: str) -> Optional[List[float]]:
        """Retrieve cluster centroid."""
        key = f"cluster:{cluster_id}:centroid"
        return await self.get(key)

    async def get_all_cluster_centroids(self) -> Dict[str, List[float]]:
        """Get all stored cluster centroids."""
        if not self.client:
            raise RuntimeError("Redis not connected")

        keys = await self.client.keys("cluster:*:centroid")
        centroids = {}
        for key in keys:
            # Extract cluster_id from key format "cluster:{cluster_id}:centroid"
            parts = key.split(":")
            if len(parts) >= 2:
                cluster_id = parts[1]
                centroid = await self.get(key)
                if centroid:
                    centroids[cluster_id] = centroid
        return centroids

    async def flush_cluster_centroids(self) -> int:
        """Delete all cached cluster centroids.

        Emergence detection treats any cluster whose centroid is already
        cached (up to 48h) as 'already seen' and suppresses it, even if it
        was never actually surfaced in the feed. This is correct behavior in
        steady state, but during dev/testing - repeatedly hitting
        /intelligence/feed/generate against a static or slow-growing corpus -
        it makes real emerging topics vanish after the first generation.
        This clears that cache so the next detection run treats everything
        as new again. Returns the number of keys deleted.
        """
        if not self.client:
            raise RuntimeError("Redis not connected")

        keys = await self.client.keys("cluster:*:centroid")
        if not keys:
            return 0

        await self.client.delete(*keys)
        return len(keys)


# Global Redis client instance
redis_client = RedisClient()
