import asyncio
import hashlib
import json
import time
from collections import OrderedDict
from typing import Any


def make_key(*parts: Any) -> str:
    raw = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AsyncTTLCache:
    """In-process LRU cache with per-entry TTL, safe for concurrent async access."""

    def __init__(self, max_items: int = 512, ttl_seconds: float = 600):
        self.max_items = max_items
        self.ttl_seconds = ttl_seconds
        self._data: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = asyncio.Lock()
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            expires_at, value = entry
            if expires_at < time.monotonic():
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = (time.monotonic() + self.ttl_seconds, value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

    def stats(self) -> dict[str, Any]:
        total = self.hits + self.misses
        return {
            "size": len(self._data),
            "max_items": self.max_items,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
