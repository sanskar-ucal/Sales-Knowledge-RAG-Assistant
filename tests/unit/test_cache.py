import asyncio

from app.cache import AsyncTTLCache, make_key


async def test_set_get_and_stats():
    cache = AsyncTTLCache(max_items=10, ttl_seconds=60)
    assert await cache.get("missing") is None
    await cache.set("k", 1)
    assert await cache.get("k") == 1
    stats = cache.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1 and stats["hit_rate"] == 0.5


async def test_lru_eviction():
    cache = AsyncTTLCache(max_items=2, ttl_seconds=60)
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.get("a")
    await cache.set("c", 3)
    assert await cache.get("b") is None
    assert await cache.get("a") == 1
    assert await cache.get("c") == 3


async def test_ttl_expiry():
    cache = AsyncTTLCache(max_items=2, ttl_seconds=0.01)
    await cache.set("a", 1)
    await asyncio.sleep(0.02)
    assert await cache.get("a") is None


def test_make_key_is_stable():
    assert make_key("rag", "q", 4) == make_key("rag", "q", 4)
    assert make_key("rag", "q", 4) != make_key("rag", "q", 5)
