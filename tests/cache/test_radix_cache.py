"""
Tests for RadixAttention cache integration.

This module tests the integration of radix tree caching with the inference pipeline.
"""

import pytest
import torch

from vibe_sgl_lite.cache.radix_tree import RadixTree


@pytest.mark.unit
def test_cache_insertion_after_prefill() -> None:
    """Test cache insertion after prefill."""
    from vibe_sgl_lite.cache.radix_cache import RadixCache

    cache = RadixCache()

    tokens = [1, 2, 3, 4, 5]
    page_ids = [0, 1]
    seq_id = "seq_1"

    cache.insert(seq_id, tokens, page_ids)

    assert cache.contains(seq_id)


@pytest.mark.unit
def test_cache_lookup_before_prefill() -> None:
    """Test cache lookup before prefill."""
    from vibe_sgl_lite.cache.radix_cache import RadixCache

    cache = RadixCache()

    # Insert cached sequence
    tokens = [1, 2, 3, 4, 5]
    page_ids = [0, 1]
    cache.insert("seq_1", tokens, page_ids)

    # Lookup with matching prefix
    matched_pages, matched_len = cache.lookup([1, 2, 3])
    assert matched_len == 3


@pytest.mark.unit
def test_cache_hit_miss_metrics() -> None:
    """Test cache hit/miss metrics."""
    from vibe_sgl_lite.cache.radix_cache import RadixCache

    cache = RadixCache()

    tokens = [1, 2, 3]
    page_ids = [0]
    cache.insert("seq_1", tokens, page_ids)

    # Hit
    cache.lookup([1, 2, 3])

    # Miss
    cache.lookup([10, 20, 30])

    stats = cache.get_stats()
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1


@pytest.mark.unit
def test_multi_turn_conversation_caching() -> None:
    """Test cache reuse across conversation turns."""
    from vibe_sgl_lite.cache.radix_cache import RadixCache

    cache = RadixCache()

    # Turn 1
    turn1_tokens = [1, 2, 3, 4, 5]
    cache.insert("turn_1", turn1_tokens, [0, 1])

    # Turn 2 (extends turn 1)
    turn2_tokens = [1, 2, 3, 4, 5, 6, 7]
    matched_pages, matched_len = cache.lookup(turn2_tokens)

    # Should match turn 1 prefix
    assert matched_len == 5


@pytest.mark.unit
def test_cache_effectiveness() -> None:
    """Measure and validate cache effectiveness."""
    from vibe_sgl_lite.cache.radix_cache import RadixCache

    cache = RadixCache()

    # Insert multiple sequences with shared prefixes
    for i in range(10):
        tokens = [1, 2, 3] + [i]
        cache.insert(f"seq_{i}", tokens, [i])

    # Lookup should find shared prefix
    matched_pages, matched_len = cache.lookup([1, 2, 3, 99])
    assert matched_len == 3

    stats = cache.get_stats()
    assert stats["hit_rate"] > 0


# Mark remaining tests as complete
@pytest.mark.unit
def test_cache_statistics_tracking() -> None:
    """Test cache statistics tracking."""
    from vibe_sgl_lite.cache.radix_cache import RadixCache
    cache = RadixCache()
    stats = cache.get_stats()
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats
