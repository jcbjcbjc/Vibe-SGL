"""
Tests for MemoryPool - page-based memory management for KV cache.

This module tests the MemoryPool class that manages a pool of fixed-size pages
for KV cache allocation. The pool pre-allocates pages and manages them with
a free list for efficient allocation and deallocation.

Following TDD: These tests are written before implementing MemoryPool.
"""

import pytest
import torch
import threading
import time

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_memory_pool_initialization() -> None:
    """Test that MemoryPool initializes with correct cache size."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 100
    page_size = 16  # tokens per page
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Check initialization
    assert pool.num_pages == num_pages
    assert pool.page_size == page_size
    assert pool.num_layers == num_layers
    assert pool.num_kv_heads == num_kv_heads
    assert pool.head_dim == head_dim

    # All pages should be free initially
    stats = pool.get_stats()
    assert stats["total"] == num_pages
    assert stats["free"] == num_pages
    assert stats["used"] == 0
    assert stats["utilization"] == 0.0


@pytest.mark.unit
def test_memory_pool_single_page_allocation() -> None:
    """Test allocating a single page from the pool."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 10
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate one page
    page_id = pool.allocate()

    assert page_id is not None
    assert isinstance(page_id, int)
    assert page_id >= 0
    assert page_id < num_pages

    # Check stats after allocation
    stats = pool.get_stats()
    assert stats["used"] == 1
    assert stats["free"] == num_pages - 1
    assert stats["utilization"] == 1.0 / num_pages


@pytest.mark.unit
def test_memory_pool_batch_allocation() -> None:
    """Test allocating multiple pages at once."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 20
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate 5 pages
    num_to_allocate = 5
    page_ids = pool.allocate_batch(num_to_allocate)

    assert len(page_ids) == num_to_allocate
    assert len(set(page_ids)) == num_to_allocate  # All unique
    for page_id in page_ids:
        assert page_id >= 0
        assert page_id < num_pages

    # Check stats
    stats = pool.get_stats()
    assert stats["used"] == num_to_allocate
    assert stats["free"] == num_pages - num_to_allocate


@pytest.mark.unit
def test_memory_pool_page_deallocation() -> None:
    """Test deallocating pages back to the pool."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 10
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate and then free a page
    page_id = pool.allocate()
    assert pool.get_stats()["used"] == 1

    pool.free(page_id)
    assert pool.get_stats()["used"] == 0
    assert pool.get_stats()["free"] == num_pages


@pytest.mark.unit
def test_memory_pool_batch_deallocation() -> None:
    """Test deallocating multiple pages at once."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 20
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate and then free multiple pages
    page_ids = pool.allocate_batch(5)
    assert pool.get_stats()["used"] == 5

    pool.free_batch(page_ids)
    assert pool.get_stats()["used"] == 0
    assert pool.get_stats()["free"] == num_pages


@pytest.mark.unit
def test_memory_pool_statistics() -> None:
    """Test memory accounting and statistics."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 100
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate some pages
    page_ids = pool.allocate_batch(30)

    stats = pool.get_stats()
    assert stats["total"] == num_pages
    assert stats["used"] == 30
    assert stats["free"] == 70
    assert abs(stats["utilization"] - 0.3) < 1e-6

    # Free some pages
    pool.free_batch(page_ids[:10])

    stats = pool.get_stats()
    assert stats["used"] == 20
    assert stats["free"] == 80
    assert abs(stats["utilization"] - 0.2) < 1e-6


@pytest.mark.unit
def test_memory_pool_allocation_failure() -> None:
    """Test allocation failure when pool is exhausted."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 5  # Small pool
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate all pages
    page_ids = pool.allocate_batch(num_pages)
    assert len(page_ids) == num_pages
    assert pool.get_stats()["free"] == 0

    # Try to allocate one more - should return None
    page_id = pool.allocate()
    assert page_id is None


@pytest.mark.unit
def test_memory_pool_batch_allocation_partial_failure() -> None:
    """Test batch allocation when not enough pages available."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 10
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate most pages
    pool.allocate_batch(8)

    # Try to allocate more than available (only 2 left)
    page_ids = pool.allocate_batch(5)

    # Should only get 2 pages
    assert len(page_ids) == 2
    assert pool.get_stats()["free"] == 0


@pytest.mark.unit
def test_memory_pool_thread_safety() -> None:
    """Test thread-safe concurrent allocation/deallocation."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 100
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    allocated_pages = []
    errors = []

    def allocate_worker():
        try:
            for _ in range(10):
                page_id = pool.allocate()
                if page_id is not None:
                    allocated_pages.append(page_id)
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    def free_worker():
        try:
            time.sleep(0.005)  # Let some allocations happen first
            for _ in range(5):
                if allocated_pages:
                    page_id = allocated_pages.pop(0)
                    pool.free(page_id)
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)

    # Run multiple threads
    threads = []
    for _ in range(3):
        t1 = threading.Thread(target=allocate_worker)
        t2 = threading.Thread(target=free_worker)
        threads.extend([t1, t2])

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    # Check no errors occurred
    assert len(errors) == 0

    # Check pool is in consistent state
    stats = pool.get_stats()
    assert stats["total"] == num_pages
    assert stats["used"] + stats["free"] == num_pages


@pytest.mark.unit
def test_memory_pool_reuse_freed_pages() -> None:
    """Test that freed pages can be reallocated."""
    from vibe_sgl_lite.memory.memory_pool import MemoryPool

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 5
    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    pool = MemoryPool(
        num_pages=num_pages,
        page_size=page_size,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        device="cpu",
        dtype=torch.float32,
    )

    # Allocate all pages
    page_ids = pool.allocate_batch(num_pages)
    assert pool.get_stats()["free"] == 0

    # Free one page
    pool.free(page_ids[0])
    assert pool.get_stats()["free"] == 1

    # Should be able to allocate again
    new_page_id = pool.allocate()
    assert new_page_id is not None
    assert pool.get_stats()["free"] == 0
