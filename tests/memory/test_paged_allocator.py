"""
Tests for PagedAllocator - page table management for sequences.

This module tests the PagedAllocator class that manages page allocation
for sequences with page tables, reference counting, and LRU eviction.

Following TDD: These tests are written before implementing PagedAllocator.
"""

import pytest
import torch

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.memory.memory_pool import MemoryPool


@pytest.mark.unit
def test_paged_allocator_initialization() -> None:
    """Test that PagedAllocator initializes correctly."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    assert allocator.page_size == page_size
    assert allocator.pool == pool


@pytest.mark.unit
def test_paged_allocator_allocate_sequence() -> None:
    """Test allocating pages for a new sequence."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate pages for a sequence of 50 tokens (needs 4 pages: 50/16 = 3.125 -> 4)
    seq_id = "seq_1"
    num_tokens = 50
    page_ids = allocator.allocate_sequence(seq_id, num_tokens)

    expected_num_pages = (num_tokens + page_size - 1) // page_size  # Ceiling division
    assert len(page_ids) == expected_num_pages
    assert all(isinstance(pid, int) for pid in page_ids)


@pytest.mark.unit
def test_paged_allocator_page_table() -> None:
    """Test page table creation and management."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate sequence
    seq_id = "seq_1"
    num_tokens = 50
    page_ids = allocator.allocate_sequence(seq_id, num_tokens)

    # Get page table for sequence
    page_table = allocator.get_page_table(seq_id)

    assert page_table is not None
    assert len(page_table) == len(page_ids)
    assert page_table == page_ids


@pytest.mark.unit
def test_paged_allocator_free_sequence() -> None:
    """Test freeing all pages for a sequence."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate and then free
    seq_id = "seq_1"
    num_tokens = 50
    allocator.allocate_sequence(seq_id, num_tokens)

    initial_free = pool.get_stats()["free"]

    allocator.free_sequence(seq_id)

    # All pages should be returned to pool
    final_free = pool.get_stats()["free"]
    assert final_free > initial_free

    # Page table should be removed
    assert allocator.get_page_table(seq_id) is None


@pytest.mark.unit
def test_paged_allocator_reference_counting() -> None:
    """Test reference counting for shared pages."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate sequence 1
    seq_id_1 = "seq_1"
    page_ids_1 = allocator.allocate_sequence(seq_id_1, 32)

    # Share first page with sequence 2
    seq_id_2 = "seq_2"
    allocator.share_pages(seq_id_2, page_ids_1[:1])

    # Get reference count for shared page
    ref_count = allocator.get_ref_count(page_ids_1[0])
    assert ref_count == 2  # Shared by 2 sequences

    # Free sequence 1 - page should still be allocated (ref count = 1)
    allocator.free_sequence(seq_id_1)
    ref_count = allocator.get_ref_count(page_ids_1[0])
    assert ref_count == 1

    # Free sequence 2 - page should now be freed (ref count = 0)
    allocator.free_sequence(seq_id_2)
    ref_count = allocator.get_ref_count(page_ids_1[0])
    assert ref_count == 0


@pytest.mark.unit
def test_paged_allocator_lru_eviction() -> None:
    """Test LRU eviction when pool is exhausted."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    num_pages = 10  # Small pool
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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate sequences until pool is full
    seq_ids = []
    for i in range(5):
        seq_id = f"seq_{i}"
        seq_ids.append(seq_id)
        allocator.allocate_sequence(seq_id, 32)  # 2 pages each

    assert pool.get_stats()["free"] == 0

    # Try to allocate one more - should trigger LRU eviction
    new_seq_id = "seq_new"
    page_ids = allocator.allocate_sequence(new_seq_id, 32, allow_eviction=True)

    # Should succeed by evicting LRU sequence
    assert len(page_ids) > 0

    # Oldest sequence should have been evicted
    assert allocator.get_page_table(seq_ids[0]) is None


@pytest.mark.unit
def test_paged_allocator_page_metadata() -> None:
    """Test page metadata tracking."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate sequence
    seq_id = "seq_1"
    page_ids = allocator.allocate_sequence(seq_id, 32)

    # Check metadata for first page
    metadata = allocator.get_page_metadata(page_ids[0])

    assert metadata is not None
    assert metadata["page_id"] == page_ids[0]
    assert metadata["owner"] == seq_id
    assert metadata["ref_count"] >= 1


@pytest.mark.unit
def test_paged_allocator_extend_sequence() -> None:
    """Test extending a sequence with more pages."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate initial sequence
    seq_id = "seq_1"
    initial_pages = allocator.allocate_sequence(seq_id, 32)  # 2 pages
    assert len(initial_pages) == 2

    # Extend sequence
    additional_tokens = 32  # Need 2 more pages
    new_pages = allocator.extend_sequence(seq_id, additional_tokens)
    assert len(new_pages) == 2

    # Total pages should be 4
    page_table = allocator.get_page_table(seq_id)
    assert len(page_table) == 4


@pytest.mark.unit
def test_paged_allocator_multiple_sequences() -> None:
    """Test managing multiple sequences simultaneously."""
    from vibe_sgl_lite.memory.paged_allocator import PagedAllocator

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate multiple sequences
    num_sequences = 10
    for i in range(num_sequences):
        seq_id = f"seq_{i}"
        allocator.allocate_sequence(seq_id, 32)

    # Check all sequences have page tables
    for i in range(num_sequences):
        seq_id = f"seq_{i}"
        page_table = allocator.get_page_table(seq_id)
        assert page_table is not None
        assert len(page_table) == 2  # 32 tokens / 16 per page = 2 pages

    # Get statistics
    stats = allocator.get_stats()
    assert stats["num_sequences"] == num_sequences
    assert stats["total_pages_allocated"] == num_sequences * 2
