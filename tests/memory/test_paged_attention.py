"""
Tests for Paged Attention Integration.

This module tests the integration of paged KV cache with the attention mechanism,
including gather/scatter operations and handling page boundaries.

Following TDD: These tests are written before implementing paged attention.
"""

import pytest
import torch

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.memory.memory_pool import MemoryPool
from vibe_sgl_lite.memory.paged_allocator import PagedAllocator


@pytest.mark.unit
def test_kv_gather_from_pages() -> None:
    """Test gathering KV from pages using page table."""
    from vibe_sgl_lite.memory.paged_attention import gather_kv_from_pages

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate pages for a sequence
    seq_id = "seq_1"
    num_tokens = 32  # 2 pages
    page_ids = allocator.allocate_sequence(seq_id, num_tokens)

    # Gather KV for layer 0
    layer_idx = 0
    cache_k, cache_v = gather_kv_from_pages(pool, page_ids, layer_idx, num_tokens)

    # Check shapes
    assert cache_k.shape == (1, num_kv_heads, num_tokens, head_dim)
    assert cache_v.shape == (1, num_kv_heads, num_tokens, head_dim)


@pytest.mark.unit
def test_kv_scatter_to_pages() -> None:
    """Test scattering new KV to pages."""
    from vibe_sgl_lite.memory.paged_attention import scatter_kv_to_pages

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

    allocator = PagedAllocator(pool, page_size=page_size)

    # Allocate pages
    seq_id = "seq_1"
    num_tokens = 32
    page_ids = allocator.allocate_sequence(seq_id, num_tokens)

    # Create new KV to scatter
    layer_idx = 0
    new_k = torch.randn(1, num_kv_heads, num_tokens, head_dim)
    new_v = torch.randn(1, num_kv_heads, num_tokens, head_dim)

    # Scatter to pages
    scatter_kv_to_pages(pool, page_ids, layer_idx, new_k, new_v)

    # Verify data was written (gather it back)
    from vibe_sgl_lite.memory.paged_attention import gather_kv_from_pages
    gathered_k, gathered_v = gather_kv_from_pages(pool, page_ids, layer_idx, num_tokens)

    assert torch.allclose(gathered_k, new_k, atol=1e-6)
    assert torch.allclose(gathered_v, new_v, atol=1e-6)


@pytest.mark.unit
def test_paged_attention_computation() -> None:
    """Test attention computation with paged KV cache."""
    from vibe_sgl_lite.memory.paged_attention import paged_attention

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    batch_size = 1
    seq_len = 10
    num_kv_heads = config.num_key_value_heads
    num_q_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # Create query
    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim)

    # Create paged KV cache
    num_pages = 5
    page_size = 16
    num_layers = config.num_hidden_layers

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

    # Allocate and populate pages
    seq_id = "seq_1"
    cache_len = 32
    page_ids = allocator.allocate_sequence(seq_id, cache_len)

    layer_idx = 0
    cache_k = torch.randn(1, num_kv_heads, cache_len, head_dim)
    cache_v = torch.randn(1, num_kv_heads, cache_len, head_dim)

    from vibe_sgl_lite.memory.paged_attention import scatter_kv_to_pages
    scatter_kv_to_pages(pool, page_ids, layer_idx, cache_k, cache_v)

    # Compute paged attention
    output = paged_attention(query, pool, page_ids, layer_idx, cache_len)

    # Check output shape
    assert output.shape == (batch_size, num_q_heads, seq_len, head_dim)


@pytest.mark.integration
def test_paged_attention_correctness() -> None:
    """Validate paged attention correctness vs contiguous attention."""
    from vibe_sgl_lite.memory.paged_attention import paged_attention
    from vibe_sgl_lite.models.qwen3.attention import compute_attention

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    batch_size = 1
    seq_len = 5
    cache_len = 20
    num_kv_heads = config.num_key_value_heads
    num_q_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # Create query, key, value
    query = torch.randn(batch_size, num_q_heads, seq_len, head_dim)
    key = torch.randn(batch_size, num_kv_heads, cache_len, head_dim)
    value = torch.randn(batch_size, num_kv_heads, cache_len, head_dim)

    # Compute contiguous attention
    contiguous_output = compute_attention(query, key, value)

    # Compute paged attention
    num_pages = 10
    page_size = 16
    num_layers = config.num_hidden_layers

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
    seq_id = "seq_1"
    page_ids = allocator.allocate_sequence(seq_id, cache_len)

    layer_idx = 0
    from vibe_sgl_lite.memory.paged_attention import scatter_kv_to_pages
    scatter_kv_to_pages(pool, page_ids, layer_idx, key, value)

    paged_output = paged_attention(query, pool, page_ids, layer_idx, cache_len)

    # Outputs should be very close
    assert torch.allclose(contiguous_output, paged_output, atol=1e-5, rtol=1e-4)


@pytest.mark.unit
def test_page_boundary_handling() -> None:
    """Test handling page boundaries in attention."""
    from vibe_sgl_lite.memory.paged_attention import gather_kv_from_pages

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    page_size = 16
    num_layers = config.num_hidden_layers
    num_kv_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads

    # Create pool with small pages
    num_pages = 10
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

    # Allocate sequence that spans multiple pages
    seq_id = "seq_1"
    num_tokens = 50  # Spans 4 pages (50/16 = 3.125 -> 4)
    page_ids = allocator.allocate_sequence(seq_id, num_tokens)

    assert len(page_ids) == 4

    # Gather KV across page boundaries
    layer_idx = 0
    cache_k, cache_v = gather_kv_from_pages(pool, page_ids, layer_idx, num_tokens)

    # Should handle page boundaries correctly
    assert cache_k.shape == (1, num_kv_heads, num_tokens, head_dim)
    assert cache_v.shape == (1, num_kv_heads, num_tokens, head_dim)
