"""
Paged attention operations for efficient KV cache management.

This module implements gather/scatter operations for paged KV cache
and paged attention computation.
"""

import torch
from typing import List, Tuple

from vibe_sgl_lite.memory.memory_pool import MemoryPool


def gather_kv_from_pages(
    pool: MemoryPool,
    page_ids: List[int],
    layer_idx: int,
    num_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gather KV from pages using page table.

    Args:
        pool: Memory pool containing pages.
        page_ids: List of page IDs for the sequence.
        layer_idx: Layer index.
        num_tokens: Total number of tokens to gather.

    Returns:
        Tuple of (cache_k, cache_v) tensors of shape
        [batch_size=1, num_kv_heads, num_tokens, head_dim].
    """
    page_size = pool.page_size
    num_kv_heads = pool.num_kv_heads
    head_dim = pool.head_dim

    # Initialize output tensors
    cache_k = torch.zeros(
        1, num_kv_heads, num_tokens, head_dim,
        device=pool.device, dtype=pool.dtype
    )
    cache_v = torch.zeros(
        1, num_kv_heads, num_tokens, head_dim,
        device=pool.device, dtype=pool.dtype
    )

    # Gather from pages
    token_idx = 0
    for page_id in page_ids:
        # Get page data
        page = pool.get_page(page_id)  # [num_layers, 2, page_size, num_kv_heads, head_dim]

        # Extract K and V for this layer
        page_k = page[layer_idx, 0]  # [page_size, num_kv_heads, head_dim]
        page_v = page[layer_idx, 1]  # [page_size, num_kv_heads, head_dim]

        # Determine how many tokens to copy from this page
        tokens_in_page = min(page_size, num_tokens - token_idx)

        # Copy to output (transpose to match expected shape)
        cache_k[0, :, token_idx:token_idx + tokens_in_page, :] = page_k[:tokens_in_page].transpose(0, 1)
        cache_v[0, :, token_idx:token_idx + tokens_in_page, :] = page_v[:tokens_in_page].transpose(0, 1)

        token_idx += tokens_in_page

        if token_idx >= num_tokens:
            break

    return cache_k, cache_v


def scatter_kv_to_pages(
    pool: MemoryPool,
    page_ids: List[int],
    layer_idx: int,
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
) -> None:
    """Scatter new KV to pages.

    Args:
        pool: Memory pool containing pages.
        page_ids: List of page IDs for the sequence.
        layer_idx: Layer index.
        cache_k: Key tensor of shape [batch_size, num_kv_heads, num_tokens, head_dim].
        cache_v: Value tensor of shape [batch_size, num_kv_heads, num_tokens, head_dim].
    """
    page_size = pool.page_size
    num_tokens = cache_k.shape[2]

    # Scatter to pages
    token_idx = 0
    for page_id in page_ids:
        # Get page data
        page = pool.get_page(page_id)  # [num_layers, 2, page_size, num_kv_heads, head_dim]

        # Determine how many tokens to copy to this page
        tokens_in_page = min(page_size, num_tokens - token_idx)

        # Copy from input (transpose to match page format)
        page[layer_idx, 0, :tokens_in_page] = cache_k[0, :, token_idx:token_idx + tokens_in_page].transpose(0, 1)
        page[layer_idx, 1, :tokens_in_page] = cache_v[0, :, token_idx:token_idx + tokens_in_page].transpose(0, 1)

        token_idx += tokens_in_page

        if token_idx >= num_tokens:
            break


def paged_attention(
    query: torch.Tensor,
    pool: MemoryPool,
    page_ids: List[int],
    layer_idx: int,
    cache_len: int,
) -> torch.Tensor:
    """Compute attention with paged KV cache.

    Args:
        query: Query tensor of shape [batch_size, num_q_heads, seq_len, head_dim].
        pool: Memory pool containing pages.
        page_ids: List of page IDs for the sequence.
        layer_idx: Layer index.
        cache_len: Length of cached KV.

    Returns:
        Attention output of shape [batch_size, num_q_heads, seq_len, head_dim].
    """
    # Gather KV from pages
    cache_k, cache_v = gather_kv_from_pages(pool, page_ids, layer_idx, cache_len)

    # Compute attention using standard attention mechanism
    from vibe_sgl_lite.models.qwen3.attention import compute_attention
    output = compute_attention(query, cache_k, cache_v)

    return output
