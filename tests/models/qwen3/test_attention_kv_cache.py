"""
Tests for attention with KV cache in Qwen3.

This module tests the KV cache functionality in attention:
- KV cache initialization and storage
- Attention computation with cached KV values
- Position offset handling for incremental decoding
- Cache update and concatenation
- Correctness of cached vs non-cached attention

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention
from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_attention_forward_accepts_kv_cache_parameter() -> None:
    """Test that attention forward method accepts KV cache parameter."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    # Should accept kv_cache parameter (None for first call)
    output = attention.forward(hidden_states, freqs_cis, kv_cache=None)

    assert output is not None


@pytest.mark.unit
def test_attention_forward_returns_kv_cache() -> None:
    """Test that attention forward returns KV cache when requested."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    # Forward should return both output and new KV cache
    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    assert output is not None
    assert kv_cache is not None


@pytest.mark.unit
def test_kv_cache_structure() -> None:
    """Test that KV cache has correct structure (tuple of K and V tensors)."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    # KV cache should be a tuple of (K, V)
    assert isinstance(kv_cache, tuple)
    assert len(kv_cache) == 2

    k_cache, v_cache = kv_cache
    assert isinstance(k_cache, torch.Tensor)
    assert isinstance(v_cache, torch.Tensor)


@pytest.mark.unit
def test_kv_cache_shape() -> None:
    """Test that KV cache tensors have correct shape."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache, v_cache = kv_cache

    # KV cache shape: [batch, num_kv_heads, seq_len, head_dim]
    expected_shape = (batch_size, config.num_key_value_heads, seq_len, head_dim)
    assert k_cache.shape == expected_shape
    assert v_cache.shape == expected_shape


@pytest.mark.unit
def test_attention_with_existing_kv_cache() -> None:
    """Test attention computation with existing KV cache."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    prefill_len = 10
    decode_len = 1

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=prefill_len + decode_len)

    # Prefill phase: process initial sequence
    prefill_hidden = torch.randn(batch_size, prefill_len, config.hidden_size)
    prefill_output, kv_cache = attention.forward(
        prefill_hidden, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    # Decode phase: process single token with cached KV
    decode_hidden = torch.randn(batch_size, decode_len, config.hidden_size)
    decode_output, new_kv_cache = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache,
        position_offset=prefill_len,
        return_kv_cache=True
    )

    assert decode_output is not None
    assert new_kv_cache is not None


@pytest.mark.unit
def test_kv_cache_concatenation() -> None:
    """Test that new KV values are concatenated with cached KV."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    prefill_len = 10
    decode_len = 1

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=prefill_len + decode_len)

    # Prefill phase
    prefill_hidden = torch.randn(batch_size, prefill_len, config.hidden_size)
    _, kv_cache = attention.forward(
        prefill_hidden, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache, v_cache = kv_cache
    assert k_cache.shape[2] == prefill_len  # seq_len dimension

    # Decode phase
    decode_hidden = torch.randn(batch_size, decode_len, config.hidden_size)
    _, new_kv_cache = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache,
        position_offset=prefill_len,
        return_kv_cache=True
    )

    new_k_cache, new_v_cache = new_kv_cache

    # New cache should have concatenated length
    expected_len = prefill_len + decode_len
    assert new_k_cache.shape[2] == expected_len
    assert new_v_cache.shape[2] == expected_len


@pytest.mark.unit
def test_kv_cache_preserves_previous_values() -> None:
    """Test that cached KV values are preserved when adding new tokens."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    prefill_len = 5
    decode_len = 1

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=prefill_len + decode_len)

    # Prefill phase
    prefill_hidden = torch.randn(batch_size, prefill_len, config.hidden_size)
    _, kv_cache = attention.forward(
        prefill_hidden, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache_old, v_cache_old = kv_cache

    # Decode phase
    decode_hidden = torch.randn(batch_size, decode_len, config.hidden_size)
    _, new_kv_cache = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache,
        position_offset=prefill_len,
        return_kv_cache=True
    )

    new_k_cache, new_v_cache = new_kv_cache

    # First prefill_len tokens should match the old cache
    assert_tensors_close(
        new_k_cache[:, :, :prefill_len, :],
        k_cache_old,
        atol=1e-6,
        rtol=1e-5
    )
    assert_tensors_close(
        new_v_cache[:, :, :prefill_len, :],
        v_cache_old,
        atol=1e-6,
        rtol=1e-5
    )


@pytest.mark.unit
def test_attention_output_with_cache_matches_without_cache() -> None:
    """Test that attention with cache produces same output as without cache."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    total_len = 11

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=total_len)

    # Generate full sequence
    full_hidden = torch.randn(batch_size, total_len, config.hidden_size)

    # Method 1: Process full sequence at once (no cache)
    output_no_cache = attention.forward(full_hidden, freqs_cis)

    # Method 2: Process with cache (prefill + decode)
    prefill_len = 10
    prefill_hidden = full_hidden[:, :prefill_len, :]
    decode_hidden = full_hidden[:, prefill_len:, :]

    # Prefill
    _, kv_cache = attention.forward(
        prefill_hidden, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    # Decode
    decode_output, _ = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache,
        position_offset=prefill_len,
        return_kv_cache=True
    )

    # The decode output should match the last token of no-cache output
    assert_tensors_close(
        decode_output[:, -1, :],
        output_no_cache[:, -1, :],
        atol=1e-5,
        rtol=1e-4
    )


@pytest.mark.unit
def test_multiple_decode_steps_with_cache() -> None:
    """Test multiple decode steps with incremental cache updates."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    prefill_len = 5
    num_decode_steps = 3

    head_dim = config.hidden_size // config.num_attention_heads
    max_len = prefill_len + num_decode_steps
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=max_len)

    # Prefill phase
    prefill_hidden = torch.randn(batch_size, prefill_len, config.hidden_size)
    _, kv_cache = attention.forward(
        prefill_hidden, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    # Multiple decode steps
    for step in range(num_decode_steps):
        decode_hidden = torch.randn(batch_size, 1, config.hidden_size)
        position_offset = prefill_len + step

        output, kv_cache = attention.forward(
            decode_hidden,
            freqs_cis,
            kv_cache=kv_cache,
            position_offset=position_offset,
            return_kv_cache=True
        )

        # Cache should grow by 1 each step
        k_cache, v_cache = kv_cache
        expected_len = prefill_len + step + 1
        assert k_cache.shape[2] == expected_len
        assert v_cache.shape[2] == expected_len


@pytest.mark.unit
def test_kv_cache_with_batch_size_one() -> None:
    """Test KV cache with batch size 1 (typical inference scenario)."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 10

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache, v_cache = kv_cache

    # Should work with batch size 1
    assert k_cache.shape[0] == 1
    assert v_cache.shape[0] == 1


@pytest.mark.unit
def test_kv_cache_position_offset_correctness() -> None:
    """Test that position offset correctly shifts RoPE for cached tokens."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    prefill_len = 10
    decode_len = 1

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=prefill_len + decode_len)

    # Create a prefill sequence
    prefill_hidden = torch.randn(batch_size, prefill_len, config.hidden_size)

    # Create identical decode hidden states for testing
    decode_hidden = torch.randn(batch_size, decode_len, config.hidden_size)

    # Scenario 1: Decode at position 0 (wrong - no prefill context)
    _, kv_cache_empty = attention.forward(
        torch.randn(batch_size, 1, config.hidden_size),  # dummy token at pos 0
        freqs_cis,
        kv_cache=None,
        return_kv_cache=True
    )
    output_wrong_pos, _ = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache_empty,
        position_offset=1,  # position 1
        return_kv_cache=True
    )

    # Scenario 2: Decode at position prefill_len (correct - after prefill)
    _, kv_cache_prefill = attention.forward(
        prefill_hidden,
        freqs_cis,
        kv_cache=None,
        return_kv_cache=True
    )
    output_correct_pos, _ = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache_prefill,
        position_offset=prefill_len,  # position 10
        return_kv_cache=True
    )

    # Outputs should be different due to different cached context and positions
    assert not torch.allclose(output_wrong_pos, output_correct_pos)


@pytest.mark.unit
def test_kv_cache_dtype_preservation() -> None:
    """Test that KV cache preserves tensor dtype."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    # Test with float32
    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size,
        dtype=torch.float32
    )

    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache, v_cache = kv_cache

    # Cache should preserve dtype
    assert k_cache.dtype == torch.float32
    assert v_cache.dtype == torch.float32


@pytest.mark.unit
def test_kv_cache_no_nan_or_inf() -> None:
    """Test that KV cache never contains NaN or Inf values."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache, v_cache = kv_cache

    # Cache should not contain NaN or Inf
    assert not torch.isnan(k_cache).any()
    assert not torch.isinf(k_cache).any()
    assert not torch.isnan(v_cache).any()
    assert not torch.isinf(v_cache).any()


@pytest.mark.unit
def test_kv_cache_with_gqa() -> None:
    """Test that KV cache works correctly with GQA (fewer KV heads than Q heads)."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,  # 14 query heads
        num_key_value_heads=2,   # 2 KV heads (GQA)
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache, v_cache = kv_cache

    # Cache should have num_key_value_heads, not num_attention_heads
    assert k_cache.shape[1] == config.num_key_value_heads
    assert v_cache.shape[1] == config.num_key_value_heads


@pytest.mark.unit
def test_empty_kv_cache_initialization() -> None:
    """Test that passing None as kv_cache initializes empty cache."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Passing None should work and create new cache
    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    assert kv_cache is not None
    k_cache, v_cache = kv_cache
    assert k_cache.shape[2] == seq_len


@pytest.mark.unit
def test_kv_cache_gradient_flow() -> None:
    """Test that gradients flow through KV cache operations."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size,
        requires_grad=True
    )

    output, kv_cache = attention.forward(
        hidden_states, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    # Compute loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Gradients should flow to input
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()


@pytest.mark.unit
def test_long_sequence_with_cache() -> None:
    """Test KV cache with longer sequences."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    prefill_len = 100
    decode_len = 1

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=prefill_len + decode_len)

    # Prefill with long sequence
    prefill_hidden = torch.randn(batch_size, prefill_len, config.hidden_size)
    _, kv_cache = attention.forward(
        prefill_hidden, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    k_cache, v_cache = kv_cache
    assert k_cache.shape[2] == prefill_len

    # Decode step
    decode_hidden = torch.randn(batch_size, decode_len, config.hidden_size)
    output, new_kv_cache = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache,
        position_offset=prefill_len,
        return_kv_cache=True
    )

    new_k_cache, new_v_cache = new_kv_cache
    assert new_k_cache.shape[2] == prefill_len + decode_len


@pytest.mark.unit
def test_kv_cache_with_qwen3_config() -> None:
    """Test KV cache with actual Qwen3 configuration."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    attention = Qwen3Attention(config)

    batch_size = 1
    prefill_len = 10
    decode_len = 1

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        head_dim,
        max_seq_len=prefill_len + decode_len,
        theta=config.rope_theta,
    )

    # Prefill
    prefill_hidden = torch.randn(batch_size, prefill_len, config.hidden_size)
    _, kv_cache = attention.forward(
        prefill_hidden, freqs_cis, kv_cache=None, return_kv_cache=True
    )

    # Decode
    decode_hidden = torch.randn(batch_size, decode_len, config.hidden_size)
    output, new_kv_cache = attention.forward(
        decode_hidden,
        freqs_cis,
        kv_cache=kv_cache,
        position_offset=prefill_len,
        return_kv_cache=True
    )

    assert output.shape == (batch_size, decode_len, config.hidden_size)
    new_k_cache, new_v_cache = new_kv_cache
    assert new_k_cache.shape[2] == prefill_len + decode_len

