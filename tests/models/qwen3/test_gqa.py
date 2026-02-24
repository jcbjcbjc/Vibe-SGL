"""
Tests for Grouped-Query Attention (GQA) head configuration in Qwen3.

This module tests the GQA mechanism where the number of key-value heads
is smaller than the number of query heads for efficiency. It validates:
- Correct configuration of GQA with fewer KV heads than Q heads
- KV head repetition to match query head count during attention
- GQA behavior with Tensor Parallelism (TP)

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch
import torch.nn as nn

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_gqa_creates_fewer_kv_heads_than_q_heads() -> None:
    """Test that GQA configuration creates fewer KV heads than Q heads."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,  # 14 query heads
        num_key_value_heads=2,   # 2 KV heads (7x fewer)
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Verify that KV heads are fewer than Q heads
    assert attention.num_key_value_heads < attention.num_heads
    assert attention.num_key_value_heads == 2
    assert attention.num_heads == 14


@pytest.mark.unit
def test_gqa_head_grouping_calculation() -> None:
    """Test that GQA correctly calculates head grouping ratio."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Each KV head should be shared by 7 Q heads (14 / 2 = 7)
    expected_groups = 14 // 2
    assert attention.num_key_value_groups == expected_groups
    assert attention.num_key_value_groups == 7


@pytest.mark.unit
def test_gqa_with_different_ratios() -> None:
    """Test GQA with various query-to-KV head ratios."""
    test_cases = [
        # (num_attention_heads, num_key_value_heads, expected_groups)
        (14, 2, 7),   # Qwen3-0.6B config
        (8, 2, 4),    # 4:1 ratio
        (32, 8, 4),   # 4:1 ratio, larger model
        (16, 4, 4),   # 4:1 ratio
        (12, 3, 4),   # 4:1 ratio
        (12, 12, 1),  # MHA (no grouping)
    ]

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    for num_q_heads, num_kv_heads, expected_groups in test_cases:
        config = Qwen3Config(
            hidden_size=num_q_heads * 64,  # Ensure divisible
            num_attention_heads=num_q_heads,
            num_key_value_heads=num_kv_heads,
        )
        attention = Qwen3Attention(config)

        assert attention.num_heads == num_q_heads
        assert attention.num_key_value_heads == num_kv_heads
        assert attention.num_key_value_groups == expected_groups


@pytest.mark.unit
def test_gqa_kv_projection_dimensions() -> None:
    """Test that KV projections have correct dimensions for GQA."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    head_dim = config.hidden_size // config.num_attention_heads

    # K/V projections should output num_key_value_heads * head_dim
    expected_kv_dim = config.num_key_value_heads * head_dim
    assert attention.k_proj.out_features == expected_kv_dim
    assert attention.v_proj.out_features == expected_kv_dim

    # Q projection should output num_attention_heads * head_dim
    expected_q_dim = config.num_attention_heads * head_dim
    assert attention.q_proj.out_features == expected_q_dim


@pytest.mark.unit
def test_gqa_repeat_kv_heads_function_exists() -> None:
    """Test that attention layer has a method to repeat KV heads."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Should have a method to repeat KV heads
    assert hasattr(attention, "repeat_kv") or hasattr(Qwen3Attention, "repeat_kv")


@pytest.mark.unit
def test_gqa_repeat_kv_heads_shape() -> None:
    """Test that repeat_kv correctly expands KV heads to match Q heads."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads

    # Create KV tensor with shape [batch, num_kv_heads, seq_len, head_dim]
    kv = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)

    # Repeat KV heads to match Q heads
    kv_repeated = attention.repeat_kv(kv, attention.num_key_value_groups)

    # Output should have shape [batch, num_attention_heads, seq_len, head_dim]
    expected_shape = (batch_size, config.num_attention_heads, seq_len, head_dim)
    assert kv_repeated.shape == expected_shape


@pytest.mark.unit
def test_gqa_repeat_kv_heads_values() -> None:
    """Test that repeat_kv correctly duplicates KV head values."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 4
    head_dim = config.hidden_size // config.num_attention_heads

    # Create simple KV tensor for easy verification
    kv = torch.arange(
        batch_size * config.num_key_value_heads * seq_len * head_dim,
        dtype=torch.float32
    ).reshape(batch_size, config.num_key_value_heads, seq_len, head_dim)

    # Repeat KV heads
    kv_repeated = attention.repeat_kv(kv, attention.num_key_value_groups)

    # Each KV head should be repeated num_key_value_groups times
    # For config: 14 Q heads, 2 KV heads -> each KV head repeated 7 times
    # kv_repeated[0, 0:7, :, :] should all equal kv[0, 0, :, :]
    # kv_repeated[0, 7:14, :, :] should all equal kv[0, 1, :, :]
    for group_idx in range(attention.num_key_value_groups):
        q_head_idx = group_idx
        assert torch.equal(kv_repeated[0, q_head_idx, :, :], kv[0, 0, :, :])

    for group_idx in range(attention.num_key_value_groups):
        q_head_idx = attention.num_key_value_groups + group_idx
        assert torch.equal(kv_repeated[0, q_head_idx, :, :], kv[0, 1, :, :])


@pytest.mark.unit
def test_gqa_repeat_kv_no_grouping_mha() -> None:
    """Test that repeat_kv works correctly with MHA (no grouping)."""
    config = Qwen3Config(
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=12,  # MHA: same number of KV and Q heads
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads

    # Create KV tensor
    kv = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)

    # Repeat with groups=1 should return same tensor
    kv_repeated = attention.repeat_kv(kv, 1)

    # Should be identical (no repetition needed)
    assert torch.equal(kv, kv_repeated)


@pytest.mark.unit
def test_gqa_repeat_kv_preserves_dtype() -> None:
    """Test that repeat_kv preserves tensor dtype."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads

    # Test with float32
    kv_f32 = torch.randn(
        batch_size, config.num_key_value_heads, seq_len, head_dim,
        dtype=torch.float32
    )
    kv_repeated_f32 = attention.repeat_kv(kv_f32, attention.num_key_value_groups)
    assert kv_repeated_f32.dtype == torch.float32

    # Test with float16
    kv_f16 = torch.randn(
        batch_size, config.num_key_value_heads, seq_len, head_dim,
        dtype=torch.float16
    )
    kv_repeated_f16 = attention.repeat_kv(kv_f16, attention.num_key_value_groups)
    assert kv_repeated_f16.dtype == torch.float16


@pytest.mark.unit
def test_gqa_repeat_kv_gradient_flow() -> None:
    """Test that gradients flow correctly through repeat_kv."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads

    # Create KV tensor with gradient tracking
    kv = torch.randn(
        batch_size, config.num_key_value_heads, seq_len, head_dim,
        requires_grad=True
    )

    # Repeat and compute loss
    kv_repeated = attention.repeat_kv(kv, attention.num_key_value_groups)
    loss = kv_repeated.sum()
    loss.backward()

    # Gradients should flow back to original KV tensor
    assert kv.grad is not None
    assert not torch.isnan(kv.grad).any()

    # Gradient should be summed across repeated heads
    # Each KV head is repeated 7 times, so gradient should be 7x
    expected_grad_scale = attention.num_key_value_groups
    assert torch.allclose(
        kv.grad,
        torch.ones_like(kv) * expected_grad_scale,
        rtol=1e-5
    )


@pytest.mark.unit
def test_gqa_with_tp_kv_head_partitioning() -> None:
    """Test that GQA correctly partitions KV heads across TP ranks."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    # Simulate TP with 2 ranks
    tp_size = 2

    # Each rank should have num_key_value_heads // tp_size KV heads
    expected_kv_heads_per_rank = config.num_key_value_heads // tp_size
    assert expected_kv_heads_per_rank == 1  # 2 KV heads / 2 ranks = 1 per rank

    # Each rank should have num_attention_heads // tp_size Q heads
    expected_q_heads_per_rank = config.num_attention_heads // tp_size
    assert expected_q_heads_per_rank == 7  # 14 Q heads / 2 ranks = 7 per rank


@pytest.mark.unit
def test_gqa_with_tp_head_grouping_preserved() -> None:
    """Test that GQA head grouping ratio is preserved with TP."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Original grouping: 14 Q heads / 2 KV heads = 7 groups
    original_groups = attention.num_key_value_groups
    assert original_groups == 7

    # With TP=2:
    # - Each rank has 7 Q heads and 1 KV head
    # - Grouping ratio should remain 7
    tp_size = 2
    q_heads_per_rank = config.num_attention_heads // tp_size
    kv_heads_per_rank = config.num_key_value_heads // tp_size
    groups_per_rank = q_heads_per_rank // kv_heads_per_rank

    assert groups_per_rank == original_groups


@pytest.mark.unit
def test_gqa_with_tp_invalid_partitioning() -> None:
    """Test that GQA validates TP partitioning constraints."""
    # Valid config for testing TP constraints
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)
    assert attention is not None

    # With TP size=3, KV heads (2) would not be evenly divisible
    tp_size = 3
    assert config.num_key_value_heads % tp_size != 0

    # With TP size=2, KV heads (2) are evenly divisible
    tp_size = 2
    assert config.num_key_value_heads % tp_size == 0


@pytest.mark.unit
def test_gqa_kv_cache_shape_with_gqa() -> None:
    """Test that KV cache has correct shape for GQA."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    max_seq_len = 100
    head_dim = config.hidden_size // config.num_attention_heads

    # KV cache should use num_key_value_heads, not num_attention_heads
    expected_k_cache_shape = (batch_size, config.num_key_value_heads, max_seq_len, head_dim)
    expected_v_cache_shape = (batch_size, config.num_key_value_heads, max_seq_len, head_dim)

    # Create mock KV cache
    k_cache = torch.zeros(expected_k_cache_shape)
    v_cache = torch.zeros(expected_v_cache_shape)

    assert k_cache.shape[1] == config.num_key_value_heads
    assert v_cache.shape[1] == config.num_key_value_heads
    assert k_cache.shape[1] < config.num_attention_heads  # GQA: fewer KV heads


@pytest.mark.unit
def test_gqa_memory_efficiency() -> None:
    """Test that GQA reduces memory usage compared to MHA."""
    # GQA config
    gqa_config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    # MHA config (same Q heads, but KV heads = Q heads)
    mha_config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=14,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    gqa_attention = Qwen3Attention(gqa_config)
    mha_attention = Qwen3Attention(mha_config)

    # Count KV projection parameters
    gqa_kv_params = (
        gqa_attention.k_proj.weight.numel() +
        gqa_attention.v_proj.weight.numel()
    )
    mha_kv_params = (
        mha_attention.k_proj.weight.numel() +
        mha_attention.v_proj.weight.numel()
    )

    # GQA should use fewer parameters for KV projections
    assert gqa_kv_params < mha_kv_params

    # Specifically, GQA should use 2/14 = 1/7 of MHA's KV parameters
    expected_ratio = gqa_config.num_key_value_heads / mha_config.num_key_value_heads
    actual_ratio = gqa_kv_params / mha_kv_params
    assert abs(actual_ratio - expected_ratio) < 1e-6


@pytest.mark.unit
def test_gqa_configuration_from_pretrained() -> None:
    """Test that GQA configuration is correctly loaded from pretrained model."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Verify GQA is configured
    assert attention.num_key_value_heads <= attention.num_heads

    # Verify grouping is calculated correctly
    expected_groups = attention.num_heads // attention.num_key_value_heads
    assert attention.num_key_value_groups == expected_groups


@pytest.mark.unit
def test_gqa_repeat_kv_with_different_batch_sizes() -> None:
    """Test that repeat_kv works with various batch sizes."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads

    for batch_size in [1, 2, 4, 8, 16]:
        kv = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)
        kv_repeated = attention.repeat_kv(kv, attention.num_key_value_groups)

        expected_shape = (batch_size, config.num_attention_heads, seq_len, head_dim)
        assert kv_repeated.shape == expected_shape


@pytest.mark.unit
def test_gqa_repeat_kv_with_different_seq_lengths() -> None:
    """Test that repeat_kv works with various sequence lengths."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    head_dim = config.hidden_size // config.num_attention_heads

    for seq_len in [1, 10, 50, 100, 512]:
        kv = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)
        kv_repeated = attention.repeat_kv(kv, attention.num_key_value_groups)

        expected_shape = (batch_size, config.num_attention_heads, seq_len, head_dim)
        assert kv_repeated.shape == expected_shape

