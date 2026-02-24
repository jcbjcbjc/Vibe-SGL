"""
Tests for Q/K/V projection initialization in Qwen3 attention layer.

This module tests the initialization and configuration of Query, Key, and Value
projection layers in the Qwen3 attention mechanism. It validates correct weight
shapes, GQA (Grouped-Query Attention) configuration, and TP (Tensor Parallelism)
awareness.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch
import torch.nn as nn

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_qkv_projection_initialization() -> None:
    """Test that Q/K/V projections can be initialized with correct dimensions."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    # Import will fail until implementation exists
    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Check that projections exist
    assert hasattr(attention, "q_proj")
    assert hasattr(attention, "k_proj")
    assert hasattr(attention, "v_proj")
    assert isinstance(attention.q_proj, nn.Linear)
    assert isinstance(attention.k_proj, nn.Linear)
    assert isinstance(attention.v_proj, nn.Linear)


@pytest.mark.unit
def test_q_projection_shape() -> None:
    """Test that Q projection has correct input/output dimensions."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Q projection: hidden_size -> num_attention_heads * head_dim
    head_dim = config.hidden_size // config.num_attention_heads
    expected_q_out = config.num_attention_heads * head_dim

    assert attention.q_proj.in_features == config.hidden_size
    assert attention.q_proj.out_features == expected_q_out
    assert attention.q_proj.weight.shape == (expected_q_out, config.hidden_size)


@pytest.mark.unit
def test_kv_projection_shapes_with_gqa() -> None:
    """Test that K/V projections have correct dimensions for GQA."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,  # GQA: fewer KV heads than Q heads
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # K/V projections: hidden_size -> num_key_value_heads * head_dim
    head_dim = config.hidden_size // config.num_attention_heads
    expected_kv_out = config.num_key_value_heads * head_dim

    assert attention.k_proj.in_features == config.hidden_size
    assert attention.k_proj.out_features == expected_kv_out
    assert attention.k_proj.weight.shape == (expected_kv_out, config.hidden_size)

    assert attention.v_proj.in_features == config.hidden_size
    assert attention.v_proj.out_features == expected_kv_out
    assert attention.v_proj.weight.shape == (expected_kv_out, config.hidden_size)


@pytest.mark.unit
def test_head_dimension_calculation() -> None:
    """Test that head dimension is correctly calculated from config."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Head dimension should be hidden_size / num_attention_heads
    expected_head_dim = 896 // 14  # = 64
    assert hasattr(attention, "head_dim")
    assert attention.head_dim == expected_head_dim


@pytest.mark.unit
def test_num_heads_stored() -> None:
    """Test that attention layer stores number of heads correctly."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    assert hasattr(attention, "num_heads")
    assert attention.num_heads == config.num_attention_heads
    assert hasattr(attention, "num_key_value_heads")
    assert attention.num_key_value_heads == config.num_key_value_heads


@pytest.mark.unit
def test_gqa_head_grouping_ratio() -> None:
    """Test that GQA head grouping ratio is correctly calculated."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Each KV head should be shared by multiple Q heads
    expected_ratio = 14 // 2  # = 7
    assert hasattr(attention, "num_key_value_groups")
    assert attention.num_key_value_groups == expected_ratio


@pytest.mark.unit
def test_projection_bias_disabled() -> None:
    """Test that Q/K/V projections have bias disabled (Qwen3 standard)."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Qwen3 uses bias=False for attention projections
    assert attention.q_proj.bias is None
    assert attention.k_proj.bias is None
    assert attention.v_proj.bias is None


@pytest.mark.unit
def test_projection_weights_initialized() -> None:
    """Test that projection weights are properly initialized."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Weights should be initialized (not all zeros)
    assert not torch.allclose(attention.q_proj.weight, torch.zeros_like(attention.q_proj.weight))
    assert not torch.allclose(attention.k_proj.weight, torch.zeros_like(attention.k_proj.weight))
    assert not torch.allclose(attention.v_proj.weight, torch.zeros_like(attention.v_proj.weight))

    # Weights should have reasonable values (not NaN or Inf)
    assert not torch.isnan(attention.q_proj.weight).any()
    assert not torch.isnan(attention.k_proj.weight).any()
    assert not torch.isnan(attention.v_proj.weight).any()
    assert not torch.isinf(attention.q_proj.weight).any()
    assert not torch.isinf(attention.k_proj.weight).any()
    assert not torch.isinf(attention.v_proj.weight).any()


@pytest.mark.unit
def test_qkv_projections_with_different_configs() -> None:
    """Test Q/K/V projections with various model configurations."""
    configs = [
        # Qwen3-0.6B-like config
        Qwen3Config(hidden_size=896, num_attention_heads=14, num_key_value_heads=2),
        # Smaller test config
        Qwen3Config(hidden_size=512, num_attention_heads=8, num_key_value_heads=2),
        # Larger config
        Qwen3Config(hidden_size=2048, num_attention_heads=32, num_key_value_heads=8),
        # MHA (Multi-Head Attention) - no GQA
        Qwen3Config(hidden_size=768, num_attention_heads=12, num_key_value_heads=12),
    ]

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    for config in configs:
        attention = Qwen3Attention(config)
        head_dim = config.hidden_size // config.num_attention_heads

        # Validate Q projection
        assert attention.q_proj.in_features == config.hidden_size
        assert attention.q_proj.out_features == config.num_attention_heads * head_dim

        # Validate K/V projections
        assert attention.k_proj.in_features == config.hidden_size
        assert attention.k_proj.out_features == config.num_key_value_heads * head_dim
        assert attention.v_proj.in_features == config.hidden_size
        assert attention.v_proj.out_features == config.num_key_value_heads * head_dim


@pytest.mark.unit
def test_qkv_forward_pass_shapes() -> None:
    """Test that Q/K/V projections produce correct output shapes."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size

    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Apply projections
    q = attention.q_proj(x)
    k = attention.k_proj(x)
    v = attention.v_proj(x)

    head_dim = config.hidden_size // config.num_attention_heads

    # Check output shapes
    assert q.shape == (batch_size, seq_len, config.num_attention_heads * head_dim)
    assert k.shape == (batch_size, seq_len, config.num_key_value_heads * head_dim)
    assert v.shape == (batch_size, seq_len, config.num_key_value_heads * head_dim)


@pytest.mark.unit
def test_qkv_projections_gradient_flow() -> None:
    """Test that gradients flow through Q/K/V projections."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

    # Forward pass
    q = attention.q_proj(x)
    k = attention.k_proj(x)
    v = attention.v_proj(x)

    # Backward pass
    loss = (q.sum() + k.sum() + v.sum())
    loss.backward()

    # Check gradients exist
    assert attention.q_proj.weight.grad is not None
    assert attention.k_proj.weight.grad is not None
    assert attention.v_proj.weight.grad is not None
    assert x.grad is not None

    # Check gradients are valid
    assert not torch.isnan(attention.q_proj.weight.grad).any()
    assert not torch.isnan(attention.k_proj.weight.grad).any()
    assert not torch.isnan(attention.v_proj.weight.grad).any()


@pytest.mark.unit
def test_qkv_projections_dtype_preservation() -> None:
    """Test that Q/K/V projections preserve input dtype."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    # Test with float32
    attention_f32 = Qwen3Attention(config)
    x_f32 = torch.randn(2, 10, config.hidden_size, dtype=torch.float32)
    q_f32 = attention_f32.q_proj(x_f32)
    k_f32 = attention_f32.k_proj(x_f32)
    v_f32 = attention_f32.v_proj(x_f32)

    assert q_f32.dtype == torch.float32
    assert k_f32.dtype == torch.float32
    assert v_f32.dtype == torch.float32


@pytest.mark.unit
def test_attention_config_stored() -> None:
    """Test that attention layer stores config reference."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    # Config should be accessible for later use
    assert hasattr(attention, "config")
    assert attention.config == config


@pytest.mark.unit
def test_qkv_projections_deterministic() -> None:
    """Test that Q/K/V projections produce deterministic outputs."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    x = torch.randn(2, 10, config.hidden_size)

    # Run twice with same input
    q1 = attention.q_proj(x)
    k1 = attention.k_proj(x)
    v1 = attention.v_proj(x)

    q2 = attention.q_proj(x)
    k2 = attention.k_proj(x)
    v2 = attention.v_proj(x)

    # Outputs should be identical
    assert torch.equal(q1, q2)
    assert torch.equal(k1, k2)
    assert torch.equal(v1, v2)


@pytest.mark.unit
def test_qkv_projections_from_pretrained_config() -> None:
    """Test Q/K/V projections with config loaded from HuggingFace."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    head_dim = config.hidden_size // config.num_attention_heads

    # Validate projections match pretrained config
    assert attention.q_proj.in_features == config.hidden_size
    assert attention.q_proj.out_features == config.num_attention_heads * head_dim
    assert attention.k_proj.out_features == config.num_key_value_heads * head_dim
    assert attention.v_proj.out_features == config.num_key_value_heads * head_dim


@pytest.mark.unit
def test_gqa_constraint_validation() -> None:
    """Test that GQA constraints are validated during initialization."""
    # Valid GQA config: num_attention_heads divisible by num_key_value_heads
    valid_config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,  # 14 % 2 == 0 ✓
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    # Should initialize successfully
    attention = Qwen3Attention(valid_config)
    assert attention is not None


@pytest.mark.unit
def test_qkv_projection_parameter_count() -> None:
    """Test that Q/K/V projections have expected parameter count."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

    attention = Qwen3Attention(config)

    head_dim = config.hidden_size // config.num_attention_heads

    # Q projection parameters
    q_params = config.hidden_size * (config.num_attention_heads * head_dim)
    assert attention.q_proj.weight.numel() == q_params

    # K/V projection parameters
    kv_params = config.hidden_size * (config.num_key_value_heads * head_dim)
    assert attention.k_proj.weight.numel() == kv_params
    assert attention.v_proj.weight.numel() == kv_params
