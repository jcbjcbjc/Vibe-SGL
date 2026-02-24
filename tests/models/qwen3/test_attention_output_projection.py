"""
Tests for attention output projection in Qwen3.

This module tests the output projection layer in the attention mechanism:
- Output projection layer initialization (o_proj)
- Projection from attention output back to hidden_size
- Integration with attention forward pass
- Shape transformations and correctness

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention
from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_attention_has_output_projection() -> None:
    """Test that attention layer has an output projection layer."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    # Should have o_proj layer
    assert hasattr(attention, "o_proj")
    assert isinstance(attention.o_proj, torch.nn.Linear)


@pytest.mark.unit
def test_output_projection_dimensions() -> None:
    """Test that output projection has correct dimensions."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    # o_proj should project from hidden_size to hidden_size
    # (num_heads * head_dim) -> hidden_size
    assert attention.o_proj.in_features == config.hidden_size
    assert attention.o_proj.out_features == config.hidden_size


@pytest.mark.unit
def test_output_projection_no_bias() -> None:
    """Test that output projection has no bias (following Qwen3 architecture)."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    # o_proj should not have bias
    assert attention.o_proj.bias is None


@pytest.mark.unit
def test_output_projection_forward_shape() -> None:
    """Test that output projection produces correct output shape."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10

    # Create attention output (before projection)
    # Shape: [batch_size, seq_len, hidden_size]
    attn_output = torch.randn(batch_size, seq_len, config.hidden_size)

    # Apply output projection
    projected = attention.o_proj(attn_output)

    # Output should have same shape
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert projected.shape == expected_shape


@pytest.mark.unit
def test_attention_forward_includes_output_projection() -> None:
    """Test that attention forward pass includes output projection."""
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

    # Forward pass should include output projection
    output = attention.forward(hidden_states, freqs_cis)

    # Output should have correct shape
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape


@pytest.mark.unit
def test_output_projection_gradient_flow() -> None:
    """Test that gradients flow through output projection."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size,
        requires_grad=True
    )

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    output = attention.forward(hidden_states, freqs_cis)

    # Compute loss and backpropagate
    loss = output.sum()
    loss.backward()

    # Gradients should flow to output projection
    assert attention.o_proj.weight.grad is not None
    assert not torch.isnan(attention.o_proj.weight.grad).any()


@pytest.mark.unit
def test_output_projection_with_different_configs() -> None:
    """Test output projection with various model configurations."""
    configs = [
        Qwen3Config(hidden_size=128, num_attention_heads=4, num_key_value_heads=2),
        Qwen3Config(hidden_size=256, num_attention_heads=8, num_key_value_heads=4),
        Qwen3Config(hidden_size=512, num_attention_heads=16, num_key_value_heads=8),
        Qwen3Config(hidden_size=896, num_attention_heads=14, num_key_value_heads=2),
    ]

    for config in configs:
        attention = Qwen3Attention(config)

        # o_proj dimensions should match hidden_size
        assert attention.o_proj.in_features == config.hidden_size
        assert attention.o_proj.out_features == config.hidden_size

        # Test forward pass
        batch_size = 2
        seq_len = 5
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        head_dim = config.hidden_size // config.num_attention_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

        output = attention.forward(hidden_states, freqs_cis)
        assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_output_projection_preserves_dtype() -> None:
    """Test that output projection preserves input dtype."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10

    # Test with float32
    attn_output_f32 = torch.randn(
        batch_size, seq_len, config.hidden_size,
        dtype=torch.float32
    )
    projected_f32 = attention.o_proj(attn_output_f32)
    assert projected_f32.dtype == torch.float32


@pytest.mark.unit
def test_output_projection_no_nan_or_inf() -> None:
    """Test that output projection never produces NaN or Inf."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10

    # Test with various input patterns
    test_inputs = [
        torch.randn(batch_size, seq_len, config.hidden_size),
        torch.zeros(batch_size, seq_len, config.hidden_size),
        torch.ones(batch_size, seq_len, config.hidden_size),
        torch.randn(batch_size, seq_len, config.hidden_size) * 100,
        torch.randn(batch_size, seq_len, config.hidden_size) * 0.01,
    ]

    for attn_output in test_inputs:
        projected = attention.o_proj(attn_output)

        assert not torch.isnan(projected).any()
        assert not torch.isinf(projected).any()


@pytest.mark.unit
def test_output_projection_deterministic() -> None:
    """Test that output projection produces deterministic outputs."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    attn_output = torch.randn(batch_size, seq_len, config.hidden_size)

    # Run twice with same input
    projected1 = attention.o_proj(attn_output)
    projected2 = attention.o_proj(attn_output)

    # Outputs should be identical
    assert torch.equal(projected1, projected2)


@pytest.mark.unit
def test_attention_forward_output_differs_from_unprojected() -> None:
    """Test that attention output with projection differs from unprojected output."""
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

    # Get attention output before projection (manually compute)
    q_proj = attention.q_proj(hidden_states)
    k_proj = attention.k_proj(hidden_states)
    v_proj = attention.v_proj(hidden_states)

    q = q_proj.view(batch_size, seq_len, config.num_attention_heads, head_dim)
    k = k_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim)
    v = v_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim)

    from vibe_sgl_lite.models.qwen3.rope import apply_rotary_emb
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)

    q_t = q_rope.transpose(1, 2)
    k_t = k_rope.transpose(1, 2)
    v_t = v.transpose(1, 2)

    k_repeated = attention.repeat_kv(k_t, attention.num_key_value_groups)
    v_repeated = attention.repeat_kv(v_t, attention.num_key_value_groups)

    scores = torch.matmul(q_t, k_repeated.transpose(-2, -1)) / (head_dim ** 0.5)
    attn_weights = torch.nn.functional.softmax(scores, dim=-1)
    attn_output_unprojected = torch.matmul(attn_weights, v_repeated)
    attn_output_unprojected = attn_output_unprojected.transpose(1, 2).contiguous()
    attn_output_unprojected = attn_output_unprojected.view(
        batch_size, seq_len, config.hidden_size
    )

    # Get full attention output (with projection)
    output_with_projection = attention.forward(hidden_states, freqs_cis)

    # Outputs should differ (unless o_proj is identity, which is unlikely)
    # We check that they're not equal with high probability
    assert not torch.allclose(attn_output_unprojected, output_with_projection, atol=1e-3)


@pytest.mark.unit
def test_output_projection_with_qwen3_config() -> None:
    """Test output projection with actual Qwen3 configuration."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    attention = Qwen3Attention(config)

    # Check o_proj dimensions
    assert attention.o_proj.in_features == config.hidden_size
    assert attention.o_proj.out_features == config.hidden_size
    assert attention.o_proj.bias is None

    # Test forward pass
    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        head_dim,
        max_seq_len=seq_len,
        theta=config.rope_theta,
    )

    output = attention.forward(hidden_states, freqs_cis)

    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape


@pytest.mark.unit
def test_output_projection_weight_initialization() -> None:
    """Test that output projection weights are properly initialized."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    # Weights should be initialized (not all zeros)
    assert not torch.all(attention.o_proj.weight == 0)

    # Weights should have reasonable magnitude
    weight_std = attention.o_proj.weight.std()
    assert 0.001 < weight_std < 1.0


@pytest.mark.unit
def test_output_projection_with_single_token() -> None:
    """Test output projection with single token (decode phase)."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 1
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=100)

    output = attention.forward(hidden_states, freqs_cis)

    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape


@pytest.mark.unit
def test_output_projection_with_long_sequence() -> None:
    """Test output projection with long sequence."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 512
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    output = attention.forward(hidden_states, freqs_cis)

    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()

