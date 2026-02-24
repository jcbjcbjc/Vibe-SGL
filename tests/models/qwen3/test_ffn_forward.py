"""
Tests for FFN forward pass in Qwen3.

This module tests the complete forward pass of the Feed-Forward Network (FFN)
layer in Qwen3, which combines:
1. SwiGLU activation: Swish(gate_proj(x)) * up_proj(x)
2. Down projection: down_proj(swiglu_output)

The forward pass should implement: FFN(x) = down_proj(SwiGLU(x))

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_ffn_forward_method_exists() -> None:
    """Test that FFN layer has a forward method."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # FFN should have a forward method
    assert hasattr(ffn, "forward")
    assert callable(ffn.forward)


@pytest.mark.unit
def test_ffn_forward_output_shape() -> None:
    """Test that FFN forward pass produces correct output shape."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass
    output = ffn.forward(x)

    # Output should have same shape as input (hidden_size preserved)
    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_ffn_forward_callable_directly() -> None:
    """Test that FFN can be called directly (nn.Module __call__)."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Should be callable as ffn(x)
    output = ffn(x)

    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_ffn_forward_matches_manual_computation() -> None:
    """Test that forward pass matches manual SwiGLU + down_proj computation."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass
    output = ffn(x)

    # Manual computation
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out
    expected_output = ffn.down_proj(swiglu_out)

    # Outputs should match
    assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-7)


@pytest.mark.unit
def test_ffn_forward_with_different_batch_sizes() -> None:
    """Test FFN forward pass with various batch sizes."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    seq_len = 10

    for batch_size in [1, 2, 4, 8, 16]:
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = ffn(x)

        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.unit
def test_ffn_forward_with_different_sequence_lengths() -> None:
    """Test FFN forward pass with various sequence lengths."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2

    for seq_len in [1, 5, 10, 32, 128]:
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        output = ffn(x)

        assert output.shape == (batch_size, seq_len, config.hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.unit
def test_ffn_forward_gradient_flow() -> None:
    """Test that gradients flow correctly through FFN forward pass."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

    # Forward pass
    output = ffn(x)

    # Backward pass
    loss = output.sum()
    loss.backward()

    # Check gradients exist for all parameters
    assert ffn.gate_proj.weight.grad is not None
    assert ffn.up_proj.weight.grad is not None
    assert ffn.down_proj.weight.grad is not None
    assert x.grad is not None

    # Check gradients are valid
    assert not torch.isnan(ffn.gate_proj.weight.grad).any()
    assert not torch.isnan(ffn.up_proj.weight.grad).any()
    assert not torch.isnan(ffn.down_proj.weight.grad).any()
    assert not torch.isnan(x.grad).any()


@pytest.mark.unit
def test_ffn_forward_deterministic() -> None:
    """Test that FFN forward pass produces deterministic outputs."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 5, config.hidden_size)

    # Run forward pass twice
    output1 = ffn(x)
    output2 = ffn(x)

    # Outputs should be identical
    assert torch.equal(output1, output2)


@pytest.mark.unit
def test_ffn_forward_dtype_preservation() -> None:
    """Test that FFN forward pass preserves input dtype."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Test with float32
    x_f32 = torch.randn(2, 5, config.hidden_size, dtype=torch.float32)
    output_f32 = ffn(x_f32)

    assert output_f32.dtype == torch.float32


@pytest.mark.unit
def test_ffn_forward_with_zero_input() -> None:
    """Test FFN forward pass with zero input."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 5
    x = torch.zeros(batch_size, seq_len, config.hidden_size)

    # Forward pass with zero input
    output = ffn(x)

    # Output should be valid (not NaN or Inf)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_ffn_forward_with_positive_input() -> None:
    """Test FFN forward pass with positive input values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 5
    x = torch.ones(batch_size, seq_len, config.hidden_size) * 2.0

    # Forward pass
    output = ffn(x)

    # Output should be valid
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_ffn_forward_with_negative_input() -> None:
    """Test FFN forward pass with negative input values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 5
    x = torch.ones(batch_size, seq_len, config.hidden_size) * -2.0

    # Forward pass
    output = ffn(x)

    # Output should be valid
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_ffn_forward_numerical_stability() -> None:
    """Test FFN forward pass numerical stability with extreme values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Test with large positive values
    x_large = torch.ones(1, 1, config.hidden_size) * 10.0
    output_large = ffn(x_large)

    assert not torch.isnan(output_large).any()
    assert not torch.isinf(output_large).any()

    # Test with large negative values
    x_small = torch.ones(1, 1, config.hidden_size) * -10.0
    output_small = ffn(x_small)

    assert not torch.isnan(output_small).any()
    assert not torch.isinf(output_small).any()


@pytest.mark.unit
def test_ffn_forward_batch_independence() -> None:
    """Test that FFN processes each batch element independently."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Create two identical inputs in a batch
    x_single = torch.randn(1, 5, config.hidden_size)
    x_batch = torch.cat([x_single, x_single], dim=0)

    # Forward pass
    output_single = ffn(x_single)
    output_batch = ffn(x_batch)

    # Both batch elements should match the single computation
    assert torch.allclose(output_batch[0], output_single[0], rtol=1e-5, atol=1e-7)
    assert torch.allclose(output_batch[1], output_single[0], rtol=1e-5, atol=1e-7)


@pytest.mark.unit
def test_ffn_forward_sequence_independence() -> None:
    """Test that FFN processes each sequence position independently."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Create input with identical sequence positions
    x_single_pos = torch.randn(2, 1, config.hidden_size)
    x_multi_pos = x_single_pos.repeat(1, 3, 1)

    # Forward pass
    output_single = ffn(x_single_pos)
    output_multi = ffn(x_multi_pos)

    # All sequence positions should produce the same output
    for i in range(3):
        assert torch.allclose(output_multi[:, i, :], output_single[:, 0, :], rtol=1e-5, atol=1e-7)


@pytest.mark.unit
def test_ffn_forward_with_different_configs() -> None:
    """Test FFN forward pass with various model configurations."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    configs = [
        # Qwen3-0.6B-like config
        Qwen3Config(hidden_size=896, intermediate_size=4864, num_attention_heads=14, num_key_value_heads=2),
        # Smaller test config
        Qwen3Config(hidden_size=512, intermediate_size=2048, num_attention_heads=8, num_key_value_heads=2),
        # Larger config
        Qwen3Config(hidden_size=2048, intermediate_size=8192, num_attention_heads=32, num_key_value_heads=8),
    ]

    for config in configs:
        ffn = Qwen3FFN(config)

        x = torch.randn(2, 5, config.hidden_size)
        output = ffn(x)

        # Verify output shape
        assert output.shape == (2, 5, config.hidden_size)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.unit
def test_ffn_forward_eval_mode() -> None:
    """Test FFN forward pass in evaluation mode."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)
    ffn.eval()

    x = torch.randn(2, 5, config.hidden_size)

    # Forward pass in eval mode
    with torch.no_grad():
        output = ffn(x)

    assert output.shape == (2, 5, config.hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.unit
def test_ffn_forward_train_mode() -> None:
    """Test FFN forward pass in training mode."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)
    ffn.train()

    x = torch.randn(2, 5, config.hidden_size)

    # Forward pass in train mode
    output = ffn(x)

    assert output.shape == (2, 5, config.hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.unit
def test_ffn_forward_uses_swiglu_method() -> None:
    """Test that forward pass uses the swiglu method internally."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 5, config.hidden_size)

    # Forward pass
    output = ffn(x)

    # Manual computation using swiglu method
    swiglu_out = ffn.swiglu(x)
    expected_output = ffn.down_proj(swiglu_out)

    # Outputs should match
    assert torch.allclose(output, expected_output, rtol=1e-5, atol=1e-7)


@pytest.mark.unit
def test_ffn_forward_intermediate_dimension() -> None:
    """Test that FFN forward pass correctly uses intermediate dimension."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 5, config.hidden_size)

    # Forward pass
    output = ffn(x)

    # Verify that intermediate computation happens
    # (we can't directly observe it, but we can verify the output shape)
    assert output.shape == (2, 5, config.hidden_size)

    # Manually compute to verify intermediate dimension is used
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)

    # These should have intermediate_size dimension
    assert gate_out.shape == (2, 5, config.intermediate_size)
    assert up_out.shape == (2, 5, config.intermediate_size)


@pytest.mark.unit
def test_ffn_forward_parameter_updates() -> None:
    """Test that FFN parameters can be updated through forward pass."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Store initial weights
    initial_gate_weight = ffn.gate_proj.weight.data.clone()
    initial_up_weight = ffn.up_proj.weight.data.clone()
    initial_down_weight = ffn.down_proj.weight.data.clone()

    # Forward and backward pass
    x = torch.randn(2, 5, config.hidden_size, requires_grad=True)
    output = ffn(x)
    loss = output.sum()
    loss.backward()

    # Update parameters (simulate optimizer step)
    with torch.no_grad():
        ffn.gate_proj.weight -= 0.01 * ffn.gate_proj.weight.grad
        ffn.up_proj.weight -= 0.01 * ffn.up_proj.weight.grad
        ffn.down_proj.weight -= 0.01 * ffn.down_proj.weight.grad

    # Weights should have changed
    assert not torch.equal(ffn.gate_proj.weight, initial_gate_weight)
    assert not torch.equal(ffn.up_proj.weight, initial_up_weight)
    assert not torch.equal(ffn.down_proj.weight, initial_down_weight)


@pytest.mark.unit
def test_ffn_forward_output_range() -> None:
    """Test that FFN forward pass produces reasonable output values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 5, config.hidden_size)

    # Forward pass
    output = ffn(x)

    # Output should have reasonable magnitude (not too large or too small)
    # This is a sanity check, not a strict requirement
    assert output.abs().max() < 1000.0  # Not exploding
    assert output.abs().mean() > 1e-6   # Not vanishing


@pytest.mark.unit
def test_ffn_forward_single_token() -> None:
    """Test FFN forward pass with single token (seq_len=1)."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 1
    seq_len = 1
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Forward pass
    output = ffn(x)

    assert output.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


@pytest.mark.unit
def test_ffn_forward_consistency_across_calls() -> None:
    """Test that multiple forward passes with same input produce same output."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)
    ffn.eval()

    x = torch.randn(2, 5, config.hidden_size)

    # Multiple forward passes
    outputs = [ffn(x) for _ in range(5)]

    # All outputs should be identical
    for i in range(1, len(outputs)):
        assert torch.equal(outputs[0], outputs[i])

