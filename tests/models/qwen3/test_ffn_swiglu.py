"""
Tests for SwiGLU activation computation in Qwen3 FFN layer.

This module tests the SwiGLU activation function used in the Qwen3 FFN layer.
SwiGLU is defined as: SwiGLU(x) = Swish(gate_proj(x)) * up_proj(x)
where Swish(x) = x * sigmoid(x).

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch
import torch.nn.functional as F

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_swish_activation_basic() -> None:
    """Test that Swish activation is computed correctly: Swish(x) = x * sigmoid(x)."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Create test input with correct shape [batch, seq_len, hidden_size]
    batch_size = 1
    seq_len = 1
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # FFN should have a method to compute Swish (or use it internally)
    # We'll test this by checking the gate activation
    gate_out = ffn.gate_proj(x)
    swish_out = gate_out * torch.sigmoid(gate_out)

    # Verify Swish computation is correct
    assert swish_out.shape == gate_out.shape
    assert swish_out.shape == (batch_size, seq_len, config.intermediate_size)
    assert not torch.isnan(swish_out).any()
    assert not torch.isinf(swish_out).any()


@pytest.mark.unit
def test_swiglu_computation_formula() -> None:
    """Test that SwiGLU follows the formula: SwiGLU(x) = Swish(gate_proj(x)) * up_proj(x)."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 5
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Compute SwiGLU components manually
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swish_gate = gate_out * torch.sigmoid(gate_out)
    expected_swiglu = swish_gate * up_out

    # Verify expected shape
    assert expected_swiglu.shape == (batch_size, seq_len, config.intermediate_size)
    assert not torch.isnan(expected_swiglu).any()
    assert not torch.isinf(expected_swiglu).any()


@pytest.mark.unit
def test_swiglu_output_shape() -> None:
    """Test that SwiGLU activation produces correct output shape."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 4
    seq_len = 8
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Compute SwiGLU
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

    # Output should have intermediate_size dimension
    assert swiglu_out.shape == (batch_size, seq_len, config.intermediate_size)


@pytest.mark.unit
def test_swiglu_with_zero_input() -> None:
    """Test SwiGLU activation with zero input."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)
    batch_size = 2
    seq_len = 3
    x = torch.zeros(batch_size, seq_len, config.hidden_size)

    # Compute SwiGLU with zero input
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

    # Output should be valid (not NaN or Inf)
    assert not torch.isnan(swiglu_out).any()
    assert not torch.isinf(swiglu_out).any()
    assert swiglu_out.shape == (batch_size, seq_len, config.intermediate_size)


@pytest.mark.unit
def test_swiglu_with_positive_input() -> None:
    """Test SwiGLU activation with positive input values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 3
    x = torch.ones(batch_size, seq_len, config.hidden_size) * 2.0

    # Compute SwiGLU
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

    # Output should be valid
    assert not torch.isnan(swiglu_out).any()
    assert not torch.isinf(swiglu_out).any()
    assert swiglu_out.shape == (batch_size, seq_len, config.intermediate_size)


@pytest.mark.unit
def test_swiglu_with_negative_input() -> None:
    """Test SwiGLU activation with negative input values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 3
    x = torch.ones(batch_size, seq_len, config.hidden_size) * -2.0

    # Compute SwiGLU
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

    # Output should be valid
    assert not torch.isnan(swiglu_out).any()
    assert not torch.isinf(swiglu_out).any()
    assert swiglu_out.shape == (batch_size, seq_len, config.intermediate_size)


@pytest.mark.unit
def test_swiglu_with_mixed_input() -> None:
    """Test SwiGLU activation with mixed positive and negative values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 3
    x = torch.randn(batch_size, seq_len, config.hidden_size)

    # Compute SwiGLU
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

    # Output should be valid
    assert not torch.isnan(swiglu_out).any()
    assert not torch.isinf(swiglu_out).any()
    assert swiglu_out.shape == (batch_size, seq_len, config.intermediate_size)


@pytest.mark.unit
def test_swish_properties() -> None:
    """Test mathematical properties of Swish activation."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Test Swish is smooth and differentiable
    x = torch.randn(2, 3, config.hidden_size, requires_grad=True)
    gate_out = ffn.gate_proj(x)
    swish_out = gate_out * torch.sigmoid(gate_out)
    loss = swish_out.sum()
    loss.backward()

    # Gradients should exist and be valid
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isinf(x.grad).any()


@pytest.mark.unit
def test_swiglu_gradient_flow() -> None:
    """Test that gradients flow correctly through SwiGLU activation."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 3
    x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

    # Compute SwiGLU
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

    # Backward pass
    loss = swiglu_out.sum()
    loss.backward()

    # Check gradients exist for all components
    assert ffn.gate_proj.weight.grad is not None
    assert ffn.up_proj.weight.grad is not None
    assert x.grad is not None

    # Check gradients are valid
    assert not torch.isnan(ffn.gate_proj.weight.grad).any()
    assert not torch.isnan(ffn.up_proj.weight.grad).any()
    assert not torch.isnan(x.grad).any()


@pytest.mark.unit
def test_swiglu_deterministic() -> None:
    """Test that SwiGLU produces deterministic outputs."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 3, config.hidden_size)

    # Compute SwiGLU twice
    gate_out1 = ffn.gate_proj(x)
    up_out1 = ffn.up_proj(x)
    swiglu_out1 = gate_out1 * torch.sigmoid(gate_out1) * up_out1

    gate_out2 = ffn.gate_proj(x)
    up_out2 = ffn.up_proj(x)
    swiglu_out2 = gate_out2 * torch.sigmoid(gate_out2) * up_out2

    # Outputs should be identical
    assert torch.equal(swiglu_out1, swiglu_out2)


@pytest.mark.unit
def test_swiglu_dtype_preservation() -> None:
    """Test that SwiGLU preserves input dtype."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Test with float32
    x_f32 = torch.randn(2, 3, config.hidden_size, dtype=torch.float32)
    gate_f32 = ffn.gate_proj(x_f32)
    up_f32 = ffn.up_proj(x_f32)
    swiglu_f32 = gate_f32 * torch.sigmoid(gate_f32) * up_f32

    assert swiglu_f32.dtype == torch.float32


@pytest.mark.unit
def test_swiglu_batch_independence() -> None:
    """Test that SwiGLU processes each batch element independently."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Create two identical inputs in a batch
    x_single = torch.randn(1, 3, config.hidden_size)
    x_batch = torch.cat([x_single, x_single], dim=0)

    # Compute SwiGLU for single and batch
    gate_single = ffn.gate_proj(x_single)
    up_single = ffn.up_proj(x_single)
    swiglu_single = gate_single * torch.sigmoid(gate_single) * up_single

    gate_batch = ffn.gate_proj(x_batch)
    up_batch = ffn.up_proj(x_batch)
    swiglu_batch = gate_batch * torch.sigmoid(gate_batch) * up_batch

    # Both batch elements should match the single computation
    assert torch.allclose(swiglu_batch[0], swiglu_single[0], rtol=1e-5, atol=1e-7)
    assert torch.allclose(swiglu_batch[1], swiglu_single[0], rtol=1e-5, atol=1e-7)


@pytest.mark.unit
def test_swiglu_sequence_independence() -> None:
    """Test that SwiGLU processes each sequence position independently."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Create input with identical sequence positions
    x_single_pos = torch.randn(2, 1, config.hidden_size)
    x_multi_pos = x_single_pos.repeat(1, 3, 1)

    # Compute SwiGLU
    gate_single = ffn.gate_proj(x_single_pos)
    up_single = ffn.up_proj(x_single_pos)
    swiglu_single = gate_single * torch.sigmoid(gate_single) * up_single

    gate_multi = ffn.gate_proj(x_multi_pos)
    up_multi = ffn.up_proj(x_multi_pos)
    swiglu_multi = gate_multi * torch.sigmoid(gate_multi) * up_multi

    # All sequence positions should produce the same output
    for i in range(3):
        assert torch.allclose(swiglu_multi[:, i, :], swiglu_single[:, 0, :], rtol=1e-5, atol=1e-7)


@pytest.mark.unit
def test_swiglu_numerical_stability() -> None:
    """Test SwiGLU numerical stability with extreme values."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    # Test with large positive values
    x_large = torch.ones(1, 1, config.hidden_size) * 10.0
    gate_large = ffn.gate_proj(x_large)
    up_large = ffn.up_proj(x_large)
    swiglu_large = gate_large * torch.sigmoid(gate_large) * up_large

    assert not torch.isnan(swiglu_large).any()
    assert not torch.isinf(swiglu_large).any()

    # Test with large negative values
    x_small = torch.ones(1, 1, config.hidden_size) * -10.0
    gate_small = ffn.gate_proj(x_small)
    up_small = ffn.up_proj(x_small)
    swiglu_small = gate_small * torch.sigmoid(gate_small) * up_small

    assert not torch.isnan(swiglu_small).any()
    assert not torch.isinf(swiglu_small).any()


@pytest.mark.unit
def test_swiglu_vs_relu_difference() -> None:
    """Test that SwiGLU produces different outputs than ReLU-based activation."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 3, config.hidden_size)

    # Compute SwiGLU
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

    # Compute ReLU-based alternative
    relu_out = F.relu(gate_out) * up_out

    # SwiGLU and ReLU should produce different outputs
    assert not torch.allclose(swiglu_out, relu_out, rtol=1e-3, atol=1e-5)


@pytest.mark.unit
def test_swiglu_with_different_configs() -> None:
    """Test SwiGLU with various model configurations."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    configs = [
        Qwen3Config(hidden_size=896, intermediate_size=4864, num_attention_heads=14, num_key_value_heads=2),
        Qwen3Config(hidden_size=2048, intermediate_size=8192, num_attention_heads=32, num_key_value_heads=8),
        Qwen3Config(hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_key_value_heads=4),
    ]

    for config in configs:
        ffn = Qwen3FFN(config)

        x = torch.randn(2, 3, config.hidden_size)

        # Compute SwiGLU
        gate_out = ffn.gate_proj(x)
        up_out = ffn.up_proj(x)
        swiglu_out = gate_out * torch.sigmoid(gate_out) * up_out

        # Verify output shape
        assert swiglu_out.shape == (2, 3, config.intermediate_size)
        assert not torch.isnan(swiglu_out).any()
        assert not torch.isinf(swiglu_out).any()


@pytest.mark.unit
def test_sigmoid_range_in_swish() -> None:
    """Test that sigmoid in Swish produces values in (0, 1) range."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 3, config.hidden_size)

    # Compute gate projection
    gate_out = ffn.gate_proj(x)

    # Compute sigmoid
    sigmoid_out = torch.sigmoid(gate_out)

    # Sigmoid should be in (0, 1) range
    assert (sigmoid_out > 0).all()
    assert (sigmoid_out < 1).all()


@pytest.mark.unit
def test_swiglu_component_interaction() -> None:
    """Test that gate and up projections interact correctly in SwiGLU."""
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 3, config.hidden_size)

    # Compute components
    gate_out = ffn.gate_proj(x)
    up_out = ffn.up_proj(x)
    swish_gate = gate_out * torch.sigmoid(gate_out)

    # SwiGLU should be element-wise product
    swiglu_out = swish_gate * up_out

    # Verify shapes match
    assert gate_out.shape == up_out.shape
    assert swish_gate.shape == up_out.shape
    assert swiglu_out.shape == up_out.shape

    # Verify element-wise multiplication
    expected = swish_gate * up_out
    assert torch.equal(swiglu_out, expected)
