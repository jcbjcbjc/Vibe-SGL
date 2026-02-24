"""
Tests for FFN layer initialization in Qwen3.

This module tests the initialization and configuration of the Feed-Forward Network
(FFN) layer in Qwen3, including up_proj, gate_proj, and down_proj projections.
It validates correct weight shapes, SwiGLU activation setup, and TP (Tensor Parallelism)
awareness.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch
import torch.nn as nn

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_ffn_initialization() -> None:
    """Test that FFN layer can be initialized with correct projections."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    # Import will fail until implementation exists
    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # Check that all three projections exist
    assert hasattr(ffn, "up_proj")
    assert hasattr(ffn, "gate_proj")
    assert hasattr(ffn, "down_proj")
    assert isinstance(ffn.up_proj, nn.Linear)
    assert isinstance(ffn.gate_proj, nn.Linear)
    assert isinstance(ffn.down_proj, nn.Linear)


@pytest.mark.unit
def test_up_proj_shape() -> None:
    """Test that up_proj has correct input/output dimensions."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # up_proj: hidden_size -> intermediate_size
    assert ffn.up_proj.in_features == config.hidden_size
    assert ffn.up_proj.out_features == config.intermediate_size
    assert ffn.up_proj.weight.shape == (config.intermediate_size, config.hidden_size)


@pytest.mark.unit
def test_gate_proj_shape() -> None:
    """Test that gate_proj has correct input/output dimensions."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # gate_proj: hidden_size -> intermediate_size
    assert ffn.gate_proj.in_features == config.hidden_size
    assert ffn.gate_proj.out_features == config.intermediate_size
    assert ffn.gate_proj.weight.shape == (config.intermediate_size, config.hidden_size)


@pytest.mark.unit
def test_down_proj_shape() -> None:
    """Test that down_proj has correct input/output dimensions."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # down_proj: intermediate_size -> hidden_size
    assert ffn.down_proj.in_features == config.intermediate_size
    assert ffn.down_proj.out_features == config.hidden_size
    assert ffn.down_proj.weight.shape == (config.hidden_size, config.intermediate_size)


@pytest.mark.unit
def test_projection_bias_disabled() -> None:
    """Test that FFN projections have bias disabled (Qwen3 standard)."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # Qwen3 uses bias=False for FFN projections
    assert ffn.up_proj.bias is None
    assert ffn.gate_proj.bias is None
    assert ffn.down_proj.bias is None


@pytest.mark.unit
def test_projection_weights_initialized() -> None:
    """Test that projection weights are properly initialized."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # Weights should be initialized (not all zeros)
    assert not torch.allclose(ffn.up_proj.weight, torch.zeros_like(ffn.up_proj.weight))
    assert not torch.allclose(ffn.gate_proj.weight, torch.zeros_like(ffn.gate_proj.weight))
    assert not torch.allclose(ffn.down_proj.weight, torch.zeros_like(ffn.down_proj.weight))

    # Weights should have reasonable values (not NaN or Inf)
    assert not torch.isnan(ffn.up_proj.weight).any()
    assert not torch.isnan(ffn.gate_proj.weight).any()
    assert not torch.isnan(ffn.down_proj.weight).any()
    assert not torch.isinf(ffn.up_proj.weight).any()
    assert not torch.isinf(ffn.gate_proj.weight).any()
    assert not torch.isinf(ffn.down_proj.weight).any()


@pytest.mark.unit
def test_ffn_config_stored() -> None:
    """Test that FFN layer stores config reference."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # Config should be accessible for later use
    assert hasattr(ffn, "config")
    assert ffn.config == config


@pytest.mark.unit
def test_ffn_with_different_configs() -> None:
    """Test FFN projections with various model configurations."""
    configs = [
        # Qwen3-0.6B-like config
        Qwen3Config(hidden_size=896, intermediate_size=4864, num_attention_heads=14, num_key_value_heads=2),
        # Smaller test config
        Qwen3Config(hidden_size=512, intermediate_size=2048, num_attention_heads=8, num_key_value_heads=2),
        # Larger config
        Qwen3Config(hidden_size=2048, intermediate_size=8192, num_attention_heads=32, num_key_value_heads=8),
        # Different ratio
        Qwen3Config(hidden_size=768, intermediate_size=3072, num_attention_heads=12, num_key_value_heads=4),
    ]

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    for config in configs:
        ffn = Qwen3FFN(config)

        # Validate up_proj and gate_proj
        assert ffn.up_proj.in_features == config.hidden_size
        assert ffn.up_proj.out_features == config.intermediate_size
        assert ffn.gate_proj.in_features == config.hidden_size
        assert ffn.gate_proj.out_features == config.intermediate_size

        # Validate down_proj
        assert ffn.down_proj.in_features == config.intermediate_size
        assert ffn.down_proj.out_features == config.hidden_size


@pytest.mark.unit
def test_ffn_forward_pass_shapes() -> None:
    """Test that FFN projections produce correct output shapes."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 10
    hidden_size = config.hidden_size

    # Create input tensor
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Apply projections individually
    up_out = ffn.up_proj(x)
    gate_out = ffn.gate_proj(x)
    down_in = torch.randn(batch_size, seq_len, config.intermediate_size)
    down_out = ffn.down_proj(down_in)

    # Check output shapes
    assert up_out.shape == (batch_size, seq_len, config.intermediate_size)
    assert gate_out.shape == (batch_size, seq_len, config.intermediate_size)
    assert down_out.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_ffn_projections_gradient_flow() -> None:
    """Test that gradients flow through FFN projections."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)

    # Forward pass through all projections
    up_out = ffn.up_proj(x)
    gate_out = ffn.gate_proj(x)
    # Simulate intermediate computation
    intermediate = up_out + gate_out
    down_out = ffn.down_proj(intermediate)

    # Backward pass
    loss = down_out.sum()
    loss.backward()

    # Check gradients exist
    assert ffn.up_proj.weight.grad is not None
    assert ffn.gate_proj.weight.grad is not None
    assert ffn.down_proj.weight.grad is not None
    assert x.grad is not None

    # Check gradients are valid
    assert not torch.isnan(ffn.up_proj.weight.grad).any()
    assert not torch.isnan(ffn.gate_proj.weight.grad).any()
    assert not torch.isnan(ffn.down_proj.weight.grad).any()


@pytest.mark.unit
def test_ffn_projections_dtype_preservation() -> None:
    """Test that FFN projections preserve input dtype."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    # Test with float32
    ffn_f32 = Qwen3FFN(config)
    x_f32 = torch.randn(2, 10, config.hidden_size, dtype=torch.float32)
    up_f32 = ffn_f32.up_proj(x_f32)
    gate_f32 = ffn_f32.gate_proj(x_f32)

    intermediate_f32 = torch.randn(2, 10, config.intermediate_size, dtype=torch.float32)
    down_f32 = ffn_f32.down_proj(intermediate_f32)

    assert up_f32.dtype == torch.float32
    assert gate_f32.dtype == torch.float32
    assert down_f32.dtype == torch.float32


@pytest.mark.unit
def test_ffn_projections_deterministic() -> None:
    """Test that FFN projections produce deterministic outputs."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    x = torch.randn(2, 10, config.hidden_size)

    # Run twice with same input
    up1 = ffn.up_proj(x)
    gate1 = ffn.gate_proj(x)

    up2 = ffn.up_proj(x)
    gate2 = ffn.gate_proj(x)

    # Outputs should be identical
    assert torch.equal(up1, up2)
    assert torch.equal(gate1, gate2)


@pytest.mark.unit
def test_ffn_from_pretrained_config() -> None:
    """Test FFN with config loaded from HuggingFace."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # Validate projections match pretrained config
    assert ffn.up_proj.in_features == config.hidden_size
    assert ffn.up_proj.out_features == config.intermediate_size
    assert ffn.gate_proj.in_features == config.hidden_size
    assert ffn.gate_proj.out_features == config.intermediate_size
    assert ffn.down_proj.in_features == config.intermediate_size
    assert ffn.down_proj.out_features == config.hidden_size


@pytest.mark.unit
def test_ffn_projection_parameter_count() -> None:
    """Test that FFN projections have expected parameter count."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # up_proj and gate_proj parameters
    up_gate_params = config.hidden_size * config.intermediate_size
    assert ffn.up_proj.weight.numel() == up_gate_params
    assert ffn.gate_proj.weight.numel() == up_gate_params

    # down_proj parameters
    down_params = config.intermediate_size * config.hidden_size
    assert ffn.down_proj.weight.numel() == down_params


@pytest.mark.unit
def test_ffn_intermediate_size_validation() -> None:
    """Test that FFN validates intermediate_size from config."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # FFN should store intermediate_size
    assert hasattr(ffn, "intermediate_size")
    assert ffn.intermediate_size == config.intermediate_size


@pytest.mark.unit
def test_ffn_hidden_size_validation() -> None:
    """Test that FFN validates hidden_size from config."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # FFN should store hidden_size
    assert hasattr(ffn, "hidden_size")
    assert ffn.hidden_size == config.hidden_size


@pytest.mark.unit
def test_ffn_projections_independent() -> None:
    """Test that up_proj and gate_proj are independent layers."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # up_proj and gate_proj should have different weights
    assert not torch.equal(ffn.up_proj.weight, ffn.gate_proj.weight)

    # They should be separate nn.Linear instances
    assert ffn.up_proj is not ffn.gate_proj


@pytest.mark.unit
def test_ffn_swiglu_components_present() -> None:
    """Test that FFN has components needed for SwiGLU activation."""
    config = Qwen3Config(
        hidden_size=896,
        intermediate_size=4864,
    )

    from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN

    ffn = Qwen3FFN(config)

    # SwiGLU requires: gate_proj (for gating), up_proj (for values), down_proj (for output)
    # All three should exist and have correct dimensions
    assert hasattr(ffn, "gate_proj")
    assert hasattr(ffn, "up_proj")
    assert hasattr(ffn, "down_proj")

    # gate_proj and up_proj should have same output dimension
    assert ffn.gate_proj.out_features == ffn.up_proj.out_features
    # down_proj input should match gate/up output
    assert ffn.down_proj.in_features == ffn.gate_proj.out_features

