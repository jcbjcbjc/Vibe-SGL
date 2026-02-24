"""
Tests for RMSNorm layer implementation.

This module tests the RMSNorm (Root Mean Square Layer Normalization) layer
used in Qwen3 model architecture. RMSNorm is a simplified normalization
technique that normalizes using only the root mean square statistic.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.rmsnorm import RMSNorm
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_rmsnorm_initialization() -> None:
    """Test that RMSNorm can be initialized with hidden_size and eps."""
    hidden_size = 896
    eps = 1e-6

    norm = RMSNorm(hidden_size, eps=eps)

    assert norm is not None
    assert isinstance(norm, RMSNorm)
    assert hasattr(norm, "weight")
    assert hasattr(norm, "eps")


@pytest.mark.unit
def test_rmsnorm_weight_shape() -> None:
    """Test that RMSNorm weight parameter has correct shape."""
    hidden_size = 896

    norm = RMSNorm(hidden_size)

    assert norm.weight.shape == (hidden_size,)
    assert isinstance(norm.weight, torch.nn.Parameter)


@pytest.mark.unit
def test_rmsnorm_weight_initialized_to_ones() -> None:
    """Test that RMSNorm weight is initialized to ones."""
    hidden_size = 896

    norm = RMSNorm(hidden_size)

    # Weight should be initialized to ones (identity scaling)
    assert torch.allclose(norm.weight, torch.ones(hidden_size))


@pytest.mark.unit
def test_rmsnorm_eps_stored() -> None:
    """Test that RMSNorm stores epsilon value correctly."""
    hidden_size = 896
    eps = 1e-6

    norm = RMSNorm(hidden_size, eps=eps)

    assert norm.eps == eps


@pytest.mark.unit
def test_rmsnorm_forward_shape() -> None:
    """Test that RMSNorm forward pass preserves input shape."""
    hidden_size = 896
    batch_size = 2
    seq_len = 10

    norm = RMSNorm(hidden_size)
    x = torch.randn(batch_size, seq_len, hidden_size)

    output = norm(x)

    assert output.shape == x.shape


@pytest.mark.unit
def test_rmsnorm_computation_formula() -> None:
    """Test that RMSNorm computes: x * rsqrt(mean(x^2) + eps) * weight."""
    hidden_size = 4
    eps = 1e-6

    norm = RMSNorm(hidden_size, eps=eps)
    # Set weight to ones for easier verification
    norm.weight.data.fill_(1.0)

    # Simple input for manual verification
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape: [1, 4]

    output = norm(x)

    # Manual computation
    # variance = mean(x^2) = (1 + 4 + 9 + 16) / 4 = 7.5
    # rsqrt(variance + eps) = 1 / sqrt(7.5 + 1e-6) ≈ 0.3651
    # output = x * rsqrt(variance + eps) * weight
    variance = torch.mean(x * x, dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps)

    assert_tensors_close(output, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_rmsnorm_with_weight_scaling() -> None:
    """Test that RMSNorm applies weight scaling correctly."""
    hidden_size = 4
    eps = 1e-6

    norm = RMSNorm(hidden_size, eps=eps)
    # Set custom weight values
    norm.weight.data = torch.tensor([1.0, 2.0, 3.0, 4.0])

    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # shape: [1, 4]

    output = norm(x)

    # Manual computation with weight scaling
    variance = torch.mean(x * x, dim=-1, keepdim=True)
    expected = x * torch.rsqrt(variance + eps) * norm.weight

    assert_tensors_close(output, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_rmsnorm_batch_processing() -> None:
    """Test that RMSNorm processes batches independently."""
    hidden_size = 4
    batch_size = 3
    eps = 1e-6

    norm = RMSNorm(hidden_size, eps=eps)
    norm.weight.data.fill_(1.0)

    # Create batch with different values
    x = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [0.5, 1.0, 1.5, 2.0],
    ])  # shape: [3, 4]

    output = norm(x)

    # Each batch element should be normalized independently
    for i in range(batch_size):
        variance = torch.mean(x[i] * x[i], dim=-1, keepdim=True)
        expected = x[i] * torch.rsqrt(variance + eps)
        assert_tensors_close(output[i], expected, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_rmsnorm_sequence_processing() -> None:
    """Test that RMSNorm processes sequence positions independently."""
    hidden_size = 4
    batch_size = 2
    seq_len = 3
    eps = 1e-6

    norm = RMSNorm(hidden_size, eps=eps)
    norm.weight.data.fill_(1.0)

    x = torch.randn(batch_size, seq_len, hidden_size)

    output = norm(x)

    # Each position in sequence should be normalized independently
    for b in range(batch_size):
        for s in range(seq_len):
            variance = torch.mean(x[b, s] * x[b, s], dim=-1, keepdim=True)
            expected = x[b, s] * torch.rsqrt(variance + eps)
            assert_tensors_close(output[b, s], expected, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_rmsnorm_eps_prevents_division_by_zero() -> None:
    """Test that epsilon prevents division by zero for zero input."""
    hidden_size = 4
    eps = 1e-6

    norm = RMSNorm(hidden_size, eps=eps)
    norm.weight.data.fill_(1.0)

    # Zero input should not cause NaN or Inf
    x = torch.zeros(1, hidden_size)

    output = norm(x)

    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
    # Output should be zero (0 * rsqrt(eps) * weight = 0)
    assert torch.allclose(output, torch.zeros_like(output))


@pytest.mark.unit
def test_rmsnorm_with_config_eps() -> None:
    """Test that RMSNorm can use epsilon from Qwen3Config."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    assert norm.eps == config.rms_norm_eps
    assert norm.weight.shape == (config.hidden_size,)


@pytest.mark.unit
def test_rmsnorm_gradient_flow() -> None:
    """Test that RMSNorm allows gradient flow through weight parameter."""
    hidden_size = 4
    norm = RMSNorm(hidden_size)

    x = torch.randn(2, 3, hidden_size, requires_grad=True)
    output = norm(x)
    loss = output.sum()
    loss.backward()

    # Check that gradients are computed
    assert norm.weight.grad is not None
    assert x.grad is not None
    assert not torch.isnan(norm.weight.grad).any()
    assert not torch.isnan(x.grad).any()


@pytest.mark.unit
def test_rmsnorm_dtype_preservation() -> None:
    """Test that RMSNorm preserves input dtype."""
    hidden_size = 4

    # Test with float32
    norm_f32 = RMSNorm(hidden_size)
    x_f32 = torch.randn(2, 3, hidden_size, dtype=torch.float32)
    output_f32 = norm_f32(x_f32)
    assert output_f32.dtype == torch.float32

    # Test with float16
    norm_f16 = RMSNorm(hidden_size)
    norm_f16.weight.data = norm_f16.weight.data.half()
    x_f16 = torch.randn(2, 3, hidden_size, dtype=torch.float16)
    output_f16 = norm_f16(x_f16)
    assert output_f16.dtype == torch.float16


@pytest.mark.unit
def test_rmsnorm_deterministic() -> None:
    """Test that RMSNorm produces deterministic outputs."""
    hidden_size = 4
    norm = RMSNorm(hidden_size)

    x = torch.randn(2, 3, hidden_size)

    # Run twice with same input
    output1 = norm(x)
    output2 = norm(x)

    # Outputs should be identical
    assert torch.equal(output1, output2)


@pytest.mark.unit
def test_rmsnorm_different_hidden_sizes() -> None:
    """Test RMSNorm with various hidden sizes."""
    hidden_sizes = [64, 128, 256, 512, 896, 1024, 2048]

    for hidden_size in hidden_sizes:
        norm = RMSNorm(hidden_size)
        x = torch.randn(2, 3, hidden_size)
        output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.unit
def test_rmsnorm_normalization_effect() -> None:
    """Test that RMSNorm reduces variance to approximately 1."""
    hidden_size = 128
    batch_size = 10
    seq_len = 20

    norm = RMSNorm(hidden_size)
    norm.weight.data.fill_(1.0)

    # Input with high variance
    x = torch.randn(batch_size, seq_len, hidden_size) * 10.0

    output = norm(x)

    # Compute RMS (root mean square) of output
    # For normalized output, RMS should be close to 1
    rms = torch.sqrt(torch.mean(output * output, dim=-1))

    # RMS should be approximately 1 (within tolerance)
    assert torch.allclose(rms, torch.ones_like(rms), atol=0.1, rtol=0.1)
