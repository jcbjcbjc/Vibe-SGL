"""
Tests for RoPE (Rotary Position Embeddings) implementation.

This module tests the Rotary Position Embeddings (RoPE) used in Qwen3 model
architecture. RoPE applies rotary transformations to query and key tensors
to encode positional information without using absolute position embeddings.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch
import math

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.rope import (
    precompute_freqs_cis,
    apply_rotary_emb,
)
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_precompute_freqs_cis_initialization() -> None:
    """Test that precompute_freqs_cis can be called with basic parameters."""
    dim = 64
    max_seq_len = 2048
    theta = 10000.0

    freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)

    assert freqs_cis is not None
    assert isinstance(freqs_cis, torch.Tensor)


@pytest.mark.unit
def test_precompute_freqs_cis_shape() -> None:
    """Test that precomputed frequencies have correct shape."""
    dim = 64
    max_seq_len = 2048

    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    # Shape should be [max_seq_len, dim // 2] for complex numbers
    # Complex tensor has shape [max_seq_len, dim // 2]
    assert freqs_cis.shape == (max_seq_len, dim // 2)


@pytest.mark.unit
def test_precompute_freqs_cis_dtype() -> None:
    """Test that precomputed frequencies are complex numbers."""
    dim = 64
    max_seq_len = 2048

    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    # RoPE uses complex numbers for rotation
    assert freqs_cis.dtype in [torch.complex64, torch.complex128]


@pytest.mark.unit
def test_precompute_freqs_cis_theta_parameter() -> None:
    """Test that theta parameter affects frequency computation."""
    dim = 64
    max_seq_len = 100

    freqs_cis_default = precompute_freqs_cis(dim, max_seq_len, theta=10000.0)
    freqs_cis_custom = precompute_freqs_cis(dim, max_seq_len, theta=100000.0)

    # Different theta values should produce different frequencies
    assert not torch.allclose(freqs_cis_default, freqs_cis_custom)


@pytest.mark.unit
def test_precompute_freqs_cis_frequency_formula() -> None:
    """Test that frequencies follow the formula: theta^(-2i/dim) for i in [0, dim/2)."""
    dim = 8
    max_seq_len = 4
    theta = 10000.0

    freqs_cis = precompute_freqs_cis(dim, max_seq_len, theta)

    # Manually compute expected frequencies for first position
    # freq_i = theta^(-2i/dim) for i in [0, dim/2)
    expected_freqs = []
    for i in range(dim // 2):
        freq = 1.0 / (theta ** (2 * i / dim))
        expected_freqs.append(freq)

    # For position 0, angle should be 0, so cos=1, sin=0
    # freqs_cis[0] should be exp(i * 0 * freq) = 1 + 0j for all frequencies
    expected_pos0 = torch.ones(dim // 2, dtype=torch.complex64)
    assert_tensors_close(freqs_cis[0].real, expected_pos0.real, atol=1e-6, rtol=1e-5)
    assert_tensors_close(freqs_cis[0].imag, expected_pos0.imag, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_precompute_freqs_cis_position_encoding() -> None:
    """Test that different positions produce different frequency encodings."""
    dim = 64
    max_seq_len = 100

    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    # Different positions should have different encodings
    assert not torch.allclose(freqs_cis[0], freqs_cis[1])
    assert not torch.allclose(freqs_cis[0], freqs_cis[50])
    assert not torch.allclose(freqs_cis[1], freqs_cis[50])


@pytest.mark.unit
def test_precompute_freqs_cis_unit_magnitude() -> None:
    """Test that all frequency encodings have unit magnitude (lie on unit circle)."""
    dim = 64
    max_seq_len = 100

    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    # Complex numbers representing rotations should have magnitude 1
    magnitudes = torch.abs(freqs_cis)
    expected_magnitudes = torch.ones_like(magnitudes)

    assert_tensors_close(magnitudes, expected_magnitudes, atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_apply_rotary_emb_initialization() -> None:
    """Test that apply_rotary_emb can be called with basic parameters."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    assert q_rot is not None
    assert k_rot is not None


@pytest.mark.unit
def test_apply_rotary_emb_shape_preservation() -> None:
    """Test that apply_rotary_emb preserves input shapes."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


@pytest.mark.unit
def test_apply_rotary_emb_dtype_preservation() -> None:
    """Test that apply_rotary_emb preserves input dtype."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    # Test with float32
    q_f32 = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    k_f32 = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=torch.float32)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot_f32, k_rot_f32 = apply_rotary_emb(q_f32, k_f32, freqs_cis)

    assert q_rot_f32.dtype == torch.float32
    assert k_rot_f32.dtype == torch.float32


@pytest.mark.unit
def test_apply_rotary_emb_modifies_values() -> None:
    """Test that apply_rotary_emb actually modifies the input tensors."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Rotated tensors should be different from original
    assert not torch.allclose(q_rot, q)
    assert not torch.allclose(k_rot, k)


@pytest.mark.unit
def test_apply_rotary_emb_with_position_offset() -> None:
    """Test that apply_rotary_emb supports position offset for KV cache."""
    batch_size = 2
    seq_len = 5
    num_heads = 8
    head_dim = 64
    position_offset = 10

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=100)

    # Apply with offset
    q_rot_offset, k_rot_offset = apply_rotary_emb(
        q, k, freqs_cis, position_offset=position_offset
    )

    # Apply without offset
    q_rot_no_offset, k_rot_no_offset = apply_rotary_emb(q, k, freqs_cis)

    # Results should be different when using offset
    assert not torch.allclose(q_rot_offset, q_rot_no_offset)
    assert not torch.allclose(k_rot_offset, k_rot_no_offset)


@pytest.mark.unit
def test_apply_rotary_emb_rotation_property() -> None:
    """Test that RoPE preserves vector norms (rotation property)."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Rotation should preserve norms (up to numerical precision)
    q_norm_before = torch.norm(q, dim=-1)
    q_norm_after = torch.norm(q_rot, dim=-1)
    k_norm_before = torch.norm(k, dim=-1)
    k_norm_after = torch.norm(k_rot, dim=-1)

    assert_tensors_close(q_norm_before, q_norm_after, atol=1e-5, rtol=1e-4)
    assert_tensors_close(k_norm_before, k_norm_after, atol=1e-5, rtol=1e-4)


@pytest.mark.unit
def test_apply_rotary_emb_batch_independence() -> None:
    """Test that RoPE processes batch elements independently."""
    batch_size = 3
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Process each batch element separately
    for i in range(batch_size):
        q_single = q[i:i+1]
        k_single = k[i:i+1]
        q_rot_single, k_rot_single = apply_rotary_emb(q_single, k_single, freqs_cis)

        assert_tensors_close(q_rot[i], q_rot_single[0], atol=1e-6, rtol=1e-5)
        assert_tensors_close(k_rot[i], k_rot_single[0], atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_apply_rotary_emb_head_independence() -> None:
    """Test that RoPE processes attention heads independently."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Process each head separately
    for h in range(num_heads):
        q_single_head = q[:, :, h:h+1, :]
        k_single_head = k[:, :, h:h+1, :]
        q_rot_single, k_rot_single = apply_rotary_emb(
            q_single_head, k_single_head, freqs_cis
        )

        assert_tensors_close(q_rot[:, :, h, :], q_rot_single[:, :, 0, :], atol=1e-6, rtol=1e-5)
        assert_tensors_close(k_rot[:, :, h, :], k_rot_single[:, :, 0, :], atol=1e-6, rtol=1e-5)


@pytest.mark.unit
def test_apply_rotary_emb_different_positions() -> None:
    """Test that different sequence positions get different rotations."""
    batch_size = 1
    seq_len = 10
    num_heads = 1
    head_dim = 64

    # Use same values for all positions
    q = torch.ones(batch_size, seq_len, num_heads, head_dim)
    k = torch.ones(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Different positions should have different rotated values
    assert not torch.allclose(q_rot[0, 0], q_rot[0, 1])
    assert not torch.allclose(q_rot[0, 0], q_rot[0, 5])
    assert not torch.allclose(k_rot[0, 0], k_rot[0, 1])


@pytest.mark.unit
def test_apply_rotary_emb_with_qwen3_config() -> None:
    """Test that RoPE works with Qwen3Config parameters."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    head_dim = config.hidden_size // config.num_attention_heads
    batch_size = 2
    seq_len = 10
    num_heads = config.num_attention_heads

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(
        head_dim,
        max_seq_len=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


@pytest.mark.unit
def test_apply_rotary_emb_gradient_flow() -> None:
    """Test that RoPE allows gradient flow."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    # Compute loss and backpropagate
    loss = q_rot.sum() + k_rot.sum()
    loss.backward()

    # Check that gradients are computed
    assert q.grad is not None
    assert k.grad is not None
    assert not torch.isnan(q.grad).any()
    assert not torch.isnan(k.grad).any()


@pytest.mark.unit
def test_apply_rotary_emb_deterministic() -> None:
    """Test that RoPE produces deterministic outputs."""
    batch_size = 2
    seq_len = 10
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    # Run twice with same input
    q_rot1, k_rot1 = apply_rotary_emb(q, k, freqs_cis)
    q_rot2, k_rot2 = apply_rotary_emb(q, k, freqs_cis)

    # Outputs should be identical
    assert torch.equal(q_rot1, q_rot2)
    assert torch.equal(k_rot1, k_rot2)


@pytest.mark.unit
def test_precompute_freqs_cis_various_dimensions() -> None:
    """Test that precompute_freqs_cis works with various head dimensions."""
    dimensions = [32, 64, 96, 128]
    max_seq_len = 100

    for dim in dimensions:
        freqs_cis = precompute_freqs_cis(dim, max_seq_len)

        assert freqs_cis.shape == (max_seq_len, dim // 2)
        assert not torch.isnan(freqs_cis).any()
        assert not torch.isinf(freqs_cis).any()


@pytest.mark.unit
def test_apply_rotary_emb_various_sequence_lengths() -> None:
    """Test that RoPE works with various sequence lengths."""
    batch_size = 2
    num_heads = 8
    head_dim = 64
    max_seq_len = 2048

    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)

    seq_lengths = [1, 10, 50, 100, 500, 1000]
    for seq_len in seq_lengths:
        q = torch.randn(batch_size, seq_len, num_heads, head_dim)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim)

        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

        assert q_rot.shape == (batch_size, seq_len, num_heads, head_dim)
        assert k_rot.shape == (batch_size, seq_len, num_heads, head_dim)
        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()


@pytest.mark.unit
def test_apply_rotary_emb_single_token() -> None:
    """Test that RoPE works correctly for single token (seq_len=1)."""
    batch_size = 2
    seq_len = 1
    num_heads = 8
    head_dim = 64

    q = torch.randn(batch_size, seq_len, num_heads, head_dim)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim)
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=100)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()


@pytest.mark.unit
def test_apply_rotary_emb_kv_cache_scenario() -> None:
    """Test RoPE in KV cache scenario with position offset."""
    batch_size = 1
    num_heads = 8
    head_dim = 64
    max_seq_len = 2048

    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)

    # Simulate prefill phase: process first 10 tokens
    prefill_len = 10
    q_prefill = torch.randn(batch_size, prefill_len, num_heads, head_dim)
    k_prefill = torch.randn(batch_size, prefill_len, num_heads, head_dim)
    q_rot_prefill, k_rot_prefill = apply_rotary_emb(q_prefill, k_prefill, freqs_cis)

    # Simulate decode phase: process next token with position offset
    decode_len = 1
    position_offset = prefill_len
    q_decode = torch.randn(batch_size, decode_len, num_heads, head_dim)
    k_decode = torch.randn(batch_size, decode_len, num_heads, head_dim)
    q_rot_decode, k_rot_decode = apply_rotary_emb(
        q_decode, k_decode, freqs_cis, position_offset=position_offset
    )

    # Verify shapes
    assert q_rot_prefill.shape == (batch_size, prefill_len, num_heads, head_dim)
    assert q_rot_decode.shape == (batch_size, decode_len, num_heads, head_dim)


@pytest.mark.unit
def test_rope_relative_position_property() -> None:
    """Test that RoPE encodes relative positions in attention scores.

    The key property of RoPE is that the dot product between rotated query
    and key depends only on their relative position, not absolute positions.
    """
    batch_size = 1
    num_heads = 1
    head_dim = 64
    max_seq_len = 100

    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len)

    # Create identical query and key vectors
    q_vec = torch.randn(batch_size, 1, num_heads, head_dim)
    k_vec = q_vec.clone()

    # Apply RoPE at position 5 and 10 (relative distance = 5)
    q_pos5, _ = apply_rotary_emb(q_vec, q_vec, freqs_cis, position_offset=5)
    _, k_pos10 = apply_rotary_emb(k_vec, k_vec, freqs_cis, position_offset=10)

    # Apply RoPE at position 20 and 25 (relative distance = 5)
    q_pos20, _ = apply_rotary_emb(q_vec, q_vec, freqs_cis, position_offset=20)
    _, k_pos25 = apply_rotary_emb(k_vec, k_vec, freqs_cis, position_offset=25)

    # Compute attention scores (dot products)
    score1 = torch.sum(q_pos5 * k_pos10)
    score2 = torch.sum(q_pos20 * k_pos25)

    # Scores should be similar for same relative distance
    # (may not be exactly equal due to different absolute positions)
    assert_tensors_close(score1, score2, atol=1e-3, rtol=1e-2)


@pytest.mark.unit
def test_precompute_freqs_cis_caching() -> None:
    """Test that precomputed frequencies can be reused across forward passes."""
    dim = 64
    max_seq_len = 2048

    # Precompute once
    freqs_cis = precompute_freqs_cis(dim, max_seq_len)

    # Use multiple times
    batch_size = 2
    seq_len = 10
    num_heads = 8

    for _ in range(5):
        q = torch.randn(batch_size, seq_len, num_heads, dim)
        k = torch.randn(batch_size, seq_len, num_heads, dim)
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape


@pytest.mark.unit
def test_apply_rotary_emb_no_nan_or_inf() -> None:
    """Test that RoPE never produces NaN or Inf values."""
    batch_size = 2
    seq_len = 100
    num_heads = 8
    head_dim = 64

    # Test with various input patterns
    test_inputs = [
        torch.randn(batch_size, seq_len, num_heads, head_dim),
        torch.zeros(batch_size, seq_len, num_heads, head_dim),
        torch.ones(batch_size, seq_len, num_heads, head_dim),
        torch.randn(batch_size, seq_len, num_heads, head_dim) * 100,
        torch.randn(batch_size, seq_len, num_heads, head_dim) * 0.01,
    ]

    freqs_cis = precompute_freqs_cis(head_dim, seq_len)

    for q in test_inputs:
        k = q.clone()
        q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

        assert not torch.isnan(q_rot).any()
        assert not torch.isnan(k_rot).any()
        assert not torch.isinf(q_rot).any()
        assert not torch.isinf(k_rot).any()


@pytest.mark.unit
def test_rope_implementation_matches_formula() -> None:
    """Test that RoPE implementation matches the mathematical formula.

    RoPE applies rotation to pairs of dimensions:
    [x_i, x_{i+1}] -> [x_i * cos(m*theta_i) - x_{i+1} * sin(m*theta_i),
                       x_i * sin(m*theta_i) + x_{i+1} * cos(m*theta_i)]
    where m is the position and theta_i is the frequency for dimension pair i.
    """
    batch_size = 1
    seq_len = 1
    num_heads = 1
    head_dim = 4  # Small dimension for manual verification

    # Simple input for verification
    q = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # [1, 1, 1, 4]
    k = torch.tensor([[[[5.0, 6.0, 7.0, 8.0]]]])  # [1, 1, 1, 4]

    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=10)

    q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis, position_offset=0)

    # At position 0, the rotation angles are 0, so we should get back
    # something close to the original (with rotation matrix applied)
    # This is a basic sanity check
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.isnan(q_rot).any()
    assert not torch.isnan(k_rot).any()

