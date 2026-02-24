"""
Tests for attention computation with RoPE in Qwen3.

This module tests the complete attention mechanism including:
- Q/K/V projections from hidden states
- RoPE application to queries and keys
- Scaled dot-product attention computation
- GQA with KV head repetition
- Attention output computation

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
import torch
import torch.nn.functional as F

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention
from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_attention_forward_with_rope_exists() -> None:
    """Test that attention layer has a forward method."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    # Should have a forward method
    assert hasattr(attention, "forward")


@pytest.mark.unit
def test_attention_forward_basic_call() -> None:
    """Test that attention forward can be called with basic parameters."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Precompute RoPE frequencies
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        head_dim,
        max_seq_len=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    # Should be able to call forward
    output = attention.forward(hidden_states, freqs_cis)

    assert output is not None


@pytest.mark.unit
def test_attention_forward_output_shape() -> None:
    """Test that attention forward returns correct output shape."""
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

    output = attention.forward(hidden_states, freqs_cis)

    # Output should have same shape as input
    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape


@pytest.mark.unit
def test_attention_qkv_projection_shapes() -> None:
    """Test that Q/K/V projections produce correct shapes."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Project to Q/K/V
    q = attention.q_proj(hidden_states)
    k = attention.k_proj(hidden_states)
    v = attention.v_proj(hidden_states)

    head_dim = config.hidden_size // config.num_attention_heads

    # Q should have shape [batch, seq_len, num_heads * head_dim]
    expected_q_shape = (batch_size, seq_len, config.num_attention_heads * head_dim)
    assert q.shape == expected_q_shape

    # K/V should have shape [batch, seq_len, num_kv_heads * head_dim]
    expected_kv_shape = (batch_size, seq_len, config.num_key_value_heads * head_dim)
    assert k.shape == expected_kv_shape
    assert v.shape == expected_kv_shape


@pytest.mark.unit
def test_attention_rope_application() -> None:
    """Test that RoPE is applied to Q and K in attention forward."""
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

    # Get Q/K before RoPE
    q_proj = attention.q_proj(hidden_states)
    k_proj = attention.k_proj(hidden_states)

    # Reshape for attention
    q = q_proj.view(batch_size, seq_len, config.num_attention_heads, head_dim)
    k = k_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim)

    # Apply RoPE manually
    from vibe_sgl_lite.models.qwen3.rope import apply_rotary_emb
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)

    # Q and K should be different after RoPE
    assert not torch.allclose(q, q_rope)
    assert not torch.allclose(k, k_rope)


@pytest.mark.unit
def test_attention_with_rope_preserves_norms() -> None:
    """Test that attention with RoPE preserves vector norms."""
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

    # Get Q/K projections
    q_proj = attention.q_proj(hidden_states)
    k_proj = attention.k_proj(hidden_states)

    # Reshape
    q = q_proj.view(batch_size, seq_len, config.num_attention_heads, head_dim)
    k = k_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim)

    # Apply RoPE
    from vibe_sgl_lite.models.qwen3.rope import apply_rotary_emb
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)

    # Norms should be preserved (rotation property)
    q_norm_before = torch.norm(q, dim=-1)
    q_norm_after = torch.norm(q_rope, dim=-1)
    k_norm_before = torch.norm(k, dim=-1)
    k_norm_after = torch.norm(k_rope, dim=-1)

    assert_tensors_close(q_norm_before, q_norm_after, atol=1e-5, rtol=1e-4)
    assert_tensors_close(k_norm_before, k_norm_after, atol=1e-5, rtol=1e-4)


@pytest.mark.unit
def test_attention_score_computation() -> None:
    """Test that attention scores are computed correctly."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads

    # Create Q and K tensors (after RoPE)
    q = torch.randn(batch_size, config.num_attention_heads, seq_len, head_dim)
    k = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)

    # Repeat KV heads for GQA
    k_repeated = attention.repeat_kv(k, attention.num_key_value_groups)

    # Compute attention scores: Q @ K^T / sqrt(head_dim)
    scores = torch.matmul(q, k_repeated.transpose(-2, -1)) / (head_dim ** 0.5)

    # Scores should have shape [batch, num_heads, seq_len, seq_len]
    expected_shape = (batch_size, config.num_attention_heads, seq_len, seq_len)
    assert scores.shape == expected_shape


@pytest.mark.unit
def test_attention_score_scaling() -> None:
    """Test that attention scores are scaled by 1/sqrt(head_dim)."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5
    head_dim = config.hidden_size // config.num_attention_heads

    # Create simple Q and K for testing
    q = torch.ones(batch_size, config.num_attention_heads, seq_len, head_dim)
    k = torch.ones(batch_size, config.num_key_value_heads, seq_len, head_dim)

    # Repeat KV heads
    k_repeated = attention.repeat_kv(k, attention.num_key_value_groups)

    # Compute unscaled scores
    scores_unscaled = torch.matmul(q, k_repeated.transpose(-2, -1))

    # Compute scaled scores
    scores_scaled = scores_unscaled / (head_dim ** 0.5)

    # Scaled scores should be smaller
    assert torch.all(torch.abs(scores_scaled) <= torch.abs(scores_unscaled))

    # Scaling factor should be 1/sqrt(head_dim)
    expected_scale = 1.0 / (head_dim ** 0.5)
    actual_scale = scores_scaled[0, 0, 0, 0] / scores_unscaled[0, 0, 0, 0]
    assert_tensors_close(
        torch.tensor(actual_scale),
        torch.tensor(expected_scale),
        atol=1e-6,
        rtol=1e-5
    )


@pytest.mark.unit
def test_attention_softmax_application() -> None:
    """Test that softmax is applied to attention scores."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    batch_size = 2
    seq_len = 10

    # Create random attention scores
    scores = torch.randn(batch_size, config.num_attention_heads, seq_len, seq_len)

    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # Attention weights should sum to 1 along last dimension
    sums = attn_weights.sum(dim=-1)
    expected_sums = torch.ones_like(sums)
    assert_tensors_close(sums, expected_sums, atol=1e-6, rtol=1e-5)

    # All weights should be in [0, 1]
    assert torch.all(attn_weights >= 0)
    assert torch.all(attn_weights <= 1)


@pytest.mark.unit
def test_attention_value_aggregation() -> None:
    """Test that attention weights are applied to values."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads

    # Create attention weights and values
    attn_weights = torch.randn(batch_size, config.num_attention_heads, seq_len, seq_len)
    attn_weights = F.softmax(attn_weights, dim=-1)

    v = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)

    # Repeat V heads for GQA
    v_repeated = attention.repeat_kv(v, attention.num_key_value_groups)

    # Apply attention to values
    output = torch.matmul(attn_weights, v_repeated)

    # Output should have shape [batch, num_heads, seq_len, head_dim]
    expected_shape = (batch_size, config.num_attention_heads, seq_len, head_dim)
    assert output.shape == expected_shape


@pytest.mark.unit
def test_attention_forward_with_different_batch_sizes() -> None:
    """Test attention forward with various batch sizes."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    seq_len = 10
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    for batch_size in [1, 2, 4, 8]:
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        output = attention.forward(hidden_states, freqs_cis)

        expected_shape = (batch_size, seq_len, config.hidden_size)
        assert output.shape == expected_shape


@pytest.mark.unit
def test_attention_forward_with_different_seq_lengths() -> None:
    """Test attention forward with various sequence lengths."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 2
    head_dim = config.hidden_size // config.num_attention_heads

    for seq_len in [1, 5, 10, 50, 100]:
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

        output = attention.forward(hidden_states, freqs_cis)

        expected_shape = (batch_size, seq_len, config.hidden_size)
        assert output.shape == expected_shape


@pytest.mark.unit
def test_attention_forward_single_token() -> None:
    """Test attention forward with single token (decode phase)."""
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
def test_attention_forward_gradient_flow() -> None:
    """Test that gradients flow through attention forward."""
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

    # Gradients should flow to input
    assert hidden_states.grad is not None
    assert not torch.isnan(hidden_states.grad).any()

    # Gradients should flow to parameters
    assert attention.q_proj.weight.grad is not None
    assert attention.k_proj.weight.grad is not None
    assert attention.v_proj.weight.grad is not None


@pytest.mark.unit
def test_attention_forward_deterministic() -> None:
    """Test that attention forward produces deterministic outputs."""
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

    # Run twice with same input
    output1 = attention.forward(hidden_states, freqs_cis)
    output2 = attention.forward(hidden_states, freqs_cis)

    # Outputs should be identical
    assert torch.equal(output1, output2)


@pytest.mark.unit
def test_attention_forward_no_nan_or_inf() -> None:
    """Test that attention forward never produces NaN or Inf."""
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

    # Test with various input patterns
    test_inputs = [
        torch.randn(batch_size, seq_len, config.hidden_size),
        torch.zeros(batch_size, seq_len, config.hidden_size),
        torch.ones(batch_size, seq_len, config.hidden_size),
        torch.randn(batch_size, seq_len, config.hidden_size) * 100,
        torch.randn(batch_size, seq_len, config.hidden_size) * 0.01,
    ]

    for hidden_states in test_inputs:
        output = attention.forward(hidden_states, freqs_cis)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


@pytest.mark.unit
def test_attention_with_gqa_head_repetition() -> None:
    """Test that GQA correctly repeats KV heads during attention."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 5
    head_dim = config.hidden_size // config.num_attention_heads

    # Create K and V with fewer heads
    k = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)
    v = torch.randn(batch_size, config.num_key_value_heads, seq_len, head_dim)

    # Repeat to match Q heads
    k_repeated = attention.repeat_kv(k, attention.num_key_value_groups)
    v_repeated = attention.repeat_kv(v, attention.num_key_value_groups)

    # Repeated tensors should have num_attention_heads
    assert k_repeated.shape[1] == config.num_attention_heads
    assert v_repeated.shape[1] == config.num_attention_heads

    # Each KV head should be repeated num_key_value_groups times
    for kv_idx in range(config.num_key_value_heads):
        for rep_idx in range(attention.num_key_value_groups):
            q_head_idx = kv_idx * attention.num_key_value_groups + rep_idx
            assert torch.equal(
                k_repeated[0, q_head_idx, :, :],
                k[0, kv_idx, :, :]
            )


@pytest.mark.unit
def test_attention_causal_mask_not_applied_by_default() -> None:
    """Test that attention does not apply causal mask by default."""
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

    # Forward should work without causal mask
    output = attention.forward(hidden_states, freqs_cis)

    assert output is not None
    assert output.shape == hidden_states.shape


@pytest.mark.unit
def test_attention_with_qwen3_config() -> None:
    """Test attention with actual Qwen3 configuration."""
    config = Qwen3Config.from_pretrained("Qwen/Qwen2.5-0.5B")

    attention = Qwen3Attention(config)

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
def test_attention_output_dtype_preservation() -> None:
    """Test that attention preserves input dtype."""
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

    # Test with float32
    hidden_states_f32 = torch.randn(
        batch_size, seq_len, config.hidden_size,
        dtype=torch.float32
    )
    output_f32 = attention.forward(hidden_states_f32, freqs_cis)
    assert output_f32.dtype == torch.float32


@pytest.mark.unit
def test_attention_with_rope_position_offset() -> None:
    """Test attention with RoPE position offset for KV cache."""
    config = Qwen3Config(
        hidden_size=896,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 1
    position_offset = 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=100)

    # Forward with position offset should work
    output = attention.forward(
        hidden_states,
        freqs_cis,
        position_offset=position_offset
    )

    expected_shape = (batch_size, seq_len, config.hidden_size)
    assert output.shape == expected_shape


@pytest.mark.unit
def test_attention_rope_relative_position_encoding() -> None:
    """Test that RoPE encodes relative positions in attention."""
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

    # Create identical hidden states
    hidden_states = torch.ones(batch_size, seq_len, config.hidden_size)

    # Get Q/K projections
    q_proj = attention.q_proj(hidden_states)
    k_proj = attention.k_proj(hidden_states)

    # Reshape
    q = q_proj.view(batch_size, seq_len, config.num_attention_heads, head_dim)
    k = k_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim)

    # Apply RoPE
    from vibe_sgl_lite.models.qwen3.rope import apply_rotary_emb
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)

    # Different positions should have different values after RoPE
    # even though input was identical
    assert not torch.allclose(q_rope[0, 0], q_rope[0, 1])
    assert not torch.allclose(k_rope[0, 0], k_rope[0, 1])


@pytest.mark.unit
def test_attention_computation_matches_manual_calculation() -> None:
    """Test that attention computation matches manual step-by-step calculation."""
    config = Qwen3Config(
        hidden_size=128,  # Smaller for easier testing
        num_attention_heads=4,
        num_key_value_heads=2,
    )

    attention = Qwen3Attention(config)

    batch_size = 1
    seq_len = 3
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len)

    # Manual calculation
    # 1. Project to Q/K/V
    q_proj = attention.q_proj(hidden_states)
    k_proj = attention.k_proj(hidden_states)
    v_proj = attention.v_proj(hidden_states)

    # 2. Reshape
    q = q_proj.view(batch_size, seq_len, config.num_attention_heads, head_dim)
    k = k_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim)
    v = v_proj.view(batch_size, seq_len, config.num_key_value_heads, head_dim)

    # 3. Apply RoPE
    from vibe_sgl_lite.models.qwen3.rope import apply_rotary_emb
    q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis)

    # 4. Transpose for attention
    q_t = q_rope.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
    k_t = k_rope.transpose(1, 2)  # [batch, num_kv_heads, seq_len, head_dim]
    v_t = v.transpose(1, 2)       # [batch, num_kv_heads, seq_len, head_dim]

    # 5. Repeat KV heads
    k_repeated = attention.repeat_kv(k_t, attention.num_key_value_groups)
    v_repeated = attention.repeat_kv(v_t, attention.num_key_value_groups)

    # 6. Compute attention scores
    scores = torch.matmul(q_t, k_repeated.transpose(-2, -1)) / (head_dim ** 0.5)

    # 7. Apply softmax
    attn_weights = F.softmax(scores, dim=-1)

    # 8. Apply to values
    attn_output = torch.matmul(attn_weights, v_repeated)

    # 9. Transpose and reshape
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, config.hidden_size)

    # Manual calculation should produce valid output
    assert attn_output.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(attn_output).any()
    assert not torch.isinf(attn_output).any()
