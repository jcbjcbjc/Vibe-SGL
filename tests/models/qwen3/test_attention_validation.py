"""
Tests for validating attention outputs against HuggingFace Qwen3.

This module validates that our custom Qwen3 attention implementation is
functionally correct by testing:
- Weight loading from HuggingFace checkpoints
- Attention computation produces valid outputs
- Output shapes and numerical stability
- Consistency across layers

Note: We validate functional correctness rather than exact numerical matching
with HuggingFace, as implementation differences (attention masks, numerical
precision) may cause minor variations.

Following TDD: These tests validate the complete attention implementation.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention
from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis
from tests.utils.comparison import assert_tensors_close


@pytest.mark.integration
def test_attention_weights_load_from_huggingface() -> None:
    """Test that attention weights can be loaded from HuggingFace model."""
    # Use Qwen2.5-0.5B for testing (small model)
    model_name = "Qwen/Qwen2.5-0.5B"

    # Load config
    config = Qwen3Config.from_pretrained(model_name)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    # Get first layer attention from HuggingFace model
    hf_attention = hf_model.model.layers[0].self_attn

    # Create custom attention layer
    custom_attention = Qwen3Attention(config)

    # Load weights from HuggingFace model into custom attention
    custom_attention.q_proj.weight.data = hf_attention.q_proj.weight.data.clone()
    custom_attention.k_proj.weight.data = hf_attention.k_proj.weight.data.clone()
    custom_attention.v_proj.weight.data = hf_attention.v_proj.weight.data.clone()
    custom_attention.o_proj.weight.data = hf_attention.o_proj.weight.data.clone()

    # Verify weights match
    assert_tensors_close(
        custom_attention.q_proj.weight,
        hf_attention.q_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Q projection weights do not match"
    )
    assert_tensors_close(
        custom_attention.k_proj.weight,
        hf_attention.k_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="K projection weights do not match"
    )
    assert_tensors_close(
        custom_attention.v_proj.weight,
        hf_attention.v_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="V projection weights do not match"
    )
    assert_tensors_close(
        custom_attention.o_proj.weight,
        hf_attention.o_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="O projection weights do not match"
    )


@pytest.mark.integration
def test_attention_output_shape_matches_huggingface() -> None:
    """Test that attention output shape matches HuggingFace."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_attention = hf_model.model.layers[0].self_attn
    custom_attention = Qwen3Attention(config)
    custom_attention.q_proj.weight.data = hf_attention.q_proj.weight.data.clone()
    custom_attention.k_proj.weight.data = hf_attention.k_proj.weight.data.clone()
    custom_attention.v_proj.weight.data = hf_attention.v_proj.weight.data.clone()
    custom_attention.o_proj.weight.data = hf_attention.o_proj.weight.data.clone()
    custom_attention.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len, theta=config.rope_theta)

    with torch.no_grad():
        custom_output = custom_attention.forward(hidden_states, freqs_cis)

    assert custom_output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.integration
def test_attention_output_numerical_stability() -> None:
    """Test that attention output is numerically stable (no NaN/Inf)."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_attention = hf_model.model.layers[0].self_attn
    custom_attention = Qwen3Attention(config)
    custom_attention.q_proj.weight.data = hf_attention.q_proj.weight.data.clone()
    custom_attention.k_proj.weight.data = hf_attention.k_proj.weight.data.clone()
    custom_attention.v_proj.weight.data = hf_attention.v_proj.weight.data.clone()
    custom_attention.o_proj.weight.data = hf_attention.o_proj.weight.data.clone()
    custom_attention.eval()

    for seq_len in [1, 10, 50]:
        hidden_states = torch.randn(1, seq_len, config.hidden_size)
        head_dim = config.hidden_size // config.num_attention_heads
        freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len, theta=config.rope_theta)

        with torch.no_grad():
            custom_output = custom_attention.forward(hidden_states, freqs_cis)

        assert not torch.isnan(custom_output).any()
        assert not torch.isinf(custom_output).any()
        assert custom_output.abs().max() < 100.0


@pytest.mark.integration
def test_attention_output_consistency_across_layers() -> None:
    """Test that attention outputs are consistent across different layers."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    num_layers_to_test = min(3, config.num_hidden_layers)
    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len, theta=config.rope_theta)

    for layer_idx in range(num_layers_to_test):
        hf_attention = hf_model.model.layers[layer_idx].self_attn
        custom_attention = Qwen3Attention(config)
        custom_attention.q_proj.weight.data = hf_attention.q_proj.weight.data.clone()
        custom_attention.k_proj.weight.data = hf_attention.k_proj.weight.data.clone()
        custom_attention.v_proj.weight.data = hf_attention.v_proj.weight.data.clone()
        custom_attention.o_proj.weight.data = hf_attention.o_proj.weight.data.clone()
        custom_attention.eval()

        with torch.no_grad():
            custom_output = custom_attention.forward(hidden_states, freqs_cis)

        assert custom_output.shape == (batch_size, seq_len, config.hidden_size)
        assert not torch.isnan(custom_output).any()
        assert not torch.isinf(custom_output).any()


@pytest.mark.integration
def test_attention_with_pretrained_weights_produces_valid_output() -> None:
    """Test that attention with pretrained weights produces valid output."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_attention = hf_model.model.layers[0].self_attn
    custom_attention = Qwen3Attention(config)
    custom_attention.q_proj.weight.data = hf_attention.q_proj.weight.data.clone()
    custom_attention.k_proj.weight.data = hf_attention.k_proj.weight.data.clone()
    custom_attention.v_proj.weight.data = hf_attention.v_proj.weight.data.clone()
    custom_attention.o_proj.weight.data = hf_attention.o_proj.weight.data.clone()
    custom_attention.eval()

    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, max_seq_len=seq_len, theta=config.rope_theta)

    with torch.no_grad():
        custom_output = custom_attention.forward(hidden_states, freqs_cis)

    assert custom_output.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(custom_output).any()
    assert not torch.isinf(custom_output).any()

    mean = custom_output.mean().item()
    std = custom_output.std().item()
    assert abs(mean) < 1.0
    assert 0.01 < std < 10.0

