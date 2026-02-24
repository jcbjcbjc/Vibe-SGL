"""
Tests for validating FFN outputs against HuggingFace Qwen3.

This module validates that our custom Qwen3 FFN implementation is
functionally correct by testing:
- Weight loading from HuggingFace checkpoints
- FFN computation produces valid outputs
- Output shapes and numerical stability
- Consistency across layers

Note: We validate functional correctness rather than exact numerical matching
with HuggingFace, as implementation differences (numerical precision) may
cause minor variations.

Following TDD: These tests validate the complete FFN implementation.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN
from tests.utils.comparison import assert_tensors_close


@pytest.mark.integration
def test_ffn_weights_load_from_huggingface() -> None:
    """Test that FFN weights can be loaded from HuggingFace model."""
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

    # Get first layer FFN from HuggingFace model
    hf_ffn = hf_model.model.layers[0].mlp

    # Create custom FFN layer
    custom_ffn = Qwen3FFN(config)

    # Load weights from HuggingFace model into custom FFN
    custom_ffn.gate_proj.weight.data = hf_ffn.gate_proj.weight.data.clone()
    custom_ffn.up_proj.weight.data = hf_ffn.up_proj.weight.data.clone()
    custom_ffn.down_proj.weight.data = hf_ffn.down_proj.weight.data.clone()

    # Verify weights match
    assert_tensors_close(
        custom_ffn.gate_proj.weight,
        hf_ffn.gate_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Gate projection weights do not match"
    )
    assert_tensors_close(
        custom_ffn.up_proj.weight,
        hf_ffn.up_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Up projection weights do not match"
    )
    assert_tensors_close(
        custom_ffn.down_proj.weight,
        hf_ffn.down_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Down projection weights do not match"
    )


@pytest.mark.integration
def test_ffn_output_shape_matches_huggingface() -> None:
    """Test that FFN output shape matches HuggingFace."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_ffn = hf_model.model.layers[0].mlp
    custom_ffn = Qwen3FFN(config)
    custom_ffn.gate_proj.weight.data = hf_ffn.gate_proj.weight.data.clone()
    custom_ffn.up_proj.weight.data = hf_ffn.up_proj.weight.data.clone()
    custom_ffn.down_proj.weight.data = hf_ffn.down_proj.weight.data.clone()
    custom_ffn.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        custom_output = custom_ffn.forward(hidden_states)

    assert custom_output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.integration
def test_ffn_output_numerical_stability() -> None:
    """Test that FFN output is numerically stable (no NaN/Inf)."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_ffn = hf_model.model.layers[0].mlp
    custom_ffn = Qwen3FFN(config)
    custom_ffn.gate_proj.weight.data = hf_ffn.gate_proj.weight.data.clone()
    custom_ffn.up_proj.weight.data = hf_ffn.up_proj.weight.data.clone()
    custom_ffn.down_proj.weight.data = hf_ffn.down_proj.weight.data.clone()
    custom_ffn.eval()

    for seq_len in [1, 10, 50]:
        hidden_states = torch.randn(1, seq_len, config.hidden_size)

        with torch.no_grad():
            custom_output = custom_ffn.forward(hidden_states)

        assert not torch.isnan(custom_output).any()
        assert not torch.isinf(custom_output).any()
        assert custom_output.abs().max() < 100.0


@pytest.mark.integration
def test_ffn_output_consistency_across_layers() -> None:
    """Test that FFN outputs are consistent across different layers."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    num_layers_to_test = min(3, config.num_hidden_layers)
    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    for layer_idx in range(num_layers_to_test):
        hf_ffn = hf_model.model.layers[layer_idx].mlp
        custom_ffn = Qwen3FFN(config)
        custom_ffn.gate_proj.weight.data = hf_ffn.gate_proj.weight.data.clone()
        custom_ffn.up_proj.weight.data = hf_ffn.up_proj.weight.data.clone()
        custom_ffn.down_proj.weight.data = hf_ffn.down_proj.weight.data.clone()
        custom_ffn.eval()

        with torch.no_grad():
            custom_output = custom_ffn.forward(hidden_states)

        assert custom_output.shape == (batch_size, seq_len, config.hidden_size)
        assert not torch.isnan(custom_output).any()
        assert not torch.isinf(custom_output).any()


@pytest.mark.integration
def test_ffn_with_pretrained_weights_produces_valid_output() -> None:
    """Test that FFN with pretrained weights produces valid output."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_ffn = hf_model.model.layers[0].mlp
    custom_ffn = Qwen3FFN(config)
    custom_ffn.gate_proj.weight.data = hf_ffn.gate_proj.weight.data.clone()
    custom_ffn.up_proj.weight.data = hf_ffn.up_proj.weight.data.clone()
    custom_ffn.down_proj.weight.data = hf_ffn.down_proj.weight.data.clone()
    custom_ffn.eval()

    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        custom_output = custom_ffn.forward(hidden_states)

    assert custom_output.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(custom_output).any()
    assert not torch.isinf(custom_output).any()

    # Check output statistics are reasonable
    mean = custom_output.mean().item()
    std = custom_output.std().item()
    assert abs(mean) < 1.0
    assert 0.01 < std < 10.0


@pytest.mark.integration
def test_ffn_swiglu_activation_with_pretrained_weights() -> None:
    """Test that SwiGLU activation works correctly with pretrained weights."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_ffn = hf_model.model.layers[0].mlp
    custom_ffn = Qwen3FFN(config)
    custom_ffn.gate_proj.weight.data = hf_ffn.gate_proj.weight.data.clone()
    custom_ffn.up_proj.weight.data = hf_ffn.up_proj.weight.data.clone()
    custom_ffn.down_proj.weight.data = hf_ffn.down_proj.weight.data.clone()
    custom_ffn.eval()

    batch_size, seq_len = 1, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        # Test SwiGLU activation
        swiglu_output = custom_ffn.swiglu(hidden_states)

        # SwiGLU output should have intermediate_size dimension
        assert swiglu_output.shape == (batch_size, seq_len, config.intermediate_size)
        assert not torch.isnan(swiglu_output).any()
        assert not torch.isinf(swiglu_output).any()

        # Full forward pass
        full_output = custom_ffn.forward(hidden_states)
        assert full_output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.integration
def test_ffn_output_matches_huggingface_computation() -> None:
    """Test that FFN output closely matches HuggingFace computation."""
    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_ffn = hf_model.model.layers[0].mlp
    custom_ffn = Qwen3FFN(config)
    custom_ffn.gate_proj.weight.data = hf_ffn.gate_proj.weight.data.clone()
    custom_ffn.up_proj.weight.data = hf_ffn.up_proj.weight.data.clone()
    custom_ffn.down_proj.weight.data = hf_ffn.down_proj.weight.data.clone()
    custom_ffn.eval()

    batch_size, seq_len = 1, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        # Custom FFN output
        custom_output = custom_ffn.forward(hidden_states)

        # HuggingFace FFN output
        hf_output = hf_ffn.forward(hidden_states)

    # Outputs should be very close (allowing for minor numerical differences)
    assert_tensors_close(
        custom_output,
        hf_output,
        atol=1e-5,
        rtol=1e-4,
        msg="FFN outputs do not match HuggingFace"
    )

