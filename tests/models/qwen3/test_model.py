"""
Tests for full Qwen3Model forward pass.

This module tests the complete Qwen3 model that combines:
- Token embedding layer
- Stack of decoder layers
- Final layer normalization
- LM head for logits computation

The model implements the standard transformer decoder architecture.

Following TDD: These tests are written before implementing Qwen3Model.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_qwen3model_initialization() -> None:
    """Test that Qwen3Model initializes with correct components."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    model = Qwen3Model(config)

    # Check that all components are initialized
    assert hasattr(model, "embed_tokens")
    assert hasattr(model, "layers")
    assert hasattr(model, "norm")

    # Check number of layers
    assert len(model.layers) == config.num_hidden_layers

    # Check final norm epsilon
    assert model.norm.eps == config.rms_norm_eps


@pytest.mark.unit
def test_qwen3model_forward_output_shape() -> None:
    """Test that Qwen3Model forward pass produces correct output shape."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    model = Qwen3Model(config)
    model.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    # Output shape should be [batch_size, seq_len, hidden_size]
    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_qwen3model_different_sequence_lengths() -> None:
    """Test that Qwen3Model works with different sequence lengths."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    model = Qwen3Model(config)
    model.eval()

    batch_size = 1
    for seq_len in [1, 5, 10, 20]:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_qwen3model_numerical_stability() -> None:
    """Test that Qwen3Model output is numerically stable (no NaN/Inf)."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    model = Qwen3Model(config)
    model.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = model(input_ids)

    assert not torch.isnan(output).any(), "NaN detected in model output"
    assert not torch.isinf(output).any(), "Inf detected in model output"


@pytest.mark.integration
def test_qwen3model_with_kv_cache() -> None:
    """Test that Qwen3Model works correctly with KV cache."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    model = Qwen3Model(config)
    model.eval()

    batch_size = 1
    seq_len_prefill = 5
    seq_len_decode = 1

    # Prefill phase
    input_ids_prefill = torch.randint(0, config.vocab_size, (batch_size, seq_len_prefill))

    with torch.no_grad():
        output_prefill, caches = model(input_ids_prefill, start_pos=0, return_kv_cache=True)

    assert output_prefill.shape == (batch_size, seq_len_prefill, config.hidden_size)
    assert len(caches) == config.num_hidden_layers

    # Each cache should be a tuple of (cache_k, cache_v)
    for cache_k, cache_v in caches:
        assert cache_k.shape[1] == seq_len_prefill  # [batch, seq_len, num_kv_heads, head_dim]
        assert cache_v.shape[1] == seq_len_prefill

    # Decode phase (use cached K/V)
    input_ids_decode = torch.randint(0, config.vocab_size, (batch_size, seq_len_decode))

    with torch.no_grad():
        output_decode, new_caches = model(
            input_ids_decode,
            start_pos=seq_len_prefill,
            kv_caches=caches,
            return_kv_cache=True,
        )

    assert output_decode.shape == (batch_size, seq_len_decode, config.hidden_size)
    assert len(new_caches) == config.num_hidden_layers

    # Cache should have grown
    for cache_k, cache_v in new_caches:
        assert cache_k.shape[1] == seq_len_prefill + seq_len_decode
        assert cache_v.shape[1] == seq_len_prefill + seq_len_decode


@pytest.mark.integration
def test_qwen3model_weights_load_from_huggingface() -> None:
    """Test that Qwen3Model weights can be loaded from HuggingFace model."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    # Create custom model
    custom_model = Qwen3Model(config)

    # Load embedding weights
    custom_model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()

    # Load layer weights (just check first layer for this test)
    hf_layer = hf_model.model.layers[0]
    custom_layer = custom_model.layers[0]

    custom_layer.self_attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
    custom_layer.self_attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
    custom_layer.self_attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
    custom_layer.self_attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()
    custom_layer.mlp.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
    custom_layer.mlp.up_proj.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
    custom_layer.mlp.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.clone()
    custom_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
    custom_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

    # Load final norm weights
    custom_model.norm.weight.data = hf_model.model.norm.weight.data.clone()

    # Verify embedding weights match
    assert_tensors_close(
        custom_model.embed_tokens.weight,
        hf_model.model.embed_tokens.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Embedding weights do not match"
    )

    # Verify final norm weights match
    assert_tensors_close(
        custom_model.norm.weight,
        hf_model.model.norm.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Final norm weights do not match"
    )


@pytest.mark.slow
@pytest.mark.integration
def test_qwen3model_end_to_end_sanity_check() -> None:
    """Test that Qwen3Model produces reasonable outputs end-to-end."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Use only first 3 layers for faster testing
    config.num_hidden_layers = 3

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    # Create custom model with 3 layers
    custom_model = Qwen3Model(config)

    # Load embedding weights
    custom_model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()

    # Load first 3 layer weights
    for layer_idx in range(3):
        hf_layer = hf_model.model.layers[layer_idx]
        custom_layer = custom_model.layers[layer_idx]

        custom_layer.self_attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
        custom_layer.self_attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
        custom_layer.self_attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
        custom_layer.self_attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()
        custom_layer.mlp.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
        custom_layer.mlp.up_proj.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
        custom_layer.mlp.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.clone()
        custom_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
        custom_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

    # Load final norm weights
    custom_model.norm.weight.data = hf_model.model.norm.weight.data.clone()
    custom_model.eval()

    batch_size, seq_len = 1, 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        # Custom model output
        custom_output = custom_model(input_ids)

    # Verify output shape and numerical stability
    assert custom_output.shape == (batch_size, seq_len, config.hidden_size)
    assert not torch.isnan(custom_output).any()
    assert not torch.isinf(custom_output).any()

    # Verify output has reasonable magnitude (not too large or too small)
    assert custom_output.abs().mean() > 0.01
    assert custom_output.abs().mean() < 100.0
    assert custom_output.abs().max() < 1000.0
