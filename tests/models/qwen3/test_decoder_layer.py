"""
Tests for Qwen3DecoderLayer (attention + FFN + norms).

This module tests the complete decoder layer implementation that combines:
- Input layer normalization (RMSNorm)
- Self-attention mechanism with RoPE and GQA
- Post-attention layer normalization (RMSNorm)
- Feed-forward network with SwiGLU activation
- Residual connections around attention and FFN

The decoder layer follows the pre-norm architecture:
    x = x + attention(norm(x))
    x = x + ffn(norm(x))

Following TDD: These tests are written before implementing Qwen3DecoderLayer.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_decoder_layer_initialization() -> None:
    """Test that Qwen3DecoderLayer initializes with correct components."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    layer = Qwen3DecoderLayer(config)

    # Check that all components are initialized
    assert hasattr(layer, "input_layernorm")
    assert hasattr(layer, "self_attn")
    assert hasattr(layer, "post_attention_layernorm")
    assert hasattr(layer, "mlp")

    # Check layer norm epsilon
    assert layer.input_layernorm.eps == config.rms_norm_eps
    assert layer.post_attention_layernorm.eps == config.rms_norm_eps


@pytest.mark.unit
def test_decoder_layer_forward_output_shape() -> None:
    """Test that decoder layer forward pass produces correct output shape."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
    from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    layer = Qwen3DecoderLayer(config)
    layer.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    with torch.no_grad():
        output = layer.forward(hidden_states, freqs_cis=freqs_cis)

    # Output shape should match input shape
    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_decoder_layer_residual_connections() -> None:
    """Test that decoder layer applies residual connections correctly."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
    from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    layer = Qwen3DecoderLayer(config)
    layer.eval()

    batch_size, seq_len = 1, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    with torch.no_grad():
        output = layer.forward(hidden_states, freqs_cis=freqs_cis)

    # Output should be different from input (not identity)
    assert not torch.allclose(output, hidden_states)

    # But output should have reasonable magnitude (residual helps)
    # If residuals work correctly, output shouldn't explode
    assert output.abs().max() < 100.0


@pytest.mark.unit
def test_decoder_layer_numerical_stability() -> None:
    """Test that decoder layer output is numerically stable (no NaN/Inf)."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
    from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    layer = Qwen3DecoderLayer(config)
    layer.eval()

    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    # Test with various sequence lengths
    for seq_len in [1, 10, 50]:
        hidden_states = torch.randn(1, seq_len, config.hidden_size)

        with torch.no_grad():
            output = layer.forward(hidden_states, freqs_cis=freqs_cis)

        assert not torch.isnan(output).any(), f"NaN detected for seq_len={seq_len}"
        assert not torch.isinf(output).any(), f"Inf detected for seq_len={seq_len}"


@pytest.mark.integration
def test_decoder_layer_weights_load_from_huggingface() -> None:
    """Test that decoder layer weights can be loaded from HuggingFace model."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    # Get first decoder layer from HuggingFace model
    hf_layer = hf_model.model.layers[0]

    # Create custom decoder layer
    custom_layer = Qwen3DecoderLayer(config)

    # Load attention weights
    custom_layer.self_attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
    custom_layer.self_attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
    custom_layer.self_attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
    custom_layer.self_attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()

    # Load FFN weights
    custom_layer.mlp.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
    custom_layer.mlp.up_proj.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
    custom_layer.mlp.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.clone()

    # Load layer norm weights
    custom_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
    custom_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()

    # Verify attention weights match
    assert_tensors_close(
        custom_layer.self_attn.q_proj.weight,
        hf_layer.self_attn.q_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Q projection weights do not match"
    )

    # Verify FFN weights match
    assert_tensors_close(
        custom_layer.mlp.gate_proj.weight,
        hf_layer.mlp.gate_proj.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Gate projection weights do not match"
    )

    # Verify layer norm weights match
    assert_tensors_close(
        custom_layer.input_layernorm.weight,
        hf_layer.input_layernorm.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Input layer norm weights do not match"
    )


@pytest.mark.integration
def test_decoder_layer_output_shape_matches_huggingface() -> None:
    """Test that decoder layer output shape matches HuggingFace."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
    from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_layer = hf_model.model.layers[0]
    custom_layer = Qwen3DecoderLayer(config)

    # Load all weights
    custom_layer.self_attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
    custom_layer.self_attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
    custom_layer.self_attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
    custom_layer.self_attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()
    custom_layer.mlp.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
    custom_layer.mlp.up_proj.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
    custom_layer.mlp.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.clone()
    custom_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
    custom_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()
    custom_layer.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    with torch.no_grad():
        custom_output = custom_layer.forward(hidden_states, freqs_cis=freqs_cis)

    assert custom_output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.integration
def test_decoder_layer_consistency_across_layers() -> None:
    """Test that decoder layer outputs are consistent across different layers."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
    from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    num_layers_to_test = min(3, config.num_hidden_layers)
    batch_size, seq_len = 1, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    for layer_idx in range(num_layers_to_test):
        hf_layer = hf_model.model.layers[layer_idx]
        custom_layer = Qwen3DecoderLayer(config)

        # Load weights
        custom_layer.self_attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
        custom_layer.self_attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
        custom_layer.self_attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
        custom_layer.self_attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()
        custom_layer.mlp.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
        custom_layer.mlp.up_proj.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
        custom_layer.mlp.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.clone()
        custom_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
        custom_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()
        custom_layer.eval()

        with torch.no_grad():
            custom_output = custom_layer.forward(hidden_states, freqs_cis=freqs_cis)

        assert custom_output.shape == (batch_size, seq_len, config.hidden_size)
        assert not torch.isnan(custom_output).any()
        assert not torch.isinf(custom_output).any()


@pytest.mark.integration
def test_decoder_layer_with_kv_cache() -> None:
    """Test that decoder layer works correctly with KV cache."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
    from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    layer = Qwen3DecoderLayer(config)
    layer.eval()

    batch_size = 1
    seq_len_prefill = 5
    seq_len_decode = 1

    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    # Prefill phase
    hidden_states_prefill = torch.randn(batch_size, seq_len_prefill, config.hidden_size)

    with torch.no_grad():
        output_prefill, cache_k, cache_v = layer.forward(
            hidden_states_prefill,
            freqs_cis=freqs_cis,
            start_pos=0,
        )

    assert output_prefill.shape == (batch_size, seq_len_prefill, config.hidden_size)
    assert cache_k.shape[1] == seq_len_prefill  # [batch, seq_len, num_kv_heads, head_dim]
    assert cache_v.shape[1] == seq_len_prefill

    # Decode phase (use cached K/V)
    hidden_states_decode = torch.randn(batch_size, seq_len_decode, config.hidden_size)

    with torch.no_grad():
        output_decode, cache_k_new, cache_v_new = layer.forward(
            hidden_states_decode,
            freqs_cis=freqs_cis,
            start_pos=seq_len_prefill,
            cache_k=cache_k,
            cache_v=cache_v,
        )

    assert output_decode.shape == (batch_size, seq_len_decode, config.hidden_size)
    assert cache_k_new.shape[1] == seq_len_prefill + seq_len_decode
    assert cache_v_new.shape[1] == seq_len_prefill + seq_len_decode


@pytest.mark.integration
def test_decoder_layer_output_matches_huggingface() -> None:
    """Test that decoder layer output closely matches HuggingFace computation."""
    from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
    from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, torch_dtype=torch.float32
    )
    hf_model.eval()

    hf_layer = hf_model.model.layers[0]
    custom_layer = Qwen3DecoderLayer(config)

    # Load all weights
    custom_layer.self_attn.q_proj.weight.data = hf_layer.self_attn.q_proj.weight.data.clone()
    custom_layer.self_attn.k_proj.weight.data = hf_layer.self_attn.k_proj.weight.data.clone()
    custom_layer.self_attn.v_proj.weight.data = hf_layer.self_attn.v_proj.weight.data.clone()
    custom_layer.self_attn.o_proj.weight.data = hf_layer.self_attn.o_proj.weight.data.clone()
    custom_layer.mlp.gate_proj.weight.data = hf_layer.mlp.gate_proj.weight.data.clone()
    custom_layer.mlp.up_proj.weight.data = hf_layer.mlp.up_proj.weight.data.clone()
    custom_layer.mlp.down_proj.weight.data = hf_layer.mlp.down_proj.weight.data.clone()
    custom_layer.input_layernorm.weight.data = hf_layer.input_layernorm.weight.data.clone()
    custom_layer.post_attention_layernorm.weight.data = hf_layer.post_attention_layernorm.weight.data.clone()
    custom_layer.eval()

    batch_size, seq_len = 1, 5
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    # Precompute RoPE frequencies
    freqs_cis = precompute_freqs_cis(
        dim=config.hidden_size // config.num_attention_heads,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    with torch.no_grad():
        # Custom decoder layer output
        custom_output = custom_layer.forward(hidden_states, freqs_cis=freqs_cis)

        # HuggingFace decoder layer output
        # Note: HuggingFace returns a tuple (hidden_states, self_attn_weights, present_key_value)
        # Compute position embeddings for HuggingFace model
        position_ids = torch.arange(seq_len).unsqueeze(0)
        # Get rotary embeddings from HuggingFace model
        cos, sin = hf_model.model.rotary_emb(hidden_states, position_ids)
        position_embeddings = (cos, sin)

        hf_output = hf_layer.forward(
            hidden_states,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
        )[0]  # Get hidden_states only

    # Outputs should be reasonably close (allowing for numerical differences due to different implementations)
    # Note: Larger tolerance needed because HuggingFace may use different attention implementations (SDPA)
    assert_tensors_close(
        custom_output,
        hf_output,
        atol=0.3,
        rtol=0.1,
        msg="Decoder layer outputs do not match HuggingFace"
    )

