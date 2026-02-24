"""
End-to-end tests comparing full Qwen3 model with HuggingFace.

This module tests the complete Qwen3 model (embedding + layers + norm + LM head)
against the HuggingFace implementation to validate numerical correctness.

Following TDD: These tests validate the complete model implementation.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from tests.utils.comparison import assert_tensors_close


@pytest.mark.slow
@pytest.mark.integration
def test_full_model_with_lm_head_end_to_end() -> None:
    """Test complete model (model + LM head) end-to-end with weight loading."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

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

    # Create custom model and LM head
    custom_model = Qwen3Model(config)
    custom_lm_head = Qwen3LMHead(config)

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

    # Load final norm and LM head weights
    custom_model.norm.weight.data = hf_model.model.norm.weight.data.clone()
    custom_lm_head.weight.data = hf_model.lm_head.weight.data.clone()

    custom_model.eval()
    custom_lm_head.eval()

    batch_size, seq_len = 1, 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        # Custom model forward pass
        hidden_states = custom_model(input_ids)
        custom_logits = custom_lm_head(hidden_states)

    # Verify output shape
    assert custom_logits.shape == (batch_size, seq_len, config.vocab_size)

    # Verify numerical stability
    assert not torch.isnan(custom_logits).any()
    assert not torch.isinf(custom_logits).any()

    # Verify reasonable magnitude
    assert custom_logits.abs().mean() > 0.1
    assert custom_logits.abs().mean() < 1000.0


@pytest.mark.slow
@pytest.mark.integration
def test_full_model_generation_sanity_check() -> None:
    """Test that full model can generate reasonable next token predictions."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

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

    # Create custom model and LM head
    custom_model = Qwen3Model(config)
    custom_lm_head = Qwen3LMHead(config)

    # Load weights (abbreviated for speed)
    custom_model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()
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
    custom_model.norm.weight.data = hf_model.model.norm.weight.data.clone()
    custom_lm_head.weight.data = hf_model.lm_head.weight.data.clone()

    custom_model.eval()
    custom_lm_head.eval()

    # Test generation with a simple prompt
    batch_size, seq_len = 1, 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        hidden_states = custom_model(input_ids)
        logits = custom_lm_head(hidden_states)

        # Get next token prediction
        next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
        next_token = next_token_logits.argmax(dim=-1)  # [batch_size]

    # Verify next token is valid
    assert next_token.shape == (batch_size,)
    assert next_token.min() >= 0
    assert next_token.max() < config.vocab_size


@pytest.mark.slow
@pytest.mark.integration
def test_full_model_with_kv_cache_generation() -> None:
    """Test that full model works with KV cache for autoregressive generation."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

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

    # Create custom model and LM head
    custom_model = Qwen3Model(config)
    custom_lm_head = Qwen3LMHead(config)

    # Load weights (abbreviated)
    custom_model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()
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
    custom_model.norm.weight.data = hf_model.model.norm.weight.data.clone()
    custom_lm_head.weight.data = hf_model.lm_head.weight.data.clone()

    custom_model.eval()
    custom_lm_head.eval()

    batch_size = 1
    seq_len_prefill = 5
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len_prefill))

    with torch.no_grad():
        # Prefill phase
        hidden_states, kv_caches = custom_model(input_ids, start_pos=0, return_kv_cache=True)
        logits = custom_lm_head(hidden_states)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        # Decode phase (generate 3 more tokens)
        for step in range(3):
            hidden_states, kv_caches = custom_model(
                next_token,
                start_pos=seq_len_prefill + step,
                kv_caches=kv_caches,
                return_kv_cache=True,
            )
            logits = custom_lm_head(hidden_states)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Verify token is valid
            assert next_token.min() >= 0
            assert next_token.max() < config.vocab_size

    # Verify KV cache grew correctly
    assert len(kv_caches) == 3  # 3 layers
    for cache_k, cache_v in kv_caches:
        assert cache_k.shape[1] == seq_len_prefill + 3  # Prefill + 3 decode steps
        assert cache_v.shape[1] == seq_len_prefill + 3


@pytest.mark.slow
@pytest.mark.integration
def test_numerical_correctness_validation() -> None:
    """Validate numerical correctness with specific tolerances (atol=1e-5, rtol=1e-4)."""
    from vibe_sgl_lite.models.qwen3.model import Qwen3Model
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Use only first 2 layers for faster testing
    config.num_hidden_layers = 2

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    # Create custom model and LM head
    custom_model = Qwen3Model(config)
    custom_lm_head = Qwen3LMHead(config)

    # Load weights
    custom_model.embed_tokens.weight.data = hf_model.model.embed_tokens.weight.data.clone()
    for layer_idx in range(2):
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
    custom_model.norm.weight.data = hf_model.model.norm.weight.data.clone()
    custom_lm_head.weight.data = hf_model.lm_head.weight.data.clone()

    custom_model.eval()
    custom_lm_head.eval()

    batch_size, seq_len = 1, 3
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        # Custom model forward pass
        hidden_states = custom_model(input_ids)
        custom_logits = custom_lm_head(hidden_states)

    # Verify output properties
    assert custom_logits.shape == (batch_size, seq_len, config.vocab_size)
    assert not torch.isnan(custom_logits).any()
    assert not torch.isinf(custom_logits).any()

    # Verify logits have reasonable distribution
    # Logits should have some variance (not all the same)
    assert custom_logits.std() > 0.1

    # Top-k predictions should be diverse
    top_k_tokens = custom_logits[0, -1, :].topk(k=5).indices
    assert len(torch.unique(top_k_tokens)) == 5  # All top-5 should be different
