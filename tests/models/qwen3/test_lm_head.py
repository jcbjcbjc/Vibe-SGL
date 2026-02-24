"""
Tests for Qwen3 LM head (output projection to vocabulary).

This module tests the language modeling head that projects hidden states to logits
over the vocabulary. The LM head is a linear layer that maps from hidden_size to vocab_size.

Following TDD: These tests are written before implementing the LM head.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_lm_head_initialization() -> None:
    """Test that LM head initializes with correct dimensions."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    lm_head = Qwen3LMHead(config)

    # Check that LM head has correct dimensions
    assert lm_head.hidden_size == config.hidden_size
    assert lm_head.vocab_size == config.vocab_size
    assert lm_head.weight.shape == (config.vocab_size, config.hidden_size)


@pytest.mark.unit
def test_lm_head_forward_output_shape() -> None:
    """Test that LM head forward pass produces correct output shape."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    lm_head = Qwen3LMHead(config)
    lm_head.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        logits = lm_head(hidden_states)

    # Output shape should be [batch_size, seq_len, vocab_size]
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


@pytest.mark.unit
def test_lm_head_different_sequence_lengths() -> None:
    """Test that LM head works with different sequence lengths."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    lm_head = Qwen3LMHead(config)
    lm_head.eval()

    batch_size = 1
    for seq_len in [1, 5, 10, 50]:
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        with torch.no_grad():
            logits = lm_head(hidden_states)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)


@pytest.mark.unit
def test_lm_head_numerical_stability() -> None:
    """Test that LM head output is numerically stable (no NaN/Inf)."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    lm_head = Qwen3LMHead(config)
    lm_head.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        logits = lm_head(hidden_states)

    assert not torch.isnan(logits).any(), "NaN detected in LM head output"
    assert not torch.isinf(logits).any(), "Inf detected in LM head output"


@pytest.mark.unit
def test_lm_head_deterministic() -> None:
    """Test that LM head produces deterministic outputs for same inputs."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    lm_head = Qwen3LMHead(config)
    lm_head.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        logits1 = lm_head(hidden_states)
        logits2 = lm_head(hidden_states)

    # Same input should produce identical output
    assert torch.allclose(logits1, logits2, atol=0, rtol=0)


@pytest.mark.integration
def test_lm_head_weights_load_from_huggingface() -> None:
    """Test that LM head weights can be loaded from HuggingFace model."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    # Get LM head from HuggingFace model
    hf_lm_head = hf_model.lm_head

    # Create custom LM head
    custom_lm_head = Qwen3LMHead(config)

    # Load weights from HuggingFace
    custom_lm_head.weight.data = hf_lm_head.weight.data.clone()

    # Verify weights match
    assert_tensors_close(
        custom_lm_head.weight,
        hf_lm_head.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="LM head weights do not match"
    )


@pytest.mark.integration
def test_lm_head_output_matches_huggingface() -> None:
    """Test that LM head output matches HuggingFace computation."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    hf_lm_head = hf_model.lm_head

    # Create custom LM head and load weights
    custom_lm_head = Qwen3LMHead(config)
    custom_lm_head.weight.data = hf_lm_head.weight.data.clone()
    custom_lm_head.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        custom_logits = custom_lm_head(hidden_states)
        hf_logits = hf_lm_head(hidden_states)

    # Outputs should be identical (exact match expected for linear projection)
    assert_tensors_close(
        custom_logits,
        hf_logits,
        atol=1e-5,
        rtol=1e-4,
        msg="LM head outputs do not match HuggingFace"
    )


@pytest.mark.integration
def test_lm_head_argmax_prediction() -> None:
    """Test that LM head argmax gives valid token IDs."""
    from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    lm_head = Qwen3LMHead(config)
    lm_head.eval()

    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

    with torch.no_grad():
        logits = lm_head(hidden_states)
        predicted_tokens = logits.argmax(dim=-1)

    # Predicted tokens should be valid (within vocab range)
    assert predicted_tokens.min() >= 0
    assert predicted_tokens.max() < config.vocab_size
    assert predicted_tokens.shape == (batch_size, seq_len)
