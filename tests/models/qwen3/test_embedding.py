"""
Tests for Qwen3 token embedding layer.

This module tests the token embedding layer that converts token IDs to dense vectors.
The embedding layer is a standard nn.Embedding that maps vocabulary indices to hidden states.

Following TDD: These tests are written before implementing the embedding layer.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from tests.utils.comparison import assert_tensors_close


@pytest.mark.unit
def test_embedding_initialization() -> None:
    """Test that embedding layer initializes with correct dimensions."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    embedding = Qwen3Embedding(config)

    # Check that embedding has correct dimensions
    assert embedding.vocab_size == config.vocab_size
    assert embedding.hidden_size == config.hidden_size
    assert embedding.weight.shape == (config.vocab_size, config.hidden_size)


@pytest.mark.unit
def test_embedding_forward_output_shape() -> None:
    """Test that embedding forward pass produces correct output shape."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    embedding = Qwen3Embedding(config)
    embedding.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = embedding(input_ids)

    # Output shape should be [batch_size, seq_len, hidden_size]
    assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_embedding_different_sequence_lengths() -> None:
    """Test that embedding works with different sequence lengths."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    embedding = Qwen3Embedding(config)
    embedding.eval()

    batch_size = 1
    for seq_len in [1, 5, 10, 50]:
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = embedding(input_ids)

        assert output.shape == (batch_size, seq_len, config.hidden_size)


@pytest.mark.unit
def test_embedding_numerical_stability() -> None:
    """Test that embedding output is numerically stable (no NaN/Inf)."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    embedding = Qwen3Embedding(config)
    embedding.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output = embedding(input_ids)

    assert not torch.isnan(output).any(), "NaN detected in embedding output"
    assert not torch.isinf(output).any(), "Inf detected in embedding output"


@pytest.mark.unit
def test_embedding_deterministic() -> None:
    """Test that embedding produces deterministic outputs for same inputs."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    embedding = Qwen3Embedding(config)
    embedding.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        output1 = embedding(input_ids)
        output2 = embedding(input_ids)

    # Same input should produce identical output
    assert torch.allclose(output1, output2, atol=0, rtol=0)


@pytest.mark.integration
def test_embedding_weights_load_from_huggingface() -> None:
    """Test that embedding weights can be loaded from HuggingFace model."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    # Get embedding from HuggingFace model
    hf_embedding = hf_model.model.embed_tokens

    # Create custom embedding
    custom_embedding = Qwen3Embedding(config)

    # Load weights from HuggingFace
    custom_embedding.weight.data = hf_embedding.weight.data.clone()

    # Verify weights match
    assert_tensors_close(
        custom_embedding.weight,
        hf_embedding.weight,
        atol=1e-6,
        rtol=1e-5,
        msg="Embedding weights do not match"
    )


@pytest.mark.integration
def test_embedding_output_matches_huggingface() -> None:
    """Test that embedding output matches HuggingFace computation."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    # Load HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    hf_embedding = hf_model.model.embed_tokens

    # Create custom embedding and load weights
    custom_embedding = Qwen3Embedding(config)
    custom_embedding.weight.data = hf_embedding.weight.data.clone()
    custom_embedding.eval()

    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    with torch.no_grad():
        custom_output = custom_embedding(input_ids)
        hf_output = hf_embedding(input_ids)

    # Outputs should be identical (exact match expected for embedding lookup)
    assert_tensors_close(
        custom_output,
        hf_output,
        atol=1e-6,
        rtol=1e-5,
        msg="Embedding outputs do not match HuggingFace"
    )


@pytest.mark.integration
def test_embedding_with_edge_case_tokens() -> None:
    """Test embedding with edge case token IDs (0, max, etc.)."""
    from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding

    model_name = "Qwen/Qwen2.5-0.5B"
    config = Qwen3Config.from_pretrained(model_name)

    embedding = Qwen3Embedding(config)
    embedding.eval()

    # Test with edge case tokens
    edge_case_ids = torch.tensor([
        [0, 1, config.vocab_size - 1],  # First, second, and last token
        [config.vocab_size // 2, 100, 1000],  # Middle and random tokens
    ])

    with torch.no_grad():
        output = embedding(edge_case_ids)

    assert output.shape == (2, 3, config.hidden_size)
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()
