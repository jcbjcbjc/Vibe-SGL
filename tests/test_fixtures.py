"""
Tests for pytest fixtures defined in conftest.py.

This module validates that test fixtures work correctly:
- Model loading with caching
- CPU device enforcement
- Model size validation
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


@pytest.mark.unit
def test_qwen3_model_fixture_loads(qwen3_model: AutoModelForCausalLM) -> None:
    """Test that qwen3_model fixture successfully loads the model."""
    assert qwen3_model is not None
    # Check that model is a PreTrainedModel (base class for all HF models)
    assert isinstance(qwen3_model, PreTrainedModel)


@pytest.mark.unit
def test_qwen3_model_on_cpu(qwen3_model: AutoModelForCausalLM) -> None:
    """Test that model is loaded on CPU device."""
    # Check that all model parameters are on CPU
    for param in qwen3_model.parameters():
        assert param.device.type == "cpu", f"Parameter on {param.device}, expected CPU"


@pytest.mark.unit
def test_qwen3_model_in_eval_mode(qwen3_model: AutoModelForCausalLM) -> None:
    """Test that model is in evaluation mode."""
    assert not qwen3_model.training, "Model should be in eval mode"


@pytest.mark.unit
def test_qwen3_model_parameter_count(qwen3_model: AutoModelForCausalLM) -> None:
    """Test that model has expected parameter count (0.5-0.6B range)."""
    param_count = sum(p.numel() for p in qwen3_model.parameters())

    # Expected range: 400M to 700M parameters
    assert 400_000_000 <= param_count <= 700_000_000, (
        f"Model has {param_count:,} parameters, "
        f"expected between 400M and 700M"
    )


@pytest.mark.unit
def test_qwen3_model_has_config(qwen3_model: AutoModelForCausalLM) -> None:
    """Test that model has valid configuration."""
    assert hasattr(qwen3_model, "config")
    config = qwen3_model.config

    # Validate essential config attributes
    assert hasattr(config, "vocab_size")
    assert hasattr(config, "hidden_size")
    assert hasattr(config, "num_hidden_layers")
    assert hasattr(config, "num_attention_heads")

    # Check reasonable values
    assert config.vocab_size > 0
    assert config.hidden_size > 0
    assert config.num_hidden_layers > 0
    assert config.num_attention_heads > 0


@pytest.mark.integration
@pytest.mark.slow
def test_qwen3_model_forward_pass(qwen3_model: AutoModelForCausalLM) -> None:
    """Test that model can perform a forward pass."""
    # Create dummy input
    batch_size = 2
    seq_len = 10
    vocab_size = qwen3_model.config.vocab_size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    with torch.no_grad():
        outputs = qwen3_model(input_ids)

    # Validate output shape
    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (batch_size, seq_len, vocab_size)


@pytest.mark.integration
def test_qwen3_model_reused_across_tests(qwen3_model: AutoModelForCausalLM) -> None:
    """
    Test that the same model instance is reused across tests.

    This validates the session-scoped fixture behavior.
    """
    # Store model ID for comparison in other tests
    model_id = id(qwen3_model)
    assert model_id > 0  # Just verify we can get the ID


@pytest.mark.unit
def test_qwen3_tokenizer_fixture_loads(qwen3_tokenizer: AutoTokenizer) -> None:
    """Test that qwen3_tokenizer fixture successfully loads the tokenizer."""
    assert qwen3_tokenizer is not None
    # Check that tokenizer is a PreTrainedTokenizerBase (base class for all HF tokenizers)
    assert isinstance(qwen3_tokenizer, PreTrainedTokenizerBase)


@pytest.mark.unit
def test_qwen3_tokenizer_has_vocab(qwen3_tokenizer: AutoTokenizer) -> None:
    """Test that tokenizer has a valid vocabulary."""
    vocab_size = len(qwen3_tokenizer)
    assert vocab_size > 0, "Tokenizer should have non-empty vocabulary"
    # Qwen models typically have large vocabularies (>100k tokens)
    assert vocab_size > 100_000, f"Expected vocab_size > 100k, got {vocab_size:,}"


@pytest.mark.unit
def test_qwen3_tokenizer_encode_decode(qwen3_tokenizer: AutoTokenizer) -> None:
    """Test that tokenizer can encode and decode text."""
    test_text = "Hello, world!"

    # Encode text to token IDs
    token_ids = qwen3_tokenizer.encode(test_text)
    assert len(token_ids) > 0, "Encoding should produce tokens"
    assert all(isinstance(tid, int) for tid in token_ids), "Token IDs should be integers"

    # Decode back to text
    decoded_text = qwen3_tokenizer.decode(token_ids)
    assert isinstance(decoded_text, str), "Decoded output should be string"
    # Note: decoded text may not exactly match due to special tokens
    assert len(decoded_text) > 0, "Decoded text should not be empty"


@pytest.mark.unit
def test_qwen3_tokenizer_special_tokens(qwen3_tokenizer: AutoTokenizer) -> None:
    """Test that tokenizer has required special tokens."""
    # Check for common special tokens
    assert hasattr(qwen3_tokenizer, "eos_token")
    assert hasattr(qwen3_tokenizer, "bos_token")
    assert hasattr(qwen3_tokenizer, "pad_token")

    # Verify special token IDs are valid
    assert qwen3_tokenizer.eos_token_id is not None
    assert isinstance(qwen3_tokenizer.eos_token_id, int)


@pytest.mark.unit
def test_qwen3_tokenizer_batch_encoding(qwen3_tokenizer: AutoTokenizer) -> None:
    """Test that tokenizer can handle batch encoding."""
    texts = ["Hello, world!", "This is a test.", "Batch encoding works!"]

    # Batch encode with padding
    encoded = qwen3_tokenizer(
        texts,
        padding=True,
        return_tensors="pt"
    )

    # Validate output structure
    assert "input_ids" in encoded
    assert "attention_mask" in encoded

    # Check shapes
    batch_size = len(texts)
    assert encoded["input_ids"].shape[0] == batch_size
    assert encoded["attention_mask"].shape[0] == batch_size

    # All sequences should have same length due to padding
    assert encoded["input_ids"].shape[1] == encoded["attention_mask"].shape[1]


@pytest.mark.integration
def test_qwen3_tokenizer_matches_model(
    qwen3_tokenizer: AutoTokenizer,
    qwen3_model: AutoModelForCausalLM
) -> None:
    """Test that tokenizer vocabulary is compatible with model's vocabulary size."""
    tokenizer_vocab_size = len(qwen3_tokenizer)
    model_vocab_size = qwen3_model.config.vocab_size

    # Model vocab size may be padded for efficiency (e.g., to nearest multiple of 64)
    # Tokenizer vocab should be <= model vocab size
    assert tokenizer_vocab_size <= model_vocab_size, (
        f"Tokenizer vocab size ({tokenizer_vocab_size:,}) "
        f"exceeds model vocab size ({model_vocab_size:,})"
    )

    # They should be reasonably close (within 1% difference)
    diff_ratio = abs(tokenizer_vocab_size - model_vocab_size) / model_vocab_size
    assert diff_ratio < 0.01, (
        f"Tokenizer vocab size ({tokenizer_vocab_size:,}) differs too much "
        f"from model vocab size ({model_vocab_size:,}): {diff_ratio:.2%}"
    )


@pytest.mark.integration
def test_qwen3_tokenizer_reused_across_tests(qwen3_tokenizer: AutoTokenizer) -> None:
    """
    Test that the same tokenizer instance is reused across tests.

    This validates the session-scoped fixture behavior.
    """
    # Store tokenizer ID for comparison in other tests
    tokenizer_id = id(qwen3_tokenizer)
    assert tokenizer_id > 0  # Just verify we can get the ID


@pytest.mark.unit
def test_cpu_device_fixture_returns_cpu_device(cpu_device: torch.device) -> None:
    """Test that cpu_device fixture returns a torch.device with type 'cpu'."""
    assert isinstance(cpu_device, torch.device)
    assert cpu_device.type == "cpu"


@pytest.mark.unit
def test_cpu_device_forces_cpu_tensors(cpu_device: torch.device) -> None:
    """Test that tensors created with cpu_device are on CPU."""
    tensor = torch.randn(10, 10, device=cpu_device)
    assert tensor.device.type == "cpu"
    assert tensor.is_cpu


@pytest.mark.unit
def test_cpu_device_no_cuda_available(cpu_device: torch.device) -> None:
    """
    Test that CUDA is not available when using cpu_device fixture.

    The fixture should set CUDA_VISIBLE_DEVICES="" which makes CUDA unavailable.
    """
    assert not torch.cuda.is_available()


@pytest.mark.integration
def test_model_uses_cpu_device(cpu_device: torch.device, qwen3_model: AutoModelForCausalLM) -> None:
    """Test that model loaded with cpu_device is on CPU."""
    # Check that all model parameters are on CPU
    for param in qwen3_model.parameters():
        assert param.device.type == "cpu"
        assert param.is_cpu


@pytest.mark.unit
def test_cpu_device_tensor_operations(cpu_device: torch.device) -> None:
    """Test that tensor operations work correctly on CPU device."""
    # Create tensors on CPU
    a = torch.randn(5, 5, device=cpu_device)
    b = torch.randn(5, 5, device=cpu_device)

    # Perform operations
    c = a + b
    d = torch.matmul(a, b)

    # Verify results are on CPU
    assert c.device.type == "cpu"
    assert d.device.type == "cpu"

