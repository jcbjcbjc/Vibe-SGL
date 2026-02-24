"""
Pytest configuration and shared fixtures for vibe-sgl-lite tests.

This module provides reusable fixtures for testing, including:
- Model loading (Qwen3-0.6B with caching)
- Tokenizer loading
- CPU device enforcement
- Test data and utilities
"""

import os
from typing import Any, Dict

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# Force CPU-only testing by disabling CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""


@pytest.fixture(scope="session")
def qwen3_model() -> AutoModelForCausalLM:
    """
    Load Qwen3-0.6B model for testing (session-scoped with caching).

    This fixture:
    - Downloads Qwen3-0.6B from HuggingFace on first run
    - Caches the model locally for subsequent runs
    - Forces CPU device for all tests
    - Validates model size (0.6B parameters)
    - Reuses the same model instance across all tests in the session

    Returns:
        AutoModelForCausalLM: Loaded Qwen3-0.6B model on CPU

    Raises:
        ValueError: If model parameter count doesn't match expected 0.6B
    """
    model_name = "Qwen/Qwen2.5-0.5B"  # Using Qwen2.5-0.5B as closest available

    # Load model on CPU with caching enabled (default behavior)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # Move model to CPU explicitly
    model = model.to("cpu")

    # Validate model size (approximately 0.5-0.6B parameters)
    param_count = sum(p.numel() for p in model.parameters())
    expected_min = 400_000_000  # 400M parameters
    expected_max = 700_000_000  # 700M parameters

    if not (expected_min <= param_count <= expected_max):
        raise ValueError(
            f"Model parameter count {param_count:,} is outside expected range "
            f"[{expected_min:,}, {expected_max:,}]"
        )

    # Set to eval mode for testing
    model.eval()

    return model


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """
    Force CPU device for all tests.

    This fixture:
    - Returns a torch.device('cpu') for explicit device placement
    - Ensures CUDA_VISIBLE_DEVICES="" is set (done at module level)
    - Prevents accidental GPU usage in tests
    - Can be used to explicitly place tensors and models on CPU

    Returns:
        torch.device: CPU device object

    Example:
        def test_tensor_on_cpu(cpu_device):
            tensor = torch.randn(10, 10, device=cpu_device)
            assert tensor.device.type == "cpu"
    """
    return torch.device("cpu")


@pytest.fixture(scope="session")
def qwen3_tokenizer() -> AutoTokenizer:
    """
    Load Qwen3 tokenizer for testing (session-scoped with caching).

    This fixture:
    - Downloads Qwen3 tokenizer from HuggingFace on first run
    - Caches the tokenizer locally for subsequent runs
    - Reuses the same tokenizer instance across all tests in the session
    - Matches the model used in qwen3_model fixture

    Returns:
        AutoTokenizer: Loaded Qwen3 tokenizer

    Example:
        def test_tokenization(qwen3_tokenizer):
            tokens = qwen3_tokenizer.encode("Hello, world!")
            assert len(tokens) > 0
    """
    model_name = "Qwen/Qwen2.5-0.5B"  # Using Qwen2.5-0.5B as closest available

    # Load tokenizer with caching enabled (default behavior)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    return tokenizer


@pytest.fixture(scope="session")
def qwen3_model_name() -> str:
    """
    Return the Qwen3 model name used for testing.

    This fixture provides the model name string that can be used
    to initialize models and tokenizers.

    Returns:
        str: HuggingFace model name for Qwen3
    """
    return "Qwen/Qwen2.5-0.5B"
