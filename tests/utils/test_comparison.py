"""
Tests for model output comparison utilities.

This module tests utilities for comparing model outputs with appropriate
numerical tolerance for floating point comparisons.
"""

import pytest
import torch

from tests.utils.comparison import (
    assert_logits_close,
    assert_tensors_close,
    assert_tokens_equal,
    compare_generation_outputs,
)


@pytest.mark.unit
def test_assert_tensors_close_identical() -> None:
    """Test that identical tensors pass comparison."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0, 2.0, 3.0])
    # Should not raise
    assert_tensors_close(tensor1, tensor2)


@pytest.mark.unit
def test_assert_tensors_close_within_tolerance() -> None:
    """Test that tensors within tolerance pass comparison."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.00001, 2.00001, 3.00001])
    # Should not raise with default tolerance
    assert_tensors_close(tensor1, tensor2, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_assert_tensors_close_exceeds_tolerance() -> None:
    """Test that tensors exceeding tolerance fail comparison."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.1, 2.1, 3.1])
    with pytest.raises(AssertionError):
        assert_tensors_close(tensor1, tensor2, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_assert_tensors_close_different_shapes() -> None:
    """Test that tensors with different shapes fail comparison."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.0, 2.0])
    with pytest.raises(AssertionError):
        assert_tensors_close(tensor1, tensor2)


@pytest.mark.unit
def test_assert_tensors_close_multidimensional() -> None:
    """Test comparison of multidimensional tensors."""
    tensor1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    tensor2 = torch.tensor([[1.00001, 2.00001], [3.00001, 4.00001]])
    assert_tensors_close(tensor1, tensor2, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_assert_logits_close_identical() -> None:
    """Test that identical logits pass comparison."""
    logits1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logits2 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert_logits_close(logits1, logits2)


@pytest.mark.unit
def test_assert_logits_close_within_tolerance() -> None:
    """Test that logits within tolerance pass comparison."""
    logits1 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    logits2 = torch.tensor([[1.00001, 2.00001, 3.00001], [4.00001, 5.00001, 6.00001]])
    assert_logits_close(logits1, logits2, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_assert_logits_close_exceeds_tolerance() -> None:
    """Test that logits exceeding tolerance fail comparison."""
    logits1 = torch.tensor([[1.0, 2.0, 3.0]])
    logits2 = torch.tensor([[1.1, 2.1, 3.1]])
    with pytest.raises(AssertionError):
        assert_logits_close(logits1, logits2, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_assert_tokens_equal_identical() -> None:
    """Test that identical token sequences pass comparison."""
    tokens1 = torch.tensor([1, 2, 3, 4, 5])
    tokens2 = torch.tensor([1, 2, 3, 4, 5])
    assert_tokens_equal(tokens1, tokens2)


@pytest.mark.unit
def test_assert_tokens_equal_different() -> None:
    """Test that different token sequences fail comparison."""
    tokens1 = torch.tensor([1, 2, 3, 4, 5])
    tokens2 = torch.tensor([1, 2, 3, 4, 6])
    with pytest.raises(AssertionError):
        assert_tokens_equal(tokens1, tokens2)


@pytest.mark.unit
def test_assert_tokens_equal_different_lengths() -> None:
    """Test that token sequences with different lengths fail comparison."""
    tokens1 = torch.tensor([1, 2, 3, 4, 5])
    tokens2 = torch.tensor([1, 2, 3, 4])
    with pytest.raises(AssertionError):
        assert_tokens_equal(tokens1, tokens2)


@pytest.mark.unit
def test_compare_generation_outputs_identical() -> None:
    """Test comparison of identical generation outputs."""
    output1 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    }
    output2 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
    }
    # Should not raise
    compare_generation_outputs(output1, output2)


@pytest.mark.unit
def test_compare_generation_outputs_within_tolerance() -> None:
    """Test comparison of generation outputs within tolerance."""
    output1 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    output2 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.00001, 2.00001, 3.00001]]),
    }
    compare_generation_outputs(output1, output2, atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_compare_generation_outputs_different_tokens() -> None:
    """Test that different tokens fail comparison."""
    output1 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    output2 = {
        "tokens": torch.tensor([1, 2, 3, 4, 6]),
        "logits": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    with pytest.raises(AssertionError):
        compare_generation_outputs(output1, output2)


@pytest.mark.unit
def test_compare_generation_outputs_different_logits() -> None:
    """Test that different logits fail comparison."""
    output1 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    output2 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.1, 2.1, 3.1]]),
    }
    with pytest.raises(AssertionError):
        compare_generation_outputs(output1, output2, atol=1e-5, rtol=1e-5)


@pytest.mark.unit
def test_compare_generation_outputs_missing_keys() -> None:
    """Test that missing keys in output dictionaries fail comparison."""
    output1 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
        "logits": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    output2 = {
        "tokens": torch.tensor([1, 2, 3, 4, 5]),
    }
    with pytest.raises(KeyError):
        compare_generation_outputs(output1, output2)


@pytest.mark.unit
def test_assert_tensors_close_with_custom_message() -> None:
    """Test that custom error messages are included in assertion errors."""
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    tensor2 = torch.tensor([1.1, 2.1, 3.1])
    with pytest.raises(AssertionError, match="Custom error message"):
        assert_tensors_close(
            tensor1, tensor2, atol=1e-5, rtol=1e-5, msg="Custom error message"
        )
