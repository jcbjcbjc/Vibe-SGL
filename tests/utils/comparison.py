"""
Utilities for comparing model outputs with numerical tolerance.

This module provides functions for comparing tensors, logits, tokens, and
generation outputs with appropriate tolerance for floating point comparisons.
"""

from typing import Any, Dict, Optional

import torch


def assert_tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that two tensors are close within specified tolerance.

    This function compares two tensors element-wise and raises an AssertionError
    if they differ by more than the specified absolute and relative tolerances.

    Args:
        tensor1: First tensor to compare
        tensor2: Second tensor to compare
        atol: Absolute tolerance (default: 1e-5)
        rtol: Relative tolerance (default: 1e-4)
        msg: Optional custom error message

    Raises:
        AssertionError: If tensors differ by more than tolerance or have different shapes

    Example:
        >>> t1 = torch.tensor([1.0, 2.0, 3.0])
        >>> t2 = torch.tensor([1.00001, 2.00001, 3.00001])
        >>> assert_tensors_close(t1, t2, atol=1e-4, rtol=1e-4)
    """
    # Check shapes match
    if tensor1.shape != tensor2.shape:
        error_msg = (
            f"Tensor shapes do not match: {tensor1.shape} vs {tensor2.shape}"
        )
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)

    # Use torch.allclose for numerical comparison
    if not torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol):
        max_diff = torch.max(torch.abs(tensor1 - tensor2)).item()
        error_msg = (
            f"Tensors are not close within tolerance. "
            f"Max difference: {max_diff:.6e}, atol: {atol:.6e}, rtol: {rtol:.6e}"
        )
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def assert_logits_close(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that two logit tensors are close within specified tolerance.

    This is a specialized version of assert_tensors_close for logits,
    with the same behavior but a more descriptive name for clarity.

    Args:
        logits1: First logits tensor to compare (shape: [batch_size, vocab_size])
        logits2: Second logits tensor to compare (shape: [batch_size, vocab_size])
        atol: Absolute tolerance (default: 1e-5)
        rtol: Relative tolerance (default: 1e-4)
        msg: Optional custom error message

    Raises:
        AssertionError: If logits differ by more than tolerance or have different shapes

    Example:
        >>> logits1 = torch.tensor([[1.0, 2.0, 3.0]])
        >>> logits2 = torch.tensor([[1.00001, 2.00001, 3.00001]])
        >>> assert_logits_close(logits1, logits2, atol=1e-4, rtol=1e-4)
    """
    custom_msg = msg or "Logits comparison failed"
    assert_tensors_close(logits1, logits2, atol=atol, rtol=rtol, msg=custom_msg)


def assert_tokens_equal(
    tokens1: torch.Tensor,
    tokens2: torch.Tensor,
    msg: Optional[str] = None,
) -> None:
    """
    Assert that two token sequences are exactly equal.

    Token sequences must match exactly (no tolerance) since they are discrete values.

    Args:
        tokens1: First token sequence (shape: [seq_len] or [batch_size, seq_len])
        tokens2: Second token sequence (shape: [seq_len] or [batch_size, seq_len])
        msg: Optional custom error message

    Raises:
        AssertionError: If token sequences differ or have different shapes

    Example:
        >>> tokens1 = torch.tensor([1, 2, 3, 4, 5])
        >>> tokens2 = torch.tensor([1, 2, 3, 4, 5])
        >>> assert_tokens_equal(tokens1, tokens2)
    """
    # Check shapes match
    if tokens1.shape != tokens2.shape:
        error_msg = (
            f"Token sequence shapes do not match: {tokens1.shape} vs {tokens2.shape}"
        )
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)

    # Check exact equality (no tolerance for discrete tokens)
    if not torch.equal(tokens1, tokens2):
        # Find first mismatch for helpful error message
        diff_mask = tokens1 != tokens2
        if diff_mask.any():
            first_diff_idx = diff_mask.nonzero()[0].item()
            error_msg = (
                f"Token sequences differ. First mismatch at index {first_diff_idx}: "
                f"{tokens1.flatten()[first_diff_idx].item()} vs "
                f"{tokens2.flatten()[first_diff_idx].item()}"
            )
        else:
            error_msg = "Token sequences are not equal"
        if msg:
            error_msg = f"{msg}: {error_msg}"
        raise AssertionError(error_msg)


def compare_generation_outputs(
    output1: Dict[str, Any],
    output2: Dict[str, Any],
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """
    Compare two generation output dictionaries.

    This function compares generation outputs from two model runs, checking
    that tokens match exactly and logits are close within tolerance.

    Args:
        output1: First generation output dict with keys "tokens" and "logits"
        output2: Second generation output dict with keys "tokens" and "logits"
        atol: Absolute tolerance for logits comparison (default: 1e-5)
        rtol: Relative tolerance for logits comparison (default: 1e-4)

    Raises:
        AssertionError: If outputs differ
        KeyError: If required keys are missing from output dictionaries

    Example:
        >>> output1 = {
        ...     "tokens": torch.tensor([1, 2, 3]),
        ...     "logits": torch.tensor([[1.0, 2.0, 3.0]])
        ... }
        >>> output2 = {
        ...     "tokens": torch.tensor([1, 2, 3]),
        ...     "logits": torch.tensor([[1.00001, 2.00001, 3.00001]])
        ... }
        >>> compare_generation_outputs(output1, output2, atol=1e-4, rtol=1e-4)
    """
    # Check required keys exist
    required_keys = ["tokens", "logits"]
    for key in required_keys:
        if key not in output1:
            raise KeyError(f"Missing key '{key}' in first output")
        if key not in output2:
            raise KeyError(f"Missing key '{key}' in second output")

    # Compare tokens (exact match required)
    assert_tokens_equal(
        output1["tokens"],
        output2["tokens"],
        msg="Generated tokens do not match",
    )

    # Compare logits (with tolerance)
    assert_logits_close(
        output1["logits"],
        output2["logits"],
        atol=atol,
        rtol=rtol,
        msg="Generated logits do not match",
    )

