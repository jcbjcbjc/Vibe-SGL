"""
Tests for ColumnParallelLinear and RowParallelLinear layers.

This module tests:
- ColumnParallelLinear layer (split output dimension)
- RowParallelLinear layer (split input dimension + all-reduce)
- Weight partitioning and initialization
- Forward pass correctness
- Integration with TP infrastructure
"""

import pytest
import torch
import torch.nn as nn
from torch.testing import assert_close

from vibe_sgl_lite.distributed.tp.parallel_linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vibe_sgl_lite.distributed.tp.infrastructure import (
    init_tp_process_group,
    cleanup_tp_process_group,
    get_tp_rank,
    get_tp_world_size,
)


@pytest.fixture
def tp_setup():
    """Setup TP environment for testing."""
    # This will be called in multi-process context
    # For now, we'll test with single process and mock TP state
    yield
    # Cleanup
    if torch.distributed.is_initialized():
        cleanup_tp_process_group()


class TestColumnParallelLinear:
    """Tests for ColumnParallelLinear layer."""

    def test_initialization(self):
        """Test ColumnParallelLinear initialization."""
        in_features = 64
        out_features = 128

        # Create layer
        layer = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=2,
            rank=0,
        )

        # Check weight shape (output dimension should be split)
        assert layer.weight.shape == (out_features // 2, in_features)

        # Check bias shape
        assert layer.bias.shape == (out_features // 2,)

        # Check attributes
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert layer.out_features_per_partition == out_features // 2
        assert layer.tp_degree == 2
        assert layer.rank == 0

    def test_initialization_without_bias(self):
        """Test ColumnParallelLinear initialization without bias."""
        layer = ColumnParallelLinear(
            in_features=64,
            out_features=128,
            bias=False,
            tp_degree=2,
            rank=0,
        )

        assert layer.bias is None

    def test_initialization_invalid_partition(self):
        """Test that initialization fails if output dimension not divisible by tp_degree."""
        with pytest.raises(ValueError, match="not evenly divisible"):
            ColumnParallelLinear(
                in_features=64,
                out_features=127,  # Not divisible by 2
                bias=True,
                tp_degree=2,
                rank=0,
            )

    def test_forward_pass_shape(self):
        """Test forward pass output shape."""
        batch_size = 4
        seq_len = 16
        in_features = 64
        out_features = 128

        layer = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=2,
            rank=0,
        )

        # Input: [batch, seq_len, in_features]
        x = torch.randn(batch_size, seq_len, in_features)

        # Forward pass
        output = layer(x)

        # Output should have partitioned output dimension
        assert output.shape == (batch_size, seq_len, out_features // 2)

    def test_forward_pass_computation(self):
        """Test that forward pass computes correct values."""
        in_features = 4
        out_features = 8
        tp_degree = 2

        # Create full weight and bias for reference
        full_weight = torch.randn(out_features, in_features)
        full_bias = torch.randn(out_features)

        # Create layer for rank 0
        layer_rank0 = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=0,
        )

        # Set partitioned weights
        layer_rank0.weight.data = full_weight[:out_features // 2, :]
        layer_rank0.bias.data = full_bias[:out_features // 2]

        # Input
        x = torch.randn(2, 3, in_features)

        # Forward pass
        output_rank0 = layer_rank0(x)

        # Reference computation
        reference_output = torch.nn.functional.linear(x, full_weight, full_bias)
        reference_rank0 = reference_output[..., :out_features // 2]

        # Check correctness
        assert_close(output_rank0, reference_rank0, atol=1e-5, rtol=1e-4)

    def test_multiple_ranks(self):
        """Test that different ranks get different partitions."""
        in_features = 4
        out_features = 8
        tp_degree = 2

        # Create full weight
        full_weight = torch.randn(out_features, in_features)
        full_bias = torch.randn(out_features)

        # Create layers for both ranks
        layer_rank0 = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=0,
        )

        layer_rank1 = ColumnParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=1,
        )

        # Set partitioned weights
        layer_rank0.weight.data = full_weight[:out_features // 2, :]
        layer_rank0.bias.data = full_bias[:out_features // 2]

        layer_rank1.weight.data = full_weight[out_features // 2:, :]
        layer_rank1.bias.data = full_bias[out_features // 2:]

        # Input
        x = torch.randn(2, 3, in_features)

        # Forward pass
        output_rank0 = layer_rank0(x)
        output_rank1 = layer_rank1(x)

        # Concatenate outputs
        combined_output = torch.cat([output_rank0, output_rank1], dim=-1)

        # Reference computation
        reference_output = torch.nn.functional.linear(x, full_weight, full_bias)

        # Check that combined output matches reference
        assert_close(combined_output, reference_output, atol=1e-5, rtol=1e-4)

    def test_no_communication(self):
        """Test that ColumnParallelLinear does not perform communication."""
        # ColumnParallelLinear should not call all-reduce
        # This is a design property - each rank computes independently
        layer = ColumnParallelLinear(
            in_features=64,
            out_features=128,
            bias=True,
            tp_degree=2,
            rank=0,
        )

        x = torch.randn(2, 3, 64)

        # Forward pass should complete without distributed communication
        # (no assertion needed, just verify it doesn't crash)
        output = layer(x)
        assert output.shape == (2, 3, 64)


class TestRowParallelLinear:
    """Tests for RowParallelLinear layer."""

    def test_initialization(self):
        """Test RowParallelLinear initialization."""
        in_features = 128
        out_features = 64

        # Create layer
        layer = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=2,
            rank=0,
        )

        # Check weight shape (input dimension should be split)
        assert layer.weight.shape == (out_features, in_features // 2)

        # Check bias shape (only rank 0 has bias)
        assert layer.bias.shape == (out_features,)

        # Check attributes
        assert layer.in_features == in_features
        assert layer.out_features == out_features
        assert layer.in_features_per_partition == in_features // 2
        assert layer.tp_degree == 2
        assert layer.rank == 0

    def test_initialization_without_bias(self):
        """Test RowParallelLinear initialization without bias."""
        layer = RowParallelLinear(
            in_features=128,
            out_features=64,
            bias=False,
            tp_degree=2,
            rank=0,
        )

        assert layer.bias is None

    def test_initialization_invalid_partition(self):
        """Test that initialization fails if input dimension not divisible by tp_degree."""
        with pytest.raises(ValueError, match="not evenly divisible"):
            RowParallelLinear(
                in_features=127,  # Not divisible by 2
                out_features=64,
                bias=True,
                tp_degree=2,
                rank=0,
            )

    def test_forward_pass_shape_without_reduce(self):
        """Test forward pass output shape without all-reduce."""
        batch_size = 4
        seq_len = 16
        in_features = 128
        out_features = 64

        layer = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=2,
            rank=0,
        )

        # Input: [batch, seq_len, in_features_per_partition]
        # Note: input is already partitioned for row parallel
        x = torch.randn(batch_size, seq_len, in_features // 2)

        # Forward pass (without actual distributed reduce)
        output = layer(x)

        # Output should have full output dimension
        assert output.shape == (batch_size, seq_len, out_features)

    def test_forward_pass_computation(self):
        """Test that forward pass computes correct partial results."""
        in_features = 8
        out_features = 4
        tp_degree = 2

        # Create full weight and bias for reference
        full_weight = torch.randn(out_features, in_features)
        full_bias = torch.randn(out_features)

        # Create layer for rank 0
        layer_rank0 = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=0,
        )

        # Set partitioned weights (rank 0 gets first half of input dimension)
        layer_rank0.weight.data = full_weight[:, :in_features // 2]
        layer_rank0.bias.data = full_bias

        # Partitioned input (rank 0 gets first half)
        x_full = torch.randn(2, 3, in_features)
        x_rank0 = x_full[..., :in_features // 2]

        # Forward pass (this computes partial result, no actual reduce)
        output_rank0 = layer_rank0(x_rank0)

        # Reference computation (partial result)
        # Note: bias is only added on rank 0
        reference_partial = torch.nn.functional.linear(
            x_rank0, full_weight[:, :in_features // 2], full_bias
        )

        # Check correctness
        assert_close(output_rank0, reference_partial, atol=1e-5, rtol=1e-4)

    def test_bias_only_on_rank_zero(self):
        """Test that bias is only applied on rank 0."""
        in_features = 8
        out_features = 4
        tp_degree = 2

        full_bias = torch.randn(out_features)

        # Create layers for both ranks
        layer_rank0 = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=0,
        )

        layer_rank1 = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=1,
        )

        # Set bias
        layer_rank0.bias.data = full_bias
        layer_rank1.bias.data = full_bias

        # Create dummy input
        x = torch.zeros(2, 3, in_features // 2)

        # Forward pass
        output_rank0 = layer_rank0(x)
        output_rank1 = layer_rank1(x)

        # Rank 0 should have bias, rank 1 should not
        assert_close(output_rank0, full_bias.view(1, 1, -1).expand(2, 3, -1), atol=1e-5, rtol=1e-4)
        assert_close(output_rank1, torch.zeros(2, 3, out_features), atol=1e-5, rtol=1e-4)

    def test_combined_output_correctness(self):
        """Test that sum of partial results equals full computation."""
        in_features = 8
        out_features = 4
        tp_degree = 2

        # Create full weight and bias
        full_weight = torch.randn(out_features, in_features)
        full_bias = torch.randn(out_features)

        # Create layers for both ranks
        layer_rank0 = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=0,
        )

        layer_rank1 = RowParallelLinear(
            in_features=in_features,
            out_features=out_features,
            bias=True,
            tp_degree=tp_degree,
            rank=1,
        )

        # Set partitioned weights
        layer_rank0.weight.data = full_weight[:, :in_features // 2]
        layer_rank0.bias.data = full_bias

        layer_rank1.weight.data = full_weight[:, in_features // 2:]
        layer_rank1.bias.data = full_bias  # Will be ignored on rank 1

        # Full input
        x_full = torch.randn(2, 3, in_features)

        # Partitioned inputs
        x_rank0 = x_full[..., :in_features // 2]
        x_rank1 = x_full[..., in_features // 2:]

        # Forward pass on both ranks
        output_rank0 = layer_rank0(x_rank0)
        output_rank1 = layer_rank1(x_rank1)

        # Sum partial results (simulating all-reduce)
        combined_output = output_rank0 + output_rank1

        # Reference computation
        reference_output = torch.nn.functional.linear(x_full, full_weight, full_bias)

        # Check that combined output matches reference
        assert_close(combined_output, reference_output, atol=1e-5, rtol=1e-4)
