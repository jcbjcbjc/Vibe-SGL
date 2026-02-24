"""
Parallel linear layers for tensor parallelism.

This module provides:
- ColumnParallelLinear: Splits output dimension across TP ranks
- RowParallelLinear: Splits input dimension across TP ranks with all-reduce
"""

from typing import Optional

import torch
import torch.nn as nn

from vibe_sgl_lite.distributed.tp.infrastructure import (
    all_reduce_wrapper,
    get_tp_group,
)


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column parallelism.

    This layer:
    - Splits the output dimension across TP ranks
    - Each rank computes a portion of the output
    - No communication needed during forward pass
    - Used for Q/K/V projections and FFN up/gate projections

    Args:
        in_features: Input feature dimension (not partitioned)
        out_features: Output feature dimension (will be partitioned)
        bias: Whether to include bias term
        tp_degree: Tensor parallelism degree (number of ranks)
        rank: Current rank (0 to tp_degree-1)

    Example:
        # Full layer would be: Linear(64, 128)
        # With TP degree 2:
        # - Rank 0: ColumnParallelLinear(64, 128, rank=0) -> output shape [..., 64]
        # - Rank 1: ColumnParallelLinear(64, 128, rank=1) -> output shape [..., 64]
        # Combined output: [..., 128]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_degree: int = 1,
        rank: int = 0,
    ):
        super().__init__()

        # Validate inputs
        if out_features % tp_degree != 0:
            raise ValueError(
                f"Output features {out_features} not evenly divisible by tp_degree {tp_degree}"
            )

        if rank < 0 or rank >= tp_degree:
            raise ValueError(f"Invalid rank {rank} for tp_degree {tp_degree}")

        # Store configuration
        self.in_features = in_features
        self.out_features = out_features
        self.tp_degree = tp_degree
        self.rank = rank
        self.out_features_per_partition = out_features // tp_degree

        # Create partitioned weight
        # Shape: [out_features_per_partition, in_features]
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )

        # Create partitioned bias (if enabled)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using standard initialization."""
        # Use Kaiming uniform initialization (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with column parallelism.

        Args:
            x: Input tensor of shape [..., in_features]

        Returns:
            Output tensor of shape [..., out_features_per_partition]
        """
        # Linear transformation: x @ weight.T + bias
        output = torch.nn.functional.linear(x, self.weight, self.bias)
        return output

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"out_features_per_partition={self.out_features_per_partition}, "
            f"bias={self.bias is not None}, "
            f"tp_degree={self.tp_degree}, "
            f"rank={self.rank}"
        )


class RowParallelLinear(nn.Module):
    """
    Linear layer with row parallelism.

    This layer:
    - Splits the input dimension across TP ranks
    - Each rank computes partial output
    - Performs all-reduce to sum partial results
    - Bias is only applied on rank 0 (to avoid duplication after all-reduce)
    - Used for attention output projection and FFN down projection

    Args:
        in_features: Input feature dimension (will be partitioned)
        out_features: Output feature dimension (not partitioned)
        bias: Whether to include bias term
        tp_degree: Tensor parallelism degree (number of ranks)
        rank: Current rank (0 to tp_degree-1)

    Example:
        # Full layer would be: Linear(128, 64)
        # With TP degree 2:
        # - Rank 0: RowParallelLinear(128, 64, rank=0) -> input shape [..., 64]
        # - Rank 1: RowParallelLinear(128, 64, rank=1) -> input shape [..., 64]
        # After all-reduce: output shape [..., 64]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        tp_degree: int = 1,
        rank: int = 0,
    ):
        super().__init__()

        # Validate inputs
        if in_features % tp_degree != 0:
            raise ValueError(
                f"Input features {in_features} not evenly divisible by tp_degree {tp_degree}"
            )

        if rank < 0 or rank >= tp_degree:
            raise ValueError(f"Invalid rank {rank} for tp_degree {tp_degree}")

        # Store configuration
        self.in_features = in_features
        self.out_features = out_features
        self.tp_degree = tp_degree
        self.rank = rank
        self.in_features_per_partition = in_features // tp_degree

        # Create partitioned weight
        # Shape: [out_features, in_features_per_partition]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )

        # Create bias (only used on rank 0 to avoid duplication)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using standard initialization."""
        # Use Kaiming uniform initialization (same as nn.Linear)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with row parallelism.

        Args:
            x: Input tensor of shape [..., in_features_per_partition]

        Returns:
            Output tensor of shape [..., out_features]
        """
        # Compute partial result
        # Only apply bias on rank 0 to avoid duplication after all-reduce
        bias = self.bias if self.rank == 0 else None
        output = torch.nn.functional.linear(x, self.weight, bias)

        # Perform all-reduce if TP is enabled
        if self.tp_degree > 1:
            tp_group = get_tp_group()
            if tp_group is not None:
                all_reduce_wrapper(output, group=tp_group)

        return output

    def extra_repr(self) -> str:
        """Extra representation for debugging."""
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"in_features_per_partition={self.in_features_per_partition}, "
            f"bias={self.bias is not None}, "
            f"tp_degree={self.tp_degree}, "
            f"rank={self.rank}"
        )
