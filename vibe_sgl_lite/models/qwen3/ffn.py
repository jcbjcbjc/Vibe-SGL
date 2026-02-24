"""
Qwen3 Feed-Forward Network (FFN) layer implementation.

This module implements the Qwen3 FFN layer with SwiGLU activation:
- up_proj: Projects hidden_size -> intermediate_size (for values)
- gate_proj: Projects hidden_size -> intermediate_size (for gating)
- down_proj: Projects intermediate_size -> hidden_size (output projection)

The FFN uses SwiGLU activation: SwiGLU(x) = Swish(gate_proj(x)) * up_proj(x)
where Swish(x) = x * sigmoid(x).

This implementation supports Tensor Parallelism (TP) where:
- up_proj and gate_proj are partitioned along output dimension (column parallel)
- down_proj is partitioned along input dimension (row parallel)
"""

import torch
import torch.nn as nn
from typing import Optional

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


class Qwen3FFN(nn.Module):
    """Qwen3 Feed-Forward Network layer with SwiGLU activation.

    This class implements the FFN layer used in Qwen3, which consists of
    three linear projections forming a gated activation network. The SwiGLU
    activation provides better performance than standard ReLU or GELU.

    Attributes:
        config: Qwen3 model configuration.
        hidden_size: Input/output dimension of the FFN.
        intermediate_size: Hidden dimension of the FFN (typically 4x hidden_size).
        up_proj: Linear projection for values (hidden_size -> intermediate_size).
        gate_proj: Linear projection for gating (hidden_size -> intermediate_size).
        down_proj: Linear projection for output (intermediate_size -> hidden_size).
    """

    def __init__(self, config: Qwen3Config, tp_degree: int = 1, rank: int = 0) -> None:
        """Initialize Qwen3FFN layer.

        Args:
            config: Qwen3 model configuration containing FFN parameters.
            tp_degree: Tensor parallelism degree (default: 1, no TP).
            rank: Current TP rank (default: 0).
        """
        super().__init__()

        self.config = config
        self.tp_degree = tp_degree
        self.rank = rank
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Calculate per-partition intermediate size for TP
        self.intermediate_size_per_partition = self.intermediate_size // tp_degree if tp_degree > 1 else self.intermediate_size

        # Initialize projection layers
        if tp_degree > 1:
            # Use parallel linear layers for TP
            from vibe_sgl_lite.distributed.tp.parallel_linear import (
                ColumnParallelLinear,
                RowParallelLinear,
            )

            # up_proj: Projects input to intermediate dimension (column parallel)
            # Shape: (hidden_size, intermediate_size)
            self.up_proj = ColumnParallelLinear(
                in_features=config.hidden_size,
                out_features=config.intermediate_size,
                bias=False,
                tp_degree=tp_degree,
                rank=rank,
            )

            # gate_proj: Projects input to intermediate dimension (column parallel)
            # Shape: (hidden_size, intermediate_size)
            self.gate_proj = ColumnParallelLinear(
                in_features=config.hidden_size,
                out_features=config.intermediate_size,
                bias=False,
                tp_degree=tp_degree,
                rank=rank,
            )

            # down_proj: Projects intermediate back to hidden dimension (row parallel)
            # Shape: (intermediate_size, hidden_size)
            self.down_proj = RowParallelLinear(
                in_features=config.intermediate_size,
                out_features=config.hidden_size,
                bias=False,
                tp_degree=tp_degree,
                rank=rank,
            )
        else:
            # Use standard linear layers (no TP)
            # up_proj: Projects input to intermediate dimension (for values)
            # Shape: (hidden_size, intermediate_size)
            self.up_proj = nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
            )

            # gate_proj: Projects input to intermediate dimension (for gating)
            # Shape: (hidden_size, intermediate_size)
            self.gate_proj = nn.Linear(
                config.hidden_size,
                config.intermediate_size,
                bias=False,
            )

            # down_proj: Projects intermediate back to hidden dimension
            # Shape: (intermediate_size, hidden_size)
            self.down_proj = nn.Linear(
                config.intermediate_size,
                config.hidden_size,
                bias=False,
            )

    def swiglu(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SwiGLU activation: Swish(gate_proj(x)) * up_proj(x).

        SwiGLU is a gated activation function that combines:
        - Swish activation: Swish(x) = x * sigmoid(x)
        - Gating mechanism: multiply Swish(gate) by up projection

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Tensor of shape [batch_size, seq_len, intermediate_size] containing
            the SwiGLU-activated values.
        """
        # Compute gate and up projections
        gate = self.gate_proj(x)  # [batch, seq_len, intermediate_size]
        up = self.up_proj(x)      # [batch, seq_len, intermediate_size]

        # Apply Swish activation to gate: Swish(x) = x * sigmoid(x)
        swish_gate = gate * torch.sigmoid(gate)

        # Multiply by up projection to get SwiGLU output
        return swish_gate * up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of FFN layer: FFN(x) = down_proj(SwiGLU(x)).

        The forward pass implements the complete FFN computation:
        1. Apply SwiGLU activation to transform input to intermediate dimension
        2. Apply down projection to transform back to hidden dimension

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Output tensor of shape [batch_size, seq_len, hidden_size].
        """
        # Apply SwiGLU activation
        swiglu_output = self.swiglu(x)  # [batch, seq_len, intermediate_size]

        # Apply down projection
        output = self.down_proj(swiglu_output)  # [batch, seq_len, hidden_size]

        return output
