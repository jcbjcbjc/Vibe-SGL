"""
RMSNorm (Root Mean Square Layer Normalization) implementation.

RMSNorm is a simplified normalization technique that normalizes using only
the root mean square statistic, without centering (no mean subtraction).
It's computationally more efficient than LayerNorm while maintaining
similar performance.

Formula: RMSNorm(x) = x * rsqrt(mean(x^2) + eps) * weight

where:
- x: input tensor
- mean(x^2): mean of squared values along the last dimension
- rsqrt: reciprocal square root (1 / sqrt(x))
- eps: small constant to prevent division by zero
- weight: learnable scaling parameter (initialized to ones)
"""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    This normalization layer computes:
        output = x * rsqrt(mean(x^2) + eps) * weight

    Args:
        hidden_size: The size of the hidden dimension (last dimension of input)
        eps: Small constant for numerical stability (default: 1e-6)

    Attributes:
        weight: Learnable scaling parameter of shape (hidden_size,)
        eps: Epsilon value for numerical stability
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        Initialize RMSNorm layer.

        Args:
            hidden_size: The size of the hidden dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        # Initialize weight parameter to ones (identity scaling)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.

        Args:
            x: Input tensor of shape [..., hidden_size]

        Returns:
            Normalized tensor of same shape as input
        """
        # Compute variance (mean of squared values) along last dimension
        # keepdim=True preserves the dimension for broadcasting
        variance = torch.mean(x * x, dim=-1, keepdim=True)

        # Normalize: x * rsqrt(variance + eps)
        # rsqrt is more numerically stable than 1/sqrt
        x_normalized = x * torch.rsqrt(variance + self.eps)

        # Apply learnable weight scaling
        return x_normalized * self.weight
