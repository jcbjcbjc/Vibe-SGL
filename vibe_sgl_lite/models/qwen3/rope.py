"""
RoPE (Rotary Position Embeddings) implementation for Qwen3.

This module implements Rotary Position Embeddings (RoPE), which encodes
positional information by rotating query and key vectors in the attention
mechanism. RoPE has the key property that the attention score between two
tokens depends only on their relative position, not their absolute positions.

References:
- RoFormer: Enhanced Transformer with Rotary Position Embedding
  https://arxiv.org/abs/2104.09864
"""

import torch
from typing import Tuple


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Precompute rotation frequencies for RoPE.

    This function computes the complex exponentials (cos + i*sin) for all
    positions up to end. The frequencies follow the formula:
        freq_i = theta^(-2i/dim) for i in [0, dim/2)

    Args:
        dim: Dimension of each attention head (must be even)
        end: Maximum sequence length to precompute
        theta: Base value for frequency computation (default: 10000.0)

    Returns:
        Complex tensor of shape [end, dim // 2] containing
        precomputed rotation frequencies for each position
    """
    # Compute frequencies: theta^(-2i/dim) for i in [0, dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Create position indices [0, 1, 2, ..., end-1]
    positions = torch.arange(end)

    # Compute angles: position * frequency for each (position, frequency) pair
    # Shape: [end, dim // 2]
    angles = torch.outer(positions, freqs).float()

    # Convert to complex exponentials: e^(i * angle) = cos(angle) + i*sin(angle)
    # This represents rotations on the unit circle
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    return freqs_cis


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: torch.Tensor,
    position_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings to query and key tensors.

    RoPE works by treating pairs of dimensions as 2D coordinates and rotating
    them by an angle that depends on the position. This encodes positional
    information while preserving the relative position property.

    Args:
        q: Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
        k: Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
        freqs_cis: Precomputed frequencies of shape [end, head_dim // 2]
        position_offset: Offset to add to positions (for KV cache scenarios)

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # Get sequence length from input
    seq_len = q.shape[1]

    # Select frequencies for the current sequence positions
    # If position_offset is provided, shift the position indices
    freqs_cis_seq = freqs_cis[position_offset : position_offset + seq_len]

    # Reshape query and key to treat pairs of dimensions as complex numbers
    # Original shape: [batch_size, seq_len, num_heads, head_dim]
    # Target shape: [batch_size, seq_len, num_heads, head_dim // 2, 2]
    q_reshaped = q.float().reshape(*q.shape[:-1], -1, 2)
    k_reshaped = k.float().reshape(*k.shape[:-1], -1, 2)

    # Convert pairs of real numbers to complex numbers
    # Shape: [batch_size, seq_len, num_heads, head_dim // 2]
    q_complex = torch.view_as_complex(q_reshaped)
    k_complex = torch.view_as_complex(k_reshaped)

    # Reshape freqs_cis to broadcast correctly
    # freqs_cis shape: [seq_len, head_dim // 2]
    # Need to add batch and head dimensions: [1, seq_len, 1, head_dim // 2]
    freqs_cis_broadcast = freqs_cis_seq.unsqueeze(0).unsqueeze(2)

    # Apply rotation by complex multiplication
    # This performs the rotation: (x + iy) * (cos(θ) + i*sin(θ))
    q_rotated = q_complex * freqs_cis_broadcast
    k_rotated = k_complex * freqs_cis_broadcast

    # Convert back to real numbers
    # Shape: [batch_size, seq_len, num_heads, head_dim // 2, 2]
    q_out = torch.view_as_real(q_rotated)
    k_out = torch.view_as_real(k_rotated)

    # Flatten last two dimensions back to head_dim
    # Shape: [batch_size, seq_len, num_heads, head_dim]
    q_out = q_out.flatten(-2)
    k_out = k_out.flatten(-2)

    # Cast back to original dtype
    q_out = q_out.type_as(q)
    k_out = k_out.type_as(k)

    return q_out, k_out
