"""
Qwen3 attention layer implementation.

This module implements the Qwen3 attention mechanism with support for:
- Grouped-Query Attention (GQA) with configurable key-value head grouping
- Rotary Position Embeddings (RoPE)
- KV cache for efficient autoregressive generation
- Tensor Parallelism (TP) support for distributed inference

The attention layer consists of:
- Q/K/V projection layers (nn.Linear)
- RoPE application to queries and keys
- Scaled dot-product attention with GQA
- Output projection layer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.rope import apply_rotary_emb


def compute_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
) -> torch.Tensor:
    """Compute scaled dot-product attention with GQA support.

    Args:
        query: Query tensor of shape [batch_size, num_q_heads, seq_len, head_dim].
        key: Key tensor of shape [batch_size, num_kv_heads, cache_len, head_dim].
        value: Value tensor of shape [batch_size, num_kv_heads, cache_len, head_dim].

    Returns:
        Attention output of shape [batch_size, num_q_heads, seq_len, head_dim].
    """
    batch_size, num_q_heads, seq_len, head_dim = query.shape
    _, num_kv_heads, cache_len, _ = key.shape

    # Handle GQA: repeat KV heads to match query heads
    if num_q_heads != num_kv_heads:
        num_groups = num_q_heads // num_kv_heads
        # Repeat each KV head num_groups times
        key = key.repeat_interleave(num_groups, dim=1)
        value = value.repeat_interleave(num_groups, dim=1)

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) / (head_dim ** 0.5)

    # Apply softmax
    attn_weights = torch.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, value)

    return output


class Qwen3Attention(nn.Module):
    """Qwen3 attention layer with GQA and RoPE support.

    This class implements the multi-head attention mechanism used in Qwen3,
    with support for Grouped-Query Attention (GQA) where the number of key-value
    heads can be smaller than the number of query heads for efficiency.

    Attributes:
        config: Qwen3 model configuration.
        num_heads: Number of query attention heads.
        num_key_value_heads: Number of key-value attention heads (for GQA).
        num_key_value_groups: Number of query heads per key-value head.
        head_dim: Dimension of each attention head.
        q_proj: Query projection layer.
        k_proj: Key projection layer.
        v_proj: Value projection layer.
        o_proj: Output projection layer.
    """

    def __init__(self, config: Qwen3Config, tp_degree: int = 1, rank: int = 0) -> None:
        """Initialize Qwen3Attention layer.

        Args:
            config: Qwen3 model configuration containing attention parameters.
            tp_degree: Tensor parallelism degree (default: 1, no TP).
            rank: Current TP rank (default: 0).
        """
        super().__init__()

        self.config = config
        self.tp_degree = tp_degree
        self.rank = rank

        # Store attention head configuration
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        # Calculate head dimension
        self.head_dim = config.hidden_size // config.num_attention_heads

        # Validate TP configuration
        if tp_degree > 1:
            if self.num_heads % tp_degree != 0:
                raise ValueError(
                    f"Number of attention heads {self.num_heads} not evenly divisible by tp_degree {tp_degree}"
                )
            if self.num_key_value_heads % tp_degree != 0:
                raise ValueError(
                    f"Number of key-value heads {self.num_key_value_heads} not evenly divisible by tp_degree {tp_degree}"
                )

        # Calculate per-partition head counts for TP
        self.num_heads_per_partition = self.num_heads // tp_degree
        self.num_kv_heads_per_partition = self.num_key_value_heads // tp_degree

        # Initialize Q/K/V projection layers
        if tp_degree > 1:
            # Use parallel linear layers for TP
            from vibe_sgl_lite.distributed.tp.parallel_linear import ColumnParallelLinear

            # Q projection: hidden_size -> num_attention_heads * head_dim (column parallel)
            self.q_proj = ColumnParallelLinear(
                in_features=config.hidden_size,
                out_features=self.num_heads * self.head_dim,
                bias=False,
                tp_degree=tp_degree,
                rank=rank,
            )

            # K/V projections: hidden_size -> num_key_value_heads * head_dim (column parallel)
            self.k_proj = ColumnParallelLinear(
                in_features=config.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=False,
                tp_degree=tp_degree,
                rank=rank,
            )

            self.v_proj = ColumnParallelLinear(
                in_features=config.hidden_size,
                out_features=self.num_key_value_heads * self.head_dim,
                bias=False,
                tp_degree=tp_degree,
                rank=rank,
            )

            # Output projection: (num_heads * head_dim) -> hidden_size (row parallel)
            from vibe_sgl_lite.distributed.tp.parallel_linear import RowParallelLinear

            self.o_proj = RowParallelLinear(
                in_features=config.hidden_size,
                out_features=config.hidden_size,
                bias=False,
                tp_degree=tp_degree,
                rank=rank,
            )
        else:
            # Use standard linear layers (no TP)
            # Q projection: hidden_size -> num_attention_heads * head_dim
            self.q_proj = nn.Linear(
                config.hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
            )

            # K/V projections: hidden_size -> num_key_value_heads * head_dim
            # For GQA, K/V have fewer heads than Q
            self.k_proj = nn.Linear(
                config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
            )

            self.v_proj = nn.Linear(
                config.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
            )

            # Output projection: (num_heads * head_dim) -> hidden_size
            # Projects attention output back to hidden dimension
            self.o_proj = nn.Linear(
                config.hidden_size,
                config.hidden_size,
                bias=False,
            )

    def repeat_kv(
        self,
        hidden_states: torch.Tensor,
        n_rep: int,
    ) -> torch.Tensor:
        """Repeat key/value heads to match the number of query heads.

        This function implements the key-value head repetition mechanism for
        Grouped-Query Attention (GQA). When the number of key-value heads is
        smaller than the number of query heads, each KV head is repeated
        multiple times to match the query head count.

        Args:
            hidden_states: Input tensor with shape [batch, num_kv_heads, seq_len, head_dim].
            n_rep: Number of times to repeat each KV head (num_key_value_groups).

        Returns:
            Tensor with shape [batch, num_kv_heads * n_rep, seq_len, head_dim].

        Example:
            For GQA with 14 query heads and 2 KV heads:
            - Input: [batch, 2, seq_len, head_dim]
            - n_rep: 7 (14 // 2)
            - Output: [batch, 14, seq_len, head_dim]
            Each of the 2 KV heads is repeated 7 times.
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape

        # If n_rep is 1, no repetition needed (MHA case)
        if n_rep == 1:
            return hidden_states

        # Expand KV heads by repeating each head n_rep times
        # Shape: [batch, num_kv_heads, 1, seq_len, head_dim]
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )

        # Reshape to merge the repetition dimension with the head dimension
        # Shape: [batch, num_kv_heads * n_rep, seq_len, head_dim]
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        position_offset: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_kv_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of Qwen3 attention with RoPE and KV cache support.

        This method implements the complete attention mechanism:
        1. Project hidden states to Q/K/V
        2. Reshape for multi-head attention
        3. Apply RoPE to queries and keys
        4. Concatenate with cached KV if provided
        5. Compute scaled dot-product attention with GQA
        6. Reshape output back to original dimensions
        7. Apply output projection

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size].
            freqs_cis: Precomputed RoPE frequencies of shape [max_seq_len, head_dim // 2].
            position_offset: Position offset for RoPE (used with KV cache). Default: 0.
            kv_cache: Optional tuple of (K, V) cached tensors from previous forward passes.
                     Each tensor has shape [batch_size, num_kv_heads, cached_seq_len, head_dim].
            return_kv_cache: If True, return both output and updated KV cache.

        Returns:
            If return_kv_cache is False:
                Attention output tensor of shape [batch_size, seq_len, hidden_size].
            If return_kv_cache is True:
                Tuple of (output, kv_cache) where kv_cache is (K, V) tuple.
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. Project to Q/K/V
        # Q: [batch_size, seq_len, num_heads_per_partition * head_dim]
        # K/V: [batch_size, seq_len, num_kv_heads_per_partition * head_dim]
        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        # 2. Reshape for multi-head attention
        # Use per-partition head counts for TP
        # Q: [batch_size, seq_len, num_heads_per_partition, head_dim]
        # K/V: [batch_size, seq_len, num_kv_heads_per_partition, head_dim]
        q = q_proj.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k_proj.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)
        v = v_proj.view(batch_size, seq_len, self.num_kv_heads_per_partition, self.head_dim)

        # 3. Apply RoPE to queries and keys
        # RoPE encodes positional information through rotation
        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis, position_offset)

        # 4. Transpose for attention computation
        # Shape: [batch_size, num_heads_per_partition, seq_len, head_dim]
        q_t = q_rope.transpose(1, 2)
        k_t = k_rope.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # 5. Handle KV cache
        # If cache is provided, concatenate cached KV with new KV
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate along sequence dimension
            k_t = torch.cat([k_cache, k_t], dim=2)
            v_t = torch.cat([v_cache, v_t], dim=2)

        # 6. Store updated KV cache if requested
        # Cache shape: [batch_size, num_kv_heads_per_partition, total_seq_len, head_dim]
        new_kv_cache = (k_t, v_t) if return_kv_cache else None

        # 7. Repeat KV heads for GQA
        # Each KV head is repeated num_key_value_groups times to match Q heads
        # Note: num_key_value_groups is the same for TP and non-TP
        k_repeated = self.repeat_kv(k_t, self.num_key_value_groups)
        v_repeated = self.repeat_kv(v_t, self.num_key_value_groups)

        # 8. Compute scaled dot-product attention scores
        # scores = Q @ K^T / sqrt(head_dim)
        # Shape: [batch_size, num_heads_per_partition, seq_len, total_seq_len]
        scores = torch.matmul(q_t, k_repeated.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # 9. Apply softmax to get attention weights
        # Shape: [batch_size, num_heads_per_partition, seq_len, total_seq_len]
        attn_weights = F.softmax(scores, dim=-1)

        # 10. Apply attention weights to values
        # attn_output = attention_weights @ V
        # Shape: [batch_size, num_heads_per_partition, seq_len, head_dim]
        attn_output = torch.matmul(attn_weights, v_repeated)

        # 11. Transpose and reshape back to original dimensions
        # Shape: [batch_size, seq_len, num_heads_per_partition, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # Shape: [batch_size, seq_len, num_heads_per_partition * head_dim]
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads_per_partition * self.head_dim)

        # 12. Apply output projection
        # Project attention output back to hidden dimension
        # Shape: [batch_size, seq_len, hidden_size]
        output = self.o_proj(attn_output)

        # Return output with or without cache
        if return_kv_cache:
            return output, new_kv_cache
        return output
