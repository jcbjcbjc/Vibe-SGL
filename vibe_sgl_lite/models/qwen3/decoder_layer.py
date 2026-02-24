"""
Qwen3 decoder layer implementation.

This module implements a complete Qwen3 decoder layer that combines:
- Input layer normalization (RMSNorm)
- Self-attention mechanism with RoPE and GQA
- Post-attention layer normalization (RMSNorm)
- Feed-forward network with SwiGLU activation
- Residual connections around attention and FFN

The decoder layer follows the pre-norm architecture:
    x = x + attention(norm(x))
    x = x + ffn(norm(x))

This architecture applies normalization before each sub-layer (attention and FFN)
rather than after, which improves training stability and gradient flow.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.rmsnorm import RMSNorm
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention
from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 decoder layer with pre-norm architecture.

    This class implements a complete transformer decoder layer used in Qwen3,
    combining self-attention and feed-forward network with residual connections
    and layer normalization.

    The layer follows the pre-norm architecture where normalization is applied
    before each sub-layer:
        1. x = x + self_attention(input_layernorm(x))
        2. x = x + ffn(post_attention_layernorm(x))

    Attributes:
        config: Qwen3 model configuration.
        input_layernorm: RMSNorm layer applied before self-attention.
        self_attn: Multi-head self-attention layer with GQA and RoPE.
        post_attention_layernorm: RMSNorm layer applied before FFN.
        mlp: Feed-forward network with SwiGLU activation.
    """

    def __init__(self, config: Qwen3Config, tp_degree: int = 1, rank: int = 0) -> None:
        """Initialize Qwen3DecoderLayer.

        Args:
            config: Qwen3 model configuration containing layer parameters.
            tp_degree: Tensor parallelism degree (default: 1, no TP).
            rank: Current TP rank (default: 0).
        """
        super().__init__()

        self.config = config
        self.tp_degree = tp_degree
        self.rank = rank

        # Input layer normalization (applied before self-attention)
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Self-attention layer with GQA and RoPE (with TP support)
        self.self_attn = Qwen3Attention(config, tp_degree=tp_degree, rank=rank)

        # Post-attention layer normalization (applied before FFN)
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Feed-forward network with SwiGLU activation (with TP support)
        self.mlp = Qwen3FFN(config, tp_degree=tp_degree, rank=rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        start_pos: Optional[int] = None,
        cache_k: Optional[torch.Tensor] = None,
        cache_v: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Forward pass of Qwen3 decoder layer.

        This method implements the complete decoder layer computation with
        pre-norm architecture and residual connections:
        1. Apply input layer norm
        2. Compute self-attention with residual connection
        3. Apply post-attention layer norm
        4. Compute FFN with residual connection

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_len, hidden_size].
            freqs_cis: Precomputed RoPE frequencies of shape [max_seq_len, head_dim // 2].
            start_pos: Starting position for RoPE (used with KV cache). If provided, KV cache will be returned.
            cache_k: Optional cached key tensor from previous forward passes.
                    Shape: [batch_size, seq_len, num_kv_heads, head_dim].
            cache_v: Optional cached value tensor from previous forward passes.
                    Shape: [batch_size, seq_len, num_kv_heads, head_dim].

        Returns:
            If cache_k and cache_v are None:
                Output tensor of shape [batch_size, seq_len, hidden_size].
            If cache_k and cache_v are provided or start_pos is provided:
                Tuple of (output, new_cache_k, new_cache_v) where:
                - output: [batch_size, seq_len, hidden_size]
                - new_cache_k: [batch_size, total_seq_len, num_kv_heads, head_dim]
                - new_cache_v: [batch_size, total_seq_len, num_kv_heads, head_dim]
        """
        # 1. Apply input layer normalization
        # Pre-norm: normalize before attention
        normed_hidden_states = self.input_layernorm(hidden_states)

        # 2. Self-attention with residual connection
        # Prepare KV cache for attention layer
        # Attention layer expects cache in shape [batch, num_kv_heads, seq_len, head_dim]
        # But decoder layer API uses [batch, seq_len, num_kv_heads, head_dim]
        # So we need to transpose when passing to/from attention
        kv_cache = None
        if cache_k is not None and cache_v is not None:
            # Transpose from [batch, seq_len, num_kv_heads, head_dim] to [batch, num_kv_heads, seq_len, head_dim]
            cache_k_transposed = cache_k.transpose(1, 2)
            cache_v_transposed = cache_v.transpose(1, 2)
            kv_cache = (cache_k_transposed, cache_v_transposed)

        # Determine if we need to return KV cache
        # Return cache if start_pos is explicitly provided (for both prefill and decode phases)
        return_kv_cache = start_pos is not None

        # Use start_pos if provided, otherwise default to 0
        position_offset = start_pos if start_pos is not None else 0

        # Compute attention
        attn_output, new_kv_cache = self.self_attn(
            normed_hidden_states,
            freqs_cis=freqs_cis,
            position_offset=position_offset,
            kv_cache=kv_cache,
            return_kv_cache=True,
        )
        # Transpose cache back from [batch, num_kv_heads, seq_len, head_dim] to [batch, seq_len, num_kv_heads, head_dim]
        new_cache_k, new_cache_v = new_kv_cache
        new_cache_k = new_cache_k.transpose(1, 2)
        new_cache_v = new_cache_v.transpose(1, 2)

        # Apply residual connection
        hidden_states = hidden_states + attn_output

        # 3. Apply post-attention layer normalization
        # Pre-norm: normalize before FFN
        normed_hidden_states = self.post_attention_layernorm(hidden_states)

        # 4. Feed-forward network with residual connection
        ffn_output = self.mlp(normed_hidden_states)

        # Apply residual connection
        hidden_states = hidden_states + ffn_output

        # Return output with or without cache
        if return_kv_cache:
            return hidden_states, new_cache_k, new_cache_v
        return hidden_states
