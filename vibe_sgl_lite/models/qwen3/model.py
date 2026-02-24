"""
Qwen3 model implementation.

This module implements the complete Qwen3 model that combines:
- Token embedding layer
- Stack of decoder layers
- Final layer normalization

The model implements the standard transformer decoder architecture.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Union

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.embedding import Qwen3Embedding
from vibe_sgl_lite.models.qwen3.decoder_layer import Qwen3DecoderLayer
from vibe_sgl_lite.models.qwen3.rmsnorm import RMSNorm
from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis


class Qwen3Model(nn.Module):
    """Qwen3 transformer model.

    This class implements the complete Qwen3 model architecture consisting of:
    1. Token embedding layer
    2. Stack of decoder layers (attention + FFN)
    3. Final layer normalization

    The model processes token IDs through embeddings, applies multiple transformer
    decoder layers, and outputs hidden states.

    Attributes:
        config: Qwen3 model configuration.
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final RMSNorm layer.
        freqs_cis: Precomputed RoPE frequencies.
    """

    def __init__(self, config: Qwen3Config, tp_degree: int = 1, rank: int = 0) -> None:
        """Initialize Qwen3Model.

        Args:
            config: Qwen3 model configuration containing model parameters.
            tp_degree: Tensor parallelism degree (default: 1, no TP).
            rank: Current TP rank (default: 0).
        """
        super().__init__()

        self.config = config
        self.tp_degree = tp_degree
        self.rank = rank

        # Token embedding layer (replicated across all TP ranks)
        self.embed_tokens = Qwen3Embedding(config)

        # Stack of decoder layers (with TP support)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, tp_degree=tp_degree, rank=rank)
            for _ in range(config.num_hidden_layers)
        ])

        # Final layer normalization (replicated across all TP ranks)
        self.norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        # Precompute RoPE frequencies
        head_dim = config.hidden_size // config.num_attention_heads
        self.freqs_cis = precompute_freqs_cis(
            dim=head_dim,
            end=config.max_position_embeddings,
            theta=config.rope_theta,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: Optional[int] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        return_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """Forward pass of Qwen3 model.

        This method implements the complete model forward pass:
        1. Convert token IDs to embeddings
        2. Apply each decoder layer sequentially
        3. Apply final layer normalization

        Args:
            input_ids: Token IDs tensor of shape [batch_size, seq_len].
            start_pos: Starting position for RoPE (used with KV cache).
                      If provided, KV cache will be returned.
            kv_caches: Optional list of KV caches from previous forward passes.
                      Each element is a tuple of (cache_k, cache_v) for one layer.
                      Shape: [batch_size, seq_len, num_kv_heads, head_dim].
            return_kv_cache: Whether to return KV caches.

        Returns:
            If return_kv_cache is False:
                Hidden states tensor of shape [batch_size, seq_len, hidden_size].
            If return_kv_cache is True:
                Tuple of (hidden_states, kv_caches) where:
                - hidden_states: [batch_size, seq_len, hidden_size]
                - kv_caches: List of (cache_k, cache_v) tuples for each layer
        """
        # 1. Token embedding
        hidden_states = self.embed_tokens(input_ids)

        # Move freqs_cis to same device as input
        freqs_cis = self.freqs_cis.to(hidden_states.device)

        # 2. Apply decoder layers
        new_kv_caches = []

        for layer_idx, decoder_layer in enumerate(self.layers):
            # Get KV cache for this layer if provided
            layer_kv_cache = None
            if kv_caches is not None and layer_idx < len(kv_caches):
                layer_kv_cache = kv_caches[layer_idx]

            # Apply decoder layer
            if return_kv_cache or start_pos is not None:
                # Return KV cache
                cache_k = layer_kv_cache[0] if layer_kv_cache is not None else None
                cache_v = layer_kv_cache[1] if layer_kv_cache is not None else None

                hidden_states, new_cache_k, new_cache_v = decoder_layer(
                    hidden_states,
                    freqs_cis=freqs_cis,
                    start_pos=start_pos,
                    cache_k=cache_k,
                    cache_v=cache_v,
                )
                new_kv_caches.append((new_cache_k, new_cache_v))
            else:
                # No KV cache
                hidden_states = decoder_layer(
                    hidden_states,
                    freqs_cis=freqs_cis,
                )

        # 3. Final layer normalization
        hidden_states = self.norm(hidden_states)

        # Return output with or without cache
        if return_kv_cache or start_pos is not None:
            return hidden_states, new_kv_caches
        return hidden_states
