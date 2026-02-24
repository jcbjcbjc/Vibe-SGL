"""
Qwen3 model implementation with native TP/EP support.

Components:
- Qwen3Model: Full model with layer-wise TP support
- Qwen3DecoderLayer: Transformer decoder layer
- Qwen3Attention: Multi-head attention with GQA and RoPE
- Qwen3MLP: Feed-forward network with SwiGLU activation
- Weight loading utilities for HuggingFace checkpoints
"""

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.rmsnorm import RMSNorm
from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis, apply_rotary_emb
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention

__all__ = [
    "Qwen3Config",
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "Qwen3Attention",
]
