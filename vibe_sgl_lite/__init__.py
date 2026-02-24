"""
vibe_sgl_lite: A lightweight LLM inference engine with advanced optimizations.

This package provides a production-ready inference engine implementing:
- Core inference backbone (model loading, tokenization, generation)
- Paged attention for efficient KV cache management
- RadixAttention for automatic prefix caching
- Continuous batching for dynamic request handling
- Tensor parallelism (TP) and expert parallelism (EP)
- Custom Qwen3 model implementation with native TP/EP support
"""

__version__ = "0.1.0"
__author__ = "vibe-sgl-lite contributors"

# Public API will be exposed here once core components are implemented
__all__ = []
