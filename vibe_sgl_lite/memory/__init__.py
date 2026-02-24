"""
Memory management subsystem for efficient KV cache handling.

Provides:
- MemoryPool: Manages free pages with LRU eviction
- PagedAllocator: Allocates/deallocates pages per request
- PageTable: Maps logical to physical pages
- Page: Page data structure and metadata
"""

from vibe_sgl_lite.memory.memory_pool import MemoryPool

__all__ = ["MemoryPool"]
