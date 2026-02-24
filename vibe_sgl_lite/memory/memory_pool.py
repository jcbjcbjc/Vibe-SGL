"""
Memory pool for page-based KV cache management.

This module implements a MemoryPool that manages a pool of fixed-size pages
for KV cache allocation. The pool pre-allocates pages and manages them with
a free list for efficient allocation and deallocation.

The pool supports:
- Fixed-size page allocation (e.g., 16 tokens per page)
- Batch allocation and deallocation
- Memory statistics tracking
- Thread-safe operations
- Allocation failure handling when pool is exhausted
"""

import torch
import threading
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class MemoryStats:
    """Statistics about memory pool usage."""
    total: int
    used: int
    free: int
    utilization: float


class MemoryPool:
    """Memory pool for managing fixed-size pages.

    This class implements a pool of pre-allocated pages for KV cache storage.
    Pages are managed with a free list for efficient allocation and deallocation.

    Attributes:
        num_pages: Total number of pages in the pool.
        page_size: Number of tokens per page.
        num_layers: Number of transformer layers.
        num_kv_heads: Number of key/value heads.
        head_dim: Dimension of each attention head.
        device: Device for page storage (cpu or cuda).
        dtype: Data type for page storage.
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize MemoryPool with pre-allocated pages.

        Args:
            num_pages: Total number of pages to pre-allocate.
            page_size: Number of tokens per page.
            num_layers: Number of transformer layers.
            num_kv_heads: Number of key/value heads.
            head_dim: Dimension of each attention head.
            device: Device for page storage.
            dtype: Data type for page storage.
        """
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Pre-allocate all pages
        # Shape: [num_pages, num_layers, 2, page_size, num_kv_heads, head_dim]
        # The "2" dimension is for key and value
        self.pages = torch.zeros(
            num_pages,
            num_layers,
            2,  # key and value
            page_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )

        # Free list: list of available page IDs
        self.free_list = list(range(num_pages))

        # Track allocated pages
        self.allocated_pages = set()

        # Thread safety lock
        self.lock = threading.Lock()

    def allocate(self) -> Optional[int]:
        """Allocate a single page from the pool.

        Returns:
            Page ID if allocation successful, None if pool is exhausted.
        """
        with self.lock:
            if not self.free_list:
                return None

            page_id = self.free_list.pop(0)
            self.allocated_pages.add(page_id)
            return page_id

    def allocate_batch(self, num_pages: int) -> List[int]:
        """Allocate multiple pages from the pool.

        Args:
            num_pages: Number of pages to allocate.

        Returns:
            List of allocated page IDs. May be shorter than requested if
            pool doesn't have enough free pages.
        """
        with self.lock:
            # Allocate as many as available (up to requested amount)
            num_to_allocate = min(num_pages, len(self.free_list))
            page_ids = []

            for _ in range(num_to_allocate):
                if self.free_list:
                    page_id = self.free_list.pop(0)
                    self.allocated_pages.add(page_id)
                    page_ids.append(page_id)

            return page_ids

    def free(self, page_id: int) -> None:
        """Free a single page back to the pool.

        Args:
            page_id: ID of the page to free.
        """
        with self.lock:
            if page_id in self.allocated_pages:
                self.allocated_pages.remove(page_id)
                self.free_list.append(page_id)

                # Zero out the page data for reuse
                self.pages[page_id].zero_()

    def free_batch(self, page_ids: List[int]) -> None:
        """Free multiple pages back to the pool.

        Args:
            page_ids: List of page IDs to free.
        """
        with self.lock:
            for page_id in page_ids:
                if page_id in self.allocated_pages:
                    self.allocated_pages.remove(page_id)
                    self.free_list.append(page_id)

                    # Zero out the page data for reuse
                    self.pages[page_id].zero_()

    def get_stats(self) -> Dict[str, float]:
        """Get memory pool statistics.

        Returns:
            Dictionary with keys:
            - total: Total number of pages
            - used: Number of allocated pages
            - free: Number of free pages
            - utilization: Fraction of pages in use (0.0 to 1.0)
        """
        with self.lock:
            used = len(self.allocated_pages)
            free = len(self.free_list)
            utilization = used / self.num_pages if self.num_pages > 0 else 0.0

            return {
                "total": self.num_pages,
                "used": used,
                "free": free,
                "utilization": utilization,
            }

    def get_page(self, page_id: int) -> torch.Tensor:
        """Get the tensor for a specific page.

        Args:
            page_id: ID of the page to retrieve.

        Returns:
            Page tensor of shape [num_layers, 2, page_size, num_kv_heads, head_dim].
        """
        return self.pages[page_id]

    def reset(self) -> None:
        """Reset the pool to initial state (all pages free)."""
        with self.lock:
            self.free_list = list(range(self.num_pages))
            self.allocated_pages.clear()
            self.pages.zero_()
