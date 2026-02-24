"""
Paged allocator for sequence-based page management.

This module implements a PagedAllocator that manages page allocation for sequences
with page tables, reference counting, and LRU eviction.
"""

import time
from typing import Optional, List, Dict, Set
from collections import OrderedDict

from vibe_sgl_lite.memory.memory_pool import MemoryPool


class PagedAllocator:
    """Paged allocator with page table management.

    This class manages page allocation for sequences, maintaining page tables
    that map logical positions to physical pages. It supports:
    - Page allocation for new sequences
    - Page sharing with reference counting
    - LRU eviction when pool is exhausted
    - Page metadata tracking

    Attributes:
        pool: Underlying memory pool for page storage.
        page_size: Number of tokens per page.
    """

    def __init__(self, pool: MemoryPool, page_size: int) -> None:
        """Initialize PagedAllocator.

        Args:
            pool: Memory pool for page allocation.
            page_size: Number of tokens per page.
        """
        self.pool = pool
        self.page_size = page_size

        # Page tables: seq_id -> list of page IDs
        self.page_tables: Dict[str, List[int]] = {}

        # Reference counting: page_id -> ref_count
        self.ref_counts: Dict[int, int] = {}

        # Page metadata: page_id -> metadata dict
        self.page_metadata: Dict[int, Dict] = {}

        # LRU tracking: seq_id -> last access time
        self.lru_tracker: OrderedDict[str, float] = OrderedDict()

        # Page owners: page_id -> set of seq_ids
        self.page_owners: Dict[int, Set[str]] = {}

    def allocate_sequence(
        self,
        seq_id: str,
        num_tokens: int,
        allow_eviction: bool = False,
    ) -> List[int]:
        """Allocate pages for a new sequence.

        Args:
            seq_id: Unique identifier for the sequence.
            num_tokens: Number of tokens in the sequence.
            allow_eviction: Whether to allow LRU eviction if pool is full.

        Returns:
            List of allocated page IDs.
        """
        # Calculate number of pages needed
        num_pages = (num_tokens + self.page_size - 1) // self.page_size

        # Try to allocate pages
        page_ids = self.pool.allocate_batch(num_pages)

        # If not enough pages and eviction is allowed, evict LRU sequences
        while len(page_ids) < num_pages and allow_eviction:
            if not self._evict_lru_sequence():
                break  # No more sequences to evict
            # Try to allocate remaining pages
            remaining = num_pages - len(page_ids)
            additional_pages = self.pool.allocate_batch(remaining)
            page_ids.extend(additional_pages)

        # Create page table for sequence
        self.page_tables[seq_id] = page_ids

        # Initialize reference counts and metadata
        for page_id in page_ids:
            if page_id not in self.ref_counts:
                self.ref_counts[page_id] = 0
            self.ref_counts[page_id] += 1

            if page_id not in self.page_owners:
                self.page_owners[page_id] = set()
            self.page_owners[page_id].add(seq_id)

            self.page_metadata[page_id] = {
                "page_id": page_id,
                "owner": seq_id,
                "ref_count": self.ref_counts[page_id],
            }

        # Update LRU tracker
        self.lru_tracker[seq_id] = time.time()
        self.lru_tracker.move_to_end(seq_id)

        return page_ids

    def free_sequence(self, seq_id: str) -> None:
        """Free all pages for a sequence.

        Args:
            seq_id: Sequence identifier.
        """
        if seq_id not in self.page_tables:
            return

        page_ids = self.page_tables[seq_id]

        # Decrement reference counts and free pages with ref_count = 0
        for page_id in page_ids:
            if page_id in self.ref_counts:
                self.ref_counts[page_id] -= 1

                # Remove from owners
                if page_id in self.page_owners:
                    self.page_owners[page_id].discard(seq_id)

                # Free page if no more references
                if self.ref_counts[page_id] == 0:
                    self.pool.free(page_id)
                    del self.ref_counts[page_id]
                    if page_id in self.page_metadata:
                        del self.page_metadata[page_id]
                    if page_id in self.page_owners:
                        del self.page_owners[page_id]

        # Remove page table and LRU entry
        del self.page_tables[seq_id]
        if seq_id in self.lru_tracker:
            del self.lru_tracker[seq_id]

    def get_page_table(self, seq_id: str) -> Optional[List[int]]:
        """Get page table for a sequence.

        Args:
            seq_id: Sequence identifier.

        Returns:
            List of page IDs, or None if sequence not found.
        """
        return self.page_tables.get(seq_id)

    def share_pages(self, seq_id: str, page_ids: List[int]) -> None:
        """Share pages with a new sequence (for prefix caching).

        Args:
            seq_id: New sequence identifier.
            page_ids: List of page IDs to share.
        """
        # Create page table for new sequence
        self.page_tables[seq_id] = page_ids.copy()

        # Increment reference counts
        for page_id in page_ids:
            if page_id in self.ref_counts:
                self.ref_counts[page_id] += 1
            else:
                self.ref_counts[page_id] = 1

            if page_id not in self.page_owners:
                self.page_owners[page_id] = set()
            self.page_owners[page_id].add(seq_id)

            # Update metadata
            if page_id in self.page_metadata:
                self.page_metadata[page_id]["ref_count"] = self.ref_counts[page_id]

        # Update LRU tracker
        self.lru_tracker[seq_id] = time.time()
        self.lru_tracker.move_to_end(seq_id)

    def get_ref_count(self, page_id: int) -> int:
        """Get reference count for a page.

        Args:
            page_id: Page identifier.

        Returns:
            Reference count (0 if page not tracked).
        """
        return self.ref_counts.get(page_id, 0)

    def get_page_metadata(self, page_id: int) -> Optional[Dict]:
        """Get metadata for a page.

        Args:
            page_id: Page identifier.

        Returns:
            Metadata dictionary, or None if page not found.
        """
        return self.page_metadata.get(page_id)

    def extend_sequence(self, seq_id: str, num_tokens: int) -> List[int]:
        """Extend a sequence with additional pages.

        Args:
            seq_id: Sequence identifier.
            num_tokens: Number of additional tokens.

        Returns:
            List of newly allocated page IDs.
        """
        if seq_id not in self.page_tables:
            return []

        # Calculate number of additional pages needed
        num_pages = (num_tokens + self.page_size - 1) // self.page_size

        # Allocate new pages
        new_page_ids = self.pool.allocate_batch(num_pages)

        # Add to page table
        self.page_tables[seq_id].extend(new_page_ids)

        # Initialize reference counts and metadata
        for page_id in new_page_ids:
            if page_id not in self.ref_counts:
                self.ref_counts[page_id] = 0
            self.ref_counts[page_id] += 1

            if page_id not in self.page_owners:
                self.page_owners[page_id] = set()
            self.page_owners[page_id].add(seq_id)

            self.page_metadata[page_id] = {
                "page_id": page_id,
                "owner": seq_id,
                "ref_count": self.ref_counts[page_id],
            }

        # Update LRU tracker
        self.lru_tracker[seq_id] = time.time()
        self.lru_tracker.move_to_end(seq_id)

        return new_page_ids

    def get_stats(self) -> Dict[str, int]:
        """Get allocator statistics.

        Returns:
            Dictionary with statistics:
            - num_sequences: Number of active sequences
            - total_pages_allocated: Total pages allocated across all sequences
        """
        total_pages = sum(len(pages) for pages in self.page_tables.values())

        return {
            "num_sequences": len(self.page_tables),
            "total_pages_allocated": total_pages,
        }

    def _evict_lru_sequence(self) -> bool:
        """Evict the least recently used sequence.

        Returns:
            True if a sequence was evicted, False if no sequences to evict.
        """
        if not self.lru_tracker:
            return False

        # Get LRU sequence (first in OrderedDict)
        lru_seq_id = next(iter(self.lru_tracker))

        # Free the sequence
        self.free_sequence(lru_seq_id)

        return True

    def touch_sequence(self, seq_id: str) -> None:
        """Update LRU timestamp for a sequence.

        Args:
            seq_id: Sequence identifier.
        """
        if seq_id in self.lru_tracker:
            self.lru_tracker[seq_id] = time.time()
            self.lru_tracker.move_to_end(seq_id)
