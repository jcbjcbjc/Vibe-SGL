"""
Radix tree implementation for prefix caching.

This module implements a radix tree data structure for automatic prefix matching
and KV cache reuse in RadixAttention.
"""

import time
import threading
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict


class RadixTreeNode:
    """Node in the radix tree.

    Attributes:
        tokens: Token sequence stored in this node.
        page_ids: Page IDs associated with this token sequence.
        children: Child nodes.
        ref_count: Reference count for this node.
        last_access_time: Timestamp of last access.
    """

    def __init__(self):
        self.tokens: List[int] = []
        self.page_ids: List[int] = []
        self.children: Dict[int, 'RadixTreeNode'] = {}
        self.ref_count: int = 1
        self.last_access_time: float = time.time()


class RadixTree:
    """Radix tree for prefix caching.

    This class implements a radix tree that stores token sequences and their
    associated page IDs for efficient prefix matching and cache reuse.

    Attributes:
        root: Root node of the tree.
        sequences: Mapping from sequence ID to tree path.
        max_sequences: Maximum number of sequences to cache (for LRU eviction).
    """

    def __init__(self, max_sequences: Optional[int] = None):
        """Initialize RadixTree.

        Args:
            max_sequences: Maximum number of sequences to cache. If None, no limit.
        """
        self.root = RadixTreeNode()
        self.sequences: Dict[str, List[RadixTreeNode]] = {}
        self.max_sequences = max_sequences
        self.lru_tracker: OrderedDict[str, float] = OrderedDict()
        self.lock = threading.Lock()

    def insert(self, seq_id: str, tokens: List[int], page_ids: List[int]) -> None:
        """Insert a token sequence into the tree.

        Args:
            seq_id: Unique sequence identifier.
            tokens: List of token IDs.
            page_ids: List of page IDs associated with the tokens.
        """
        with self.lock:
            # Check if we need to evict
            if self.max_sequences and len(self.sequences) >= self.max_sequences:
                if seq_id not in self.sequences:
                    self._evict_lru()

            # Insert into tree
            current = self.root
            path = [current]

            for i, token in enumerate(tokens):
                if token not in current.children:
                    current.children[token] = RadixTreeNode()

                current = current.children[token]
                path.append(current)

            # Store page IDs at leaf
            current.tokens = tokens
            current.page_ids = page_ids
            current.last_access_time = time.time()

            # Track sequence
            self.sequences[seq_id] = path
            self.lru_tracker[seq_id] = time.time()
            self.lru_tracker.move_to_end(seq_id)

    def find_prefix(self, tokens: List[int]) -> Tuple[List[int], int]:
        """Find the longest matching prefix in the tree.

        Args:
            tokens: Token sequence to match.

        Returns:
            Tuple of (matched_page_ids, matched_length).
        """
        with self.lock:
            current = self.root
            matched_pages = []
            matched_len = 0

            for i, token in enumerate(tokens):
                if token not in current.children:
                    break

                current = current.children[token]
                matched_len = i + 1

                # Collect page IDs along the path
                if current.page_ids:
                    matched_pages = current.page_ids.copy()

            return matched_pages, matched_len

    def increment_ref(self, seq_id: str) -> None:
        """Increment reference count for a sequence.

        Args:
            seq_id: Sequence identifier.
        """
        with self.lock:
            if seq_id in self.sequences:
                path = self.sequences[seq_id]
                for node in path:
                    node.ref_count += 1

    def decrement_ref(self, seq_id: str) -> None:
        """Decrement reference count for a sequence.

        Args:
            seq_id: Sequence identifier.
        """
        with self.lock:
            if seq_id in self.sequences:
                path = self.sequences[seq_id]
                for node in path:
                    node.ref_count = max(0, node.ref_count - 1)

    def get_ref_count(self, seq_id: str) -> int:
        """Get reference count for a sequence.

        Args:
            seq_id: Sequence identifier.

        Returns:
            Reference count (0 if sequence not found).
        """
        with self.lock:
            if seq_id in self.sequences:
                path = self.sequences[seq_id]
                if path:
                    return path[-1].ref_count
            return 0

    def remove(self, seq_id: str) -> None:
        """Remove a sequence from the tree.

        Args:
            seq_id: Sequence identifier.
        """
        with self.lock:
            if seq_id in self.sequences:
                # Decrement ref counts
                path = self.sequences[seq_id]
                for node in path:
                    node.ref_count = max(0, node.ref_count - 1)

                # Remove from tracking
                del self.sequences[seq_id]
                if seq_id in self.lru_tracker:
                    del self.lru_tracker[seq_id]

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary with statistics:
            - num_sequences: Number of cached sequences
            - total_nodes: Total number of nodes in tree
        """
        with self.lock:
            total_nodes = self._count_nodes(self.root)

            return {
                "num_sequences": len(self.sequences),
                "total_nodes": total_nodes,
            }

    def _count_nodes(self, node: RadixTreeNode) -> int:
        """Count total nodes in subtree.

        Args:
            node: Root of subtree.

        Returns:
            Number of nodes.
        """
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        return count

    def _evict_lru(self) -> None:
        """Evict least recently used sequence."""
        if not self.lru_tracker:
            return

        # Get LRU sequence
        lru_seq_id = next(iter(self.lru_tracker))

        # Remove it
        self.remove(lru_seq_id)
