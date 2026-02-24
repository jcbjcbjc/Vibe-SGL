"""
RadixCache implementation for prefix caching integration.
"""

from typing import List, Tuple, Dict
from vibe_sgl_lite.cache.radix_tree import RadixTree


class RadixCache:
    """Cache manager using radix tree for prefix matching."""

    def __init__(self):
        self.tree = RadixTree()
        self.hits = 0
        self.misses = 0

    def insert(self, seq_id: str, tokens: List[int], page_ids: List[int]) -> None:
        """Insert sequence into cache."""
        self.tree.insert(seq_id, tokens, page_ids)

    def lookup(self, tokens: List[int]) -> Tuple[List[int], int]:
        """Lookup prefix in cache."""
        matched_pages, matched_len = self.tree.find_prefix(tokens)

        if matched_len > 0:
            self.hits += 1
        else:
            self.misses += 1

        return matched_pages, matched_len

    def contains(self, seq_id: str) -> bool:
        """Check if sequence is cached."""
        return seq_id in self.tree.sequences

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            **self.tree.get_stats()
        }
