"""
Tests for RadixAttention - Radix Tree implementation.

This module tests the radix tree data structure for automatic prefix matching
and KV cache reuse.

Following TDD: These tests are written before implementing the radix tree.
"""

import pytest
import torch


@pytest.mark.unit
def test_radix_tree_node_structure() -> None:
    """Test RadixTreeNode structure."""
    from vibe_sgl_lite.cache.radix_tree import RadixTreeNode

    node = RadixTreeNode()

    assert hasattr(node, "tokens")
    assert hasattr(node, "page_ids")
    assert hasattr(node, "children")
    assert hasattr(node, "ref_count")
    assert hasattr(node, "last_access_time")


@pytest.mark.unit
def test_radix_tree_insert_sequence() -> None:
    """Test inserting token sequences into tree."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree

    tree = RadixTree()

    # Insert a sequence
    tokens = [1, 2, 3, 4, 5]
    page_ids = [0, 1]
    seq_id = "seq_1"

    tree.insert(seq_id, tokens, page_ids)

    # Verify insertion
    assert tree.root is not None
    assert len(tree.sequences) == 1


@pytest.mark.unit
def test_radix_tree_prefix_matching() -> None:
    """Test prefix matching (exact and partial)."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree

    tree = RadixTree()

    # Insert sequences
    tokens1 = [1, 2, 3, 4, 5]
    page_ids1 = [0, 1]
    tree.insert("seq_1", tokens1, page_ids1)

    # Find exact match
    matched_pages, matched_len = tree.find_prefix([1, 2, 3, 4, 5])
    assert matched_len == 5
    assert matched_pages == page_ids1

    # Find partial match
    matched_pages, matched_len = tree.find_prefix([1, 2, 3])
    assert matched_len == 3

    # No match
    matched_pages, matched_len = tree.find_prefix([10, 20, 30])
    assert matched_len == 0


@pytest.mark.unit
def test_radix_tree_reference_counting() -> None:
    """Test reference counting on tree nodes."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree

    tree = RadixTree()

    tokens = [1, 2, 3]
    page_ids = [0]

    # Insert and increment ref count
    tree.insert("seq_1", tokens, page_ids)
    tree.increment_ref("seq_1")

    # Check ref count
    ref_count = tree.get_ref_count("seq_1")
    assert ref_count == 2  # Initial + increment

    # Decrement
    tree.decrement_ref("seq_1")
    ref_count = tree.get_ref_count("seq_1")
    assert ref_count == 1


@pytest.mark.unit
def test_radix_tree_lru_eviction() -> None:
    """Test LRU eviction of tree branches."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree

    tree = RadixTree(max_sequences=3)

    # Insert sequences
    for i in range(4):
        tokens = [i, i+1, i+2]
        page_ids = [i]
        tree.insert(f"seq_{i}", tokens, page_ids)

    # Should have evicted oldest (seq_0)
    assert len(tree.sequences) == 3
    assert "seq_0" not in tree.sequences


@pytest.mark.unit
def test_radix_tree_shared_prefix() -> None:
    """Test handling shared prefixes between sequences."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree

    tree = RadixTree()

    # Insert sequences with shared prefix
    tokens1 = [1, 2, 3, 4, 5]
    tokens2 = [1, 2, 3, 6, 7]
    page_ids1 = [0, 1]
    page_ids2 = [0, 2]  # Shares page 0

    tree.insert("seq_1", tokens1, page_ids1)
    tree.insert("seq_2", tokens2, page_ids2)

    # Both should share prefix [1, 2, 3]
    matched_pages1, matched_len1 = tree.find_prefix([1, 2, 3])
    assert matched_len1 == 3

    # Full sequences should be different
    matched_pages2, matched_len2 = tree.find_prefix([1, 2, 3, 4, 5])
    assert matched_len2 == 5


@pytest.mark.unit
def test_radix_tree_remove_sequence() -> None:
    """Test removing sequences from tree."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree

    tree = RadixTree()

    tokens = [1, 2, 3]
    page_ids = [0]
    tree.insert("seq_1", tokens, page_ids)

    assert "seq_1" in tree.sequences

    tree.remove("seq_1")

    assert "seq_1" not in tree.sequences


@pytest.mark.unit
def test_radix_tree_thread_safety() -> None:
    """Test thread-safe tree operations."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree
    import threading

    tree = RadixTree()
    errors = []

    def insert_worker(worker_id):
        try:
            for i in range(10):
                tokens = [worker_id, i]
                page_ids = [worker_id * 10 + i]
                tree.insert(f"seq_{worker_id}_{i}", tokens, page_ids)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=insert_worker, args=(i,)) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(tree.sequences) == 30


@pytest.mark.unit
def test_radix_tree_cache_statistics() -> None:
    """Test cache statistics tracking."""
    from vibe_sgl_lite.cache.radix_tree import RadixTree

    tree = RadixTree()

    # Insert sequences
    for i in range(5):
        tokens = [i, i+1]
        page_ids = [i]
        tree.insert(f"seq_{i}", tokens, page_ids)

    stats = tree.get_stats()

    assert stats["num_sequences"] == 5
    assert stats["total_nodes"] > 0
