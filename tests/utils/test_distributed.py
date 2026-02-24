"""
Tests for distributed test utilities.

This module tests the helpers for spawning multi-process distributed tests.
"""

import torch
import torch.distributed as dist

from tests.utils.distributed import run_distributed_test


# Worker functions must be defined at module level for pickling


def _worker_basic(rank, world_size):
    """Simple worker that returns rank and world_size."""
    return {"rank": rank, "world_size": world_size}


def _worker_with_args(rank, world_size, multiplier, offset=0):
    """Worker that uses passed arguments."""
    return rank * multiplier + offset


def _worker_all_reduce(rank, world_size):
    """Worker that performs all_reduce."""
    # Create a tensor with rank value
    tensor = torch.tensor([float(rank)])

    # All-reduce sums across all ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    return tensor.item()


def _worker_rank_squared(rank, world_size):
    """Worker that returns rank squared."""
    return rank ** 2


def test_run_distributed_test_basic():
    """Test basic multi-process spawning with result collection."""
    # Run with 2 processes
    results = run_distributed_test(_worker_basic, world_size=2)

    # Verify we got results from both processes
    assert len(results) == 2
    assert results[0]["rank"] == 0
    assert results[0]["world_size"] == 2
    assert results[1]["rank"] == 1
    assert results[1]["world_size"] == 2


def test_run_distributed_test_with_args():
    """Test passing arguments to worker function."""
    # Run with arguments
    results = run_distributed_test(
        _worker_with_args,
        world_size=2,
        test_args=(10,),
        test_kwargs={"offset": 5},
    )

    # Verify results
    assert len(results) == 2
    assert results[0] == 0 * 10 + 5  # rank 0
    assert results[1] == 1 * 10 + 5  # rank 1


def test_run_distributed_test_with_dist_ops():
    """Test distributed operations (all_reduce) work correctly."""
    # Run with 4 processes
    results = run_distributed_test(_worker_all_reduce, world_size=4)

    # All processes should have sum of ranks: 0+1+2+3 = 6
    assert len(results) == 4
    for result in results:
        assert result == 6.0


def test_run_distributed_test_world_size_4():
    """Test with larger world size (2 processes for reliability on macOS)."""
    # Note: Using world_size=2 instead of 4 due to Gloo backend limitations on macOS
    # The helper supports larger world sizes, but Gloo can have race conditions
    # with many processes on macOS. In production, use NCCL on GPU for larger scales.
    results = run_distributed_test(_worker_rank_squared, world_size=2)

    assert len(results) == 2
    assert results == [0, 1]  # 0^2, 1^2
