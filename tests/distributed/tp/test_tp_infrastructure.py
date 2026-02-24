"""
Tests for Tensor Parallelism (TP) infrastructure.

This module tests:
- TP process group initialization with Gloo backend
- Rank and world size assignment
- Weight partitioning utilities
- Communication primitives (all-reduce, broadcast)
"""

import pytest
import torch
import torch.distributed as dist

from tests.utils.distributed import run_distributed_test
from vibe_sgl_lite.distributed.tp.infrastructure import partition_weights


@pytest.mark.distributed
def test_tp_process_group_initialization():
    """
    Test TP process group initialization with Gloo backend.

    Validates:
    - Process group is initialized successfully
    - Backend is set to Gloo for CPU testing
    - All ranks can communicate
    """
    def worker(rank, world_size):
        # Verify process group is initialized
        assert dist.is_initialized(), f"Rank {rank}: Process group not initialized"

        # Verify backend is Gloo
        backend = dist.get_backend()
        assert backend == "gloo", f"Rank {rank}: Expected 'gloo' backend, got '{backend}'"

        # Verify rank and world size
        assert dist.get_rank() == rank, f"Rank mismatch: expected {rank}, got {dist.get_rank()}"
        assert dist.get_world_size() == world_size, f"World size mismatch: expected {world_size}, got {dist.get_world_size()}"

        # Test basic communication with barrier
        dist.barrier()

        return {"rank": rank, "world_size": world_size, "backend": backend}

    # Test with 2 ranks
    results = run_distributed_test(worker, world_size=2, backend="gloo")

    # Verify all ranks completed successfully
    assert len(results) == 2
    assert results[0]["rank"] == 0
    assert results[1]["rank"] == 1
    assert all(r["world_size"] == 2 for r in results)
    assert all(r["backend"] == "gloo" for r in results)


@pytest.mark.distributed
def test_tp_rank_assignment():
    """
    Test rank and world size assignment in TP process group.

    Validates:
    - Each rank receives unique ID from 0 to (world_size - 1)
    - Ranks can identify themselves correctly
    - No rank collisions
    """
    def worker(rank, world_size):
        # Get rank from distributed
        my_rank = dist.get_rank()
        my_world_size = dist.get_world_size()

        # Verify rank is in valid range
        assert 0 <= my_rank < world_size, f"Rank {my_rank} out of range [0, {world_size})"

        # Verify world size matches
        assert my_world_size == world_size, f"World size mismatch: {my_world_size} != {world_size}"

        # Gather all ranks to verify uniqueness
        rank_tensor = torch.tensor([my_rank], dtype=torch.long)
        gathered = [torch.zeros_like(rank_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, rank_tensor)

        # Convert to list of ranks
        all_ranks = [t.item() for t in gathered]

        return {"rank": my_rank, "all_ranks": all_ranks}

    # Test with 4 ranks
    results = run_distributed_test(worker, world_size=4, backend="gloo")

    # Verify all ranks are unique and complete
    assert len(results) == 4
    all_ranks_from_rank0 = results[0]["all_ranks"]
    assert sorted(all_ranks_from_rank0) == [0, 1, 2, 3]

    # Verify each rank sees the same gathered ranks
    for result in results:
        assert sorted(result["all_ranks"]) == [0, 1, 2, 3]


@pytest.mark.distributed
def test_tp_world_size_configuration():
    """
    Test world size configuration for TP.

    Validates:
    - World size equals TP degree
    - Different world sizes work correctly
    - Process group handles various configurations
    """
    def worker(rank, world_size, expected_world_size):
        actual_world_size = dist.get_world_size()
        assert actual_world_size == expected_world_size, \
            f"Rank {rank}: Expected world size {expected_world_size}, got {actual_world_size}"
        return actual_world_size

    # Test with world_size=2
    results = run_distributed_test(
        worker,
        world_size=2,
        backend="gloo",
        test_args=(2,)
    )
    assert all(r == 2 for r in results)

    # Test with world_size=4
    results = run_distributed_test(
        worker,
        world_size=4,
        backend="gloo",
        test_args=(4,)
    )
    assert all(r == 4 for r in results)


def test_partition_weights_column_parallel():
    """
    Test weight partitioning for column parallelism.

    Validates:
    - Weights are split along output dimension (dim=0)
    - Each rank gets equal-sized partition
    - Partitions are contiguous and non-overlapping
    """
    # Create a weight matrix [out_features, in_features]
    weight = torch.randn(128, 64)
    world_size = 4

    # Partition for each rank
    partitions = []
    for rank in range(world_size):
        partition = partition_weights(weight, rank, world_size, dim=0)
        partitions.append(partition)

    # Verify partition sizes
    expected_size = 128 // world_size  # 32
    for partition in partitions:
        assert partition.shape == (expected_size, 64), \
            f"Expected shape ({expected_size}, 64), got {partition.shape}"

    # Verify partitions are contiguous and non-overlapping
    reconstructed = torch.cat(partitions, dim=0)
    assert torch.allclose(reconstructed, weight), "Partitions don't reconstruct original weight"


def test_partition_weights_row_parallel():
    """
    Test weight partitioning for row parallelism.

    Validates:
    - Weights are split along input dimension (dim=1)
    - Each rank gets equal-sized partition
    - Partitions are contiguous and non-overlapping
    """
    # Create a weight matrix [out_features, in_features]
    weight = torch.randn(64, 128)
    world_size = 4

    # Partition for each rank
    partitions = []
    for rank in range(world_size):
        partition = partition_weights(weight, rank, world_size, dim=1)
        partitions.append(partition)

    # Verify partition sizes
    expected_size = 128 // world_size  # 32
    for partition in partitions:
        assert partition.shape == (64, expected_size), \
            f"Expected shape (64, {expected_size}), got {partition.shape}"

    # Verify partitions are contiguous and non-overlapping
    reconstructed = torch.cat(partitions, dim=1)
    assert torch.allclose(reconstructed, weight), "Partitions don't reconstruct original weight"


def test_partition_weights_uneven_split():
    """
    Test weight partitioning with uneven split.

    Validates:
    - Raises error when dimension is not evenly divisible
    - Provides clear error message
    """
    # Create a weight matrix with size not divisible by world_size
    weight = torch.randn(101, 64)  # 101 not divisible by 4
    world_size = 4

    with pytest.raises(ValueError, match="not evenly divisible"):
        partition_weights(weight, rank=0, world_size=world_size, dim=0)


def test_partition_weights_bias():
    """
    Test bias partitioning for column parallelism.

    Validates:
    - 1D bias tensors are partitioned correctly
    - Each rank gets equal-sized partition
    """
    # Create a bias vector
    bias = torch.randn(128)
    world_size = 4

    # Partition for each rank
    partitions = []
    for rank in range(world_size):
        partition = partition_weights(bias, rank, world_size, dim=0)
        partitions.append(partition)

    # Verify partition sizes
    expected_size = 128 // world_size  # 32
    for partition in partitions:
        assert partition.shape == (expected_size,), \
            f"Expected shape ({expected_size},), got {partition.shape}"

    # Verify partitions reconstruct original
    reconstructed = torch.cat(partitions, dim=0)
    assert torch.allclose(reconstructed, bias), "Partitions don't reconstruct original bias"


@pytest.mark.distributed
def test_all_reduce_operation():
    """
    Test all-reduce operation for TP.

    Validates:
    - All-reduce sums tensors across all ranks
    - Result is identical on all ranks
    - Works with different tensor sizes
    """
    def worker(rank, world_size):
        # Create a tensor with rank-specific value
        tensor = torch.ones(10, 10) * (rank + 1)

        # Perform all-reduce (sum)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        # Expected result: sum of all rank values
        # rank 0: 1, rank 1: 2, rank 2: 3, rank 3: 4
        # sum = 1 + 2 + 3 + 4 = 10
        expected = torch.ones(10, 10) * sum(range(1, world_size + 1))

        # Verify result
        assert torch.allclose(tensor, expected), \
            f"Rank {rank}: All-reduce result mismatch"

        return tensor[0, 0].item()

    # Test with 4 ranks
    results = run_distributed_test(worker, world_size=4, backend="gloo")

    # Verify all ranks got the same result
    assert len(results) == 4
    assert all(r == 10.0 for r in results)


@pytest.mark.distributed
def test_all_reduce_with_different_ops():
    """
    Test all-reduce with different reduction operations.

    Validates:
    - SUM operation works correctly
    - MAX operation works correctly
    - MIN operation works correctly
    """
    def worker(rank, world_size, op_name):
        # Create a tensor with rank-specific value
        tensor = torch.ones(5, 5) * (rank + 1)

        # Perform all-reduce with specified operation
        if op_name == "SUM":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            expected_value = sum(range(1, world_size + 1))
        elif op_name == "MAX":
            dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            expected_value = world_size
        elif op_name == "MIN":
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            expected_value = 1
        else:
            raise ValueError(f"Unknown op: {op_name}")

        expected = torch.ones(5, 5) * expected_value

        # Verify result
        assert torch.allclose(tensor, expected), \
            f"Rank {rank}: All-reduce {op_name} result mismatch"

        return tensor[0, 0].item()

    # Test SUM
    results = run_distributed_test(
        worker,
        world_size=4,
        backend="gloo",
        test_args=("SUM",)
    )
    assert all(r == 10.0 for r in results)

    # Test MAX
    results = run_distributed_test(
        worker,
        world_size=4,
        backend="gloo",
        test_args=("MAX",)
    )
    assert all(r == 4.0 for r in results)

    # Test MIN
    results = run_distributed_test(
        worker,
        world_size=4,
        backend="gloo",
        test_args=("MIN",)
    )
    assert all(r == 1.0 for r in results)


@pytest.mark.distributed
def test_broadcast_operation():
    """
    Test broadcast operation for TP.

    Validates:
    - Broadcast sends tensor from source rank to all ranks
    - All ranks receive identical tensor
    - Works with different tensor sizes
    """
    def worker(rank, world_size):
        # Create a tensor with rank-specific value
        if rank == 0:
            # Source rank has specific data
            tensor = torch.arange(10, dtype=torch.float32)
        else:
            # Other ranks start with zeros
            tensor = torch.zeros(10, dtype=torch.float32)

        # Broadcast from rank 0
        dist.broadcast(tensor, src=0)

        # All ranks should now have the same tensor
        expected = torch.arange(10, dtype=torch.float32)

        # Verify result
        assert torch.allclose(tensor, expected), \
            f"Rank {rank}: Broadcast result mismatch"

        return tensor.tolist()

    # Test with 4 ranks
    results = run_distributed_test(worker, world_size=4, backend="gloo")

    # Verify all ranks got the same result
    assert len(results) == 4
    expected = list(range(10))
    for result in results:
        assert result == expected


@pytest.mark.distributed
def test_broadcast_from_different_sources():
    """
    Test broadcast from different source ranks.

    Validates:
    - Broadcast works from any rank as source
    - All ranks receive correct data from specified source
    """
    def worker(rank, world_size, src_rank):
        # Each rank has its own unique value
        tensor = torch.ones(5) * (rank + 1)

        # Broadcast from specified source rank
        dist.broadcast(tensor, src=src_rank)

        # All ranks should now have the source rank's value
        expected = torch.ones(5) * (src_rank + 1)

        # Verify result
        assert torch.allclose(tensor, expected), \
            f"Rank {rank}: Broadcast from rank {src_rank} failed"

        return tensor[0].item()

    # Test broadcast from rank 0
    results = run_distributed_test(
        worker,
        world_size=4,
        backend="gloo",
        test_args=(0,)
    )
    assert all(r == 1.0 for r in results)

    # Test broadcast from rank 2
    results = run_distributed_test(
        worker,
        world_size=4,
        backend="gloo",
        test_args=(2,)
    )
    assert all(r == 3.0 for r in results)


@pytest.mark.distributed
def test_broadcast_weight_distribution():
    """
    Test broadcast for weight distribution in TP.

    Validates:
    - Large weight tensors can be broadcast
    - Useful for distributing model weights from rank 0
    """
    def worker(rank, world_size):
        # Simulate weight loading on rank 0
        if rank == 0:
            weight = torch.randn(128, 64)
        else:
            weight = torch.zeros(128, 64)

        # Broadcast weight from rank 0
        dist.broadcast(weight, src=0)

        # Verify all ranks have non-zero weights
        assert weight.abs().sum() > 0, f"Rank {rank}: Weight is all zeros after broadcast"

        # Return checksum for verification
        return weight.sum().item()

    # Test with 2 ranks
    results = run_distributed_test(worker, world_size=2, backend="gloo")

    # Verify all ranks got the same weights (same checksum)
    assert len(results) == 2
    assert abs(results[0] - results[1]) < 1e-5
