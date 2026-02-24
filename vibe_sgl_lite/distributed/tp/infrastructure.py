"""
Tensor Parallelism (TP) infrastructure implementation.

This module provides:
- Process group initialization for TP
- Rank and world size management
- Weight partitioning utilities
- Communication wrappers (all-reduce, broadcast)
"""

from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist


# Global TP state
_TP_GROUP: Optional[dist.ProcessGroup] = None
_TP_RANK: Optional[int] = None
_TP_WORLD_SIZE: Optional[int] = None


def init_tp_process_group(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    timeout_seconds: int = 30,
) -> dist.ProcessGroup:
    """
    Initialize tensor parallelism process group.

    This function:
    - Initializes torch.distributed process group for TP
    - Sets up Gloo backend for CPU or NCCL for GPU
    - Assigns rank and world size for TP workers
    - Returns the process group for TP communication

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of TP workers (TP degree)
        backend: torch.distributed backend ("gloo" for CPU, "nccl" for GPU)
        timeout_seconds: Timeout for distributed operations (default: 30)

    Returns:
        ProcessGroup: Initialized TP process group

    Raises:
        RuntimeError: If process group is already initialized
        ValueError: If rank or world_size is invalid

    Example:
        # Initialize TP with 2 workers
        tp_group = init_tp_process_group(rank=0, world_size=2, backend="gloo")

        # Now can use TP communication
        tensor = torch.randn(10, 10)
        dist.all_reduce(tensor, group=tp_group)
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE

    # Validate inputs
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}")
    if world_size <= 0:
        raise ValueError(f"Invalid world_size {world_size}, must be > 0")

    # Check if already initialized
    if _TP_GROUP is not None:
        raise RuntimeError("TP process group already initialized. Call cleanup_tp_process_group() first.")

    # Initialize process group if not already done
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_seconds),
        )
        _TP_GROUP = dist.group.WORLD
    else:
        # If already initialized, create a new group (for hybrid parallelism)
        ranks = list(range(world_size))
        _TP_GROUP = dist.new_group(ranks=ranks, backend=backend)

    # Store TP state
    _TP_RANK = rank
    _TP_WORLD_SIZE = world_size

    return _TP_GROUP


def get_tp_group() -> Optional[dist.ProcessGroup]:
    """
    Get the current TP process group.

    Returns:
        ProcessGroup or None: Current TP process group, or None if not initialized
    """
    return _TP_GROUP


def get_tp_rank() -> int:
    """
    Get the current TP rank.

    Returns:
        int: Current TP rank (0 to world_size-1)

    Raises:
        RuntimeError: If TP is not initialized
    """
    if _TP_RANK is None:
        raise RuntimeError("TP not initialized. Call init_tp_process_group() first.")
    return _TP_RANK


def get_tp_world_size() -> int:
    """
    Get the current TP world size.

    Returns:
        int: Current TP world size (TP degree)

    Raises:
        RuntimeError: If TP is not initialized
    """
    if _TP_WORLD_SIZE is None:
        raise RuntimeError("TP not initialized. Call init_tp_process_group() first.")
    return _TP_WORLD_SIZE


def is_tp_initialized() -> bool:
    """
    Check if TP is initialized.

    Returns:
        bool: True if TP is initialized, False otherwise
    """
    return _TP_GROUP is not None


def cleanup_tp_process_group() -> None:
    """
    Clean up TP process group.

    This function:
    - Destroys the TP process group
    - Resets TP state variables
    - Should be called in teardown or finally block

    Example:
        try:
            init_tp_process_group(rank=0, world_size=2)
            # Use TP
        finally:
            cleanup_tp_process_group()
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE

    if _TP_GROUP is not None:
        # Only destroy if we're using a custom group (not WORLD)
        if _TP_GROUP != dist.group.WORLD and dist.is_initialized():
            dist.destroy_process_group(_TP_GROUP)

    # Reset state
    _TP_GROUP = None
    _TP_RANK = None
    _TP_WORLD_SIZE = None


def partition_weights(
    weight: torch.Tensor,
    rank: int,
    world_size: int,
    dim: int = 0,
) -> torch.Tensor:
    """
    Partition weight tensor for tensor parallelism.

    This function:
    - Splits weight tensor along specified dimension
    - Returns the partition for the given rank
    - Used for column parallelism (dim=0) and row parallelism (dim=1)

    Args:
        weight: Weight tensor to partition (e.g., [out_features, in_features])
        rank: Process rank (0 to world_size-1)
        world_size: Total number of TP workers
        dim: Dimension to partition along (0 for column, 1 for row)

    Returns:
        torch.Tensor: Partitioned weight for this rank

    Raises:
        ValueError: If dimension is not evenly divisible by world_size

    Example:
        # Column parallelism: split output dimension
        weight = torch.randn(128, 64)  # [out_features, in_features]
        partition = partition_weights(weight, rank=0, world_size=4, dim=0)
        # partition.shape = [32, 64]

        # Row parallelism: split input dimension
        weight = torch.randn(64, 128)  # [out_features, in_features]
        partition = partition_weights(weight, rank=0, world_size=4, dim=1)
        # partition.shape = [64, 32]
    """
    # Validate inputs
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid rank {rank} for world_size {world_size}")
    if dim < 0 or dim >= weight.ndim:
        raise ValueError(f"Invalid dim {dim} for tensor with {weight.ndim} dimensions")

    # Check if dimension is evenly divisible
    dim_size = weight.shape[dim]
    if dim_size % world_size != 0:
        raise ValueError(
            f"Dimension {dim} with size {dim_size} is not evenly divisible by world_size {world_size}"
        )

    # Calculate partition size and offset
    partition_size = dim_size // world_size
    offset = rank * partition_size

    # Slice along the specified dimension
    if dim == 0:
        return weight[offset:offset + partition_size, ...].contiguous()
    elif dim == 1:
        return weight[:, offset:offset + partition_size, ...].contiguous()
    else:
        # For higher dimensions, use narrow
        return weight.narrow(dim, offset, partition_size).contiguous()


def all_reduce_wrapper(
    tensor: torch.Tensor,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[dist.Work]:
    """
    Wrapper for torch.distributed.all_reduce operation.

    This function:
    - Performs all-reduce across all ranks in the group
    - Supports different reduction operations (SUM, MAX, MIN, etc.)
    - Can be used for row parallelism output aggregation
    - Optionally performs async operation

    Args:
        tensor: Tensor to reduce (modified in-place)
        op: Reduction operation (default: SUM)
        group: Process group (default: TP group or WORLD)
        async_op: Whether to perform async operation (default: False)

    Returns:
        Work handle if async_op=True, None otherwise

    Example:
        # Synchronous all-reduce
        tensor = torch.randn(10, 10)
        all_reduce_wrapper(tensor, op=dist.ReduceOp.SUM)

        # Async all-reduce
        work = all_reduce_wrapper(tensor, async_op=True)
        # Do other work...
        work.wait()
    """
    if group is None:
        group = _TP_GROUP if _TP_GROUP is not None else dist.group.WORLD

    return dist.all_reduce(tensor, op=op, group=group, async_op=async_op)


def broadcast_wrapper(
    tensor: torch.Tensor,
    src: int = 0,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Optional[dist.Work]:
    """
    Wrapper for torch.distributed.broadcast operation.

    This function:
    - Broadcasts tensor from source rank to all ranks in the group
    - Useful for distributing weights from rank 0 to all workers
    - Can be used for shared data distribution
    - Optionally performs async operation

    Args:
        tensor: Tensor to broadcast (modified in-place on non-source ranks)
        src: Source rank to broadcast from (default: 0)
        group: Process group (default: TP group or WORLD)
        async_op: Whether to perform async operation (default: False)

    Returns:
        Work handle if async_op=True, None otherwise

    Example:
        # Rank 0 loads weights
        if rank == 0:
            weight = load_weights()
        else:
            weight = torch.zeros_like(...)

        # Broadcast to all ranks
        broadcast_wrapper(weight, src=0)
    """
    if group is None:
        group = _TP_GROUP if _TP_GROUP is not None else dist.group.WORLD

    return dist.broadcast(tensor, src=src, group=group, async_op=async_op)
