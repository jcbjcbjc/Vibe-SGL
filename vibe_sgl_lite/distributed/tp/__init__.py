"""
Tensor Parallelism (TP) implementation.

Provides:
- ColumnParallelLinear: Column-wise weight partitioning
- RowParallelLinear: Row-wise weight partitioning with all-reduce
- Process group initialization for TP
- Weight partitioning utilities
- Communication wrappers (all-reduce, broadcast)
"""

from .infrastructure import (
    all_reduce_wrapper,
    broadcast_wrapper,
    cleanup_tp_process_group,
    get_tp_group,
    get_tp_rank,
    get_tp_world_size,
    init_tp_process_group,
    is_tp_initialized,
    partition_weights,
)
from .parallel_linear import ColumnParallelLinear, RowParallelLinear

__all__ = [
    "init_tp_process_group",
    "get_tp_group",
    "get_tp_rank",
    "get_tp_world_size",
    "is_tp_initialized",
    "cleanup_tp_process_group",
    "partition_weights",
    "all_reduce_wrapper",
    "broadcast_wrapper",
    "ColumnParallelLinear",
    "RowParallelLinear",
]
