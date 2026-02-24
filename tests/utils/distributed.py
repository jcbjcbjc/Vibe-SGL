"""
Utilities for spawning and managing multi-process distributed tests.

This module provides helpers for testing tensor parallelism (TP) and expert
parallelism (EP) using torch.distributed with multiple processes on CPU.
"""

import os
import pickle
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run_distributed_test(
    test_func: Callable,
    world_size: int,
    backend: str = "gloo",
    test_args: Optional[Tuple] = None,
    test_kwargs: Optional[Dict[str, Any]] = None,
    master_port: Optional[str] = None,
) -> List[Any]:
    """
    Spawn multiple processes to run a distributed test function.

    This helper:
    - Spawns `world_size` processes using torch.multiprocessing
    - Initializes torch.distributed with specified backend (default: Gloo for CPU)
    - Runs the test function in each process with proper rank assignment
    - Collects and returns results from all processes
    - Handles cleanup and error propagation

    Args:
        test_func: Function to run in each process. Should accept (rank, world_size, *args, **kwargs)
        world_size: Number of processes to spawn (TP degree or EP degree)
        backend: torch.distributed backend ("gloo" for CPU, "nccl" for GPU)
        test_args: Optional tuple of positional arguments to pass to test_func
        test_kwargs: Optional dict of keyword arguments to pass to test_func
        master_port: Optional master port (default: auto-assigned)

    Returns:
        List of results from each process (one per rank)

    Example:
        def test_tp_forward(rank, world_size, model, input_ids):
            # Initialize distributed
            # Run forward pass with TP
            return output

        results = run_distributed_test(
            test_tp_forward,
            world_size=2,
            test_args=(model, input_ids)
        )
    """
    if test_args is None:
        test_args = ()
    if test_kwargs is None:
        test_kwargs = {}

    # Auto-assign port if not provided
    if master_port is None:
        import random
        master_port = str(29500 + random.randint(0, 1000))

    # Create temporary directory for inter-process communication
    with tempfile.TemporaryDirectory() as tmpdir:
        result_dir = Path(tmpdir)

        # Spawn processes
        mp.spawn(
            _worker_process,
            args=(
                world_size,
                backend,
                test_func,
                test_args,
                test_kwargs,
                result_dir,
                master_port,
            ),
            nprocs=world_size,
            join=True,
        )

        # Collect results from all processes
        results = []
        for rank in range(world_size):
            result_file = result_dir / f"rank_{rank}_result.pkl"
            with open(result_file, "rb") as f:
                result = pickle.load(f)
            results.append(result)

        return results


def _worker_process(
    rank: int,
    world_size: int,
    backend: str,
    test_func: Callable,
    test_args: Tuple,
    test_kwargs: Dict[str, Any],
    result_dir: Path,
    master_port: str,
) -> None:
    """
    Worker process that initializes distributed and runs the test function.

    This function:
    - Sets up environment variables for torch.distributed
    - Initializes the process group with specified backend
    - Runs the test function with proper rank and world_size
    - Saves the result to a file for collection by main process
    - Handles cleanup and error propagation

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: torch.distributed backend
        test_func: Test function to run
        test_args: Positional arguments for test_func
        test_kwargs: Keyword arguments for test_func
        result_dir: Directory to save results
        master_port: Master port for distributed communication
    """
    try:
        # Set up environment for torch.distributed
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = master_port
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)

        # Initialize process group with timeout
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=30),
        )

        # Run the test function
        result = test_func(rank, world_size, *test_args, **test_kwargs)

        # Save result to file
        result_file = result_dir / f"rank_{rank}_result.pkl"
        with open(result_file, "wb") as f:
            pickle.dump(result, f)

    except Exception as e:
        # Save exception for propagation
        result_file = result_dir / f"rank_{rank}_result.pkl"
        with open(result_file, "wb") as f:
            pickle.dump(e, f)
        raise

    finally:
        # Clean up process group
        if dist.is_initialized():
            dist.destroy_process_group()


def init_distributed_test_env(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> None:
    """
    Initialize torch.distributed environment for testing.

    This is a convenience function for manually setting up distributed
    environment in tests. Use run_distributed_test() for automatic
    multi-process spawning.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: torch.distributed backend ("gloo" for CPU, "nccl" for GPU)
        master_addr: Master node address (default: localhost)
        master_port: Master node port (default: 29500)

    Example:
        # In a manually spawned process
        init_distributed_test_env(rank=0, world_size=2)
        # Now can use dist.all_reduce(), etc.
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed_test_env() -> None:
    """
    Clean up torch.distributed environment after testing.

    This function destroys the process group if it's initialized.
    Should be called in test teardown or finally block.

    Example:
        try:
            init_distributed_test_env(rank=0, world_size=2)
            # Run distributed test
        finally:
            cleanup_distributed_test_env()
    """
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedTestContext:
    """
    Context manager for distributed test environment setup and cleanup.

    This provides a clean way to set up and tear down distributed
    environment in tests using a with statement.

    Args:
        rank: Process rank (0 to world_size-1)
        world_size: Total number of processes
        backend: torch.distributed backend ("gloo" for CPU, "nccl" for GPU)
        master_addr: Master node address (default: localhost)
        master_port: Master node port (default: 29500)

    Example:
        with DistributedTestContext(rank=0, world_size=2):
            # Distributed environment is initialized
            tensor = torch.ones(10)
            dist.all_reduce(tensor)
        # Automatically cleaned up
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        backend: str = "gloo",
        master_addr: str = "127.0.0.1",
        master_port: str = "29500",
    ):
        self.rank = rank
        self.world_size = world_size
        self.backend = backend
        self.master_addr = master_addr
        self.master_port = master_port

    def __enter__(self):
        init_distributed_test_env(
            rank=self.rank,
            world_size=self.world_size,
            backend=self.backend,
            master_addr=self.master_addr,
            master_port=self.master_port,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed_test_env()
        return False
