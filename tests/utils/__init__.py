"""Test utilities for vibe_sgl_lite."""

from tests.utils.comparison import (
    assert_logits_close,
    assert_tensors_close,
    assert_tokens_equal,
    compare_generation_outputs,
)
from tests.utils.distributed import (
    DistributedTestContext,
    cleanup_distributed_test_env,
    init_distributed_test_env,
    run_distributed_test,
)

__all__ = [
    # Comparison utilities
    "assert_tensors_close",
    "assert_logits_close",
    "assert_tokens_equal",
    "compare_generation_outputs",
    # Distributed utilities
    "run_distributed_test",
    "init_distributed_test_env",
    "cleanup_distributed_test_env",
    "DistributedTestContext",
]
