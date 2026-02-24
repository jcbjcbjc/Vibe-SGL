"""First-Come-First-Serve scheduling policy."""

from typing import List
from vibe_sgl_lite.scheduler.policy import SchedulerPolicy
from vibe_sgl_lite.batch.request import Request


class FCFSPolicy(SchedulerPolicy):
    """First-Come-First-Serve scheduling policy.

    Selects requests in arrival order (queue-based). This is the simplest
    scheduling policy that ensures fairness by processing requests in the
    order they arrive.

    Characteristics:
    - Simple queue-based implementation
    - No complex scoring or prioritization
    - Fair processing in arrival order
    - Predictable latency for users
    """

    @property
    def name(self) -> str:
        """Get the policy name.

        Returns:
            "FCFS"
        """
        return "FCFS"

    def select_requests(
        self,
        waiting_requests: List[Request],
        max_batch_size: int
    ) -> List[Request]:
        """Select requests in arrival order.

        Args:
            waiting_requests: List of requests waiting to be processed
            max_batch_size: Maximum number of requests that can be selected

        Returns:
            First N requests from the queue (up to max_batch_size)
        """
        # Simply return the first max_batch_size requests
        return waiting_requests[:max_batch_size]
