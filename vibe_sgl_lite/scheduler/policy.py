"""Abstract base class for scheduling policies."""

from abc import ABC, abstractmethod
from typing import List
from vibe_sgl_lite.batch.request import Request


class SchedulerPolicy(ABC):
    """Abstract base class for scheduling policies.

    Scheduling policies determine which waiting requests should be selected
    for batch processing. Different policies can optimize for different goals:
    - FCFS: Fair processing in arrival order
    - LPM: Cache-aware scheduling to maximize prefix reuse
    - Custom: User-defined policies for specific workloads
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the policy name.

        Returns:
            Policy name string
        """
        pass

    @abstractmethod
    def select_requests(
        self,
        waiting_requests: List[Request],
        max_batch_size: int
    ) -> List[Request]:
        """Select requests to add to the current batch.

        Args:
            waiting_requests: List of requests waiting to be processed
            max_batch_size: Maximum number of requests that can be selected

        Returns:
            List of selected requests (up to max_batch_size)
        """
        pass
