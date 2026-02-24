"""
BatchManager for continuous batching.
"""

from typing import List, Optional, Dict
import torch
from vibe_sgl_lite.batch.request import Request, RequestState


class BatchManager:
    """Manages continuous batching of generation requests.

    The BatchManager coordinates the lifecycle of multiple concurrent generation
    requests, handling dynamic batch composition, iteration-level scheduling,
    and resource management.

    Attributes:
        max_batch_size: Maximum number of requests in active batch
        active_requests: Currently executing requests
        waiting_requests: Requests waiting to be added to batch
    """

    def __init__(self, max_batch_size: int = 8):
        """Initialize BatchManager.

        Args:
            max_batch_size: Maximum number of requests in active batch
        """
        self.max_batch_size = max_batch_size
        self.active_requests: List[Request] = []
        self.waiting_requests: List[Request] = []
        self._request_map: Dict[str, Request] = {}

        # Metrics
        self._total_requests = 0
        self._completed_requests = 0

    def add_request(self, request: Request) -> None:
        """Add a new request to the waiting queue.

        Args:
            request: Request to add
        """
        self.waiting_requests.append(request)
        self._request_map[request.request_id] = request
        self._total_requests += 1

    def remove_request(self, request_id: str) -> Optional[Request]:
        """Remove a request from active batch.

        Args:
            request_id: ID of request to remove

        Returns:
            Removed request, or None if not found
        """
        # Find and remove from active requests
        for i, req in enumerate(self.active_requests):
            if req.request_id == request_id:
                removed = self.active_requests.pop(i)
                if request_id in self._request_map:
                    del self._request_map[request_id]
                return removed

        # Also check waiting requests
        for i, req in enumerate(self.waiting_requests):
            if req.request_id == request_id:
                removed = self.waiting_requests.pop(i)
                if request_id in self._request_map:
                    del self._request_map[request_id]
                return removed

        return None

    def step(self) -> None:
        """Perform one iteration step.

        This method:
        1. Removes completed requests from active batch
        2. Adds waiting requests to active batch (up to max_batch_size)
        3. Updates request states
        """
        # Count and remove completed requests
        completed_count = sum(1 for req in self.active_requests if req.is_completed())
        self._completed_requests += completed_count

        self.active_requests = [
            req for req in self.active_requests if not req.is_completed()
        ]

        # Add waiting requests to active batch
        available_slots = self.max_batch_size - len(self.active_requests)
        if available_slots > 0 and self.waiting_requests:
            # Move waiting requests to active
            to_add = self.waiting_requests[:available_slots]
            self.waiting_requests = self.waiting_requests[available_slots:]

            for req in to_add:
                req.state = RequestState.PREFILLING
                self.active_requests.append(req)

    def get_active_requests(self) -> List[Request]:
        """Get list of active requests.

        Returns:
            List of currently active requests
        """
        return self.active_requests.copy()

    def is_empty(self) -> bool:
        """Check if batch is empty.

        Returns:
            True if no active requests
        """
        return len(self.active_requests) == 0

    def has_waiting_requests(self) -> bool:
        """Check if there are waiting requests.

        Returns:
            True if there are requests waiting to be added
        """
        return len(self.waiting_requests) > 0

    def get_batch_size(self) -> int:
        """Get current batch size.

        Returns:
            Number of active requests
        """
        return len(self.active_requests)

    def get_prefill_requests(self) -> List[Request]:
        """Get requests in prefill phase.

        Returns:
            List of requests currently being prefilled
        """
        return [
            req
            for req in self.active_requests
            if req.state == RequestState.PREFILLING
        ]

    def get_decode_requests(self) -> List[Request]:
        """Get requests in decode phase.

        Returns:
            List of requests currently decoding
        """
        return [
            req for req in self.active_requests if req.state == RequestState.DECODING
        ]

    def create_padded_batch(
        self, pad_token_id: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create padded batch from active requests.

        Args:
            pad_token_id: Token ID to use for padding

        Returns:
            Tuple of (batch_input_ids, attention_mask)
            - batch_input_ids: [batch_size, max_seq_len]
            - attention_mask: [batch_size, max_seq_len]
        """
        if not self.active_requests:
            return torch.empty(0, 0, dtype=torch.long), torch.empty(
                0, 0, dtype=torch.long
            )

        # Get current sequence for each request
        sequences = []
        for req in self.active_requests:
            if req.output_ids is not None:
                sequences.append(req.output_ids[0])  # Remove batch dim
            else:
                sequences.append(req.input_ids[0])  # Remove batch dim

        # Find max length
        max_len = max(seq.shape[0] for seq in sequences)

        # Create padded batch
        batch_size = len(sequences)
        batch_input_ids = torch.full(
            (batch_size, max_len), pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

        # Fill in sequences and masks
        for i, seq in enumerate(sequences):
            seq_len = seq.shape[0]
            batch_input_ids[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

        return batch_input_ids, attention_mask

    def get_metrics(self) -> Dict[str, float]:
        """Get throughput and utilization metrics.

        Returns:
            Dictionary containing:
            - total_requests: Total requests added
            - completed_requests: Total requests completed
            - batch_utilization: Current batch size / max batch size
        """
        batch_utilization = (
            len(self.active_requests) / self.max_batch_size
            if self.max_batch_size > 0
            else 0.0
        )

        return {
            "total_requests": self._total_requests,
            "completed_requests": self._completed_requests,
            "batch_utilization": batch_utilization,
        }
