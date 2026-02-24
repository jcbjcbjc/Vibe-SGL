"""Longest Prefix Match scheduling policy."""

from typing import List
import torch
from vibe_sgl_lite.scheduler.policy import SchedulerPolicy
from vibe_sgl_lite.batch.request import Request
from vibe_sgl_lite.cache.radix_cache import RadixCache


class LPMPolicy(SchedulerPolicy):
    """Longest Prefix Match scheduling policy.

    Prioritizes requests with longest cached prefix matches to maximize
    cache reuse and improve throughput. Uses RadixCache to query for
    prefix matches and scores requests accordingly.

    Characteristics:
    - Cache-aware scheduling
    - Maximizes prefix reuse across batch
    - Falls back to FCFS for equal scores
    - Improves throughput by reducing redundant computation

    Args:
        cache: RadixCache instance for prefix matching
        cache_weight: Weight for cache score (default: 1.0)
    """

    def __init__(self, cache: RadixCache, cache_weight: float = 1.0):
        """Initialize LPM policy.

        Args:
            cache: RadixCache instance for prefix matching
            cache_weight: Weight for cache score (default: 1.0)
        """
        self.cache = cache
        self.cache_weight = cache_weight

    @property
    def name(self) -> str:
        """Get the policy name.

        Returns:
            "LPM"
        """
        return "LPM"

    def score_request(self, request: Request) -> float:
        """Score a request based on cached prefix length.

        Args:
            request: Request to score

        Returns:
            Score as (matched_tokens / total_tokens) * cache_weight
        """
        # Query cache for longest prefix match
        tokens = request.input_ids.tolist()
        _, match_len = self.cache.lookup(tokens)

        # Calculate score as ratio of matched tokens
        total_tokens = len(request.input_ids)
        if total_tokens == 0:
            return 0.0

        score = (match_len / total_tokens) * self.cache_weight
        return score

    def select_requests(
        self,
        waiting_requests: List[Request],
        max_batch_size: int
    ) -> List[Request]:
        """Select requests prioritizing longest cached prefixes.

        Args:
            waiting_requests: List of requests waiting to be processed
            max_batch_size: Maximum number of requests that can be selected

        Returns:
            Requests sorted by cache score (highest first), up to max_batch_size
        """
        if not waiting_requests:
            return []

        # Score all requests
        scored_requests = [
            (self.score_request(req), idx, req)
            for idx, req in enumerate(waiting_requests)
        ]

        # Sort by score (descending), then by index (ascending) for FCFS tiebreaker
        scored_requests.sort(key=lambda x: (-x[0], x[1]))

        # Select top max_batch_size requests
        selected = [req for _, _, req in scored_requests[:max_batch_size]]

        return selected
