"""Tests for SchedulerPolicy abstract interface."""

import pytest
import torch
from abc import ABC
from vibe_sgl_lite.scheduler.policy import SchedulerPolicy
from vibe_sgl_lite.batch.request import Request, RequestState
from vibe_sgl_lite.sampling.sampling import SamplingParams


class TestSchedulerPolicyInterface:
    """Test SchedulerPolicy abstract interface."""

    def test_is_abstract_class(self):
        """Test SchedulerPolicy is an abstract base class."""
        assert issubclass(SchedulerPolicy, ABC)

    def test_cannot_instantiate_directly(self):
        """Test cannot instantiate SchedulerPolicy directly."""
        with pytest.raises(TypeError):
            SchedulerPolicy()

    def test_has_select_requests_method(self):
        """Test SchedulerPolicy defines select_requests method."""
        assert hasattr(SchedulerPolicy, 'select_requests')

    def test_select_requests_is_abstract(self):
        """Test select_requests is an abstract method."""
        # Create a concrete class without implementing select_requests
        class IncompletePolicy(SchedulerPolicy):
            pass

        with pytest.raises(TypeError):
            IncompletePolicy()

    def test_concrete_implementation_works(self):
        """Test concrete implementation of SchedulerPolicy works."""
        class ConcretePolicy(SchedulerPolicy):
            @property
            def name(self):
                return "concrete"

            def select_requests(self, waiting_requests, max_batch_size):
                return waiting_requests[:max_batch_size]

        policy = ConcretePolicy()
        assert policy is not None

        # Test select_requests works
        requests = [
            Request("req1", torch.tensor([1, 2, 3]), max_new_tokens=10),
            Request("req2", torch.tensor([4, 5, 6]), max_new_tokens=10),
        ]
        selected = policy.select_requests(requests, max_batch_size=1)
        assert len(selected) == 1

    def test_policy_name_property(self):
        """Test SchedulerPolicy has name property."""
        class NamedPolicy(SchedulerPolicy):
            @property
            def name(self):
                return "test_policy"

            def select_requests(self, waiting_requests, max_batch_size):
                return []

        policy = NamedPolicy()
        assert policy.name == "test_policy"

    def test_select_requests_signature(self):
        """Test select_requests has correct signature."""
        class TestPolicy(SchedulerPolicy):
            @property
            def name(self):
                return "test"

            def select_requests(self, waiting_requests, max_batch_size):
                return []

        policy = TestPolicy()

        # Should accept list of requests and max_batch_size
        result = policy.select_requests([], max_batch_size=8)
        assert isinstance(result, list)


class TestFCFSPolicy:
    """Test First-Come-First-Serve scheduling policy."""

    def test_fcfs_policy_name(self):
        """Test FCFS policy has correct name."""
        from vibe_sgl_lite.scheduler.fcfs_policy import FCFSPolicy
        policy = FCFSPolicy()
        assert policy.name == "FCFS"

    def test_fcfs_selects_in_arrival_order(self):
        """Test FCFS selects requests in arrival order."""
        from vibe_sgl_lite.scheduler.fcfs_policy import FCFSPolicy
        policy = FCFSPolicy()

        # Create requests with different arrival times
        requests = [
            Request("req1", torch.tensor([1, 2, 3]), max_new_tokens=10),
            Request("req2", torch.tensor([4, 5, 6]), max_new_tokens=10),
            Request("req3", torch.tensor([7, 8, 9]), max_new_tokens=10),
        ]

        # Select 2 requests
        selected = policy.select_requests(requests, max_batch_size=2)

        assert len(selected) == 2
        assert selected[0].request_id == "req1"
        assert selected[1].request_id == "req2"

    def test_fcfs_respects_batch_size_limit(self):
        """Test FCFS respects maximum batch size."""
        from vibe_sgl_lite.scheduler.fcfs_policy import FCFSPolicy
        policy = FCFSPolicy()

        requests = [
            Request(f"req{i}", torch.tensor([i]), max_new_tokens=10)
            for i in range(10)
        ]

        # Request batch size of 3
        selected = policy.select_requests(requests, max_batch_size=3)

        assert len(selected) == 3
        assert selected[0].request_id == "req0"
        assert selected[1].request_id == "req1"
        assert selected[2].request_id == "req2"

    def test_fcfs_handles_empty_queue(self):
        """Test FCFS handles empty request queue."""
        from vibe_sgl_lite.scheduler.fcfs_policy import FCFSPolicy
        policy = FCFSPolicy()

        selected = policy.select_requests([], max_batch_size=5)

        assert len(selected) == 0

    def test_fcfs_handles_fewer_requests_than_batch_size(self):
        """Test FCFS handles fewer requests than batch size."""
        from vibe_sgl_lite.scheduler.fcfs_policy import FCFSPolicy
        policy = FCFSPolicy()

        requests = [
            Request("req1", torch.tensor([1, 2]), max_new_tokens=10),
            Request("req2", torch.tensor([3, 4]), max_new_tokens=10),
        ]

        selected = policy.select_requests(requests, max_batch_size=5)

        assert len(selected) == 2

    def test_fcfs_simple_queue_implementation(self):
        """Test FCFS uses simple queue without complex scoring."""
        from vibe_sgl_lite.scheduler.fcfs_policy import FCFSPolicy
        policy = FCFSPolicy()

        # FCFS should not have scoring logic
        assert not hasattr(policy, 'score_request')
        assert not hasattr(policy, 'priority_queue')

    def test_fcfs_fair_processing(self):
        """Test FCFS processes oldest request first."""
        from vibe_sgl_lite.scheduler.fcfs_policy import FCFSPolicy
        policy = FCFSPolicy()

        # Create requests
        requests = [
            Request("old", torch.tensor([1]), max_new_tokens=10),
            Request("new", torch.tensor([2]), max_new_tokens=10),
        ]

        # Select one request
        selected = policy.select_requests(requests, max_batch_size=1)

        # Should select the first (oldest) request
        assert len(selected) == 1
        assert selected[0].request_id == "old"


class TestLPMPolicy:
    """Test Longest Prefix Match scheduling policy."""

    def test_lpm_policy_name(self):
        """Test LPM policy has correct name."""
        from vibe_sgl_lite.scheduler.lpm_policy import LPMPolicy
        from vibe_sgl_lite.cache.radix_cache import RadixCache

        cache = RadixCache()
        policy = LPMPolicy(cache)
        assert policy.name == "LPM"

    def test_lpm_requires_cache(self):
        """Test LPM policy requires RadixCache."""
        from vibe_sgl_lite.scheduler.lpm_policy import LPMPolicy
        from vibe_sgl_lite.cache.radix_cache import RadixCache

        cache = RadixCache()
        policy = LPMPolicy(cache)
        assert policy.cache is not None

    def test_lpm_scores_by_prefix_length(self):
        """Test LPM scores requests by cached prefix length."""
        from vibe_sgl_lite.scheduler.lpm_policy import LPMPolicy
        from vibe_sgl_lite.cache.radix_cache import RadixCache

        cache = RadixCache()
        policy = LPMPolicy(cache)

        # Insert some prefixes into cache
        cache.insert("seq1", [1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        cache.insert("seq2", [1, 2], [0, 1])

        # Create requests with different prefix matches
        req1 = Request("req1", torch.tensor([1, 2, 3, 4, 5, 6]), max_new_tokens=10)  # 5 tokens match
        req2 = Request("req2", torch.tensor([1, 2, 7, 8]), max_new_tokens=10)  # 2 tokens match
        req3 = Request("req3", torch.tensor([9, 10, 11]), max_new_tokens=10)  # 0 tokens match

        # Score requests
        score1 = policy.score_request(req1)
        score2 = policy.score_request(req2)
        score3 = policy.score_request(req3)

        # req1 should have highest score (longest match)
        assert score1 > score2
        assert score2 > score3

    def test_lpm_prioritizes_cache_hits(self):
        """Test LPM prioritizes requests with longest cached prefixes."""
        from vibe_sgl_lite.scheduler.lpm_policy import LPMPolicy
        from vibe_sgl_lite.cache.radix_cache import RadixCache

        cache = RadixCache()
        policy = LPMPolicy(cache)

        # Insert prefix
        cache.insert("seq1", [1, 2, 3, 4], [0, 1, 2, 3])

        # Create requests
        req_hit = Request("hit", torch.tensor([1, 2, 3, 4, 5]), max_new_tokens=10)  # Cache hit
        req_miss = Request("miss", torch.tensor([9, 10, 11]), max_new_tokens=10)  # Cache miss

        requests = [req_miss, req_hit]  # Miss comes first

        # Select 1 request
        selected = policy.select_requests(requests, max_batch_size=1)

        # Should select the cache hit
        assert len(selected) == 1
        assert selected[0].request_id == "hit"

    def test_lpm_fallback_to_fcfs(self):
        """Test LPM falls back to FCFS for equal prefix matches."""
        from vibe_sgl_lite.scheduler.lpm_policy import LPMPolicy
        from vibe_sgl_lite.cache.radix_cache import RadixCache

        cache = RadixCache()
        policy = LPMPolicy(cache)

        # No cache entries - all requests have score 0
        req1 = Request("req1", torch.tensor([1, 2]), max_new_tokens=10)
        req2 = Request("req2", torch.tensor([3, 4]), max_new_tokens=10)
        req3 = Request("req3", torch.tensor([5, 6]), max_new_tokens=10)

        requests = [req1, req2, req3]

        # Select 2 requests
        selected = policy.select_requests(requests, max_batch_size=2)

        # Should select in arrival order (FCFS tiebreaker)
        assert len(selected) == 2
        assert selected[0].request_id == "req1"
        assert selected[1].request_id == "req2"

    def test_lpm_maximizes_cache_reuse(self):
        """Test LPM maximizes total cached tokens across batch."""
        from vibe_sgl_lite.scheduler.lpm_policy import LPMPolicy
        from vibe_sgl_lite.cache.radix_cache import RadixCache

        cache = RadixCache()
        policy = LPMPolicy(cache)

        # Insert prefixes
        cache.insert("seq1", [1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 3, 4, 5, 6, 7])
        cache.insert("seq2", [10, 11, 12], [10, 11, 12])

        # Create requests
        req1 = Request("req1", torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]), max_new_tokens=10)  # 8 match
        req2 = Request("req2", torch.tensor([10, 11, 12, 13]), max_new_tokens=10)  # 3 match
        req3 = Request("req3", torch.tensor([20, 21]), max_new_tokens=10)  # 0 match

        requests = [req3, req2, req1]  # Worst to best

        # Select 2 requests
        selected = policy.select_requests(requests, max_batch_size=2)

        # Should select req1 and req2 (best cache reuse)
        assert len(selected) == 2
        assert selected[0].request_id == "req1"
        assert selected[1].request_id == "req2"

    def test_lpm_respects_batch_size(self):
        """Test LPM respects maximum batch size."""
        from vibe_sgl_lite.scheduler.lpm_policy import LPMPolicy
        from vibe_sgl_lite.cache.radix_cache import RadixCache

        cache = RadixCache()
        policy = LPMPolicy(cache)

        requests = [
            Request(f"req{i}", torch.tensor([i]), max_new_tokens=10)
            for i in range(10)
        ]

        selected = policy.select_requests(requests, max_batch_size=3)

        assert len(selected) == 3
