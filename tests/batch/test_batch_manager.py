"""
Tests for BatchManager.
"""

import pytest
import torch
from vibe_sgl_lite.batch.batch_manager import BatchManager
from vibe_sgl_lite.batch.request import Request, RequestState
from vibe_sgl_lite.sampling.sampling import SamplingParams


@pytest.mark.unit
def test_batch_manager_initialization():
    """Test BatchManager initialization."""
    manager = BatchManager(max_batch_size=8)

    assert manager.max_batch_size == 8
    assert len(manager.active_requests) == 0
    assert len(manager.waiting_requests) == 0


@pytest.mark.unit
def test_batch_manager_custom_max_batch_size():
    """Test BatchManager with custom max batch size."""
    manager = BatchManager(max_batch_size=16)
    assert manager.max_batch_size == 16


@pytest.mark.unit
def test_batch_manager_add_request():
    """Test adding a request to BatchManager."""
    manager = BatchManager(max_batch_size=8)

    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    manager.add_request(request)

    # Request should be in waiting queue
    assert len(manager.waiting_requests) == 1
    assert manager.waiting_requests[0].request_id == "req_1"


@pytest.mark.unit
def test_batch_manager_add_multiple_requests():
    """Test adding multiple requests."""
    manager = BatchManager(max_batch_size=8)

    for i in range(5):
        request = Request(
            request_id=f"req_{i}",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
        )
        manager.add_request(request)

    assert len(manager.waiting_requests) == 5


@pytest.mark.unit
def test_batch_manager_remove_request():
    """Test removing a request from BatchManager."""
    manager = BatchManager(max_batch_size=8)

    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    manager.add_request(request)
    # Move to active
    manager.step()

    # Remove request
    manager.remove_request("req_1")

    assert len(manager.active_requests) == 0


@pytest.mark.unit
def test_batch_manager_step_moves_waiting_to_active():
    """Test that step() moves waiting requests to active batch."""
    manager = BatchManager(max_batch_size=8)

    # Add requests
    for i in range(3):
        request = Request(
            request_id=f"req_{i}",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
        )
        manager.add_request(request)

    assert len(manager.waiting_requests) == 3
    assert len(manager.active_requests) == 0

    # Step should move requests to active
    manager.step()

    assert len(manager.waiting_requests) == 0
    assert len(manager.active_requests) == 3


@pytest.mark.unit
def test_batch_manager_respects_max_batch_size():
    """Test that BatchManager respects max batch size."""
    manager = BatchManager(max_batch_size=2)

    # Add 5 requests
    for i in range(5):
        request = Request(
            request_id=f"req_{i}",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
        )
        manager.add_request(request)

    # Step should only move 2 to active
    manager.step()

    assert len(manager.active_requests) == 2
    assert len(manager.waiting_requests) == 3


@pytest.mark.unit
def test_batch_manager_get_active_requests():
    """Test getting active requests."""
    manager = BatchManager(max_batch_size=8)

    for i in range(3):
        request = Request(
            request_id=f"req_{i}",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
        )
        manager.add_request(request)

    manager.step()

    active = manager.get_active_requests()
    assert len(active) == 3
    assert all(isinstance(r, Request) for r in active)


@pytest.mark.unit
def test_batch_manager_is_empty():
    """Test checking if batch is empty."""
    manager = BatchManager(max_batch_size=8)

    assert manager.is_empty()

    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )
    manager.add_request(request)
    manager.step()

    assert not manager.is_empty()


@pytest.mark.unit
def test_batch_manager_has_waiting_requests():
    """Test checking if there are waiting requests."""
    manager = BatchManager(max_batch_size=8)

    assert not manager.has_waiting_requests()

    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )
    manager.add_request(request)

    assert manager.has_waiting_requests()

    manager.step()

    assert not manager.has_waiting_requests()


@pytest.mark.unit
def test_batch_manager_mixed_prefill_decode():
    """Test handling mixed prefill and decode requests."""
    manager = BatchManager(max_batch_size=8)

    # Add first request and start it (will be in decode phase)
    req1 = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )
    manager.add_request(req1)
    manager.step()

    # Simulate that req1 is now decoding
    req1.state = RequestState.DECODING
    req1.generated_tokens = 2

    # Add new request (will be in prefill phase)
    req2 = Request(
        request_id="req_2",
        input_ids=torch.tensor([[4, 5, 6, 7]]),
        max_new_tokens=10,
    )
    manager.add_request(req2)
    manager.step()

    # Both should be active
    assert len(manager.active_requests) == 2

    # Check states
    prefill_reqs = [r for r in manager.active_requests if r.state == RequestState.PREFILLING]
    decode_reqs = [r for r in manager.active_requests if r.state == RequestState.DECODING]

    assert len(prefill_reqs) == 1
    assert len(decode_reqs) == 1


@pytest.mark.unit
def test_batch_manager_get_prefill_requests():
    """Test getting requests in prefill phase."""
    manager = BatchManager(max_batch_size=8)

    # Add requests
    for i in range(3):
        req = Request(
            request_id=f"req_{i}",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
        )
        manager.add_request(req)

    manager.step()

    # All should be in prefill
    prefill_reqs = manager.get_prefill_requests()
    assert len(prefill_reqs) == 3


@pytest.mark.unit
def test_batch_manager_get_decode_requests():
    """Test getting requests in decode phase."""
    manager = BatchManager(max_batch_size=8)

    req = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )
    manager.add_request(req)
    manager.step()

    # Move to decode
    req.state = RequestState.DECODING

    decode_reqs = manager.get_decode_requests()
    assert len(decode_reqs) == 1


@pytest.mark.unit
def test_batch_manager_create_padded_batch():
    """Test creating padded batch from requests."""
    manager = BatchManager(max_batch_size=8)

    # Add requests with different lengths
    req1 = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )
    req2 = Request(
        request_id="req_2",
        input_ids=torch.tensor([[4, 5, 6, 7, 8]]),
        max_new_tokens=10,
    )

    manager.add_request(req1)
    manager.add_request(req2)
    manager.step()

    # Create padded batch
    batch_input_ids, attention_mask = manager.create_padded_batch()

    # Check shapes
    assert batch_input_ids.shape[0] == 2  # batch size
    assert batch_input_ids.shape[1] == 5  # max length in batch

    # Check padding
    assert attention_mask.shape == batch_input_ids.shape


@pytest.mark.unit
def test_batch_manager_attention_mask():
    """Test attention mask generation for padded sequences."""
    manager = BatchManager(max_batch_size=8)

    # Add requests with different lengths
    req1 = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2]]),
        max_new_tokens=10,
    )
    req2 = Request(
        request_id="req_2",
        input_ids=torch.tensor([[3, 4, 5, 6]]),
        max_new_tokens=10,
    )

    manager.add_request(req1)
    manager.add_request(req2)
    manager.step()

    batch_input_ids, attention_mask = manager.create_padded_batch()

    # First sequence should have 2 real tokens, 2 padding
    assert attention_mask[0, 0] == 1
    assert attention_mask[0, 1] == 1
    assert attention_mask[0, 2] == 0
    assert attention_mask[0, 3] == 0

    # Second sequence should have all real tokens
    assert torch.all(attention_mask[1] == 1)


@pytest.mark.unit
def test_batch_manager_throughput_metrics():
    """Test throughput metrics tracking."""
    manager = BatchManager(max_batch_size=8)

    # Initially no metrics
    metrics = manager.get_metrics()
    assert metrics["total_requests"] == 0
    assert metrics["completed_requests"] == 0


@pytest.mark.unit
def test_batch_manager_track_completed_requests():
    """Test tracking completed requests."""
    manager = BatchManager(max_batch_size=8)

    req = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )
    manager.add_request(req)
    manager.step()

    # Mark as completed
    req.state = RequestState.COMPLETED

    # Step should remove it
    manager.step()

    metrics = manager.get_metrics()
    assert metrics["completed_requests"] == 1


@pytest.mark.unit
def test_batch_manager_batch_utilization():
    """Test batch utilization calculation."""
    manager = BatchManager(max_batch_size=4)

    # Add 2 requests (50% utilization)
    for i in range(2):
        req = Request(
            request_id=f"req_{i}",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=10,
        )
        manager.add_request(req)

    manager.step()

    metrics = manager.get_metrics()
    assert metrics["batch_utilization"] == 0.5  # 2/4


@pytest.mark.integration
def test_batch_manager_end_to_end():
    """Test BatchManager end-to-end workflow."""
    manager = BatchManager(max_batch_size=4)

    # Add initial batch
    for i in range(3):
        req = Request(
            request_id=f"req_{i}",
            input_ids=torch.tensor([[1, 2, 3]]),
            max_new_tokens=5,
        )
        manager.add_request(req)

    # Step 1: Move to active
    manager.step()
    assert len(manager.active_requests) == 3

    # Simulate generation
    for req in manager.active_requests:
        req.state = RequestState.DECODING
        req.generated_tokens = 1

    # Add new request while batch is running
    new_req = Request(
        request_id="req_3",
        input_ids=torch.tensor([[4, 5, 6]]),
        max_new_tokens=5,
    )
    manager.add_request(new_req)

    # Step 2: Add new request to batch
    manager.step()
    assert len(manager.active_requests) == 4

    # Complete first request
    manager.active_requests[0].state = RequestState.COMPLETED

    # Step 3: Remove completed
    manager.step()
    assert len(manager.active_requests) == 3

    # Verify metrics
    metrics = manager.get_metrics()
    assert metrics["completed_requests"] == 1
    assert metrics["total_requests"] == 4
