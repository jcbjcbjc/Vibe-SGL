"""
Tests for Request dataclass.
"""

import pytest
import torch
from vibe_sgl_lite.batch.request import Request, RequestState
from vibe_sgl_lite.sampling.sampling import SamplingParams


@pytest.mark.unit
def test_request_initialization():
    """Test Request initialization with required fields."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    assert request.request_id == "req_1"
    assert torch.equal(request.input_ids, torch.tensor([[1, 2, 3]]))
    assert request.max_new_tokens == 10
    assert request.state == RequestState.WAITING
    assert request.generated_tokens == 0
    assert request.output_ids is None


@pytest.mark.unit
def test_request_with_sampling_params():
    """Test Request initialization with custom sampling parameters."""
    sampling_params = SamplingParams(temperature=0.8, top_p=0.9)
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
        sampling_params=sampling_params,
    )

    assert request.sampling_params == sampling_params
    assert request.sampling_params.temperature == 0.8
    assert request.sampling_params.top_p == 0.9


@pytest.mark.unit
def test_request_default_sampling_params():
    """Test Request uses default sampling params if not provided."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    assert request.sampling_params is not None
    assert isinstance(request.sampling_params, SamplingParams)


@pytest.mark.unit
def test_request_state_transitions():
    """Test Request state transitions."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    # Initial state
    assert request.state == RequestState.WAITING

    # Transition to prefilling
    request.state = RequestState.PREFILLING
    assert request.state == RequestState.PREFILLING

    # Transition to decoding
    request.state = RequestState.DECODING
    assert request.state == RequestState.DECODING

    # Transition to completed
    request.state = RequestState.COMPLETED
    assert request.state == RequestState.COMPLETED


@pytest.mark.unit
def test_request_track_generated_tokens():
    """Test tracking generated tokens count."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    assert request.generated_tokens == 0

    # Simulate token generation
    request.generated_tokens = 5
    assert request.generated_tokens == 5


@pytest.mark.unit
def test_request_output_ids():
    """Test storing output token IDs."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    assert request.output_ids is None

    # Set output IDs
    request.output_ids = torch.tensor([[1, 2, 3, 4, 5]])
    assert torch.equal(request.output_ids, torch.tensor([[1, 2, 3, 4, 5]]))


@pytest.mark.unit
def test_request_is_completed():
    """Test checking if request is completed."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    # Not completed initially
    assert not request.is_completed()

    # Completed when state is COMPLETED
    request.state = RequestState.COMPLETED
    assert request.is_completed()


@pytest.mark.unit
def test_request_is_finished_max_tokens():
    """Test checking if request reached max tokens."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )

    assert not request.is_finished()

    # Reached max tokens
    request.generated_tokens = 10
    assert request.is_finished()


@pytest.mark.unit
def test_request_streaming_mode():
    """Test request with streaming enabled."""
    request = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
        stream=True,
    )

    assert request.stream is True


@pytest.mark.unit
def test_request_unique_ids():
    """Test that requests have unique IDs."""
    request1 = Request(
        request_id="req_1",
        input_ids=torch.tensor([[1, 2, 3]]),
        max_new_tokens=10,
    )
    request2 = Request(
        request_id="req_2",
        input_ids=torch.tensor([[4, 5, 6]]),
        max_new_tokens=10,
    )

    assert request1.request_id != request2.request_id
