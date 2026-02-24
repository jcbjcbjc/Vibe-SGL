"""
Request dataclass for continuous batching.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import torch
from vibe_sgl_lite.sampling.sampling import SamplingParams


class RequestState(Enum):
    """State of a generation request."""

    WAITING = "waiting"  # Request is waiting to be added to batch
    PREFILLING = "prefilling"  # Request is being prefilled
    DECODING = "decoding"  # Request is generating tokens
    COMPLETED = "completed"  # Request has finished generation


@dataclass
class Request:
    """Represents a single generation request in continuous batching.

    Attributes:
        request_id: Unique identifier for the request
        input_ids: Input token IDs [1, seq_len]
        max_new_tokens: Maximum number of tokens to generate
        sampling_params: Sampling parameters for generation
        stream: Whether to stream tokens as they are generated
        state: Current state of the request
        generated_tokens: Number of tokens generated so far
        output_ids: Generated token IDs (including input)
        kv_cache_refs: References to KV cache pages (for paged attention)
    """

    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    stream: bool = False
    state: RequestState = RequestState.WAITING
    generated_tokens: int = 0
    output_ids: Optional[torch.Tensor] = None
    kv_cache_refs: Optional[list] = None

    def is_completed(self) -> bool:
        """Check if request is completed."""
        return self.state == RequestState.COMPLETED

    def is_finished(self) -> bool:
        """Check if request has reached max tokens or generated EOS."""
        return self.generated_tokens >= self.max_new_tokens

    def get_current_length(self) -> int:
        """Get current sequence length (input + generated)."""
        if self.output_ids is not None:
            return self.output_ids.shape[1]
        return self.input_ids.shape[1]
