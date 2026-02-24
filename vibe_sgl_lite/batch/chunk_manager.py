"""ChunkManager for handling chunked prefill of long sequences."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch


@dataclass
class ChunkState:
    """State for a chunked sequence."""
    request_id: str
    input_ids: torch.Tensor
    chunk_size: int
    current_chunk_idx: int = 0
    total_chunks: int = 0
    completed_chunks: List[int] = field(default_factory=list)
    allocated_pages: List[int] = field(default_factory=list)


class ChunkManager:
    """Manages chunked prefill for long sequences.

    Splits long input sequences into configurable chunks and processes them
    incrementally to avoid memory spikes and enable interleaved prefill-decode.

    Args:
        chunk_size: Maximum tokens per chunk (default: 8192)
    """

    def __init__(self, chunk_size: int = 8192):
        """Initialize ChunkManager.

        Args:
            chunk_size: Maximum tokens per chunk (default: 8192)

        Raises:
            ValueError: If chunk_size is zero or negative
        """
        if chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {chunk_size}")

        self.chunk_size = chunk_size
        self.active_chunks: Dict[str, ChunkState] = {}
        self.completed_chunks: Dict[str, ChunkState] = {}

    def split_sequence(self, request_id: str, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Split input sequence into chunks.

        Args:
            request_id: Unique identifier for the request
            input_ids: Input token IDs to split

        Returns:
            List of chunk tensors
        """
        seq_len = input_ids.size(0)

        # If sequence is shorter than chunk size, return as single chunk
        if seq_len <= self.chunk_size:
            chunks = [input_ids]
            total_chunks = 1
        else:
            # Split into chunks
            chunks = []
            for i in range(0, seq_len, self.chunk_size):
                chunk = input_ids[i:i + self.chunk_size]
                chunks.append(chunk)
            total_chunks = len(chunks)

        # Create ChunkState for this request
        state = ChunkState(
            request_id=request_id,
            input_ids=input_ids,
            chunk_size=self.chunk_size,
            current_chunk_idx=0,
            total_chunks=total_chunks,
            completed_chunks=[],
            allocated_pages=[]
        )
        self.active_chunks[request_id] = state

        return chunks

    def process_chunk(self, request_id: str, chunk: torch.Tensor) -> Optional[torch.Tensor]:
        """Process a single chunk for a request.

        Args:
            request_id: Unique identifier for the request
            chunk: Chunk tensor to process

        Returns:
            The processed chunk tensor, or None if request not found
        """
        if request_id not in self.active_chunks:
            return None

        state = self.active_chunks[request_id]

        # Mark current chunk as completed
        state.completed_chunks.append(state.current_chunk_idx)

        # Move to next chunk
        state.current_chunk_idx += 1

        # If all chunks are complete, move to completed_chunks
        if len(state.completed_chunks) == state.total_chunks:
            self.completed_chunks[request_id] = state
            del self.active_chunks[request_id]

        return chunk

    def is_complete(self, request_id: str) -> bool:
        """Check if all chunks for a request are complete.

        Args:
            request_id: Unique identifier for the request

        Returns:
            True if all chunks are complete, False otherwise
        """
        # Check if in completed_chunks
        if request_id in self.completed_chunks:
            return True

        # Check if in active_chunks and all chunks are complete
        if request_id in self.active_chunks:
            state = self.active_chunks[request_id]
            return len(state.completed_chunks) == state.total_chunks

        return False

    def get_next_chunk(self, request_id: str) -> Optional[torch.Tensor]:
        """Get the next chunk to process for a request.

        Args:
            request_id: Unique identifier for the request

        Returns:
            Next chunk tensor, or None if no more chunks or request not found
        """
        if request_id not in self.active_chunks:
            return None

        state = self.active_chunks[request_id]

        # Check if all chunks are complete
        if state.current_chunk_idx >= state.total_chunks:
            return None

        # Get the next chunk
        start_idx = state.current_chunk_idx * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(state.input_ids))
        chunk = state.input_ids[start_idx:end_idx]

        return chunk

    def has_pending_chunks(self, request_id: str) -> bool:
        """Check if request has pending chunks to process.

        Args:
            request_id: Unique identifier for the request

        Returns:
            True if request has pending chunks, False otherwise
        """
        if request_id not in self.active_chunks:
            return False

        state = self.active_chunks[request_id]
        return state.current_chunk_idx < state.total_chunks

    def get_pending_requests(self) -> List[str]:
        """Get all request IDs with pending chunks.

        Returns:
            List of request IDs with pending chunks
        """
        return [
            request_id
            for request_id, state in self.active_chunks.items()
            if state.current_chunk_idx < state.total_chunks
        ]

    def allocate_pages_for_chunk(self, request_id: str, num_pages: int) -> List[int]:
        """Allocate pages for a chunk.

        Args:
            request_id: Unique identifier for the request
            num_pages: Number of pages to allocate

        Returns:
            List of allocated page IDs
        """
        if request_id not in self.active_chunks:
            return []

        state = self.active_chunks[request_id]

        # Generate page IDs (in real implementation, these would come from PagedAllocator)
        # For now, use simple incrementing IDs
        start_id = len(state.allocated_pages)
        page_ids = list(range(start_id, start_id + num_pages))

        # Add to allocated pages
        state.allocated_pages.extend(page_ids)

        return page_ids

    def get_allocated_pages(self, request_id: str) -> List[int]:
        """Get all allocated pages for a request.

        Args:
            request_id: Unique identifier for the request

        Returns:
            List of allocated page IDs
        """
        if request_id in self.active_chunks:
            return self.active_chunks[request_id].allocated_pages
        elif request_id in self.completed_chunks:
            return self.completed_chunks[request_id].allocated_pages
        return []

    def calculate_pages_needed(self, chunk_size: int, page_size: int) -> int:
        """Calculate number of pages needed for a chunk.

        Args:
            chunk_size: Number of tokens in chunk
            page_size: Number of tokens per page

        Returns:
            Number of pages needed
        """
        import math
        return math.ceil(chunk_size / page_size)

    def get_total_allocated_pages(self, request_id: str) -> int:
        """Get total number of allocated pages for a request.

        Args:
            request_id: Unique identifier for the request

        Returns:
            Total number of allocated pages
        """
        return len(self.get_allocated_pages(request_id))

    def set_chunk_size(self, chunk_size: int) -> None:
        """Set the chunk size.

        Args:
            chunk_size: New chunk size

        Raises:
            ValueError: If chunk_size is zero or negative
        """
        if chunk_size <= 0:
            raise ValueError(f"Chunk size must be positive, got {chunk_size}")

        self.chunk_size = chunk_size

    def get_chunk_size(self) -> int:
        """Get the current chunk size.

        Returns:
            Current chunk size
        """
        return self.chunk_size
