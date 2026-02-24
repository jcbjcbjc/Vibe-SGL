"""Tests for ChunkManager class."""

import pytest
import torch
from vibe_sgl_lite.batch.chunk_manager import ChunkManager


class TestChunkManagerInitialization:
    """Test ChunkManager initialization."""

    def test_default_chunk_size(self):
        """Test ChunkManager uses default chunk size of 8192."""
        manager = ChunkManager()
        assert manager.chunk_size == 8192

    def test_custom_chunk_size(self):
        """Test ChunkManager accepts custom chunk size."""
        manager = ChunkManager(chunk_size=4096)
        assert manager.chunk_size == 4096

    def test_invalid_chunk_size_zero(self):
        """Test ChunkManager raises ValueError for zero chunk size."""
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            ChunkManager(chunk_size=0)

    def test_invalid_chunk_size_negative(self):
        """Test ChunkManager raises ValueError for negative chunk size."""
        with pytest.raises(ValueError, match="Chunk size must be positive"):
            ChunkManager(chunk_size=-100)

    def test_initialization_state(self):
        """Test ChunkManager initializes with empty state."""
        manager = ChunkManager()
        assert len(manager.active_chunks) == 0
        assert len(manager.completed_chunks) == 0


class TestSequenceChunking:
    """Test sequence chunking functionality."""

    def test_split_long_sequence(self):
        """Test splitting sequence longer than chunk size."""
        manager = ChunkManager(chunk_size=8)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])

        chunks = manager.split_sequence("req1", input_ids)

        # Should split into 3 chunks: [1-8], [9-16], [17]
        assert len(chunks) == 3
        assert torch.equal(chunks[0], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]))
        assert torch.equal(chunks[1], torch.tensor([9, 10, 11, 12, 13, 14, 15, 16]))
        assert torch.equal(chunks[2], torch.tensor([17]))

    def test_split_exact_multiple(self):
        """Test splitting sequence that's exact multiple of chunk size."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])

        chunks = manager.split_sequence("req2", input_ids)

        # Should split into 2 chunks: [1-4], [5-8]
        assert len(chunks) == 2
        assert torch.equal(chunks[0], torch.tensor([1, 2, 3, 4]))
        assert torch.equal(chunks[1], torch.tensor([5, 6, 7, 8]))

    def test_short_sequence_passthrough(self):
        """Test short sequence processed as single chunk."""
        manager = ChunkManager(chunk_size=10)
        input_ids = torch.tensor([1, 2, 3, 4, 5])

        chunks = manager.split_sequence("req3", input_ids)

        # Should return single chunk
        assert len(chunks) == 1
        assert torch.equal(chunks[0], input_ids)

    def test_chunk_state_created(self):
        """Test ChunkState is created when splitting sequence."""
        manager = ChunkManager(chunk_size=8)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        chunks = manager.split_sequence("req4", input_ids)

        # Should create ChunkState in active_chunks
        assert "req4" in manager.active_chunks
        state = manager.active_chunks["req4"]
        assert state.request_id == "req4"
        assert torch.equal(state.input_ids, input_ids)
        assert state.chunk_size == 8
        assert state.total_chunks == 2
        assert state.current_chunk_idx == 0
        assert len(state.completed_chunks) == 0

    def test_default_chunk_size_8192(self):
        """Test default chunk size of 8192 tokens."""
        manager = ChunkManager()
        # Create sequence of 20000 tokens
        input_ids = torch.arange(20000)

        chunks = manager.split_sequence("req5", input_ids)

        # Should split into 3 chunks: [0-8191], [8192-16383], [16384-19999]
        assert len(chunks) == 3
        assert len(chunks[0]) == 8192
        assert len(chunks[1]) == 8192
        assert len(chunks[2]) == 20000 - 16384


class TestIncrementalProcessing:
    """Test incremental chunk processing."""

    def test_process_first_chunk(self):
        """Test processing first chunk."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        chunks = manager.split_sequence("req1", input_ids)

        # Process first chunk
        result = manager.process_chunk("req1", chunks[0])

        assert result is not None
        assert torch.equal(result, chunks[0])
        state = manager.active_chunks["req1"]
        assert state.current_chunk_idx == 1
        assert 0 in state.completed_chunks

    def test_process_subsequent_chunks(self):
        """Test processing subsequent chunks."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        chunks = manager.split_sequence("req2", input_ids)

        # Process all chunks
        manager.process_chunk("req2", chunks[0])
        manager.process_chunk("req2", chunks[1])
        result = manager.process_chunk("req2", chunks[2])

        assert result is not None
        # After all chunks complete, should be in completed_chunks
        state = manager.completed_chunks["req2"]
        assert state.current_chunk_idx == 3
        assert state.completed_chunks == [0, 1, 2]

    def test_track_chunk_progress(self):
        """Test chunk progress tracking."""
        manager = ChunkManager(chunk_size=5)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        chunks = manager.split_sequence("req3", input_ids)

        # Process first two chunks
        manager.process_chunk("req3", chunks[0])
        manager.process_chunk("req3", chunks[1])

        state = manager.active_chunks["req3"]
        assert state.current_chunk_idx == 2
        assert len(state.completed_chunks) == 2
        assert state.completed_chunks == [0, 1]
        # Still has one chunk remaining
        assert state.current_chunk_idx < state.total_chunks

    def test_complete_all_chunks(self):
        """Test completing all chunks moves to completed_chunks."""
        manager = ChunkManager(chunk_size=3)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        chunks = manager.split_sequence("req4", input_ids)

        # Process all chunks
        for chunk in chunks:
            manager.process_chunk("req4", chunk)

        # Should move to completed_chunks
        assert "req4" not in manager.active_chunks
        assert "req4" in manager.completed_chunks
        state = manager.completed_chunks["req4"]
        assert len(state.completed_chunks) == state.total_chunks

    def test_is_complete(self):
        """Test checking if chunked sequence is complete."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        chunks = manager.split_sequence("req5", input_ids)

        # Not complete initially
        assert not manager.is_complete("req5")

        # Process first chunk
        manager.process_chunk("req5", chunks[0])
        assert not manager.is_complete("req5")

        # Process second chunk
        manager.process_chunk("req5", chunks[1])
        assert manager.is_complete("req5")


class TestInterleavedPrefillDecode:
    """Test interleaved prefill-decode functionality."""

    def test_get_next_chunk_for_request(self):
        """Test getting next chunk to process for a request."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        chunks = manager.split_sequence("req1", input_ids)

        # Get first chunk
        chunk = manager.get_next_chunk("req1")
        assert chunk is not None
        assert torch.equal(chunk, chunks[0])

        # Process it
        manager.process_chunk("req1", chunk)

        # Get second chunk
        chunk = manager.get_next_chunk("req1")
        assert chunk is not None
        assert torch.equal(chunk, chunks[1])

    def test_get_next_chunk_returns_none_when_complete(self):
        """Test get_next_chunk returns None when all chunks complete."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4])
        chunks = manager.split_sequence("req2", input_ids)

        # Process the only chunk
        chunk = manager.get_next_chunk("req2")
        manager.process_chunk("req2", chunk)

        # Should return None
        chunk = manager.get_next_chunk("req2")
        assert chunk is None

    def test_has_pending_chunks(self):
        """Test checking if request has pending chunks."""
        manager = ChunkManager(chunk_size=3)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        chunks = manager.split_sequence("req3", input_ids)

        # Initially has pending chunks
        assert manager.has_pending_chunks("req3")

        # Process first chunk
        chunk = manager.get_next_chunk("req3")
        manager.process_chunk("req3", chunk)
        assert manager.has_pending_chunks("req3")

        # Process second chunk
        chunk = manager.get_next_chunk("req3")
        manager.process_chunk("req3", chunk)
        assert not manager.has_pending_chunks("req3")

    def test_get_all_pending_requests(self):
        """Test getting all requests with pending chunks."""
        manager = ChunkManager(chunk_size=4)

        # Add multiple requests
        input_ids1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        input_ids2 = torch.tensor([10, 11, 12, 13])
        input_ids3 = torch.tensor([20, 21, 22, 23, 24, 25])

        manager.split_sequence("req1", input_ids1)
        manager.split_sequence("req2", input_ids2)
        manager.split_sequence("req3", input_ids3)

        # All should have pending chunks
        pending = manager.get_pending_requests()
        assert len(pending) == 3
        assert "req1" in pending
        assert "req2" in pending
        assert "req3" in pending

        # Complete req2
        chunk = manager.get_next_chunk("req2")
        manager.process_chunk("req2", chunk)

        # Should only have req1 and req3
        pending = manager.get_pending_requests()
        assert len(pending) == 2
        assert "req1" in pending
        assert "req3" in pending
        assert "req2" not in pending


class TestIncrementalKVCacheAllocation:
    """Test incremental KV cache allocation during chunking."""

    def test_allocate_pages_for_chunk(self):
        """Test allocating pages for a single chunk."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
        manager.split_sequence("req1", input_ids)

        # Allocate pages for first chunk (4 tokens)
        page_ids = manager.allocate_pages_for_chunk("req1", num_pages=2)

        assert len(page_ids) == 2
        state = manager.active_chunks["req1"]
        assert len(state.allocated_pages) == 2
        assert state.allocated_pages == page_ids

    def test_incremental_page_allocation(self):
        """Test pages are allocated incrementally per chunk."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        chunks = manager.split_sequence("req2", input_ids)

        # Allocate for first chunk
        pages1 = manager.allocate_pages_for_chunk("req2", num_pages=2)
        assert len(manager.active_chunks["req2"].allocated_pages) == 2

        # Process first chunk
        manager.process_chunk("req2", chunks[0])

        # Allocate for second chunk
        pages2 = manager.allocate_pages_for_chunk("req2", num_pages=2)
        assert len(manager.active_chunks["req2"].allocated_pages) == 4

        # Pages should be different
        assert pages1 != pages2

    def test_get_allocated_pages(self):
        """Test getting all allocated pages for a request."""
        manager = ChunkManager(chunk_size=3)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6])
        manager.split_sequence("req3", input_ids)

        # Allocate pages
        manager.allocate_pages_for_chunk("req3", num_pages=1)
        manager.allocate_pages_for_chunk("req3", num_pages=1)

        pages = manager.get_allocated_pages("req3")
        assert len(pages) == 2

    def test_calculate_pages_needed(self):
        """Test calculating pages needed for a chunk."""
        manager = ChunkManager(chunk_size=16)

        # 16 tokens with page_size=16 needs 1 page
        pages = manager.calculate_pages_needed(chunk_size=16, page_size=16)
        assert pages == 1

        # 17 tokens with page_size=16 needs 2 pages
        pages = manager.calculate_pages_needed(chunk_size=17, page_size=16)
        assert pages == 2

        # 32 tokens with page_size=16 needs 2 pages
        pages = manager.calculate_pages_needed(chunk_size=32, page_size=16)
        assert pages == 2

    def test_track_total_allocated_pages(self):
        """Test tracking total pages allocated across all chunks."""
        manager = ChunkManager(chunk_size=4)
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        manager.split_sequence("req4", input_ids)

        # Allocate for each chunk
        manager.allocate_pages_for_chunk("req4", num_pages=1)
        manager.allocate_pages_for_chunk("req4", num_pages=1)
        manager.allocate_pages_for_chunk("req4", num_pages=1)

        total = manager.get_total_allocated_pages("req4")
        assert total == 3


class TestConfigurableChunkSize:
    """Test configurable chunk size functionality."""

    def test_set_chunk_size_after_init(self):
        """Test setting chunk size after initialization."""
        manager = ChunkManager(chunk_size=4096)
        assert manager.chunk_size == 4096

        # Update chunk size
        manager.set_chunk_size(2048)
        assert manager.chunk_size == 2048

    def test_chunk_size_affects_splitting(self):
        """Test chunk size affects how sequences are split."""
        input_ids = torch.arange(100)

        # With chunk size 50
        manager1 = ChunkManager(chunk_size=50)
        chunks1 = manager1.split_sequence("req1", input_ids)
        assert len(chunks1) == 2

        # With chunk size 25
        manager2 = ChunkManager(chunk_size=25)
        chunks2 = manager2.split_sequence("req2", input_ids)
        assert len(chunks2) == 4

    def test_invalid_chunk_size_update(self):
        """Test updating to invalid chunk size raises error."""
        manager = ChunkManager()

        with pytest.raises(ValueError, match="Chunk size must be positive"):
            manager.set_chunk_size(0)

        with pytest.raises(ValueError, match="Chunk size must be positive"):
            manager.set_chunk_size(-100)

    def test_get_chunk_size(self):
        """Test getting current chunk size."""
        manager = ChunkManager(chunk_size=1024)
        assert manager.get_chunk_size() == 1024

    def test_chunk_size_configuration_per_request(self):
        """Test different chunk sizes can be used for different requests."""
        manager = ChunkManager(chunk_size=8)

        # First request with default chunk size
        input_ids1 = torch.arange(20)
        chunks1 = manager.split_sequence("req1", input_ids1)
        assert len(chunks1) == 3  # 20 / 8 = 3 chunks

        # Change chunk size
        manager.set_chunk_size(5)

        # Second request with new chunk size
        input_ids2 = torch.arange(20)
        chunks2 = manager.split_sequence("req2", input_ids2)
        assert len(chunks2) == 4  # 20 / 5 = 4 chunks
