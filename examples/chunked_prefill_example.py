"""Example demonstrating chunked prefill for long sequences.

This example shows how to use ChunkManager to process long input sequences
incrementally, avoiding memory spikes and enabling interleaved prefill-decode.
"""

import torch
from vibe_sgl_lite.batch.chunk_manager import ChunkManager


def main():
    """Demonstrate chunked prefill functionality."""
    print("=== Chunked Prefill Example ===\n")

    # Create ChunkManager with 8K chunk size (default)
    manager = ChunkManager(chunk_size=8192)
    print(f"ChunkManager initialized with chunk_size={manager.get_chunk_size()}")

    # Simulate a long input sequence (20K tokens)
    long_input = torch.arange(20000)
    print(f"\nInput sequence length: {len(long_input)} tokens")

    # Split sequence into chunks
    chunks = manager.split_sequence("long_request", long_input)
    print(f"Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i}: {len(chunk)} tokens")

    # Process chunks incrementally
    print("\n--- Processing chunks incrementally ---")
    request_id = "long_request"

    while manager.has_pending_chunks(request_id):
        # Get next chunk to process
        chunk = manager.get_next_chunk(request_id)
        print(f"\nProcessing chunk {manager.active_chunks[request_id].current_chunk_idx}")
        print(f"  Chunk size: {len(chunk)} tokens")

        # Calculate pages needed for this chunk (assuming 16 tokens per page)
        pages_needed = manager.calculate_pages_needed(len(chunk), page_size=16)
        print(f"  Pages needed: {pages_needed}")

        # Allocate pages incrementally
        page_ids = manager.allocate_pages_for_chunk(request_id, pages_needed)
        print(f"  Allocated pages: {page_ids}")

        # Process the chunk (in real implementation, this would run model forward pass)
        manager.process_chunk(request_id, chunk)

        # Show progress
        state = manager.active_chunks.get(request_id) or manager.completed_chunks.get(request_id)
        print(f"  Progress: {len(state.completed_chunks)}/{state.total_chunks} chunks complete")

    # Check completion
    print(f"\n--- Chunked prefill complete ---")
    print(f"Request complete: {manager.is_complete(request_id)}")
    print(f"Total pages allocated: {manager.get_total_allocated_pages(request_id)}")

    # Demonstrate configurable chunk size
    print("\n\n=== Configurable Chunk Size ===\n")

    # Create manager with smaller chunk size
    manager2 = ChunkManager(chunk_size=4096)
    print(f"ChunkManager with chunk_size={manager2.get_chunk_size()}")

    # Same input, different chunking
    chunks2 = manager2.split_sequence("req2", long_input)
    print(f"Same 20K input split into {len(chunks2)} chunks with 4K chunk size")

    # Demonstrate interleaved processing
    print("\n\n=== Interleaved Prefill-Decode ===\n")

    manager3 = ChunkManager(chunk_size=1000)

    # Add multiple requests
    input1 = torch.arange(3000)  # 3 chunks
    input2 = torch.arange(500)   # 1 chunk
    input3 = torch.arange(2500)  # 3 chunks

    manager3.split_sequence("req_a", input1)
    manager3.split_sequence("req_b", input2)
    manager3.split_sequence("req_c", input3)

    print("Added 3 requests:")
    print(f"  req_a: {len(input1)} tokens -> 3 chunks")
    print(f"  req_b: {len(input2)} tokens -> 1 chunk")
    print(f"  req_c: {len(input3)} tokens -> 3 chunks")

    # Process chunks in round-robin fashion (interleaved)
    print("\nProcessing chunks in round-robin (interleaved):")
    iteration = 0
    while True:
        pending = manager3.get_pending_requests()
        if not pending:
            break

        iteration += 1
        print(f"\nIteration {iteration}:")

        # Process one chunk from each pending request
        for req_id in pending:
            chunk = manager3.get_next_chunk(req_id)
            if chunk is not None:
                manager3.process_chunk(req_id, chunk)
                state = manager3.active_chunks.get(req_id) or manager3.completed_chunks.get(req_id)
                status = "COMPLETE" if manager3.is_complete(req_id) else "IN_PROGRESS"
                print(f"  {req_id}: processed chunk, {len(state.completed_chunks)}/{state.total_chunks} done [{status}]")

    print("\nAll requests complete!")


if __name__ == "__main__":
    main()
