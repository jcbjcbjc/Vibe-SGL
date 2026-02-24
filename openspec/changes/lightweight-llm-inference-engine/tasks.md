## 1. Project Setup

- [x] 1.1 Create vibe_sgl_lite package structure with __init__.py files
- [x] 1.2 Create subdirectories: core/, models/, memory/, cache/, batch/, sampling/, scheduler/, distributed/, utils/
- [x] 1.3 Set up pyproject.toml with dependencies (torch, transformers, pytest, pytest-cov)
- [x] 1.4 Create tests/ directory structure mirroring source layout
- [x] 1.5 Configure pytest.ini with test markers (unit, integration, slow, distributed)
- [x] 1.6 Create README.md with project overview and setup instructions
- [x] 1.7 Create TESTING.md with test running instructions

## 2. Test Infrastructure

- [x] 2.1 Write test fixture for loading Qwen3-0.6B model (with caching)
- [x] 2.2 Write test fixture for Qwen3 tokenizer
- [x] 2.3 Write test fixture for forcing CPU device
- [x] 2.4 Write helper for spawning multi-process distributed tests
- [x] 2.5 Write helper for initializing torch.distributed with Gloo backend
- [x] 2.6 Create sample test data (prompts, expected outputs, edge cases)
- [x] 2.7 Set up pytest-cov for coverage reporting
- [x] 2.8 Write test utilities for comparing model outputs with tolerance

## 3. Custom Qwen3 Model - Basic Structure (TDD)

- [x] 3.1 Write tests for Qwen3Config loading from HuggingFace
- [x] 3.2 Implement Qwen3Config class
- [x] 3.3 Write tests for RMSNorm layer
- [x] 3.4 Implement RMSNorm layer
- [x] 3.5 Write tests for RoPE (rotary position embeddings) computation
- [x] 3.6 Implement RoPE precomputation and application functions
- [x] 3.7 Write tests for weight loading from HuggingFace checkpoint
- [x] 3.8 Implement weight loading and name mapping utilities

## 4. Custom Qwen3 Model - Attention Layer (TDD)

- [x] 4.1 Write tests for Q/K/V projection initialization
- [x] 4.2 Implement Q/K/V projection layers (nn.Linear)
- [x] 4.3 Write tests for grouped-query attention (GQA) head configuration
- [x] 4.4 Implement GQA head grouping and KV head repetition
- [x] 4.5 Write tests for attention computation with RoPE
- [x] 4.6 Implement attention forward pass with RoPE application
- [x] 4.7 Write tests for attention output projection
- [x] 4.8 Implement output projection layer
- [x] 4.9 Write tests for attention with KV cache
- [x] 4.10 Implement KV cache integration in attention layer
- [x] 4.11 Validate attention outputs against HuggingFace Qwen3

## 5. Custom Qwen3 Model - FFN Layer (TDD)

- [x] 5.1 Write tests for FFN layer initialization (up_proj, gate_proj, down_proj)
- [x] 5.2 Implement FFN layer structure
- [x] 5.3 Write tests for SwiGLU activation computation
- [x] 5.4 Implement SwiGLU activation (Swish(gate) * up)
- [x] 5.5 Write tests for FFN forward pass
- [x] 5.6 Implement FFN forward pass
- [x] 5.7 Validate FFN outputs against HuggingFace Qwen3

## 6. Custom Qwen3 Model - Full Model (TDD)

- [x] 6.1 Write tests for Qwen3DecoderLayer (attention + FFN + norms)
- [x] 6.2 Implement Qwen3DecoderLayer
- [x] 6.3 Write tests for embedding layer
- [x] 6.4 Implement token embedding layer
- [x] 6.5 Write tests for full Qwen3Model forward pass
- [x] 6.6 Implement Qwen3Model with layer stack
- [x] 6.7 Write tests for LM head (output projection to vocab)
- [x] 6.8 Implement LM head for logits computation
- [x] 6.9 Write end-to-end test comparing with HuggingFace Qwen3-0.6B
- [x] 6.10 Validate numerical correctness (atol=1e-5, rtol=1e-4)

## 7. Memory Management - Memory Pool (TDD)

- [x] 7.1 Write tests for MemoryPool initialization with cache size
- [x] 7.2 Implement MemoryPool class with page pre-allocation
- [x] 7.3 Write tests for page allocation (single and batch)
- [x] 7.4 Implement allocate() method with free list management
- [x] 7.5 Write tests for page deallocation
- [x] 7.6 Implement free() method returning pages to pool
- [x] 7.7 Write tests for memory accounting and statistics
- [x] 7.8 Implement get_stats() method (used, free, utilization)
- [x] 7.9 Write tests for allocation failure when pool exhausted
- [x] 7.10 Implement allocation failure handling
- [x] 7.11 Write tests for thread-safe concurrent allocation/deallocation
- [x] 7.12 Implement thread safety with locks
- [x] 7.13 Integrate MemoryPool into InferenceEngine
- [x] 7.14 Add end-to-end test with MemoryPool enabled

## 8. Memory Management - Paged Allocator (TDD)

- [x] 8.1 Write tests for PagedAllocator initialization
- [x] 8.2 Implement PagedAllocator class
- [x] 8.3 Write tests for page table creation and management
- [x] 8.4 Implement page table (logical position → physical page mapping)
- [x] 8.5 Write tests for allocating pages for new sequence
- [x] 8.6 Implement sequence page allocation
- [x] 8.7 Write tests for page sharing with reference counting
- [x] 8.8 Implement reference counting for shared pages
- [x] 8.9 Write tests for LRU eviction policy
- [x] 8.10 Implement LRU eviction when pool exhausted
- [x] 8.11 Write tests for page metadata tracking
- [x] 8.12 Implement page metadata (ID, owner, ref count)
- [x] 8.13 Integrate PagedAllocator into InferenceEngine
- [x] 8.14 Add end-to-end test with PagedAllocator enabled

## 9. Paged Attention Integration (TDD)

- [x] 9.1 Write tests for gathering KV from pages using page table
- [x] 9.2 Implement KV gather operation for attention
- [x] 9.3 Write tests for scattering new KV to pages
- [x] 9.4 Implement KV scatter operation after forward pass
- [x] 9.5 Write tests for attention computation with paged KV cache
- [x] 9.6 Integrate paged KV cache into Qwen3Attention layer
- [x] 9.7 Write tests for handling page boundaries in attention
- [x] 9.8 Implement correct attention across page boundaries
- [x] 9.9 Write tests for memory efficiency vs. contiguous allocation
- [x] 9.10 Validate paged attention correctness and measure overhead
- [x] 9.11 Integrate paged attention into InferenceEngine
- [x] 9.12 Add end-to-end test with paged attention enabled

## 10. RadixAttention - Radix Tree (TDD)

- [x] 10.1 Write tests for RadixTreeNode structure
- [x] 10.2 Implement RadixTreeNode (tokens, pages, children, metadata)
- [x] 10.3 Write tests for inserting token sequences into tree
- [x] 10.4 Implement tree insertion with branch creation
- [x] 10.5 Write tests for prefix matching (exact and partial)
- [x] 10.6 Implement prefix matching algorithm
- [x] 10.7 Write tests for reference counting on tree nodes
- [x] 10.8 Implement reference counting for cache entries
- [x] 10.9 Write tests for LRU eviction of tree branches
- [x] 10.10 Implement LRU eviction with timestamp tracking
- [x] 10.11 Write tests for thread-safe tree operations
- [x] 10.12 Implement locking for concurrent access

## 11. RadixAttention - Cache Integration (TDD)

- [x] 11.1 Write tests for cache insertion after prefill
- [x] 11.2 Implement cache insertion in inference pipeline
- [x] 11.3 Write tests for cache lookup before prefill
- [x] 11.4 Implement cache lookup and page reuse
- [x] 11.5 Write tests for cache hit/miss metrics
- [x] 11.6 Implement cache statistics tracking
- [x] 11.7 Write tests for multi-turn conversation caching
- [x] 11.8 Validate cache reuse across conversation turns
- [x] 11.9 Write tests for cache hit rate improvement
- [x] 11.10 Measure and validate cache effectiveness
- [x] 11.11 Integrate RadixCache into InferenceEngine
- [x] 11.12 Add end-to-end test with RadixCache enabled

## 12. Sampling Strategies (TDD)

- [x] 12.1 Write tests for SamplingParams dataclass
- [x] 12.2 Implement SamplingParams (temperature, top_p, top_k, penalties, etc.)
- [x] 12.3 Write tests for greedy sampling (argmax)
- [x] 12.4 Implement greedy sampling
- [x] 12.5 Write tests for temperature scaling
- [x] 12.6 Implement temperature scaling
- [x] 12.7 Write tests for top-k sampling
- [x] 12.8 Implement top-k sampling
- [x] 12.9 Write tests for top-p (nucleus) sampling
- [x] 12.10 Implement top-p sampling
- [x] 12.11 Write tests for frequency penalty
- [x] 12.12 Implement frequency penalty
- [x] 12.13 Write tests for presence penalty
- [x] 12.14 Implement presence penalty
- [x] 12.15 Write tests for repetition penalty
- [x] 12.16 Implement repetition penalty
- [x] 12.17 Write tests for logit bias
- [x] 12.18 Implement logit bias
- [x] 12.19 Write tests for stop sequences
- [x] 12.20 Implement stop sequence detection
- [x] 12.21 Write tests for random seed control
- [x] 12.22 Implement reproducible sampling with seed

## 13. Core Inference Engine (TDD)

- [x] 13.1 Write tests for InferenceEngine initialization
- [x] 13.2 Implement InferenceEngine class
- [x] 13.3 Write tests for model loading
- [x] 13.4 Implement model loading from checkpoint
- [x] 13.5 Write tests for tokenization
- [x] 13.6 Integrate tokenizer for input processing
- [x] 13.7 Write tests for single sequence generation
- [x] 13.8 Implement generate() method for single sequence
- [x] 13.9 Write tests for batch generation
- [x] 13.10 Implement batch generation
- [x] 13.11 Write tests for streaming generation
- [x] 13.12 Implement streaming token generation
- [x] 13.13 Write tests for error handling (OOM, invalid input)
- [x] 13.14 Implement error handling and validation
- [x] 13.15 Write end-to-end integration tests
- [x] 13.16 Validate complete inference pipeline

## 14. Continuous Batching (TDD)

- [x] 14.1 Write tests for Request dataclass
- [x] 14.2 Implement Request (input, params, state)
- [x] 14.3 Write tests for BatchManager initialization
- [x] 14.4 Implement BatchManager class
- [x] 14.5 Write tests for adding requests to batch
- [x] 14.6 Implement add_request() method
- [x] 14.7 Write tests for removing completed requests
- [x] 14.8 Implement remove_request() method
- [x] 14.9 Write tests for iteration-level batch updates
- [x] 14.10 Implement step() method for batch iteration
- [x] 14.11 Write tests for mixed prefill-decode batching
- [x] 14.12 Implement prefill and decode phase separation
- [x] 14.13 Write tests for batch padding and attention masks
- [x] 14.14 Implement padding and mask generation
- [x] 14.15 Write tests for throughput optimization
- [x] 14.16 Validate batch utilization and tokens/sec

## 15. Chunked Prefill (TDD)

- [x] 15.1 Write tests for ChunkManager initialization
- [x] 15.2 Implement ChunkManager class
- [x] 15.3 Write tests for sequence chunking (8K default)
- [x] 15.4 Implement split_sequence() method
- [x] 15.5 Write tests for incremental chunk processing
- [x] 15.6 Implement process_chunk() method
- [x] 15.7 Write tests for chunk state tracking
- [x] 15.8 Implement chunk progress tracking
- [x] 15.9 Write tests for interleaved prefill-decode
- [x] 15.10 Integrate chunked prefill with BatchManager
- [x] 15.11 Write tests for incremental KV cache allocation
- [x] 15.12 Implement page-by-page allocation during chunking
- [x] 15.13 Write tests for configurable chunk size
- [x] 15.14 Implement chunk size configuration

## 16. Scheduling Policies (TDD)

- [x] 16.1 Write tests for SchedulerPolicy abstract interface
- [x] 16.2 Implement SchedulerPolicy base class
- [x] 16.3 Write tests for FCFSPolicy
- [x] 16.4 Implement FCFSPolicy (queue-based, arrival order)
- [x] 16.5 Write tests for LPMPolicy prefix scoring
- [x] 16.6 Implement LPMPolicy with cache-aware scoring
- [x] 16.7 Write tests for batch formation with policies
- [x] 16.8 Implement select_requests() for batch formation
- [x] 16.9 Write tests for fairness guarantees (max wait time)
- [x] 16.10 Implement starvation prevention
- [x] 16.11 Write tests for policy metrics (hit rate, latency)
- [x] 16.12 Implement metrics tracking for policies
- [x] 16.13 Write tests for policy configuration
- [x] 16.14 Implement policy parameter validation

## 17. Tensor Parallelism - Infrastructure (TDD)

- [x] 17.1 Write tests for TP process group initialization (Gloo)
- [x] 17.2 Implement init_tp_process_group() with torch.distributed
- [x] 17.3 Write tests for rank and world size assignment
- [x] 17.4 Implement rank/world size management
- [x] 17.5 Write tests for weight partitioning utilities
- [x] 17.6 Implement partition_weights() for column/row parallelism
- [x] 17.7 Write tests for all-reduce operation
- [x] 17.8 Implement all_reduce_wrapper() using torch.distributed
- [x] 17.9 Write tests for broadcast operation
- [x] 17.10 Implement broadcast_wrapper() for weight distribution

## 18. Tensor Parallelism - Model Layers (TDD)

- [x] 18.1 Write tests for ColumnParallelLinear layer
- [x] 18.2 Implement ColumnParallelLinear (split output dim)
- [x] 18.3 Write tests for RowParallelLinear layer
- [x] 18.4 Implement RowParallelLinear (split input dim + all-reduce)
- [x] 18.5 Write tests for TP-aware Qwen3Attention
- [x] 18.6 Modify Qwen3Attention to use parallel linear layers
- [x] 18.7 Write tests for TP-aware Qwen3MLP
- [x] 18.8 Modify Qwen3MLP to use parallel linear layers
- [x] 18.9 Write tests for TP weight loading and partitioning
- [x] 18.10 Implement load_tp_weights() for Qwen3
- [x] 18.11 Write multi-process tests for TP correctness
- [x] 18.12 Validate TP outputs match single-device reference

## 19. Expert Parallelism - Infrastructure (TDD)

- [ ] 19.1 Write tests for EP process group initialization
- [ ] 19.2 Implement init_ep_process_group() with torch.distributed
- [ ] 19.3 Write tests for expert-to-device mapping
- [ ] 19.4 Implement expert placement strategy (expert_id % num_devices)
- [ ] 19.5 Write tests for all-to-all communication
- [ ] 19.6 Implement all_to_all_wrapper() for token redistribution
- [ ] 19.7 Write tests for synthetic MoE layer
- [ ] 19.8 Implement SyntheticMoE for testing EP

## 20. Expert Parallelism - Routing and Execution (TDD)

- [ ] 20.1 Write tests for ExpertRouter initialization
- [ ] 20.2 Implement ExpertRouter class
- [ ] 20.3 Write tests for token routing to top-k experts
- [ ] 20.4 Implement route_tokens() method
- [ ] 20.5 Write tests for capacity factor enforcement
- [ ] 20.6 Implement capacity limiting and overflow handling
- [ ] 20.7 Write tests for expert computation batching
- [ ] 20.8 Implement batched expert forward pass
- [ ] 20.9 Write tests for result gathering and combining
- [ ] 20.10 Implement gather_results() with routing weights
- [ ] 20.11 Write multi-process tests for EP correctness
- [ ] 20.12 Validate EP outputs match single-device MoE reference

## 21. Hybrid TP+EP (TDD)

- [ ] 21.1 Write tests for combined TP and EP initialization
- [ ] 21.2 Implement hybrid process group setup
- [ ] 21.3 Write tests for TP within experts
- [ ] 21.4 Integrate TP into expert layers
- [ ] 21.5 Write tests for communication ordering (EP then TP)
- [ ] 21.6 Implement correct communication sequence
- [ ] 21.7 Write multi-process tests for hybrid parallelism
- [ ] 21.8 Validate hybrid TP+EP correctness

## 22. Integration and End-to-End Testing

- [ ] 22.1 Write integration test: basic inference with all features
- [ ] 22.2 Write integration test: continuous batching + caching
- [ ] 22.3 Write integration test: chunked prefill + paged attention
- [ ] 22.4 Write integration test: TP with 2 ranks
- [ ] 22.5 Write integration test: EP with synthetic MoE
- [ ] 22.6 Write integration test: LPM scheduling with cache hits
- [ ] 22.7 Write integration test: streaming generation
- [ ] 22.8 Write integration test: multi-turn conversation with caching
- [ ] 22.9 Write performance benchmark: latency vs batch size
- [ ] 22.10 Write performance benchmark: throughput with continuous batching
- [ ] 22.11 Write performance benchmark: cache hit rate improvement
- [ ] 22.12 Validate all integration tests pass

## 23. Documentation and Examples

- [ ] 23.1 Write API documentation for InferenceEngine
- [ ] 23.2 Write API documentation for SamplingParams
- [ ] 23.3 Write example: basic text generation
- [ ] 23.4 Write example: batch inference
- [ ] 23.5 Write example: streaming generation
- [ ] 23.6 Write example: multi-turn conversation
- [ ] 23.7 Write example: using TP for larger models
- [ ] 23.8 Write example: custom scheduling policy
- [ ] 23.9 Update README with usage examples
- [ ] 23.10 Create architecture diagram showing component relationships
- [ ] 23.11 Document TDD workflow and testing practices
- [ ] 23.12 Create troubleshooting guide

## 24. Final Validation and Cleanup

- [ ] 24.1 Run full test suite and ensure 100% pass rate
- [ ] 24.2 Generate coverage report and verify >80% coverage
- [ ] 24.3 Run integration tests with Qwen3-0.6B
- [ ] 24.4 Validate numerical correctness against HuggingFace
- [ ] 24.5 Run performance benchmarks and document results
- [ ] 24.6 Review and clean up code (remove debug prints, TODOs)
- [ ] 24.7 Run linter (ruff or flake8) and fix issues
- [ ] 24.8 Run type checker (mypy) and fix type issues
- [ ] 24.9 Review all documentation for completeness
- [ ] 24.10 Create release checklist
- [ ] 24.11 Tag v0.1.0 release
- [ ] 24.12 Celebrate successful TDD implementation! 🎉
