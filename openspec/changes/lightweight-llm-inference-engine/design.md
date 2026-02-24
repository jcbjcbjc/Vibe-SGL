## Context

Building a lightweight LLM inference engine inspired by vibe-sgl's architecture, focusing on educational value and production-ready optimizations. The engine targets developers learning advanced LLM serving techniques through hands-on implementation with strict TDD practices.

**Current State:**
- No existing inference engine in the project
- Starting from scratch with clean architecture
- Target model: Qwen3-0.6B for testing
- Development environment: MacBook (CPU-only testing)

**Constraints:**
- Must support CPU testing with torch.distributed (Gloo backend)
- All features must have comprehensive test coverage (TDD)
- Custom Qwen3 implementation required for TP/EP support
- No external distributed frameworks (pure PyTorch)

**Stakeholders:**
- Developers learning LLM inference optimization techniques
- Users needing lightweight, understandable inference engine
- Contributors extending the engine with new features

## Goals / Non-Goals

**Goals:**
- Build modular inference engine with clear separation of concerns
- Implement core optimizations: paged attention, prefix caching, continuous batching
- Support tensor parallelism (TP) and expert parallelism (EP) via torch.distributed
- Provide custom Qwen3 model with native TP/EP support
- Achieve 100% test coverage following TDD methodology
- Enable CPU-based testing for accessibility
- Create educational codebase demonstrating modern inference techniques

**Non-Goals:**
- Supporting models beyond Qwen3 (single model focus)
- GPU-specific optimizations (FlashAttention, CUDA graphs) in initial version
- Production deployment infrastructure (Docker, Kubernetes)
- Speculative decoding (EAGLE) - deferred to future iteration
- Constrained generation (JSON schema, regex) - deferred to future iteration
- Multi-LoRA batching - deferred to future iteration
- Pipeline parallelism (PP) - focus on TP/EP only

## Decisions

### 1. Layered Architecture: Backbone + Features

**Decision:** Implement a layered architecture where all advanced features build on top of a core inference backbone.

**Rationale:**
- Enables incremental development and testing
- Clear dependency hierarchy: Core → Memory → Caching → Batching → Scheduling → Parallelism
- Each layer can be tested independently
- Aligns with TDD approach (test backbone first, then add features)

**Alternatives Considered:**
- Monolithic design: Rejected due to testing complexity and tight coupling
- Plugin architecture: Rejected as over-engineered for initial version

### 2. Custom Qwen3 Implementation

**Decision:** Rewrite Qwen3 model from scratch instead of wrapping HuggingFace Transformers.

**Rationale:**
- Native TP/EP support requires control over weight partitioning
- Column/row parallelism needs custom linear layers with all-reduce
- Transformers models not designed for distributed inference patterns
- Educational value: understanding model architecture deeply
- Flexibility to optimize for inference (remove training-only code)

**Alternatives Considered:**
- Monkey-patching Transformers: Rejected due to fragility and maintenance burden
- Using vLLM's model implementations: Rejected to maintain independence and simplicity

**Implementation Details:**
- Qwen3Attention: Custom attention with TP-aware Q/K/V projections
- Qwen3MLP: Custom FFN with column/row parallel linear layers
- Qwen3Model: Full model with layer-wise TP support
- Weight loading: Convert HuggingFace checkpoints to partitioned weights

### 3. Paged Attention Memory Management

**Decision:** Use fixed-size pages (16 tokens/page) for KV cache allocation.

**Rationale:**
- Eliminates memory fragmentation
- Enables efficient sharing for prefix caching
- Predictable memory usage
- Industry standard (vLLM, TGI use similar approach)

**Alternatives Considered:**
- Contiguous allocation: Rejected due to fragmentation and inflexibility
- Variable-size pages: Rejected due to complexity without clear benefits
- 32 tokens/page: Rejected as 16 provides better granularity for short sequences

**Implementation Details:**
- MemoryPool: Manages free pages with LRU eviction
- PagedAllocator: Allocates/deallocates pages per request
- Page format: [batch_size, num_heads, page_size, head_dim]

### 4. RadixAttention for Prefix Caching

**Decision:** Implement radix tree for automatic prefix matching and KV cache reuse.

**Rationale:**
- Automatic detection of shared prefixes (no manual cache management)
- Efficient O(prefix_length) lookup time
- Supports multi-turn conversations and batch processing
- Reference counting prevents premature eviction

**Alternatives Considered:**
- Hash-based caching: Rejected due to no partial matching support
- Manual cache keys: Rejected as error-prone and not automatic
- No caching: Rejected as misses major optimization opportunity

**Implementation Details:**
- RadixTreeNode: Stores token sequences and page references
- LRU eviction: When memory full, evict least recently used branches
- Reference counting: Track active requests using cached prefixes
- Cache hit metrics: Track hit rate for monitoring

### 5. Continuous Batching Strategy

**Decision:** Implement iteration-level batching where requests can join/leave at each decode step.

**Rationale:**
- Maximizes GPU/CPU utilization by keeping batches full
- Reduces latency (no waiting for slowest request)
- Handles variable-length sequences efficiently
- Standard in modern serving systems

**Alternatives Considered:**
- Static batching: Rejected due to poor utilization and high latency
- Separate prefill/decode batches: Deferred to future (adds complexity)

**Implementation Details:**
- BatchManager: Maintains active requests, adds/removes dynamically
- Separate prefill and decode phases within iteration
- Padding to max sequence length in batch
- Attention masks handle variable lengths

### 6. Scheduling Policies

**Decision:** Implement FCFS (First-Come-First-Serve) and LPM (Longest Prefix Match) policies.

**Rationale:**
- FCFS: Simple, fair, good baseline
- LPM: Cache-aware, maximizes prefix reuse
- Two policies demonstrate trade-offs (fairness vs. efficiency)
- Extensible design for adding more policies

**Alternatives Considered:**
- Only FCFS: Rejected as misses cache optimization opportunity
- Many policies (LOF, DFS-Weight, etc.): Deferred to keep initial version focused

**Implementation Details:**
- SchedulerPolicy interface: Abstract base class
- FCFSPolicy: Queue-based, arrival order
- LPMPolicy: Scores requests by cached prefix length
- Pluggable: Easy to add new policies

### 7. Tensor Parallelism via torch.distributed

**Decision:** Use torch.distributed with column/row parallelism for TP.

**Rationale:**
- No external dependencies (pure PyTorch)
- Well-documented and stable API
- Supports both CPU (Gloo) and GPU (NCCL) backends
- Process groups enable flexible parallelism strategies

**Alternatives Considered:**
- Megatron-LM: Rejected as heavy dependency
- Custom NCCL wrappers: Rejected as reinventing the wheel
- Ray: Rejected as adds complexity and overhead

**Implementation Details:**
- Column parallelism: Split Q/K/V projections, up_proj, gate_proj
- Row parallelism: Split output projections, down_proj (with all-reduce)
- Process groups: One group per TP rank
- Weight partitioning: Slice pretrained weights along appropriate dimensions

### 8. Expert Parallelism Design

**Decision:** Implement EP with expert-to-device mapping and all-to-all communication.

**Rationale:**
- Prepares for future MoE model support
- Demonstrates distributed routing patterns
- Complements TP for hybrid parallelism

**Alternatives Considered:**
- Skip EP: Rejected as it's a stated requirement
- Expert replication: Deferred to future (adds load balancing complexity)

**Implementation Details:**
- ExpertRouter: Maps tokens to experts
- All-to-all communication: Redistribute tokens to expert devices
- Expert placement: Static mapping (expert_id % num_devices)
- Capacity factor: Limit tokens per expert to prevent imbalance

### 9. Chunked Prefill Strategy

**Decision:** Split prefill sequences into 8K token chunks.

**Rationale:**
- Prevents memory spikes from very long inputs
- Enables mixed prefill-decode batching
- Provides predictable latency
- 8K balances memory and computation efficiency

**Alternatives Considered:**
- No chunking: Rejected due to memory unpredictability
- 4K chunks: Rejected as too small (overhead)
- 16K chunks: Rejected as too large for CPU testing

**Implementation Details:**
- ChunkManager: Splits sequences, tracks progress
- Incremental KV cache allocation (page-by-page)
- Interleave chunks with decode steps
- Resume from last chunk on next iteration

### 10. Test Infrastructure

**Decision:** Use pytest with CPU-based testing and Qwen3-0.6B model.

**Rationale:**
- pytest: Industry standard, rich plugin ecosystem
- CPU testing: Accessible on MacBook, no GPU required
- Qwen3-0.6B: Small enough for fast tests, real model behavior
- Gloo backend: Enables distributed testing on CPU

**Alternatives Considered:**
- Mock models: Rejected as doesn't test real inference behavior
- Larger models: Rejected due to slow test execution
- GPU-only tests: Rejected as limits accessibility

**Implementation Details:**
- Fixtures: Shared model loading, tokenizer setup
- Parametrized tests: Test multiple configurations
- Integration tests: End-to-end inference with all features
- Performance benchmarks: Track latency and throughput

## Risks / Trade-offs

### Risk: Custom Qwen3 Implementation Complexity
**Mitigation:**
- Start with single-device implementation, add TP/EP incrementally
- Extensive unit tests for each layer
- Validate outputs against HuggingFace reference implementation

### Risk: CPU Testing May Miss GPU-Specific Issues
**Mitigation:**
- Focus on algorithmic correctness, not performance optimization
- Document GPU-specific optimizations as future work
- Design interfaces to support GPU backends later

### Risk: torch.distributed on CPU May Have Different Behavior
**Mitigation:**
- Test with both Gloo (CPU) and NCCL (GPU) backends where possible
- Document known differences
- Use backend-agnostic communication patterns

### Risk: Paged Attention Overhead on Small Batches
**Trade-off:** Accept overhead for consistency and simplicity
- Paged attention adds indirection cost
- Benefit increases with batch size and sequence length
- Educational value outweighs performance cost for small models

### Risk: RadixCache Memory Overhead
**Trade-off:** Memory for speed
- Radix tree consumes memory for cache metadata
- Configurable cache size with LRU eviction
- Monitor cache hit rate to validate benefit

### Risk: Continuous Batching Complexity
**Mitigation:**
- Comprehensive tests for join/leave scenarios
- Clear state management for active requests
- Logging and metrics for debugging

### Risk: TDD May Slow Initial Development
**Trade-off:** Slower start, faster iteration
- Writing tests first takes more time upfront
- Catches bugs early, reduces debugging time
- Enables confident refactoring

## Migration Plan

N/A - This is a new project with no existing users or systems to migrate.

## Open Questions

1. **Chunked Prefill Chunk Size:** Is 8K optimal for CPU testing, or should it be configurable?
   - **Resolution:** Make configurable with 8K default, allow tuning based on hardware

2. **Expert Parallelism Testing:** How to test EP without MoE model?
   - **Resolution:** Create synthetic MoE layer for testing, document real MoE support as future work

3. **Attention Backend:** Should we support multiple attention implementations (naive, optimized)?
   - **Resolution:** Start with naive PyTorch attention, add FlashAttention as optional backend later

4. **Sampling Strategies:** Which penalties to implement (frequency, presence, repetition)?
   - **Resolution:** Implement all three, they're standard and straightforward

5. **Metrics and Monitoring:** What metrics to expose (latency, throughput, cache hit rate)?
   - **Resolution:** Expose all three plus memory usage, make extensible for custom metrics

## File Organization Architecture

### Top-Level Structure

```
vibe-sgl/
├── vibe_sgl_lite/           # Main Python package
│   ├── __init__.py          # Package initialization, version info
│   ├── core/                # Core inference engine
│   ├── models/              # Custom model implementations
│   ├── memory/              # Memory management subsystem
│   ├── cache/               # Caching mechanisms
│   ├── batch/               # Batching logic
│   ├── sampling/            # Sampling strategies
│   ├── scheduler/           # Request scheduling
│   ├── distributed/         # Parallelism implementations
│   └── utils/               # Utilities and helpers
├── tests/                   # Test suite (mirrors source structure)
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── distributed/         # Distributed tests (TP/EP)
│   ├── fixtures/            # Shared test fixtures
│   └── conftest.py          # pytest configuration
├── examples/                # Usage examples
│   ├── basic_generation.py
│   ├── batch_inference.py
│   ├── streaming.py
│   ├── multi_turn.py
│   └── tensor_parallel.py
├── benchmarks/              # Performance benchmarks
│   ├── latency_benchmark.py
│   ├── throughput_benchmark.py
│   └── cache_benchmark.py
├── docs/                    # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   └── tdd_workflow.md
├── pyproject.toml           # Project configuration and dependencies
├── pytest.ini               # pytest configuration
├── README.md                # Project overview
└── TESTING.md               # Testing guide
```

---

### Core Subsystems

#### 1. Core Engine (`vibe_sgl_lite/core/`)

The core inference engine provides the main API and orchestrates all components.

**Key Files:**
- **`engine.py`**: Main InferenceEngine class, public API entry point
- **`model_runner.py`**: Model execution loop, forward pass coordination
- **`tokenizer_manager.py`**: Tokenization and detokenization
- **`request.py`**: Request data structure and state management
- **`config.py`**: Engine configuration and parameters
- **`io_struct.py`**: Input/output data structures

**Purpose:** Provides high-level API for users to perform inference, coordinates all subsystems.

**Dependencies:** All other subsystems (models, memory, cache, batch, sampling, scheduler)

---

#### 2. Models (`vibe_sgl_lite/models/`)

Custom model implementations with native TP/EP support.

**Key Files:**
- **`qwen3/`**: Qwen3 model implementation
  - **`config.py`**: Qwen3Config for model architecture
  - **`modeling.py`**: Qwen3Model, Qwen3DecoderLayer
  - **`attention.py`**: Qwen3Attention with GQA and RoPE
  - **`mlp.py`**: Qwen3MLP with SwiGLU activation
  - **`layers.py`**: RMSNorm, embedding layers
  - **`rope.py`**: Rotary position embeddings
  - **`weight_loader.py`**: Load and partition HuggingFace weights
- **`base.py`**: Base model interface
- **`registry.py`**: Model registry for extensibility

**Purpose:** Provides custom Qwen3 implementation with TP/EP support built from scratch.

**Dependencies:** distributed/ (for TP/EP layers)

---

#### 3. Memory Management (`vibe_sgl_lite/memory/`)

Memory pool and paged allocator for efficient KV cache management.

**Key Files:**
- **`memory_pool.py`**: MemoryPool class managing free pages
- **`paged_allocator.py`**: PagedAllocator for per-request allocation
- **`page_table.py`**: Page table mapping logical to physical pages
- **`page.py`**: Page data structure and metadata
- **`allocator_interface.py`**: Abstract allocator interface
- **`stats.py`**: Memory usage statistics and tracking

**Purpose:** Eliminates memory fragmentation, enables efficient page sharing.

**Dependencies:** None (low-level subsystem)

---

#### 4. Cache (`vibe_sgl_lite/cache/`)

Prefix caching with radix tree for automatic KV cache reuse.

**Key Files:**
- **`radix_cache.py`**: RadixCache main class
- **`radix_tree.py`**: RadixTreeNode and tree operations
- **`prefix_matcher.py`**: Prefix matching algorithm
- **`eviction_policy.py`**: LRU eviction implementation
- **`cache_metrics.py`**: Cache hit rate and statistics
- **`kv_cache.py`**: KV cache data structure

**Purpose:** Automatically detects and reuses shared prefixes across requests.

**Dependencies:** memory/ (for page management)

---

#### 5. Batch Management (`vibe_sgl_lite/batch/`)

Continuous batching for dynamic request handling.

**Key Files:**
- **`batch_manager.py`**: BatchManager coordinating batch lifecycle
- **`batch.py`**: Batch data structure
- **`prefill_batch.py`**: Prefill phase batch handling
- **`decode_batch.py`**: Decode phase batch handling
- **`padding.py`**: Sequence padding and attention mask generation
- **`chunked_prefill.py`**: ChunkManager for long sequence handling

**Purpose:** Maximizes throughput by allowing requests to join/leave dynamically.

**Dependencies:** memory/, cache/ (for KV cache management)

---

#### 6. Sampling (`vibe_sgl_lite/sampling/`)

Token sampling strategies and generation control.

**Key Files:**
- **`sampling_params.py`**: SamplingParams dataclass
- **`sampler.py`**: Main Sampler class
- **`strategies.py`**: Greedy, top-k, top-p implementations
- **`penalties.py`**: Frequency, presence, repetition penalties
- **`logit_processor.py`**: Logit bias and processing
- **`stop_checker.py`**: Stop sequence detection

**Purpose:** Provides flexible token sampling with various strategies and penalties.

**Dependencies:** None (operates on logits)

---

#### 7. Scheduler (`vibe_sgl_lite/scheduler/`)

Request scheduling policies for batch formation.

**Key Files:**
- **`scheduler.py`**: Main Scheduler class
- **`policy_interface.py`**: SchedulerPolicy abstract base class
- **`fcfs_policy.py`**: First-Come-First-Serve policy
- **`lpm_policy.py`**: Longest Prefix Match (cache-aware) policy
- **`priority_queue.py`**: Priority queue for request ordering
- **`fairness.py`**: Starvation prevention and fairness guarantees
- **`metrics.py`**: Scheduling metrics (latency, throughput)

**Purpose:** Optimizes batch formation for throughput or cache efficiency.

**Dependencies:** cache/ (for LPM policy to query cache state)

---

#### 8. Distributed (`vibe_sgl_lite/distributed/`)

Tensor parallelism and expert parallelism implementations.

**Key Files:**
- **`tp/`**: Tensor Parallelism
  - **`parallel_linear.py`**: ColumnParallelLinear, RowParallelLinear
  - **`process_group.py`**: TP process group initialization
  - **`weight_partition.py`**: Weight partitioning utilities
  - **`communication.py`**: All-reduce, broadcast wrappers
- **`ep/`**: Expert Parallelism
  - **`expert_router.py`**: ExpertRouter for token routing
  - **`expert_placement.py`**: Expert-to-device mapping
  - **`all_to_all.py`**: All-to-all communication for token redistribution
  - **`synthetic_moe.py`**: Synthetic MoE layer for testing
- **`hybrid.py`**: Hybrid TP+EP coordination
- **`backend.py`**: Backend selection (Gloo for CPU, NCCL for GPU)

**Purpose:** Enables distributed inference with TP and EP using torch.distributed.

**Dependencies:** None (low-level, used by models/)

---

#### 9. Utils (`vibe_sgl_lite/utils/`)

Utilities and helper functions.

**Key Files:**
- **`logging.py`**: Logging configuration
- **`metrics.py`**: Metrics collection and reporting
- **`timer.py`**: Performance timing utilities
- **`device.py`**: Device management (CPU/GPU)
- **`validation.py`**: Input validation helpers
- **`constants.py`**: Global constants

**Purpose:** Provides common utilities used across subsystems.

**Dependencies:** None

---

### Test Organization

```
tests/
├── unit/                    # Unit tests (one file per source file)
│   ├── test_memory_pool.py
│   ├── test_radix_cache.py
│   ├── test_qwen3_attention.py
│   └── ...
├── integration/             # Integration tests
│   ├── test_end_to_end.py
│   ├── test_continuous_batching.py
│   ├── test_cache_integration.py
│   └── ...
├── distributed/             # Distributed tests (multi-process)
│   ├── test_tensor_parallel.py
│   ├── test_expert_parallel.py
│   └── test_hybrid_parallel.py
├── fixtures/                # Shared fixtures
│   ├── model_fixtures.py    # Qwen3-0.6B loading
│   ├── tokenizer_fixtures.py
│   └── distributed_fixtures.py
└── conftest.py              # pytest configuration and global fixtures
```

---

### Data Flow Through Architecture

```
User Request
    ↓
InferenceEngine (core/engine.py)
    ↓
TokenizerManager (core/tokenizer_manager.py) - Tokenization
    ↓
Scheduler (scheduler/scheduler.py) - Request prioritization
    ↓
RadixCache (cache/radix_cache.py) - Prefix matching
    ↓
PagedAllocator (memory/paged_allocator.py) - Memory allocation
    ↓
BatchManager (batch/batch_manager.py) - Batch formation
    ↓
ModelRunner (core/model_runner.py) - Forward pass coordination
    ↓
Qwen3Model (models/qwen3/modeling.py) - Model computation
    ├─ Qwen3Attention (with paged KV cache)
    └─ Qwen3MLP
    ↓
Sampler (sampling/sampler.py) - Token sampling
    ↓
TokenizerManager - Detokenization
    ↓
Response Stream
```

---

### Module Dependencies

```
core/
  ├─ depends on: models/, memory/, cache/, batch/, sampling/, scheduler/
  └─ provides: Public API

models/
  ├─ depends on: distributed/
  └─ provides: Model implementations

memory/
  ├─ depends on: (none)
  └─ provides: Memory management primitives

cache/
  ├─ depends on: memory/
  └─ provides: Prefix caching

batch/
  ├─ depends on: memory/, cache/
  └─ provides: Batch management

sampling/
  ├─ depends on: (none)
  └─ provides: Token sampling

scheduler/
  ├─ depends on: cache/
  └─ provides: Request scheduling

distributed/
  ├─ depends on: (none)
  └─ provides: Parallelism primitives

utils/
  ├─ depends on: (none)
  └─ provides: Common utilities
```

This architecture ensures:
- **Clear separation of concerns**: Each subsystem has a single responsibility
- **Testability**: Each module can be tested independently
- **Extensibility**: New features can be added without modifying core logic
- **Educational value**: Code organization mirrors conceptual architecture
