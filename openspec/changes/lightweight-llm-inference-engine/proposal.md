## Why

Building a lightweight, production-ready LLM inference engine that implements core optimization techniques from vibe-sgl while maintaining simplicity and testability. This engine will serve as a learning platform and foundation for understanding advanced LLM serving techniques through strict TDD practices, enabling developers to grasp how modern inference optimizations work from first principles.

## What Changes

- **New Python package** `vibe_sgl_lite` with modular architecture
- **Core inference backbone**: Model loading, tokenization, forward pass, and token generation
- **Memory optimization**: Paged attention for efficient KV cache management
- **Prefix caching**: RadixAttention with radix tree for automatic prefix reuse
- **Batch processing**: Continuous batching for dynamic request handling
- **Scheduling**: Multiple policies (FCFS, LPM) for request prioritization
- **Chunked prefill**: Split long sequences into manageable chunks
- **Tensor Parallelism (TP)**: Distribute model layers across multiple devices with custom Qwen3 implementation
- **Expert Parallelism (EP)**: Support for MoE models with expert distribution across devices
- **Custom Qwen3 model**: Rewritten Qwen3 model definition with TP/EP support
- **Comprehensive test suite**: All features tested with Qwen/Qwen3-0.6B on CPU using PyTorch
- **TDD workflow**: Tests written before implementation for all components
- **Documentation**: English documentation with clear examples and architecture diagrams

## Capabilities

### New Capabilities

- `core-inference-engine`: Basic model loading, tokenization, forward pass, and token generation with sampling
- `paged-attention`: Page-based KV cache allocation and management for memory efficiency
- `radix-attention`: Prefix caching using radix tree data structure for automatic prefix reuse
- `continuous-batching`: Dynamic batch composition allowing requests to join/leave during execution
- `chunked-prefill`: Split long input sequences into chunks for incremental processing
- `scheduling-policies`: Request prioritization strategies (FCFS, LPM, cache-aware)
- `sampling-strategies`: Token sampling with temperature, top-p, top-k, and penalties
- `memory-management`: Memory pool and allocator for efficient KV cache handling
- `tensor-parallelism`: Distribute model computation across multiple devices (column/row parallelism)
- `expert-parallelism`: Expert distribution and routing for MoE models
- `model-qwen3`: Custom Qwen3 model implementation with TP/EP support built from scratch
- `test-infrastructure`: TDD framework with CPU-based testing using Qwen/Qwen3-0.6B

### Modified Capabilities

<!-- No existing capabilities are being modified -->

## Impact

**New Code:**
- `vibe_sgl_lite/` - Main package directory
  - `core/` - Core inference engine (model runner, tokenizer)
  - `models/` - Custom model implementations (Qwen3 with TP/EP support)
  - `memory/` - Memory management (paged allocator, memory pool)
  - `cache/` - Caching mechanisms (radix cache, KV cache)
  - `batch/` - Batching logic (continuous batching, batch scheduler)
  - `sampling/` - Sampling strategies and parameters
  - `scheduler/` - Request scheduling policies
  - `distributed/` - Parallelism implementations (TP, EP)
  - `utils/` - Utilities and helpers
- `tests/` - Comprehensive test suite following TDD
- `examples/` - Usage examples and benchmarks

**Dependencies:**
- PyTorch (CPU mode for testing, with `torch.distributed` for all TP/EP operations)
- Transformers (for tokenization only - model will be custom implemented)
- Qwen/Qwen3-0.6B (test model - custom implementation)
- pytest (testing framework)
- numpy (numerical operations)

**Distributed Implementation:**
- All communication via `torch.distributed` primitives (all-reduce, send/recv, broadcast)
- Process groups for organizing TP/EP workers
- Gloo backend for CPU testing, NCCL backend for GPU production
- No external distributed frameworks required

**Development Approach:**
- Strict TDD: Write tests first, then implementation
- Incremental feature development: Core backbone → Custom Qwen3 → Memory → Caching → Batching → Scheduling → Parallelism
- CPU-only testing on MacBook for accessibility (TP/EP tested with CPU process groups using Gloo backend)
- Modular design for easy understanding and extension
- Custom Qwen3 implementation from scratch to support TP/EP natively
- All distributed operations implemented using `torch.distributed` primitives (no external frameworks)
