# vibe-sgl Overview

## Introduction

vibe-sgl (Structured Generation Language) is a high-performance serving framework for large language models (LLMs). It provides both an intuitive frontend programming interface and a powerful backend runtime engine optimized for efficient model serving.

---

## Core Features

### 1. RadixAttention for Prefix Caching

RadixAttention is vibe-sgl's intelligent caching mechanism that automatically detects and reuses shared prompt prefixes across different requests.

**How it works:**
- Uses a radix tree data structure to store previously computed key-value (KV) cache
- When a new request arrives, the system searches for matching prefixes in the cache
- Shared prefixes are reused, eliminating redundant computation
- Implements LRU (Least Recently Used) eviction when cache is full
- Supports reference counting to prevent premature eviction of active cache entries

**Benefits:**
- Dramatically reduces computation for requests with common prefixes
- Enables efficient multi-turn conversations and batch processing
- Automatic and transparent - no manual cache management required

---

### 2. Continuous Batching

Continuous batching allows the system to dynamically compose and update batches without waiting for all requests to complete.

**How it works:**
- New requests can join a running batch at any time
- Completed requests leave the batch immediately
- The system maintains separate prefill (initial processing) and decode (token generation) batches
- Batches are automatically merged and split based on workload

**Benefits:**
- Maximizes GPU utilization by keeping batches full
- Reduces latency by not waiting for slow requests
- Handles variable-length sequences efficiently

---

### 3. Paged Attention

Paged attention manages KV cache memory using a page-based allocation strategy, similar to virtual memory in operating systems.

**How it works:**
- KV cache is divided into fixed-size pages (typically 16-32 tokens)
- Memory is allocated page-by-page as sequences grow
- Pages can be shared across requests (for prefix caching)
- Freed pages are returned to a memory pool for reuse

**Benefits:**
- Eliminates memory fragmentation
- Enables efficient memory sharing
- Supports variable-length sequences without pre-allocation

---

### 4. Chunked Prefill

Chunked prefill splits long input sequences into smaller chunks for incremental processing.

**How it works:**
- Long prefill sequences are divided into chunks (e.g., 8K-16K tokens)
- Each chunk is processed separately
- Chunks can be interleaved with decode operations
- Memory is allocated incrementally as chunks are processed

**Benefits:**
- Prevents memory spikes from very long inputs
- Enables mixed prefill-decode batching for better GPU utilization
- Provides more predictable latency

---

### 5. Speculative Decoding (EAGLE)

Speculative decoding uses a smaller "draft" model to predict multiple future tokens, which are then verified by the main model in parallel.

**How it works:**
- Draft model generates multiple candidate token sequences (tree structure)
- Main model verifies all candidates in a single forward pass
- Accepted tokens are kept, rejected tokens are discarded
- Process repeats for the next generation step

**Benefits:**
- Speeds up generation by 1.5-3x for compatible models
- No quality degradation - output is identical to standard decoding
- Particularly effective for long-form generation

---

### 6. Advanced Scheduling Policies

vibe-sgl provides multiple scheduling policies to optimize different objectives.

**Cache-Aware Policies:**
- **LPM (Longest Prefix Match)**: Prioritizes requests with the longest cached prefix to maximize cache hits
- **DFS-Weight**: Uses depth-first search with weighting for tree-based scheduling

**Cache-Agnostic Policies:**
- **FCFS (First Come First Serve)**: Processes requests in arrival order
- **LOF (Longest Output First)**: Prioritizes requests expecting longer outputs
- **RANDOM**: Random selection for load balancing

**In-Batch Prefix Caching:**
- Detects shared prefixes within the waiting queue
- Groups requests with common prefixes together
- Increases overall cache hit rate

---

### 7. Constrained Generation

Constrained generation ensures model outputs conform to specific formats or patterns.

**Supported Formats:**
- **JSON Schema**: Generate valid JSON matching a schema
- **Regular Expressions**: Match specific patterns
- **EBNF Grammars**: Follow formal grammar rules
- **Structural Tags**: Generate structured markup

**How it works:**
- Converts constraints into finite state machines (FSM)
- Masks invalid tokens during sampling
- Implements jump-forward optimization to skip deterministic sequences
- Caches compiled grammars for reuse

**Benefits:**
- Guarantees valid output format
- Eliminates post-processing and retry logic
- Supports complex structured generation tasks

---

### 8. Multi-LoRA Batching

Multi-LoRA batching allows serving multiple LoRA (Low-Rank Adaptation) adapters simultaneously in a single batch.

**How it works:**
- Base model is loaded once
- Multiple LoRA adapters are loaded into memory
- Each request specifies which adapter to use
- System efficiently switches between adapters within a batch

**Benefits:**
- Serve multiple fine-tuned models with minimal overhead
- Efficient memory usage (only store adapter weights)
- Dynamic adapter loading and unloading

---

### 9. Tensor Parallelism

Tensor parallelism distributes model layers across multiple GPUs for large models.

**How it works:**
- Model weights are split across GPUs
- Each GPU processes a portion of the computation
- Results are synchronized via all-reduce operations
- Supports both row and column parallelism

**Benefits:**
- Enables serving models larger than single GPU memory
- Increases throughput for large models
- Transparent to the user

---

### 10. Expert Parallelism

Expert parallelism is designed for Mixture-of-Experts (MoE) models.

**How it works:**
- Different experts are placed on different GPUs
- Tokens are routed to appropriate experts
- Expert locations are dynamically updated
- Supports expert replication for load balancing

**Benefits:**
- Efficient serving of MoE models (e.g., Mixtral, DeepSeek)
- Balances load across experts
- Reduces communication overhead

---

## File Organization Architecture

### Top-Level Structure

```
vibe-sgl/
├── python/vibe-sgl/          # Main Python package
│   ├── lang/               # Frontend: Programming interface
│   ├── srt/                # Backend: vibe-sgl Runtime
│   ├── eval/               # Evaluation utilities
│   └── test/               # Testing utilities
├── benchmark/              # Performance benchmarks
├── docker/                 # Docker configurations
├── docs/                   # Documentation
├── examples/               # Example code
└── scripts/                # Utility scripts
```

---

### Frontend: `python/vibe-sgl/lang/`

The frontend provides an intuitive Python API for LLM programming.

**Key Components:**
- **`api.py`**: Public API functions (`gen()`, `select()`, `image()`, etc.)
- **`interpreter.py`**: Executes vibe-sgl programs
- **`compiler.py`**: Compiles vibe-sgl functions
- **`ir.py`**: Intermediate representation
- **`chat_template.py`**: Chat template handling
- **`backend/`**: Connectors to various LLM backends (OpenAI, Anthropic, etc.)

**Purpose:** Enables users to write LLM applications with structured generation, control flow, and multi-modal inputs.

---

### Backend: `python/vibe-sgl/srt/`

The vibe-sgl Runtime (SRT) is the high-performance serving engine.

#### Core Subsystems:

##### **1. Managers (`srt/managers/`)**
Orchestrates request processing and scheduling.

- **`scheduler.py`**: Main scheduler coordinating all operations
- **`schedule_batch.py`**: Batch data structures and request representation
- **`schedule_policy.py`**: Scheduling policies (LPM, FCFS, LOF, etc.)
- **`tokenizer_manager.py`**: Tokenization and detokenization
- **`tp_worker.py`**: Tensor parallel worker coordination
- **`data_parallel_controller.py`**: Data parallelism management
- **`io_struct.py`**: Input/output data structures

##### **2. Memory Cache (`srt/mem_cache/`)**
Advanced caching mechanisms for KV cache management.

- **`radix_cache.py`**: RadixAttention implementation with radix tree
- **`hiradix_cache.py`**: Hierarchical radix cache with host memory offloading
- **`memory_pool.py`**: Memory pool management for KV cache
- **`paged_allocator.py`**: Paged attention allocator
- **`chunk_cache.py`**: Cache for chunked prefill
- **`flush_cache.py`**: Cache flushing utilities

##### **3. Attention Layers (`srt/layers/`)**
Multiple attention backend implementations.

- **`radix_attention.py`**: Main attention interface
- **`attention/`**: Backend-specific implementations
  - FlashInfer, FlashAttention, Triton, PyTorch Native
  - Multi-Latent Attention (MLA), Double Sparsity
  - Automatic backend selection based on hardware

##### **4. Speculative Decoding (`srt/speculative/`)**
EAGLE speculative decoding implementation.

- **`eagle_utils.py`**: Core EAGLE logic and data structures
- **`eagle_worker.py`**: EAGLE worker implementation
- **`build_eagle_tree.py`**: Tree construction for speculative sampling
- **`spec_info.py`**: Speculative algorithm metadata

##### **5. Sampling (`srt/sampling/`)**
Token sampling and generation control.

- **`sampling_params.py`**: Sampling parameters (temperature, top-p, penalties, etc.)
- **`sampling_batch_info.py`**: Batch sampling information
- **`penaltylib/`**: Penalty implementations (frequency, presence, repetition)

##### **6. Constrained Decoding (`srt/constrained/`)**
Structured output generation.

- **`base_grammar_backend.py`**: Grammar backend interface
- **`xgrammar_backend.py`**: XGrammar integration
- **`outlines_backend.py`**: Outlines integration
- **`llguidance_backend.py`**: LLGuidance integration
- **`outlines_jump_forward.py`**: Jump-forward optimization

##### **7. Model Executor (`srt/model_executor/`)**
Core model execution and optimization.

- **`model_runner.py`**: Main model execution loop
- **`cuda_graph_runner.py`**: CUDA graph optimization
- **`forward_batch_info.py`**: Forward pass metadata
- **`expert_location_updater.py`**: Expert parallelism support

##### **8. LoRA Support (`srt/lora/`)**
Multi-LoRA batching implementation.

- **`lora_manager.py`**: LoRA adapter management
- **`lora.py`**: LoRA implementation
- **`layers.py`**: LoRA layers
- **`mem_pool.py`**: LoRA memory pool

##### **9. Disaggregation (`srt/disaggregation/`)**
Prefill-decode disaggregation for large-scale deployments.

- **`prefill.py`**: Prefill server logic
- **`decode.py`**: Decode server logic
- **`kv_events.py`**: KV cache event tracking
- Connection backends for KV transfer

##### **10. Distributed (`srt/distributed/`)**
Distributed training and inference support.

- Tensor parallelism (TP)
- Pipeline parallelism (PP)
- Data parallelism (DP)
- Expert parallelism (EP)

##### **11. Entrypoints (`srt/entrypoints/`)**
Server and engine interfaces.

- **`engine.py`**: Main engine interface
- **`http_server.py`**: HTTP server implementation
- **`openai_api_adapter.py`**: OpenAI-compatible API
- **`verl_engine.py`**: VERL integration

##### **12. Models (`srt/models/`)**
Model architecture implementations.

- Support for 50+ model architectures
- Llama, Mistral, Qwen, DeepSeek, Gemma, Phi, etc.
- Vision-language models (LLaVA)
- Embedding and reward models

---

## Data Flow

```
User Request
    ↓
Tokenizer Manager (tokenization)
    ↓
Scheduler (queue management)
    ↓
Schedule Policy (request prioritization)
    ↓
Memory Allocation (paged allocator)
    ↓
RadixCache (prefix matching)
    ↓
Model Runner (forward pass)
    ↓
Attention Backend (attention computation)
    ↓
Sampler (token sampling with constraints)
    ↓
Detokenizer Manager (streaming detokenization)
    ↓
Response Stream
```

---

## Configuration

vibe-sgl provides extensive configuration options through:

- **`server_args.py`**: 200+ server configuration options
- **`global_config.py`**: Global runtime settings
- **`model_config.py`**: Model-specific configurations

Common configuration categories:
- Model loading and quantization
- Memory management (cache size, page size)
- Scheduling policies and batch sizes
- Parallelism settings (TP, DP, PP, EP)
- Attention backend selection
- LoRA and speculative decoding settings

---

## Summary

vibe-sgl is a production-grade LLM serving framework that combines:

1. **Intelligent Caching**: RadixAttention automatically shares prefixes
2. **Efficient Scheduling**: Continuous batching with multiple policies
3. **Memory Optimization**: Paged attention and chunked prefill
4. **Performance**: Speculative decoding and optimized attention backends
5. **Flexibility**: Multi-LoRA, constrained generation, and disaggregation
6. **Scalability**: Distributed serving with multiple parallelism strategies

The codebase is well-organized with clear separation between:
- **Frontend (`lang/`)**: User-facing programming interface
- **Backend (`srt/`)**: High-performance serving engine
- **Infrastructure**: Benchmarks, examples, and deployment tools

All core features are implemented in Python with performance-critical operations delegated to optimized kernels, making the codebase accessible for research and development while maintaining production-level performance.
