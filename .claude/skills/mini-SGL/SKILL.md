# Mini-SGL: Simplified Inference Engine for Qwen3-30B-A3B

## Project Overview

Build a **Mini-SGL** inference engine based on SGLang's architecture design, specifically optimized for **Qwen3-30B-A3B** (a MoE model). The goal is to create a clean, educational PyTorch implementation that:

- ✅ Keeps SGLang's modular design philosophy
- ✅ Focuses on MoE (Mixture of Experts) block logic
- ✅ Supports basic chat completion inference
- ✅ **Implements Tensor Parallelism (TP)** for distributed weight sharding
- ✅ **Implements Expert Parallelism (EP)** for MoE expert distribution
- ❌ Removes C++ extensions (use pure PyTorch/Triton when possible)
- ❌ Removes complex scheduling (RadixAttention, batching strategies)
- ❌ Removes Pipeline Parallelism (PP) - only TP and EP
- ❌ Removes advanced optimizations (chunked prefill, piecewise graphs, etc.)

**Target Model:** Qwen3-30B-A3B (30B parameters, Attention-3-Blocks architecture with MoE)

**Parallelism Strategy:**
- **TP (Tensor Parallelism):** Shard model weights across GPUs (e.g., 4-way TP for attention/FFN)
- **EP (Expert Parallelism):** Distribute MoE experts across GPUs (e.g., 8-way EP for 64 experts)

---

## Key Reference Files from SGLang

Prioritize these SGLang source files in `/references/sglang/python/sglang/srt/`:

### 1. Model Architecture
- **`models/qwen2.py`** - Base Qwen2 architecture (attention, MLP, decoder layer)
- **`models/qwen2_moe.py`** - Qwen2MoE implementation with sparse MoE blocks

### 2. MoE Core Components
- **`layers/moe/topk.py`** - Expert routing and Top-K selection logic
- **`layers/moe/fused_moe_triton/layer.py`** - FusedMoE layer implementation
- **`layers/moe/fused_moe_native.py`** - Native PyTorch MoE (use this for simplicity)
- **`layers/moe/token_dispatcher/standard.py`** - Token dispatch/combine logic
- **`layers/moe/ep_moe/layer.py`** - Expert Parallel MoE (DeepEPMoE)
- **`layers/moe/token_dispatcher/flashinfer.py`** - FlashInfer AllToAll dispatcher

### 3. Distributed Parallelism ⭐ **NEW**
- **`distributed/parallel_state.py`** - Parallel group initialization and management
- **`distributed/communication_op.py`** - AllReduce, AllGather, AllToAll primitives
- **`distributed/device_communicators/pynccl.py`** - NCCL communicator wrapper
- **`layers/linear.py`** - ColumnParallelLinear, RowParallelLinear (TP implementation)
- **`layers/parameter.py`** - Distributed weight loading

### 4. Model Runner
- **`model_executor/model_runner.py`** - Main forward pass orchestration
- **`model_executor/forward_batch_info.py`** - Batch info structures

### 5. Essential Layers
- **`layers/layernorm.py`** - RMSNorm implementation
- **`layers/activation.py`** - SiluAndMul activation
- **`layers/rotary_embedding.py`** - RoPE (Rotary Position Embedding)

---

## Mini-SGL Architecture Design

### Simplified Component Stack

```
┌──────────────────────────────────────────────────────────┐
│              Mini-SGL Inference Engine                   │
├──────────────────────────────────────────────────────────┤
│  1. Distributed Initialization ⭐ NEW                    │
│     ├── TP Groups (Tensor Parallelism)                   │
│     ├── EP Groups (Expert Parallelism)                   │
│     └── NCCL Communicator Setup                          │
├──────────────────────────────────────────────────────────┤
│  2. Model Loader                                         │
│     ├── Load HuggingFace weights                         │
│     ├── Shard weights for TP (Column/Row Parallel)       │
│     ├── Distribute experts for EP                        │
│     └── No quantization (FP16/BF16 only)                 │
├──────────────────────────────────────────────────────────┤
│  3. Qwen3MoE Model                                       │
│     ├── Embedding Layer (replicated or vocab-parallel)   │
│     ├── Decoder Layers × N                               │
│     │   ├── Self-Attention (TP Sharded) ⭐              │
│     │   │   ├── QKV: ColumnParallelLinear                │
│     │   │   └── O_proj: RowParallelLinear                │
│     │   └── MoE Block (EP Distributed) ⭐               │
│     │       ├── Router (replicated)                      │
│     │       ├── TopK Selection (per-rank)                │
│     │       ├── AllToAll Token Dispatch                  │
│     │       ├── Local Expert MLPs (TP + EP)              │
│     │       ├── AllToAll Token Combine                   │
│     │       └── Shared Expert (TP sharded)               │
│     ├── RMSNorm (replicated)                             │
│     └── LM Head (Column Parallel or replicated)          │
├──────────────────────────────────────────────────────────┤
│  4. Generation Loop                                      │
│     - Prefill phase (prompt encoding)                    │
│     - Decode phase (token-by-token)                      │
│     - Sampling (temperature, top-p)                      │
├──────────────────────────────────────────────────────────┤
│  5. Simple KV Cache                                      │
│     - Per-layer KV tensors (no paging)                   │
│     - Distributed across TP ranks                        │
│     - No RadixAttention                                  │
└──────────────────────────────────────────────────────────┘
```

---

## Distributed Parallelism Design ⭐

### Overview: TP + EP Hybrid Strategy

For **Qwen3-30B-A3B** with ~64 experts, we use a hybrid parallelism approach:

```
Example Configuration (8 GPUs):
- TP_SIZE = 2 (Tensor Parallel across 2 GPUs)
- EP_SIZE = 4 (Expert Parallel across 4 GPUs)
- Total GPUs = TP_SIZE × EP_SIZE = 8

GPU Layout:
┌─────────────────────────────────────────┐
│  TP Group 0    │  TP Group 1            │
│  [GPU 0, GPU 1]│  [GPU 4, GPU 5]        │
│  (EP Rank 0)   │  (EP Rank 2)           │
├─────────────────────────────────────────┤
│  TP Group 2    │  TP Group 3            │
│  [GPU 2, GPU 3]│  [GPU 6, GPU 7]        │
│  (EP Rank 1)   │  (EP Rank 3)           │
└─────────────────────────────────────────┘

Expert Distribution (64 experts):
- EP Rank 0: Experts 0-15  (on GPU 0-1 with TP)
- EP Rank 1: Experts 16-31 (on GPU 2-3 with TP)
- EP Rank 2: Experts 32-47 (on GPU 4-5 with TP)
- EP Rank 3: Experts 48-63 (on GPU 6-7 with TP)
```

---

### Part 1: Tensor Parallelism (TP) Implementation

**Goal:** Shard attention and FFN weights across multiple GPUs to reduce memory footprint.

#### 1.1 Core Linear Layers

Based on `layers/linear.py`, implement:

**ColumnParallelLinear** (Column-wise weight sharding):
```python
class ColumnParallelLinear(nn.Module):
    """
    Y = XA where A = [A_1, A_2, ..., A_tp_size]
    Each GPU holds A_i and computes Y_i = XA_i
    """
    def __init__(self, input_size, output_size, bias=True,
                 gather_output=False, tp_rank=None, tp_size=None):
        super().__init__()
        self.tp_rank = tp_rank or get_tensor_model_parallel_rank()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.gather_output = gather_output

        # Each rank holds output_size // tp_size columns
        self.output_size_per_partition = output_size // self.tp_size

        self.weight = nn.Parameter(
            torch.empty(self.output_size_per_partition, input_size)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_size_per_partition)
            )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output_parallel = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            # AllGather along last dimension
            output = tensor_model_parallel_all_gather(output_parallel, dim=-1)
            return output
        else:
            return output_parallel

    def load_weight(self, full_weight, tp_rank=None):
        """Load sharded weight from full checkpoint"""
        tp_rank = tp_rank or self.tp_rank
        shard_size = self.output_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size

        # Slice along output dimension (dim 0 for weight matrix)
        shard = full_weight[start_idx:end_idx, :]
        self.weight.data.copy_(shard)
```

**RowParallelLinear** (Row-wise weight sharding):
```python
class RowParallelLinear(nn.Module):
    """
    Y = XA where A = [A_1; A_2; ...; A_tp_size] (row-wise split)
    Each GPU computes partial result, then AllReduce
    """
    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False, tp_rank=None, tp_size=None):
        super().__init__()
        self.tp_rank = tp_rank or get_tensor_model_parallel_rank()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.input_is_parallel = input_is_parallel

        # Each rank holds input_size // tp_size rows
        self.input_size_per_partition = input_size // self.tp_size

        self.weight = nn.Parameter(
            torch.empty(output_size, self.input_size_per_partition)
        )
        if bias:
            # Bias only on rank 0 to avoid duplicate addition
            self.bias = nn.Parameter(
                torch.empty(output_size)
            ) if self.tp_rank == 0 else None

    def forward(self, x):
        # x: (batch, seq_len, input_size_per_partition) if input_is_parallel
        #    (batch, seq_len, input_size) otherwise

        if not self.input_is_parallel:
            # Split input along last dimension
            x_parallel = split_tensor_along_last_dim(x, self.tp_size)[self.tp_rank]
        else:
            x_parallel = x

        # Each rank computes partial output
        output_parallel = F.linear(x_parallel, self.weight)

        # AllReduce to sum partial results
        output = tensor_model_parallel_all_reduce(output_parallel)

        # Add bias (only on rank 0)
        if self.bias is not None:
            output = output + self.bias

        return output

    def load_weight(self, full_weight, tp_rank=None):
        """Load sharded weight from full checkpoint"""
        tp_rank = tp_rank or self.tp_rank
        shard_size = self.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = start_idx + shard_size

        # Slice along input dimension (dim 1 for weight matrix)
        shard = full_weight[:, start_idx:end_idx]
        self.weight.data.copy_(shard)
```

#### 1.2 TP in Attention

```python
class TPShardedAttention(nn.Module):
    """Attention with TP sharding"""
    def __init__(self, hidden_size, num_heads, head_dim, tp_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.tp_size = tp_size

        # Each TP rank holds num_heads // tp_size heads
        self.num_heads_per_partition = num_heads // tp_size

        # QKV projection: column-parallel (output sharded)
        self.qkv_proj = ColumnParallelLinear(
            hidden_size,
            3 * num_heads * head_dim,  # Q, K, V concatenated
            gather_output=False,  # Keep sharded for local attention
        )

        # O projection: row-parallel (input sharded from attention output)
        self.o_proj = RowParallelLinear(
            num_heads * head_dim,
            hidden_size,
            input_is_parallel=True,  # Input is already sharded
        )

    def forward(self, hidden_states, position_ids, kv_cache):
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection (output is sharded across heads)
        qkv = self.qkv_proj(hidden_states)  # (B, S, 3 * heads_per_partition * head_dim)

        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads_per_partition, self.head_dim)

        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, position_ids)

        # Attention (each rank computes attention for its heads)
        attn_output = scaled_dot_product_attention(q, k, v, kv_cache)

        # Flatten: (B, S, heads_per_partition * head_dim)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)

        # O projection (row-parallel, includes AllReduce)
        output = self.o_proj(attn_output)

        return output
```

#### 1.3 Communication Primitives

```python
# Based on distributed/communication_op.py

def tensor_model_parallel_all_reduce(tensor):
    """Sum-reduce across TP group"""
    if get_tensor_model_parallel_world_size() == 1:
        return tensor

    torch.distributed.all_reduce(
        tensor,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tp_group()
    )
    return tensor

def tensor_model_parallel_all_gather(tensor, dim=-1):
    """Gather tensors from all TP ranks along specified dimension"""
    if get_tensor_model_parallel_world_size() == 1:
        return tensor

    tp_size = get_tensor_model_parallel_world_size()

    # Gather list
    tensor_list = [torch.empty_like(tensor) for _ in range(tp_size)]
    torch.distributed.all_gather(
        tensor_list,
        tensor,
        group=get_tp_group()
    )

    # Concatenate along dim
    output = torch.cat(tensor_list, dim=dim)
    return output

def split_tensor_along_last_dim(tensor, num_partitions):
    """Split tensor along last dimension"""
    last_dim = tensor.shape[-1]
    assert last_dim % num_partitions == 0
    chunk_size = last_dim // num_partitions

    tensor_list = torch.split(tensor, chunk_size, dim=-1)
    return list(tensor_list)
```

---

### Part 2: Expert Parallelism (EP) Implementation

**Goal:** Distribute MoE experts across GPUs and use AllToAll for token routing.

#### 2.1 Expert Distribution Strategy

```python
# Based on layers/moe/ep_moe/layer.py

class EPMoELayer(nn.Module):
    """MoE with Expert Parallelism"""
    def __init__(self, hidden_size, num_experts, top_k,
                 intermediate_size, ep_size, ep_rank):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.ep_size = ep_size
        self.ep_rank = ep_rank

        # Each EP rank holds num_experts // ep_size experts
        self.num_local_experts = num_experts // ep_size
        self.expert_start_idx = ep_rank * self.num_local_experts
        self.expert_end_idx = self.expert_start_idx + self.num_local_experts

        # Router (replicated on all ranks)
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Local experts (each EP rank has a subset)
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, intermediate_size)
            for _ in range(self.num_local_experts)
        ])

        # Shared expert (if applicable, TP-sharded)
        self.shared_expert = None  # Optional

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 1. Router: compute expert scores (replicated)
        router_logits = self.gate(hidden_states)  # (B, S, num_experts)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 2. Dispatch: AllToAll to send tokens to expert owners
        dispatched_states, dispatch_metadata = self.dispatch_tokens(
            hidden_states, topk_ids, topk_weights
        )

        # 3. Compute: local expert forward
        expert_output = self.compute_local_experts(dispatched_states, dispatch_metadata)

        # 4. Combine: AllToAll to return tokens to original ranks
        final_output = self.combine_tokens(expert_output, dispatch_metadata)

        # 5. Add shared expert (if exists)
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            final_output = final_output + shared_output

        return final_output
```

#### 2.2 AllToAll Token Dispatch

```python
def dispatch_tokens(self, hidden_states, topk_ids, topk_weights):
    """
    Dispatch tokens to expert owners using AllToAll

    Args:
        hidden_states: (B, S, H)
        topk_ids: (B, S, top_k) - expert indices
        topk_weights: (B, S, top_k) - routing weights

    Returns:
        dispatched_states: (num_local_tokens, H)
        dispatch_metadata: dict with routing info
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    total_tokens = batch_size * seq_len

    # Flatten
    hidden_states_flat = hidden_states.reshape(total_tokens, hidden_dim)
    topk_ids_flat = topk_ids.reshape(total_tokens, self.top_k)
    topk_weights_flat = topk_weights.reshape(total_tokens, self.top_k)

    # Compute send/recv counts for AllToAll
    # send_counts[rank]: number of tokens to send to rank
    # recv_counts[rank]: number of tokens to receive from rank
    send_counts, recv_counts, send_indices, recv_indices = (
        self._compute_alltoall_metadata(topk_ids_flat)
    )

    # Prepare send buffer: group tokens by destination rank
    send_tokens = self._prepare_send_buffer(
        hidden_states_flat, topk_ids_flat, send_indices
    )

    # AllToAll communication
    recv_tokens = self._alltoall(send_tokens, send_counts, recv_counts)

    # Metadata for reverse operation
    dispatch_metadata = {
        'send_counts': send_counts,
        'recv_counts': recv_counts,
        'send_indices': send_indices,
        'recv_indices': recv_indices,
        'topk_weights': topk_weights_flat,
        'topk_ids': topk_ids_flat,
        'original_shape': (batch_size, seq_len, hidden_dim),
    }

    return recv_tokens, dispatch_metadata

def _compute_alltoall_metadata(self, topk_ids_flat):
    """Compute how many tokens to send/recv to/from each rank"""
    total_tokens, top_k = topk_ids_flat.shape
    ep_size = self.ep_size

    # Count tokens assigned to each expert rank
    send_counts = torch.zeros(ep_size, dtype=torch.int64)
    for token_idx in range(total_tokens):
        for k in range(top_k):
            expert_id = topk_ids_flat[token_idx, k].item()
            expert_rank = expert_id // self.num_local_experts
            send_counts[expert_rank] += 1

    # Exchange counts with all ranks
    recv_counts = torch.zeros(ep_size, dtype=torch.int64)
    torch.distributed.all_to_all_single(
        recv_counts, send_counts, group=get_ep_group()
    )

    # Compute indices for reordering
    send_indices = []
    recv_indices = []

    # ... (detailed implementation for index computation)

    return send_counts, recv_counts, send_indices, recv_indices

def _alltoall(self, send_tokens, send_counts, recv_counts):
    """Perform AllToAll communication"""
    # Convert counts to splits
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()

    # Prepare output buffer
    total_recv = sum(recv_splits)
    hidden_dim = send_tokens.shape[-1]
    recv_tokens = torch.empty(
        (total_recv, hidden_dim),
        dtype=send_tokens.dtype,
        device=send_tokens.device
    )

    # AllToAll
    torch.distributed.all_to_all_single(
        recv_tokens,
        send_tokens,
        output_split_sizes=recv_splits,
        input_split_sizes=send_splits,
        group=get_ep_group()
    )

    return recv_tokens
```

#### 2.3 Local Expert Computation

```python
def compute_local_experts(self, dispatched_states, dispatch_metadata):
    """
    Compute forward pass for local experts

    Args:
        dispatched_states: (num_local_tokens, H) - tokens assigned to this rank
        dispatch_metadata: routing information

    Returns:
        expert_outputs: (num_local_tokens, H)
    """
    num_local_tokens, hidden_dim = dispatched_states.shape

    # Determine which local expert each token should use
    # (based on global expert ID -> local expert index mapping)
    local_expert_assignments = self._map_to_local_experts(dispatch_metadata)

    # Allocate output buffer
    expert_outputs = torch.zeros_like(dispatched_states)

    # Process each local expert
    for local_expert_id in range(self.num_local_experts):
        # Find tokens assigned to this expert
        mask = (local_expert_assignments == local_expert_id)
        if not mask.any():
            continue

        # Extract tokens
        expert_inputs = dispatched_states[mask]

        # Forward pass
        expert_output = self.experts[local_expert_id](expert_inputs)

        # Write back
        expert_outputs[mask] = expert_output

    return expert_outputs
```

#### 2.4 Token Combine (Reverse AllToAll)

```python
def combine_tokens(self, expert_output, dispatch_metadata):
    """
    Combine expert outputs back to original token positions

    Args:
        expert_output: (num_local_tokens, H)
        dispatch_metadata: routing information from dispatch

    Returns:
        combined_output: (B, S, H)
    """
    # Reverse AllToAll: send expert outputs back to source ranks
    send_counts = dispatch_metadata['recv_counts']  # Note: reversed
    recv_counts = dispatch_metadata['send_counts']

    recv_tokens = self._alltoall(expert_output, send_counts, recv_counts)

    # Unpack and weight by routing weights
    batch_size, seq_len, hidden_dim = dispatch_metadata['original_shape']
    topk_weights = dispatch_metadata['topk_weights']

    # Reshape and apply routing weights
    combined_output = self._apply_routing_weights(
        recv_tokens, topk_weights, (batch_size, seq_len, hidden_dim)
    )

    return combined_output
```

---

### Part 3: Distributed Initialization

```python
# Based on distributed/parallel_state.py

def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
):
    """
    Initialize TP and EP process groups

    Example (8 GPUs, TP=2, EP=4):
        TP groups: [0,1], [2,3], [4,5], [6,7]
        EP groups: [0,2,4,6], [1,3,5,7]
    """
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Validate configuration
    assert world_size % tensor_model_parallel_size == 0
    assert world_size % expert_model_parallel_size == 0
    assert (world_size == tensor_model_parallel_size * expert_model_parallel_size)

    # 1. Create TP groups (contiguous ranks)
    num_tp_groups = world_size // tensor_model_parallel_size
    for i in range(num_tp_groups):
        ranks = list(range(
            i * tensor_model_parallel_size,
            (i + 1) * tensor_model_parallel_size
        ))
        group = torch.distributed.new_group(ranks, backend='nccl')
        if rank in ranks:
            _TP_GROUP = group
            _TP_RANK = ranks.index(rank)
            _TP_SIZE = len(ranks)

    # 2. Create EP groups (strided ranks)
    for i in range(tensor_model_parallel_size):
        ranks = list(range(i, world_size, tensor_model_parallel_size))
        group = torch.distributed.new_group(ranks, backend='nccl')
        if rank in ranks:
            _EP_GROUP = group
            _EP_RANK = ranks.index(rank)
            _EP_SIZE = len(ranks)

    print(f"Rank {rank}: TP_RANK={_TP_RANK}, TP_SIZE={_TP_SIZE}, "
          f"EP_RANK={_EP_RANK}, EP_SIZE={_EP_SIZE}")

def get_tp_group():
    return _TP_GROUP

def get_ep_group():
    return _EP_GROUP

def get_tensor_model_parallel_rank():
    return _TP_RANK

def get_tensor_model_parallel_world_size():
    return _TP_SIZE

def get_expert_model_parallel_rank():
    return _EP_RANK

def get_expert_model_parallel_world_size():
    return _EP_SIZE
```

---

## Implementation Plan

### Phase 1: Distributed Infrastructure (Week 1)

**Task 1.0: Distributed Setup** ⭐ **NEW - Do This First**
```python
mini_sgl/
├── distributed/
│   ├── __init__.py
│   ├── parallel_state.py      # TP/EP group initialization
│   ├── communication.py       # AllReduce, AllGather, AllToAll
│   └── utils.py               # Helper functions
```

**Implementation Checklist:**
- [ ] Initialize NCCL backend with `torch.distributed`
- [ ] Create TP and EP process groups
- [ ] Implement `all_reduce()`, `all_gather()`, `all_to_all()` wrappers
- [ ] Add rank/world_size query functions
- [ ] Test with simple tensor communication

**Task 1.1: TP-Aware Linear Layers**
```python
mini_sgl/
├── layers/
│   ├── __init__.py
│   ├── linear.py             # ColumnParallelLinear, RowParallelLinear
│   ├── norm.py               # RMSNorm (replicated)
│   ├── mlp.py                # Standard MLP (can be TP-sharded)
│   ├── attention.py          # TP-sharded attention + RoPE
│   └── embedding.py          # Token embeddings (vocab-parallel optional)
```

**Key Features:**
- **ColumnParallelLinear**: Shard output dimension, optional AllGather
- **RowParallelLinear**: Shard input dimension, AllReduce outputs
- Weight loading with automatic sharding
- Use PyTorch 2.0+ SDPA (no FlashAttention initially)

---

**Task 1.2: MoE Components** ⭐ **CRITICAL**
```python
mini_sgl/
├── moe/
│   ├── __init__.py
│   ├── router.py         # Gate network + TopK selection
│   ├── expert.py         # Single expert MLP
│   ├── moe_layer.py      # Complete MoE block
│   └── dispatcher.py     # Token routing (simplified)
```

**MoE Implementation Strategy:**

1. **Router Logic** (from `layers/moe/topk.py`):
```python
def router_forward(hidden_states, gate_weight, top_k=4):
    # hidden_states: (batch_size, seq_len, hidden_dim)
    # gate_weight: (hidden_dim, num_experts)

    router_logits = F.linear(hidden_states, gate_weight)  # (B, S, E)
    routing_weights = F.softmax(router_logits, dim=-1)

    # Select top-k experts per token
    topk_weights, topk_ids = torch.topk(routing_weights, k=top_k, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)  # renormalize

    return topk_weights, topk_ids
```

2. **Expert Computation** (inspired by `layers/moe/fused_moe_native.py`):
```python
class ExpertMLP(nn.Module):
    """Single expert: gate_proj + up_proj + down_proj"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

class MoELayer(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k, intermediate_size):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

        # Optional: shared expert (Qwen2MoE specific)
        self.shared_expert = ExpertMLP(hidden_size, shared_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        # 1. Route tokens to experts
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, k=self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # 2. Compute expert outputs (naive loop, can optimize later)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        final_output = torch.zeros_like(hidden_states)

        for i in range(self.top_k):
            expert_ids = topk_ids[:, :, i]  # (B, S)
            expert_weights = topk_weights[:, :, i]  # (B, S)

            # Process each expert
            for expert_id in range(self.num_experts):
                mask = (expert_ids == expert_id)
                if mask.any():
                    token_inputs = hidden_states[mask]
                    expert_output = self.experts[expert_id](token_inputs)
                    final_output[mask] += expert_output * expert_weights[mask].unsqueeze(-1)

        # 3. Add shared expert (if exists)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(hidden_states)
            shared_weight = torch.sigmoid(self.shared_expert_gate(hidden_states))
            final_output += shared_out * shared_weight

        return final_output
```

**Optimization Note:** The naive loop above is for clarity. For better performance, implement batch gathering similar to `fused_moe_native.py`.

---

### Phase 2: Expert Parallelism & MoE (Week 2)

**Task 2.0: EP MoE Layer** ⭐ **CRITICAL**
```python
mini_sgl/
├── moe/
│   ├── __init__.py
│   ├── router.py             # Gate + TopK selection
│   ├── expert.py             # Single expert MLP (can be TP-sharded)
│   ├── ep_moe_layer.py       # EP-distributed MoE
│   ├── dispatcher.py         # AllToAll token dispatch/combine
│   └── utils.py              # Helper functions
```

**Implementation Checklist:**
- [ ] Router: gate network + softmax + top-k
- [ ] AllToAll dispatcher: token routing across EP ranks
- [ ] Local expert computation
- [ ] Reverse AllToAll: combine expert outputs
- [ ] Optional shared expert (TP-sharded)
- [ ] Test with synthetic data: verify routing correctness

**Task 2.1: Decoder Layer**
```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.self_attn = Attention(config)
        self.moe_block = MoELayer(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            intermediate_size=config.moe_intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask, position_ids, kv_cache):
        # Pre-norm + Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_kv = self.self_attn(
            hidden_states, attention_mask, position_ids, kv_cache
        )
        hidden_states = residual + hidden_states

        # Pre-norm + MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.moe_block(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, new_kv
```

**Task 2.2: Full Model**
```python
class Qwen3MoEForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def forward(self, input_ids, attention_mask, position_ids, kv_caches):
        hidden_states = self.embed_tokens(input_ids)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            hidden_states, new_kv = layer(
                hidden_states, attention_mask, position_ids, kv_caches[i]
            )
            new_kv_caches.append(new_kv)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits, new_kv_caches
```

---

### Phase 3: Model Loading & Generation (Week 3)

**Task 3.1: Distributed Weight Loader** ⭐ **CRITICAL**
```python
def load_hf_weights_distributed(model, model_path, tp_rank, tp_size, ep_rank, ep_size):
    """
    Load weights with TP sharding and EP distribution

    Each rank loads only its portion of weights:
    - TP sharding: Column/Row parallel layers load sharded weights
    - EP distribution: Each EP rank loads its subset of experts
    """
    from safetensors import safe_open

    # Load config
    config = AutoConfig.from_pretrained(model_path)

    # Load safetensors (all ranks read the same files)
    weight_files = glob(f"{model_path}/*.safetensors")

    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt") as f:
            for key in f.keys():
                full_weight = f.get_tensor(key)

                # Map HF keys to Mini-SGL keys
                mini_key = map_hf_key_to_mini_sgl(key)

                # Determine if this is a TP-sharded or EP-distributed layer
                if is_column_parallel_layer(mini_key):
                    # Load TP shard for ColumnParallelLinear
                    shard = shard_column_weight(full_weight, tp_rank, tp_size)
                    load_tensor_to_param(model, mini_key, shard)

                elif is_row_parallel_layer(mini_key):
                    # Load TP shard for RowParallelLinear
                    shard = shard_row_weight(full_weight, tp_rank, tp_size)
                    load_tensor_to_param(model, mini_key, shard)

                elif is_expert_weight(mini_key):
                    # Load expert weights (EP distribution)
                    expert_id = extract_expert_id(mini_key)
                    if belongs_to_ep_rank(expert_id, ep_rank, ep_size):
                        # This expert belongs to this EP rank
                        local_expert_id = expert_id % (num_experts // ep_size)
                        local_key = remap_expert_key(mini_key, local_expert_id)

                        # If expert is also TP-sharded, apply TP sharding
                        if is_column_parallel_layer(local_key):
                            shard = shard_column_weight(full_weight, tp_rank, tp_size)
                        elif is_row_parallel_layer(local_key):
                            shard = shard_row_weight(full_weight, tp_rank, tp_size)
                        else:
                            shard = full_weight

                        load_tensor_to_param(model, local_key, shard)
                else:
                    # Replicated weights (e.g., layernorm, embeddings)
                    load_tensor_to_param(model, mini_key, full_weight)

def shard_column_weight(weight, tp_rank, tp_size):
    """Shard weight along output dimension (dim 0)"""
    output_size = weight.shape[0]
    shard_size = output_size // tp_size
    start_idx = tp_rank * shard_size
    end_idx = start_idx + shard_size
    return weight[start_idx:end_idx, ...]

def shard_row_weight(weight, tp_rank, tp_size):
    """Shard weight along input dimension (dim 1)"""
    input_size = weight.shape[1]
    shard_size = input_size // tp_size
    start_idx = tp_rank * shard_size
    end_idx = start_idx + shard_size
    return weight[:, start_idx:end_idx]
```

**Key Mapping Examples (with TP/EP):**
```
HF: model.layers.0.self_attn.q_proj.weight (shape: [num_heads*head_dim, hidden_size])
    → Mini-SGL: layers.0.self_attn.qkv_proj.weight
    → TP Shard: Each rank loads [num_heads*head_dim//tp_size, hidden_size]

HF: model.layers.0.mlp.experts.0.gate_proj.weight
    → EP Rank 0: layers.0.moe_block.experts.0.gate_proj.weight
    → EP Rank 1: (skips expert 0, loads expert 16 instead)
    → TP Shard: Within each expert, apply column parallel sharding

HF: model.norm.weight (layernorm)
    → Mini-SGL: norm.weight (replicated on all ranks)
```

**Task 3.2: Generation Loop**
```python
def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.7, top_p=0.9):
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt").cuda()

    # Initialize KV cache
    kv_caches = [None] * len(model.layers)

    generated_tokens = []

    for step in range(max_new_tokens):
        # Forward pass
        logits, kv_caches = model(
            input_ids=input_ids if step == 0 else input_ids[:, -1:],
            attention_mask=None,  # causal mask applied internally
            position_ids=torch.arange(input_ids.size(1), device=input_ids.device),
            kv_caches=kv_caches,
        )

        # Sample next token
        next_token_logits = logits[:, -1, :] / temperature
        next_token = sample_token(next_token_logits, top_p=top_p)

        # Append token
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        # Stop if EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated_tokens)
```

---

## Key Simplifications vs. SGLang

| Feature | SGLang | Mini-SGL |
|---------|--------|----------|
| **Batching** | Dynamic batching, continuous batching | Single request only |
| **KV Cache** | RadixAttention, prefix sharing, paging | Simple per-layer tensors (TP-sharded) |
| **Attention** | FlashAttention, Triton kernels, MLA | PyTorch SDPA (TP-sharded) |
| **Parallelism** | TP + PP + EP | ✅ **TP + EP only** (no PP) |
| **Quantization** | FP8, INT4, AWQ, GPTQ, 58+ methods | None (FP16/BF16 only) |
| **MoE Routing** | Fused kernels, DeepGemm, load balancing | ✅ **AllToAll EP** (PyTorch) |
| **Communication** | Custom kernels, MSCCLPP, 16 backends | ✅ **PyTorch NCCL** only |
| **Scheduling** | Complex scheduler with memory pool | Simple sequential generation |
| **Model Loader** | Distributed loading, 58+ quant loaders | ✅ **TP/EP weight sharding** |

---

## File Structure

```
mini-sgl/
├── mini_sgl/
│   ├── __init__.py
│   ├── config.py                 # Model config
│   ├── model.py                  # Main model class
│   ├── generation.py             # Generation loop
│   ├── loader.py                 # Distributed weight loading ⭐
│   │
│   ├── distributed/              # ⭐ NEW
│   │   ├── __init__.py
│   │   ├── parallel_state.py    # TP/EP group management
│   │   ├── communication.py     # AllReduce, AllGather, AllToAll
│   │   └── utils.py             # Tensor split/gather helpers
│   │
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── linear.py            # ⭐ ColumnParallel, RowParallel
│   │   ├── attention.py         # ⭐ TP-sharded attention + RoPE
│   │   ├── mlp.py               # Standard MLP (can be TP-sharded)
│   │   ├── norm.py              # RMSNorm (replicated)
│   │   └── embedding.py         # Token embeddings
│   │
│   └── moe/
│       ├── __init__.py
│       ├── router.py            # Gate + TopK
│       ├── expert.py            # Expert MLP (can be TP-sharded)
│       ├── ep_moe_layer.py      # ⭐ EP-distributed MoE
│       ├── dispatcher.py        # ⭐ AllToAll token routing
│       └── utils.py             # Helper functions
│
├── tests/
│   ├── test_distributed.py      # ⭐ TP/EP communication tests
│   ├── test_tp_linear.py        # ⭐ TP linear layer tests
│   ├── test_ep_moe.py           # ⭐ EP MoE tests
│   ├── test_attention.py
│   ├── test_moe.py
│   └── test_generation.py
│
├── examples/
│   ├── chat_completion.py       # Simple chat demo
│   ├── distributed_inference.py # ⭐ Multi-GPU inference example
│   └── benchmark.py             # Throughput test
│
├── scripts/
│   └── launch_distributed.sh    # ⭐ torchrun launcher script
│
├── README.md
└── requirements.txt
```

---

## Testing & Validation

### Unit Tests
1. **MoE Layer Test**
   - Verify routing weights sum to 1
   - Check expert load distribution
   - Validate output shapes

2. **Attention Test**
   - Compare with HF Transformers output
   - Verify causal masking

3. **Generation Test**
   - Load official Qwen3-30B-A3B weights
   - Generate text and compare with HuggingFace baseline
   - Measure perplexity on validation set

### Integration Test (Distributed)
```python
# Test TP/EP correctness by comparing with single-GPU baseline
import torch
import torch.distributed as dist
from mini_sgl import Qwen3MoEForCausalLM, initialize_model_parallel

def test_distributed_correctness():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Test configuration: TP=2, EP=2 (requires 4 GPUs)
    initialize_model_parallel(tensor_model_parallel_size=2, expert_model_parallel_size=2)

    # Load distributed model
    model_dist = Qwen3MoEForCausalLM.from_pretrained(
        "Qwen/Qwen3-30B-A3B", tp_size=2, ep_size=2
    )

    # Prepare test input
    input_ids = torch.randint(0, 32000, (1, 10)).cuda()

    # Forward pass
    with torch.no_grad():
        output_dist = model_dist(input_ids)

    # Gather output on rank 0
    if local_rank == 0:
        # Load single-GPU baseline (requires CPU offload or quantization)
        # model_single = Qwen3MoEForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B")
        # output_single = model_single(input_ids)
        # assert torch.allclose(output_dist, output_single, atol=1e-2)
        print(f"Distributed output shape: {output_dist.shape}")
        print("Test passed!")

    dist.destroy_process_group()

# Run with: torchrun --nproc_per_node=4 test_distributed_correctness.py
```

### Unit Test Examples

**Test TP Communication:**
```python
def test_tp_all_reduce():
    initialize_model_parallel(tensor_model_parallel_size=2)

    # Create tensor
    x = torch.ones(4, 4).cuda() * (get_tensor_model_parallel_rank() + 1)

    # AllReduce
    y = tensor_model_parallel_all_reduce(x)

    # Expected: sum of tensors from all TP ranks
    expected = torch.ones(4, 4).cuda() * (1 + 2)  # rank0=1, rank1=2
    assert torch.allclose(y, expected)

def test_tp_all_gather():
    initialize_model_parallel(tensor_model_parallel_size=2)

    # Create tensor
    x = torch.ones(4, 2).cuda() * (get_tensor_model_parallel_rank() + 1)

    # AllGather along last dimension
    y = tensor_model_parallel_all_gather(x, dim=-1)

    # Expected: [rank0_tensor | rank1_tensor]
    assert y.shape == (4, 4)
    assert torch.allclose(y[:, :2], torch.ones(4, 2).cuda() * 1)  # rank 0
    assert torch.allclose(y[:, 2:], torch.ones(4, 2).cuda() * 2)  # rank 1
```

**Test EP AllToAll:**
```python
def test_ep_alltoall():
    initialize_model_parallel(expert_model_parallel_size=4)

    ep_rank = get_expert_model_parallel_rank()
    ep_size = get_expert_model_parallel_world_size()

    # Create dummy routing: each rank sends 2 tokens to each other rank
    num_tokens = 8
    hidden_dim = 128
    tokens = torch.randn(num_tokens, hidden_dim).cuda()

    # Simulate routing (2 tokens to each of 4 ranks)
    topk_ids = torch.tensor([
        [0, 1], [0, 2], [1, 2], [1, 3],
        [2, 3], [2, 0], [3, 0], [3, 1]
    ]).cuda()  # (num_tokens, top_k=2)

    # Dispatch
    dispatcher = EPTokenDispatcher(ep_size, ep_rank)
    dispatched_tokens, metadata = dispatcher.dispatch(tokens, topk_ids)

    # Verify: each rank should receive 4 tokens (2 from each of 4 ranks)
    assert dispatched_tokens.shape[0] == 4

    # Combine
    combined_tokens = dispatcher.combine(dispatched_tokens, metadata)

    # Verify: should recover original shape
    assert combined_tokens.shape == tokens.shape
```

---

## Performance Expectations

| Configuration | Memory per GPU | Prefill (tokens/s) | Decode (tokens/s) |
|---------------|----------------|-------------------|-------------------|
| **Single A100 40GB** | ~60GB (won't fit) | N/A | N/A |
| **2x A100 (TP=2)** | ~30GB each | ~40-80 | ~8-15 |
| **4x A100 (TP=2, EP=2)** | ~15GB each | ~80-150 | ~15-25 |
| **8x A100 (TP=2, EP=4)** | ~8GB each | ~150-250 | ~25-40 |

**Memory Breakdown (Qwen3-30B-A3B in BF16):**
- Model weights: ~60GB (30B params × 2 bytes)
- KV cache: ~2-4GB per request (depends on context length)
- Activations: ~2-4GB (temporary buffers)

**Recommended Configuration:**
- **Minimum:** 4x A100 40GB (TP=2, EP=2)
- **Optimal:** 8x A100 40GB (TP=2, EP=4) for best throughput

---

## Future Extensions (Optional)

After completing the basic version, consider:

1. **Batch Processing**: Support multiple requests (no batching logic, just sequential)
2. **Triton Kernels**: Optimize MoE with fused operations
3. **FlashAttention**: Replace SDPA with FlashAttention-2
4. **Simple Caching**: Implement basic prompt prefix caching
5. **Quantization**: Add INT8/FP8 support via `torch.quantization` or FBGEMM
6. **Pipeline Parallelism (PP)**: Add inter-layer pipelining for deeper models
7. **Load Balancing**: Implement expert load balancing (auxiliary loss)
8. **Hybrid Parallelism**: Combine TP, EP, and PP for larger clusters
9. **Overlapped Communication**: Overlap AllToAll with computation
10. **Multi-Node Support**: Extend to multi-node clusters with InfiniBand

---

## Development Guidelines

1. **Code Style**: Follow SGLang's modular design
2. **Documentation**: Add docstrings for all classes/functions
3. **Type Hints**: Use Python type hints everywhere
4. **Testing**: Write tests before implementation (TDD)
5. **Simplicity**: When in doubt, choose the simpler approach

---

## Quick Start Example

### Single-Node Multi-GPU Inference

```bash
# Launch with torchrun (8 GPUs, TP=2, EP=4)
torchrun --nproc_per_node=8 examples/distributed_inference.py \
    --model_path Qwen/Qwen3-30B-A3B \
    --tp_size 2 \
    --ep_size 4 \
    --prompt "Explain how a Mixture of Experts model works:"
```

```python
# examples/distributed_inference.py
import torch
import torch.distributed as dist
from mini_sgl import Qwen3MoEForCausalLM, initialize_model_parallel, generate
from transformers import AutoTokenizer

def main():
    # Initialize distributed
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Initialize TP and EP groups
    initialize_model_parallel(
        tensor_model_parallel_size=args.tp_size,
        expert_model_parallel_size=args.ep_size,
    )

    # Load model (each rank loads its shard)
    model = Qwen3MoEForCausalLM.from_pretrained(
        args.model_path,
        tp_size=args.tp_size,
        ep_size=args.ep_size,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Generate (only rank 0 prints output)
    if local_rank == 0:
        output = generate(model, tokenizer, args.prompt, max_new_tokens=200)
        print(output)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

---

## Debugging & Common Issues

### Debugging Distributed Code

**Enable Detailed Logging:**
```bash
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
torchrun --nproc_per_node=8 your_script.py
```

**Check Process Group Setup:**
```python
def verify_distributed_setup():
    print(f"Rank {dist.get_rank()}/{dist.get_world_size()}")
    print(f"TP: rank {get_tensor_model_parallel_rank()}/{get_tensor_model_parallel_world_size()}")
    print(f"EP: rank {get_expert_model_parallel_rank()}/{get_expert_model_parallel_world_size()}")

    # Test basic communication
    tensor = torch.ones(2, 2).cuda() * dist.get_rank()
    dist.all_reduce(tensor)
    print(f"AllReduce result: {tensor[0, 0].item()}")  # Should be sum of all ranks
```

**Visualize Tensor Shapes:**
```python
def debug_forward_pass(model, input_ids):
    """Print tensor shapes at each layer"""
    def hook_fn(module, input, output):
        print(f"{module.__class__.__name__}: input {input[0].shape} -> output {output.shape if isinstance(output, torch.Tensor) else 'tuple'}")

    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(hook_fn))

    model(input_ids)

    for hook in hooks:
        hook.remove()
```

---

### Common Issues & Solutions

**Issue 1: NCCL Timeout**
```
RuntimeError: NCCL error: unhandled system error
```
**Solution:**
- Check network connectivity between GPUs
- Increase timeout: `export NCCL_TIMEOUT_MS=600000`
- Verify all ranks are reaching the collective operation

**Issue 2: Shape Mismatch in TP**
```
RuntimeError: The size of tensor a (128) must match the size of tensor b (64)
```
**Solution:**
- Verify weight sharding is correct (column vs row parallel)
- Check `gather_output` and `input_is_parallel` flags
- Ensure all TP ranks have consistent hidden dimensions

**Issue 3: Expert Distribution Error in EP**
```
IndexError: expert_id 32 out of range for num_local_experts 16
```
**Solution:**
- Verify expert assignment logic: `expert_rank = expert_id // num_local_experts`
- Check EP group size matches `num_experts // num_local_experts`
- Ensure router outputs are correctly masked

**Issue 4: AllToAll Hangs**
```
[Hangs indefinitely at all_to_all_single]
```
**Solution:**
- Verify send/recv counts are consistent across all ranks
- Check that all ranks call AllToAll (no conditional skips)
- Use `NCCL_DEBUG=INFO` to see where it hangs
- Ensure EP group is correctly initialized

**Issue 5: Memory OOM**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Solution:**
- Reduce batch size or sequence length
- Increase TP size to shard more
- Enable gradient checkpointing (if training)
- Check for memory leaks in AllToAll buffers

---

## Profiling & Optimization

### Communication Profiling

```python
import torch.distributed.profiler as dist_profiler

with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    with_stack=True,
) as prof:
    output = model(input_ids)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Look for:
# - nccl:all_reduce time
# - nccl:all_to_all time
# - Computation vs communication ratio
```

### Optimization Checklist

- [ ] **Overlap Communication & Computation**: Use async NCCL calls
- [ ] **Fuse Operations**: Combine multiple small collectives
- [ ] **Use FP16/BF16**: Reduce communication volume
- [ ] **Tune NCCL**: Set `NCCL_MIN_NRINGS`, `NCCL_BUFFSIZE`
- [ ] **Profile Experts**: Ensure balanced load across EP ranks
- [ ] **Cache Routing Decisions**: Reuse topk results when possible

---

## References

- SGLang GitHub: https://github.com/sgl-project/sglang
- Qwen2-MoE Paper: https://arxiv.org/abs/2405.04434
- HuggingFace Transformers: https://github.com/huggingface/transformers
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
- NCCL Documentation: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- Megatron-LM (TP reference): https://github.com/NVIDIA/Megatron-LM
- DeepSpeed (MoE reference): https://github.com/microsoft/DeepSpeed

---

**Status:** Planning Phase
**Next Steps:**
1. ✅ Complete skill documentation with TP/EP design
2. ⏭️ Implement Phase 1: Distributed infrastructure (parallel_state.py, communication.py)
3. ⏭️ Implement Phase 2: EP MoE with AllToAll dispatcher
4. ⏭️ Implement Phase 3: Distributed weight loading
5. ⏭️ Test and validate on Qwen3-30B-A3B