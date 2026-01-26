# Mini-SGL: Simplified Inference Engine for Qwen3-30B-A3B

A clean, educational PyTorch implementation of a distributed inference engine for MoE models, based on SGLang's architecture.

## Features

- ✅ **Tensor Parallelism (TP)**: Shard model weights across GPUs
- ✅ **Expert Parallelism (EP)**: Distribute MoE experts across GPUs
- ✅ **Clean PyTorch Implementation**: No C++ extensions
- ✅ **Educational**: Clear, well-documented code

## Architecture

```
Mini-SGL (TP=2, EP=4 on 8 GPUs)
├── Distributed Infrastructure (NCCL + PyTorch Distributed)
├── TP-Sharded Attention (ColumnParallel QKV + RowParallel O)
├── EP-Distributed MoE (AllToAll token routing)
└── Simple Generation Loop (Prefill + Decode)
```

## Installation

```bash
# Clone repository
cd Mini-SGL

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Single-GPU Testing (CPU/MacBook)
```bash
# Test basic components (no GPU required)
python -m pytest tests/test_layers.py
python -m pytest tests/test_moe.py
```

### Multi-GPU Inference (Linux with CUDA)
```bash
# Launch with 8 GPUs (TP=2, EP=4)
torchrun --nproc_per_node=8 examples/distributed_inference.py \
    --model_path Qwen/Qwen3-30B-A3B \
    --tp_size 2 \
    --ep_size 4 \
    --prompt "Explain how Mixture of Experts works"
```

## Project Structure

```
mini_sgl/
├── distributed/       # TP/EP parallelism infrastructure
├── layers/           # TP-aware linear, attention, norm layers
├── moe/              # EP-distributed MoE with AllToAll routing
├── config.py         # Model configuration
├── model.py          # Main Qwen3MoE model
├── loader.py         # Distributed weight loading
└── generation.py     # Generation loop

tests/                # Unit tests and integration tests
examples/             # Usage examples
docs/                 # Additional documentation
```

## Testing Environment Requirements

See [TESTING_REQUIREMENTS.md](docs/TESTING_REQUIREMENTS.md) for detailed hardware requirements.

**Summary:**
- **Local MacBook**: Unit tests for layers, MoE logic (CPU only)
- **Single CUDA GPU**: Basic forward pass, small models
- **Multi-GPU (4-8 A100s)**: Full distributed tests, Qwen3-30B-A3B

## Development

```bash
# Run all local tests (no GPU)
python -m pytest tests/ -m "not gpu"

# Run GPU tests (requires CUDA)
python -m pytest tests/ -m "gpu"

# Run distributed tests (requires multi-GPU)
bash scripts/test_distributed.sh
```

## Performance

| Configuration | Memory/GPU | Prefill (tok/s) | Decode (tok/s) |
|--------------|------------|----------------|---------------|
| 4x A100 (TP=2, EP=2) | ~15GB | ~80-150 | ~15-25 |
| 8x A100 (TP=2, EP=4) | ~8GB  | ~150-250 | ~25-40 |

## References

- [SGLang](https://github.com/sgl-project/sglang) - Original inspiration
- [Qwen2-MoE Paper](https://arxiv.org/abs/2405.04434)
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html)

## License

Apache 2.0

## Status

🚧 **Work in Progress** - Phase 1: Distributed Infrastructure
