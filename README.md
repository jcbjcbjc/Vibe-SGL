# vibe-sgl-lite

A lightweight, production-ready LLM inference engine implementing advanced optimization techniques for efficient large language model serving.

## Overview

vibe-sgl-lite is an educational and production-ready inference engine that demonstrates modern LLM serving optimizations from first principles. Built with strict Test-Driven Development (TDD) practices, it provides a clear, modular implementation of techniques used in state-of-the-art serving systems.

### Key Features

- **Core Inference Engine**: Model loading, tokenization, forward pass, and autoregressive token generation
- **Paged Attention**: Efficient KV cache management using page-based memory allocation
- **Prefix Caching (RadixAttention)**: Automatic prefix reuse with radix tree data structure
- **Continuous Batching**: Dynamic request batching for optimal throughput
- **Chunked Prefill**: Split long sequences into manageable chunks for incremental processing
- **Scheduling Policies**: Multiple strategies (FCFS, LPM, cache-aware) for request prioritization
- **Sampling Strategies**: Flexible token sampling with temperature, top-p, top-k, and penalties
- **Tensor Parallelism (TP)**: Distribute model computation across multiple devices
- **Expert Parallelism (EP)**: Support for Mixture-of-Experts (MoE) models
- **Custom Qwen3 Implementation**: Native TP/EP support built from scratch

### Architecture

```
vibe_sgl_lite/
├── core/          # Core inference engine (model runner, tokenizer)
├── models/        # Custom model implementations (Qwen3 with TP/EP)
├── memory/        # Memory management (paged allocator, memory pool)
├── cache/         # Caching mechanisms (radix cache, KV cache)
├── batch/         # Batching logic (continuous batching, scheduler)
├── sampling/      # Sampling strategies and parameters
├── scheduler/     # Request scheduling policies
├── distributed/   # Parallelism implementations (TP, EP)
└── utils/         # Utilities and helpers
```

## Why vibe-sgl-lite?

This project serves dual purposes:

1. **Learning Platform**: Understand how modern LLM inference optimizations work by studying clean, well-tested implementations
2. **Production Foundation**: Use as a lightweight inference engine for applications requiring efficient LLM serving

Unlike heavyweight frameworks, vibe-sgl-lite prioritizes:
- **Simplicity**: Clear, readable code over complex abstractions
- **Testability**: 100% test coverage with CPU-based testing
- **Modularity**: Each optimization is a separate, composable component
- **Educational Value**: Learn by reading and extending the codebase

## Requirements

- Python 3.9 or higher
- PyTorch 2.0.0 or higher
- Transformers 4.30.0 or higher (for tokenization)

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/vibe-sgl-lite.git
cd vibe-sgl-lite

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

### Using pip (once published)

```bash
pip install vibe-sgl-lite
```

## Quick Start

### Basic Text Generation

```python
from vibe_sgl_lite.core import ModelRunner

# Initialize the model
runner = ModelRunner(model_path="Qwen/Qwen3-0.6B")

# Generate text
output = runner.generate(
    prompt="Once upon a time",
    max_tokens=50,
    temperature=0.7
)

print(output)
```

### Batch Inference

```python
from vibe_sgl_lite.core import ModelRunner

runner = ModelRunner(model_path="Qwen/Qwen3-0.6B")

# Process multiple prompts in a batch
prompts = [
    "The capital of France is",
    "Machine learning is",
    "Python programming language"
]

outputs = runner.generate_batch(
    prompts=prompts,
    max_tokens=30,
    temperature=0.8
)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output}\n")
```

### Streaming Generation

```python
from vibe_sgl_lite.core import ModelRunner

runner = ModelRunner(model_path="Qwen/Qwen3-0.6B")

# Stream tokens as they are generated
for token in runner.generate_stream(
    prompt="Explain quantum computing in simple terms:",
    max_tokens=100
):
    print(token, end="", flush=True)
```

### Using Advanced Features

```python
from vibe_sgl_lite.core import ModelRunner
from vibe_sgl_lite.cache import RadixCache
from vibe_sgl_lite.scheduler import LPMScheduler

# Initialize with prefix caching and custom scheduler
runner = ModelRunner(
    model_path="Qwen/Qwen3-0.6B",
    enable_prefix_cache=True,
    cache=RadixCache(),
    scheduler=LPMScheduler()
)

# Benefit from automatic prefix reuse
prompts = [
    "Translate to French: Hello, how are you?",
    "Translate to French: Good morning!",
    "Translate to French: Thank you very much."
]

outputs = runner.generate_batch(prompts, max_tokens=20)
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m distributed   # Distributed tests (TP/EP)

# Run with coverage report
pytest --cov=vibe_sgl_lite --cov-report=html

# Run tests in parallel
pytest -n auto
```

### Code Quality

```bash
# Format code
black vibe_sgl_lite/ tests/

# Sort imports
isort vibe_sgl_lite/ tests/

# Type checking
mypy vibe_sgl_lite/
```

## Project Structure

```
vibe-sgl-lite/
├── vibe_sgl_lite/          # Main package
│   ├── core/               # Core inference engine
│   ├── models/             # Custom model implementations
│   ├── memory/             # Memory management
│   ├── cache/              # Caching mechanisms
│   ├── batch/              # Batching logic
│   ├── sampling/           # Sampling strategies
│   ├── scheduler/          # Scheduling policies
│   ├── distributed/        # Parallelism (TP, EP)
│   └── utils/              # Utilities
├── tests/                  # Test suite (mirrors source structure)
├── examples/               # Usage examples
├── docs/                   # Documentation
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Testing Philosophy

This project follows strict Test-Driven Development (TDD):

1. **Tests First**: All features have tests written before implementation
2. **CPU Testing**: All tests run on CPU for accessibility (using Gloo backend for distributed tests)
3. **Comprehensive Coverage**: Target 80%+ code coverage
4. **Test Model**: Uses Qwen/Qwen3-0.6B for all testing

## Distributed Testing

Tensor Parallelism and Expert Parallelism are tested using `torch.distributed` with CPU process groups:

```bash
# Run distributed tests
pytest -m distributed

# Test specific parallelism features
pytest tests/distributed/test_tensor_parallelism.py
pytest tests/distributed/test_expert_parallelism.py
```

## Roadmap

### Current Features (v0.1.0)
- ✅ Core inference engine
- ✅ Paged attention
- ✅ Prefix caching (RadixAttention)
- ✅ Continuous batching
- ✅ Chunked prefill
- ✅ Tensor Parallelism (TP)
- ✅ Expert Parallelism (EP)
- ✅ Custom Qwen3 implementation

### Future Enhancements
- 🔄 Speculative decoding (EAGLE)
- 🔄 Constrained generation (JSON schema, regex)
- 🔄 Multi-LoRA batching
- 🔄 Pipeline Parallelism (PP)
- 🔄 GPU optimizations (FlashAttention, CUDA graphs)
- 🔄 Additional model architectures

## Contributing

Contributions are welcome! Please follow these guidelines:

1. **TDD Approach**: Write tests before implementation
2. **Code Quality**: Run `black`, `isort`, and `mypy` before submitting
3. **Documentation**: Update docs for new features
4. **Test Coverage**: Maintain 80%+ coverage

## License

MIT License - see LICENSE file for details

## Acknowledgments

Inspired by advanced techniques from:
- vibe-sgl
- vLLM
- TensorRT-LLM
- SGLang

## Citation

If you use vibe-sgl-lite in your research, please cite:

```bibtex
@software{vibe_sgl_lite,
  title = {vibe-sgl-lite: A Lightweight LLM Inference Engine},
  author = {vibe-sgl-lite contributors},
  year = {2026},
  url = {https://github.com/yourusername/vibe-sgl-lite}
}
```

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/yourusername/vibe-sgl-lite/issues)
- **Discussions**: Ask questions on [GitHub Discussions](https://github.com/yourusername/vibe-sgl-lite/discussions)
- **Documentation**: Full documentation at [docs/](docs/)

## Contact

For questions and feedback, please open an issue on GitHub.
