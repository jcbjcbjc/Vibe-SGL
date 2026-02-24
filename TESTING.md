# Testing Guide

This document provides comprehensive instructions for running tests in the vibe-sgl-lite project.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Organization](#test-organization)
- [Running Tests](#running-tests)
- [Test Markers](#test-markers)
- [Test Coverage](#test-coverage)
- [Distributed Testing](#distributed-testing)
- [Test Fixtures](#test-fixtures)
- [Writing Tests](#writing-tests)
- [Debugging Tests](#debugging-tests)
- [Continuous Integration](#continuous-integration)
- [Performance Benchmarks](#performance-benchmarks)

## Quick Start

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=vibe_sgl_lite --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m distributed   # Distributed tests (TP/EP)
```

## Test Organization

Tests are organized to mirror the source code structure:

```
tests/
├── core/               # Core inference engine tests
├── models/             # Model implementation tests
├── memory/             # Memory management tests
├── cache/              # Caching mechanism tests
├── batch/              # Batching logic tests
├── sampling/           # Sampling strategy tests
├── scheduler/          # Scheduling policy tests
├── distributed/        # Parallelism tests (TP, EP)
├── integration/        # End-to-end integration tests
└── conftest.py         # Shared fixtures and configuration
```

Each test file follows the naming convention `test_*.py` and contains tests for the corresponding source module.

## Running Tests

### Basic Test Execution

```bash
# Run all tests with verbose output
pytest

# Run tests in a specific directory
pytest tests/core/

# Run a specific test file
pytest tests/core/test_model_runner.py

# Run a specific test function
pytest tests/core/test_model_runner.py::test_basic_generation

# Run tests matching a pattern
pytest -k "test_generation"
```

### Parallel Test Execution

```bash
# Run tests in parallel using all available CPU cores
pytest -n auto

# Run tests using a specific number of workers
pytest -n 4
```

### Verbose Output

```bash
# Show detailed test output
pytest -v

# Show even more detailed output (including print statements)
pytest -vv -s

# Show short traceback format (default)
pytest --tb=short

# Show full traceback
pytest --tb=long
```

## Test Markers

Tests are categorized using pytest markers for selective execution:

### Available Markers

- **`unit`**: Unit tests for individual components (fast, isolated)
- **`integration`**: Integration tests for end-to-end workflows (slower, uses real models)
- **`slow`**: Tests that take longer to run (>5 seconds)
- **`distributed`**: Tests requiring multiple processes (TP/EP with torch.distributed)

### Running Tests by Marker

```bash
# Run only unit tests (fast)
pytest -m unit

# Run only integration tests
pytest -m integration

# Run distributed tests (TP/EP)
pytest -m distributed

# Exclude slow tests
pytest -m "not slow"

# Run unit tests but exclude slow ones
pytest -m "unit and not slow"

# Run integration or distributed tests
pytest -m "integration or distributed"
```

## Test Coverage

### Generating Coverage Reports

```bash
# Run tests with coverage (terminal report)
pytest --cov=vibe_sgl_lite --cov-report=term-missing

# Generate HTML coverage report
pytest --cov=vibe_sgl_lite --cov-report=html

# View HTML report (opens in browser)
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Generate XML coverage report (for CI)
pytest --cov=vibe_sgl_lite --cov-report=xml
```

### Coverage Configuration

The project enforces a minimum coverage threshold of **80%**. Tests will fail if coverage drops below this threshold.

Coverage settings are configured in `pyproject.toml`:

```toml
[tool.coverage.run]
source = ["vibe_sgl_lite"]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/site-packages/*",
]

[tool.coverage.report]
precision = 2
show_missing = true
skip_covered = false
```

### Checking Coverage for Specific Modules

```bash
# Coverage for a specific module
pytest --cov=vibe_sgl_lite.core --cov-report=term-missing tests/core/

# Coverage for multiple modules
pytest --cov=vibe_sgl_lite.core --cov=vibe_sgl_lite.memory --cov-report=html
```

## Distributed Testing

Tensor Parallelism (TP) and Expert Parallelism (EP) tests use `torch.distributed` with the Gloo backend for CPU-based testing.

### Running Distributed Tests

```bash
# Run all distributed tests
pytest -m distributed

# Run specific distributed test files
pytest tests/distributed/test_tensor_parallelism.py
pytest tests/distributed/test_expert_parallelism.py

# Run with verbose output to see process coordination
pytest -m distributed -vv -s
```

### How Distributed Tests Work

1. **Process Spawning**: Tests use `torch.multiprocessing` to spawn multiple processes
2. **Backend**: Gloo backend is used for CPU-based distributed operations
3. **Synchronization**: Processes synchronize using `torch.distributed` primitives
4. **Result Collection**: Main process collects and validates results from all workers

### Example Distributed Test Structure

```python
import torch.multiprocessing as mp
import torch.distributed as dist

def test_tensor_parallelism():
    world_size = 2
    mp.spawn(
        _run_tp_test,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

def _run_tp_test(rank, world_size):
    # Initialize process group
    dist.init_process_group(
        backend="gloo",
        init_method="tcp://localhost:29500",
        rank=rank,
        world_size=world_size
    )

    # Run distributed test logic
    # ...

    dist.destroy_process_group()
```

## Test Fixtures

Common test fixtures are defined in `tests/conftest.py` and are automatically available to all tests.

### Available Fixtures

#### Model Fixtures

```python
@pytest.fixture(scope="session")
def test_model_path():
    """Returns the path to the test model (Qwen3-0.6B)."""
    return "Qwen/Qwen3-0.6B"

@pytest.fixture(scope="session")
def model(test_model_path):
    """Loads and caches the Qwen3-0.6B model for reuse across tests."""
    # Model is loaded once and reused
    pass

@pytest.fixture(scope="session")
def tokenizer(test_model_path):
    """Loads and caches the Qwen3 tokenizer."""
    pass
```

#### Sample Data Fixtures

```python
@pytest.fixture
def sample_prompts():
    """Provides diverse sample prompts for testing."""
    return [
        "Once upon a time",  # Short prompt
        "Explain quantum computing in simple terms: " * 10,  # Long prompt
        "Hello, how are you?",  # Conversational
        "",  # Edge case: empty
    ]

@pytest.fixture
def sample_tokens():
    """Provides sample token IDs for testing."""
    pass
```

#### Cleanup Fixtures

```python
@pytest.fixture(autouse=True)
def cleanup_memory():
    """Automatically cleans up memory after each test."""
    yield
    # Cleanup logic
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

@pytest.fixture(autouse=True)
def cleanup_processes():
    """Ensures distributed processes are properly terminated."""
    yield
    # Cleanup logic
```

### Using Fixtures in Tests

```python
def test_model_loading(model, tokenizer):
    """Test uses model and tokenizer fixtures."""
    assert model is not None
    assert tokenizer is not None

def test_generation(model, tokenizer, sample_prompts):
    """Test uses multiple fixtures."""
    for prompt in sample_prompts:
        output = model.generate(prompt)
        assert output is not None
```

## Writing Tests

### Test Structure

Follow this structure for writing tests:

```python
import pytest
from vibe_sgl_lite.core import ModelRunner

class TestModelRunner:
    """Test suite for ModelRunner class."""

    @pytest.mark.unit
    def test_initialization(self, test_model_path):
        """Test ModelRunner initialization."""
        runner = ModelRunner(model_path=test_model_path)
        assert runner is not None

    @pytest.mark.integration
    def test_basic_generation(self, model, tokenizer):
        """Test basic text generation."""
        runner = ModelRunner(model=model, tokenizer=tokenizer)
        output = runner.generate("Hello", max_tokens=10)
        assert len(output) > 0

    @pytest.mark.slow
    @pytest.mark.integration
    def test_long_generation(self, model, tokenizer):
        """Test generation with long sequences."""
        runner = ModelRunner(model=model, tokenizer=tokenizer)
        output = runner.generate("Once upon a time", max_tokens=500)
        assert len(output) > 0
```

### Test Naming Conventions

- Test files: `test_<module_name>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<functionality>`
- Use descriptive names that explain what is being tested

### Assertions and Validation

```python
# Basic assertions
assert result is not None
assert len(output) > 0
assert output.startswith("Hello")

# Numerical comparisons with tolerance
import torch
assert torch.allclose(output, expected, atol=1e-5, rtol=1e-4)

# Exception testing
with pytest.raises(ValueError, match="Invalid input"):
    runner.generate("")

# Parametrized tests
@pytest.mark.parametrize("temperature", [0.0, 0.5, 1.0, 1.5])
def test_temperature_sampling(model, temperature):
    output = model.generate("Hello", temperature=temperature)
    assert output is not None
```

### Mocking Dependencies

```python
from unittest.mock import Mock, patch

def test_with_mock():
    """Test using mocked dependencies."""
    mock_model = Mock()
    mock_model.generate.return_value = "mocked output"

    runner = ModelRunner(model=mock_model)
    output = runner.generate("test")

    assert output == "mocked output"
    mock_model.generate.assert_called_once()
```

## Debugging Tests

### Running Tests in Debug Mode

```bash
# Run with print statements visible
pytest -s

# Run with Python debugger on failure
pytest --pdb

# Run with detailed traceback
pytest --tb=long

# Stop on first failure
pytest -x

# Show local variables in traceback
pytest -l
```

### Debugging Specific Tests

```python
# Add breakpoint in test code
def test_something():
    result = some_function()
    breakpoint()  # Execution will pause here
    assert result == expected
```

### Common Issues and Solutions

#### Issue: Tests fail with "CUDA out of memory"

**Solution**: Tests should run on CPU. Ensure `CUDA_VISIBLE_DEVICES=""` is set:

```bash
export CUDA_VISIBLE_DEVICES=""
pytest
```

#### Issue: Distributed tests hang or timeout

**Solution**: Check process synchronization and ensure proper cleanup:

```python
# Always destroy process group
try:
    # Test logic
    pass
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
```

#### Issue: Model download fails

**Solution**: Manually download the model first:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

#### Issue: Tests are too slow

**Solution**: Use markers to skip slow tests during development:

```bash
pytest -m "not slow"
```

## Continuous Integration

### GitHub Actions Configuration

The project includes CI configuration for automated testing on multiple Python versions.

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11"]

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest --cov=vibe_sgl_lite --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Running CI Tests Locally

```bash
# Simulate CI environment
export CUDA_VISIBLE_DEVICES=""
pytest --cov=vibe_sgl_lite --cov-report=xml --cov-fail-under=80
```

## Performance Benchmarks

### Running Benchmarks

```bash
# Run all benchmark tests
pytest tests/benchmarks/ -v

# Run specific benchmark
pytest tests/benchmarks/test_latency.py

# Run benchmarks with detailed output
pytest tests/benchmarks/ -vv -s
```

### Benchmark Categories

#### Latency Benchmarks

Measure time per token for various configurations:

```bash
pytest tests/benchmarks/test_latency.py -v
```

#### Throughput Benchmarks

Measure tokens per second for continuous batching:

```bash
pytest tests/benchmarks/test_throughput.py -v
```

#### Memory Benchmarks

Measure peak memory usage:

```bash
pytest tests/benchmarks/test_memory.py -v
```

### Interpreting Benchmark Results

Benchmarks output metrics in the following format:

```
Latency (batch_size=1): 45.2ms per token
Latency (batch_size=4): 38.7ms per token
Latency (batch_size=8): 35.1ms per token

Throughput (continuous batching): 127.3 tokens/sec
Throughput (static batching): 98.6 tokens/sec

Peak Memory (paged attention): 1.2GB
Peak Memory (no paging): 2.4GB
```

## Test Model

All tests use **Qwen/Qwen3-0.6B** as the standard test model:

- **Size**: 600M parameters
- **Context Length**: 32K tokens
- **Architecture**: Qwen3 (custom implementation)
- **Download**: Automatically downloaded from HuggingFace on first run
- **Cache**: Stored in `~/.cache/huggingface/hub/`

### Model Caching

The model is downloaded once and cached for subsequent test runs:

```python
# First run: Downloads model (~1.2GB)
pytest

# Subsequent runs: Uses cached model (fast)
pytest
```

### Clearing Model Cache

```bash
# Remove cached models
rm -rf ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B
```

## Environment Variables

### Test Configuration

```bash
# Force CPU testing (recommended)
export CUDA_VISIBLE_DEVICES=""

# Set HuggingFace cache directory
export HF_HOME=/path/to/cache

# Enable debug logging
export VIBE_SGL_DEBUG=1

# Set number of threads for CPU testing
export OMP_NUM_THREADS=4
```

### Running Tests with Environment Variables

```bash
# Single command
CUDA_VISIBLE_DEVICES="" pytest

# Multiple variables
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 pytest -m unit
```

## Best Practices

### For Test Writers

1. **Write tests first** (TDD): Write failing tests before implementation
2. **Use appropriate markers**: Tag tests with `unit`, `integration`, `slow`, or `distributed`
3. **Keep tests isolated**: Each test should be independent and not rely on others
4. **Use fixtures**: Leverage shared fixtures for common setup
5. **Test edge cases**: Include tests for empty inputs, very long sequences, etc.
6. **Validate correctness**: Compare outputs with reference implementations when possible
7. **Document tests**: Add docstrings explaining what each test validates

### For Test Runners

1. **Run unit tests frequently**: Fast feedback during development
2. **Run full suite before commits**: Ensure nothing breaks
3. **Check coverage**: Maintain 80%+ coverage
4. **Use parallel execution**: Speed up test runs with `-n auto`
5. **Skip slow tests during development**: Use `-m "not slow"` for faster iteration

## Troubleshooting

### Tests Pass Locally but Fail in CI

- Check Python version compatibility
- Verify all dependencies are installed
- Ensure environment variables are set correctly
- Check for platform-specific issues (macOS vs Linux)

### Flaky Tests

- Add proper synchronization for distributed tests
- Use fixed random seeds for deterministic behavior
- Increase timeouts for slow operations
- Check for race conditions in parallel tests

### Memory Issues

- Run tests sequentially instead of in parallel
- Reduce batch sizes in tests
- Clear cache between test runs
- Use `pytest-xdist` with `--dist=loadfile` to avoid memory spikes

## Additional Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [PyTorch Distributed Documentation](https://pytorch.org/docs/stable/distributed.html)
- [Project README](README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Getting Help

If you encounter issues with testing:

1. Check this guide for common solutions
2. Review test output and error messages carefully
3. Run tests with `-vv -s` for detailed output
4. Open an issue on GitHub with test failure details
5. Ask questions in GitHub Discussions

---

**Note**: This project follows strict TDD practices. All features should have tests written before implementation, and all tests should pass before code is merged.
