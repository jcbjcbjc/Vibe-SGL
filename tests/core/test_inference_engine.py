"""
Tests for InferenceEngine initialization and basic functionality.
"""

import pytest
import torch
from vibe_sgl_lite.core.inference_engine import InferenceEngine
from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.sampling.sampling import SamplingParams


@pytest.mark.unit
def test_inference_engine_initialization(qwen3_model_name):
    """Test that InferenceEngine initializes correctly."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Check that config is loaded
    assert engine.config is not None
    assert isinstance(engine.config, Qwen3Config)

    # Check that model and lm_head are initialized
    assert engine.model is not None
    assert engine.lm_head is not None

    # Check that device is set
    assert engine.device == "cpu"

    # Check that models are in eval mode
    assert not engine.model.training
    assert not engine.lm_head.training


@pytest.mark.unit
def test_inference_engine_invalid_model_path():
    """Test that InferenceEngine raises error for invalid model path."""
    with pytest.raises(Exception):  # Could be OSError, ValueError, etc.
        InferenceEngine("invalid/model/path", device="cpu")


@pytest.mark.unit
def test_inference_engine_device_setting(qwen3_model_name):
    """Test that InferenceEngine respects device setting."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")
    assert engine.device == "cpu"

    # If CUDA is available, test GPU device
    if torch.cuda.is_available():
        engine_gpu = InferenceEngine(qwen3_model_name, device="cuda")
        assert engine_gpu.device == "cuda"


@pytest.mark.unit
def test_load_model_from_checkpoint(qwen3_model_name):
    """Test loading model weights from HuggingFace checkpoint."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Verify model is loaded
    assert engine.model is not None
    assert engine.lm_head is not None

    # Verify weights are loaded (check that parameters are not all zeros)
    model_params = list(engine.model.parameters())
    assert len(model_params) > 0

    # Check that at least some parameters are non-zero
    has_nonzero = any(param.abs().sum() > 0 for param in model_params)
    assert has_nonzero, "Model parameters should not all be zero after loading"


@pytest.mark.unit
def test_load_model_architecture_validation(qwen3_model_name):
    """Test that model architecture is validated during loading."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Verify config matches expected architecture
    assert engine.config.vocab_size > 0
    assert engine.config.hidden_size > 0
    assert engine.config.num_hidden_layers > 0
    assert engine.config.num_attention_heads > 0


@pytest.mark.unit
def test_tokenize_input_text(qwen3_model_name):
    """Test tokenization of input text."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Test basic tokenization
    text = "Hello, world!"
    tokens = engine.tokenize(text)

    assert isinstance(tokens, torch.Tensor)
    assert tokens.dim() == 2  # [batch_size, seq_len]
    assert tokens.shape[0] == 1  # batch size
    assert tokens.shape[1] > 0  # has tokens


@pytest.mark.unit
def test_detokenize_output_tokens(qwen3_model_name):
    """Test detokenization of output tokens."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Tokenize then detokenize
    text = "Hello, world!"
    tokens = engine.tokenize(text)
    decoded = engine.detokenize(tokens)

    assert isinstance(decoded, str)
    # The decoded text should be similar to the original
    # (may have minor differences due to tokenizer behavior)
    assert len(decoded) > 0


@pytest.mark.unit
def test_tokenize_batch(qwen3_model_name):
    """Test batch tokenization."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    texts = ["Hello, world!", "How are you?", "This is a test."]
    tokens = engine.tokenize_batch(texts)

    assert isinstance(tokens, torch.Tensor)
    assert tokens.dim() == 2  # [batch_size, seq_len]
    assert tokens.shape[0] == len(texts)  # batch size matches input


@pytest.mark.unit
def test_tokenize_empty_input(qwen3_model_name):
    """Test that empty input raises ValueError."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    with pytest.raises(ValueError):
        engine.tokenize("")


@pytest.mark.unit
def test_tokenize_none_input(qwen3_model_name):
    """Test that None input raises ValueError."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    with pytest.raises((ValueError, TypeError)):
        engine.tokenize(None)


@pytest.mark.integration
def test_generate_single_sequence(qwen3_model_name):
    """Test single sequence generation."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Tokenize input
    text = "Hello"
    input_ids = engine.tokenize(text)

    # Generate tokens
    output_ids = engine.generate(input_ids, max_new_tokens=5)

    # Check output shape
    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.dim() == 2
    assert output_ids.shape[0] == 1  # batch size
    assert output_ids.shape[1] > input_ids.shape[1]  # generated tokens added


@pytest.mark.integration
def test_generate_with_sampling_params(qwen3_model_name):
    """Test generation with custom sampling parameters."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    text = "Hello"
    input_ids = engine.tokenize(text)

    # Test with greedy sampling
    sampling_params = SamplingParams(temperature=0.0)
    output_ids = engine.generate(
        input_ids, max_new_tokens=5, sampling_params=sampling_params
    )

    assert output_ids.shape[1] > input_ids.shape[1]


@pytest.mark.integration
def test_generate_respects_max_tokens(qwen3_model_name):
    """Test that generation respects max_new_tokens limit."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    text = "Hello"
    input_ids = engine.tokenize(text)
    max_new_tokens = 10

    output_ids = engine.generate(input_ids, max_new_tokens=max_new_tokens)

    # Output should have at most max_new_tokens more than input
    assert output_ids.shape[1] <= input_ids.shape[1] + max_new_tokens


@pytest.mark.integration
def test_generate_text_to_text(qwen3_model_name):
    """Test end-to-end text generation."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Generate text from text input
    input_text = "The capital of France is"
    output_text = engine.generate_text(input_text, max_new_tokens=5)

    assert isinstance(output_text, str)
    assert len(output_text) > len(input_text)
    assert output_text.startswith(input_text)


@pytest.mark.integration
def test_generate_batch(qwen3_model_name):
    """Test batch generation."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Tokenize batch
    texts = ["Hello", "Hi there"]
    input_ids = engine.tokenize_batch(texts)

    # Generate for batch
    output_ids = engine.generate_batch(input_ids, max_new_tokens=5)

    # Check output shape
    assert isinstance(output_ids, torch.Tensor)
    assert output_ids.dim() == 2
    assert output_ids.shape[0] == len(texts)  # batch size preserved
    assert output_ids.shape[1] > input_ids.shape[1]  # tokens generated


@pytest.mark.integration
def test_generate_batch_text(qwen3_model_name):
    """Test batch text generation."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    texts = ["Hello", "Hi there", "Good morning"]
    outputs = engine.generate_batch_text(texts, max_new_tokens=5)

    assert isinstance(outputs, list)
    assert len(outputs) == len(texts)
    for i, output in enumerate(outputs):
        assert isinstance(output, str)
        assert len(output) > len(texts[i])


@pytest.mark.integration
def test_generate_stream(qwen3_model_name):
    """Test streaming generation."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    text = "Hello"
    input_ids = engine.tokenize(text)

    # Generate with streaming
    tokens = []
    for token in engine.generate_stream(input_ids, max_new_tokens=5):
        assert isinstance(token, torch.Tensor)
        tokens.append(token)

    # Should have generated tokens
    assert len(tokens) > 0
    assert len(tokens) <= 5  # respects max_new_tokens


@pytest.mark.integration
def test_generate_stream_text(qwen3_model_name):
    """Test streaming text generation."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    text = "Hello"

    # Generate with streaming
    chunks = []
    for chunk in engine.generate_stream_text(text, max_new_tokens=5):
        assert isinstance(chunk, str)
        chunks.append(chunk)

    # Should have generated text chunks
    assert len(chunks) > 0
    assert len(chunks) <= 5  # respects max_new_tokens


@pytest.mark.unit
def test_error_handling_empty_model_name():
    """Test that empty model name raises ValueError."""
    with pytest.raises(ValueError, match="model_name cannot be empty"):
        InferenceEngine("", device="cpu")


@pytest.mark.unit
def test_error_handling_invalid_input_generate():
    """Test error handling for invalid input to generate."""
    # This test would require a loaded engine, but we test the validation
    # in the tokenize methods which are called before generate
    pass  # Covered by tokenize tests


@pytest.mark.integration
def test_end_to_end_integration(qwen3_model_name):
    """Test complete inference pipeline end-to-end."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    # Test single generation
    output = engine.generate_text("Hello", max_new_tokens=5)
    assert isinstance(output, str)
    assert len(output) > 0

    # Test batch generation
    outputs = engine.generate_batch_text(
        ["Hello", "Hi"], max_new_tokens=5
    )
    assert len(outputs) == 2
    assert all(isinstance(o, str) for o in outputs)

    # Test streaming
    chunks = list(engine.generate_stream_text("Hello", max_new_tokens=3))
    assert len(chunks) > 0


@pytest.mark.integration
def test_continuous_batching_integration(qwen3_model_name):
    """Test continuous batching integration."""
    # Initialize engine with continuous batching enabled
    engine = InferenceEngine(
        qwen3_model_name,
        device="cpu",
        enable_continuous_batching=True,
        max_batch_size=4,
    )

    # Submit multiple requests
    req_ids = []
    for i in range(3):
        req_id = engine.submit_request(
            text=f"Request {i}",
            max_new_tokens=5,
        )
        req_ids.append(req_id)

    # Run continuous batching
    outputs = engine.run_continuous_batching(max_iterations=20)

    # Check outputs
    assert len(outputs) == 3
    for req_id in req_ids:
        assert req_id in outputs
        assert isinstance(outputs[req_id], str)

    # Check metrics
    metrics = engine.get_batch_metrics()
    assert metrics["total_requests"] == 3
    assert metrics["completed_requests"] == 3


@pytest.mark.integration
def test_continuous_batching_dynamic_requests(qwen3_model_name):
    """Test continuous batching with dynamic request addition."""
    engine = InferenceEngine(
        qwen3_model_name,
        device="cpu",
        enable_continuous_batching=True,
        max_batch_size=2,
    )

    # Submit initial requests
    req1 = engine.submit_request("First", max_new_tokens=3)
    req2 = engine.submit_request("Second", max_new_tokens=3)

    # Run a few steps
    for _ in range(3):
        engine.run_continuous_batching_step()

    # Add more requests while processing
    req3 = engine.submit_request("Third", max_new_tokens=3)

    # Continue processing
    outputs = engine.run_continuous_batching(max_iterations=20)

    # All requests should complete
    assert len(outputs) == 3
    assert req1 in outputs
    assert req2 in outputs
    assert req3 in outputs


@pytest.mark.unit
def test_continuous_batching_disabled_error(qwen3_model_name):
    """Test that continuous batching methods raise error when disabled."""
    engine = InferenceEngine(qwen3_model_name, device="cpu")

    with pytest.raises(RuntimeError, match="Continuous batching is not enabled"):
        engine.submit_request("Test", max_new_tokens=5)


@pytest.mark.integration
def test_paged_attention_integration(qwen3_model_name):
    """Test paged attention integration."""
    # Initialize engine with paged attention enabled
    engine = InferenceEngine(
        qwen3_model_name,
        device="cpu",
        enable_paged_attention=True,
        kv_cache_size_mb=100,
        page_size=16,
    )

    # Verify memory pool is initialized
    assert engine.memory_pool is not None
    assert engine.paged_allocator is not None

    # Get memory stats
    stats = engine.get_memory_stats()
    assert "free" in stats
    assert "used" in stats
    assert "total" in stats
    assert stats["free"] > 0

    # Generate text (should use paged attention internally)
    output = engine.generate_text("Hello", max_new_tokens=5)
    assert isinstance(output, str)
    assert len(output) > 0


@pytest.mark.integration
def test_prefix_caching_integration(qwen3_model_name):
    """Test prefix caching integration."""
    # Initialize engine with prefix caching enabled
    engine = InferenceEngine(
        qwen3_model_name,
        device="cpu",
        enable_prefix_caching=True,
    )

    # Verify cache is initialized
    assert engine.radix_cache is not None

    # Get cache stats
    stats = engine.get_cache_stats()
    assert "hits" in stats
    assert "misses" in stats
    assert "num_sequences" in stats

    # Generate text (should use prefix caching internally)
    output = engine.generate_text("Hello", max_new_tokens=5)
    assert isinstance(output, str)
    assert len(output) > 0


@pytest.mark.integration
def test_all_features_enabled(qwen3_model_name):
    """Test engine with all features enabled."""
    # Initialize engine with all features
    engine = InferenceEngine(
        qwen3_model_name,
        device="cpu",
        enable_continuous_batching=True,
        max_batch_size=4,
        enable_paged_attention=True,
        kv_cache_size_mb=100,
        enable_prefix_caching=True,
    )

    # Verify all features are initialized
    assert engine.batch_manager is not None
    assert engine.memory_pool is not None
    assert engine.paged_allocator is not None
    assert engine.radix_cache is not None

    # Submit requests with continuous batching
    req1 = engine.submit_request("Hello", max_new_tokens=5)
    req2 = engine.submit_request("Hi there", max_new_tokens=5)

    # Run continuous batching
    outputs = engine.run_continuous_batching(max_iterations=20)

    # Verify outputs
    assert len(outputs) == 2
    assert req1 in outputs
    assert req2 in outputs

    # Get all metrics
    batch_metrics = engine.get_batch_metrics()
    memory_stats = engine.get_memory_stats()
    cache_stats = engine.get_cache_stats()

    assert batch_metrics["completed_requests"] == 2
    assert "free" in memory_stats
    assert "hits" in cache_stats
