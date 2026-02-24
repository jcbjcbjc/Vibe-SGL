"""
Sample test data for vibe-sgl-lite tests.

This module provides diverse test data for various testing scenarios:
- Sample prompts (short, long, multi-turn)
- Expected outputs for validation
- Edge cases (empty, very long, special characters)
- Benchmark data for performance testing

All test data is designed to work with Qwen3-0.6B model on CPU.
"""

from typing import Dict, List, Tuple


# ============================================================================
# Sample Prompts - Diverse inputs for testing
# ============================================================================

SAMPLE_PROMPTS_SHORT: List[str] = [
    "Hello, world!",
    "What is 2+2?",
    "Tell me a joke.",
    "Translate 'hello' to French.",
    "Who is the president?",
]

SAMPLE_PROMPTS_MEDIUM: List[str] = [
    "Explain the concept of machine learning in simple terms.",
    "Write a short poem about the ocean.",
    "What are the benefits of regular exercise?",
    "Describe the process of photosynthesis.",
    "How does a computer processor work?",
]

SAMPLE_PROMPTS_LONG: List[str] = [
    (
        "Write a detailed explanation of how neural networks work, "
        "including the concepts of forward propagation, backpropagation, "
        "activation functions, and gradient descent. Please provide examples "
        "and explain why deep learning has become so successful in recent years."
    ),
    (
        "Describe the history of artificial intelligence from its inception "
        "in the 1950s to modern day, covering key milestones such as the "
        "Dartmouth Conference, expert systems, the AI winter, the resurgence "
        "with deep learning, and recent breakthroughs in large language models."
    ),
    (
        "Explain quantum computing and how it differs from classical computing. "
        "Discuss qubits, superposition, entanglement, and quantum gates. "
        "What are the potential applications of quantum computers, and what "
        "are the current challenges in building practical quantum computers?"
    ),
]

# Multi-turn conversation prompts
SAMPLE_PROMPTS_MULTI_TURN: List[List[str]] = [
    [
        "What is Python?",
        "What are its main features?",
        "Can you give me an example?",
    ],
    [
        "Tell me about machine learning.",
        "What's the difference between supervised and unsupervised learning?",
        "Give me an example of each.",
    ],
    [
        "What is the capital of France?",
        "What is its population?",
        "What are some famous landmarks there?",
    ],
]

# ============================================================================
# Edge Cases - Testing robustness
# ============================================================================

EDGE_CASE_PROMPTS: List[Tuple[str, str]] = [
    ("", "empty_string"),
    ("   ", "whitespace_only"),
    ("a", "single_character"),
    ("Hello" * 1000, "repeated_word"),
    ("🚀🌟💻🎉", "emoji_only"),
    ("Hello\nWorld\n\nTest", "multiline_with_newlines"),
    ("Special chars: @#$%^&*()", "special_characters"),
    ("Mixed 中文 English 日本語", "mixed_languages"),
    ("A" * 2048, "very_long_single_token"),
    ("<html><body>Test</body></html>", "html_tags"),
    ("SELECT * FROM users;", "sql_injection_attempt"),
    ("'; DROP TABLE users; --", "sql_injection_malicious"),
    ("../../../etc/passwd", "path_traversal_attempt"),
    ("\x00\x01\x02", "null_bytes"),
    ("Test\r\nCRLF\r\nLine", "crlf_injection"),
]

# ============================================================================
# Expected Outputs - For correctness validation
# ============================================================================

# Note: These are approximate expected outputs for Qwen3-0.6B
# Actual outputs may vary slightly due to model updates or randomness
EXPECTED_OUTPUTS: Dict[str, str] = {
    "Hello, world!": "Hello! How can I assist you today?",
    "What is 2+2?": "2+2 equals 4.",
    "Tell me a joke.": "Why did the chicken cross the road? To get to the other side!",
}

# ============================================================================
# Benchmark Data - For performance testing
# ============================================================================

# Batch sizes for throughput testing
BENCHMARK_BATCH_SIZES: List[int] = [1, 2, 4, 8, 16, 32]

# Sequence lengths for latency testing
BENCHMARK_SEQ_LENGTHS: List[int] = [16, 32, 64, 128, 256, 512, 1024]

# Generation lengths for end-to-end testing
BENCHMARK_GEN_LENGTHS: List[int] = [10, 20, 50, 100, 200]

# Prompts for benchmarking (various lengths)
BENCHMARK_PROMPTS: Dict[str, str] = {
    "short": "Hello",
    "medium": "Explain machine learning in simple terms.",
    "long": (
        "Write a comprehensive guide to deep learning, covering neural networks, "
        "backpropagation, optimization algorithms, regularization techniques, "
        "and modern architectures like transformers and attention mechanisms."
    ),
    "very_long": " ".join(["This is a test sentence."] * 100),
}

# ============================================================================
# Test Scenarios - Structured test cases
# ============================================================================

TEST_SCENARIOS: List[Dict[str, any]] = [
    {
        "name": "basic_generation",
        "prompt": "Hello, world!",
        "max_tokens": 20,
        "temperature": 1.0,
        "top_p": 1.0,
        "expected_min_tokens": 5,
    },
    {
        "name": "short_generation",
        "prompt": "What is AI?",
        "max_tokens": 10,
        "temperature": 0.7,
        "top_p": 0.9,
        "expected_min_tokens": 5,
    },
    {
        "name": "long_generation",
        "prompt": "Explain quantum computing.",
        "max_tokens": 100,
        "temperature": 0.8,
        "top_p": 0.95,
        "expected_min_tokens": 50,
    },
    {
        "name": "deterministic_generation",
        "prompt": "Count from 1 to 5.",
        "max_tokens": 20,
        "temperature": 0.0,  # Greedy decoding
        "top_p": 1.0,
        "expected_min_tokens": 5,
    },
    {
        "name": "creative_generation",
        "prompt": "Write a creative story.",
        "max_tokens": 50,
        "temperature": 1.2,  # High temperature for creativity
        "top_p": 0.95,
        "expected_min_tokens": 30,
    },
]

# ============================================================================
# Prefix Caching Test Data
# ============================================================================

# Prompts with common prefixes for testing prefix caching
PREFIX_CACHE_PROMPTS: List[str] = [
    "Translate the following to French: Hello",
    "Translate the following to French: Goodbye",
    "Translate the following to French: Thank you",
    "Translate the following to Spanish: Hello",
    "Translate the following to Spanish: Goodbye",
]

# System prompts for multi-turn conversations
SYSTEM_PROMPTS: List[str] = [
    "You are a helpful assistant.",
    "You are a coding expert specializing in Python.",
    "You are a creative writer who loves storytelling.",
    "You are a math tutor helping students learn algebra.",
]

# ============================================================================
# Batching Test Data
# ============================================================================

# Requests with varying lengths for continuous batching
BATCHING_REQUESTS: List[Dict[str, any]] = [
    {"prompt": "Short", "max_tokens": 10, "priority": 1},
    {"prompt": "Medium length prompt here", "max_tokens": 20, "priority": 2},
    {"prompt": "This is a longer prompt for testing batching", "max_tokens": 30, "priority": 1},
    {"prompt": "Very long prompt " * 10, "max_tokens": 50, "priority": 3},
    {"prompt": "Another short one", "max_tokens": 10, "priority": 2},
]

# ============================================================================
# Distributed Testing Data
# ============================================================================

# Test configurations for tensor parallelism
TP_TEST_CONFIGS: List[Dict[str, any]] = [
    {"world_size": 2, "prompt": "Hello", "max_tokens": 10},
    {"world_size": 4, "prompt": "Explain AI", "max_tokens": 20},
]

# Test configurations for expert parallelism (MoE models)
EP_TEST_CONFIGS: List[Dict[str, any]] = [
    {"world_size": 2, "num_experts": 4, "prompt": "Test", "max_tokens": 10},
    {"world_size": 4, "num_experts": 8, "prompt": "Explain", "max_tokens": 20},
]

# ============================================================================
# Correctness Validation Data
# ============================================================================

# Deterministic prompts for comparing with HuggingFace reference
CORRECTNESS_PROMPTS: List[Dict[str, any]] = [
    {
        "prompt": "The capital of France is",
        "max_tokens": 5,
        "temperature": 0.0,
        "seed": 42,
    },
    {
        "prompt": "2 + 2 =",
        "max_tokens": 3,
        "temperature": 0.0,
        "seed": 42,
    },
    {
        "prompt": "Once upon a time",
        "max_tokens": 10,
        "temperature": 0.0,
        "seed": 42,
    },
]

# ============================================================================
# Memory Testing Data
# ============================================================================

# Prompts for testing memory management and KV cache
MEMORY_TEST_PROMPTS: List[Dict[str, any]] = [
    {"prompt": "Short", "expected_kv_cache_size": "small"},
    {"prompt": "A" * 100, "expected_kv_cache_size": "medium"},
    {"prompt": "B" * 500, "expected_kv_cache_size": "large"},
]

# ============================================================================
# Sampling Strategy Test Data
# ============================================================================

SAMPLING_TEST_CASES: List[Dict[str, any]] = [
    {
        "name": "greedy",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    },
    {
        "name": "nucleus_sampling",
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": -1,
        "repetition_penalty": 1.0,
    },
    {
        "name": "top_k_sampling",
        "temperature": 0.8,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.0,
    },
    {
        "name": "with_repetition_penalty",
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": -1,
        "repetition_penalty": 1.2,
    },
]

# ============================================================================
# Utility Functions
# ============================================================================

def get_prompt_by_length(length: str) -> str:
    """
    Get a sample prompt by length category.

    Args:
        length: One of "short", "medium", "long"

    Returns:
        Sample prompt string
    """
    if length == "short":
        return SAMPLE_PROMPTS_SHORT[0]
    elif length == "medium":
        return SAMPLE_PROMPTS_MEDIUM[0]
    elif length == "long":
        return SAMPLE_PROMPTS_LONG[0]
    else:
        raise ValueError(f"Unknown length category: {length}")


def get_edge_case_by_type(case_type: str) -> str:
    """
    Get an edge case prompt by type.

    Args:
        case_type: Type of edge case (e.g., "empty_string", "emoji_only")

    Returns:
        Edge case prompt string
    """
    for prompt, prompt_type in EDGE_CASE_PROMPTS:
        if prompt_type == case_type:
            return prompt
    raise ValueError(f"Unknown edge case type: {case_type}")


def get_test_scenario(name: str) -> Dict[str, any]:
    """
    Get a test scenario by name.

    Args:
        name: Name of the test scenario

    Returns:
        Test scenario dictionary
    """
    for scenario in TEST_SCENARIOS:
        if scenario["name"] == name:
            return scenario
    raise ValueError(f"Unknown test scenario: {name}")
