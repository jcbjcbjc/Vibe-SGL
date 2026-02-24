"""
Tests for test data module.

This module validates that test data is properly structured and accessible.
"""

import pytest

from tests.data import (
    BENCHMARK_BATCH_SIZES,
    BENCHMARK_GEN_LENGTHS,
    BENCHMARK_PROMPTS,
    BENCHMARK_SEQ_LENGTHS,
    BATCHING_REQUESTS,
    CORRECTNESS_PROMPTS,
    EDGE_CASE_PROMPTS,
    EP_TEST_CONFIGS,
    EXPECTED_OUTPUTS,
    MEMORY_TEST_PROMPTS,
    PREFIX_CACHE_PROMPTS,
    SAMPLE_PROMPTS_LONG,
    SAMPLE_PROMPTS_MEDIUM,
    SAMPLE_PROMPTS_MULTI_TURN,
    SAMPLE_PROMPTS_SHORT,
    SAMPLING_TEST_CASES,
    SYSTEM_PROMPTS,
    TEST_SCENARIOS,
    TP_TEST_CONFIGS,
    get_edge_case_by_type,
    get_prompt_by_length,
    get_test_scenario,
)


@pytest.mark.unit
def test_sample_prompts_short_not_empty() -> None:
    """Test that short sample prompts list is not empty."""
    assert len(SAMPLE_PROMPTS_SHORT) > 0
    assert all(isinstance(p, str) for p in SAMPLE_PROMPTS_SHORT)
    assert all(len(p) > 0 for p in SAMPLE_PROMPTS_SHORT)


@pytest.mark.unit
def test_sample_prompts_medium_not_empty() -> None:
    """Test that medium sample prompts list is not empty."""
    assert len(SAMPLE_PROMPTS_MEDIUM) > 0
    assert all(isinstance(p, str) for p in SAMPLE_PROMPTS_MEDIUM)
    assert all(len(p) > len(SAMPLE_PROMPTS_SHORT[0]) for p in SAMPLE_PROMPTS_MEDIUM)


@pytest.mark.unit
def test_sample_prompts_long_not_empty() -> None:
    """Test that long sample prompts list is not empty."""
    assert len(SAMPLE_PROMPTS_LONG) > 0
    assert all(isinstance(p, str) for p in SAMPLE_PROMPTS_LONG)
    assert all(len(p) > len(SAMPLE_PROMPTS_MEDIUM[0]) for p in SAMPLE_PROMPTS_LONG)


@pytest.mark.unit
def test_sample_prompts_multi_turn_structure() -> None:
    """Test that multi-turn prompts have correct structure."""
    assert len(SAMPLE_PROMPTS_MULTI_TURN) > 0
    for conversation in SAMPLE_PROMPTS_MULTI_TURN:
        assert isinstance(conversation, list)
        assert len(conversation) > 1  # Multi-turn means at least 2 turns
        assert all(isinstance(turn, str) for turn in conversation)


@pytest.mark.unit
def test_edge_case_prompts_structure() -> None:
    """Test that edge case prompts have correct structure."""
    assert len(EDGE_CASE_PROMPTS) > 0
    for prompt, case_type in EDGE_CASE_PROMPTS:
        assert isinstance(prompt, str)
        assert isinstance(case_type, str)
        assert len(case_type) > 0


@pytest.mark.unit
def test_edge_case_prompts_include_empty() -> None:
    """Test that edge cases include empty string."""
    case_types = [case_type for _, case_type in EDGE_CASE_PROMPTS]
    assert "empty_string" in case_types


@pytest.mark.unit
def test_edge_case_prompts_include_special_chars() -> None:
    """Test that edge cases include special characters."""
    case_types = [case_type for _, case_type in EDGE_CASE_PROMPTS]
    assert "special_characters" in case_types


@pytest.mark.unit
def test_expected_outputs_structure() -> None:
    """Test that expected outputs dictionary is properly structured."""
    assert len(EXPECTED_OUTPUTS) > 0
    for prompt, output in EXPECTED_OUTPUTS.items():
        assert isinstance(prompt, str)
        assert isinstance(output, str)
        assert len(prompt) > 0
        assert len(output) > 0


@pytest.mark.unit
def test_benchmark_batch_sizes_valid() -> None:
    """Test that benchmark batch sizes are valid."""
    assert len(BENCHMARK_BATCH_SIZES) > 0
    assert all(isinstance(size, int) for size in BENCHMARK_BATCH_SIZES)
    assert all(size > 0 for size in BENCHMARK_BATCH_SIZES)
    # Should be in ascending order
    assert BENCHMARK_BATCH_SIZES == sorted(BENCHMARK_BATCH_SIZES)


@pytest.mark.unit
def test_benchmark_seq_lengths_valid() -> None:
    """Test that benchmark sequence lengths are valid."""
    assert len(BENCHMARK_SEQ_LENGTHS) > 0
    assert all(isinstance(length, int) for length in BENCHMARK_SEQ_LENGTHS)
    assert all(length > 0 for length in BENCHMARK_SEQ_LENGTHS)
    # Should be in ascending order
    assert BENCHMARK_SEQ_LENGTHS == sorted(BENCHMARK_SEQ_LENGTHS)


@pytest.mark.unit
def test_benchmark_gen_lengths_valid() -> None:
    """Test that benchmark generation lengths are valid."""
    assert len(BENCHMARK_GEN_LENGTHS) > 0
    assert all(isinstance(length, int) for length in BENCHMARK_GEN_LENGTHS)
    assert all(length > 0 for length in BENCHMARK_GEN_LENGTHS)


@pytest.mark.unit
def test_benchmark_prompts_structure() -> None:
    """Test that benchmark prompts have correct structure."""
    assert len(BENCHMARK_PROMPTS) > 0
    assert "short" in BENCHMARK_PROMPTS
    assert "medium" in BENCHMARK_PROMPTS
    assert "long" in BENCHMARK_PROMPTS
    # Verify length ordering
    assert len(BENCHMARK_PROMPTS["short"]) < len(BENCHMARK_PROMPTS["medium"])
    assert len(BENCHMARK_PROMPTS["medium"]) < len(BENCHMARK_PROMPTS["long"])


@pytest.mark.unit
def test_test_scenarios_structure() -> None:
    """Test that test scenarios have correct structure."""
    assert len(TEST_SCENARIOS) > 0
    required_keys = ["name", "prompt", "max_tokens", "temperature", "top_p"]
    for scenario in TEST_SCENARIOS:
        for key in required_keys:
            assert key in scenario, f"Missing key '{key}' in scenario"
        assert isinstance(scenario["name"], str)
        assert isinstance(scenario["prompt"], str)
        assert isinstance(scenario["max_tokens"], int)
        assert scenario["max_tokens"] > 0


@pytest.mark.unit
def test_prefix_cache_prompts_have_common_prefix() -> None:
    """Test that prefix cache prompts share common prefixes."""
    assert len(PREFIX_CACHE_PROMPTS) > 0
    # Check that some prompts share prefixes
    prefixes = set()
    for prompt in PREFIX_CACHE_PROMPTS:
        # Extract first few words as prefix
        words = prompt.split()[:3]
        prefix = " ".join(words)
        prefixes.add(prefix)
    # Should have fewer unique prefixes than total prompts (indicating sharing)
    assert len(prefixes) < len(PREFIX_CACHE_PROMPTS)


@pytest.mark.unit
def test_system_prompts_not_empty() -> None:
    """Test that system prompts list is not empty."""
    assert len(SYSTEM_PROMPTS) > 0
    assert all(isinstance(p, str) for p in SYSTEM_PROMPTS)
    assert all(len(p) > 0 for p in SYSTEM_PROMPTS)


@pytest.mark.unit
def test_batching_requests_structure() -> None:
    """Test that batching requests have correct structure."""
    assert len(BATCHING_REQUESTS) > 0
    required_keys = ["prompt", "max_tokens", "priority"]
    for request in BATCHING_REQUESTS:
        for key in required_keys:
            assert key in request
        assert isinstance(request["prompt"], str)
        assert isinstance(request["max_tokens"], int)
        assert isinstance(request["priority"], int)


@pytest.mark.unit
def test_tp_test_configs_structure() -> None:
    """Test that TP test configs have correct structure."""
    assert len(TP_TEST_CONFIGS) > 0
    required_keys = ["world_size", "prompt", "max_tokens"]
    for config in TP_TEST_CONFIGS:
        for key in required_keys:
            assert key in config
        assert config["world_size"] > 1  # TP requires multiple processes


@pytest.mark.unit
def test_ep_test_configs_structure() -> None:
    """Test that EP test configs have correct structure."""
    assert len(EP_TEST_CONFIGS) > 0
    required_keys = ["world_size", "num_experts", "prompt", "max_tokens"]
    for config in EP_TEST_CONFIGS:
        for key in required_keys:
            assert key in config
        assert config["world_size"] > 1  # EP requires multiple processes
        assert config["num_experts"] > 0


@pytest.mark.unit
def test_correctness_prompts_deterministic() -> None:
    """Test that correctness prompts are configured for deterministic output."""
    assert len(CORRECTNESS_PROMPTS) > 0
    for prompt_config in CORRECTNESS_PROMPTS:
        assert "temperature" in prompt_config
        assert prompt_config["temperature"] == 0.0  # Greedy decoding
        assert "seed" in prompt_config


@pytest.mark.unit
def test_memory_test_prompts_structure() -> None:
    """Test that memory test prompts have correct structure."""
    assert len(MEMORY_TEST_PROMPTS) > 0
    for prompt_config in MEMORY_TEST_PROMPTS:
        assert "prompt" in prompt_config
        assert "expected_kv_cache_size" in prompt_config


@pytest.mark.unit
def test_sampling_test_cases_structure() -> None:
    """Test that sampling test cases have correct structure."""
    assert len(SAMPLING_TEST_CASES) > 0
    required_keys = ["name", "temperature", "top_p", "top_k", "repetition_penalty"]
    for test_case in SAMPLING_TEST_CASES:
        for key in required_keys:
            assert key in test_case
        # Validate parameter ranges
        assert test_case["temperature"] >= 0.0
        assert 0.0 <= test_case["top_p"] <= 1.0
        assert test_case["repetition_penalty"] >= 1.0


@pytest.mark.unit
def test_get_prompt_by_length_short() -> None:
    """Test get_prompt_by_length for short prompts."""
    prompt = get_prompt_by_length("short")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


@pytest.mark.unit
def test_get_prompt_by_length_medium() -> None:
    """Test get_prompt_by_length for medium prompts."""
    prompt = get_prompt_by_length("medium")
    assert isinstance(prompt, str)
    assert len(prompt) > len(get_prompt_by_length("short"))


@pytest.mark.unit
def test_get_prompt_by_length_long() -> None:
    """Test get_prompt_by_length for long prompts."""
    prompt = get_prompt_by_length("long")
    assert isinstance(prompt, str)
    assert len(prompt) > len(get_prompt_by_length("medium"))


@pytest.mark.unit
def test_get_prompt_by_length_invalid() -> None:
    """Test get_prompt_by_length with invalid length."""
    with pytest.raises(ValueError):
        get_prompt_by_length("invalid")


@pytest.mark.unit
def test_get_edge_case_by_type_empty() -> None:
    """Test get_edge_case_by_type for empty string."""
    prompt = get_edge_case_by_type("empty_string")
    assert prompt == ""


@pytest.mark.unit
def test_get_edge_case_by_type_emoji() -> None:
    """Test get_edge_case_by_type for emoji only."""
    prompt = get_edge_case_by_type("emoji_only")
    assert isinstance(prompt, str)
    assert len(prompt) > 0


@pytest.mark.unit
def test_get_edge_case_by_type_invalid() -> None:
    """Test get_edge_case_by_type with invalid type."""
    with pytest.raises(ValueError):
        get_edge_case_by_type("invalid_type")


@pytest.mark.unit
def test_get_test_scenario_basic() -> None:
    """Test get_test_scenario for basic generation."""
    scenario = get_test_scenario("basic_generation")
    assert isinstance(scenario, dict)
    assert scenario["name"] == "basic_generation"
    assert "prompt" in scenario
    assert "max_tokens" in scenario


@pytest.mark.unit
def test_get_test_scenario_invalid() -> None:
    """Test get_test_scenario with invalid name."""
    with pytest.raises(ValueError):
        get_test_scenario("invalid_scenario")
