"""
Tests for Qwen3Config loading from HuggingFace.

This module tests the Qwen3Config class which loads model configuration
from HuggingFace checkpoints and validates the configuration parameters.

Following TDD: These tests are written BEFORE implementation.
"""

import pytest
from transformers import AutoConfig

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_qwen3_config_from_pretrained_loads() -> None:
    """Test that Qwen3Config can load from HuggingFace model path."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert config is not None
    assert isinstance(config, Qwen3Config)


@pytest.mark.unit
def test_qwen3_config_has_vocab_size() -> None:
    """Test that loaded config has vocab_size attribute."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "vocab_size")
    assert isinstance(config.vocab_size, int)
    assert config.vocab_size > 0
    # Qwen models typically have vocab size around 150k
    assert config.vocab_size > 100_000


@pytest.mark.unit
def test_qwen3_config_has_hidden_size() -> None:
    """Test that loaded config has hidden_size attribute."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "hidden_size")
    assert isinstance(config.hidden_size, int)
    assert config.hidden_size > 0


@pytest.mark.unit
def test_qwen3_config_has_num_hidden_layers() -> None:
    """Test that loaded config has num_hidden_layers attribute."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "num_hidden_layers")
    assert isinstance(config.num_hidden_layers, int)
    assert config.num_hidden_layers > 0


@pytest.mark.unit
def test_qwen3_config_has_num_attention_heads() -> None:
    """Test that loaded config has num_attention_heads attribute."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "num_attention_heads")
    assert isinstance(config.num_attention_heads, int)
    assert config.num_attention_heads > 0


@pytest.mark.unit
def test_qwen3_config_has_num_key_value_heads() -> None:
    """Test that loaded config has num_key_value_heads for GQA."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "num_key_value_heads")
    assert isinstance(config.num_key_value_heads, int)
    assert config.num_key_value_heads > 0
    # For GQA, num_key_value_heads should be <= num_attention_heads
    assert config.num_key_value_heads <= config.num_attention_heads


@pytest.mark.unit
def test_qwen3_config_has_intermediate_size() -> None:
    """Test that loaded config has intermediate_size for FFN."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "intermediate_size")
    assert isinstance(config.intermediate_size, int)
    assert config.intermediate_size > 0
    # Intermediate size is typically larger than hidden_size
    assert config.intermediate_size > config.hidden_size


@pytest.mark.unit
def test_qwen3_config_has_max_position_embeddings() -> None:
    """Test that loaded config has max_position_embeddings."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "max_position_embeddings")
    assert isinstance(config.max_position_embeddings, int)
    assert config.max_position_embeddings > 0


@pytest.mark.unit
def test_qwen3_config_has_rms_norm_eps() -> None:
    """Test that loaded config has rms_norm_eps for RMSNorm."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "rms_norm_eps")
    assert isinstance(config.rms_norm_eps, float)
    assert config.rms_norm_eps > 0
    # Typical epsilon value is around 1e-6
    assert config.rms_norm_eps < 1e-3


@pytest.mark.unit
def test_qwen3_config_has_rope_theta() -> None:
    """Test that loaded config has rope_theta for RoPE."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "rope_theta")
    assert isinstance(config.rope_theta, (int, float))
    assert config.rope_theta > 0


@pytest.mark.unit
def test_qwen3_config_has_hidden_act() -> None:
    """Test that loaded config has hidden_act for activation function."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    assert hasattr(config, "hidden_act")
    assert isinstance(config.hidden_act, str)
    # Qwen3 uses SwiGLU activation
    assert len(config.hidden_act) > 0


@pytest.mark.unit
def test_qwen3_config_validates_architecture() -> None:
    """Test that config validates Qwen3 architecture constraints."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)

    # Validate head dimension is consistent
    head_dim = config.hidden_size // config.num_attention_heads
    assert head_dim > 0
    assert config.hidden_size % config.num_attention_heads == 0

    # Validate GQA constraint: num_attention_heads % num_key_value_heads == 0
    assert config.num_attention_heads % config.num_key_value_heads == 0


@pytest.mark.unit
def test_qwen3_config_matches_huggingface_config() -> None:
    """Test that Qwen3Config values match HuggingFace AutoConfig."""
    model_name = "Qwen/Qwen2.5-0.5B"

    # Load both configs
    hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    qwen3_config = Qwen3Config.from_pretrained(model_name)

    # Compare key attributes
    assert qwen3_config.vocab_size == hf_config.vocab_size
    assert qwen3_config.hidden_size == hf_config.hidden_size
    assert qwen3_config.num_hidden_layers == hf_config.num_hidden_layers
    assert qwen3_config.num_attention_heads == hf_config.num_attention_heads
    assert qwen3_config.num_key_value_heads == hf_config.num_key_value_heads
    assert qwen3_config.intermediate_size == hf_config.intermediate_size
    assert qwen3_config.max_position_embeddings == hf_config.max_position_embeddings
    assert qwen3_config.rms_norm_eps == hf_config.rms_norm_eps
    assert qwen3_config.rope_theta == hf_config.rope_theta


@pytest.mark.unit
def test_qwen3_config_from_dict() -> None:
    """Test that Qwen3Config can be created from a dictionary."""
    config_dict = {
        "vocab_size": 151936,
        "hidden_size": 896,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 4864,
        "max_position_embeddings": 32768,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "hidden_act": "silu",
    }

    config = Qwen3Config(**config_dict)

    assert config.vocab_size == 151936
    assert config.hidden_size == 896
    assert config.num_hidden_layers == 24
    assert config.num_attention_heads == 14
    assert config.num_key_value_heads == 2


@pytest.mark.unit
def test_qwen3_config_to_dict() -> None:
    """Test that Qwen3Config can be converted to a dictionary."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)
    config_dict = config.to_dict()

    assert isinstance(config_dict, dict)
    assert "vocab_size" in config_dict
    assert "hidden_size" in config_dict
    assert "num_hidden_layers" in config_dict
    assert config_dict["vocab_size"] == config.vocab_size


@pytest.mark.unit
def test_qwen3_config_repr() -> None:
    """Test that Qwen3Config has a readable string representation."""
    model_name = "Qwen/Qwen2.5-0.5B"

    config = Qwen3Config.from_pretrained(model_name)
    config_str = repr(config)

    assert isinstance(config_str, str)
    assert len(config_str) > 0
    # Should contain class name
    assert "Qwen3Config" in config_str

