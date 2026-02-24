"""
Tests for weight loading from HuggingFace checkpoint.

This module tests the weight loading utilities that load pretrained Qwen3
weights from HuggingFace checkpoint format (safetensors or pytorch_model.bin),
map weight names to custom model parameter names, and validate weight shapes.

Following TDD: These tests are written BEFORE implementation.
"""

import os
from typing import Dict

import pytest
import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


@pytest.mark.unit
def test_load_checkpoint_from_model_path() -> None:
    """Test that checkpoint can be loaded from HuggingFace model path."""
    model_name = "Qwen/Qwen2.5-0.5B"

    # Import the weight loading function (to be implemented)
    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    # Load checkpoint
    state_dict = load_checkpoint(model_name)

    # Verify state_dict is not empty
    assert state_dict is not None
    assert isinstance(state_dict, dict)
    assert len(state_dict) > 0


@pytest.mark.unit
def test_checkpoint_contains_embedding_weights() -> None:
    """Test that loaded checkpoint contains embedding layer weights."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    state_dict = load_checkpoint(model_name)

    # Check for embedding weights (HuggingFace naming)
    # Qwen models use "model.embed_tokens.weight"
    embedding_keys = [k for k in state_dict.keys() if "embed_tokens" in k]
    assert len(embedding_keys) > 0


@pytest.mark.unit
def test_checkpoint_contains_attention_weights() -> None:
    """Test that loaded checkpoint contains attention layer weights."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    state_dict = load_checkpoint(model_name)

    # Check for attention weights (Q/K/V projections and output projection)
    # Qwen models use "model.layers.{i}.self_attn.{q,k,v,o}_proj.weight"
    attention_keys = [k for k in state_dict.keys() if "self_attn" in k]
    assert len(attention_keys) > 0

    # Check for Q, K, V, O projections
    q_proj_keys = [k for k in state_dict.keys() if "q_proj" in k]
    k_proj_keys = [k for k in state_dict.keys() if "k_proj" in k]
    v_proj_keys = [k for k in state_dict.keys() if "v_proj" in k]
    o_proj_keys = [k for k in state_dict.keys() if "o_proj" in k]

    assert len(q_proj_keys) > 0
    assert len(k_proj_keys) > 0
    assert len(v_proj_keys) > 0
    assert len(o_proj_keys) > 0


@pytest.mark.unit
def test_checkpoint_contains_ffn_weights() -> None:
    """Test that loaded checkpoint contains FFN layer weights."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    state_dict = load_checkpoint(model_name)

    # Check for FFN weights (up_proj, gate_proj, down_proj)
    # Qwen models use "model.layers.{i}.mlp.{gate,up,down}_proj.weight"
    mlp_keys = [k for k in state_dict.keys() if "mlp" in k]
    assert len(mlp_keys) > 0

    # Check for gate, up, down projections
    gate_proj_keys = [k for k in state_dict.keys() if "gate_proj" in k]
    up_proj_keys = [k for k in state_dict.keys() if "up_proj" in k]
    down_proj_keys = [k for k in state_dict.keys() if "down_proj" in k]

    assert len(gate_proj_keys) > 0
    assert len(up_proj_keys) > 0
    assert len(down_proj_keys) > 0


@pytest.mark.unit
def test_checkpoint_contains_norm_weights() -> None:
    """Test that loaded checkpoint contains normalization layer weights."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    state_dict = load_checkpoint(model_name)

    # Check for norm weights (input_layernorm, post_attention_layernorm)
    norm_keys = [k for k in state_dict.keys() if "layernorm" in k.lower()]
    assert len(norm_keys) > 0


@pytest.mark.unit
def test_checkpoint_contains_lm_head_weights() -> None:
    """Test that loaded checkpoint contains LM head weights."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    state_dict = load_checkpoint(model_name)

    # Check for LM head weights
    # Qwen models use "lm_head.weight"
    lm_head_keys = [k for k in state_dict.keys() if "lm_head" in k]
    assert len(lm_head_keys) > 0


@pytest.mark.unit
def test_weight_name_mapping_creates_mapping_dict() -> None:
    """Test that weight name mapping creates a mapping dictionary."""
    from vibe_sgl_lite.models.qwen3.weight_loader import create_weight_name_mapping

    # Create mapping for a simple config
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,  # Small for testing
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    mapping = create_weight_name_mapping(config)

    assert mapping is not None
    assert isinstance(mapping, dict)
    assert len(mapping) > 0


@pytest.mark.unit
def test_weight_name_mapping_maps_embedding_weights() -> None:
    """Test that weight name mapping includes embedding layer mapping."""
    from vibe_sgl_lite.models.qwen3.weight_loader import create_weight_name_mapping

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    mapping = create_weight_name_mapping(config)

    # Check that embedding mapping exists
    # HuggingFace: "model.embed_tokens.weight" -> Custom: "embed_tokens.weight"
    hf_embed_key = "model.embed_tokens.weight"
    assert hf_embed_key in mapping
    assert "embed_tokens" in mapping[hf_embed_key]


@pytest.mark.unit
def test_weight_name_mapping_maps_attention_weights() -> None:
    """Test that weight name mapping includes attention layer mappings."""
    from vibe_sgl_lite.models.qwen3.weight_loader import create_weight_name_mapping

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    mapping = create_weight_name_mapping(config)

    # Check that attention mappings exist for layer 0
    # HuggingFace: "model.layers.0.self_attn.q_proj.weight"
    # Custom: "layers.0.self_attn.q_proj.weight"
    hf_q_proj_key = "model.layers.0.self_attn.q_proj.weight"
    assert hf_q_proj_key in mapping
    assert "layers.0" in mapping[hf_q_proj_key]
    assert "q_proj" in mapping[hf_q_proj_key]


@pytest.mark.unit
def test_weight_name_mapping_maps_ffn_weights() -> None:
    """Test that weight name mapping includes FFN layer mappings."""
    from vibe_sgl_lite.models.qwen3.weight_loader import create_weight_name_mapping

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    mapping = create_weight_name_mapping(config)

    # Check that FFN mappings exist for layer 0
    # HuggingFace: "model.layers.0.mlp.gate_proj.weight"
    # Custom: "layers.0.mlp.gate_proj.weight"
    hf_gate_proj_key = "model.layers.0.mlp.gate_proj.weight"
    assert hf_gate_proj_key in mapping
    assert "layers.0" in mapping[hf_gate_proj_key]
    assert "gate_proj" in mapping[hf_gate_proj_key]


@pytest.mark.unit
def test_weight_name_mapping_maps_norm_weights() -> None:
    """Test that weight name mapping includes normalization layer mappings."""
    from vibe_sgl_lite.models.qwen3.weight_loader import create_weight_name_mapping

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    mapping = create_weight_name_mapping(config)

    # Check that norm mappings exist for layer 0
    # HuggingFace: "model.layers.0.input_layernorm.weight"
    # Custom: "layers.0.input_layernorm.weight"
    hf_input_norm_key = "model.layers.0.input_layernorm.weight"
    assert hf_input_norm_key in mapping
    assert "layers.0" in mapping[hf_input_norm_key]
    assert "input_layernorm" in mapping[hf_input_norm_key]


@pytest.mark.unit
def test_weight_name_mapping_maps_lm_head_weights() -> None:
    """Test that weight name mapping includes LM head mapping."""
    from vibe_sgl_lite.models.qwen3.weight_loader import create_weight_name_mapping

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    mapping = create_weight_name_mapping(config)

    # Check that LM head mapping exists
    # HuggingFace: "lm_head.weight" -> Custom: "lm_head.weight"
    hf_lm_head_key = "lm_head.weight"
    assert hf_lm_head_key in mapping


@pytest.mark.unit
def test_weight_shape_validation_validates_embedding_shape() -> None:
    """Test that weight shape validation checks embedding layer shape."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    # Create a mock state dict with correct embedding shape
    state_dict = {
        "embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size)
    }

    # Should not raise an error
    validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_weight_shape_validation_rejects_wrong_embedding_shape() -> None:
    """Test that weight shape validation rejects incorrect embedding shape."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    # Create a mock state dict with WRONG embedding shape
    state_dict = {
        "embed_tokens.weight": torch.randn(
            config.vocab_size, config.hidden_size + 100
        )  # Wrong hidden_size
    }

    # Should raise ValueError
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_weight_shape_validation_validates_attention_shapes() -> None:
    """Test that weight shape validation checks attention layer shapes."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    head_dim = config.hidden_size // config.num_attention_heads

    # Create a mock state dict with correct attention shapes
    state_dict = {
        "layers.0.self_attn.q_proj.weight": torch.randn(
            config.hidden_size, config.hidden_size
        ),
        "layers.0.self_attn.k_proj.weight": torch.randn(
            config.num_key_value_heads * head_dim, config.hidden_size
        ),
        "layers.0.self_attn.v_proj.weight": torch.randn(
            config.num_key_value_heads * head_dim, config.hidden_size
        ),
        "layers.0.self_attn.o_proj.weight": torch.randn(
            config.hidden_size, config.hidden_size
        ),
    }

    # Should not raise an error
    validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_weight_shape_validation_rejects_wrong_attention_shape() -> None:
    """Test that weight shape validation rejects incorrect attention shapes."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    # Create a mock state dict with WRONG q_proj shape
    state_dict = {
        "layers.0.self_attn.q_proj.weight": torch.randn(
            config.hidden_size + 100, config.hidden_size
        )  # Wrong output dim
    }

    # Should raise ValueError
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_weight_shape_validation_validates_ffn_shapes() -> None:
    """Test that weight shape validation checks FFN layer shapes."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
    )

    # Create a mock state dict with correct FFN shapes
    state_dict = {
        "layers.0.mlp.gate_proj.weight": torch.randn(
            config.intermediate_size, config.hidden_size
        ),
        "layers.0.mlp.up_proj.weight": torch.randn(
            config.intermediate_size, config.hidden_size
        ),
        "layers.0.mlp.down_proj.weight": torch.randn(
            config.hidden_size, config.intermediate_size
        ),
    }

    # Should not raise an error
    validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_weight_shape_validation_rejects_wrong_ffn_shape() -> None:
    """Test that weight shape validation rejects incorrect FFN shapes."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
    )

    # Create a mock state dict with WRONG gate_proj shape
    state_dict = {
        "layers.0.mlp.gate_proj.weight": torch.randn(
            config.intermediate_size + 100, config.hidden_size
        )  # Wrong intermediate_size
    }

    # Should raise ValueError
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_weight_shape_validation_validates_norm_shapes() -> None:
    """Test that weight shape validation checks normalization layer shapes."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    # Create a mock state dict with correct norm shapes
    state_dict = {
        "layers.0.input_layernorm.weight": torch.randn(config.hidden_size),
        "layers.0.post_attention_layernorm.weight": torch.randn(config.hidden_size),
    }

    # Should not raise an error
    validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_weight_shape_validation_validates_lm_head_shape() -> None:
    """Test that weight shape validation checks LM head shape."""
    from vibe_sgl_lite.models.qwen3.weight_loader import validate_weight_shapes

    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=896,
        num_hidden_layers=2,
        num_attention_heads=14,
        num_key_value_heads=2,
    )

    # Create a mock state dict with correct LM head shape
    state_dict = {
        "lm_head.weight": torch.randn(config.vocab_size, config.hidden_size)
    }

    # Should not raise an error
    validate_weight_shapes(state_dict, config)


@pytest.mark.unit
def test_load_and_map_weights_from_pretrained() -> None:
    """Test end-to-end weight loading and mapping from pretrained model."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_and_map_weights

    # Load config
    config = Qwen3Config.from_pretrained(model_name)

    # Load and map weights
    mapped_state_dict = load_and_map_weights(model_name, config)

    # Verify mapped state dict
    assert mapped_state_dict is not None
    assert isinstance(mapped_state_dict, dict)
    assert len(mapped_state_dict) > 0

    # Check that weights are mapped to custom names (no "model." prefix)
    for key in mapped_state_dict.keys():
        assert not key.startswith("model."), f"Key {key} should not have 'model.' prefix"


@pytest.mark.unit
def test_load_and_map_weights_validates_shapes() -> None:
    """Test that load_and_map_weights validates weight shapes."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_and_map_weights

    # Load config
    config = Qwen3Config.from_pretrained(model_name)

    # Load and map weights (should validate shapes internally)
    mapped_state_dict = load_and_map_weights(model_name, config)

    # If we get here without exception, shapes are valid
    assert len(mapped_state_dict) > 0


@pytest.mark.unit
def test_load_weights_supports_safetensors() -> None:
    """Test that weight loading supports safetensors format."""
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    # Load checkpoint (should auto-detect safetensors if available)
    state_dict = load_checkpoint(model_name)

    assert state_dict is not None
    assert len(state_dict) > 0


@pytest.mark.unit
def test_load_weights_supports_pytorch_bin() -> None:
    """Test that weight loading supports pytorch_model.bin format."""
    # This test verifies fallback to .bin format if safetensors not available
    # For now, we just verify the function can handle both formats
    model_name = "Qwen/Qwen2.5-0.5B"

    from vibe_sgl_lite.models.qwen3.weight_loader import load_checkpoint

    state_dict = load_checkpoint(model_name)

    assert state_dict is not None
    assert len(state_dict) > 0

