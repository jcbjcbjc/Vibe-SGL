"""
Weight loading utilities for Qwen3 model.

This module provides utilities for loading pretrained Qwen3 weights from
HuggingFace checkpoint format (safetensors or pytorch_model.bin), mapping
weight names from HuggingFace format to custom model parameter names, and
validating weight shapes against model architecture.
"""

import os
from typing import Dict

import torch
from transformers import AutoModelForCausalLM

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


def load_checkpoint(model_name_or_path: str) -> Dict[str, torch.Tensor]:
    """Load checkpoint weights from HuggingFace model.

    This function loads pretrained weights from a HuggingFace model checkpoint,
    supporting both safetensors and pytorch_model.bin formats. It automatically
    detects the available format and loads accordingly.

    Args:
        model_name_or_path: Model identifier (e.g., "Qwen/Qwen2.5-0.5B") or
            path to local model directory.

    Returns:
        Dictionary mapping weight names to tensors.

    Raises:
        ValueError: If model cannot be loaded or checkpoint not found.
    """
    try:
        # Load model using HuggingFace AutoModel
        # This automatically handles safetensors and .bin formats
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        )

        # Extract state dict
        state_dict = model.state_dict()

        return state_dict

    except Exception as e:
        raise ValueError(
            f"Failed to load checkpoint from {model_name_or_path}: {str(e)}"
        )


def create_weight_name_mapping(config: Qwen3Config) -> Dict[str, str]:
    """Create mapping from HuggingFace weight names to custom model names.

    HuggingFace Qwen models use a "model." prefix for most weights. This function
    creates a mapping that removes this prefix and maps to our custom model structure.

    Mapping examples:
        - "model.embed_tokens.weight" -> "embed_tokens.weight"
        - "model.layers.0.self_attn.q_proj.weight" -> "layers.0.self_attn.q_proj.weight"
        - "lm_head.weight" -> "lm_head.weight" (no change)

    Args:
        config: Qwen3Config instance specifying model architecture.

    Returns:
        Dictionary mapping HuggingFace weight names to custom model names.
    """
    mapping = {}

    # Embedding layer
    mapping["model.embed_tokens.weight"] = "embed_tokens.weight"

    # Transformer layers
    for layer_idx in range(config.num_hidden_layers):
        # Attention weights
        mapping[f"model.layers.{layer_idx}.self_attn.q_proj.weight"] = (
            f"layers.{layer_idx}.self_attn.q_proj.weight"
        )
        mapping[f"model.layers.{layer_idx}.self_attn.k_proj.weight"] = (
            f"layers.{layer_idx}.self_attn.k_proj.weight"
        )
        mapping[f"model.layers.{layer_idx}.self_attn.v_proj.weight"] = (
            f"layers.{layer_idx}.self_attn.v_proj.weight"
        )
        mapping[f"model.layers.{layer_idx}.self_attn.o_proj.weight"] = (
            f"layers.{layer_idx}.self_attn.o_proj.weight"
        )

        # Attention biases (if present)
        mapping[f"model.layers.{layer_idx}.self_attn.q_proj.bias"] = (
            f"layers.{layer_idx}.self_attn.q_proj.bias"
        )
        mapping[f"model.layers.{layer_idx}.self_attn.k_proj.bias"] = (
            f"layers.{layer_idx}.self_attn.k_proj.bias"
        )
        mapping[f"model.layers.{layer_idx}.self_attn.v_proj.bias"] = (
            f"layers.{layer_idx}.self_attn.v_proj.bias"
        )
        mapping[f"model.layers.{layer_idx}.self_attn.o_proj.bias"] = (
            f"layers.{layer_idx}.self_attn.o_proj.bias"
        )

        # FFN weights
        mapping[f"model.layers.{layer_idx}.mlp.gate_proj.weight"] = (
            f"layers.{layer_idx}.mlp.gate_proj.weight"
        )
        mapping[f"model.layers.{layer_idx}.mlp.up_proj.weight"] = (
            f"layers.{layer_idx}.mlp.up_proj.weight"
        )
        mapping[f"model.layers.{layer_idx}.mlp.down_proj.weight"] = (
            f"layers.{layer_idx}.mlp.down_proj.weight"
        )

        # Normalization layers
        mapping[f"model.layers.{layer_idx}.input_layernorm.weight"] = (
            f"layers.{layer_idx}.input_layernorm.weight"
        )
        mapping[f"model.layers.{layer_idx}.post_attention_layernorm.weight"] = (
            f"layers.{layer_idx}.post_attention_layernorm.weight"
        )

    # Final layer norm
    mapping["model.norm.weight"] = "norm.weight"

    # LM head (no "model." prefix in HuggingFace)
    mapping["lm_head.weight"] = "lm_head.weight"

    return mapping


def validate_weight_shapes(
    state_dict: Dict[str, torch.Tensor], config: Qwen3Config
) -> None:
    """Validate that weight shapes match model architecture.

    This function checks that all weights in the state dict have the correct
    shapes according to the model configuration. It validates embedding layers,
    attention projections, FFN layers, normalization layers, and LM head.

    Args:
        state_dict: Dictionary of weight tensors (using custom naming).
        config: Qwen3Config instance specifying expected dimensions.

    Raises:
        ValueError: If any weight has incorrect shape.
    """
    head_dim = config.hidden_size // config.num_attention_heads

    # Define expected shapes for each weight type
    expected_shapes = {
        # Embedding layer
        "embed_tokens.weight": (config.vocab_size, config.hidden_size),
        # LM head
        "lm_head.weight": (config.vocab_size, config.hidden_size),
        # Final norm
        "norm.weight": (config.hidden_size,),
    }

    # Add layer-specific shapes
    for layer_idx in range(config.num_hidden_layers):
        # Attention projections
        expected_shapes[f"layers.{layer_idx}.self_attn.q_proj.weight"] = (
            config.hidden_size,
            config.hidden_size,
        )
        expected_shapes[f"layers.{layer_idx}.self_attn.k_proj.weight"] = (
            config.num_key_value_heads * head_dim,
            config.hidden_size,
        )
        expected_shapes[f"layers.{layer_idx}.self_attn.v_proj.weight"] = (
            config.num_key_value_heads * head_dim,
            config.hidden_size,
        )
        expected_shapes[f"layers.{layer_idx}.self_attn.o_proj.weight"] = (
            config.hidden_size,
            config.hidden_size,
        )

        # FFN projections
        expected_shapes[f"layers.{layer_idx}.mlp.gate_proj.weight"] = (
            config.intermediate_size,
            config.hidden_size,
        )
        expected_shapes[f"layers.{layer_idx}.mlp.up_proj.weight"] = (
            config.intermediate_size,
            config.hidden_size,
        )
        expected_shapes[f"layers.{layer_idx}.mlp.down_proj.weight"] = (
            config.hidden_size,
            config.intermediate_size,
        )

        # Normalization layers
        expected_shapes[f"layers.{layer_idx}.input_layernorm.weight"] = (
            config.hidden_size,
        )
        expected_shapes[f"layers.{layer_idx}.post_attention_layernorm.weight"] = (
            config.hidden_size,
        )

    # Validate shapes
    for weight_name, tensor in state_dict.items():
        if weight_name in expected_shapes:
            expected_shape = expected_shapes[weight_name]
            actual_shape = tuple(tensor.shape)

            if actual_shape != expected_shape:
                raise ValueError(
                    f"Weight '{weight_name}' shape mismatch: "
                    f"expected {expected_shape}, got {actual_shape}"
                )


def load_and_map_weights(
    model_name_or_path: str, config: Qwen3Config
) -> Dict[str, torch.Tensor]:
    """Load and map weights from HuggingFace checkpoint to custom model format.

    This is the main entry point for loading pretrained weights. It performs
    the complete workflow:
    1. Load checkpoint from HuggingFace
    2. Create name mapping from HuggingFace to custom format
    3. Map weights to custom names
    4. Validate weight shapes

    Args:
        model_name_or_path: Model identifier or path to model directory.
        config: Qwen3Config instance specifying model architecture.

    Returns:
        Dictionary mapping custom weight names to tensors, ready to load
        into custom Qwen3 model.

    Raises:
        ValueError: If checkpoint cannot be loaded or shapes are invalid.
    """
    # Load checkpoint
    hf_state_dict = load_checkpoint(model_name_or_path)

    # Create name mapping
    name_mapping = create_weight_name_mapping(config)

    # Map weights to custom names
    mapped_state_dict = {}
    for hf_name, custom_name in name_mapping.items():
        if hf_name in hf_state_dict:
            mapped_state_dict[custom_name] = hf_state_dict[hf_name]

    # Validate shapes
    validate_weight_shapes(mapped_state_dict, config)

    return mapped_state_dict


def load_tp_weights(
    model_name_or_path: str,
    config: Qwen3Config,
    tp_degree: int,
    rank: int,
) -> Dict[str, torch.Tensor]:
    """Load and partition weights for tensor parallelism.

    This function loads weights from a HuggingFace checkpoint and partitions them
    across TP ranks. It handles:
    - Column parallel layers (Q/K/V projections, FFN up/gate): split output dimension
    - Row parallel layers (O projection, FFN down): split input dimension
    - Non-partitioned layers (embeddings, norms): replicate across all ranks

    Args:
        model_name_or_path: Model identifier or path to model directory.
        config: Qwen3Config instance specifying model architecture.
        tp_degree: Tensor parallelism degree (number of ranks).
        rank: Current TP rank (0 to tp_degree-1).

    Returns:
        Dictionary mapping weight names to partitioned tensors for this rank.

    Raises:
        ValueError: If checkpoint cannot be loaded, shapes are invalid, or
            dimensions are not divisible by tp_degree.
    """
    # Load full weights
    full_state_dict = load_and_map_weights(model_name_or_path, config)

    # If no TP, return full weights
    if tp_degree == 1:
        return full_state_dict

    # Validate rank
    if rank < 0 or rank >= tp_degree:
        raise ValueError(f"Invalid rank {rank} for tp_degree {tp_degree}")

    # Partition weights for this rank
    partitioned_state_dict = {}

    for weight_name, weight_tensor in full_state_dict.items():
        # Determine if this weight should be partitioned
        if "q_proj.weight" in weight_name or "k_proj.weight" in weight_name or "v_proj.weight" in weight_name:
            # Column parallel: split output dimension (dim 0)
            out_features = weight_tensor.shape[0]
            if out_features % tp_degree != 0:
                raise ValueError(
                    f"Weight '{weight_name}' output features {out_features} not divisible by tp_degree {tp_degree}"
                )
            partition_size = out_features // tp_degree
            start_idx = rank * partition_size
            end_idx = start_idx + partition_size
            partitioned_state_dict[weight_name] = weight_tensor[start_idx:end_idx, :]

        elif "o_proj.weight" in weight_name:
            # Row parallel: split input dimension (dim 1)
            in_features = weight_tensor.shape[1]
            if in_features % tp_degree != 0:
                raise ValueError(
                    f"Weight '{weight_name}' input features {in_features} not divisible by tp_degree {tp_degree}"
                )
            partition_size = in_features // tp_degree
            start_idx = rank * partition_size
            end_idx = start_idx + partition_size
            partitioned_state_dict[weight_name] = weight_tensor[:, start_idx:end_idx]

        elif "up_proj.weight" in weight_name or "gate_proj.weight" in weight_name:
            # Column parallel: split output dimension (dim 0)
            out_features = weight_tensor.shape[0]
            if out_features % tp_degree != 0:
                raise ValueError(
                    f"Weight '{weight_name}' output features {out_features} not divisible by tp_degree {tp_degree}"
                )
            partition_size = out_features // tp_degree
            start_idx = rank * partition_size
            end_idx = start_idx + partition_size
            partitioned_state_dict[weight_name] = weight_tensor[start_idx:end_idx, :]

        elif "down_proj.weight" in weight_name:
            # Row parallel: split input dimension (dim 1)
            in_features = weight_tensor.shape[1]
            if in_features % tp_degree != 0:
                raise ValueError(
                    f"Weight '{weight_name}' input features {in_features} not divisible by tp_degree {tp_degree}"
                )
            partition_size = in_features // tp_degree
            start_idx = rank * partition_size
            end_idx = start_idx + partition_size
            partitioned_state_dict[weight_name] = weight_tensor[:, start_idx:end_idx]

        else:
            # Non-partitioned weights (embeddings, norms, etc.): replicate
            partitioned_state_dict[weight_name] = weight_tensor

    return partitioned_state_dict

