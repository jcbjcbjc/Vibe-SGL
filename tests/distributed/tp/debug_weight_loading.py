"""
Debug script to test weight loading and model architecture.

This script helps identify issues with:
- Weight loading from HuggingFace
- Model architecture differences
- TP implementation correctness
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.model import Qwen3Model
from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead


def test_single_layer_no_tp():
    """Test a single layer without TP to verify basic architecture."""
    print("=" * 80)
    print("Test 1: Single layer without TP")
    print("=" * 80)

    # Load HuggingFace model
    model_name = "Qwen/Qwen2.5-0.5B"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Get config
    hf_config = hf_model.config
    config = Qwen3Config(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        num_hidden_layers=1,  # Only 1 layer for testing
        intermediate_size=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        rope_theta=hf_config.rope_theta,
        rms_norm_eps=hf_config.rms_norm_eps,
    )

    # Create our model
    our_model = Qwen3Model(config, tp_degree=1, rank=0)
    our_lm_head = Qwen3LMHead(config)

    # Prepare input
    text = "Hello"
    input_ids = tokenizer.encode(text, return_tensors="pt")
    print(f"Input text: {text}")
    print(f"Input IDs shape: {input_ids.shape}")

    # Get HuggingFace output
    with torch.no_grad():
        hf_output = hf_model(input_ids)
        hf_logits = hf_output.logits

    print(f"\nHuggingFace output shape: {hf_logits.shape}")
    print(f"HuggingFace logits range: [{hf_logits.min().item():.4f}, {hf_logits.max().item():.4f}]")

    # Get our model output (without loading weights)
    with torch.no_grad():
        our_hidden = our_model(input_ids)
        our_logits = our_lm_head(our_hidden)

    print(f"\nOur model output shape: {our_logits.shape}")
    print(f"Our model logits range: [{our_logits.min().item():.4f}, {our_logits.max().item():.4f}]")

    # Check if shapes match
    if our_logits.shape == hf_logits.shape:
        print("\n✅ Output shapes match!")
    else:
        print(f"\n❌ Output shapes don't match!")

    # Check weight names
    print("\n" + "=" * 80)
    print("HuggingFace model weight names (first 10):")
    print("=" * 80)
    for i, name in enumerate(list(hf_model.state_dict().keys())[:10]):
        print(f"  {i+1}. {name}")

    print("\n" + "=" * 80)
    print("Our model weight names (first 10):")
    print("=" * 80)
    for i, name in enumerate(list(our_model.state_dict().keys())[:10]):
        print(f"  {i+1}. {name}")


def test_weight_name_mapping():
    """Test weight name mapping between HuggingFace and our model."""
    print("\n" + "=" * 80)
    print("Test 2: Weight name mapping")
    print("=" * 80)

    # Load HuggingFace model
    model_name = "Qwen/Qwen2.5-0.5B"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    hf_config = hf_model.config
    config = Qwen3Config(
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        num_hidden_layers=1,
        intermediate_size=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        rope_theta=hf_config.rope_theta,
        rms_norm_eps=hf_config.rms_norm_eps,
    )

    our_model = Qwen3Model(config, tp_degree=1, rank=0)

    # Try to map weights
    hf_state_dict = hf_model.state_dict()
    our_state_dict = our_model.state_dict()

    print(f"\nHuggingFace model has {len(hf_state_dict)} parameters")
    print(f"Our model has {len(our_state_dict)} parameters")

    # Check for layer 0 attention weights
    print("\n" + "-" * 80)
    print("Layer 0 attention weights:")
    print("-" * 80)

    hf_attn_keys = [k for k in hf_state_dict.keys() if "layers.0.self_attn" in k]
    our_attn_keys = [k for k in our_state_dict.keys() if "layers.0.self_attn" in k]

    print(f"\nHuggingFace attention keys ({len(hf_attn_keys)}):")
    for key in hf_attn_keys:
        shape = hf_state_dict[key].shape
        print(f"  {key}: {shape}")

    print(f"\nOur model attention keys ({len(our_attn_keys)}):")
    for key in our_attn_keys:
        shape = our_state_dict[key].shape
        print(f"  {key}: {shape}")


if __name__ == "__main__":
    test_single_layer_no_tp()
    test_weight_name_mapping()
