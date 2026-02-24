"""
Tests for TP-aware Qwen3FFN (MLP) layer.

This module tests:
- Qwen3FFN with tensor parallelism
- up_proj and gate_proj using ColumnParallelLinear
- down_proj using RowParallelLinear
- Correctness of TP FFN vs single-device reference
"""

import pytest
import torch
from torch.testing import assert_close

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN


@pytest.fixture
def small_config():
    """Create a small Qwen3 config for testing."""
    return Qwen3Config(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=1000,
        max_position_embeddings=512,
        rope_theta=10000.0,
    )


class TestTPQwen3FFN:
    """Tests for TP-aware Qwen3FFN."""

    def test_tp_ffn_initialization(self, small_config):
        """Test that TP FFN layer initializes correctly."""
        # Create TP FFN layer
        ffn = Qwen3FFN(small_config, tp_degree=2, rank=0)

        # Check that projections are using parallel linear layers
        from vibe_sgl_lite.distributed.tp.parallel_linear import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        assert isinstance(ffn.up_proj, ColumnParallelLinear)
        assert isinstance(ffn.gate_proj, ColumnParallelLinear)
        assert isinstance(ffn.down_proj, RowParallelLinear)

        # Check dimensions
        assert ffn.up_proj.out_features_per_partition == (
            small_config.intermediate_size // 2
        )
        assert ffn.gate_proj.out_features_per_partition == (
            small_config.intermediate_size // 2
        )
        assert ffn.down_proj.in_features_per_partition == (
            small_config.intermediate_size // 2
        )

    def test_tp_ffn_forward_shape(self, small_config):
        """Test that TP FFN forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 16

        # Create TP FFN layer
        ffn = Qwen3FFN(small_config, tp_degree=2, rank=0)

        # Input
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

        # Forward pass
        output = ffn(hidden_states)

        # Check output shape
        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_tp_ffn_vs_single_device(self, small_config):
        """Test that TP FFN output matches single-device reference."""
        batch_size = 2
        seq_len = 16

        # Create single-device reference FFN
        ffn_ref = Qwen3FFN(small_config, tp_degree=1, rank=0)

        # Create TP FFN layers for both ranks
        ffn_rank0 = Qwen3FFN(small_config, tp_degree=2, rank=0)
        ffn_rank1 = Qwen3FFN(small_config, tp_degree=2, rank=1)

        # Copy weights from reference to TP layers
        # up_proj (column parallel - split output dimension)
        ffn_rank0.up_proj.weight.data = ffn_ref.up_proj.weight.data[
            :ffn_rank0.up_proj.out_features_per_partition, :
        ]
        ffn_rank1.up_proj.weight.data = ffn_ref.up_proj.weight.data[
            ffn_rank0.up_proj.out_features_per_partition:, :
        ]

        # gate_proj (column parallel - split output dimension)
        ffn_rank0.gate_proj.weight.data = ffn_ref.gate_proj.weight.data[
            :ffn_rank0.gate_proj.out_features_per_partition, :
        ]
        ffn_rank1.gate_proj.weight.data = ffn_ref.gate_proj.weight.data[
            ffn_rank0.gate_proj.out_features_per_partition:, :
        ]

        # down_proj (row parallel - split input dimension)
        ffn_rank0.down_proj.weight.data = ffn_ref.down_proj.weight.data[
            :, :ffn_rank0.down_proj.in_features_per_partition
        ]
        ffn_rank1.down_proj.weight.data = ffn_ref.down_proj.weight.data[
            :, ffn_rank0.down_proj.in_features_per_partition:
        ]

        # Input
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

        # Reference forward pass
        with torch.no_grad():
            output_ref = ffn_ref(hidden_states)

        # TP forward pass (simulating distributed execution)
        with torch.no_grad():
            # Each rank computes its portion
            output_rank0 = ffn_rank0(hidden_states)
            output_rank1 = ffn_rank1(hidden_states)

            # Simulate all-reduce by summing outputs
            output_tp = output_rank0 + output_rank1

        # Check that TP output matches reference
        assert_close(output_tp, output_ref, atol=1e-4, rtol=1e-3)

    def test_tp_ffn_swiglu_correctness(self, small_config):
        """Test that SwiGLU activation works correctly with TP."""
        batch_size = 2
        seq_len = 8

        # Create TP FFN layer
        ffn = Qwen3FFN(small_config, tp_degree=2, rank=0)

        # Input
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

        # Forward pass should work without errors
        output = ffn(hidden_states)

        # Check output shape
        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

        # Check that output is not all zeros (activation is working)
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_tp_ffn_invalid_partition(self):
        """Test that initialization fails if intermediate_size not divisible by tp_degree."""
        config = Qwen3Config(
            hidden_size=64,
            num_attention_heads=8,
            num_key_value_heads=4,
            num_hidden_layers=2,
            intermediate_size=101,  # Not divisible by 4
            vocab_size=1000,
        )

        with pytest.raises(ValueError, match="Output features .* not evenly divisible"):
            Qwen3FFN(config, tp_degree=4, rank=0)

    def test_tp_ffn_different_batch_sizes(self, small_config):
        """Test that TP FFN works with different batch sizes."""
        ffn = Qwen3FFN(small_config, tp_degree=2, rank=0)

        for batch_size in [1, 2, 4, 8]:
            seq_len = 16
            hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
            output = ffn(hidden_states)
            assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_tp_ffn_different_seq_lengths(self, small_config):
        """Test that TP FFN works with different sequence lengths."""
        ffn = Qwen3FFN(small_config, tp_degree=2, rank=0)

        batch_size = 2
        for seq_len in [1, 8, 16, 32]:
            hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)
            output = ffn(hidden_states)
            assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_tp_ffn_gradient_flow(self, small_config):
        """Test that gradients flow correctly through TP FFN."""
        ffn = Qwen3FFN(small_config, tp_degree=2, rank=0)

        batch_size = 2
        seq_len = 8
        hidden_states = torch.randn(
            batch_size, seq_len, small_config.hidden_size, requires_grad=True
        )

        # Forward pass
        output = ffn(hidden_states)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Check that gradients exist
        assert hidden_states.grad is not None
        assert not torch.allclose(hidden_states.grad, torch.zeros_like(hidden_states.grad))

        # Check that all parameters have gradients
        for name, param in ffn.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"
