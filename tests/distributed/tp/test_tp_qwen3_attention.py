"""
Tests for TP-aware Qwen3Attention layer.

This module tests:
- Qwen3Attention with tensor parallelism
- Q/K/V projections using ColumnParallelLinear
- Output projection using RowParallelLinear
- Correctness of TP attention vs single-device reference
"""

import pytest
import torch
from torch.testing import assert_close

from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention
from vibe_sgl_lite.models.qwen3.rope import precompute_freqs_cis


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


@pytest.fixture
def freqs_cis(small_config):
    """Precompute RoPE frequencies for testing."""
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    return precompute_freqs_cis(
        dim=head_dim,
        end=small_config.max_position_embeddings,
        theta=small_config.rope_theta,
    )


class TestTPQwen3Attention:
    """Tests for TP-aware Qwen3Attention."""

    def test_tp_attention_initialization(self, small_config):
        """Test that TP attention layer initializes correctly."""
        # Create TP attention layer
        attention = Qwen3Attention(small_config, tp_degree=2, rank=0)

        # Check that projections are using parallel linear layers
        from vibe_sgl_lite.distributed.tp.parallel_linear import (
            ColumnParallelLinear,
            RowParallelLinear,
        )

        assert isinstance(attention.q_proj, ColumnParallelLinear)
        assert isinstance(attention.k_proj, ColumnParallelLinear)
        assert isinstance(attention.v_proj, ColumnParallelLinear)
        assert isinstance(attention.o_proj, RowParallelLinear)

        # Check dimensions
        assert attention.q_proj.out_features_per_partition == (
            small_config.num_attention_heads * attention.head_dim // 2
        )
        assert attention.k_proj.out_features_per_partition == (
            small_config.num_key_value_heads * attention.head_dim // 2
        )
        assert attention.v_proj.out_features_per_partition == (
            small_config.num_key_value_heads * attention.head_dim // 2
        )

    def test_tp_attention_forward_shape(self, small_config, freqs_cis):
        """Test that TP attention forward pass produces correct output shape."""
        batch_size = 2
        seq_len = 16

        # Create TP attention layer
        attention = Qwen3Attention(small_config, tp_degree=2, rank=0)

        # Input
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

        # Forward pass
        output = attention(hidden_states, freqs_cis)

        # Check output shape
        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

    def test_tp_attention_vs_single_device(self, small_config, freqs_cis):
        """Test that TP attention output matches single-device reference."""
        batch_size = 2
        seq_len = 16

        # Create single-device reference attention
        attention_ref = Qwen3Attention(small_config, tp_degree=1, rank=0)

        # Create TP attention layers for both ranks
        attention_rank0 = Qwen3Attention(small_config, tp_degree=2, rank=0)
        attention_rank1 = Qwen3Attention(small_config, tp_degree=2, rank=1)

        # Copy weights from reference to TP layers
        # Q projection
        attention_rank0.q_proj.weight.data = attention_ref.q_proj.weight.data[
            :attention_rank0.q_proj.out_features_per_partition, :
        ]
        attention_rank1.q_proj.weight.data = attention_ref.q_proj.weight.data[
            attention_rank0.q_proj.out_features_per_partition:, :
        ]

        # K projection
        attention_rank0.k_proj.weight.data = attention_ref.k_proj.weight.data[
            :attention_rank0.k_proj.out_features_per_partition, :
        ]
        attention_rank1.k_proj.weight.data = attention_ref.k_proj.weight.data[
            attention_rank0.k_proj.out_features_per_partition:, :
        ]

        # V projection
        attention_rank0.v_proj.weight.data = attention_ref.v_proj.weight.data[
            :attention_rank0.v_proj.out_features_per_partition, :
        ]
        attention_rank1.v_proj.weight.data = attention_ref.v_proj.weight.data[
            attention_rank0.v_proj.out_features_per_partition:, :
        ]

        # O projection (row parallel - split input dimension)
        attention_rank0.o_proj.weight.data = attention_ref.o_proj.weight.data[
            :, :attention_rank0.o_proj.in_features_per_partition
        ]
        attention_rank1.o_proj.weight.data = attention_ref.o_proj.weight.data[
            :, attention_rank0.o_proj.in_features_per_partition:
        ]

        # Input
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

        # Reference forward pass
        with torch.no_grad():
            output_ref = attention_ref(hidden_states, freqs_cis)

        # TP forward pass (simulating distributed execution)
        with torch.no_grad():
            # Each rank computes its portion
            output_rank0 = attention_rank0(hidden_states, freqs_cis)
            output_rank1 = attention_rank1(hidden_states, freqs_cis)

            # Simulate all-reduce by summing outputs
            output_tp = output_rank0 + output_rank1

        # Check that TP output matches reference
        assert_close(output_tp, output_ref, atol=1e-4, rtol=1e-3)

    def test_tp_attention_with_kv_cache(self, small_config, freqs_cis):
        """Test that TP attention works correctly with KV cache."""
        batch_size = 2
        prefill_len = 8
        decode_len = 1

        # Create TP attention layer
        attention = Qwen3Attention(small_config, tp_degree=2, rank=0)

        # Prefill phase
        prefill_input = torch.randn(batch_size, prefill_len, small_config.hidden_size)
        output_prefill, kv_cache = attention(
            prefill_input, freqs_cis, return_kv_cache=True
        )

        # Check prefill output shape
        assert output_prefill.shape == (batch_size, prefill_len, small_config.hidden_size)

        # Check KV cache shapes
        k_cache, v_cache = kv_cache
        # Note: num_key_value_heads is partitioned across TP ranks
        expected_kv_heads = small_config.num_key_value_heads // 2
        assert k_cache.shape == (
            batch_size,
            expected_kv_heads,
            prefill_len,
            attention.head_dim,
        )
        assert v_cache.shape == (
            batch_size,
            expected_kv_heads,
            prefill_len,
            attention.head_dim,
        )

        # Decode phase
        decode_input = torch.randn(batch_size, decode_len, small_config.hidden_size)
        output_decode, kv_cache_updated = attention(
            decode_input,
            freqs_cis,
            position_offset=prefill_len,
            kv_cache=kv_cache,
            return_kv_cache=True,
        )

        # Check decode output shape
        assert output_decode.shape == (batch_size, decode_len, small_config.hidden_size)

        # Check updated KV cache shapes
        k_cache_updated, v_cache_updated = kv_cache_updated
        assert k_cache_updated.shape == (
            batch_size,
            expected_kv_heads,
            prefill_len + decode_len,
            attention.head_dim,
        )
        assert v_cache_updated.shape == (
            batch_size,
            expected_kv_heads,
            prefill_len + decode_len,
            attention.head_dim,
        )

    def test_tp_attention_head_partitioning(self, small_config):
        """Test that attention heads are correctly partitioned across ranks."""
        # Create TP attention layers
        attention_rank0 = Qwen3Attention(small_config, tp_degree=2, rank=0)
        attention_rank1 = Qwen3Attention(small_config, tp_degree=2, rank=1)

        # Check that each rank has half the heads
        assert attention_rank0.num_heads_per_partition == small_config.num_attention_heads // 2
        assert attention_rank1.num_heads_per_partition == small_config.num_attention_heads // 2

        # Check KV heads partitioning
        assert attention_rank0.num_kv_heads_per_partition == small_config.num_key_value_heads // 2
        assert attention_rank1.num_kv_heads_per_partition == small_config.num_key_value_heads // 2

    def test_tp_attention_invalid_partition(self):
        """Test that initialization fails if heads not divisible by tp_degree."""
        config = Qwen3Config(
            hidden_size=96,  # Divisible by 6
            num_attention_heads=6,  # Not divisible by 4
            num_key_value_heads=2,
            num_hidden_layers=2,
            intermediate_size=128,
            vocab_size=1000,
        )

        with pytest.raises(ValueError, match="not evenly divisible"):
            Qwen3Attention(config, tp_degree=4, rank=0)

    def test_tp_attention_gqa_correctness(self, small_config, freqs_cis):
        """Test that GQA works correctly with TP."""
        batch_size = 2
        seq_len = 8

        # Ensure GQA is being used (num_q_heads > num_kv_heads)
        assert small_config.num_attention_heads > small_config.num_key_value_heads

        # Create TP attention layer
        attention = Qwen3Attention(small_config, tp_degree=2, rank=0)

        # Input
        hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

        # Forward pass should work without errors
        output = attention(hidden_states, freqs_cis)

        # Check output shape
        assert output.shape == (batch_size, seq_len, small_config.hidden_size)

        # Check that GQA grouping is correct
        assert attention.num_key_value_groups == (
            attention.num_heads_per_partition // attention.num_kv_heads_per_partition
        )
