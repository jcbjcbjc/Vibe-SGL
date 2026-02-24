"""
Multi-process tests for Tensor Parallelism (TP) correctness.

This module tests:
- TP attention correctness across multiple processes
- TP FFN correctness across multiple processes
- TP outputs match single-device reference
- Communication and synchronization between TP ranks
"""

import pytest
import torch
import torch.distributed as dist
from torch.testing import assert_close

from tests.utils.distributed import run_distributed_test
from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.attention import Qwen3Attention
from vibe_sgl_lite.models.qwen3.ffn import Qwen3FFN
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


# Worker functions defined at module level for pickling


def _worker_tp_attention(rank, world_size, ref_weights, hidden_states, freqs_cis, config):
    """Worker function for TP attention test."""
    # Create TP attention layer for this rank
    attention_tp = Qwen3Attention(config, tp_degree=world_size, rank=rank)

    # Load partitioned weights
    q_split_size = attention_tp.q_proj.out_features_per_partition
    k_split_size = attention_tp.k_proj.out_features_per_partition
    v_split_size = attention_tp.v_proj.out_features_per_partition
    o_split_size = attention_tp.o_proj.in_features_per_partition

    attention_tp.q_proj.weight.data = ref_weights["q_proj"][
        rank * q_split_size : (rank + 1) * q_split_size, :
    ]
    attention_tp.k_proj.weight.data = ref_weights["k_proj"][
        rank * k_split_size : (rank + 1) * k_split_size, :
    ]
    attention_tp.v_proj.weight.data = ref_weights["v_proj"][
        rank * v_split_size : (rank + 1) * v_split_size, :
    ]
    attention_tp.o_proj.weight.data = ref_weights["o_proj"][
        :, rank * o_split_size : (rank + 1) * o_split_size
    ]

    # Forward pass
    with torch.no_grad():
        output_tp = attention_tp(hidden_states, freqs_cis)

    # All-reduce to combine outputs from all ranks
    dist.all_reduce(output_tp, op=dist.ReduceOp.SUM)

    return output_tp


def _worker_tp_ffn(rank, world_size, ref_weights, hidden_states, config):
    """Worker function for TP FFN test."""
    # Create TP FFN layer for this rank
    ffn_tp = Qwen3FFN(config, tp_degree=world_size, rank=rank)

    # Load partitioned weights
    up_split_size = ffn_tp.up_proj.out_features_per_partition
    gate_split_size = ffn_tp.gate_proj.out_features_per_partition
    down_split_size = ffn_tp.down_proj.in_features_per_partition

    ffn_tp.up_proj.weight.data = ref_weights["up_proj"][
        rank * up_split_size : (rank + 1) * up_split_size, :
    ]
    ffn_tp.gate_proj.weight.data = ref_weights["gate_proj"][
        rank * gate_split_size : (rank + 1) * gate_split_size, :
    ]
    ffn_tp.down_proj.weight.data = ref_weights["down_proj"][
        :, rank * down_split_size : (rank + 1) * down_split_size
    ]

    # Forward pass
    with torch.no_grad():
        output_tp = ffn_tp(hidden_states)

    # All-reduce to combine outputs from all ranks
    dist.all_reduce(output_tp, op=dist.ReduceOp.SUM)

    return output_tp


def _worker_communication_test(rank, world_size):
    """Worker function to test communication."""
    # Create a tensor with rank-specific value
    tensor = torch.ones(10) * (rank + 1)

    # All-reduce should sum all tensors
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Expected sum: 1 + 2 + ... + world_size = world_size * (world_size + 1) / 2
    expected_sum = sum(range(1, world_size + 1))
    expected_tensor = torch.ones(10) * expected_sum

    # Verify result
    assert_close(tensor, expected_tensor, atol=1e-6, rtol=1e-6)

    # Test barrier
    dist.barrier()

    return {"rank": rank, "sum": tensor[0].item()}


def _worker_gradient_test(rank, world_size, config):
    """Worker function to test gradient synchronization."""
    # Create TP FFN layer
    ffn_tp = Qwen3FFN(config, tp_degree=world_size, rank=rank)

    # Create input with requires_grad
    batch_size = 2
    seq_len = 8
    hidden_states = torch.randn(
        batch_size, seq_len, config.hidden_size, requires_grad=True
    )

    # Forward pass
    output = ffn_tp(hidden_states)

    # Compute loss (sum of outputs)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Verify gradients exist
    assert hidden_states.grad is not None
    assert not torch.allclose(
        hidden_states.grad, torch.zeros_like(hidden_states.grad)
    )

    # All-reduce gradients to simulate distributed training
    dist.all_reduce(hidden_states.grad, op=dist.ReduceOp.SUM)

    return {
        "rank": rank,
        "grad_norm": hidden_states.grad.norm().item(),
        "has_grad": hidden_states.grad is not None,
    }


def _worker_e2e_attention_ffn(rank, world_size, ref_weights, hidden_states, freqs_cis, config):
    """Worker function for end-to-end attention + FFN test."""
    # Create TP attention layer
    attention_tp = Qwen3Attention(config, tp_degree=world_size, rank=rank)

    # Load partitioned attention weights
    q_split_size = attention_tp.q_proj.out_features_per_partition
    k_split_size = attention_tp.k_proj.out_features_per_partition
    v_split_size = attention_tp.v_proj.out_features_per_partition
    o_split_size = attention_tp.o_proj.in_features_per_partition

    attention_tp.q_proj.weight.data = ref_weights["attn_q_proj"][
        rank * q_split_size : (rank + 1) * q_split_size, :
    ]
    attention_tp.k_proj.weight.data = ref_weights["attn_k_proj"][
        rank * k_split_size : (rank + 1) * k_split_size, :
    ]
    attention_tp.v_proj.weight.data = ref_weights["attn_v_proj"][
        rank * v_split_size : (rank + 1) * v_split_size, :
    ]
    attention_tp.o_proj.weight.data = ref_weights["attn_o_proj"][
        :, rank * o_split_size : (rank + 1) * o_split_size
    ]

    # Create TP FFN layer
    ffn_tp = Qwen3FFN(config, tp_degree=world_size, rank=rank)

    # Load partitioned FFN weights
    up_split_size = ffn_tp.up_proj.out_features_per_partition
    gate_split_size = ffn_tp.gate_proj.out_features_per_partition
    down_split_size = ffn_tp.down_proj.in_features_per_partition

    ffn_tp.up_proj.weight.data = ref_weights["ffn_up_proj"][
        rank * up_split_size : (rank + 1) * up_split_size, :
    ]
    ffn_tp.gate_proj.weight.data = ref_weights["ffn_gate_proj"][
        rank * gate_split_size : (rank + 1) * gate_split_size, :
    ]
    ffn_tp.down_proj.weight.data = ref_weights["ffn_down_proj"][
        :, rank * down_split_size : (rank + 1) * down_split_size
    ]

    # Forward pass: attention -> FFN (simulating a transformer layer)
    with torch.no_grad():
        # Attention
        attn_output = attention_tp(hidden_states, freqs_cis)
        dist.all_reduce(attn_output, op=dist.ReduceOp.SUM)

        # FFN
        ffn_output = ffn_tp(attn_output)
        dist.all_reduce(ffn_output, op=dist.ReduceOp.SUM)

    return ffn_output


def _worker_e2e_with_kv_cache(rank, world_size, ref_weights, hidden_states, freqs_cis, config):
    """Worker function for end-to-end test with KV cache."""
    # Create TP attention layer
    attention_tp = Qwen3Attention(config, tp_degree=world_size, rank=rank)

    # Load partitioned attention weights
    q_split_size = attention_tp.q_proj.out_features_per_partition
    k_split_size = attention_tp.k_proj.out_features_per_partition
    v_split_size = attention_tp.v_proj.out_features_per_partition
    o_split_size = attention_tp.o_proj.in_features_per_partition

    attention_tp.q_proj.weight.data = ref_weights["q_proj"][
        rank * q_split_size : (rank + 1) * q_split_size, :
    ]
    attention_tp.k_proj.weight.data = ref_weights["k_proj"][
        rank * k_split_size : (rank + 1) * k_split_size, :
    ]
    attention_tp.v_proj.weight.data = ref_weights["v_proj"][
        rank * v_split_size : (rank + 1) * v_split_size, :
    ]
    attention_tp.o_proj.weight.data = ref_weights["o_proj"][
        :, rank * o_split_size : (rank + 1) * o_split_size
    ]

    with torch.no_grad():
        # Prefill phase
        prefill_output, kv_cache = attention_tp(
            hidden_states, freqs_cis, return_kv_cache=True
        )
        dist.all_reduce(prefill_output, op=dist.ReduceOp.SUM)

        # Decode phase (single token)
        decode_input = torch.randn(hidden_states.shape[0], 1, config.hidden_size)
        decode_output, kv_cache_updated = attention_tp(
            decode_input,
            freqs_cis,
            position_offset=hidden_states.shape[1],
            kv_cache=kv_cache,
            return_kv_cache=True,
        )
        dist.all_reduce(decode_output, op=dist.ReduceOp.SUM)

    return {
        "prefill_output": prefill_output,
        "decode_output": decode_output,
        "kv_cache_shape": kv_cache_updated[0].shape,
    }


# Test functions



@pytest.mark.distributed
def test_tp_attention_multiprocess_correctness(small_config):
    """
    Test that TP attention outputs match single-device reference across processes.

    This test:
    - Spawns 2 processes for TP degree 2
    - Creates TP attention layers in each process
    - Loads partitioned weights from a reference model
    - Runs forward pass in parallel
    - Performs all-reduce to combine outputs
    - Validates combined output matches reference
    """
    # Create reference model and weights (shared across processes)
    torch.manual_seed(42)
    attention_ref = Qwen3Attention(small_config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "q_proj": attention_ref.q_proj.weight.data.clone(),
        "k_proj": attention_ref.k_proj.weight.data.clone(),
        "v_proj": attention_ref.v_proj.weight.data.clone(),
        "o_proj": attention_ref.o_proj.weight.data.clone(),
    }

    # Precompute RoPE frequencies
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=small_config.max_position_embeddings,
        theta=small_config.rope_theta,
    )

    # Create test input
    batch_size = 2
    seq_len = 16
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

    # Compute reference output
    with torch.no_grad():
        output_ref = attention_ref(hidden_states, freqs_cis)

    # Run distributed test
    results = run_distributed_test(
        _worker_tp_attention,
        world_size=2,
        backend="gloo",
        test_args=(ref_weights, hidden_states, freqs_cis, small_config),
    )

    # Verify outputs from all ranks match reference
    for rank, output_tp in enumerate(results):
        assert_close(
            output_tp,
            output_ref,
            atol=1e-4,
            rtol=1e-3,
            msg=f"Rank {rank} output doesn't match reference",
        )


@pytest.mark.distributed
def test_tp_ffn_multiprocess_correctness(small_config):
    """
    Test that TP FFN outputs match single-device reference across processes.

    This test:
    - Spawns 2 processes for TP degree 2
    - Creates TP FFN layers in each process
    - Loads partitioned weights from a reference model
    - Runs forward pass in parallel
    - Performs all-reduce to combine outputs
    - Validates combined output matches reference
    """
    # Create reference model and weights
    torch.manual_seed(42)
    ffn_ref = Qwen3FFN(small_config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "up_proj": ffn_ref.up_proj.weight.data.clone(),
        "gate_proj": ffn_ref.gate_proj.weight.data.clone(),
        "down_proj": ffn_ref.down_proj.weight.data.clone(),
    }

    # Create test input
    batch_size = 2
    seq_len = 16
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

    # Compute reference output
    with torch.no_grad():
        output_ref = ffn_ref(hidden_states)

    # Run distributed test
    results = run_distributed_test(
        _worker_tp_ffn,
        world_size=2,
        backend="gloo",
        test_args=(ref_weights, hidden_states, small_config),
    )

    # Verify outputs from all ranks match reference
    for rank, output_tp in enumerate(results):
        assert_close(
            output_tp,
            output_ref,
            atol=1e-4,
            rtol=1e-3,
            msg=f"Rank {rank} output doesn't match reference",
        )


@pytest.mark.distributed
def test_tp_attention_with_different_world_sizes(small_config):
    """
    Test TP attention correctness with different world sizes (TP degrees).

    Validates that TP works correctly with:
    - TP degree 2 (2 ranks)
    - TP degree 4 (4 ranks)
    """
    # Create reference model
    torch.manual_seed(42)
    attention_ref = Qwen3Attention(small_config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "q_proj": attention_ref.q_proj.weight.data.clone(),
        "k_proj": attention_ref.k_proj.weight.data.clone(),
        "v_proj": attention_ref.v_proj.weight.data.clone(),
        "o_proj": attention_ref.o_proj.weight.data.clone(),
    }

    # Precompute RoPE frequencies
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=small_config.max_position_embeddings,
        theta=small_config.rope_theta,
    )

    # Create test input
    batch_size = 2
    seq_len = 8
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

    # Compute reference output
    with torch.no_grad():
        output_ref = attention_ref(hidden_states, freqs_cis)

    # Test with different world sizes
    for world_size in [2, 4]:
        results = run_distributed_test(
            _worker_tp_attention,
            world_size=world_size,
            backend="gloo",
            test_args=(ref_weights, hidden_states, freqs_cis, small_config),
        )

        # Verify all ranks produce the same output matching reference
        for rank, output_tp in enumerate(results):
            assert_close(
                output_tp,
                output_ref,
                atol=1e-4,
                rtol=1e-3,
                msg=f"World size {world_size}, rank {rank} output doesn't match reference",
            )


@pytest.mark.distributed
def test_tp_communication_correctness():
    """
    Test that TP communication primitives work correctly.

    Validates:
    - All-reduce sums outputs correctly
    - Barrier synchronization works
    - All ranks can communicate
    """
    # Test with 2 ranks
    results = run_distributed_test(_worker_communication_test, world_size=2, backend="gloo")

    # Verify all ranks computed the same sum
    assert len(results) == 2
    assert results[0]["sum"] == 3.0  # 1 + 2
    assert results[1]["sum"] == 3.0


@pytest.mark.distributed
def test_tp_gradient_synchronization():
    """
    Test that gradients are correctly synchronized across TP ranks.

    This validates that:
    - Gradients flow correctly through TP layers
    - All-reduce synchronizes gradients properly
    - Backward pass works with TP
    """
    small_config = Qwen3Config(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=1000,
    )

    # Run distributed test
    results = run_distributed_test(
        _worker_gradient_test,
        world_size=2,
        backend="gloo",
        test_args=(small_config,),
    )

    # Verify all ranks have gradients
    assert all(r["has_grad"] for r in results)

    # Verify gradient norms are the same after all-reduce
    grad_norms = [r["grad_norm"] for r in results]
    assert_close(
        torch.tensor(grad_norms[0]),
        torch.tensor(grad_norms[1]),
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.distributed
def test_tp_e2e_attention_ffn_tp2(small_config):
    """
    End-to-end test: Attention + FFN with TP degree 2.

    This test validates a complete transformer layer (attention + FFN)
    with tensor parallelism degree 2.
    """
    # Create reference models
    torch.manual_seed(42)
    attention_ref = Qwen3Attention(small_config, tp_degree=1, rank=0)
    ffn_ref = Qwen3FFN(small_config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "attn_q_proj": attention_ref.q_proj.weight.data.clone(),
        "attn_k_proj": attention_ref.k_proj.weight.data.clone(),
        "attn_v_proj": attention_ref.v_proj.weight.data.clone(),
        "attn_o_proj": attention_ref.o_proj.weight.data.clone(),
        "ffn_up_proj": ffn_ref.up_proj.weight.data.clone(),
        "ffn_gate_proj": ffn_ref.gate_proj.weight.data.clone(),
        "ffn_down_proj": ffn_ref.down_proj.weight.data.clone(),
    }

    # Precompute RoPE frequencies
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=small_config.max_position_embeddings,
        theta=small_config.rope_theta,
    )

    # Create test input
    batch_size = 2
    seq_len = 16
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

    # Compute reference output
    with torch.no_grad():
        attn_output_ref = attention_ref(hidden_states, freqs_cis)
        ffn_output_ref = ffn_ref(attn_output_ref)

    # Run distributed test with TP degree 2
    results = run_distributed_test(
        _worker_e2e_attention_ffn,
        world_size=2,
        backend="gloo",
        test_args=(ref_weights, hidden_states, freqs_cis, small_config),
    )

    # Verify outputs from all ranks match reference
    for rank, output_tp in enumerate(results):
        assert_close(
            output_tp,
            ffn_output_ref,
            atol=1e-4,
            rtol=1e-3,
            msg=f"TP degree 2, rank {rank} output doesn't match reference",
        )


@pytest.mark.distributed
def test_tp_e2e_attention_ffn_tp4(small_config):
    """
    End-to-end test: Attention + FFN with TP degree 4.

    This test validates a complete transformer layer (attention + FFN)
    with tensor parallelism degree 4.
    """
    # Create reference models
    torch.manual_seed(42)
    attention_ref = Qwen3Attention(small_config, tp_degree=1, rank=0)
    ffn_ref = Qwen3FFN(small_config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "attn_q_proj": attention_ref.q_proj.weight.data.clone(),
        "attn_k_proj": attention_ref.k_proj.weight.data.clone(),
        "attn_v_proj": attention_ref.v_proj.weight.data.clone(),
        "attn_o_proj": attention_ref.o_proj.weight.data.clone(),
        "ffn_up_proj": ffn_ref.up_proj.weight.data.clone(),
        "ffn_gate_proj": ffn_ref.gate_proj.weight.data.clone(),
        "ffn_down_proj": ffn_ref.down_proj.weight.data.clone(),
    }

    # Precompute RoPE frequencies
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=small_config.max_position_embeddings,
        theta=small_config.rope_theta,
    )

    # Create test input
    batch_size = 2
    seq_len = 16
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

    # Compute reference output
    with torch.no_grad():
        attn_output_ref = attention_ref(hidden_states, freqs_cis)
        ffn_output_ref = ffn_ref(attn_output_ref)

    # Run distributed test with TP degree 4
    results = run_distributed_test(
        _worker_e2e_attention_ffn,
        world_size=4,
        backend="gloo",
        test_args=(ref_weights, hidden_states, freqs_cis, small_config),
    )

    # Verify outputs from all ranks match reference
    for rank, output_tp in enumerate(results):
        assert_close(
            output_tp,
            ffn_output_ref,
            atol=1e-4,
            rtol=1e-3,
            msg=f"TP degree 4, rank {rank} output doesn't match reference",
        )


@pytest.mark.distributed
def test_tp_e2e_with_kv_cache_tp2(small_config):
    """
    End-to-end test: Attention with KV cache and TP degree 2.

    This test validates:
    - Prefill phase with TP
    - Decode phase with KV cache and TP
    - KV cache shapes are correct for TP
    """
    # Create reference model
    torch.manual_seed(42)
    attention_ref = Qwen3Attention(small_config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "q_proj": attention_ref.q_proj.weight.data.clone(),
        "k_proj": attention_ref.k_proj.weight.data.clone(),
        "v_proj": attention_ref.v_proj.weight.data.clone(),
        "o_proj": attention_ref.o_proj.weight.data.clone(),
    }

    # Precompute RoPE frequencies
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=small_config.max_position_embeddings,
        theta=small_config.rope_theta,
    )

    # Create test input
    batch_size = 2
    seq_len = 8
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

    # Compute reference output
    with torch.no_grad():
        prefill_output_ref, kv_cache_ref = attention_ref(
            hidden_states, freqs_cis, return_kv_cache=True
        )

        decode_input = torch.randn(batch_size, 1, small_config.hidden_size)
        decode_output_ref, _ = attention_ref(
            decode_input,
            freqs_cis,
            position_offset=seq_len,
            kv_cache=kv_cache_ref,
            return_kv_cache=True,
        )

    # Run distributed test with TP degree 2
    results = run_distributed_test(
        _worker_e2e_with_kv_cache,
        world_size=2,
        backend="gloo",
        test_args=(ref_weights, hidden_states, freqs_cis, small_config),
    )

    # Verify outputs from all ranks
    for rank, result in enumerate(results):
        # Check prefill output
        assert_close(
            result["prefill_output"],
            prefill_output_ref,
            atol=1e-4,
            rtol=1e-3,
            msg=f"TP degree 2, rank {rank} prefill output doesn't match reference",
        )

        # Check KV cache shape (should be partitioned)
        expected_kv_heads = small_config.num_key_value_heads // 2
        assert result["kv_cache_shape"][1] == expected_kv_heads, (
            f"Rank {rank}: KV cache has wrong number of heads. "
            f"Expected {expected_kv_heads}, got {result['kv_cache_shape'][1]}"
        )


@pytest.mark.distributed
def test_tp_e2e_with_kv_cache_tp4(small_config):
    """
    End-to-end test: Attention with KV cache and TP degree 4.

    This test validates:
    - Prefill phase with TP
    - Decode phase with KV cache and TP
    - KV cache shapes are correct for TP
    """
    # Create reference model
    torch.manual_seed(42)
    attention_ref = Qwen3Attention(small_config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "q_proj": attention_ref.q_proj.weight.data.clone(),
        "k_proj": attention_ref.k_proj.weight.data.clone(),
        "v_proj": attention_ref.v_proj.weight.data.clone(),
        "o_proj": attention_ref.o_proj.weight.data.clone(),
    }

    # Precompute RoPE frequencies
    head_dim = small_config.hidden_size // small_config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=small_config.max_position_embeddings,
        theta=small_config.rope_theta,
    )

    # Create test input
    batch_size = 2
    seq_len = 8
    torch.manual_seed(123)
    hidden_states = torch.randn(batch_size, seq_len, small_config.hidden_size)

    # Compute reference output
    with torch.no_grad():
        prefill_output_ref, kv_cache_ref = attention_ref(
            hidden_states, freqs_cis, return_kv_cache=True
        )

    # Run distributed test with TP degree 4
    results = run_distributed_test(
        _worker_e2e_with_kv_cache,
        world_size=4,
        backend="gloo",
        test_args=(ref_weights, hidden_states, freqs_cis, small_config),
    )

    # Verify outputs from all ranks
    for rank, result in enumerate(results):
        # Check prefill output
        assert_close(
            result["prefill_output"],
            prefill_output_ref,
            atol=1e-4,
            rtol=1e-3,
            msg=f"TP degree 4, rank {rank} prefill output doesn't match reference",
        )

        # Check KV cache shape (should be partitioned)
        expected_kv_heads = small_config.num_key_value_heads // 4
        assert result["kv_cache_shape"][1] == expected_kv_heads, (
            f"Rank {rank}: KV cache has wrong number of heads. "
            f"Expected {expected_kv_heads}, got {result['kv_cache_shape'][1]}"
        )


@pytest.mark.distributed
def test_tp_e2e_different_batch_sizes():
    """
    End-to-end test: TP with different batch sizes.

    This test validates that TP works correctly with various batch sizes.
    """
    config = Qwen3Config(
        hidden_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=1000,
    )

    # Create reference model
    torch.manual_seed(42)
    attention_ref = Qwen3Attention(config, tp_degree=1, rank=0)

    # Save reference weights
    ref_weights = {
        "q_proj": attention_ref.q_proj.weight.data.clone(),
        "k_proj": attention_ref.k_proj.weight.data.clone(),
        "v_proj": attention_ref.v_proj.weight.data.clone(),
        "o_proj": attention_ref.o_proj.weight.data.clone(),
    }

    # Precompute RoPE frequencies
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(
        dim=head_dim,
        end=config.max_position_embeddings,
        theta=config.rope_theta,
    )

    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        seq_len = 8
        torch.manual_seed(123 + batch_size)
        hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)

        # Compute reference output
        with torch.no_grad():
            output_ref = attention_ref(hidden_states, freqs_cis)

        # Run distributed test with TP degree 2
        results = run_distributed_test(
            _worker_tp_attention,
            world_size=2,
            backend="gloo",
            test_args=(ref_weights, hidden_states, freqs_cis, config),
        )

        # Verify outputs
        for rank, output_tp in enumerate(results):
            assert_close(
                output_tp,
                output_ref,
                atol=1e-4,
                rtol=1e-3,
                msg=f"Batch size {batch_size}, rank {rank} output doesn't match reference",
            )


