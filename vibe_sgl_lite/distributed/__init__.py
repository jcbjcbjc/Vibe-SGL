"""
Distributed parallelism implementations using torch.distributed.

Provides:
- Tensor Parallelism (TP): Column/row parallel layers
- Expert Parallelism (EP): Expert routing and placement
- Hybrid parallelism: TP+EP coordination
- Backend selection: Gloo (CPU) and NCCL (GPU)
"""

__all__ = []
