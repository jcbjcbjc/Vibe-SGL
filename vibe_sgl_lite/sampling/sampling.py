"""
Sampling strategies for text generation.

This module implements various sampling methods for token generation.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class SamplingParams:
    """Parameters for sampling strategies."""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    logit_bias: Optional[dict] = None
    stop_sequences: Optional[List[str]] = None
    seed: Optional[int] = None


def greedy_sampling(logits: torch.Tensor) -> torch.Tensor:
    """Greedy sampling (argmax)."""
    return logits.argmax(dim=-1)


def temperature_scaling(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling."""
    return logits / temperature


def top_k_sampling(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k sampling."""
    if k <= 0:
        return logits

    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Create mask for top-k
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, top_k_indices, top_k_logits)

    return mask


def top_p_sampling(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Top-p (nucleus) sampling."""
    if p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Create mask
    mask = torch.full_like(logits, float('-inf'))
    mask.scatter_(-1, sorted_indices, sorted_logits)
    mask[sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)] = float('-inf')

    return mask


def apply_frequency_penalty(logits: torch.Tensor, token_counts: torch.Tensor, penalty: float) -> torch.Tensor:
    """Apply frequency penalty."""
    if penalty == 0.0:
        return logits
    return logits - penalty * token_counts


def apply_presence_penalty(logits: torch.Tensor, token_presence: torch.Tensor, penalty: float) -> torch.Tensor:
    """Apply presence penalty."""
    if penalty == 0.0:
        return logits
    return logits - penalty * token_presence


def apply_repetition_penalty(logits: torch.Tensor, previous_tokens: torch.Tensor, penalty: float) -> torch.Tensor:
    """Apply repetition penalty."""
    if penalty == 1.0:
        return logits

    for token in previous_tokens:
        if logits[token] > 0:
            logits[token] /= penalty
        else:
            logits[token] *= penalty

    return logits


def sample(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Sample next token using specified parameters."""
    # Apply temperature
    if params.temperature != 1.0:
        logits = temperature_scaling(logits, params.temperature)

    # Apply top-k
    if params.top_k > 0:
        logits = top_k_sampling(logits, params.top_k)

    # Apply top-p
    if params.top_p < 1.0:
        logits = top_p_sampling(logits, params.top_p)

    # Sample from distribution
    if params.temperature == 0.0:
        return greedy_sampling(logits)

    probs = torch.softmax(logits, dim=-1)

    # Set seed if specified
    if params.seed is not None:
        torch.manual_seed(params.seed)

    return torch.multinomial(probs, num_samples=1).squeeze(-1)
