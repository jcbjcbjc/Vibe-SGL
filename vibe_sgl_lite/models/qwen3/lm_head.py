"""
Qwen3 LM head (language modeling head) implementation.

This module implements the language modeling head that projects hidden states to logits
over the vocabulary. The LM head is a linear layer that maps from hidden_size to vocab_size.
"""

import torch
import torch.nn as nn

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


class Qwen3LMHead(nn.Module):
    """Qwen3 language modeling head.

    This class implements the final projection layer that converts hidden states
    to logits over the vocabulary. It's a standard linear layer without bias.

    Attributes:
        hidden_size: Dimension of the input hidden states.
        vocab_size: Size of the vocabulary (output dimension).
        weight: Projection weight matrix of shape [vocab_size, hidden_size].
    """

    def __init__(self, config: Qwen3Config) -> None:
        """Initialize Qwen3LMHead.

        Args:
            config: Qwen3 model configuration containing LM head parameters.
        """
        super().__init__()

        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size

        # Linear projection to vocabulary size (no bias)
        self.linear = nn.Linear(
            in_features=config.hidden_size,
            out_features=config.vocab_size,
            bias=False,
        )

    @property
    def weight(self) -> torch.Tensor:
        """Get LM head weight matrix.

        Returns:
            Weight tensor of shape [vocab_size, hidden_size].
        """
        return self.linear.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of LM head.

        Args:
            hidden_states: Hidden states tensor of shape [batch_size, seq_len, hidden_size].

        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size].
        """
        return self.linear(hidden_states)
