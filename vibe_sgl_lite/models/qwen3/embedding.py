"""
Qwen3 token embedding layer implementation.

This module implements the token embedding layer that converts token IDs to dense vectors.
The embedding layer is a standard nn.Embedding that maps vocabulary indices to hidden states.
"""

import torch
import torch.nn as nn

from vibe_sgl_lite.models.qwen3.config import Qwen3Config


class Qwen3Embedding(nn.Module):
    """Qwen3 token embedding layer.

    This class implements a standard token embedding layer that converts token IDs
    to dense vector representations. It's a wrapper around torch.nn.Embedding with
    dimensions specified by the Qwen3 configuration.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the embedding vectors.
        weight: Embedding weight matrix of shape [vocab_size, hidden_size].
    """

    def __init__(self, config: Qwen3Config) -> None:
        """Initialize Qwen3Embedding.

        Args:
            config: Qwen3 model configuration containing embedding parameters.
        """
        super().__init__()

        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size

        # Standard embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

    @property
    def weight(self) -> torch.Tensor:
        """Get embedding weight matrix.

        Returns:
            Embedding weight tensor of shape [vocab_size, hidden_size].
        """
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass of embedding layer.

        Args:
            input_ids: Token IDs tensor of shape [batch_size, seq_len].

        Returns:
            Embedding tensor of shape [batch_size, seq_len, hidden_size].
        """
        return self.embedding(input_ids)
