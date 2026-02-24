"""
Qwen3 model configuration.

This module defines the Qwen3Config class which stores all configuration
parameters for the Qwen3 model architecture, including dimensions, layer counts,
attention parameters, and normalization settings.
"""

from typing import Any, Dict, Optional

from transformers import AutoConfig


class Qwen3Config:
    """Configuration class for Qwen3 model.

    This class stores all hyperparameters needed to instantiate a Qwen3 model,
    including architecture dimensions, attention configuration, and normalization
    parameters. It can load configurations from HuggingFace checkpoints.

    Attributes:
        vocab_size: Size of the vocabulary.
        hidden_size: Dimension of the hidden representations.
        num_hidden_layers: Number of transformer decoder layers.
        num_attention_heads: Number of attention heads for queries.
        num_key_value_heads: Number of attention heads for keys/values (GQA).
        intermediate_size: Dimension of the FFN intermediate layer.
        max_position_embeddings: Maximum sequence length supported.
        rms_norm_eps: Epsilon value for RMSNorm stability.
        rope_theta: Base frequency for rotary position embeddings.
        hidden_act: Activation function name (e.g., "silu").
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 896,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        max_position_embeddings: int = 32768,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
        hidden_act: str = "silu",
        **kwargs: Any,
    ) -> None:
        """Initialize Qwen3Config with model hyperparameters.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Dimension of the hidden representations.
            num_hidden_layers: Number of transformer decoder layers.
            num_attention_heads: Number of attention heads for queries.
            num_key_value_heads: Number of attention heads for keys/values (GQA).
            intermediate_size: Dimension of the FFN intermediate layer.
            max_position_embeddings: Maximum sequence length supported.
            rms_norm_eps: Epsilon value for RMSNorm stability.
            rope_theta: Base frequency for rotary position embeddings.
            hidden_act: Activation function name (e.g., "silu").
            **kwargs: Additional configuration parameters (ignored).
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act

        # Validate architecture constraints
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters satisfy Qwen3 architecture constraints.

        Raises:
            ValueError: If configuration parameters are invalid.
        """
        # Validate head dimension is consistent
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )

        # Validate GQA constraint
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({self.num_key_value_heads})"
            )

        # Validate positive values
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive, got {self.num_hidden_layers}"
            )
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be positive, got {self.num_attention_heads}"
            )
        if self.num_key_value_heads <= 0:
            raise ValueError(
                f"num_key_value_heads must be positive, got {self.num_key_value_heads}"
            )
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, got {self.intermediate_size}"
            )
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, got {self.max_position_embeddings}"
            )
        if self.rms_norm_eps <= 0:
            raise ValueError(f"rms_norm_eps must be positive, got {self.rms_norm_eps}")
        if self.rope_theta <= 0:
            raise ValueError(f"rope_theta must be positive, got {self.rope_theta}")

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "Qwen3Config":
        """Load configuration from a HuggingFace pretrained model.

        Args:
            model_name_or_path: Model identifier or path to model directory.

        Returns:
            Qwen3Config instance with parameters loaded from the pretrained model.
        """
        # Load HuggingFace config
        hf_config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )

        # Extract relevant parameters
        return cls(
            vocab_size=hf_config.vocab_size,
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            num_key_value_heads=hf_config.num_key_value_heads,
            intermediate_size=hf_config.intermediate_size,
            max_position_embeddings=hf_config.max_position_embeddings,
            rms_norm_eps=hf_config.rms_norm_eps,
            rope_theta=hf_config.rope_theta,
            hidden_act=hf_config.hidden_act,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary.

        Returns:
            Dictionary containing all configuration parameters.
        """
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "hidden_act": self.hidden_act,
        }

    def __repr__(self) -> str:
        """Return a readable string representation of the configuration.

        Returns:
            String representation showing key configuration parameters.
        """
        return (
            f"Qwen3Config("
            f"vocab_size={self.vocab_size}, "
            f"hidden_size={self.hidden_size}, "
            f"num_hidden_layers={self.num_hidden_layers}, "
            f"num_attention_heads={self.num_attention_heads}, "
            f"num_key_value_heads={self.num_key_value_heads}, "
            f"intermediate_size={self.intermediate_size}, "
            f"max_position_embeddings={self.max_position_embeddings}, "
            f"rms_norm_eps={self.rms_norm_eps}, "
            f"rope_theta={self.rope_theta}, "
            f"hidden_act='{self.hidden_act}'"
            f")"
        )
