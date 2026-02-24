"""
Core inference engine for text generation.
"""

import torch
from typing import List, Optional, Iterator, Dict
from transformers import AutoTokenizer
from vibe_sgl_lite.models.qwen3.model import Qwen3Model
from vibe_sgl_lite.models.qwen3.lm_head import Qwen3LMHead
from vibe_sgl_lite.models.qwen3.config import Qwen3Config
from vibe_sgl_lite.models.qwen3.weight_loader import load_and_map_weights
from vibe_sgl_lite.sampling.sampling import SamplingParams, sample
from vibe_sgl_lite.batch.batch_manager import BatchManager
from vibe_sgl_lite.batch.request import Request, RequestState
from vibe_sgl_lite.memory.memory_pool import MemoryPool
from vibe_sgl_lite.memory.paged_allocator import PagedAllocator
from vibe_sgl_lite.cache.radix_cache import RadixCache


class InferenceEngine:
    """Core inference engine for text generation."""

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        enable_continuous_batching: bool = False,
        max_batch_size: int = 8,
        enable_paged_attention: bool = False,
        kv_cache_size_mb: int = 1024,
        page_size: int = 16,
        enable_prefix_caching: bool = False,
    ):
        """Initialize inference engine.

        Args:
            model_name: HuggingFace model name or path to model directory
            device: Device to run inference on ("cpu" or "cuda")
            enable_continuous_batching: Enable continuous batching mode
            max_batch_size: Maximum batch size for continuous batching
            enable_paged_attention: Enable paged attention for memory efficiency
            kv_cache_size_mb: KV cache size in MB for paged attention
            page_size: Page size (number of tokens per page) for paged attention
            enable_prefix_caching: Enable prefix caching with RadixCache

        Raises:
            ValueError: If model cannot be loaded or path is invalid
            RuntimeError: If model architecture is incompatible
        """
        if not model_name:
            raise ValueError("model_name cannot be empty")

        try:
            # Load config
            self.config = Qwen3Config.from_pretrained(model_name)

            # Initialize model and lm_head
            self.model = Qwen3Model(self.config)
            self.lm_head = Qwen3LMHead(self.config)

            # Load weights from checkpoint
            state_dict = load_and_map_weights(model_name, self.config)

            # Split state dict for model and lm_head
            model_state_dict = {}
            lm_head_state_dict = {}

            for name, param in state_dict.items():
                if name.startswith("lm_head."):
                    lm_head_state_dict[name.replace("lm_head.", "")] = param
                else:
                    model_state_dict[name] = param

            # Load weights into model and lm_head
            self.model.load_state_dict(model_state_dict, strict=False)
            self.lm_head.load_state_dict(lm_head_state_dict, strict=False)

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )

            self.device = device
            self.model.eval()
            self.lm_head.eval()

            # Continuous batching support
            self.enable_continuous_batching = enable_continuous_batching
            if enable_continuous_batching:
                self.batch_manager = BatchManager(max_batch_size=max_batch_size)
            else:
                self.batch_manager = None

            # Paged attention support
            self.enable_paged_attention = enable_paged_attention
            if enable_paged_attention:
                # Calculate number of pages based on cache size
                # Each page stores KV for page_size tokens
                # KV size per token = 2 * num_layers * num_kv_heads * head_dim * sizeof(float32)
                bytes_per_token = (
                    2
                    * self.config.num_hidden_layers
                    * self.config.num_key_value_heads
                    * (self.config.hidden_size // self.config.num_attention_heads)
                    * 4
                )
                bytes_per_page = bytes_per_token * page_size
                num_pages = (kv_cache_size_mb * 1024 * 1024) // bytes_per_page

                self.memory_pool = MemoryPool(
                    num_pages=num_pages,
                    page_size=page_size,
                    num_layers=self.config.num_hidden_layers,
                    num_kv_heads=self.config.num_key_value_heads,
                    head_dim=self.config.hidden_size
                    // self.config.num_attention_heads,
                    device=device,
                )
                self.paged_allocator = PagedAllocator(self.memory_pool, page_size)
            else:
                self.memory_pool = None
                self.paged_allocator = None

            # Prefix caching support
            self.enable_prefix_caching = enable_prefix_caching
            if enable_prefix_caching:
                self.radix_cache = RadixCache()
            else:
                self.radix_cache = None

        except Exception as e:
            raise ValueError(f"Failed to initialize InferenceEngine: {str(e)}")

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize input text.

        Args:
            text: Input text to tokenize

        Returns:
            Token IDs as tensor of shape [1, seq_len]

        Raises:
            ValueError: If text is empty or None
        """
        if text is None:
            raise TypeError("text cannot be None")
        if not text or not text.strip():
            raise ValueError("text cannot be empty")

        # Tokenize using the tokenizer
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        return tokens

    def detokenize(self, tokens: torch.Tensor) -> str:
        """Detokenize token IDs to text.

        Args:
            tokens: Token IDs as tensor of shape [batch_size, seq_len] or [seq_len]

        Returns:
            Decoded text string
        """
        # Handle both 1D and 2D tensors
        if tokens.dim() == 2:
            tokens = tokens[0]  # Take first sequence if batched

        # Decode tokens to text
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return text

    def tokenize_batch(self, texts: List[str]) -> torch.Tensor:
        """Tokenize a batch of input texts.

        Args:
            texts: List of input texts to tokenize

        Returns:
            Token IDs as tensor of shape [batch_size, seq_len]

        Raises:
            ValueError: If any text is empty or None
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        for text in texts:
            if text is None:
                raise TypeError("text cannot be None")
            if not text or not text.strip():
                raise ValueError("text cannot be empty")

        # Tokenize batch with padding
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return tokens["input_ids"]

    def generate_text(
        self,
        text: str,
        max_new_tokens: int = 50,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """Generate text from text input (convenience method).

        Args:
            text: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            sampling_params: Sampling parameters for generation

        Returns:
            Generated text including the input prompt

        Raises:
            ValueError: If text is empty or None
        """
        # Tokenize input
        input_ids = self.tokenize(text)

        # Generate tokens
        output_ids = self.generate(input_ids, max_new_tokens, sampling_params)

        # Detokenize output
        output_text = self.detokenize(output_ids)

        return output_text

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        sampling_params: Optional[SamplingParams] = None,
    ) -> torch.Tensor:
        """Generate tokens."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                hidden_states = self.model(generated)
                logits = self.lm_head(hidden_states)

                # Sample next token
                next_token_logits = logits[:, -1, :]
                next_token = sample(next_token_logits, sampling_params)

                # Append to sequence
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        return generated

    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Iterator[torch.Tensor]:
        """Generate tokens with streaming."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                hidden_states = self.model(generated)
                logits = self.lm_head(hidden_states)

                next_token_logits = logits[:, -1, :]
                next_token = sample(next_token_logits, sampling_params)

                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

                yield next_token

    def generate_batch(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        sampling_params: Optional[SamplingParams] = None,
    ) -> torch.Tensor:
        """Generate tokens for a batch of sequences.

        Args:
            input_ids: Input token IDs of shape [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            sampling_params: Sampling parameters for generation

        Returns:
            Generated token IDs of shape [batch_size, seq_len + max_new_tokens]
        """
        if sampling_params is None:
            sampling_params = SamplingParams()

        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                hidden_states = self.model(generated)
                logits = self.lm_head(hidden_states)

                # Sample next token for each sequence in batch
                next_token_logits = logits[:, -1, :]
                next_token = sample(next_token_logits, sampling_params)

                # Append to sequences
                generated = torch.cat([generated, next_token.unsqueeze(-1)], dim=-1)

        return generated

    def generate_batch_text(
        self,
        texts: List[str],
        max_new_tokens: int = 50,
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """Generate text for a batch of input texts.

        Args:
            texts: List of input text prompts
            max_new_tokens: Maximum number of new tokens to generate
            sampling_params: Sampling parameters for generation

        Returns:
            List of generated texts including the input prompts
        """
        # Tokenize batch
        input_ids = self.tokenize_batch(texts)

        # Generate tokens
        output_ids = self.generate_batch(input_ids, max_new_tokens, sampling_params)

        # Detokenize each sequence
        outputs = []
        for i in range(output_ids.shape[0]):
            text = self.tokenizer.decode(output_ids[i], skip_special_tokens=True)
            outputs.append(text)

        return outputs

    def generate_stream_text(
        self,
        text: str,
        max_new_tokens: int = 50,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Iterator[str]:
        """Generate text with streaming (yields decoded tokens).

        Args:
            text: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            sampling_params: Sampling parameters for generation

        Yields:
            Decoded text for each generated token
        """
        # Tokenize input
        input_ids = self.tokenize(text)

        # Generate with streaming
        for token in self.generate_stream(input_ids, max_new_tokens, sampling_params):
            # Decode token to text
            token_text = self.tokenizer.decode(token[0], skip_special_tokens=True)
            yield token_text

    # Continuous Batching Methods

    def submit_request(
        self,
        text: str,
        max_new_tokens: int = 50,
        sampling_params: Optional[SamplingParams] = None,
        request_id: Optional[str] = None,
    ) -> str:
        """Submit a request for continuous batching.

        Args:
            text: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            sampling_params: Sampling parameters for generation
            request_id: Optional request ID (auto-generated if not provided)

        Returns:
            Request ID for tracking

        Raises:
            RuntimeError: If continuous batching is not enabled
        """
        if not self.enable_continuous_batching:
            raise RuntimeError(
                "Continuous batching is not enabled. "
                "Initialize with enable_continuous_batching=True"
            )

        # Generate request ID if not provided
        if request_id is None:
            import uuid
            request_id = f"req_{uuid.uuid4().hex[:8]}"

        # Tokenize input
        input_ids = self.tokenize(text)

        # Create request
        request = Request(
            request_id=request_id,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            sampling_params=sampling_params or SamplingParams(),
        )

        # Add to batch manager
        self.batch_manager.add_request(request)

        return request_id

    def get_request_output(self, request_id: str) -> Optional[str]:
        """Get output for a completed request.

        Args:
            request_id: Request ID to query

        Returns:
            Generated text if request is completed, None otherwise
        """
        if not self.enable_continuous_batching:
            raise RuntimeError("Continuous batching is not enabled")

        # Find request in batch manager
        request = self.batch_manager._request_map.get(request_id)
        if request is None:
            return None

        # Return output if completed
        if request.is_completed() and request.output_ids is not None:
            return self.detokenize(request.output_ids)

        return None

    def run_continuous_batching_step(self) -> None:
        """Run one step of continuous batching.

        This method:
        1. Updates batch composition (add/remove requests)
        2. Processes prefill requests
        3. Processes decode requests
        4. Updates request states
        """
        if not self.enable_continuous_batching:
            raise RuntimeError("Continuous batching is not enabled")

        # Step 1: Update batch composition
        self.batch_manager.step()

        if self.batch_manager.is_empty():
            return

        # Step 2: Get prefill and decode requests
        prefill_requests = self.batch_manager.get_prefill_requests()
        decode_requests = self.batch_manager.get_decode_requests()

        # Step 3: Process prefill requests
        for request in prefill_requests:
            with torch.no_grad():
                # Forward pass for prefill
                hidden_states = self.model(request.input_ids)
                logits = self.lm_head(hidden_states)

                # Sample first token
                next_token_logits = logits[:, -1, :]
                next_token = sample(next_token_logits, request.sampling_params)

                # Update request
                request.output_ids = torch.cat(
                    [request.input_ids, next_token.unsqueeze(-1)], dim=-1
                )
                request.generated_tokens = 1
                request.state = RequestState.DECODING

        # Step 4: Process decode requests
        if decode_requests:
            # Create padded batch for decode
            batch_input_ids, attention_mask = self.batch_manager.create_padded_batch()

            with torch.no_grad():
                # Forward pass for batch
                hidden_states = self.model(batch_input_ids)
                logits = self.lm_head(hidden_states)

                # Sample next token for each request
                next_token_logits = logits[:, -1, :]

                for i, request in enumerate(decode_requests):
                    # Sample token for this request
                    token_logits = next_token_logits[i : i + 1]
                    next_token = sample(token_logits, request.sampling_params)

                    # Update request
                    request.output_ids = torch.cat(
                        [request.output_ids, next_token.unsqueeze(-1)], dim=-1
                    )
                    request.generated_tokens += 1

                    # Check if finished
                    if request.is_finished():
                        request.state = RequestState.COMPLETED

    def run_continuous_batching(
        self, max_iterations: int = 1000
    ) -> Dict[str, str]:
        """Run continuous batching until all requests complete.

        Args:
            max_iterations: Maximum number of iterations to run

        Returns:
            Dictionary mapping request IDs to generated texts
        """
        if not self.enable_continuous_batching:
            raise RuntimeError("Continuous batching is not enabled")

        for _ in range(max_iterations):
            self.run_continuous_batching_step()

            # Check if all requests are done
            if (
                self.batch_manager.is_empty()
                and not self.batch_manager.has_waiting_requests()
            ):
                break

        # Collect outputs
        outputs = {}
        for request_id in list(self.batch_manager._request_map.keys()):
            output = self.get_request_output(request_id)
            if output is not None:
                outputs[request_id] = output

        return outputs

    def get_batch_metrics(self) -> Dict[str, float]:
        """Get continuous batching metrics.

        Returns:
            Dictionary containing batch metrics
        """
        if not self.enable_continuous_batching:
            raise RuntimeError("Continuous batching is not enabled")

        return self.batch_manager.get_metrics()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory pool statistics.

        Returns:
            Dictionary containing memory statistics
        """
        if not self.enable_paged_attention:
            raise RuntimeError("Paged attention is not enabled")

        return self.memory_pool.get_stats()

    def get_cache_stats(self) -> Dict[str, float]:
        """Get prefix cache statistics.

        Returns:
            Dictionary containing cache statistics
        """
        if not self.enable_prefix_caching:
            raise RuntimeError("Prefix caching is not enabled")

        return self.radix_cache.get_stats()
