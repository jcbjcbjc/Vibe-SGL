"""
Example demonstrating continuous batching with InferenceEngine.

This example shows how to use the continuous batching feature to process
multiple generation requests concurrently with dynamic batch composition.
"""

from vibe_sgl_lite.core.inference_engine import InferenceEngine
from vibe_sgl_lite.sampling.sampling import SamplingParams

# Initialize engine with continuous batching enabled
print("Initializing InferenceEngine with continuous batching...")
engine = InferenceEngine(
    model_name="Qwen/Qwen2.5-0.5B",
    device="cpu",
    enable_continuous_batching=True,
    max_batch_size=4,
)

# Submit multiple requests
print("\nSubmitting requests...")
requests = [
    ("The capital of France is", 10),
    ("Hello, my name is", 8),
    ("In the year 2024,", 12),
]

request_ids = []
for text, max_tokens in requests:
    req_id = engine.submit_request(
        text=text,
        max_new_tokens=max_tokens,
        sampling_params=SamplingParams(temperature=0.7),
    )
    request_ids.append(req_id)
    print(f"  Submitted: {req_id} - '{text}'")

# Run continuous batching
print("\nRunning continuous batching...")
outputs = engine.run_continuous_batching(max_iterations=50)

# Display results
print("\nGenerated outputs:")
for req_id, output in outputs.items():
    print(f"\n{req_id}:")
    print(f"  {output}")

# Show metrics
print("\nBatch metrics:")
metrics = engine.get_batch_metrics()
print(f"  Total requests: {metrics['total_requests']}")
print(f"  Completed requests: {metrics['completed_requests']}")
print(f"  Final batch utilization: {metrics['batch_utilization']:.2%}")
