## ADDED Requirements

### Requirement: Dynamic Batch Composition
The system SHALL allow requests to join and leave the active batch at any iteration boundary.

#### Scenario: Add request to running batch
- **WHEN** new request arrives while batch is executing
- **THEN** system adds request to batch at next iteration boundary

#### Scenario: Remove completed request
- **WHEN** request generates EOS token or reaches max_tokens
- **THEN** system removes request from batch immediately

#### Scenario: Batch size variation
- **WHEN** requests join or leave
- **THEN** batch size dynamically adjusts without waiting for all requests to complete

### Requirement: Iteration-Level Scheduling
The system SHALL schedule batch composition decisions at each decode iteration.

#### Scenario: Pre-iteration scheduling
- **WHEN** iteration begins
- **THEN** system decides which waiting requests to add to batch

#### Scenario: Post-iteration cleanup
- **WHEN** iteration completes
- **THEN** system removes finished requests and updates batch state

#### Scenario: Maximum batch size enforcement
- **WHEN** batch reaches configured maximum size
- **THEN** system queues additional requests until space available

### Requirement: Prefill and Decode Separation
The system SHALL handle prefill (initial processing) and decode (token generation) phases within the same iteration.

#### Scenario: Prefill new requests
- **WHEN** new requests join batch
- **THEN** system performs prefill to process their input tokens

#### Scenario: Decode existing requests
- **WHEN** requests are already in batch
- **THEN** system performs decode to generate next token

#### Scenario: Mixed prefill-decode batch
- **WHEN** batch contains both new and existing requests
- **THEN** system processes prefill and decode in same iteration

### Requirement: Request State Management
The system SHALL maintain state for each request across iterations.

#### Scenario: Track generation progress
- **WHEN** request generates tokens across iterations
- **THEN** system maintains token count, KV cache references, and generation state

#### Scenario: Preserve request context
- **WHEN** request spans multiple iterations
- **THEN** system preserves all request parameters (sampling, max_tokens, etc.)

#### Scenario: Clean up completed requests
- **WHEN** request completes
- **THEN** system releases all associated resources (KV cache, state)

### Requirement: Batch Padding
The system SHALL pad sequences to uniform length within each batch for efficient computation.

#### Scenario: Pad to max length
- **WHEN** batch contains sequences of different lengths
- **THEN** system pads shorter sequences to match longest sequence in batch

#### Scenario: Attention mask for padding
- **WHEN** sequences are padded
- **THEN** system creates attention masks to ignore padding tokens

#### Scenario: Minimize padding overhead
- **WHEN** forming batch
- **THEN** system groups requests with similar lengths to reduce padding waste

### Requirement: Throughput Optimization
The system SHALL maximize throughput by keeping batches full whenever possible.

#### Scenario: Fill batch eagerly
- **WHEN** batch has available slots and requests are waiting
- **THEN** system adds waiting requests up to maximum batch size

#### Scenario: Avoid idle iterations
- **WHEN** batch becomes empty
- **THEN** system immediately starts new batch with waiting requests

#### Scenario: Throughput metrics
- **WHEN** system processes requests
- **THEN** system tracks tokens per second and batch utilization rate

### Requirement: Latency Fairness
The system SHALL ensure no request is starved due to continuous batching.

#### Scenario: Maximum wait time
- **WHEN** request waits in queue
- **THEN** system MUST add request to batch within configured maximum wait time

#### Scenario: Priority for waiting requests
- **WHEN** batch has available slots
- **THEN** system prioritizes requests that have waited longest (unless overridden by policy)

#### Scenario: Prevent starvation
- **WHEN** high-throughput scenario with constant new requests
- **THEN** system ensures all requests eventually get processed

### Requirement: Streaming Support
The system SHALL support streaming token generation for continuous batching.

#### Scenario: Stream tokens as generated
- **WHEN** request is in streaming mode
- **THEN** system yields each generated token immediately after decode

#### Scenario: Concurrent streaming
- **WHEN** multiple requests in batch use streaming
- **THEN** system streams tokens for all requests concurrently

#### Scenario: Stream completion
- **WHEN** request completes generation
- **THEN** system sends final stream event and closes stream
