## ADDED Requirements

### Requirement: Chunk Size Configuration
The system SHALL split long input sequences into configurable chunks with default size of 8192 tokens.

#### Scenario: Configure chunk size
- **WHEN** user specifies chunk size parameter
- **THEN** system uses specified size for splitting sequences

#### Scenario: Default chunk size
- **WHEN** no chunk size specified
- **THEN** system uses 8192 tokens as default chunk size

#### Scenario: Chunk size validation
- **WHEN** user provides invalid chunk size (zero or negative)
- **THEN** system raises ValueError with valid range

### Requirement: Sequence Chunking
The system SHALL split input sequences longer than chunk size into multiple chunks.

#### Scenario: Split long sequence
- **WHEN** input sequence exceeds chunk size
- **THEN** system divides sequence into chunks of configured size

#### Scenario: Last chunk handling
- **WHEN** sequence length not evenly divisible by chunk size
- **THEN** last chunk contains remaining tokens (may be smaller than chunk size)

#### Scenario: Short sequence passthrough
- **WHEN** input sequence shorter than chunk size
- **THEN** system processes sequence as single chunk without splitting

### Requirement: Incremental Processing
The system SHALL process chunks incrementally, one chunk at a time.

#### Scenario: Process first chunk
- **WHEN** chunked prefill begins
- **THEN** system processes first chunk and allocates initial KV cache pages

#### Scenario: Process subsequent chunks
- **WHEN** first chunk completes
- **THEN** system processes next chunk and extends KV cache

#### Scenario: Track chunk progress
- **WHEN** processing chunks
- **THEN** system maintains state tracking which chunks are complete

### Requirement: Interleaved Prefill-Decode
The system SHALL interleave chunk processing with decode operations for other requests.

#### Scenario: Interleave with decode
- **WHEN** processing chunk for one request
- **THEN** system can perform decode steps for other requests in same iteration

#### Scenario: Chunk scheduling
- **WHEN** batch has both chunked prefill and decode requests
- **THEN** system schedules chunk processing to avoid blocking decode requests

#### Scenario: Fairness across phases
- **WHEN** multiple requests need prefill chunks
- **THEN** system distributes compute fairly between prefill and decode

### Requirement: Memory Allocation
The system SHALL allocate KV cache pages incrementally as chunks are processed.

#### Scenario: Allocate pages per chunk
- **WHEN** chunk is processed
- **THEN** system allocates only pages needed for that chunk

#### Scenario: Prevent memory spikes
- **WHEN** very long sequence is processed
- **THEN** memory usage grows incrementally rather than all at once

#### Scenario: Page allocation tracking
- **WHEN** chunked prefill progresses
- **THEN** system tracks total pages allocated across all chunks

### Requirement: Chunk State Management
The system SHALL maintain state for partially processed chunked sequences.

#### Scenario: Save chunk position
- **WHEN** chunk completes
- **THEN** system saves position for resuming with next chunk

#### Scenario: Resume from checkpoint
- **WHEN** next chunk begins
- **THEN** system resumes from saved position with existing KV cache

#### Scenario: Handle interruption
- **WHEN** chunked prefill is interrupted (e.g., eviction)
- **THEN** system can resume from last completed chunk

### Requirement: Attention Mask Handling
The system SHALL correctly handle attention masks across chunk boundaries.

#### Scenario: Intra-chunk attention
- **WHEN** processing tokens within a chunk
- **THEN** tokens attend to all previous tokens in current and prior chunks

#### Scenario: Cross-chunk attention
- **WHEN** processing new chunk
- **THEN** tokens attend to KV cache from all previous chunks

#### Scenario: Causal masking
- **WHEN** computing attention
- **THEN** system enforces causal masking across chunk boundaries

### Requirement: Performance Metrics
The system SHALL track metrics for chunked prefill performance.

#### Scenario: Track chunk processing time
- **WHEN** chunk is processed
- **THEN** system records time taken for that chunk

#### Scenario: Monitor memory efficiency
- **WHEN** chunked prefill is used
- **THEN** system tracks peak memory usage vs. non-chunked baseline

#### Scenario: Report chunking overhead
- **WHEN** sequence completes
- **THEN** system reports total overhead from chunking (if any)
