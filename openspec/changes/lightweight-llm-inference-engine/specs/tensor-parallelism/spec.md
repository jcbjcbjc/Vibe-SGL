## ADDED Requirements

### Requirement: Process Group Initialization
The system SHALL initialize torch.distributed process groups for tensor parallel workers.

#### Scenario: Initialize TP process group
- **WHEN** system starts with TP degree > 1
- **THEN** system initializes process group with Gloo backend for CPU or NCCL for GPU

#### Scenario: Rank assignment
- **WHEN** process group is created
- **THEN** each worker receives unique rank from 0 to (tp_degree - 1)

#### Scenario: World size configuration
- **WHEN** TP is enabled
- **THEN** system sets world size equal to TP degree

### Requirement: Column Parallelism
The system SHALL implement column parallelism for splitting weight matrices across devices.

#### Scenario: Split Q/K/V projections
- **WHEN** attention layer is initialized with TP
- **THEN** system splits Q, K, V projection weights along output dimension

#### Scenario: Split FFN up/gate projections
- **WHEN** FFN layer is initialized with TP
- **THEN** system splits up_proj and gate_proj weights along output dimension

#### Scenario: Local computation
- **WHEN** forward pass executes
- **THEN** each rank computes its portion of output without communication

### Requirement: Row Parallelism
The system SHALL implement row parallelism with all-reduce for output aggregation.

#### Scenario: Split output projection
- **WHEN** attention output layer is initialized with TP
- **THEN** system splits output projection weights along input dimension

#### Scenario: Split FFN down projection
- **WHEN** FFN down_proj is initialized with TP
- **THEN** system splits down_proj weights along input dimension

#### Scenario: All-reduce after computation
- **WHEN** row parallel layer completes forward pass
- **THEN** system performs all-reduce to sum partial results across ranks

### Requirement: Weight Partitioning
The system SHALL partition pretrained weights correctly for TP workers.

#### Scenario: Load full checkpoint
- **WHEN** system loads Qwen3 checkpoint
- **THEN** rank 0 loads full weights from HuggingFace format

#### Scenario: Partition weights
- **WHEN** weights are loaded
- **THEN** system slices weights along appropriate dimensions for each rank

#### Scenario: Distribute weights
- **WHEN** weights are partitioned
- **THEN** system distributes weight slices to corresponding ranks

### Requirement: Communication Primitives
The system SHALL use torch.distributed primitives for all TP communication.

#### Scenario: All-reduce operation
- **WHEN** row parallel layer needs aggregation
- **THEN** system uses torch.distributed.all_reduce() with SUM operation

#### Scenario: Broadcast operation
- **WHEN** shared data needs distribution
- **THEN** system uses torch.distributed.broadcast() from rank 0

#### Scenario: Barrier synchronization
- **WHEN** synchronization needed
- **THEN** system uses torch.distributed.barrier() to sync all ranks

### Requirement: Attention with TP
The system SHALL correctly implement attention computation with tensor parallelism.

#### Scenario: Parallel Q/K/V computation
- **WHEN** attention forward pass executes
- **THEN** each rank computes Q, K, V for its head partition

#### Scenario: Local attention computation
- **WHEN** computing attention scores
- **THEN** each rank computes attention for its head partition independently

#### Scenario: Aggregate attention output
- **WHEN** attention output projection executes
- **THEN** system performs all-reduce to combine results from all ranks

### Requirement: FFN with TP
The system SHALL correctly implement FFN computation with tensor parallelism.

#### Scenario: Parallel up/gate projection
- **WHEN** FFN forward pass executes
- **THEN** each rank computes its portion of up_proj and gate_proj

#### Scenario: Local activation
- **WHEN** applying activation function
- **THEN** each rank applies activation to its local outputs

#### Scenario: Aggregate down projection
- **WHEN** down_proj executes
- **THEN** system performs all-reduce to combine results

### Requirement: Embedding and Output Layers
The system SHALL handle embedding and output layers with TP.

#### Scenario: Replicated embeddings
- **WHEN** embedding layer is used
- **THEN** system replicates embedding weights on all ranks (no partitioning)

#### Scenario: Parallel output projection
- **WHEN** final output layer computes logits
- **THEN** system partitions output projection and gathers results

#### Scenario: Vocabulary partitioning
- **WHEN** vocabulary is very large
- **THEN** system optionally partitions vocabulary across ranks

### Requirement: TP Testing on CPU
The system SHALL support TP testing using CPU with Gloo backend.

#### Scenario: Initialize Gloo backend
- **WHEN** running tests on CPU
- **THEN** system initializes torch.distributed with Gloo backend

#### Scenario: Multi-process testing
- **WHEN** running TP tests
- **THEN** system spawns multiple processes for different TP ranks

#### Scenario: Validate correctness
- **WHEN** TP forward pass completes
- **THEN** system validates output matches single-device reference

### Requirement: TP Configuration
The system SHALL provide configuration options for tensor parallelism.

#### Scenario: Configure TP degree
- **WHEN** user specifies tp_degree parameter
- **THEN** system initializes TP with specified number of workers

#### Scenario: Disable TP
- **WHEN** tp_degree is 1
- **THEN** system runs in single-device mode without TP overhead

#### Scenario: Validate TP degree
- **WHEN** user provides tp_degree
- **THEN** system validates it divides model dimensions evenly
