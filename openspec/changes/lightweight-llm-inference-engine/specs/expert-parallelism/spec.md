## ADDED Requirements

### Requirement: Expert Router
The system SHALL implement expert routing to distribute tokens to appropriate experts.

#### Scenario: Route tokens to experts
- **WHEN** MoE layer receives input tokens
- **THEN** system computes routing scores and assigns tokens to top-k experts

#### Scenario: Top-k expert selection
- **WHEN** routing scores are computed
- **THEN** system selects k experts with highest scores for each token

#### Scenario: Routing weights
- **WHEN** experts are selected
- **THEN** system computes normalized routing weights for combining expert outputs

### Requirement: Expert Placement
The system SHALL distribute experts across devices for parallel execution.

#### Scenario: Static expert placement
- **WHEN** system initializes with EP
- **THEN** system assigns experts to devices using (expert_id % num_devices) mapping

#### Scenario: Expert-to-device mapping
- **WHEN** system needs to locate expert
- **THEN** system uses mapping table to find which device holds the expert

#### Scenario: Load balancing
- **WHEN** experts are distributed
- **THEN** system distributes experts evenly across available devices

### Requirement: Token Redistribution
The system SHALL redistribute tokens to devices based on expert assignments.

#### Scenario: All-to-all communication
- **WHEN** tokens are routed to experts
- **THEN** system uses torch.distributed.all_to_all() to redistribute tokens

#### Scenario: Gather tokens for local experts
- **WHEN** device receives tokens
- **THEN** system gathers all tokens assigned to its local experts

#### Scenario: Scatter results back
- **WHEN** expert computation completes
- **THEN** system scatters results back to original token positions

### Requirement: Expert Computation
The system SHALL execute expert computations on assigned devices.

#### Scenario: Local expert forward pass
- **WHEN** tokens arrive at expert device
- **THEN** system executes expert FFN for those tokens

#### Scenario: Batch expert computation
- **WHEN** multiple tokens assigned to same expert
- **THEN** system batches computation for efficiency

#### Scenario: Empty expert handling
- **WHEN** expert receives no tokens
- **THEN** system skips computation for that expert

### Requirement: Capacity Factor
The system SHALL implement capacity factor to limit tokens per expert.

#### Scenario: Configure capacity factor
- **WHEN** user specifies capacity_factor parameter
- **THEN** system limits tokens per expert to (capacity_factor * avg_tokens_per_expert)

#### Scenario: Drop overflow tokens
- **WHEN** expert capacity exceeded
- **THEN** system drops overflow tokens or routes to auxiliary expert

#### Scenario: Capacity metrics
- **WHEN** system processes MoE layer
- **THEN** system tracks how many tokens were dropped due to capacity

### Requirement: Expert Parallelism with TP
The system SHALL support combining expert parallelism with tensor parallelism.

#### Scenario: Hybrid EP+TP
- **WHEN** both EP and TP are enabled
- **THEN** system partitions experts across EP dimension and uses TP within each expert

#### Scenario: Process group hierarchy
- **WHEN** EP and TP are combined
- **THEN** system creates separate process groups for EP and TP communication

#### Scenario: Communication ordering
- **WHEN** hybrid parallelism is used
- **THEN** system performs EP all-to-all before TP all-reduce

### Requirement: Synthetic MoE for Testing
The system SHALL provide synthetic MoE layer for testing EP without full MoE model.

#### Scenario: Create synthetic MoE
- **WHEN** testing EP functionality
- **THEN** system creates simple MoE layer with configurable number of experts

#### Scenario: Synthetic routing
- **WHEN** synthetic MoE processes tokens
- **THEN** system uses simple routing logic (e.g., round-robin or random)

#### Scenario: Validate EP correctness
- **WHEN** synthetic MoE completes
- **THEN** system validates output matches single-device reference

### Requirement: Expert Load Balancing
The system SHALL track and report expert load distribution.

#### Scenario: Track tokens per expert
- **WHEN** routing completes
- **THEN** system records how many tokens were assigned to each expert

#### Scenario: Compute load imbalance
- **WHEN** layer completes
- **THEN** system calculates load imbalance metric (std dev of tokens per expert)

#### Scenario: Report underutilized experts
- **WHEN** some experts receive few tokens
- **THEN** system logs which experts are underutilized

### Requirement: EP Configuration
The system SHALL provide configuration options for expert parallelism.

#### Scenario: Configure EP degree
- **WHEN** user specifies ep_degree parameter
- **THEN** system initializes EP with specified number of devices

#### Scenario: Configure number of experts
- **WHEN** user specifies num_experts parameter
- **THEN** system creates MoE layer with specified expert count

#### Scenario: Validate EP configuration
- **WHEN** user provides EP configuration
- **THEN** system validates num_experts is divisible by ep_degree

### Requirement: EP Testing on CPU
The system SHALL support EP testing using CPU with Gloo backend.

#### Scenario: Initialize EP process group
- **WHEN** running EP tests on CPU
- **THEN** system initializes separate process group for EP communication

#### Scenario: Multi-process EP testing
- **WHEN** running EP tests
- **THEN** system spawns processes for different EP ranks

#### Scenario: Validate EP correctness
- **WHEN** EP forward pass completes
- **THEN** system validates output matches single-device MoE reference
