## ADDED Requirements

### Requirement: Policy Interface
The system SHALL provide an abstract interface for implementing different scheduling policies.

#### Scenario: Define policy interface
- **WHEN** system initializes scheduler
- **THEN** system provides SchedulerPolicy base class with select_requests() method

#### Scenario: Pluggable policies
- **WHEN** user specifies scheduling policy
- **THEN** system loads and uses specified policy implementation

#### Scenario: Policy switching
- **WHEN** user changes policy configuration
- **THEN** system switches to new policy without restart

### Requirement: FCFS Policy
The system SHALL implement First-Come-First-Serve scheduling policy.

#### Scenario: Queue-based ordering
- **WHEN** FCFS policy selects requests
- **THEN** system selects requests in arrival order

#### Scenario: Fair processing
- **WHEN** multiple requests waiting
- **THEN** system processes oldest request first

#### Scenario: Simple implementation
- **WHEN** FCFS policy is used
- **THEN** system uses simple queue without complex scoring

### Requirement: LPM Policy
The system SHALL implement Longest Prefix Match scheduling policy for cache-aware scheduling.

#### Scenario: Score by prefix length
- **WHEN** LPM policy evaluates requests
- **THEN** system scores each request by length of cached prefix match

#### Scenario: Prioritize cache hits
- **WHEN** LPM policy selects requests
- **THEN** system selects requests with longest cached prefixes first

#### Scenario: Maximize cache reuse
- **WHEN** batch is formed with LPM
- **THEN** system maximizes total cached tokens across batch

#### Scenario: Fallback to FCFS
- **WHEN** multiple requests have equal prefix match length
- **THEN** system uses arrival order as tiebreaker

### Requirement: Batch Formation
The system SHALL use scheduling policy to form batches from waiting requests.

#### Scenario: Select batch members
- **WHEN** batch has available slots
- **THEN** system uses policy to select which waiting requests to add

#### Scenario: Respect batch size limit
- **WHEN** policy selects requests
- **THEN** system enforces maximum batch size constraint

#### Scenario: Consider sequence length
- **WHEN** forming batch
- **THEN** system considers sequence lengths to minimize padding overhead

### Requirement: Priority Queues
The system SHALL maintain priority queues for efficient request selection.

#### Scenario: Maintain sorted queue
- **WHEN** requests arrive
- **THEN** system maintains queue sorted by policy-specific priority

#### Scenario: Efficient selection
- **WHEN** policy selects requests
- **THEN** system retrieves highest priority requests in O(log n) time

#### Scenario: Dynamic priority updates
- **WHEN** request priorities change (e.g., cache state changes)
- **THEN** system updates queue ordering accordingly

### Requirement: Cache-Aware Scoring
The system SHALL integrate with RadixAttention to score requests based on cache state.

#### Scenario: Query cache for prefix match
- **WHEN** LPM policy evaluates request
- **THEN** system queries radix tree for longest prefix match

#### Scenario: Score calculation
- **WHEN** cache match found
- **THEN** system calculates score as (matched_tokens / total_tokens)

#### Scenario: Cache miss handling
- **WHEN** no cache match found
- **THEN** system assigns lowest priority score

### Requirement: Fairness Guarantees
The system SHALL prevent request starvation regardless of scheduling policy.

#### Scenario: Maximum wait time
- **WHEN** request waits beyond configured threshold
- **THEN** system overrides policy and prioritizes waiting request

#### Scenario: Age-based boost
- **WHEN** request ages in queue
- **THEN** system gradually increases priority to ensure eventual processing

#### Scenario: Starvation metrics
- **WHEN** system processes requests
- **THEN** system tracks maximum and average wait times

### Requirement: Policy Metrics
The system SHALL track metrics for evaluating scheduling policy effectiveness.

#### Scenario: Track cache hit rate
- **WHEN** using cache-aware policy
- **THEN** system reports cache hit rate improvement vs. FCFS baseline

#### Scenario: Measure latency distribution
- **WHEN** policy processes requests
- **THEN** system tracks p50, p95, p99 latency percentiles

#### Scenario: Throughput measurement
- **WHEN** system operates under load
- **THEN** system reports requests per second and tokens per second

### Requirement: Policy Configuration
The system SHALL allow configuration of policy-specific parameters.

#### Scenario: Configure FCFS
- **WHEN** FCFS policy is selected
- **THEN** system accepts no additional parameters (simple queue)

#### Scenario: Configure LPM
- **WHEN** LPM policy is selected
- **THEN** system accepts cache weight and fairness threshold parameters

#### Scenario: Validate configuration
- **WHEN** user provides policy configuration
- **THEN** system validates parameters and raises error if invalid
