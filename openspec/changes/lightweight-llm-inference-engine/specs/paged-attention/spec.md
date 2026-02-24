## ADDED Requirements

### Requirement: Page-Based Allocation
The system SHALL allocate KV cache memory in fixed-size pages of 16 tokens each.

#### Scenario: Allocate new page
- **WHEN** sequence requires additional KV cache space
- **THEN** system allocates a new 16-token page from the memory pool

#### Scenario: Page size consistency
- **WHEN** system allocates any page
- **THEN** page MUST have exactly 16 token slots with shape [num_heads, 16, head_dim]

#### Scenario: Multiple pages per sequence
- **WHEN** sequence length exceeds 16 tokens
- **THEN** system allocates multiple pages to accommodate the full sequence

### Requirement: Memory Pool Management
The system SHALL maintain a pool of free pages and reuse pages when sequences complete.

#### Scenario: Initialize memory pool
- **WHEN** system starts with specified cache size
- **THEN** system pre-allocates pool of pages up to the cache size limit

#### Scenario: Allocate from pool
- **WHEN** new sequence requests pages
- **THEN** system provides pages from the free pool

#### Scenario: Return pages to pool
- **WHEN** sequence completes or is evicted
- **THEN** system returns all pages to the free pool for reuse

#### Scenario: Pool exhaustion
- **WHEN** free pool is empty and new allocation requested
- **THEN** system triggers eviction policy to free pages

### Requirement: Page Table Management
The system SHALL maintain a page table mapping logical token positions to physical page locations.

#### Scenario: Create page table entry
- **WHEN** sequence allocates a new page
- **THEN** system creates mapping from logical token range to physical page

#### Scenario: Lookup token position
- **WHEN** attention computation needs KV for specific token
- **THEN** system uses page table to find physical page containing that token

#### Scenario: Update page table
- **WHEN** sequence grows and allocates additional pages
- **THEN** system updates page table with new mappings

### Requirement: Page Sharing
The system SHALL support sharing pages across multiple sequences for prefix caching.

#### Scenario: Share prefix pages
- **WHEN** two sequences have identical prefix tokens
- **THEN** system allows both sequences to reference the same physical pages

#### Scenario: Reference counting
- **WHEN** multiple sequences share a page
- **THEN** system maintains reference count to prevent premature deallocation

#### Scenario: Copy-on-write
- **WHEN** shared page needs modification
- **THEN** system creates a copy before modification to preserve shared state

### Requirement: Attention Kernel Integration
The system SHALL integrate paged attention with the attention computation kernel.

#### Scenario: Gather KV from pages
- **WHEN** attention computation needs KV cache
- **THEN** system gathers keys and values from multiple pages using page table

#### Scenario: Scatter new KV to pages
- **WHEN** forward pass produces new key-value pairs
- **THEN** system scatters them to appropriate pages based on page table

#### Scenario: Handle page boundaries
- **WHEN** attention spans multiple pages
- **THEN** system correctly handles page boundaries without data loss

### Requirement: Memory Efficiency Metrics
The system SHALL track and report memory utilization metrics for paged attention.

#### Scenario: Report cache utilization
- **WHEN** user queries memory stats
- **THEN** system reports percentage of pages in use vs. total pool size

#### Scenario: Track fragmentation
- **WHEN** system has partially filled pages
- **THEN** system reports internal fragmentation (unused slots in allocated pages)

#### Scenario: Monitor page allocation rate
- **WHEN** system processes requests
- **THEN** system tracks pages allocated and deallocated per second

### Requirement: Eviction Policy
The system SHALL implement LRU eviction when memory pool is exhausted.

#### Scenario: LRU eviction
- **WHEN** pool is full and new allocation needed
- **THEN** system evicts least recently used pages that are not actively referenced

#### Scenario: Preserve active sequences
- **WHEN** eviction is triggered
- **THEN** system MUST NOT evict pages from currently executing sequences

#### Scenario: Eviction metrics
- **WHEN** eviction occurs
- **THEN** system logs eviction count and reclaimed page count
