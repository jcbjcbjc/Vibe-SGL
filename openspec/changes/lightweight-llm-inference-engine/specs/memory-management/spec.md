## ADDED Requirements

### Requirement: Memory Pool Initialization
The system SHALL initialize a memory pool with configurable total cache size.

#### Scenario: Create memory pool
- **WHEN** system starts with specified cache size in bytes
- **THEN** system creates pool capable of holding that many bytes of KV cache

#### Scenario: Calculate page capacity
- **WHEN** pool is initialized
- **THEN** system calculates total number of pages based on page size and cache size

#### Scenario: Pre-allocate pages
- **WHEN** pool is created
- **THEN** system pre-allocates all pages to avoid runtime allocation overhead

### Requirement: Page Allocation
The system SHALL allocate pages from the pool on demand.

#### Scenario: Allocate single page
- **WHEN** request needs one page
- **THEN** system provides one free page from pool

#### Scenario: Allocate multiple pages
- **WHEN** request needs multiple pages
- **THEN** system provides requested number of pages atomically

#### Scenario: Allocation failure
- **WHEN** pool has insufficient free pages
- **THEN** system returns allocation failure and triggers eviction

### Requirement: Page Deallocation
The system SHALL return pages to the pool when no longer needed.

#### Scenario: Free single page
- **WHEN** page is no longer referenced
- **THEN** system returns page to free pool

#### Scenario: Free page batch
- **WHEN** sequence completes
- **THEN** system returns all pages used by sequence in batch

#### Scenario: Clear page data
- **WHEN** page is freed
- **THEN** system optionally clears page data for security (configurable)

### Requirement: Free List Management
The system SHALL maintain a free list of available pages for fast allocation.

#### Scenario: Initialize free list
- **WHEN** pool is created
- **THEN** system initializes free list with all pages

#### Scenario: Pop from free list
- **WHEN** allocation requested
- **THEN** system pops page from free list in O(1) time

#### Scenario: Push to free list
- **WHEN** page is freed
- **THEN** system pushes page to free list in O(1) time

### Requirement: Memory Accounting
The system SHALL track memory usage and provide statistics.

#### Scenario: Track allocated pages
- **WHEN** pages are allocated or freed
- **THEN** system maintains count of allocated vs. free pages

#### Scenario: Calculate memory usage
- **WHEN** user queries memory stats
- **THEN** system reports bytes used, bytes free, and utilization percentage

#### Scenario: Track peak usage
- **WHEN** system operates
- **THEN** system tracks peak memory usage for monitoring

### Requirement: Page Metadata
The system SHALL maintain metadata for each page.

#### Scenario: Store page ID
- **WHEN** page is allocated
- **THEN** system assigns unique page ID for tracking

#### Scenario: Track page owner
- **WHEN** page is allocated to sequence
- **THEN** system records which sequence owns the page

#### Scenario: Reference counting
- **WHEN** page is shared
- **THEN** system maintains reference count in page metadata

### Requirement: Memory Limits
The system SHALL enforce memory limits and prevent over-allocation.

#### Scenario: Enforce cache size limit
- **WHEN** allocation would exceed cache size
- **THEN** system rejects allocation and triggers eviction

#### Scenario: Per-request memory limit
- **WHEN** single request exceeds per-request limit
- **THEN** system rejects request with OOM error

#### Scenario: Reserve memory
- **WHEN** system operates
- **THEN** system reserves small buffer for critical operations

### Requirement: Allocator Interface
The system SHALL provide a clean allocator interface for KV cache management.

#### Scenario: Allocate interface
- **WHEN** component needs pages
- **THEN** system provides allocate(num_pages) method returning page IDs

#### Scenario: Free interface
- **WHEN** component releases pages
- **THEN** system provides free(page_ids) method accepting list of page IDs

#### Scenario: Query interface
- **WHEN** component needs memory info
- **THEN** system provides get_stats() method returning usage statistics

### Requirement: Thread Safety
The system SHALL ensure thread-safe memory pool operations.

#### Scenario: Concurrent allocation
- **WHEN** multiple threads allocate simultaneously
- **THEN** system uses locking to prevent race conditions

#### Scenario: Concurrent deallocation
- **WHEN** multiple threads free pages simultaneously
- **THEN** system safely returns pages to pool without corruption

#### Scenario: Lock-free fast path
- **WHEN** possible
- **THEN** system uses lock-free operations for common allocation patterns

### Requirement: Memory Defragmentation
The system SHALL handle memory fragmentation gracefully.

#### Scenario: No external fragmentation
- **WHEN** pages are allocated and freed
- **THEN** system has no external fragmentation (all free pages usable)

#### Scenario: Internal fragmentation tracking
- **WHEN** pages are partially filled
- **THEN** system tracks internal fragmentation (unused slots in pages)

#### Scenario: Fragmentation metrics
- **WHEN** user queries stats
- **THEN** system reports fragmentation percentage
