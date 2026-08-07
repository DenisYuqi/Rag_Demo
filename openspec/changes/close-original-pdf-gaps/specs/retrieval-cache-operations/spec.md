## Purpose

Defines safe production retrieval caching that reduces repeated retrieval work without changing answers, crossing corpus versions, biasing official evaluations, or leaking query content.

## ADDED Requirements

### Requirement: Production retrieval cache behavior
When retrieval caching is enabled, the production retrieval path SHALL reuse a previously successful eligible result without invoking retrieval or reranking providers again. A cache hit SHALL return the same ordered chunk identifiers, scores, evidence metadata, and corpus revision as the cacheable uncached result, subject only to request-specific safe rendering.

#### Scenario: Repeated eligible retrieval hits cache
- **WHEN** two equivalent eligible retrieval requests use the same effective configuration and active corpus revision within the cache lifetime
- **THEN** the second request returns an equivalent result, records a cache hit, and does not invoke the underlying retrieval or reranking providers

#### Scenario: First retrieval fails
- **WHEN** retrieval or reranking fails, times out, or returns an invalid result
- **THEN** the failure is not cached and a later equivalent request may retry the underlying operation

### Requirement: Complete version-safe cache identity
Cache identity SHALL distinguish at least a privacy-safe digest of the normalized query, retrieval mode, result depth, hybrid weights, reranking state and model, relevant filters, embedding or provider configuration, and active corpus or index revision. Raw query text and supported PII MUST NOT be stored in cache keys or cache telemetry.

#### Scenario: Corpus revision changes
- **WHEN** ingestion activates a different corpus or index revision
- **THEN** requests cannot read results cached for the previous revision even if all other inputs match

#### Scenario: Retrieval configuration changes
- **WHEN** mode, depth, weights, filters, provider configuration, or reranking configuration differs
- **THEN** the request uses a distinct cache identity and cannot receive the incompatible cached result

### Requirement: Bounded lifetime and fail-open operation
The cache SHALL enforce configured time-to-live and capacity bounds. Expiration, eviction, corrupt entries, cache read errors, and cache write errors SHALL fall back to uncached retrieval without making an otherwise answerable QA request fail.

#### Scenario: Entry expires
- **WHEN** an eligible request matches an entry whose time-to-live has elapsed
- **THEN** the system records a miss, performs uncached retrieval, and may replace the expired entry after successful validation

#### Scenario: Cache backend raises an error
- **WHEN** cache access fails but the retrieval provider remains available
- **THEN** the request continues through uncached retrieval and emits a privacy-safe cache-error metric or diagnostic

### Requirement: Explicit cache bypass
The system SHALL provide a per-operation cache-bypass control used by official latency and controlled-comparison runs. A bypassed request SHALL neither read nor populate retrieval cache and SHALL record the bypass separately from hits and misses.

#### Scenario: Acceptance run bypasses cache
- **WHEN** an official acceptance candidate performs retrieval
- **THEN** the request invokes the configured retrieval path, records a bypass, and leaves existing cache contents and counters for eligible lookups unmodified except for the bypass counter

### Requirement: Observable cache rates
The system SHALL emit privacy-safe counts for eligible lookups, hits, misses, bypasses, evictions, expirations, and cache errors, with run and configuration correlation where applicable. Cache-hit rate SHALL use eligible lookups as its denominator and SHALL be unavailable when that denominator is zero.

#### Scenario: Mixed cache traffic is reported
- **WHEN** a reporting interval contains hits, misses, and bypasses
- **THEN** the operations evidence reports each count and calculates hit rate from hits divided by eligible lookups without treating bypasses as misses
