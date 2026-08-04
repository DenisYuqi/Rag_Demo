## ADDED Requirements

### Requirement: Single-instance performance acceptance
The system SHALL provide an end-to-end QA load test measured from request submission until the complete answer, refusal, or error response is received. The acceptance workload MUST use exactly one warm application instance, maintain at least five concurrent requests, bypass answer and retrieval-result caches, and include at least 500 successful requests.

#### Scenario: Performance targets pass
- **WHEN** a valid acceptance run completes against one warm instance
- **THEN** it SHALL pass only if nearest-rank P90 end-to-end latency is at most 10 seconds and total error rate is below 1 percent

#### Scenario: Acceptance workload is invalid
- **WHEN** fewer than five concurrent requests are maintained, multiple instances serve traffic, caches satisfy measured requests, or fewer than 500 requests succeed
- **THEN** the run SHALL be marked invalid and MUST NOT be reported as passing

#### Scenario: A request fails and is retried
- **WHEN** an attempt times out, is incomplete, or returns an unsuccessful response
- **THEN** that attempt SHALL remain in the error-rate denominator even if another attempt later succeeds

### Requirement: Warm-instance definition
A warm instance SHALL have completed startup, opened and validated persistent indexes, initialized required model and safety clients, passed readiness, and completed documented warm-up requests. Warm-up traffic MUST be excluded from measured latency and reported separately.

#### Scenario: Begin a measured run
- **WHEN** readiness and configured warm-up requests complete
- **THEN** the load runner MAY begin measurement and SHALL record excluded warm-up count and duration

### Requirement: Bounded concurrency and deadlines
The application SHALL allow at least five QA pipelines to execute concurrently on one instance, enforce a configurable admission limit and total deadline, and reject excess queued work before unbounded queue delay can invalidate the latency target.

#### Scenario: Five requests arrive together
- **WHEN** five admissible QA requests are submitted concurrently
- **THEN** the service SHALL process them without serializing the full pipelines behind one global request lock

#### Scenario: Capacity and queue are exhausted
- **WHEN** active and queued requests reach configured bounds
- **THEN** the service SHALL return a retryable capacity response rather than accept unbounded waiting

#### Scenario: End-to-end deadline expires
- **WHEN** the total QA deadline is reached
- **THEN** unfinished downstream work SHALL be cancelled and the response SHALL fail closed with a safe timeout outcome

### Requirement: Stage latency budget and degradation
The system SHALL measure queue, validation, embedding, dense retrieval, BM25, fusion, reranking, generation, grounding, redaction, and serialization time. Optional reranking MUST have a bounded sub-deadline and SHALL degrade to base ranking when it cannot complete in budget.

#### Scenario: Reranker exceeds its budget
- **WHEN** reranking does not complete within its configured sub-deadline
- **THEN** the system SHALL cancel it, continue with base ranking, and record a degraded reason

#### Scenario: Required generation cannot fit the deadline
- **WHEN** the remaining deadline is insufficient for a validated answer
- **THEN** the system SHALL stop and return a safe timeout rather than emit incomplete unvalidated output

### Requirement: Privacy-safe structured logs
The system SHALL emit JSON logs with timestamp, level, service and config versions, event name, request ID, trace ID, operation, outcome, safe error category, stage duration, cache status, counts, model identity, token usage, estimated cost, and redaction metadata where applicable. Content fields prohibited by the privacy specification MUST NOT be logged.

#### Scenario: QA completes successfully
- **WHEN** a request completes
- **THEN** correlated request and stage events SHALL be emitted with timings and outcomes but without raw user, model, or document content

#### Scenario: QA fails
- **WHEN** a pipeline stage fails
- **THEN** the system SHALL log a stable safe error category and correlation identifiers without unrestricted exception content

### Requirement: Operational metrics
The system SHALL expose bounded-cardinality counters, gauges, and histograms for request volume, outcomes, errors, in-flight work, queue rejection, end-to-end and stage latency, cache outcomes, retrieval counts, degradation, tokens, and estimated cost. Request IDs, questions, answers, and document text MUST NOT be metric labels.

#### Scenario: Requests execute
- **WHEN** QA traffic is processed
- **THEN** metrics SHALL support calculating P50, P90, P95, throughput, error rate, stage bottlenecks, token usage, and cost

### Requirement: Correlated traces
The system SHALL create a root trace for every QA request and child spans for major RAG stages. Asynchronous and provider calls MUST preserve trace context, and spans SHALL contain only privacy-safe metadata.

#### Scenario: Inspect a slow request
- **WHEN** a request exceeds a diagnostic threshold
- **THEN** its trace SHALL identify queue and stage durations, provider/model aliases, token counts, degradation, and safe errors using the same request correlation

### Requirement: Token-cost estimate
The system SHALL aggregate provider-reported or explicitly estimated token usage and calculate cost from versioned provider/model pricing. It SHALL produce a cost estimate per request and per 1,000 QA calls with assumptions and unknown values identified.

#### Scenario: Price and token usage are known
- **WHEN** an evaluation run has complete usage and matching versioned pricing
- **THEN** its report SHALL include input, output, reranking, and embedding token totals and estimated cost per 1,000 calls

#### Scenario: Usage or pricing is incomplete
- **WHEN** required usage or pricing is unavailable
- **THEN** the system SHALL mark the estimate incomplete and MUST NOT represent missing cost as zero

### Requirement: Performance evidence bundle
Each acceptance run SHALL write a privacy-safe machine-readable evidence bundle containing run ID, code and configuration versions, timestamps, instance count, warm-up details, concurrency, cache policy, attempt and success counts, error count, latency percentiles, stage summaries, token/cost totals, thresholds, and final validity and pass status.

#### Scenario: Acceptance run finishes
- **WHEN** load generation completes
- **THEN** the evidence bundle SHALL contain sufficient correlated metric, log, and representative trace references to reproduce the pass or failure decision
