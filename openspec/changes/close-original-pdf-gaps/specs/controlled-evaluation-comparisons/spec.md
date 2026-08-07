## Purpose

Defines controlled, reproducible experiments that compare model versions and retrieval strategies on identical evidence and turn the measured trade-offs into auditable production recommendations.

## ADDED Requirements

### Requirement: Immutable comparison plan
Every comparison SHALL persist an immutable plan before execution containing its candidate configurations, baseline, dataset and corpus versions, scorer and prompt versions, retrieval settings, generation settings, cache policy, timeouts, concurrency, pricing version, repetition policy, and predeclared decision criteria. Candidate results SHALL be rejected as incompatible if any controlled dimension differs unexpectedly.

#### Scenario: Compatible candidates are executed
- **WHEN** all candidates share the controlled dimensions declared by the plan
- **THEN** the system assigns a comparison identifier and records each candidate under that plan with reproducible configuration provenance

#### Scenario: Incompatible runs are selected
- **WHEN** an operator attempts to compare runs whose controlled dimensions do not match
- **THEN** the system refuses the comparison and lists the incompatible dimensions without fabricating deltas

### Requirement: Model-version comparison
A model-selection comparison SHALL evaluate the current production generation model and at least one exact configured alternative version while holding the dataset, corpus, retrieval strategy, prompt, scorers, concurrency, and cache policy fixed. It SHALL report per candidate and relative to baseline the advanced quality metrics, all-attempt p50/p90/p95 latency, successful-only latency, input/output tokens, cost per 1,000 attempts, error and timeout rates, degradation behavior, and provider/model identifiers.

#### Scenario: Model comparison completes
- **WHEN** all planned model candidates reach a terminal state
- **THEN** the comparison presents aligned absolute values and baseline deltas and records a quality/cost/latency selection rationale derived from the predeclared criteria

#### Scenario: A model candidate partially fails
- **WHEN** a candidate returns errors or timeouts for some cases
- **THEN** those attempts remain in its denominators and the comparison exposes the failure rate rather than dropping the candidate's failed observations

### Requirement: Retrieval-strategy comparison
A retrieval comparison SHALL evaluate `dense`, `hybrid`, and `hybrid-rerank` using the selected generation model and identical dataset, corpus revision, prompts, scorers, candidate depths, timeouts, and official cache-bypass policy. It SHALL report retrieval and answer quality, all-attempt p50/p90/p95 latency, token and cost measures, error/degradation rates, and tagged challenge-category results for every strategy.

#### Scenario: Three-strategy comparison completes
- **WHEN** all three required retrieval candidates finish
- **THEN** the comparison shows absolute results and baseline deltas for every required metric and challenge category under one compatible plan and proves that at least one eligible rerank-sensitive case invoked the configured reranker and changed pre-rerank ordering or selected context

#### Scenario: Reranking is configured but not exercised
- **WHEN** no eligible case invokes the reranking provider or no rerank-sensitive case changes ordering or selected context
- **THEN** the comparison is marked non-discriminating and cannot support a recommendation to enable reranking

#### Scenario: Reranking is not justified
- **WHEN** `hybrid-rerank` fails the plan's predeclared minimum quality benefit or causes all-attempt p90 latency to exceed 10 seconds
- **THEN** the recommendation keeps reranking disabled and states the measured reason

### Requirement: Cache benefit is measured separately
Official model and retrieval acceptance comparisons SHALL bypass retrieval cache so that cached responses cannot improve the service-level result. A separate cache experiment SHALL compare cold and warm eligible traffic with the same corpus revision and SHALL report hit rate, latency delta, provider-call delta, and output-equivalence checks.

#### Scenario: Official comparison bypasses cache
- **WHEN** a model or retrieval acceptance candidate runs
- **THEN** its provenance records cache bypass and its official latency metrics contain no cache hits

#### Scenario: Warm-cache experiment completes
- **WHEN** the same eligible workload is executed cold and then warm
- **THEN** the report quantifies cache behavior separately and confirms that cached and uncached retrieval outputs are equivalent for the same configuration and corpus revision

### Requirement: Deterministic recommendation and preserved history
The system SHALL derive the recommended production model and retrieval configuration from the plan's predeclared gates and tie-break rules, preserve both successful and failed comparison histories, and record the reason when no candidate is acceptable. Reopening a completed comparison SHALL read persisted evidence rather than rerunning provider calls.

#### Scenario: One candidate satisfies the decision policy
- **WHEN** a completed comparison has at least one candidate that satisfies all mandatory gates
- **THEN** the system marks exactly one recommendation using the declared tie-break rules and records a human-readable rationale linked to measured values

#### Scenario: No candidate satisfies mandatory gates
- **WHEN** every candidate violates at least one mandatory gate
- **THEN** the comparison records no recommendation, identifies each failed gate, and remains available for diagnosis and download
