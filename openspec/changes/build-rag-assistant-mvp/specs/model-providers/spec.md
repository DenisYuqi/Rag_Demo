## ADDED Requirements

### Requirement: Replaceable model-provider contracts
The system SHALL define independent asynchronous contracts for embedding, generation, and optional reranking. RAG services MUST depend on these contracts rather than vendor SDK types.

#### Scenario: Replace a compatible provider
- **WHEN** a compatible implementation is selected through configuration
- **THEN** the corresponding RAG stage SHALL use it without changes to the ingestion, retrieval, or QA API contracts

#### Scenario: Provider lacks a required capability
- **WHEN** a provider is assigned to an unsupported role
- **THEN** configuration validation SHALL fail before requests are accepted

### Requirement: OpenAI-compatible configuration
The MVP SHALL support configurable OpenAI-compatible embedding and chat-generation endpoints. It SHALL support optional listwise reranking through a configured chat-generation model. Configuration MUST include base URL, model or deployment name, timeout, retry limit, and secret reference without exposing credential values.

#### Scenario: Configure the official OpenAI API
- **WHEN** a valid API key, base URL, and model names are provided
- **THEN** the system SHALL initialize the configured embedding, generation, and optional reranking roles

#### Scenario: Configure another compatible endpoint
- **WHEN** an endpoint implements the required OpenAI-compatible contract
- **THEN** the system SHALL normalize its responses through the same provider interfaces

#### Scenario: Required configuration is absent
- **WHEN** an embedding or generation role lacks required configuration
- **THEN** application readiness SHALL fail with a safe configuration error

### Requirement: Embedding integrity and identity
The embedding provider SHALL preserve input order and return exactly one finite, fixed-dimension vector for every input. Every vector space MUST be identified by provider, model, dimension, normalization policy, and adapter version.

#### Scenario: Embed a valid batch
- **WHEN** the provider returns one compatible vector for every submitted text
- **THEN** the system SHALL preserve input order and attach the configured embedding-space identity

#### Scenario: Embedding response is incompatible
- **WHEN** vector count, dimension, or numeric validity does not match the request and active embedding identity
- **THEN** the operation SHALL fail and MUST NOT persist or query with those vectors

### Requirement: Validated listwise reranking
The reranking provider SHALL receive a bounded query and candidate list with stable IDs and truncated text. It MUST return only a duplicate-free ordering of supplied IDs; unknown, duplicate, or omitted IDs SHALL invalidate the reranking result.

#### Scenario: Reranking succeeds
- **WHEN** the model returns a valid permutation of candidate IDs
- **THEN** the system SHALL accept the order and record provider, model, prompt version, token usage, and latency

#### Scenario: Reranking output is invalid
- **WHEN** the model introduces, duplicates, or omits candidate IDs
- **THEN** the system SHALL reject that ranking and make the base retrieval order available for fallback

### Requirement: Bounded timeouts, retries, and cancellation
Every provider call SHALL have a per-attempt timeout and fit within the request's remaining end-to-end deadline. Retries MUST be bounded and limited to configured transient failures; authentication and invalid-request failures MUST NOT be retried by default.

#### Scenario: A transient call fails
- **WHEN** a provider returns a network error, timeout, rate limit, or server error and deadline remains
- **THEN** the system SHALL perform no more than the configured retry count with bounded backoff

#### Scenario: A request is cancelled or expires
- **WHEN** cancellation occurs or no deadline remains
- **THEN** the active provider operation SHALL be cancelled and no new attempt SHALL start

### Requirement: Role-specific fallback behavior
Generation and reranking roles MAY configure ordered compatible fallback routes; one route per required role is sufficient for the MVP. If an embedding fallback is configured, it MUST use the exact vector-space identity of the active index. Reranking failure SHALL degrade to the deterministic base ranking rather than fail the QA request.

#### Scenario: Generation fallback succeeds
- **WHEN** the primary generation route fails with an eligible error and a fallback succeeds within the deadline
- **THEN** the system SHALL return the fallback result and record every attempt

#### Scenario: No compatible embedding fallback exists
- **WHEN** the primary embedding route fails and available fallbacks use incompatible vector spaces
- **THEN** the embedding operation SHALL fail without querying or modifying the index

#### Scenario: All reranking routes fail
- **WHEN** reranking routes time out, fail, or return invalid output
- **THEN** the caller SHALL receive a degraded status and retain the base ranking

### Requirement: Usage and cost accounting
The system SHALL record each provider attempt's role, provider, model, latency, status, fallback indicator, and provider-reported input/output tokens. Missing usage MUST be represented as unknown, not zero. Cost SHALL be calculated only from a versioned pricing configuration.

#### Scenario: Primary fails and fallback succeeds
- **WHEN** more than one provider attempt occurs for a logical operation
- **THEN** usage and estimated cost SHALL include all attempts, not only the successful one

#### Scenario: Usage or pricing is unavailable
- **WHEN** a provider omits usage or no matching price is configured
- **THEN** the system SHALL report the affected token or cost value as unknown

### Requirement: Safe provider diagnostics
The system SHALL expose readiness and safe error categories separately for embedding, generation, and reranking roles. Diagnostics and logs MUST NOT contain API keys, authorization headers, unrestricted provider payloads, prompts, questions, answers, or document text.

#### Scenario: Optional reranking is unavailable
- **WHEN** embedding and generation are ready but reranking is not
- **THEN** overall QA readiness SHALL remain true and diagnostics SHALL mark reranking unavailable

#### Scenario: Provider authentication fails
- **WHEN** a provider rejects credentials
- **THEN** diagnostics SHALL report a non-retriable authentication category without logging the credential or raw response body
