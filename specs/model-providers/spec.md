# ADDED Requirements

### Requirement: Replaceable model-provider contracts
The system SHALL define independent asynchronous contracts for embedding, generation, and optional reranking. RAG services MUST depend on these contracts rather than vendor SDK types.

#### Scenario: Replace a compatible provider
→ WHEN a compatible implementation is selected through configuration
→ THEN the corresponding RAG stage SHALL use it without changes to the ingestion, retrieval, or QA API contracts

#### Scenario: Provider lacks a required capability
→ WHEN a provider is assigned to an unsupported role
→ THEN configuration validation SHALL fail before requests are accepted

### Requirement: OpenAI-compatible configuration
The RAG SHALL support configurable OpenAI-compatible embedding and chat-generation endpoints. It SHALL support optional litmus reranking through configured chat-generation model. Configuration MUST include base URL, model, deployment name, timeout, retry limit, secret reference without exposing credential values.

#### Scenario: Configure the official OpenAI API
→ WHEN a valid API key, base URL, and model names are provided
→ THEN the system SHALL initialize the configured embedding, generation, and optional reranking roles

#### Scenario: Configure another compatible endpoint
→ WHEN an endpoint implements the OpenAI-compatible contract
→ THEN the system SHALL normalize its responses through the same provider interfaces

#### Scenario: Required configuration is absent
→ WHEN an embedding or generation role lacks required configuration
→ THEN application readiness SHALL fail with a safe configuration error

### Requirement: Embedding integrity and identity
The embedding provider SHALL preserve input order and return exactly one finite, fixed-dimension vector for every input. Every vector space MUST be identified by provider, model, dimension, normalization policy, and adapter version.

#### Scenario: Embed a valid batch
→ WHEN the provider returns one compatible vector for every submitted text
→ THEN the system SHALL preserve input order and attach the configured embedding-space identity

#### Scenario: Embedding space is incompatible
→ WHEN retrieved stored embedding vectors do not match the request and active embedding identity
→ THEN the operation SHALL fail and MUST NOT persist or query with those vectors

### Requirement: Validated listwise reranking
The reranking provider SHALL receive a bounded query and candidate list with stable IDs and truncated text. It MUST return only duplicate-free ordering of supplied IDs; unknown, duplicates, or omitted IDs SHALL invalidate the reranking result.

#### Scenario: Reranking succeeds
→ WHEN the model returns a valid permutation of candidate IDs
→ THEN the system SHALL accept the order and record provider, model version, token usage, and latency

#### Scenario: Reranking output is invalid
→ WHEN the model introduces, duplicates, or omits candidate IDs
→ THEN the system SHALL reject that ranking and make the base retrieval order available for fallback

### Requirement: Bounded timeouts, retries, and cancellation
Every provider call SHALL have a per-attempt timeout and fit within the request’s remaining end-to-end deadline. Retries MUST be bounded and limited to configured transient failures; authentication and invalid-request failures MUST NOT be retried by default.

#### Scenario: A transient call fails
→ WHEN a provider returns a network error, timeout, rate limit, or server error and deadline remains
→ THEN the system SHALL perform no more than the configured retry count with bounded backoff

#### Scenario: A request can be expired
→ WHEN cancellation occurs or no deadline remains
→ THEN the active provider operation SHALL be cancelled and no new attempt SHALL start

### Requirement: Role-specific fallback behavior
Generation and reranking roles SHALL allow ordered configurable fallback routes. Embedding fallback MUST use the exact vector-space identity of the active index. Compatible failure SHALL degrade to the deterministic base ranking rather than fail the QA request.

#### Scenario: Generation fallback succeeds
→ WHEN the primary generation route fails with an eligible error and a fallback succeeds within the deadline
→ THEN the system SHALL record the fallback route and record every attempt

#### Scenario: No compatible embedding fallback exists
→ WHEN the primary embedding fails and available fallbacks use incompatible vector spaces
→ THEN the system SHALL fail querying or indexing without modifying the index

#### Scenario: All reranking routes fail
→ WHEN all reranking routes time out, fail, or return invalid output
→ THEN the system SHALL use base retrieval order and omit the base ranking

### Requirement: Usage and cost accounting
The system SHALL record per provider, route, model, provider model, latency, status, fallback indicator, and token counts. Diagnostics and logs MUST NOT contain API keys, authorization headers, unrestricted provider payloads, prompts, questions, answers, or document text. Cost SHALL be calculated only using verified pricing configuration.

#### Scenario: Primary fails and fallback succeeds
→ WHEN more than one provider attempt occurs for a logical operation
→ THEN usage SHALL account for all attempts, not only the successful one

#### Scenario: Usage or pricing is unavailable
→ WHEN a provider omits usage or no matching price is configured
→ THEN the system SHALL report the affected token or cost value as unknown

### Requirement: Safe provider diagnostics
The system SHALL expose readiness and safe error categories separately for embedding, generation, and reranking roles. Diagnostics and logs MUST NOT contain API keys, authorization headers, unrestricted provider payloads, prompts, questions, answers, or document text.

#### Scenario: Optional reranking is unavailable
→ WHEN embedding and generation are ready but reranking is not
→ THEN overall QA readiness SHALL remain true and diagnostics SHALL mark reranking unavailable

#### Scenario: Provider authentication fails
→ WHEN a provider rejects credentials
→ THEN diagnostics SHALL report a non-retriable authentication category without logging the credential or raw response body