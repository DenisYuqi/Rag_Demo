# ADDED Requirements

### Requirement: Configurable retrieval modes
The system SHALL support `dense`, `hybrid`, and `hybrid-rerank` retrieval modes with `hybrid-rerank` as the configurable default. Each request MUST bind to one committed index revision and use only active chunks from that revision.

#### Scenario: Execute dense mode
→ WHEN a request selects `dense`
→ THEN the system SHALL perform compatible query embedding and Chroma search without BM25 or reranking

#### Scenario: Execute hybrid mode
→ WHEN a request selects `hybrid`
→ THEN the system SHALL execute dense and BM25 retrieval and fuse their candidate rankings

#### Scenario: Execute hybrid reranking mode
→ WHEN a request selects `hybrid-rerank`
→ THEN the system SHALL fuse dense and BM25 results and attempt reranking of a bounded candidate set

### Requirement: Bilingual dense and lexical retrieval
Dense retrieval SHALL use an embedding-space identity compatible with the active Chroma collection. BM25 retrieval SHALL use a versioned tokenizer that supports Latin token boundaries and contiguous Chinese text.

#### Scenario: Retrieve cross-language semantic evidence
→ WHEN a Chinese or English query is semantically related to an indexed chunk
→ THEN dense retrieval SHALL make that chunk eligible regardless of exact term overlap

#### Scenario: Retrieve Chinese lexical evidence
→ WHEN a Chinese query contains an exact term present in an indexed chunk without whitespace boundaries
→ THEN the verified BM25 tokenizer SHALL make that chunk eligible

#### Scenario: Query embedding is incompatible
→ WHEN the configured query embedding identity differs from the active index identity
→ THEN dense retriever SHALL fail without submitting an incompatible vector to Chroma

### Requirement: Parallel hybrid candidate collection
In hybrid modes, dense and BM25 retrieval SHALL execute independently and may run concurrently. Candidates MUST be indexed by stable chunk ID while preserving each retriever’s rank and raw score.

#### Scenario: The same chunk appears in both result sets
→ WHEN dense and BM25 retrieval return the same chunk ID
→ THEN the system SHALL retain one candidate with both rank contributions

#### Scenario: One retriever returns no matches
→ WHEN one retriever succeeds with an empty list and the other returns candidates
→ THEN the system SHALL continue with the available candidates without treating the empty result as a failure

### Requirement: Reciprocal Rank Fusion
Hybrid retrieval SHALL combine rankings using versioned weighted Reciprocal Rank Fusion. It MUST use ranks rather than raw scores, add discoverable raw scores and SHALL apply stable tie-breaking by best individual rank and chunk ID.

#### Scenario: Candidate appears in both rankings
→ WHEN a candidate has dense and BM25 ranks
→ THEN its RRF score SHALL contain a contribution from each ranking using configured positive weights and constant

#### Scenario: Fusion scores tie
→ WHEN candidates have equal RRF score and best individual rank
→ THEN the system SHALL order them deterministically by stable chunk ID

### Requirement: Bounded optional reranking
Reranking SHALL receive no more than the configured candidate list, and candidate text SHALL be truncated to a configured token limit. A successful validated reranking SHALL determine the final top results.

#### Scenario: Reranking succeeds within budget
→ WHEN a valid reranking runs inside its duration identity, token usage, and latency in diagnostics
→ THEN the system SHALL apply it and expose reranker identity, token usage, and latency in diagnostics

#### Scenario: Reranking fails or exceeds budget
→ WHEN reranking times out, fails, or returns invalid output
→ THEN rely upon the deterministic RRF order, flag the request as rerank-degraded, and MUST NOT retry beyond the deadline budget

### Requirement: Bounded retrieved results and evidence
The system SHALL respect minimum and final-result limits. Every final result MUST include used chunk ID, chunk text, chunk offset, display title, final rank, token counts, and explicable dense, BM25, and reranking scores.

#### Scenario: More candidates exist than requested
→ WHEN RRF produces more than the configured final result count
→ THEN the system SHALL return only the top results and report full stage counts in diagnostics

#### Scenario: Evidence originates from a PDF
→ WHEN a returned chunk originates from a PDF
→ THEN its evidence metadata SHALL include at least one valid page number

### Requirement: Version-safe retrieval caching
Query embedding, retrieval, and reranking caches SHALL use keys that include canonical query, active index revision, retrieval mode, model identities, ranking configuration, prompt version, and relevant limits. Failed, cancelled, or degraded results MUST NOT be stored as successful final results.

#### Scenario: Repeat an identical compatible request
→ WHEN a valid uncached entry exactly matches query and configuration versions
→ THEN the system SHALL return it without repeating the cached operation and SHALL mark the cache hit

#### Scenario: Corpus or ranking configuration changes
→ WHEN the index revision, embedding identity, BM25 version, RRF settings, or reranker version changes
→ THEN the prior cache entry MUST NOT satisfy the new request

### Requirement: Retrieval diagnostics and safe degradation
Every retrieval response SHALL report request ID, effective mode, index revision, candidate counts, stage timings, cache status, provider identities, and degradation reasons without exposing credentials or hidden content. A hybrid request MAY continue with one successful retriever when configured; failure of both retrievers MUST fail the request.

#### Scenario: One hybrid retriever fails
→ WHEN exactly one retriever fails and degraded hybrid behavior is enabled
→ THEN the system SHALL rank available candidates, mark the failed stage, and MUST NOT cache the degraded final result

#### Scenario: Both retrievers fail
→ WHEN dense and BM25 retrieval both fail
→ THEN the system SHALL return a structured retrieval-unavailable error without stale or fabricated evidence

#### Scenario: No valid index is active
→ WHEN retrieval starts without a valid committed index revision
→ THEN the system SHALL return an index-not-ready error and no candidates