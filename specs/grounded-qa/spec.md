# ADDED Requirements

### Requirement: Bilingual grounded question answering
The system SHALL accept Chinese, English, and mixed-language questions and SHALL answer in the explicitly requested language or, when unspecified, the predominant language of the latest user turn. Every substantive factual claim MUST be supported by retrieved evidence from the active corpus.

#### Scenario: Chinese question uses English evidence
- **WHEN** a user asks in Chinese and sufficient supporting evidence is in English
- **THEN** the system SHALL answer in Chinese while preserving and citing the source evidence

#### Scenario: Requested information is not in the corpus
- **WHEN** retrieved evidence does not support the requested information
- **THEN** the system SHALL refuse or state the unsupported portion instead of answering from model memory

### Requirement: Evidence-only multi-turn context
The system SHALL maintain isolated conversation history for each session. It MAY use prior user turns to resolve references, but it MUST perform retrieval for every question and MUST NOT treat previous assistant answers as factual evidence.

#### Scenario: Resolve a follow-up question
- **WHEN** a user asks a follow-up whose subject is clear from the current session
- **THEN** the system SHALL form a standalone retrieval query and retrieve fresh evidence before answering

#### Scenario: Reset the session
- **WHEN** a user resets the conversation
- **THEN** subsequent questions SHALL NOT use turns from the reset session

### Requirement: Claim-level source citations
The system SHALL attach citations to the claims they support. Each citation MUST identify source title, document version, chunk ID, and an exact page number, section path, or text range available in the indexed metadata.

#### Scenario: Cite a PDF claim
- **WHEN** a factual claim is supported by a PDF chunk
- **THEN** the displayed citation SHALL contain the source title and corresponding page number

#### Scenario: Validate citation metadata
- **WHEN** a generated citation references an unknown chunk or locator
- **THEN** the system SHALL withhold the affected claim and MUST NOT invent a source or page

### Requirement: Grounding validation
Before release, the system SHALL validate that answer claims are entailed by cited retrieved chunks and that citations resolve to the active index revision. Unsupported claims MUST be removed, regenerated within the remaining deadline, or replaced by a refusal.

#### Scenario: All answer claims are supported
- **WHEN** every substantive claim is supported by one or more valid citations
- **THEN** the system SHALL release the answer with machine-readable and display citations

#### Scenario: A generated claim is unsupported
- **WHEN** grounding validation rejects a substantive claim
- **THEN** the system SHALL NOT expose that claim and SHALL return a supported partial answer or refusal

### Requirement: Appropriate refusal and partial answers
The system SHALL distinguish insufficient evidence, conflicting evidence, unsafe request, dependency failure, and deadline expiration. A refusal MUST be concise, localized, and MUST NOT imply that absent information is false.

#### Scenario: Evidence supports only part of a question
- **WHEN** retrieved evidence supports some but not all requested facts
- **THEN** the system SHALL answer the supported portion with citations and identify the unsupported portion

#### Scenario: Evidence materially conflicts
- **WHEN** active authoritative chunks conflict and metadata cannot resolve the conflict
- **THEN** the system SHALL describe the conflict with citations and MUST NOT choose an unsupported conclusion

### Requirement: Safe response contract
Every QA request SHALL return a structured outcome of `answer`, `refusal`, or `error`, including request ID, session ID, response language, citations, and safe diagnostics. Dynamic answer content MUST pass grounding and privacy checks before transport.

#### Scenario: QA succeeds
- **WHEN** retrieval, generation, grounding, citation, and privacy validation succeed
- **THEN** the response SHALL contain a grounded answer, valid citations, and safe request metadata

#### Scenario: A required stage fails
- **WHEN** retrieval, generation, grounding, citation, or privacy validation fails or times out
- **THEN** the system SHALL fail closed without returning pending unvalidated model content

### Requirement: Validated streaming
When streaming is enabled, the system SHALL buffer model output and emit only complete units that have passed citation, grounding, injection, and PII checks. It MUST NOT expose raw model token deltas.

#### Scenario: A complete sentence passes validation
- **WHEN** a generated sentence and its citation pass every release check
- **THEN** the system SHALL emit them atomically as a validated stream event

#### Scenario: A later sentence fails validation
- **WHEN** earlier validated sentences were emitted but a pending sentence fails a release check
- **THEN** the system SHALL discard the failing content and end with a safe localized limitation