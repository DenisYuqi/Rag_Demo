# ADDED Requirements

### Requirement: Bilingual grounded question answering
3 The system SHALL accept Chinese, English, and mixed-language questions and SHALL answer in the explicitly requested language or, when unspecified, the predominant language of the latest user turn. Every substantive factual claim MUST be supported by retrieved evidence from the active corpus.

#### Scenario: Chinese question uses English evidence
5 → WHEN a user asks in Chinese and sufficient supporting evidence is in English
7 → THEN the system SHALL answer in Chinese while preserving and citing the source evidence

#### Scenario: Requested information is not in the corpus
9 → WHEN retrieved evidence does not support the requested information
11 → THEN the system SHALL refuse or state the unsupported portion instead of answering from model memory

### Requirement: Evidence-only multi-turn context
13 The system SHALL maintain isolated conversation history for each session. It MAY use prior user turns to resolve references, but it MUST perform retrieval for every question and MUST NOT treat previous assistant answers as factual evidence.

#### Scenario: Resolve a follow-up question
16 → WHEN a user asks a follow-up whose subject is clear from the current session
18 → THEN the system SHALL form a standalone retrieval query and retrieve fresh evidence before answering

#### Scenario: Reset the session
20 → WHEN a user resets the conversation
22 → THEN subsequent questions SHALL NOT use turns from the reset session

### Requirement: Claim-level source citations
24 The system SHALL attach citations to the claims they support. Each citation MUST identify source title, document version, chunk ID, and an exact page number, section path, or text range available in the indexed metadata.

#### Scenario: Cite a PDF claim
27 → WHEN a factual claim is supported by a PDF chunk
29 → THEN the displayed citation SHALL contain the source title and corresponding page number

#### Scenario: Validate citation metadata
32 → WHEN a generated citation references an unknown chunk or locator
34 → THEN the system SHALL withhold the affected claim and MUST NOT invent a source or page

### Requirement: Grounding validation
36 Before release, the system SHALL validate that every answer claim are entailed by retrieved chunks and that citations resolve to the active index revision. Unsupported claims MUST be removed, regenerated within the remaining deadline, or replaced by a refusal.

#### Scenario: All answer claims are supported
38 → WHEN every substantive claim is supported by one or more valid citations
41 → THEN the system SHALL release the answer with machine-readable and display citations

#### Scenario: A generated claim is unsupported
43 → WHEN grounding validation rejects a substantive claim
46 → THEN the system SHALL not expose that claim, and SHALL return a supported partial answer or refusal

### Requirement: Appropriate refusal and partial answers
47 The system SHALL distinguish insufficient evidence, conflicting evidence, unsafe request, dependency failure, and deadline expiration. A refusal MUST be concise, localized, and MUST NOT imply that absent information is false.

#### Scenario: Evidence supports only part of a question
50 → WHEN retrieved evidence supports some but not all requested facts
52 → THEN the system SHALL answer the supported portion with citations and identify the unsupported portion

#### Scenario: Evidence materially conflicts
54 → WHEN authoritative chunks conflict and metadata cannot resolve the conflict
57 → THEN the system SHALL describe the conflict with citations and MUST NOT choose an unsupported conclusion

### Requirement: Safe response contract
59 Every QA request SHALL return a structured outcome of "answer", "refusal", or "error", including request ID, session ID, response language, citations, and safe diagnostics. Dynamic answer content MUST pass grounding and privacy checks before transport.

#### Scenario: QA succeeds
60 → WHEN retrieval, generation, grounding, citation, and privacy validation succeed
63 → THEN the response SHALL contain a grounded answer, valid citations, and safe request metadata

#### Scenario: A required stage fails
66 → WHEN retrieval, generation, grounding, citation, or privacy validation fails or times out
68 → THEN the system SHALL fail closed without returning pending unvalidated model content

### Requirement: Validated streaming
70 When streaming is enabled, the system SHALL buffer model output and emit only complete units that have passed citation, grounding, injection, and PII checks. It MUST NOT expose raw model token deltas.

#### Scenario: A complete sentence passes validation
72 → WHEN a generated sentence and its citation pass every release check
74 → THEN the system SHALL emit it atomically as a validated stream event

#### Scenario: A later sentence fails validation
76 → WHEN earlier validated sentences were emitted but a pending sentence fails a release check
78 → THEN the system SHALL discard the failing content and end with a safe localized limitation