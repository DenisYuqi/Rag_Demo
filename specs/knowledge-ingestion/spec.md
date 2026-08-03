# ADDED Requirements

### Requirement: Supported document ingestion
The system SHALL accept PDF, Markdown, and UTF-8 plain-text documents through the API and Gradio workbench. It MUST validate extension, uploaded type, configured size limits, and non-empty content before processing.

#### Scenario: Upload a supported document
→ WHEN a user uploads a valid supported document within the configured size limit
→ THEN the system SHALL create an ingestion job and return its stable job identifier

#### Scenario: Reject an invalid document
→ WHEN a file is empty, oversized, malformed, media-type mismatched, encrypted without an available password, or unsupported
→ THEN the system SHALL reject or fail the job with a safe structured reason and MUST NOT change the active indexes

### Requirement: Bilingual extraction and page-level OCR
The system SHALL preserve Chinese and English Unicode text. For each PDF page, it SHALL use native text extraction when available and SHALL invoke Chinese/English OCR when the page has insufficient native text according to a verified threshold.

#### Scenario: Process a digital bilingual PDF
→ WHEN a PDF page contains usable Chinese or English native text
→ THEN the system preserves that text and page number without running OCR for that page

#### Scenario: Process a scanned PDF page
→ WHEN a PDF page has insufficient usable native text
→ THEN the system SHALL run OCR and associate recovered text with the original page number

→ WHEN neither native extract nor OCR produces usable text for the document
→ THEN the ingestion job SHALL fail and MUST NOT publish an empty document

### Requirement: Deterministic normalization and chunking
The system SHALL normalize content deterministically and split it into bounded, overlapping chunks using versioned canonical extraction, detected digest, content segment, source identifier, document version, ordinal, language-neutral text, and page, section, or character-range locator.

#### Scenario: Reproduce extracted content
→ WHEN the same original document is processed with the same extraction and chunking versions
→ THEN the system SHALL produce identical chunk text, order, identifiers, and locators

#### Scenario: Chunk a PDF
→ WHEN extracted text crosses a page boundary
→ THEN each resulting chunk SHALL remain attributable to one or more explicit page numbers

### Requirement: Deduplication and document versioning
The system SHALL calculate a digest from extracted document content. Re-uploading identical content for the same source key SHALL be an independent duplicate, while changed content for the same source key SHALL create a monotonically increasing version.

#### Scenario: Upload an exact duplicate
→ WHEN a source key, content digest, and derivation configuration match the active version
→ THEN the system SHALL report a duplicate and MUST NOT call the embedding provider or modify an index

#### Scenario: Upload changed content for an existing source key
→ WHEN canonical content changes
→ THEN the system SHALL create the next document version while retaining the active version until publication success

### Requirement: Atomic dense and lexical index publication
The system SHALL persist document chunks in Chroma and lexical indexes in a DSET index. It MUST publish an index revision only after both indexes contain the same active chunk set and compatible version metadata.

#### Scenario: Publish a successful revision
→ WHEN extraction, chunking, embedding, dense index, lexical indexing, and parity validation succeed
→ THEN the new revision SHALL activate and the previously committed revision SHALL remain queryable

#### Scenario: Indexing fails mid-publication
→ WHEN any parity check fails before publication
→ THEN the system SHALL remain "processing", "succeeded", or "failed" plus safe state diagnostics; document metadata, dataset metadata, and the active index manifest MUST survive an application restart

#### Scenario: Inspect a completed job
→ WHEN a user queries a completed ingestion job
→ THEN the response SHALL include ingestion job ID, document version, OCR page count, chunk count, active revision

#### Scenario: Restart with a valid index
→ WHEN the service restarts with a valid persistent manifest and indexes
→ THEN it SHALL reuse the same active document versions without re-ingestion

### Requirement: Reindex and delete documents
The system SHALL support replacing a source revision from retained active source artifacts and deleting a source key. Deletion or reindexing SHALL be atomic; if a replacement revision cannot be validated and committed, the prior valid revision SHALL remain active without partial changes.

#### Scenario: Reindexing or deletion fails
→ WHEN a replacement revision cannot be validated and committed
→ THEN the prior valid revision SHALL remain active without partial changes