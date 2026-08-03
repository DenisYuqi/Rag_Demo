## ADDED Requirements

### Requirement: Supported document ingestion
The system SHALL accept PDF, Markdown, and UTF-8 plain-text documents through the API and Gradio workbench. It MUST validate extension, detected media type, configured size limit, and non-empty content before processing.

#### Scenario: Upload a supported document
- **WHEN** a user uploads a valid supported document within the configured size limit
- **THEN** the system SHALL create an ingestion job and return its stable job identifier

#### Scenario: Reject an invalid document
- **WHEN** a file is empty, oversized, malformed, media-type mismatched, encrypted without an available password, or unsupported
- **THEN** the system SHALL reject or fail the job with a safe structured reason and MUST NOT change the active indexes

### Requirement: Bilingual extraction and page-level OCR
The system SHALL preserve Chinese and English Unicode text. For each PDF page, it SHALL use native text extraction when usable text exists and SHALL invoke Chinese/English OCR when the page has insufficient native text according to a versioned threshold.

#### Scenario: Process a digital bilingual PDF
- **WHEN** a PDF page contains usable Chinese or English native text
- **THEN** the system SHALL preserve that text and page number without running OCR for that page

#### Scenario: Process a scanned PDF page
- **WHEN** a PDF page has insufficient usable native text
- **THEN** the system SHALL run OCR and associate recovered text with the original page number

#### Scenario: No usable text can be recovered
- **WHEN** neither native extraction nor OCR produces usable text for the document
- **THEN** the ingestion job SHALL fail and MUST NOT publish an empty document

### Requirement: Deterministic normalization and chunking
The system SHALL normalize content deterministically and split it into bounded, overlapping chunks using versioned configuration. Every chunk MUST have a stable identifier, content digest, source identifier, document version, ordinal, language-neutral text, and page, section, or character-range locator.

#### Scenario: Reprocess unchanged content
- **WHEN** the same canonical document is processed with the same extraction and chunking versions
- **THEN** the system SHALL produce identical chunk text, order, identifiers, and locators

#### Scenario: Chunk a PDF
- **WHEN** extracted PDF text crosses a page boundary
- **THEN** each resulting chunk SHALL remain attributable to one or more explicit page numbers

### Requirement: Deduplication and document versioning
The system SHALL calculate a digest from canonical document content. Re-uploading identical content for the same source key SHALL be an idempotent duplicate, while changed content for that source key SHALL create a monotonically increasing version.

#### Scenario: Upload an exact duplicate
- **WHEN** a source key, content digest, and derivation configuration match the active version
- **THEN** the system SHALL report a duplicate and MUST NOT call the embedding provider or modify an index

#### Scenario: Upload changed content
- **WHEN** canonical content changes for an existing source key
- **THEN** the system SHALL create the next document version while retaining the previous active version until publication succeeds

### Requirement: Atomic dense and lexical index publication
The system SHALL persist dense vectors in Chroma and lexical records in a BM25 index. It MUST publish an index revision only after both indexes contain the same validated active chunk set and compatible version metadata.

#### Scenario: Publish a successful ingestion
- **WHEN** extraction, chunking, embedding, dense indexing, lexical indexing, and parity validation succeed
- **THEN** the system SHALL atomically make the new index revision active

#### Scenario: Indexing fails part way through
- **WHEN** any embedding or index stage fails before publication
- **THEN** the staged revision SHALL remain inactive and the previously committed revision SHALL remain queryable 

### Requirement: Persistent ingestion status
Each ingestion job SHALL expose `queued`, `processing`, `succeeded`, or `failed` status plus safe stage diagnostics. Terminal job status, document metadata, and the active index manifest MUST survive an application restart.

#### Scenario: Inspect a completed job
- **WHEN** a user requests a terminal ingestion job
- **THEN** the system SHALL return its outcome, source ID, document version, OCR page count, chunk count, active index revision, stage timings, and safe warnings

#### Scenario: Restart with a valid index
- **WHEN** the application restarts with a valid persistent manifest and indexes
- **THEN** it SHALL expose the same active documents without requiring re-ingestion

### Requirement: Reindex and delete documents
The system SHALL support rebuilding a new index revision from retained active source artifacts and deleting a source from subsequent revisions. A change to embedding space, extraction rules, chunking rules, or BM25 tokenization MUST require reindexing rather than mixing incompatible data.

#### Scenario: Embedding model changes
- **WHEN** the configured embedding-space identity differs from the active manifest
- **THEN** the system SHALL report reindexing as required and MUST NOT append incompatible vectors

#### Scenario: Delete an indexed source
- **WHEN** a user confirms deletion and replacement index publication succeeds
- **THEN** the source SHALL no longer appear in active dense, lexical, or retrieval results

#### Scenario: Reindexing or deletion fails
- **WHEN** a replacement revision cannot be validated and committed
- **THEN** the prior valid revision SHALL remain active without partial changes