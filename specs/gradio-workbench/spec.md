## ADDED Requirements

#### Requirement: FastAPI-mounted Gradio workbench
The system SHALL mount one Gradio application on the FastAPI service at a configurable path and SHALL use the same application services as the HTTP APIs. Health, metrics, and API routes MUST remain independently reachable.

#### Scenario: Start the application
- **WHEN** the FastAPI application becomes ready with the workbench enabled
- **THEN** the Gradio workbench and non-UI API routes SHALL be reachable from the same server process

#### Scenario: UI mounting fails
- **WHEN** the configured path conflicts or Gradio initialization fails
- **THEN** readiness SHALL fail with a safe configuration error rather than silently selecting another path

#### Requirement: Product-oriented primary views
The workbench SHALL provide `Chat`, `Documents`, `Evaluation`, and `Diagnostics` tabs. It MUST present clear loading, success, refusal, empty, and error states in both Chinese and English content flows.

#### Scenario: Open the workbench
- **WHEN** a user loads the workbench
- **THEN** all four tabs SHALL be available and 'Chat' SHALL be the initial view

#### Scenario: A backend capability is unavailable
- **WHEN** a tab depends on an unavailable service
- **THEN** that tab SHALL display a safe unavailable state and MUST NOT bypass the shared backend pipeline

#### Requirement: Chat experience
The Chat tab SHALL support multi-turn questions, retrieval-mode selection, session reset, request cancellation, validated answer streaming, visible citations, source previews, and distinct refusal messages.

#### Scenario: Ask a supported question
- **WHEN** the QA pipeline returns a grounded answer
- **THEN** the Chat tab SHALL display the answer, inline citation markers, and expandable redacted source previews

#### Scenario: Receive a refusal
- **WHEN** the QA pipeline returns a refusal
- **THEN** the Chat tab SHALL distinguish it from a system error and SHALL NOT replace it with an ungrounded response

#### Scenario: Cancel an active answer
- **WHEN** a user cancels an in-progress request
- **THEN** the workbench SHALL stop consuming events and SHALL NOT display pending unvalidated text

### Requirement: Document management experience
The Documents tab SHALL allow a user to upload supported documents, inspect ingestion progress and metadata, reindex active sources, and confirm deletion. It MUST NOT display unrestricted raw document contents.

#### Scenario: Upload and index a document
- **WHEN** an ingestion job succeeds
- **THEN** the tab SHALL show source ID, version, type, OCR usage, chunk count, and active index revision

#### Scenario: Delete a document
- **WHEN** a user confirms deletion
- **THEN** the tab SHALL report completion only after a replacement active index excludes the source

### Requirement: Evaluation experience
The Evaluation tab SHALL let a user select a versioned dataset, run an evaluation or compare two compatible runs, monitor progress, inspect failed cases, and download JSON and HTML reports.

#### Scenario: Run an evaluation
- **WHEN** a valid dataset and configuration are selected
- **THEN** the tab SHALL execute cases through the production RAG pipeline and display progress and aggregate metrics

#### Scenario: Compare incompatible runs
- **WHEN** selected runs use incompatible dataset versions or eligible case sets
- **THEN** the tab SHALL reject the improvement comparison and explain the compatibility failure

### Requirement: Privacy-safe diagnostics experience
The Diagnostics tab SHALL display health, non-secret configuration versions, stage timings, cache outcomes, model identities, token usage, estimated cost, request outcomes, redacted errors, and trace IDs. It MUST NOT display credentials, full prompts, raw model payloads, unrestricted chunks, or unredacted PII.

#### Scenario: Inspect a request
- **WHEN** a user enters a valid recent request or trace ID
- **THEN** the tab SHALL show correlated safe stage evidence and redacted source metadata

#### Scenario: Diagnostic data contains a forbidden field
- **WHEN** a diagnostic record contains a secret or non-allowlisted content field
- **THEN** the workbench SHALL omit or redact it before display

### Requirement: Session isolation and safe failures
Each browser session SHALL have isolated chat and mutable UI state. Unexpected backend exceptions or malformed stream events MUST result in a safe message with a request or trace ID and MUST NOT expose stack traces or raw content.

#### Scenario: Two users interact concurrently
- **WHEN** separate browser sessions submit questions at the same time
- **THEN** each SHALL see only its own conversation and evaluation state

#### Scenario: A malformed event reaches the UI adapter
- **WHEN** an event lacks required validation or response fields
- **THEN** the UI SHALL discard it and end the operation with a safe error containing a correlation ID