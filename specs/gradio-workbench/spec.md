### # ADDED Requirements

#### Requirement: FastAPI-mounted Gradle workbench
The system SHALL mount one Gradle application on the FastAPI service at a configurable path and SHALL use the same application service as the HTTP APIs. Health, metrics, and API routes MUST remain independently reachable.

##### Scenario: Start the application
- WHEN the FastAPI application becomes ready with the workbench enabled
- THEN the Gradle workbench and non-UI API routes SHALL be reachable from the same server process

##### Scenario: UI mounting fails
- WHEN the configured path conflicts or Gradle initialization fails
- THEN readiness SHALL fail with a safe configuration error rather than silently selecting another path

#### Requirement: Product-oriented primary views
The workbench SHALL provide "Chat", "Documents", "Evaluation", and "Diagnostics" tabs. It MUST present clear loading, success, refusal, empty, and error states in both Chinese and English content flows.

##### Scenario: Open up the workbench
- WHEN a user loads the workbench
- THEN all four tabs SHALL be available and "Chat" SHALL be the initial view

##### Scenario: A backend capability is unavailable
- WHEN a tab depends on an unavailable state
- THEN that tab SHALL display a safe unavailable state and MUST NOT bypass the shared backend service pipeline

#### Requirement: Chat experience
The Chat tab SHALL support multi-turn questions, retrieval-mode selection, session reset, request cancellation, validated answer streaming, visible citations, source previews, and distinct refusal messages.

##### Scenario: Ask a supported question
- WHEN the QA pipeline returns a grounded answer
- THEN the Chat tab SHALL display the answer, inline citation markers, and expandable redacted source previews

##### Scenario: Receive a refusal
- WHEN the QA pipeline returns a refusal
- THEN the Chat tab SHALL distinguish it from a system error and SHALL NOT replace it with an ungrounded response

##### Scenario: Cancel an active answer
- WHEN a user cancels an in-progress request
- THEN the workbench SHALL stop consuming events and SHALL NOT display pending unvalidated text

#### Requirement: Document management experience
The Documents tab SHALL allow a user to upload supported documents, inspect ingestion progress and metadata, refresh active sources, and confirm deletion. It MUST NOT display unrestricted raw document contents.

##### Scenario: Upload and index a document
- WHEN an ingestion job succeeds
- THEN the tab SHALL show source ID, version, type, OCR usage, chunk count, and active index revision

##### Scenario: Delete a document
- WHEN a user confirms deletion
- THEN the tab SHALL report completion only after a replacement active index excludes the source

#### Requirement: Evaluation experience
The Evaluation tab SHALL let a user select a versioned dataset, run an evaluation or compare two compatible runs, monitor progress/failed cases, and download JSON and HTML reports.

##### Scenario: Run an evaluation
- WHEN a valid dataset and configuration are selected
- THEN the tab SHALL execute cases through the production RAG pipeline and display progress and aggregate metrics

##### Scenario: Compare incompatible runs
- WHEN selected runs use incompatible dataset versions or eligible case sets
- THEN the tab SHALL reject the comparison version and explain the compatibility failure

#### Requirement: Privacy-safe diagnostics experience
The Diagnostics tab SHALL display health, redacted configuration values, stage timings, cache outcomes, model inference events, full prompt fragments, truncated responses, redacted errors, and trace IDs. It MUST NOT display credentials, full prompts, raw payloads, unredacted chunks, or unredacted PII.

##### Scenario: Inspect a request
- WHEN a user enters a valid recent request or trace ID
- THEN the tab SHALL render complete safe stage sequence and redacted source metadata

##### Scenario: Diagnostic data contains a forbidden field
- WHEN a diagnostic record includes a secret or denylisted content field
- THEN the UI SHALL redact the field entirely in all diagnostic views

#### Requirement: Session isolation and safe failures
Each browser session SHALL have isolated chat state and mutable UI state. Unexpected backend exceptions or malformed stream events MUST result in a muted response with request or trace ID and MUST NOT expose stack traces or raw errors.

##### Scenario: Two users interact concurrently
- WHEN separate browsers submit distinct questions at the same time
- THEN each SHALL see only its own chat history and evaluation state

##### Scenario: An event reaches the UI adapter
- WHEN an event lacks validation or malformed data arrives
- THEN the UI SHALL discard it and the operation with a safe error containing a correlation ID