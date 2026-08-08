## Purpose

Provide profile-isolated evaluation execution and evidence so operators can generate and inspect truthful reports for the retrieval model selected in the workbench.

## ADDED Requirements

### Requirement: Every retrieval profile has independent evaluation state
The system SHALL expose an Evaluation service for each enabled retrieval profile and MUST persist its runs, comparisons, summaries, manifests, and report artifacts beneath that profile's isolated data root.

#### Scenario: BGE evaluation is executed
- **WHEN** an operator starts an evaluation while `bge-local` is selected
- **THEN** the run uses the BGE retrieval configuration and is persisted only in the BGE profile's evaluation repository and artifact directories

#### Scenario: Evaluation catalogs differ between profiles
- **WHEN** the operator switches between profiles whose evaluation histories differ
- **THEN** the workbench shows only the datasets, plans, runs, comparisons, and artifacts available from the selected profile

### Requirement: Evaluation evidence identifies the selected retrieval configuration
Generated evaluation and comparison evidence MUST bind to the selected profile's truthful configuration and model identities, including its embedding and reranking models.

#### Scenario: BGE report is generated
- **WHEN** a `bge-local` evaluation completes successfully
- **THEN** its canonical report identifies BGE-M3 embedding and BGE reranking instead of the OpenAI API retrieval models

#### Scenario: Runs from different profiles are compared
- **WHEN** a comparison request attempts to combine run identifiers from different retrieval profiles
- **THEN** the operation fails safely and does not produce cross-profile evidence

### Requirement: Workbench evaluation routing follows explicit profile selection
The workbench MUST pass the selected retrieval profile explicitly for every Evaluation and Comparison refresh, preview, start, poll, compare, and artifact operation.

#### Scenario: Profile selection changes on the Evaluation page
- **WHEN** the operator changes the shared retrieval-profile selector
- **THEN** Evaluation and Comparison data refresh from the newly selected profile and prior profile run selections are cleared

#### Scenario: Unknown profile is submitted
- **WHEN** an Evaluation or Comparison action contains an unregistered profile identifier
- **THEN** the workbench returns a safe unavailable result without executing against another profile

### Requirement: Profile reports remain downloadable through same-origin APIs
The system SHALL expose profile-qualified same-origin report and artifact downloads while preserving the existing unqualified Evaluation and Comparison API behavior for `openai-api`.

#### Scenario: Existing API client omits a profile
- **WHEN** a client calls an existing Evaluation or Comparison endpoint without a retrieval-profile qualifier
- **THEN** the request uses `openai-api` with the existing response contract

#### Scenario: BGE artifact link is opened
- **WHEN** the workbench renders or opens an artifact link for a BGE run
- **THEN** the request is explicitly qualified as `bge-local` and resolves only through the BGE Evaluation service

### Requirement: All profile evaluation services share the application lifecycle
The application SHALL start and close every composed profile Evaluation service exactly once while retaining the existing ownership of the default API service.

#### Scenario: Application starts with BGE enabled
- **WHEN** executable startup completes
- **THEN** both the OpenAI and BGE Evaluation supervisors are ready before traffic is accepted

#### Scenario: Application shuts down
- **WHEN** graceful shutdown begins
- **THEN** active Evaluation jobs for every profile receive bounded shutdown and all profile resources are closed without double-closing the default service
