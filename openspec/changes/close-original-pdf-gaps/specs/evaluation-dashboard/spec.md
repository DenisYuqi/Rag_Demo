## Purpose

Provides an operator-facing workbench surface for launching, monitoring, comparing, understanding, and downloading the same evaluation and acceptance evidence used by APIs, CLI workflows, and release gates.

## ADDED Requirements

### Requirement: Evaluation and comparison launch controls
The Evaluation workbench SHALL let an operator choose a standard evaluation, model comparison, retrieval comparison, cache experiment, or full acceptance workflow; select the available dataset and version; configure the allowed candidates; and see the planned case count, candidate count, maximum logical calls, cache policy, and available cost estimate before explicitly starting the run. Opening or refreshing the page MUST NOT start provider calls.

#### Scenario: Operator starts a retrieval comparison
- **WHEN** the operator selects a valid dataset, the required three retrieval strategies, and activates the run control
- **THEN** the UI creates one persisted comparison plan, displays its identifier and scope, and begins reporting progress without blocking other workbench views

#### Scenario: Launch configuration is invalid
- **WHEN** required candidates, dataset metadata, pricing, or provider configuration is missing
- **THEN** the UI prevents launch and displays a safe, actionable validation message without exposing credentials or internal paths

### Requirement: Persistent run and comparison history
The dashboard SHALL list persisted evaluations and comparisons across browser sessions and application restarts with type, status, progress, dataset version, corpus revision, candidate configurations, start and completion times, and gate result. Operators SHALL be able to refresh and select a historical item without causing recomputation.

#### Scenario: Application restarts during a completed history
- **WHEN** the operator opens the dashboard after an application restart
- **THEN** prior completed and failed items remain selectable and render from persisted evidence

#### Scenario: A comparison is still running
- **WHEN** the operator refreshes a running comparison
- **THEN** the UI shows completed, failed, active, and remaining candidates or cases and retains any partial evidence already committed

### Requirement: Acceptance scorecards and gate explanations
For a selected run, the dashboard SHALL display a clear overall status plus scorecards for Faithfulness, Context Precision, Answer Compliance, Style, Refusal Appropriateness, all-attempt p50/p90/p95 latency, successful-only latency, concurrency, input/output tokens, total cost, cost per 1,000 attempts, cost per 1,000 successes, error/timeout rate, cache-hit rate, and refusal rate. Each scorecard SHALL expose its value, unit, threshold where applicable, numerator, denominator, and pass/fail or unavailable state.

#### Scenario: A quality gate fails
- **WHEN** Answer Compliance is below its configured threshold
- **THEN** the dashboard marks the metric and overall advanced gate failed and shows the measured value, threshold, numerator, denominator, and scorer version

#### Scenario: A rate has no denominator
- **WHEN** a selected run has no eligible observations for a rate
- **THEN** the dashboard renders the metric as unavailable and does not display a zero-percent success claim

### Requirement: Quantitative comparison visualization
For a selected comparison, the dashboard SHALL render an aligned candidate table and at least one compact visual of baseline deltas covering required quality, latency, cost, token, error, and degradation measures. It SHALL identify controlled dimensions, highlight the selected recommendation or lack of one, show the evidence-backed rationale, and provide challenge-category drill-down for retrieval comparisons.

#### Scenario: Retrieval comparison is selected
- **WHEN** a completed dense, hybrid, and hybrid-rerank comparison is opened
- **THEN** the UI displays all three candidates side by side, their absolute metrics and deltas, category results, gate states, selected configuration, and recommendation rationale

#### Scenario: Partial comparison is selected
- **WHEN** one candidate failed while other candidates completed
- **THEN** the UI retains the failed candidate, displays its failure counts and safe reason, and labels any recommendation provisional or unavailable according to the plan

### Requirement: Operations and case-level diagnostics
The dashboard SHALL show the operations-summary measures together in one view and SHALL allow authorized local operators to inspect failed or low-scoring cases using privacy-safe case identifiers, tags, metric contributions, refusal reason, citation identifiers, request or trace identifier, and sanitized error codes. Raw prompts, raw document text, credentials, supported PII, and unrestricted filesystem paths MUST NOT be displayed.

#### Scenario: Operator inspects a failed case
- **WHEN** a failed case is selected
- **THEN** the dashboard shows only allowlisted diagnostic fields and a safe link or identifier for correlated request diagnostics

### Requirement: Evidence downloads use the displayed source of truth
The dashboard SHALL provide downloads for completed JSON, HTML, text, and CSV artifacts plus the comparison plan and integrity manifest when available. Values displayed in the UI, returned through the API, emitted by the CLI, and packaged for release SHALL be derived from the same persisted result records.

#### Scenario: Operator downloads displayed evidence
- **WHEN** the operator downloads any available artifact for a selected run
- **THEN** the artifact identifier and metrics match the displayed run and the service returns content without revealing its backing filesystem path

#### Scenario: An artifact is not ready
- **WHEN** the operator requests an artifact that has not passed generation and integrity validation
- **THEN** the UI reports it unavailable and does not offer a stale artifact from another run

### Requirement: Bilingual, non-misleading dashboard states
Dashboard labels, validation messages, empty states, unavailable states, and failure explanations SHALL be understandable in Chinese and English. The UI MUST distinguish failed, incomplete, unavailable, and passing results and MUST NOT infer success from missing evidence.

#### Scenario: Required backend is unavailable
- **WHEN** evaluation or comparison services are not ready
- **THEN** the dashboard remains loadable, disables affected actions, and presents bilingual recovery guidance while other workbench views remain usable
