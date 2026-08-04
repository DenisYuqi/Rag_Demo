## ADDED Requirements

### Requirement: Versioned evaluation dataset
The system SHALL evaluate against an immutable versioned dataset manifest containing dataset ID, semantic version, content hash, corpus snapshot, stable case IDs, question, language, answerability/refusal label, expected facts, authoritative evidence IDs, category, and applicable style expectations.

#### Scenario: Load a valid dataset
- **WHEN** evaluation starts
- **THEN** the runner SHALL verify the manifest hash, unique case IDs, corpus version, and non-empty eligible set for every required metric

#### Scenario: Dataset content changes
- **WHEN** a question, label, expected fact, evidence mapping, or rubric changes
- **THEN** the dataset SHALL receive a new version and MUST NOT overwrite the prior fixture

### Requirement: Representative evaluation coverage
The dataset SHALL cover Chinese and English answerable questions, multi-turn questions, scanned-document evidence, unanswerable questions, expected refusals, prompt injection, and PII. Each category MUST report its own result count and metrics.

#### Scenario: Validate dataset coverage
- **WHEN** an evaluation dataset is selected
- **THEN** the runner SHALL reject acceptance mode if any required category has no eligible case

### Requirement: Reproducible evaluation run
Every run SHALL record code revision, dataset and corpus versions, prompt versions, provider and model identities, generation settings, embedding and chunking identities, retrieval/ranking configuration, scorer versions, pricing version, random seeds, cache policy, and environment. Acceptance runs MUST disable answer and final-retrieval caches.

#### Scenario: Reproduce a run
- **WHEN** a user invokes the documented command with an immutable run manifest
- **THEN** the runner SHALL recreate the same eligible case set and threshold decisions within documented scoring tolerance

#### Scenario: Required identity cannot be pinned
- **WHEN** a dataset, corpus, model, prompt, scorer, or retrieval configuration cannot be identified
- **THEN** the run SHALL be marked invalid

### Requirement: Quantified RAG quality metrics
The runner SHALL calculate per-case and aggregate Faithfulness, Context Precision, Answer Completeness, Style Consistency, and Refusal Appropriateness using versioned scoring definitions. Model-based judging MUST use a versioned prompt and deterministic settings.

#### Scenario: Score an evaluation run
- **WHEN** all eligible cases complete
- **THEN** the runner SHALL report scores, eligibility, rationale, and evidence references for each case and metric

#### Scenario: A metric has no denominator
- **WHEN** a required metric has no eligible cases
- **THEN** the run SHALL be invalid rather than treating the metric as passing

### Requirement: Quality acceptance thresholds
The quality gate SHALL pass only when unrounded Faithfulness is greater than 0.85, Context Precision is greater than 0.70, Answer Completeness is at least 80 percent, Style Consistency is at least 80 percent, and Refusal Appropriateness is at least 80 percent. No weighted average MAY compensate for a failed metric.

#### Scenario: Every quality threshold passes
- **WHEN** all five unrounded aggregate values meet their individual operators and thresholds
- **THEN** the quality gate SHALL pass

#### Scenario: Faithfulness equals 0.85
- **WHEN** aggregate Faithfulness is exactly 0.85
- **THEN** the quality gate SHALL fail because the required comparison is strictly greater

#### Scenario: One quality metric fails
- **WHEN** any required metric misses its threshold
- **THEN** the complete quality gate SHALL fail regardless of other scores

### Requirement: HTML and JSON validation reports
Each completed run SHALL generate one versioned-schema JSON report and one human-readable HTML report derived from the JSON. Both SHALL share a run ID and contain provenance, configuration, thresholds, aggregate and category metrics, failed cases, performance, usage/cost, privacy checks, issue investigations, and final gate status.

#### Scenario: Generate reports
- **WHEN** evaluation and scoring complete
- **THEN** matching JSON and HTML reports SHALL be written and made downloadable from the workbench

#### Scenario: Report values disagree
- **WHEN** an HTML value cannot be reconciled with its JSON source
- **THEN** the run SHALL be marked invalid

### Requirement: Evidence-backed issue investigations
The final validation report SHALL document at least two distinct genuine baseline issues or explicitly labeled controlled baseline defects. Each record MUST include issue ID, classification, affected cases, symptom, privacy-safe logs and metrics, run and trace references, root-cause rationale, exact fix, and why the fix was chosen.

#### Scenario: A genuine issue is documented
- **WHEN** baseline evaluation reveals a compliance drop, false-refusal spike, retrieval defect, or other quality problem
- **THEN** the report SHALL correlate affected cases to actual metric, log, and trace evidence and the implemented fix

#### Scenario: A controlled baseline is necessary
- **WHEN** fewer than two genuine issues emerge
- **THEN** the project MAY use an explicitly labeled test-only configuration defect that is reproducible, isolated from production defaults, and disabled in the final candidate

#### Scenario: Evidence is incomplete
- **WHEN** fewer than two issues qualify or an issue lacks required evidence and rationale
- **THEN** final evaluation acceptance SHALL fail

### Requirement: Same-dataset post-fix improvement
Each issue SHALL use paired pre-fix and post-fix runs with the same dataset version, corpus snapshot, case IDs, scorer versions, and eligible denominator. Its declared primary metric MUST improve by at least 10 percent relative using unrounded values, and the final configuration MUST meet all quality and performance gates.

#### Scenario: A higher-is-better metric improves
- **WHEN** the primary metric increases from a nonzero baseline
- **THEN** relative improvement SHALL equal `(post - pre) / pre * 100`

#### Scenario: A lower-is-better error metric improves
- **WHEN** the primary metric is an error rate such as false refusal
- **THEN** relative improvement SHALL equal `(pre - post) / pre * 100`

#### Scenario: Comparison data differs
- **WHEN** pre-fix and post-fix runs use different datasets, corpus snapshots, case IDs, or eligible denominators
- **THEN** the comparison SHALL be rejected

#### Scenario: Improvement is below target
- **WHEN** either issue improves its primary metric by less than 10 percent
- **THEN** final evaluation acceptance SHALL fail

### Requirement: Automated evaluation gate
The evaluation command SHALL return a nonzero result when a run is invalid, a quality threshold fails, privacy verification fails, required reports are absent, or either issue investigation fails evidence or improvement requirements.

#### Scenario: Complete validation passes
- **WHEN** reproducibility, quality, privacy, reporting, and both issue comparisons pass
- **THEN** the command SHALL return success and identify the accepted code, dataset, corpus, configuration, and run IDs
