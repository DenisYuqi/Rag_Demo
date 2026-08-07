## Purpose

Defines the versioned, reproducible evidence contract used to prove that the RAG assistant satisfies the quality, performance, cost, operations, privacy, and delivery requirements in the original assignment.

## ADDED Requirements

### Requirement: Versioned acceptance contract
The system SHALL evaluate every acceptance run against a versioned contract that treats Faithfulness, Context Precision, Answer Compliance, Style, and Refusal Appropriateness as distinct metrics. Every eligible answerable case SHALL declare non-empty, versioned compliance obligations; a case SHALL be compliant only when every applicable obligation is satisfied, and aggregate Answer Compliance SHALL equal compliant eligible cases divided by all eligible answerable cases. The advanced quality gate SHALL use unrounded values and non-zero denominators and SHALL require Faithfulness at least 0.85, Context Precision at least 0.70, Answer Compliance at least 0.90, Style at least 0.85, and Refusal Appropriateness at least 0.90 without weighted compensation; Answer Completeness MAY be reported diagnostically but MUST NOT substitute for Answer Compliance.

#### Scenario: Advanced quality gate passes
- **WHEN** a completed run meets or exceeds every versioned advanced threshold
- **THEN** the report records a passing advanced quality gate with each metric value, threshold, numerator, denominator, scorer version, and dataset version

#### Scenario: Compliance is not replaced by completeness
- **WHEN** an answer covers expected facts but violates one or more explicit response instructions
- **THEN** Answer Completeness may pass while Answer Compliance fails and the overall advanced quality gate remains failed

### Requirement: Discriminating bilingual acceptance dataset
The acceptance dataset SHALL contain at least 24 versioned cases, including at least eight primarily Chinese cases, eight primarily English cases, four multi-turn cases spanning both languages, and at least two tagged cases for each of exact-identifier lexical retrieval, cross-language semantic retrieval, plausible distractors, technical-specification or architecture content, and scanned-document content. Cases MAY satisfy more than one category but SHALL declare expected facts, permitted sources, response instructions, refusal expectations, and challenge tags.

#### Scenario: Dataset validation succeeds
- **WHEN** the one-command acceptance workflow validates a dataset
- **THEN** it confirms the case-count, language, multi-turn, challenge-category, expected-fact, source, instruction, and refusal coverage requirements before any provider call is made

#### Scenario: Dataset provenance is captured
- **WHEN** an acceptance run starts
- **THEN** its evidence records immutable dataset, corpus, case, scorer, prompt, pricing, and configuration identifiers or content hashes

### Requirement: Unbiased performance and cost denominators
The official service-level result SHALL measure at least five concurrent users against one application instance, obtain at least 500 successful measured requests, keep the measured error rate below 1 percent, and count every logical QA attempt, including errors and timeouts, in the latency and success denominators. Warm-up traffic SHALL be excluded and reported separately. At least 90 percent of all measured QA attempts SHALL complete within 10 seconds. Reports SHALL present p50, p90, p95, and p99 for all attempts and separately for successful attempts, and SHALL report cost per 1,000 logical attempts and per 1,000 successful answers.

#### Scenario: Failed attempts remain in the official denominator
- **WHEN** a load run contains successful, failed, and timed-out QA attempts
- **THEN** the official latency and pass/fail calculation includes all attempts and separately labels the successful-only supplementary statistics

#### Scenario: Cost denominators are visible
- **WHEN** a load or acceptance run completes
- **THEN** the report shows total requests, successful requests, token counts, total estimated cost, cost per 1,000 logical attempts, and cost per 1,000 successful answers with the pricing version and currency

### Requirement: Text and CSV operations summary
Every completed acceptance run SHALL produce one canonical operations record and derive semantically equivalent text and CSV operations summaries from it. They SHALL contain configuration provenance, request counts, p50 and p95 latency, input and output tokens, cache hits and eligible lookups, cache-hit rate, refusals and answered requests, refusal rate, compliant and scored answers, Answer Compliance rate, both cost-per-1,000 measures, all units, and all denominators. A zero denominator SHALL be represented explicitly as unavailable rather than as a misleading zero percent.

#### Scenario: Operations artifacts agree
- **WHEN** the text and CSV artifacts for the same run are parsed
- **THEN** every shared metric, count, denominator, unit, run identifier, and configuration identifier has the same value

#### Scenario: No eligible cache lookups occurred
- **WHEN** a run contains no cache-eligible retrievals
- **THEN** both operations artifacts show the cache-hit rate as unavailable and retain hit and eligible-lookup counts of zero

### Requirement: Log dictionary and privacy-safe sample
The deliverables SHALL include a user-facing structured-log field dictionary and a representative JSONL sample that describe field meaning, type, unit, cardinality, presence conditions, and redaction rules. The sample and documentation MUST NOT contain credentials, raw prompts, raw retrieved text, filesystem paths, or supported PII.

#### Scenario: Log documentation validation
- **WHEN** release validation inspects the field dictionary and sample
- **THEN** it confirms that every sample field is documented, every documented sensitive field has a redaction rule, and privacy scanners find no prohibited value

### Requirement: One-command non-overwriting acceptance release
The repository SHALL provide one documented command that validates prerequisites, runs the versioned acceptance plan, writes all required evidence, evaluates all gates, and returns a non-zero exit status when a required gate or artifact fails. Each invocation SHALL use a unique release-v2 output directory and MUST NOT overwrite the Phase 12 release or a prior acceptance run. Publication SHALL pass only after every required artifact is written, hashed, schema-validated, parity-checked, privacy-scanned, and recorded in an immutable manifest with format, media type, SHA-256 digest, byte size, and relative path.

#### Scenario: Clean-environment acceptance succeeds
- **WHEN** an operator runs the documented command in a correctly configured clean environment
- **THEN** it produces a sealed release-v2 bundle containing the contract, provenance, JSON and HTML reports, text and CSV operations summaries, comparison evidence, log documentation, sample logs, integrity manifest, and explicit gate results

#### Scenario: Required evidence is missing
- **WHEN** any required artifact, comparison, threshold result, or integrity check is absent or invalid
- **THEN** the command exits non-zero, preserves diagnostic evidence, and does not label the release bundle accepted
