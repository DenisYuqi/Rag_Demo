## ADDED Requirements

### Requirement: Basic PII and secret detection
The system SHALL detect at minimum email addresses, phone numbers, Chinese national ID numbers, US Social Security numbers, plausible payment card numbers, IPv4 and IPv6 addresses, bearer tokens, API keys, password assignments, and private-key blocks. It SHALL replace complete detected values with typed redaction placeholders.

#### Scenario: Detect supported PII
- **WHEN** user-visible content contains a supported email, phone, identity number, payment card, or IP address
- **THEN** the complete value SHALL be replaced with its typed `[REDACTED_*]` placeholder

#### Scenario: Detect a secret
- **WHEN** dynamic content contains a recognized credential, authorization value, password assignment, or private key
- **THEN** the sensitive value or block SHALL be replaced with `[REDACTED_SECRET]`

### Requirement: Redact every output surface
The system SHALL apply PII and secret redaction to answer text, refusals, citations, source previews and titles, errors, diagnostics, evaluation case output, and downloadable generated reports before they leave the service boundary.

#### Scenario: Evidence contains personal information
- **WHEN** a grounded answer or cited source preview contains detected PII
- **THEN** every user-visible occurrence SHALL be redacted while preserving usable citation identity

#### Scenario: Output redaction fails
- **WHEN** the redaction component errors, times out, or is not initialized
- **THEN** the system SHALL withhold dynamic content and return only a fixed pre-vetted unavailable message

### Requirement: Content-minimized and redacted telemetry
Application logs and traces SHALL use allowlisted structured fields and MUST NOT record raw questions, answers, prompts, conversation histories, retrieved text, uploaded content, authorization headers, or credentials. Any permitted free-text error or metadata field MUST pass redaction before export.

#### Scenario: A request contains PII
- **WHEN** a user submits a question containing PII
- **THEN** telemetry SHALL retain correlation, timing, status, counts, and PII category metadata without retaining the raw question or value

#### Scenario: An exception contains sensitive content
- **WHEN** a dependency exception includes request, document, network, or credential data
- **THEN** logs and traces SHALL retain only a sanitized error category, component, and correlation identifiers

#### Scenario: Telemetry redaction fails
- **WHEN** a telemetry event cannot be safely filtered
- **THEN** the system SHALL drop the event and MAY increment a content-free failure counter

### Requirement: Safe streaming across chunk boundaries
The system SHALL maintain a private output buffer across model deltas. It MUST NOT emit a prefix that could become part of a sensitive value until a safe boundary has been validated. The MVP MAY validate and emit the complete response as one event; if it emits smaller units, detector state MUST span model deltas and emitted-unit boundaries.

#### Scenario: An email is split across deltas
- **WHEN** one delta ends with `person@` and a later delta completes the domain
- **THEN** no raw fragment SHALL be emitted and the client SHALL receive only the redacted placeholder

#### Scenario: Generation ends with buffered content
- **WHEN** generation finishes without a normal sentence boundary
- **THEN** the complete pending content SHALL be scanned and either safely emitted or discarded

### Requirement: Basic prompt-injection defense
The system SHALL treat user input and retrieved documents as untrusted data. It MUST refuse attempts to override higher-priority policy, disable grounding or redaction, reveal system prompts or hidden context, disclose secrets, or access other sessions. Retrieved text MUST NOT trigger tools, external URLs, commands, or policy changes.

#### Scenario: User asks to ignore safety instructions
- **WHEN** a user requests an answer without grounding, citations, privacy checks, or hidden-policy restrictions
- **THEN** the system SHALL refuse the override while retaining all controls

#### Scenario: A retrieved chunk contains instructions
- **WHEN** document text requests prompt disclosure, command execution, external access, or safety bypass
- **THEN** the system SHALL treat it only as untrusted content and SHALL NOT execute or follow those instructions

#### Scenario: A legitimate question quotes injection language
- **WHEN** a user asks a grounded analytical question about text that includes injection phrases
- **THEN** the system SHALL evaluate intent and MUST NOT refuse solely because a trigger phrase appears

### Requirement: Fail-closed safety readiness
Answer-producing endpoints SHALL be ready only when required detector rules, redactors, injection policy, and safe-stream state initialize successfully. The system MUST NOT fall back to unfiltered output or telemetry.

#### Scenario: A required safety component fails at startup
- **WHEN** required privacy or injection controls cannot initialize
- **THEN** readiness SHALL fail and dynamic answer endpoints SHALL remain unavailable

#### Scenario: Safety becomes uncertain during a request
- **WHEN** the system cannot establish that pending content is safe
- **THEN** it SHALL discard pending content and return a fixed localized failure message

### Requirement: Privacy verification corpus
The project SHALL include deterministic tests for every supported PII and secret category in answers, citations, diagnostics, logs, reports, and split streaming chunks. Verification MUST assert zero raw matches in captured output and telemetry fixtures.

#### Scenario: Run the privacy test suite
- **WHEN** privacy tests execute against representative Chinese and English values
- **THEN** all supported values SHALL be redacted and no raw fixture value SHALL appear in captured outputs or logs
