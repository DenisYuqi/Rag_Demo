## Purpose

Defines grounded, bilingual refusals that safely explain why an answer cannot be given and provide a concrete next step that helps the user reformulate or locate appropriate internal evidence.

## ADDED Requirements

### Requirement: Reason-specific actionable refusal
Every low-confidence, out-of-scope, conflicting-evidence, and safety refusal SHALL contain a stable safe reason code, a concise explanation, and at least one actionable next step appropriate to the reason. Guidance MAY ask the user to narrow or rephrase the question, identify a relevant document or section, clarify an ambiguous term or timeframe, or contact the responsible internal owner, but MUST NOT invent facts, owners, policies, or sources.

#### Scenario: Low-confidence evidence
- **WHEN** retrieved evidence is relevant but insufficient to support a grounded answer
- **THEN** the response refuses, identifies insufficient support, and asks for a narrower question or a specific relevant document or section

#### Scenario: Conflicting evidence
- **WHEN** authoritative retrieved sources conflict and the conflict cannot be resolved safely
- **THEN** the response refuses to choose an unsupported interpretation and guides the user to clarify the applicable version, date, department, or document owner

#### Scenario: Out-of-scope question
- **WHEN** the knowledge base contains no relevant evidence
- **THEN** the response states that the request is outside the available knowledge and suggests a safer scope or relevant source to provide without inventing one

### Requirement: Same-language refusal experience
The user-facing explanation and guidance SHALL use the selected response language and SHALL work for Chinese and English first-turn and multi-turn requests. Stable reason codes and observability fields SHALL remain language-neutral.

#### Scenario: Chinese follow-up is refused
- **WHEN** a Chinese multi-turn follow-up cannot be grounded
- **THEN** the system returns natural Chinese explanation and guidance while preserving the same language-neutral reason code used for equivalent English behavior

### Requirement: Safety and privacy take precedence
Refusal guidance MUST NOT echo detected prompt injection, unsafe instructions, secrets, credentials, raw retrieved text, or supported PII. Safety refusals SHALL provide only safe redirection and SHALL not reveal internal detection rules or hidden prompts.

#### Scenario: Prompt-injection request is refused
- **WHEN** a request asks the assistant to ignore grounding rules or reveal protected instructions
- **THEN** the response refuses in the selected language, offers a safe knowledge-base question pattern, and exposes no protected content or detector detail

### Requirement: Refusal evidence is evaluable and observable
Evaluation and operations evidence SHALL record the expected and actual refusal outcome, safe reason code, guidance-present result, language, Refusal Appropriateness score, and denominator without storing the raw user prompt. Answer Compliance scoring SHALL verify explicit refusal-format and guidance instructions independently of Refusal Appropriateness.

#### Scenario: Refusal evaluation completes
- **WHEN** an evaluation case expects a refusal with guidance
- **THEN** the report independently scores refusal appropriateness and answer compliance and exposes any missing or unsafe guidance as a failed criterion
