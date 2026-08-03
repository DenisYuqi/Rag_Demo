# tasks.md
openspec > changes > build-rag-assistant-mvp > tasks.md

## #1. Application Foundation
- [ ] 1.1.1 Create `pyproject.toml` with Python 3.12 runtime and test dependency groups, generate `uv.lock`, and verify a clean environment with `uv sync --frozen`.
- [ ] 1.1.2 Create the `src/rag_mvp` package and module directories from `design.md`; verify imports with `uv run python -c "import rag_mvp"`.
- [ ] 1.1.3 Add code-quality configuration for Ruff, mypy, and pytest; verify the empty package with `uv run ruff check && uv run src`.
- [ ] 1.1.4 Implement validated environment settings with safe defaults and secret-field masking; verify with `uv run pytest tests/unit/config/test-settings.py -q`.
- [ ] 1.1.5 Implement the configurable data-root directory layout and safe path creation; verify with `uv run pytest tests/unit/storage/test-data-layout.py -q`.
- [ ] 1.1.6 Implement the FastAPI application factory plus `/healthz`; verify with `uv run pytest tests/api/test-health.py -q && health.py -k ready`.
- [ ] 1.1.7 Implement coherent request aggregation and `/readyz`; verify ready and unready cases with `uv run pytest tests/unit/test-health.py -k readyz`.

## #2. Domain Contracts and Persistence
- [ ] 2.1 Define provider-neutral document, chunk, index-revision, and ingestion-job models; verify validation and serialization with `uv run pytest tests/unit/domain/test-ingestion-models.py -q`.
- [ ] 2.2 Define provider-neutral retrieval candidate, ranking verdict, cite, and diagnostic models; verify with `uv run pytest tests/unit/domain/test-retrieval-models.py -q`.
- [ ] 2.3 Define answer, refusal, error, citation, session, and validated-stream models; verify with `uv run pytest tests/unit/domain/test-qa-models.py -q`.
- [ ] 2.4 Define audit-event, evaluation-run, evaluation-run, and issue-evidence models; verify with `uv run pytest tests/unit/domain/test-evaluation-models.py -q`.
- [ ] 2.5 Implement SQLite initialization and MVP metadata; verify create and update paths with `uv run pytest tests/unit/storage/test-metadata-store.py -q`.
- [ ] 2.6 Implement repositories for documents, versions, ingestion jobs, and active index manifests; verify CRUD and transaction rollback with `uv run pytest tests/unit/storage/test-knowledge-repositories.py -q`.
- [ ] 2.7 Implement repositories for sessions, request diagnostics, provider usage, evaluation runs, and report manifests; verify with `uv run pytest tests/unit/storage/test-runtime-repositories.py -q`.

## #3. OpenAI-Compatible Model Providers
- [ ] 3.1 Define asynchronous embedding, generation, and reranking protocols plus normalized error categories; verify protocol test doubles with `-q`.
- [ ] 3.2 Implement deterministic fake embedding, generation, and reranking providers for offline tests; verify reproducibility with `uv run pytest tests/unit/providers/test-providers.py -q`.
- [ ] 3.3 Implement shared synchronous client factory without logging credentials; verify configuration using mocked HTTP with `uv run pytest tests/unit/providers/test-openai-client.py -q`.
- [ ] 3.4 Implement request batching, response chunk dimension/validation, and embed-clause span identity; verify malformed request rejection with `uv run pytest tests/unit/providers/test-batching.py -q`.
- [ ] 3.5 Implement bounded rate limiting with configurable backoff and jitter; verify wait, empty, and malformed responses with `uv run pytest tests/unit/providers/test-rate-limiting.py -q`.
- [ ] 3.6 Implement listwise reranking with versioned prompt and exact candidate-ID permutation validation; verify reordered, duplicate, and missing IDs with `uv run pytest tests/unit/providers/test-rerank.py -q`.
- [ ] 3.7 Implement deadline-line-wait timeout, cancellation, transient, and non-retriable authentication behavior; verify with `uv run pytest tests/unit/providers/test-transient-reliability.py -q`.
- [ ] 3.8 Implement ordered generation, truncation, compatible-only embedding fallback, rule readiness, and attempt-level usage recording; verify with `uv run pytest tests/unit/providers/test-routing.py -q`.

## #4. Privacy and Prompt Safety
- [ ] 4.1 Implement the typed redaction result and detector registry; verify deterministic overlap precedence with `uv run pytest tests/unit/safety/test-redactor-core.py -q`.
- [ ] 4.2 Add email and international/Chinese phone recognition; verify positive and negative bilingual fixtures with `uv run pytest tests/unit/safety/test-pii-email-phone.py -q`.
- [ ] 4.3 Add Chinese ID and US SSN recognition with format/checksum validation; verify with `uv run pytest tests/unit/safety/test-pii-id-ssn.py -q`.
- [ ] 4.4 Add document fragmentation with separator normalization and bulk redactions; verify plausible and false-positive fixtures with `uv run pytest tests/unit/safety/test-pii-bulk.py -q`.
- [ ] 4.5 Add API keys/Tokens and account API-key, bearer-token, password, and private-key redaction; verify masking and retention controls with `uv run pytest tests/unit/safety/test-secret-redaction.py -q`.
- [ ] 4.6 Implement span merging, trimming, injection detection, error, diagnostics, and repeat suppression; verify with `uv run pytest tests/unit/safety/test-injection-filter.py -q`.
- [ ] 4.7 Implement allowlisted metadata filtering that truncates injection detection failure warnings; verify no future false warnings with `uv run pytest tests/unit/safety/test-metadata-filter.py -q`.
- [ ] 4.8 Implement sanitization on all input chunk texts and retrieved-context isolation; verify bypass, masked diagnostics, and document-citation masking with `uv run pytest tests/unit/safety/test-context-sanitize.py -q`.
- [ ] 4.9 Implement bounded sentence buffering with cross-data state and full-chunk flush; verify email, phone, card, IP, and private-key leakage with `uv run pytest tests/unit/safety/test-stream-pii.py -q`.

## #5. Knowledge Ingestion
- [ ] 5.1 Implement layered upload extension, MIME, filenames, and safe-storage validation workflow; verify rejected malformed paths with `uv run pytest tests/unit/ingestion/test-upload-validation.py -q`.
- [ ] 5.2 Implement persistent Chroma revision handling and persistent indexed BM25 snapshot creation with bilingual tokenization; verify restart persistence with `uv run pytest tests/ingestion/test-index-stage.py -q`.
- [ ] 5.10 Implement batched document embedding plus client-digest embedding cache; verify duplicate chunks avoid provider calls with `uv run pytest tests/unit/ingestion/test-embedding-stage.py -q`.
- [ ] 5.11 Implement immutable Chroma revision snapshot and persistent indexed BM25 snapshot creation with bilingual tokenization; verify restart persistence with `uv run pytest tests/ingestion/test-index-stage.py -q`.
- [ ] 5.12 Implement dense/lexical chunk-parity validation and active manifest-publication; verify partial failures preserve the prior revision with `uv run pytest tests/integration/test-ingestion-service.py -q`.
- [ ] 5.13 Implement full ingestion orchestration, same-source update ordering, reindex, deletion, and startup recovery; verify end-to-end state transitions with `uv run pytest tests/integration/test-ingestion-service.py -q`.

## #6. Retrieval and Ranking
- [ ] 6.1 Implement query validation, Unicode canonicalization, mode selection, and active-revision snapshot binding; verify invalid and concurrent-publication cases with `uv run pytest tests/unit/retrieval/test-query-context.py -q`.
- [ ] 6.2 Implement compatible query embedding and Chroma dense search with deterministic tie-breaking; verify with `uv run pytest tests/unit/retrieval/test-dense.py -q`.
- [ ] 6.3 Implement persistent BM25 search using the versioned Chinese/English tokenizer; verify exact English terms and unspaced Chinese fixtures with `uv run pytest tests/unit/retrieval/test-bm25.py -q`.
- [ ] 6.4 Implement parallel dense/BM25 collection and candidate merge by stable chunk ID; verify ranks and raw scores are preserved with `uv run pytest tests/unit/retrieval/test-hybrid-collection.py -q`.
- [ ] 6.5 Implement weighted Reciprocal Rank Fusion and deterministic tie-breaking; verify formula examples with `uv run pytest tests/unit/retrieval/test-rrf.py -q`.
- [ ] 6.6 Implement bounded rerank orchestration and RRF fallback on timeout or invalid output; verify with `uv run pytest tests/unit/retrieval/test-rerank-stage.py -q`.
- [ ] 6.7 Implement combined `hybrid` and `hybrid-rerank` orchestration with configured one-tier rerank delegation; verify each mode end-to-end with `uv run pytest tests/integration/test-retrieval-service.py -q`.
- [ ] 6.8 Implement ranked complete embedding, retrieval, and real stale score only; verify PDF and text evidence mixing plus metadata masking with `uv run pytest tests/unit/retrieval/test-results.py -q`.
- [ ] 6.9 Implement version-complete embedding, retrieval, and caching key plus TTL/size bounds; verify corpus/chunks changes purge prior entries with `uv run pytest tests/unit/retrieval/test-cache.py -q`.
- [ ] 6.10 Add a persistent-index retrieval integration suite with bilingual semantic, exact, empty, and restricted tests; verify restarts with `uv run pytest tests/integration/test-retrieval-pipeline.py -q`.

## #7. Grounded QA Pipeline
- [ ] 7.1 Implement isolated conversation-session creation, turn storage, reset, and ownership checks; verify with `uv run pytest tests/unit/qa/test-sessions.py -q`.
- [ ] 7.2 Implement multi-turn language selection and conditional follow-up query rewriting that never treats assistant history as evidence; verify with `uv run pytest tests/unit/qa/test-query-rewrite.py -q`.
- [ ] 7.3 Implement context extraction from chunk-token bounds, cut/break-token bounds; verify ordering and truncation with `uv run pytest tests/unit/qa/test-context-build.py -q`.
- [ ] 7.4 Implement the grounded prompt template that labels retrieved chunks untrusted and requires structured claims with chunk IDs; verify prompt boundaries without snapshotting secret configuration with `uv run pytest tests/unit/qa/test-prompt-builder.py -q`.
- [ ] 7.5 Implement structured citation parsing and deterministic citation existence/locator validation; verify near-match suppression with `uv run pytest tests/unit/qa/test-citations.py -q`.
- [ ] 7.6 Implement ungrounded refusal with source citations for every unsupported claim; verify claims reach assembly with `uv run pytest tests/unit/qa/test-grounding-rules.py -q`.
- [ ] 7.7 Implement grounded insufficient/conflicting-evidence refusal decisions with stable refusal codes; verify average, partial, absent, and conflict fixtures with `uv run pytest tests/unit/qa/test-refusal-policy.py -q`.
- [ ] 7.8 Implement the async QA orchestrator with one total deadline, stage budgets, cancellation, retry limits, and optional rerank override; verify with fake-clock time budgets with `uv run pytest tests/integration/test-orchestrator.py -q`.
- [ ] 7.9 Connect validated sentence streaming to citation, injection, and redaction rules; verify no raw model text end-to-end if malformed with `uv run pytest tests/unit/qa/test-streaming.py -q`.
- [ ] 7.10 Add end-to-end QA integration tests for bilingual answers, mixed media, citation extraction, reusability, refusal, PII, provider failure, and deadline failure; verify with `uv run pytest tests/integration/test-qa-pipeline.py -q`.

## #8. HTTP API and Gradio Workbench
- [ ] 8.0.1 Implement document upload/list/job-status/delete/reindex API routes with shared schemas; verify status codes and idempotency with `uv run pytest tests/api/test-documents.py -q`.
- [ ] 8.0.2 Implement the full QA API with streaming and non-stream variants; verify answer, refusal, error, and malformed-request handling with `uv run pytest tests/api/test-qa.py -q`.
- [ ] 8.0.3 Implement evaluation start/status/stop/result API routes; verify with `uv run pytest tests/api/test-evaluation.py -q`.
- [ ] 8.0.4 Mount Gradio FastAPI Path at `/ui`; pass through diagnostics API routes with route coexistence and isolation; verify with `uv run pytest tests/api/test-gradio-mount.py -q`.
- [ ] 8.0.5 Implement Chat controls with message text, attachments, inline citations, source previews, reset, and cancel; verify callback success, progress, state updates, chunk selection, and deletion dialog flow; verify callbacks with `uv run pytest tests/ui/test-chat.py -q`.
- [ ] 8.0.6 Add Documents UI for uploads, list, download, and Diagnostics health/request-trace viewers; verify page states with `uv run pytest tests/ui/test-documents.py -q`.
- [ ] 8.0.7 Add a Gradio status page for session isolation and all UI error states; verify with `uv run pytest tests/ui/test-status.py -q`.

## #9. Observability, Cost, and Performance Controls
- [ ] 9.3 Add OpenTelemetry root and stage spans with async context propagation and no content attributes; verify with an in-memory exporter using `uv run pytest tests/unit/observability/test-tracing.py -q`.
- [ ] 9.4 Instrument ingestion, retrieval, provider, QA, safety, and evaluation stages with consistent request/run IDs and timing; verify span/log correlation with `uv run pytest tests/integration/test-observability.py -q`.
- [ ] 9.5 Implement versioned model price per-token/at-request cost aggregation; verify unknown usage/pricing and per-1,000-call calculation with `uv run pytest tests/unit/observability/test-costs.py -q`.
- [ ] 9.6 Implement a QA admission controller supporting at least five active pipelines and a bounded queue; verify full work excess is rejected with `uv run pytest tests/unit/performance/admission.py -q`.
- [ ] 9.7 Move synchronous Chroma, BM25, OCR, and report work to bounded worker pools; verify the event loop remains responsive with `uv run pytest tests/unit/performance/test-worker-pool.py -q`.
- [ ] 9.8 Implement the 9.5-second total deadline and configurable stage latency budgets with reranker degradation; verify fake-provider timing cases with `uv run pytest tests/unit/performance/test-deadlines.py -q`.
- [ ] 9.9 Implement safe persisted request diagnostics and lookup by request/trace ID; verify retention bounds on redaction with `uv run pytest tests/unit/observability/test-diagnostics.py -q`.
- [ ] 9.10 Add a deterministic single-process concurrency integration test proving full QA pipelines make overlapping provider calls; verify with `uv run pytest tests/integration/test_five_concurrent_qa.py -q`.

## #10. RAG Evaluation and Issue Evidence
- [ ] 10.1 Implement versioned dataset/manifest loading, content hashing, corpus-version checks, and category eligibility validation; verify with `uv run pytest tests/unit/evaluation/test-dataset.py -q`.
- [ ] 10.2 Add non-sensitive bilingual sample documents and an initial unbiased Chinese/English case set with authoritative chunk mappings; validate with `python -m rag_mvp.evaluation.validate_dataset/`.
- [ ] 10.3 Extend the same dataset with multi-turn, OCR, unanswerable, required-refusal, injection, and PII categories; run the dataset validator and confirm every required category is eligible.
- [ ] 10.4 Implement the evaluation runner through the full production QA pipeline with isolated sessions, disabled final caches, progress persistence, and immutable run metadata; verify with `uv run pytest tests/unit/evaluation/test-runner.py -q`.
- [ ] 10.5 Implement versioned Faithfulness and Context Precision scoring with per-case rationale/evidence; verify formulas and strict eligibility with `uv run pytest tests/unit/evaluation/test-grounding-metrics.py -q`.
- [ ] 10.6 Implement Answer Completeness, Style Consistency, and Refusal Appropriateness scoring; verify boundary and inappropriate-refusal cases with `uv run pytest tests/unit/evaluation/test-refusal-metrics.py -q`.
- [ ] 10.7 Implement the ungraded quality gate for >=0.85, >=0.70, >=0.80, >=0.80; verify every operator with test varied JSON report outputs and JSON Schema validation; verify generated metric, performance, cost, privacy, and issue sections with `uv run pytest tests/unit/evaluation/test_json_report.py -q`.
- [ ] 10.9 Implement Jinja2 HTML rendering from JSON and automated validity checks; verify with `uv run pytest tests/unit/evaluation/test_html_report.py -q`.
- [ ] 10.10 Run the full evaluation suite, persist its immutable report and manifests, and verify it will not reprocess the first genuine baseline issue, or an explicitly test-only controlled compliance baseline is defined.
- [ ] 10.12 Implement the issue capture case with log/messages/run metadata, root cause, and proposed fix to the report.
- [ ] 10.13 Select and evidence a distinct set of test cases where a strict metric test fails only controlled refusal baseline if needed; verify it uses the same dataset/corpus/scorers as the identical full-run.
- [ ] 10.14 Implement the smallest justified fix range at least 10% relative improved accepted one; verify its configuration identities, and report parity with `uv run rag_mvp.evaluation.verify_report <final-report.json>`.

## #11. Load Evidence and Local Container Deployment
- [ ] 11.1 Implement an HTTP load-test harness with warm-up, fixed five-user concurrency, cache-bypass headers, at least 500-success iterations, nearest-rerank percent, and error accounting; verify perf logic with `uv run pytest tests/performance/test-load-report.py -q`.
- [ ] 11.2 Package the machine-readable performance evidence bundle with run/model, data, cost, and representative trace references; verify schema with `uv run pytest tests/deploy/test-evidence-bundle.py -q`.
- [ ] 11.3 Add a multi-stage Dockerfile with `dockerignore`, run-as-nonroot, OCI labels, and `dockeignore`; verify build and UID mapping with `docker run --rm -v ./tests/deploy:/test compose full-build && compose run test-deploy`.
- [ ] 11.4 Add Docker Compose with exactly one app instance, loopback-exposed ingest, persistent data volume, compose-healthcheck; verify compose commands with `uv run pytest tests/deploy/test-compose.py -q`.
- [ ] 11.5 Implement adaptive concurrency throttling to drain transient load without degrading behavior; verify with `uv run pytest tests/unit/performance/test-throttle.py -q`.
- [ ] 11.6 Verify an example and a local runbook covering build, start, stop, logs, diagnostics, and upgrade; verify all shared ground truth data is mounted in the runbook against a controlled set of cases.
- [ ] 11.7 Verify only fixed paths are persisted; verify upgrade steps retain corpus, evaluation runs, and running the same image against the same manifest yields identical answers without reindexing full corpus.
- [ ] 11.8 Implement release smoke scanner and `try` image `--exit-code 1` critical runtime checks; verify release checks.

## #12. Final Acceptance
- [ ] 12.1 Run formatting, linting, typing, unit, API, integration, UI, and privacy suites from a clean environment with a full test report; verify zero new regressions.
- [ ] 12.2 Run the complete performance load test from a warm-up state; collect logs, traces, diagnostics, and all cost counters.
- [ ] 12.3 Confirm validated PII and secret fixture masking; verify the candidate image Faithfulness >=0.85.
- [ ] 12.4 Confirm model price and cost aggregation across all evaluation runs and load test runs; verify reported cost per 1,000 calls.
- [ ] 12.5 Confirm Compose runbook usability: Style, Metrics, and Refusal Appropriateness; verify required evidence has no hidden failure paths across at least 30 run cases, 100 complete latency bounds, exact "cta".
- [ ] 12.6 Publish the accepted image tag, Compose file, runbook, corpus version, final JSON/HTML evaluation report bundle, cost per 1,000 calls, known limitations and exact reproduction commands in the release README.