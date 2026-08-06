## Delivery Order

Implement and integrate in vertical gates. A gate is complete only when its end-to-end test passes; horizontal task numbers below remain the detailed acceptance checklist.

- **Gate A - Walking skeleton:** 1.1-1.7, minimum models/repositories from 2, fake embedding/generation from 3, text/Markdown path from 5, dense path from 6, non-streamed grounded QA from 7, QA/document APIs from 8, and one restart integration test.
- **Gate B - Required RAG path:** selective PDF OCR, deterministic recursive chunking, atomic dense/BM25 publication, hybrid RRF, refusal/citations, and Chat/Documents UI.
- **Gate C - Safe deployable candidate:** redaction and prompt isolation, concurrency/deadlines, structured diagnostics, evaluation corpus/gates, Docker persistence, and acceptance harness.
- **Gate D - Measured enhancements:** enable reranking, extra caches, provider fallback, validated sentence streaming, and richer telemetry only when required by an unmet gate or supported by measured quality/latency/cost evidence.

## 1. Application Foundation

- [x] 1.1 Create `pyproject.toml` with Python 3.12 runtime and test dependency groups, generate `uv.lock`, and verify a clean environment with `uv sync --frozen`.
- [x] 1.2 Create the `src/rag_mvp` package and module directories from `design.md`; verify imports with `uv run python -c "import rag_mvp"`.
- [x] 1.3 Add code-quality configuration for Ruff, mypy, and pytest; verify the empty package with `uv run ruff check . && uv run mypy src`.
- [x] 1.4 Implement validated environment settings with safe defaults and secret-field masking; verify with `uv run pytest tests/unit/config/test_settings.py -q`.
- [x] 1.5 Implement the configurable data-root directory layout and safe path creation; verify with `uv run pytest tests/unit/storage/test_data_layout.py -q`.
- [x] 1.6 Implement the FastAPI application factory plus `/healthz`; verify with `uv run pytest tests/api/test_health.py -q -k healthz`.
- [x] 1.7 Implement component readiness aggregation and `/readyz`; verify ready and unready cases with `uv run pytest tests/api/test_health.py -q -k readyz`.

## 2. Domain Contracts and Persistence

- [x] 2.1 Define provider-neutral document, chunk, index-revision, and ingestion-job models; verify validation and serialization with `uv run pytest tests/unit/domain/test_ingestion_models.py -q`.
- [x] 2.2 Define provider-neutral retrieval candidate, ranking evidence, mode, and diagnostics models; verify with `uv run pytest tests/unit/domain/test_retrieval_models.py -q`.
- [x] 2.3 Define QA answer, refusal, error, citation, session, and validated-stream models; verify with `uv run pytest tests/unit/domain/test_qa_models.py -q`.
- [x] 2.4 Define model-attempt, token-usage, pricing, evaluation-run, and issue-evidence models; verify with `uv run pytest tests/unit/domain/test_evaluation_models.py -q`.
- [x] 2.5 Implement SQLite initialization and versioned schema migration for MVP metadata; verify create and upgrade paths with `uv run pytest tests/unit/storage/test_database.py -q`.
- [x] 2.6 Implement repositories for documents, versions, ingestion jobs, and active index manifests; verify CRUD and transaction rollback with `uv run pytest tests/unit/storage/test_knowledge_repositories.py -q`.
- [x] 2.7 Implement repositories for sessions, request diagnostics, provider usage, evaluation runs, and report manifests; verify with `uv run pytest tests/unit/storage/test_runtime_repositories.py -q`.

## 3. OpenAI-Compatible Model Providers

- [x] 3.1 Define asynchronous embedding, generation, and reranking protocols plus normalized error categories; verify protocol test doubles with `uv run pytest tests/unit/providers/test_protocols.py -q`.
- [x] 3.2 Implement deterministic fake embedding, generation, and reranking providers for offline tests; verify reproducibility with `uv run pytest tests/unit/providers/test_fakes.py -q`.
- [x] 3.3 Implement the shared asynchronous OpenAI-compatible client factory without logging credentials; verify configuration using mocked HTTP with `uv run pytest tests/unit/providers/test_openai_client.py -q`.
- [x] 3.4 Implement embedding batching, response count/dimension validation, and embedding-space identity; verify malformed responses are rejected with `uv run pytest tests/unit/providers/test_openai_embedding.py -q`.
- [x] 3.5 Implement bounded chat generation and normalized content/finish/usage results; verify valid, empty, and malformed responses with `uv run pytest tests/unit/providers/test_openai_generation.py -q`.
- [x] 3.6 Implement listwise reranking with versioned prompt and exact candidate-ID permutation validation; verify unknown, duplicate, and missing IDs with `uv run pytest tests/unit/providers/test_openai_reranker.py -q`.
- [x] 3.7 Implement deadline-aware timeout, cancellation, transient retry, and non-retriable authentication behavior; verify with `uv run pytest tests/unit/providers/test_resilience.py -q`.
- [x] 3.8 Implement role readiness and attempt-level usage recording; retain ordered generation/reranking and compatible-only embedding fallback support, but configure additional routes only when acceptance evidence requires them; verify with `uv run pytest tests/unit/providers/test_routing.py -q`.

## 4. Privacy and Prompt Safety

- [x] 4.1 Implement the typed redaction result and detector registry; verify deterministic overlap precedence with `uv run pytest tests/unit/safety/test_redactor_core.py -q`.
- [x] 4.2 Add email and international/Chinese phone recognition; verify positive and negative bilingual fixtures with `uv run pytest tests/unit/safety/test_contact_redaction.py -q`.
- [x] 4.3 Add Chinese ID and US SSN recognition with format/checksum validation; verify with `uv run pytest tests/unit/safety/test_identity_redaction.py -q`.
- [x] 4.4 Add payment-card recognition with separator normalization and Luhn validation; verify plausible and false-positive fixtures with `uv run pytest tests/unit/safety/test_payment_redaction.py -q`.
- [x] 4.5 Add IPv4/IPv6 and common API-key, bearer-token, password, and private-key recognition; verify full-value masking with `uv run pytest tests/unit/safety/test_network_secret_redaction.py -q`.
- [x] 4.6 Implement recursive output redaction for answers, citations, metadata, errors, diagnostics, and report objects; verify with `uv run pytest tests/unit/safety/test_output_redaction.py -q`.
- [x] 4.7 Implement allowlisted telemetry filtering that drops unsafe events on redaction failure; verify captured JSON contains no fixture values with `uv run pytest tests/unit/safety/test_telemetry_filter.py -q`.
- [x] 4.8 Implement basic intent-aware user injection checks and retrieved-context instruction isolation; verify bypass, hidden-context, quoted-analysis, and document-injection cases with `uv run pytest tests/unit/safety/test_injection_policy.py -q`.
- [x] 4.9 Implement complete-response buffering and fail-closed validation first; add sentence-level emission only as a measured optimization, preserving cross-delta detector state; verify split email, phone, card, IP, and private-key cases with `uv run pytest tests/unit/safety/test_safe_stream.py -q`.

## 5. Knowledge Ingestion

- [x] 5.1 Implement upload size, extension, MIME, filename, emptiness, and safe-storage validation; verify rejected uploads leave no artifacts with `uv run pytest tests/unit/ingestion/test_upload_validation.py -q`.
- [x] 5.2 Implement persistent ingestion-job state transitions and safe stage diagnostics; verify terminal state survives repository reopen with `uv run pytest tests/unit/ingestion/test_jobs.py -q`.
- [x] 5.3 Implement deterministic UTF-8 text extraction and Markdown heading locators; verify Chinese/English preservation with `uv run pytest tests/unit/ingestion/test_text_extractors.py -q`.
- [x] 5.4 Implement page-level PyMuPDF native extraction, page metadata, and encrypted/corrupt PDF errors; verify generated PDF fixtures with `uv run pytest tests/unit/ingestion/test_pdf_extractor.py -q`.
- [x] 5.5 Implement the versioned page-usability decision and Tesseract Chinese/English OCR adapter; verify OCR is requested only for insufficient pages with `uv run pytest tests/unit/ingestion/test_ocr.py -q`.
- [x] 5.6 Implement mixed digital/scanned PDF orchestration and all-pages-empty failure; verify page order and extraction-method diagnostics with `uv run pytest tests/unit/ingestion/test_pdf_pipeline.py -q`.
- [x] 5.7 Implement Unicode NFC normalization, header/footer cleanup, and stable canonical content digest; verify idempotence with `uv run pytest tests/unit/ingestion/test_normalization.py -q`.
- [x] 5.8 Implement bounded page/structure-aware chunking with overlap and stable chunk IDs; verify deterministic locators and no lost content with `uv run pytest tests/unit/ingestion/test_chunking.py -q`.
- [x] 5.9 Implement source-key deduplication, monotonically increasing versions, and retained canonical source artifacts; verify duplicate and update behavior with `uv run pytest tests/unit/ingestion/test_versioning.py -q`.
- [x] 5.10 Implement batched document embedding plus content-digest embedding cache; verify duplicate chunks avoid provider calls with `uv run pytest tests/unit/ingestion/test_embedding_stage.py -q`.
- [x] 5.11 Implement immutable Chroma revision staging and persistent BM25 snapshot creation with bilingual tokenization; verify restart persistence with `uv run pytest tests/integration/test_index_staging.py -q`.
- [x] 5.12 Implement dense/lexical chunk-parity validation and atomic active-manifest publication; verify partial failures preserve the prior revision with `uv run pytest tests/integration/test_index_publication.py -q`.
- [x] 5.13 Implement full ingestion orchestration, same-source update ordering, reindex, deletion, and startup recovery; verify end-to-end state transitions with `uv run pytest tests/integration/test_ingestion_service.py -q`.

## 6. Retrieval and Ranking

- [x] 6.1 Implement query validation, Unicode canonicalization, mode selection, and active-revision snapshot binding; verify invalid and concurrent-publication cases with `uv run pytest tests/unit/retrieval/test_request_context.py -q`.
- [x] 6.2 Implement compatible query embedding and Chroma dense search with deterministic tie-breaking; verify with `uv run pytest tests/unit/retrieval/test_dense.py -q`.
- [x] 6.3 Implement persistent BM25 search using the versioned Chinese/English tokenizer; verify exact English terms and unspaced Chinese fixtures with `uv run pytest tests/unit/retrieval/test_bm25.py -q`.
- [x] 6.4 Implement parallel dense/BM25 collection and candidate merge by stable chunk ID; verify ranks and raw scores are preserved with `uv run pytest tests/unit/retrieval/test_hybrid_collection.py -q`.
- [x] 6.5 Implement weighted Reciprocal Rank Fusion and deterministic tie-breaking; verify formula examples with `uv run pytest tests/unit/retrieval/test_rrf.py -q`.
- [x] 6.6 Implement bounded reranking orchestration and RRF fallback on timeout or invalid output; verify with `uv run pytest tests/unit/retrieval/test_rerank_stage.py -q`.
- [x] 6.7 Implement `dense`, `hybrid`, and `hybrid-rerank` orchestration with configured one-retriever degradation; verify each mode with `uv run pytest tests/unit/retrieval/test_retrieval_service.py -q`.
- [x] 6.8 Implement ranked evidence assembly with source/version/locator and real stage scores only; verify PDF and text evidence with `uv run pytest tests/unit/retrieval/test_evidence.py -q`.
- [x] 6.9 Ensure retrieval works with caches disabled, then implement version-complete keys and TTL/size bounds for each cache actually enabled by the accepted configuration; verify corpus/config changes miss prior entries with `uv run pytest tests/unit/retrieval/test_cache.py -q`.
- [x] 6.10 Add a persistent-index retrieval integration suite covering bilingual semantic, exact-term, empty, degraded, and restart cases; verify with `uv run pytest tests/integration/test_retrieval_pipeline.py -q`.

## 7. Grounded QA Pipeline

- [x] 7.1 Implement isolated conversation-session creation, turn storage, reset, and ownership checks; verify with `uv run pytest tests/unit/qa/test_sessions.py -q`.
- [x] 7.2 Implement latest-turn language selection and conditional follow-up query rewriting that never treats assistant history as evidence; verify with `uv run pytest tests/unit/qa/test_query_rewrite.py -q`.
- [x] 7.3 Implement context selection with chunk-count, per-chunk, and total-token bounds; verify ordering and truncation with `uv run pytest tests/unit/qa/test_context_builder.py -q`.
- [x] 7.4 Implement the versioned generator prompt that labels retrieved chunks untrusted and requires structured claims with chunk IDs; verify prompt boundaries without snapshotting secret configuration with `uv run pytest tests/unit/qa/test_prompt_builder.py -q`.
- [x] 7.5 Implement structured answer parsing and deterministic citation existence/locator validation; verify invented and stale citations are rejected with `uv run pytest tests/unit/qa/test_citations.py -q`.
- [x] 7.6 Implement deterministic factual-unit citation coverage and request-scoped candidate validation; withhold the complete generated response on invalid coverage and leave semantic scoring to evaluation; verify invalid claims/citations never reach output with `uv run pytest tests/unit/qa/test_grounding.py -q`.
- [x] 7.7 Implement calibrated insufficient/conflicting-evidence refusal decisions with stable safe reason codes; verify answerable, partial, absent, and conflict fixtures with `uv run pytest tests/unit/qa/test_refusal_policy.py -q`.
- [x] 7.8 Implement the async QA orchestrator with one total deadline, stage budgets, cancellation, retry limits, optional rerank degradation, and an injected `FactEvidenceAssessor`; verify with fake-clock tests using `uv run pytest tests/unit/qa/test_orchestrator.py -q`.
- [x] 7.8a Implement and calibrate the versioned production `FactEvidenceAssessor` before end-to-end QA integration; bind signals to current request candidates/revision, detect material conflicts without treating raw dense/BM25/RRF scores as interchangeable probabilities, and verify with `uv run pytest tests/unit/qa/test_evidence_assessor.py -q`.
- [x] 7.9 Connect complete-response validated emission to grounding, citation, injection, and redaction gates; add sentence-level validated events only if needed for perceived latency; verify no raw model delta or pending unsafe tail is emitted with `uv run pytest tests/unit/qa/test_streaming.py -q`.
- [x] 7.10 Add end-to-end QA integration tests for bilingual answers, multi-turn retrieval, citations, partial answers, refusals, injection, PII, provider failure, and deadline failure; verify with `uv run pytest tests/integration/test_qa_pipeline.py -q`.

## 8. HTTP API and Gradio Workbench

- [x] 8.1 Implement document upload/list/job-status/delete/reindex API routes with shared schemas; verify status codes and atomic behavior with `uv run pytest tests/api/test_documents.py -q`.
- [x] 8.2 Implement the QA API route and validated streaming response contract; verify answer, refusal, error, cancellation, and malformed-event cases with `uv run pytest tests/api/test_qa.py -q`.
- [x] 8.3 Implement evaluation start/status/report and safe request-diagnostics API routes; verify with `uv run pytest tests/api/test_evaluation_diagnostics.py -q`.
- [x] 8.4 Mount Gradio at the configured FastAPI path and add the four primary tabs; verify route coexistence and component labels with `uv run pytest tests/ui/test_mount.py -q`.
- [x] 8.5 Implement Chat controls, mode selection, validated updates, inline citations, source previews, reset, and cancel; verify callback behavior with `uv run pytest tests/ui/test_chat.py -q`.
- [x] 8.6 Implement Documents upload, progress, metadata, reindex, and confirmed deletion views; verify callbacks with `uv run pytest tests/ui/test_documents.py -q`.
- [x] 8.7 Implement Evaluation run/compare/failure-table/download views and Diagnostics health/request-trace views; verify redacted callbacks with `uv run pytest tests/ui/test_evaluation_diagnostics.py -q`.
- [x] 8.8 Add a Gradio client smoke test for session isolation and safe UI error states; verify with `uv run pytest tests/ui/test_workbench_smoke.py -q`.

## 9. Observability, Cost, and Performance Controls

- [ ] 9.1 Configure Structlog JSON output, request/trace correlation middleware, and safe exception categories; verify event schema with `uv run pytest tests/unit/observability/test_logging.py -q`.
- [ ] 9.2 Expose bounded-cardinality Prometheus counters, gauges, and histograms for QA outcomes, concurrency, caches, stages, tokens, cost, and degradation; verify with `uv run pytest tests/unit/observability/test_metrics.py -q`.
- [ ] 9.3 Add OpenTelemetry root and stage spans with async context propagation and no content attributes; verify with an in-memory exporter using `uv run pytest tests/unit/observability/test_tracing.py -q`.
- [ ] 9.4 Instrument ingestion, retrieval, provider, QA, safety, and evaluation stages with consistent request/run IDs and timings; verify span/log correlation with `uv run pytest tests/integration/test_observability.py -q`.
- [ ] 9.5 Implement versioned model pricing lookup and per-attempt/request/run cost aggregation; verify unknown usage/pricing and per-1,000-call calculation with `uv run pytest tests/unit/observability/test_costs.py -q`.
- [ ] 9.6 Implement a QA admission controller supporting at least five active pipelines and a bounded queue; verify five overlap while excess work is rejected with `uv run pytest tests/unit/performance/test_admission.py -q`.
- [ ] 9.7 Move synchronous Chroma, BM25, OCR, and report work to bounded worker pools; verify the event loop remains responsive with `uv run pytest tests/unit/performance/test_worker_pools.py -q`.
- [ ] 9.8 Implement the 9.5-second total deadline and configurable stage latency budgets with reranker degradation; verify fake-provider timing cases with `uv run pytest tests/unit/performance/test_deadlines.py -q`.
- [ ] 9.9 Implement safe persisted request diagnostics and lookup by request ID; verify retention bounds and redaction with `uv run pytest tests/unit/observability/test_diagnostics.py -q`.
- [ ] 9.10 Add a deterministic single-process concurrency integration test proving five QA pipelines make overlapping provider calls; verify with `uv run pytest tests/integration/test_five_concurrent_qa.py -q`.

## 10. RAG Evaluation and Issue Evidence

- [ ] 10.1 Implement versioned dataset/manifest loading, content hashing, corpus-version checks, and category eligibility validation; verify with `uv run pytest tests/unit/evaluation/test_dataset.py -q`.
- [ ] 10.2 Add non-sensitive bilingual sample documents and an initial answerable Chinese/English case set with authoritative chunk mappings; validate with `uv run python -m rag_mvp.evaluation.validate_dataset evaluations/datasets/mvp-v1`.
- [ ] 10.3 Extend the same dataset with multi-turn, OCR, unanswerable, required-refusal, injection, and PII categories; rerun the dataset validator and confirm every required category is eligible.
- [ ] 10.4 Implement the evaluation runner through the production QA pipeline with isolated sessions, disabled final caches, progress persistence, and immutable run manifest; verify with `uv run pytest tests/unit/evaluation/test_runner.py -q`.
- [ ] 10.5 Implement versioned Faithfulness and Context Precision scoring with per-case rationale/evidence; verify formulas and strict eligibility with `uv run pytest tests/unit/evaluation/test_grounding_metrics.py -q`.
- [ ] 10.6 Implement Answer Completeness, Style Consistency, and Refusal Appropriateness scoring; verify boundary and inappropriate-refusal cases with `uv run pytest tests/unit/evaluation/test_answer_metrics.py -q`.
- [ ] 10.7 Implement the unrounded quality gate for `>0.85`, `>0.70`, `>=80%`, `>=80%`, and `>=80%`; verify every boundary operator with `uv run pytest tests/unit/evaluation/test_quality_gate.py -q`.
- [ ] 10.8 Implement versioned JSON report generation and JSON Schema validation; verify required provenance, metric, performance, cost, privacy, and issue sections with `uv run pytest tests/unit/evaluation/test_json_report.py -q`.
- [ ] 10.9 Implement Jinja2 HTML rendering from JSON and automated value-parity checks; verify with `uv run pytest tests/unit/evaluation/test_html_report.py -q`.
- [ ] 10.10 Run the first real baseline evaluation, persist its immutable report and manifests, and verify it with `uv run python -m rag_mvp.evaluation.verify_report <baseline-report.json>`.
- [ ] 10.11 Select the first genuine baseline issue, or an explicitly test-only controlled compliance baseline if needed; attach affected cases, config delta, safe logs/metrics/traces, root cause, and proposed fix to the report.
- [ ] 10.12 Implement the smallest justified fix for issue one, rerun the identical case set, and verify its declared primary metric improves at least 10% relative without failing global gates.
- [ ] 10.13 Select and evidence a distinct second genuine issue, or an explicitly test-only controlled refusal baseline if needed; verify it uses the same dataset/corpus/scorers as its planned post-fix run.
- [ ] 10.14 Implement the smallest justified fix for issue two, rerun the identical case set, and verify its declared primary metric improves at least 10% relative without regressing accepted issue one.
- [ ] 10.15 Generate the final combined JSON/HTML validation report and verify both issue records, calculations, configuration identities, and report parity with `uv run python -m rag_mvp.evaluation.verify_report <final-report.json>`.

## 11. Load Evidence and Local Container Deployment

- [ ] 11.1 Implement an HTTP load-test harness with warm-up, fixed five-user concurrency, cache-bypass headers, at least 500-success validation, nearest-rank percentiles, and error accounting; verify parser logic with `uv run pytest tests/unit/performance/test_load_report.py -q`.
- [ ] 11.2 Implement the machine-readable performance evidence bundle with run/config/model IDs, warm-up, attempts, successes, errors, latency, stages, tokens, cost, and representative trace references; verify schema with `uv run pytest tests/unit/performance/test_evidence_bundle.py -q`.
- [ ] 11.3 Add a multi-stage non-root `Dockerfile`, pinned runtime command, OCI labels, and `.dockerignore`; verify build and UID with `docker build -t rag-mvp:dev . && docker run --rm --entrypoint id rag-mvp:dev`.
- [ ] 11.4 Add Docker Compose with exactly one app instance, loopback-default ingress, persistent data volume, runtime secrets, and health checks; validate with `docker compose config`.
- [ ] 11.5 Implement exclusive data-root writer locking and graceful shutdown/drain behavior; verify competing-writer rejection and termination cleanup with `uv run pytest tests/integration/test_lifecycle.py -q`.
- [ ] 11.6 Add `.env.example` and a local runbook covering build, start, health, sample ingestion, smoke QA, logs, backup, restore, and stop; execute every command in the runbook against a clean volume.
- [ ] 11.7 Verify volume persistence by ingesting sample documents, recreating the container, and running the same grounded smoke query without re-ingestion; retain the smoke output as release evidence.
- [ ] 11.8 Scan the image for embedded secrets and unresolved critical vulnerabilities; verify release checks with the documented secret scanner and `trivy image --exit-code 1 --severity CRITICAL rag-mvp:dev`.

## 12. Final Acceptance

- [ ] 12.1 Run formatting, lint, typing, unit, API, integration, UI, and privacy suites from a clean environment with `uv run ruff format --check . && uv run ruff check . && uv run mypy src && uv run pytest`.
- [ ] 12.2 Run the complete privacy corpus and scan captured output, logs, traces, diagnostics, and reports; verify zero raw supported PII or secret fixture matches.
- [ ] 12.3 Run the final versioned RAG evaluation against the candidate image; verify Faithfulness `>0.85`, Context Precision `>0.70`, Completeness `>=80%`, Style `>=80%`, and Refusal Appropriateness `>=80%`.
- [ ] 12.4 Confirm model quota and estimated acceptance-run cost, warm one container, then execute the uncached five-concurrent-request load run until at least 500 requests succeed.
- [ ] 12.5 Verify the load evidence reports exactly one instance, P90 complete latency `<=10s`, error rate `<1%`, no hidden failed attempts, and correlated safe log/metric/trace evidence.
- [ ] 12.6 Publish the accepted image digest, configuration manifest, dataset/corpus versions, final JSON/HTML report, performance bundle, cost per 1,000 calls, known limitations, and exact reproduction commands in the release README.
