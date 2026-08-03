## Context

The repository currently contains only the case-study source material and this OpenSpec change. The product must therefore establish a small but complete foundation rather than integrate with an existing application. Its primary users are employees asking questions over bilingual internal documents, knowledge administrators maintaining the corpus, and reviewers verifying the system's quality, performance, cost, safety, and issue diagnosis.

The MVP must ingest digital and scanned PDFs, answer only from retrieved evidence, provide a simple Gradio UI, and produce reproducible validation evidence. It must satisfy a strict single-instance target: at least five concurrent uncached QA requests, at least 500 successful requests in the acceptance run, P90 complete response time at most 10 seconds, and error rate below 1 percent. It also must meet the quality thresholds defined in `rag-evaluation` and redact basic PII from all output and telemetry.

This change targets local execution and Docker Compose. Azure deployment is intentionally deferred. The design still externalizes provider, data-root, port, secret, and telemetry settings so a later managed-cloud change does not require rewriting the RAG domain.

## Goals / Non-Goals

**Goals:**

- Deliver an end-to-end product loop: ingest knowledge, ask questions, inspect evidence, run evaluations, diagnose failures, and download validation reports.
- Ground Chinese and English answers in active indexed chunks with verifiable citations and appropriate refusal.
- Make dense retrieval, hybrid retrieval, and reranking configurable and measurable.
- Support OpenAI-compatible embeddings, optional LLM listwise reranking, and answer generation through replaceable asynchronous interfaces.
- Meet the quality, privacy, single-instance concurrency, latency, and cost-reporting requirements with reproducible evidence.
- Package one secure, persistent, locally deployable container while retaining future provider, retriever, storage, and telemetry extension points.

**Non-Goals:**

- Azure resources, cloud identity, cloud ingress, managed secrets, or cloud deployment automation.
- Production multi-tenancy, enterprise RBAC, or per-document authorization beyond a future-facing scope field.
- Horizontally scaling embedded Chroma or supporting multiple concurrent index writers.
- Full GraphRAG indexing, graph extraction, or graph traversal in the MVP.
- Broad office-format support, video/audio ingestion, web crawling, or automatic external data synchronization.
- Perfect semantic PII recognition for names and postal addresses; the MVP covers the deterministic classes listed in `privacy-and-safety`.
- Autonomous tools, web access, code execution, or actions triggered from retrieved content.

## Decisions

### 1. Use a Python modular monolith with FastAPI and mounted Gradio

One Python 3.12 process will expose FastAPI routes and mount Gradio on `/workbench`. Domain packages will be separated by responsibility:

```text
src/rag_mvp/
  api/             # FastAPI routes and schemas
  ui/              # Gradio composition and adapters
  config/          # validated settings and version identities
  domain/          # provider-neutral models and protocols
  ingestion/       # extraction, OCR, chunking, publication
  providers/       # OpenAI-compatible adapters and test fakes
  retrieval/       # Chroma, BM25, RRF, reranking, cache
  qa/              # conversation, orchestration, grounding, citations
  safety/          # injection policy and PII redaction
  observability/   # logs, metrics, traces, usage/cost
  evaluation/      # datasets, scorers, comparisons, reports
  storage/         # SQLite metadata, manifests, file layout
```

Rationale: the workload is small, mostly I/O-bound, and benefits from Python's document, ML, evaluation, and Gradio ecosystem. A modular monolith avoids network hops and deployment complexity while retaining interfaces at likely future seams.

Alternatives considered: a separate React frontend would improve UI control but add disproportionate MVP work; separate ingestion and QA services would improve independent scaling but complicate atomic index publication and operations; Node/Bun would fit general web development but has weaker native support for the selected RAG/OCR/evaluation libraries.

### 2. Keep HTTP contracts separate from the Gradio view layer

Core routes will include:

```text
POST   /api/v1/documents
GET    /api/v1/ingestion-jobs/{job_id}
GET    /api/v1/documents
DELETE /api/v1/documents/{source_id}
POST   /api/v1/index/rebuild
POST   /api/v1/qa
POST   /api/v1/evaluations
GET    /api/v1/evaluations/{run_id}
GET    /api/v1/reports/{run_id}.{json|html}
GET    /api/v1/diagnostics/requests/{request_id}
GET    /healthz
GET    /readyz
GET    /metrics
```

Gradio callbacks call application services directly in-process, while tests and load tools use the HTTP API. Response schemas use stable outcome enums and machine-readable citations rather than UI-specific objects.

Rationale: this preserves a simple MVP deployment while allowing Gradio to be replaced or an external client to be added later.

Alternative considered: only Gradio callbacks would be faster initially but would make realistic load testing, automation, and future UI replacement harder.

### 3. Persist metadata in SQLite and content/index data under one versioned data root

SQLite will hold documents, document versions, ingestion jobs, index revisions, sessions, request diagnostics, evaluation runs, provider usage, and report manifests. Files below `DATA_ROOT` will hold uploaded source payloads, canonical extraction artifacts, BM25 snapshots, Chroma persistence, caches, and generated reports.

An `index_manifest` identifies active source versions, chunk-set digest, embedding-space identity, extraction/chunking/tokenizer versions, and dense/lexical storage locations. Ingestion builds a staged revision, validates Chroma/BM25 chunk parity, commits metadata in SQLite, and atomically replaces the active-manifest pointer. Retrieval binds to the pointer at request start.

Rationale: SQLite and a filesystem volume are transparent, backup-friendly, and sufficient for one instance. Immutable revisions make failed ingestion recoverable and prevent mixed dense/lexical results.

Alternatives considered: PostgreSQL plus pgvector would better support replicas but adds infrastructure outside MVP scope; directly mutating one Chroma collection is simpler but makes rollback and parity validation unsafe.

### 4. Extract text page-by-page and OCR only when needed

PyMuPDF will extract native PDF text and metadata. A configurable text-density/character-quality check decides whether a page requires OCR. OCR will use a local Tesseract Chinese/English adapter for the baseline implementation. Markdown and text use deterministic UTF-8 parsers. Canonical text uses Unicode NFC and normalized line endings.

Chunking is structure-aware where headings are available and page-aware for PDFs. Defaults target roughly 500 tokens with 80-token overlap, but both values and tokenizer identity are versioned. Chunk IDs derive from source, document version, ordinal, and content digest.

Rationale: selective OCR controls ingestion cost and preserves reliable page citations. Determinism enables exact reindex and evaluation reproduction.

Alternatives considered: OCR every page is simpler but slower and can degrade digital text; managed OCR can improve difficult scans but would introduce another external provider and cost before the basic workflow is proven.

### 5. Use Chroma plus bilingual BM25 with weighted RRF

The retrieval abstraction exposes `dense`, `hybrid`, and `hybrid-rerank` modes. Chroma stores dense vectors and metadata. BM25 uses a deterministic tokenizer supporting English boundaries and Chinese segmentation. In hybrid modes, dense and lexical retrieval run concurrently, candidates join by chunk ID, and weighted Reciprocal Rank Fusion combines ranks without comparing incompatible raw scores.

The default candidate flow is:

```text
Dense top 20 + BM25 top 20
        -> weighted RRF
        -> top 10 optional rerank
        -> top 5 answer context
```

Exact limits remain configurable and are included in cache and evaluation identities. A `Retriever` protocol admits a future `GraphRetriever` or managed search implementation.

Rationale: dense retrieval covers semantic and cross-language similarity; BM25 recovers identifiers and exact policy terms; RRF is deterministic and robust without score calibration.

Alternatives considered: FAISS is lightweight but requires more metadata/persistence work; Chroma-only is simpler but weaker on exact terms; GraphRAG can improve relationship questions but its extraction, indexing cost, and evaluation complexity are not justified for this MVP.

### 6. Use OpenAI-compatible adapters with explicit compatibility identities

Separate `EmbeddingProvider`, `GenerationProvider`, and `RerankingProvider` protocols use provider-neutral request/result models. The initial adapters use the asynchronous OpenAI client and allow configurable base URLs. Embedding defaults to a small embedding model chosen during configuration. Generation uses a low-latency bounded-output chat model. Optional reranking asks a chat model to return an exact JSON permutation of candidate IDs.

Embedding identity includes provider alias, model, dimension, normalization, and adapter version. An index refuses incompatible query vectors. Every provider attempt records safe model identity, latency, retry/fallback state, and token usage.

Rationale: OpenAI APIs satisfy the requested API-based embedding and reranking path while interfaces preserve future local, Azure-compatible, Cohere, or other providers.

Alternatives considered: a local BGE reranker can reduce token cost but increases image size, CPU contention, and single-instance latency; a dedicated rerank API may be faster but adds a provider. Both can be added behind the protocol after baseline measurement.

### 7. Treat reranking as optional and deadline-aware

Reranking receives only the top fused candidates, truncates candidate text to a configured token budget, has a short sub-deadline, and produces only candidate IDs. Invalid output, provider failure, or timeout falls back to RRF without retrying past the latency budget. Degraded results are marked and excluded from final-result cache writes.

Rationale: LLM reranking may improve Context Precision but is the largest optional threat to the 10-second SLA. Explicit fallback allows an evidence-backed quality/latency trade-off.

Alternative considered: always require reranking would simplify behavior but turn an optional quality enhancer into a QA availability dependency.

### 8. Orchestrate QA as a bounded evidence-first pipeline

The QA sequence is:

1. Validate request, capacity, session, language, and injection policy.
2. Rewrite only contextual follow-ups into a standalone query.
3. Bind an active index revision and retrieve evidence.
4. Apply optional reranking and context/token limits.
5. Refuse when evidence is absent or below a calibrated decision policy.
6. Generate a structured answer containing claims and chunk IDs.
7. Validate citation existence and claim support.
8. Apply injection/output policy and PII redaction.
9. Serialize a structured answer/refusal/error and safe diagnostics.

Conversation history assists query interpretation but is never evidence. The generator prompt wraps chunks as explicitly untrusted context and prohibits outside knowledge. Citation validation is deterministic; claim entailment can use a versioned evaluator model where necessary, bounded by the request deadline.

Rationale: explicit stages make failures diagnosable and enforce the grounding contract outside model instructions.

Alternative considered: a framework-managed agent chain would reduce initial code but obscure deadlines, privacy boundaries, and stage-level evidence.

### 9. Stream validated sentences, not raw tokens

Raw model deltas remain server-side in a bounded sentence buffer. A complete sentence plus citations passes grounding, injection, and PII checks before an event is emitted. Detector state retains enough suffix context to catch PII split across deltas. On ambiguity, buffer overflow, or safety failure, pending text is discarded and a pre-vetted message ends the stream.

Rationale: raw token streaming can leak an email or card-number prefix before detection. Sentence-level streaming retains some perceived responsiveness without violating output-redaction requirements.

Alternatives considered: buffering the complete answer is safest but worsens perceived latency; raw streaming is fastest but cannot meet the privacy contract.

### 10. Minimize telemetry content and redact as a second line of defense

Structlog emits allowlisted JSON events. Prometheus-compatible metrics use bounded labels. OpenTelemetry spans represent each RAG stage. Questions, answers, prompts, chunks, histories, credentials, and unrestricted exceptions are not recorded. Any permitted title, error, or preview passes the same local redactor used for output; unsafe telemetry events are dropped.

The deterministic redactor combines patterns and validators for email, phone, Chinese ID, SSN, payment cards, IPs, and common secret formats. Redaction occurs locally and fails closed. Detection categories and counts may be logged, but values and reversible mappings may not.

Rationale: not collecting content is safer than relying on perfect detection. Local deterministic checks avoid sending sensitive material to another model and provide predictable latency.

Alternatives considered: LLM-based PII classification could detect names/addresses but increases latency, cost, nondeterminism, and data exposure; comprehensive DLP is deferred.

### 11. Enforce concurrency through async I/O, admission control, and one-instance storage ownership

FastAPI handlers and model clients are asynchronous. Synchronous Chroma, BM25, OCR, and report work run in bounded worker pools so they do not block the event loop. Hybrid retrievers run concurrently. A QA semaphore admits at least five active pipelines; a small bounded queue returns a retryable capacity error when full. One process owns embedded Chroma and SQLite writes.

The internal P90 latency budget is 8.5 seconds, leaving margin below the 10-second gate:

| Stage | P90 budget |
|---|---:|
| Admission, validation, safety input check | 0.2 s |
| Query embedding and parallel retrieval | 0.8 s |
| Optional reranking | 1.2 s |
| Generation | 5.3 s |
| Grounding, citation, redaction, serialization | 0.6 s |
| Network/UI margin | 0.4 s |

The total hard deadline is 9.5 seconds. Output tokens, context chunks, candidate counts, retries, and reranking work are bounded. The load test bypasses answer and final-result caches.

Rationale: the selected model calls are I/O-bound, so one async process can support five concurrent requests without unsafe multi-writer storage. An internal margin accommodates network variance.

Alternatives considered: multiple Uvicorn workers improve HTTP capacity but risk embedded-store ownership and duplicate in-memory indexes; multiple containers require a networked vector/metadata architecture and are deferred.

### 12. Version every cache and never use cache to prove the SLA

Embedding, retrieval, reranking, and complete-answer caches have separate schemas and bounded TTL/size. Keys include canonical input plus corpus/index, prompt, model, embedding, BM25, ranking, safety, response-schema, and relevant limit versions. Failures and degraded final results are not cached. Diagnostics distinguish each cache level.

Rationale: caching reduces normal cost and latency, but stale RAG answers are dangerous. Version-complete keys make invalidation explicit, while uncached acceptance testing demonstrates real capacity.

Alternative considered: one answer cache is easier but cannot safely reuse intermediate work or explain invalidation.

### 13. Make evaluation a first-class product workflow

Evaluation datasets are JSONL plus a manifest and immutable source snapshot. Cases include expected facts/evidence, answerability/refusal, language, and categories. The runner invokes the production QA service with caches bypassed and records a complete run manifest. Deterministic metric implementations are preferred; model judges use fixed settings and versioned prompts.

The JSON report is the source of truth. Jinja2 renders HTML from it, and validation checks value parity. A Gradio tab starts runs, displays progress and failures, compares compatible runs, and downloads reports.

Two issue investigations use actual baseline observations when available. If the baseline does not naturally expose two useful issues, controlled test-only configurations may demonstrate, for example, excessive context causing compliance failure and an over-strict refusal threshold causing false refusals. Each is clearly labeled, reproduced on the same dataset before and after one documented fix, correlated with privacy-safe logs/metrics/traces, and required to improve its primary metric by at least 10 percent relative. The final configuration must still pass all global quality and performance gates.

Rationale: this turns the case-study requirements into executable evidence rather than a narrative assembled after development.

Alternative considered: using RAGAS output alone is faster but insufficient for custom style/refusal definitions, reproducibility provenance, and issue-evidence reporting.

### 14. Package one local Docker service with a persistent volume

A multi-stage image installs pinned Python/runtime dependencies and runs as a non-root user. Docker Compose starts exactly one application instance, mounts `DATA_ROOT`, passes model settings and secrets at runtime, and configures `/healthz` and `/readyz`. Startup locking prevents a second writer from opening the same embedded data root. Graceful shutdown stops admission, drains or cancels work, flushes telemetry, and closes stores.

Rationale: one image and volume are easy for reviewers to run and preserve data. External settings and provider protocols keep the application cloud-portable without implementing cloud infrastructure now.

Alternatives considered: Kubernetes or Azure Container Apps would exceed the current scope; baking sample indexes into the image would hurt reproducibility and persistence.

## Risks / Trade-offs

- **OpenAI latency or quota prevents the P90 target** -> Use a low-latency generation model, cap tokens/context, preflight RPM/TPM, warm connections, set sub-deadlines, and degrade optional reranking. Report failures rather than hiding retries.
- **LLM listwise reranking adds more latency than quality** -> Measure all three retrieval modes and make RRF the fallback/default if reranking cannot produce a net accepted result.
- **Embedded Chroma and SQLite limit scaling** -> Enforce one process/replica and one writer. Preserve repository/provider interfaces for a later networked-store migration.
- **Local OCR quality is weak on difficult scans** -> Record page-level extraction method and OCR failures, include OCR cases in evaluation, and preserve an OCR adapter for later replacement.
- **Chinese BM25 tokenization choice changes results** -> Pin tokenizer dictionaries/version in the index manifest and require reindex after changes.
- **Grounding validation may consume deadline or make false refusals** -> Prefer deterministic citation checks, bound model-based validation, calibrate refusal on the versioned dataset, and inspect category-level regressions.
- **Basic regex PII misses names or addresses and may over-redact technical identifiers** -> State supported classes explicitly, use checksum/context validators, maintain bilingual adversarial tests, and defer broader DLP to a separate capability.
- **Sentence buffering reduces streaming immediacy** -> Measure time-to-first-validated-sentence separately while keeping complete-response latency as the contractual metric.
- **Controlled issue baselines could be mistaken for real incidents** -> Label them test-only in configuration and reports, retain exact baseline deltas, and prove the controls are absent from the accepted candidate.
- **Evaluation model nondeterminism weakens reproducibility** -> Use fixed settings, version prompts/models, repeat key runs, record per-case evidence, and define tolerances instead of claiming bit-for-bit identity.
- **The 500-request acceptance run can create material API cost** -> Estimate cost before running, use a representative bounded output length, require explicit acceptance-run confirmation, and retain evidence so it need not be repeated unnecessarily.
- **No authentication in the local MVP exposes internal documents if bound publicly** -> Default bind to loopback, document trusted-network-only use, and treat authentication/public exposure as a required future change before internet deployment.

## Migration Plan

1. Create the project skeleton, pinned dependency files, validated settings, health routes, test fakes, and local data layout.
2. Implement metadata persistence and immutable index manifests without enabling user traffic.
3. Implement extraction, OCR, deterministic chunking, providers, Chroma/BM25 staging, and atomic publication; validate with sample documents.
4. Implement retrieval modes, caches, QA orchestration, citations, safety, and diagnostics behind the API.
5. Add Gradio views using the same services and enable only after API and privacy tests pass.
6. Add evaluation datasets, scorers, reports, load tests, and two paired issue investigations.
7. Build and scan the Docker image, run Compose persistence and lifecycle tests, then execute quality and single-instance performance gates against the exact image.
8. Tag the accepted image/configuration and retain reports, manifests, and reproduction commands.

Rollback consists of stopping the candidate container and restarting the previous image with the same data volume. Index revisions are immutable, so an operator can repoint to the prior validated manifest if a new revision is defective. Before migrations that alter SQLite or manifest schemas, the runbook requires a data-volume backup and documents restoration. Destructive migrations are not permitted in this MVP.

## Open Questions

- Which OpenAI-compatible endpoint and exact embedding, generation, and evaluator model versions will be available for final acceptance?
- What RPM/TPM quota is guaranteed during the five-concurrent-request load run?
- What maximum document size, corpus size, and scanned-page count should the default configuration support?
- May the repository include non-sensitive synthetic bilingual documents derived from the case-study themes for repeatable acceptance?
- Is local Tesseract Chinese/English OCR available in the target Docker environment, or must OCR be optional until language data is supplied?
- Should the first MVP bind only to loopback, or will it be used on a trusted LAN requiring basic authentication in this change?
- Which pricing source and currency should be frozen for the per-1,000-call cost report?
