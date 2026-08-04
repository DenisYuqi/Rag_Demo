## Why

Employees need a usable way to query bilingual internal knowledge, including scanned PDFs, without accepting unsupported answers as
trustworthy. This change creates an MVP that can be demonstrated, measured against explicit quality and latency targets, diagnosed from
privacy-safe evidence, and run consistently as a local Docker container.

## What Changes

- Add a versioned ingestion workflow for text documents and OCR-backed scanned PDFs.
- Add multi-turn, Chinese/English grounded question answering with source citations and evidence-based refusal.
- Add configurable dense and hybrid retrieval, optional reranking, generation controls, and cache behavior.
- Add OpenAI-compatible providers for embeddings, listwise reranking, and answer generation while keeping provider interfaces replaceable.
- Add a lightweight Gradio workbench for chat, document administration, evaluation, and diagnostics.
- Add basic prompt-injection defenses and deterministic PII redactionfor outputs and logs.
- Add structured logs, distributed traces, stage-level metrics, cost accounting, and bounded concurrency.
- Add a reproducible RAG evaluation harness and HTML/JSON report,including two evidence-backed issue investigations with at least 20%post-fix improvement.
- Add production-oriented Docker packaging, health checks, persistent local storage, and repeatable local deployment instructions.

## Capabilities

### New Capabilities

- `knowledge-ingestion`: Import, OCR, normalize, chunk, version, and index supported internal knowledge documents.
- `model-providers`: Configure replaceable OpenAI-compatible embedding, reranking, and generation providers with usage accounting and safe failure behavior.
- `etrieval-and-ranking`: Execute configurable dense or hybrid retrieval, rank candidates, apply caches, and expose retrieval evidence.
- `grounded-qa`: Answer bilingual multi-turn questions from retrieved evidence, cite sources, and refuse unsupported requests.
- `gradio-workbench`: Provide a simple UI for chat, document management, evaluation, and request diagnostics.
- `privacy-and-safety`: Defend against basic prompt injection and redact supported PII classes from user-visible output and telemetry.
- `performance-observability`: Enforce single-instance concurrency and latency targets while emitting privacy-safe logs, metrics, traces, and token-cost data.
- `rag-evaluation`: Run versioned, reproducible quality and performance evaluations and produce evidence-backed validation reports.
- `container-deployment`: Package the MVP as a secure container and document a repeatable local deployment procedure while preserving cloud-portable boundaries.

### Modified Capabilities

None. This repository has no existing product capabilities.

## Impact

- Introduces a Python 3.12 application composed of FastAPI, Gradio, Chroma, BM25 retrieval, OpenAI-compatible API clients, and an evaluation/reporting module.
- Introduces persisted document, index, evaluation, and operational metadata under a configurable data volume.
- Introduces an external dependency on a configurable OpenAI-compatible API; provider contracts and externalized configuration keep future managed-cloud integrations possible.
- Introduces Docker and Docker Compose configuration, automated tests, a versioned evaluation dataset, and operator documentation.
- Requires API credentials, model deployment names, token quotas, and non-sensitive sample knowledge for acceptance testing.
