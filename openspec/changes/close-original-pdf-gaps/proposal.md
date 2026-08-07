## Why

The Phase 12 release provides a working RAG assistant, but a strict audit against `Asst Manager, Backend Developer,AKP.pdf` found that several acceptance claims are not yet backed by the required runtime behavior or quantitative evidence. The remaining evaluation and comparison evidence must also be visible and actionable in the workbench UI, rather than existing only as offline files or command-line output.

## What Changes

- Define a corrected acceptance contract and expanded bilingual evaluation dataset, including a distinct Answer Compliance metric, advanced quality thresholds, all-request and successful-request latency/cost denominators, and cases that can meaningfully distinguish retrieval and reranking strategies.
- Wire the existing retrieval-cache primitives into the production retrieval path with version-safe invalidation, bypass support, failure isolation, and observable hit/miss behavior.
- Add localized, actionable refusal guidance for low-confidence, out-of-scope, conflicting-evidence, and safety refusals while preserving strict grounding and privacy controls.
- Produce a single operations summary in both text and CSV containing p50/p95 latency, token usage, cache-hit rate, refusal rate, Answer Compliance rate, denominators, and configuration provenance; add a user-facing log-field dictionary and privacy-safe sample logs.
- Extend the Evaluation workbench into an acceptance dashboard where operators can launch standard evaluations and controlled model/retrieval comparisons, inspect progress and history, view quality/performance/cost/operations scorecards and comparison tables, understand gate failures and recommendations, and download JSON, HTML, text, and CSV evidence.
- Run reproducible model-version and retrieval-mode experiments, including vector-only, hybrid, and hybrid-plus-reranking configurations, with fixed datasets and explicit quality, latency, token, cost, error, and degradation comparisons. Surface the resulting conclusions and selected production configuration in the UI.
- Add a one-command acceptance workflow and publish a non-overwriting Phase 16 release-v2 evidence bundle that closes the original PDF requirements and remains reproducible from a clean environment.
- Organize delivery as Phases 13 through 16 with a verification gate and a separate code commit after each completed phase.

## Capabilities

### New Capabilities

- `acceptance-evidence`: Define and generate the complete quality, performance, cost, operations, log-documentation, and release evidence required by the original PDF.
- `controlled-evaluation-comparisons`: Execute reproducible model-version and retrieval-strategy experiment matrices and derive evidence-backed configuration recommendations.
- `evaluation-dashboard`: Run evaluations and comparisons from the workbench and present their progress, metrics, gates, provenance, conclusions, history, and downloadable artifacts in the UI.
- `retrieval-cache-operations`: Cache production retrieval results safely with version-aware keys, bounded lifetime, bypass controls, failure isolation, and observable cache metrics.
- `guided-refusal`: Return bilingual, reason-specific, actionable refusal guidance without weakening grounding, safety, or privacy guarantees.

### Modified Capabilities

None. The repository has no archived base capabilities under `openspec/specs`; this follow-up declares narrowly scoped capabilities that extend the completed MVP change.

## Impact

- Affects the Gradio workbench, UI models/services/callbacks, evaluation and diagnostics APIs, API schemas/composition, evaluation runner and report builders, performance evidence, retrieval service/cache, QA refusal handling, configuration, and persisted evaluation metadata.
- Adds versioned evaluation cases and experiment plans plus JSON/HTML/text/CSV report artifacts, operator documentation, privacy-safe sample logs, and a Phase 16 release-v2 bundle.
- May increase evaluation runtime and provider spend when controlled experiment matrices are launched; the UI and CLI must show the estimated scope and require an explicit run action, while normal chat traffic remains unaffected.
- Reuses the existing FastAPI, Gradio, provider, storage, observability, and report infrastructure and does not require a second UI stack.
