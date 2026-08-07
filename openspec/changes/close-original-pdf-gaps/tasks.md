## 1. Phase 13 — Acceptance Contract and Dataset

- [x] 1.1 Add schema-v2 domain models for metric observations, gate results, acceptance contracts, operations summaries, artifact descriptors, and explicit unavailable values while retaining read-only v1 adapters.
- [x] 1.2 Extend the evaluation dataset schema with versioned compliance obligations, challenge tags, refusal guidance expectations, language coverage, and fail-fast coverage validation before provider calls.
- [x] 1.3 Add versioned technical-specification, architecture, distractor, exact-identifier, cross-language, and scanned-document corpus assets and generate immutable source/corpus manifests without changing `mvp-v1`.
- [x] 1.4 Author and validate at least 24 acceptance cases meeting the Chinese, English, multi-turn, retrieval-challenge, rerank-sensitive, permitted-source, expected-fact, and compliance-obligation requirements.
- [x] 1.5 Implement the independent all-obligations Answer Compliance scorer, aggregate numerator/denominator evidence, and a regression fixture where Completeness passes but Compliance fails.
- [x] 1.6 Add schema-v2 quality aggregation and gate profiles using unrounded Faithfulness, Context Precision, Answer Compliance, Style, and Refusal Appropriateness values with non-zero denominators and the advanced thresholds.
- [x] 1.7 Extend performance evidence and semantic validation with warm-up separation plus all-attempt and successful-only p50/p90/p95/p99 summaries recomputed from the immutable attempt ledger.
- [x] 1.8 Extend cost evidence with all measured provider attempts, role/direction token totals, exact pricing provenance, cost per 1,000 logical attempts, cost per 1,000 successes, and fail-closed unknown handling.
- [x] 1.9 Add immutable experiment-plan and registry models covering the declared axis, fixed identities, variants, baseline, repeat/order policy, cache policy, call/cost caps, and deterministic selection policy.
- [x] 1.10 Update JSON schema, JSON/HTML rendering, parity checks, fixtures, and unit tests for v2 evidence while proving Phase 12 v1 reports remain unchanged and readable.
- [x] 1.11 Run Phase 13 dataset/schema/scorer/report/performance/cost tests plus Ruff, mypy, and relevant regression gates, then commit the completed phase as `phase 13: define original PDF acceptance contract`.

## 2. Phase 14 — Runtime Gaps and Operational UI

- [ ] 2.1 Inject the bounded TTL/LRU retrieval cache into production composition and implement `USE` hit/miss/write plus `BYPASS` behavior using the complete revision/configuration identity.
- [ ] 2.2 Add fail-open cache handling and privacy-safe eligible/hit/miss/bypass/expiry/eviction/error counters with correct cache-rate denominators.
- [ ] 2.3 Add unit and integration tests proving provider-call elimination on hits, output equivalence, version/configuration isolation, expiry/eviction, non-caching of failures/degradation, bypass behavior, and cache-error fallback.
- [ ] 2.4 Add a versioned Chinese/English refusal-guidance catalog for low-confidence, out-of-scope, conflicting-evidence, prompt-injection, and safety reason codes.
- [ ] 2.5 Integrate deterministic guidance into terminal QA refusals and add first-turn, multi-turn, language, grounding, injection, PII, telemetry, Refusal Appropriateness, and Answer Compliance tests.
- [ ] 2.6 Implement the canonical operations model plus deterministic TXT and CSV renderers with identity, p50/p95, token, cache, refusal, compliance, cost, denominator, unknown-value, parity, and privacy validation.
- [ ] 2.7 Implement immutable multi-format artifact publication and manifest verification for JSON, HTML, TXT, and CSV, including safe filenames, media-type allowlists, hashes, sizes, relative paths, and no-overwrite behavior.
- [ ] 2.8 Add the structured-log field dictionary, privacy-safe JSONL sample, and automated checks for field coverage, units/types, redaction rules, prohibited fields, secrets, paths, and supported PII.
- [ ] 2.9 Extract a shared evaluation application service from the CLI and add a bounded background supervisor, persisted catalog/progress, safe terminal errors, active-job limits, and startup reconciliation of interrupted work.
- [ ] 2.10 Implement production API schemas/routes for dataset and plan catalogs, evaluation list/start/get/summary/failed cases, artifact manifests/downloads, stable conflict/capacity errors, and validated no-store/nosniff responses.
- [ ] 2.11 Wire the concrete evaluation service through application composition and lifespan shutdown, and make the CLI use the same service without changing normal QA readiness or the active online index.
- [ ] 2.12 Replace the placeholder Evaluation UI with typed Run, Overview, Operations, and Artifacts secondary views showing bilingual progress, quality/performance/cost/operations denominators, gate explanations, safe failed cases, and API-backed downloads.
- [ ] 2.13 Add API, UI, integration, restart, concurrency, privacy, and download tests proving explicit start is non-blocking, refresh is read-only, production evaluation is wired, missing evidence is not shown as success, and no filesystem path is exposed.
- [ ] 2.14 Run the complete Phase 14 focused and repository-wide tests, static analysis, privacy scans, and OpenSpec validation, then commit the completed phase as `phase 14: implement PDF requirement gaps and evaluation UI`.

## 3. Phase 15 — Controlled Comparisons and Comparison UI

- [ ] 3.1 Add persisted comparison/suite states, candidate references, progress, partial-failure handling, immutable plan hashes, and historical list/get repositories.
- [ ] 3.2 Implement candidate execution through normal evaluation plans with isolated data roots when identities differ, safe immutable-index reuse when identities match, seeded ordering, cache bypass, and bounded provider work.
- [ ] 3.3 Implement compatibility validation that fixes every controlled identity, permits only the declared experimental axis, reports precise incompatibilities, and does not require equal configuration IDs.
- [ ] 3.4 Implement comparison aggregation, absolute/baseline-delta metrics, challenge-category results, reranker invocation/discrimination proof, gate profiles, deterministic recommendations, and no-recommendation reasons.
- [ ] 3.5 Add registered model, dense/hybrid/hybrid-rerank, and cold/warm cache experiment plans with exact provider/model variants, fixed datasets/corpora/scorers/prompts, cost caps, and predeclared selection policies.
- [ ] 3.6 Add comparison catalog/list/start/get/artifact API routes that accept only registered plan IDs, return 202 with progress locations, and preserve failed candidates and safe error evidence.
- [ ] 3.7 Implement the Compare secondary UI with persisted experiment selectors, controlled-dimension and compatibility displays, authoritative candidate/delta tables, compact plots, category drill-down, gate states, and recommendation rationale.
- [ ] 3.8 Add unit, API, UI, integration, privacy, compatibility, cost-cap, partial-failure, non-discriminating-reranker, restart-history, and cross-format comparison-report tests.
- [ ] 3.9 Preflight and execute the real generation-model comparison, validate aligned quality/latency/token/cost/error evidence, and select a model only through the registered decision policy.
- [ ] 3.10 Using the selected model, execute the real dense, hybrid, and hybrid-rerank matrix with cache bypass and verify a real reranker call plus a discriminating rerank-sensitive case.
- [ ] 3.11 Execute the separate cold/warm cache experiment and validate hit rate, provider-call reduction, latency delta, corpus revision, and cached/uncached output equivalence without using it for the official SLA.
- [ ] 3.12 Verify that model, retrieval, and cache results, deltas, conclusions, provenance, failures, and JSON/HTML evidence are visible and downloadable from the workbench after an application restart.
- [ ] 3.13 Run the complete Phase 15 repository, UI, API, integration, privacy, report-integrity, and OpenSpec gates, then commit the completed phase as `phase 15: publish model and retrieval comparisons in UI`.

## 4. Phase 16 — Final Acceptance and Release v2

- [ ] 4.1 Add the documented `uv run python -m rag_mvp.acceptance` workflow that performs preflight validation, invokes the shared registered plans, evaluates every gate, preserves diagnostics, and exits non-zero on missing or failed evidence.
- [ ] 4.2 Implement a unique, non-overwriting release-v2 builder and staging/sealing lifecycle that cannot modify the Phase 12 release or a prior v2 run.
- [ ] 4.3 Validate the acceptance command from a clean configured environment before provider calls, including dataset/corpus/plan hashes, exact model pricing, service readiness, call/cost caps, output location, and required privacy tooling.
- [ ] 4.4 Execute and validate the selected configuration's full bilingual quality evaluation with the five independent advanced metrics, required denominators, category results, refusal guidance, and gate evidence.
- [ ] 4.5 Execute and validate the single-instance performance run with observed concurrency at least five, at least 500 successes, measured error rate below one percent, all-attempt P90 at most 10 seconds, successful-only supplementary latency, tokens, and both cost denominators.
- [ ] 4.6 Generate and parity-check the canonical operations record, TXT/CSV summaries, JSON/HTML reports, artifact descriptors, log dictionary, privacy-safe sample logs, pricing evidence, and all-request attempt ledger.
- [ ] 4.7 Package the real model/retrieval/cache comparison plans and reports, selected configuration rationale, Phase 10 two-issue evidence, limitations, and run/report/trace/metric references into the release-v2 bundle.
- [ ] 4.8 Start the packaged application against sealed persisted evidence and verify that Run, Overview, Compare, Operations, and Artifacts views show the final metrics, denominators, gates, conclusions, history, and validated downloads without rerunning providers.
- [ ] 4.9 Run the complete injection, grounding, refusal, PII, telemetry, artifact privacy, sample-log, unsafe-path, secret-scan, and release-manifest integrity suites with zero prohibited publishable findings.
- [ ] 4.10 Run Ruff format/check, mypy, the full pytest suite, schema/parity validators, reproducibility checks, container/startup checks, and `openspec validate close-original-pdf-gaps --strict` against the final code and artifacts.
- [ ] 4.11 Audit every requirement in `Asst Manager, Backend Developer,AKP.pdf` against concrete UI/API/code/test/report evidence, document the crosswalk and residual limitations, and seal the unique release-v2 manifest only if every required gate passes.
- [ ] 4.12 Confirm the Phase 16 diff contains the sealed release-v2 evidence without unrelated user files, then commit the completed phase as `phase 16: close original PDF acceptance with UI evidence`.
