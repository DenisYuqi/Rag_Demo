## Context

See `proposal.md` for motivation and the five delta specs for behavior. The existing evaluation runner already executes through the production QA boundary, persists write-once run/case evidence, and pins dataset, corpus, prompt, provider, model, scorer, pricing, retrieval, and cache identities. The JSON/HTML report builders, load-test attempt ledger, token-cost accounting, privacy filters, repositories, and Gradio workbench are also reusable.

The missing architectural link is a production application service. The HTTP routes and `EvaluationGateway` are currently protocols without a concrete service in normal composition, so the Evaluation tab is unavailable outside tests. Its current comparison control accepts two free-text run IDs and renders an arbitrary mapping as JSON Markdown; completed run refresh does not show quality, performance, or cost. The CLI is the only real evaluation entry point, report v1 supports only JSON/HTML, the official performance evidence uses a successful-request latency scope, and enabling the existing retrieval-cache setting currently fails composition.

The application remains a local, single-instance FastAPI and Gradio deployment with an external OpenAI-compatible provider. Evaluation corpora must remain isolated from the active chat index, provider secrets must remain server-side, and `mount_workbench(..., allowed_paths=[])` must remain unchanged.

## Goals / Non-Goals

**Goals:**

- Make CLI, HTTP API, workbench, and release packaging invoke one shared evaluation and comparison application service and read one persisted evidence model.
- Keep paid and long-running evaluation work outside Gradio callbacks and normal QA admission capacity, with bounded execution and monotonic persisted progress.
- Add schema-v2 evidence without rewriting or invalidating Phase 12 artifacts.
- Make every displayed metric traceable to a validated run, denominator, configuration, and artifact hash.
- Preserve strict privacy, grounding, and local deployment boundaries while adding cache and guidance behavior.

**Non-Goals:**

- Building a distributed scheduler, multi-node load generator, hosted experiment platform, or general-purpose BI product.
- Allowing a browser user to enter arbitrary provider URLs, model names, credentials, filesystem paths, Python expressions, or unregistered experiment axes.
- Replacing Gradio, redesigning Chat/Documents/Diagnostics, adding remote multi-tenant authorization, or expanding basic PII detection into a general DLP system.
- Automatically enabling a model or reranker before its controlled comparison and acceptance gates pass.

## Decisions

### 1. Introduce a canonical evidence schema v2 and immutable artifact catalog

Add typed v2 models for metric observations, gate results, acceptance contracts, operations summaries, artifact descriptors, experiment plans, candidate results, comparisons, and recommendations. A metric observation carries value, unit, numerator, denominator, eligibility, threshold/operator, scorer version, and status; missing or unknown data is represented explicitly and never converted to zero.

Each run or comparison owns an immutable directory and a manifest whose artifact descriptors contain opaque artifact ID, format, media type, relative path, SHA-256 digest, byte size, schema version, and creation time. The catalog repository stores only safe searchable metadata and manifest references. JSON remains the canonical rich report; HTML, text, and CSV are deterministic projections. Publication becomes a final step after schema validation, cross-format parity, hash verification, and privacy scanning, rather than a Boolean set by the report builder before files exist.

Existing v1 artifacts remain readable through an adapter that marks unavailable v2-only fields. New code never edits a v1 report in place.

Alternative considered: let each UI panel parse whichever report file it needs. Rejected because duplicated parsing would create conflicting denominators, unsafe path handling, and UI/release drift.

### 2. Extract one evaluation application service and bounded supervisor

Refactor the executable part of the current evaluation CLI into an application service with operations to list the dataset/plan catalog, queue a standard evaluation, queue a registered comparison, list/get summaries, get safe case diagnostics, and resolve validated artifacts. The CLI, API adapter, and in-process workbench gateway call this same service.

Queueing persists the immutable plan and `queued` record before provider calls. A bounded single-process supervisor executes jobs outside request callbacks, reserves capacity separately from QA admission, and permits only a configured number of active paid jobs. Duplicate starts return a stable conflict and capacity exhaustion returns a stable throttling result without changing QA readiness. Terminal exceptions store only safe codes. Startup reconciliation marks stale `queued` or `running` work as failed/interrupted while retaining its partial evidence; an operator may launch a new immutable run rather than risk duplicating unknown provider calls under the old ID.

The API retains the current standard start/get routes and adds list, summary, failed-case, registered-plan, comparison list/start/get, and artifact-manifest operations. Comparison start accepts only `{experiment_plan_id}` from the server-side registry. Downloads accept an opaque run/comparison and artifact identifier, resolve only manifest-allowlisted relative paths, verify digest and media type, and return `no-store` plus `nosniff`. The compatibility `/reports/{run_id}.{format}` route may remain for JSON/HTML while new artifacts use the catalog endpoint.

Alternative considered: have Gradio run the existing CLI in a subprocess. Rejected because it duplicates lifecycle/configuration logic, makes cancellation and progress unreliable, and encourages filesystem-path downloads.

### 3. Represent comparisons as immutable plans composed of normal candidate runs

An `ExperimentPlan` declares one experimental axis, ordered candidate variants, baseline, dataset/corpus/case identities, repeat and seeded order policy, fixed configuration, cache policy, pricing digest, maximum calls/cost, gate profile, and deterministic selection policy. Plans live in a validated registry; the browser receives safe labels and estimates, not mutable provider configuration.

Each candidate is a normal evaluation plan executed by the production QA runner with an isolated run root. Candidates may reuse an immutable installed corpus only when chunking and embedding identities match; otherwise they receive isolated data roots and indexes. A compatibility validator compares all identities except the plan's declared axis. `configuration_id` is expected to differ and is not itself a reason for incompatibility.

The model plan fixes hybrid retrieval and varies the current production generation model against at least one exact alternative. The retrieval plan fixes the selected generation model and contains dense, hybrid, and hybrid-rerank candidates. Candidate/case ordering is seeded and interleaved where practical to reduce temporal provider bias, while persisted timestamps retain evidence of provider drift. All official candidates bypass retrieval cache. The separate cache plan runs equivalent cold/warm traffic and cannot contribute cache hits to SLA evidence.

The comparison builder reads validated candidate artifacts, preserves failed observations, emits absolute values and baseline deltas, verifies a real reranker call and discriminating rerank-sensitive case, and applies the predeclared gates/tie-breaks. It emits no recommendation when evidence is incomplete, incompatible, non-discriminating, or outside a cost cap. Comparison gate profiles do not require the Phase 10 two-issue investigation; only the final acceptance profile does.

Alternative considered: compare arbitrary historical runs. Retained only as a diagnostic compatibility check; it cannot produce an acceptance recommendation because arbitrary runs lack a precommitted axis and decision policy.

### 4. Define Answer Compliance as instruction-obligation satisfaction

Dataset v2 adds versioned, machine-checkable compliance obligations for every eligible answerable case. Obligations cover required or prohibited content, selected language, requested response form, citation behavior, and other explicit case instructions. Deterministic validators evaluate each obligation and retain safe per-obligation evidence. A case is compliant only if every applicable obligation passes; the aggregate is compliant eligible cases divided by eligible answerable cases. Answer Completeness remains a separate expected-fact coverage diagnostic.

The quality gate evaluates unrounded metric values independently, requires a non-zero denominator for every mandatory metric, and does not average away a failed metric. A regression fixture deliberately has complete expected facts but violates a response instruction, proving that Completeness can equal one while Compliance equals zero.

Alternative considered: rename Answer Completeness. Rejected because expected-fact coverage and instruction compliance answer different questions and the audit specifically identified this semantic mismatch.

### 5. Derive performance, cost, and operations views from canonical ledgers

Performance evidence v2 keeps the existing immutable HTTP attempt ledger but derives two latency summaries: all measured HTTP attempts including terminal errors/timeouts, and successful complete HTTP attempts. Warm-up has its own scope. Nearest-rank p50/p90/p95/p99 values are recomputed during validation; the official 10-second P90 gate uses all measured attempts on one instance with configured and observed concurrency of at least five. The selected final configuration runs the full validity sample of at least 500 successes and less than one-percent measured errors; bounded candidate comparisons use the identical repeat policy declared by their plan and are not mislabeled as the final service-level run.

Cost includes every measured provider attempt, including retry and fallback. It records role/direction token totals, exact provider/model rate match, pricing version/digest/source, currency, total, cost per 1,000 logical attempts, and cost per 1,000 successful answers. Missing usage or pricing makes cost unknown and invalidates an acceptance publication instead of being treated as free.

One canonical operations model joins the validated quality report, performance ledger, cost ledger, and cache/refusal counters by run/configuration identity. The text renderer emits a readable key/value report and the CSV renderer emits a stable tabular schema; both undergo exact parity checks against the canonical model. This same model feeds the Operations UI.

### 6. Upgrade the existing Evaluation tab into five typed secondary views

Keep the four current top-level workbench tabs. Inside Evaluation, add:

- **Run / 运行**: registry-backed dataset and run-type selectors, allowlisted variants, immutable identity preview, planned case/candidate/call counts, cache policy, cost estimate/cap, explicit Start, and background progress.
- **Overview / 结果总览**: overall gate banner, quality, all-attempt and successful-only performance, cost/usage, category results, and failed-case tables.
- **Compare / 对比**: completed experiment selector, controlled-dimension summary, aligned candidate table, baseline deltas, compact native Gradio plot, recommendation and measured rationale. The authoritative table accompanies every plot.
- **Operations / 运维**: p50/p95, tokens, cache hits/eligible/rate, refusals/outcomes/rate, compliant/eligible/rate, denominators, and text/CSV preview/download status.
- **Artifacts / 报告下载**: JSON/HTML/TXT/CSV, comparison evidence, performance evidence, plan, manifest, field dictionary, and sample-log descriptors with schema, digest, size, and generation time.

Replace free-text dataset/run/model inputs with dropdowns populated from safe catalogs. Typed UI render models contain only allowlisted scalar/table/plot data; callbacks never dump arbitrary report mappings. Manual refresh remains available and a timer polls only while the selected item is active. Page load and refresh are read-only; provider calls require the explicit Start event.

The production workbench gateway is wired during application composition and reads the same application service as HTTP/CLI. Downloads use same-origin validated API URLs or streamed bytes and never return an arbitrary local path; Gradio `allowed_paths` stays empty. Backend absence, incomplete artifacts, zero denominators, invalid comparisons, and failed jobs receive distinct bilingual states rather than a generic success or zero.

Alternative considered: build a separate dashboard application. Rejected because the current Gradio workbench already has the correct operator context and a second stack would duplicate configuration, security, and deployment work.

### 7. Wire the existing retrieval cache as a fail-open production adapter

Reuse the current version-rich retrieval cache identity and bounded TTL/LRU primitive. Composition injects one cache into the retrieval service only when enabled. A `USE` request builds the identity after binding the immutable snapshot, hashes it, reads the cache, and returns only validated immutable retrieval evidence. It writes only successful, non-cancelled, non-degraded results. `BYPASS` neither reads nor writes. The revision and chunk-set identity prevent cross-version hits; snapshot replacement also clears obsolete entries to reclaim memory.

Cache access is wrapped fail-open so cache corruption or exceptions emit safe counters and continue uncached. Telemetry records hits, eligible lookups, misses, bypasses, expirations/evictions where observable, and errors, but never raw canonical queries. Official comparison and SLA paths already propagate `CachePolicy.BYPASS`; the separate cache experiment uses `USE` explicitly.

Alternative considered: persist retrieval cache in the evaluation database. Rejected for this local single-instance scope because it increases privacy and migration surface without improving the acceptance evidence.

### 8. Centralize refusal guidance by reason and response language

Add a versioned guidance catalog keyed by stable refusal reason and `zh`/`en`. The QA orchestrator selects a bounded template only after the refusal policy decides; generation does not invent guidance. Templates explain the safe reason and offer allowed next actions without naming an owner/source that is not in evidence. Safety and injection reasons use stricter templates that never echo the triggering content or detection rules.

The terminal event retains a language-neutral reason code and adds only a safe guidance/template identifier if needed for evaluation. User-visible content is rendered in the selected response language. Dataset obligations and refusal scoring independently verify reason appropriateness, guidance presence, language, compliance, and privacy.

Alternative considered: ask the generation model to write a helpful refusal. Rejected because it adds latency and creates avoidable grounding, injection, consistency, and privacy risk.

### 9. Version the dataset and corpus instead of mutating MVP evidence

Create a new acceptance dataset/corpus version with at least the specified 24 cases and challenge coverage, including technical specification and architecture documents, exact identifiers, cross-language retrieval, plausible distractors, rerank-sensitive candidates, scanned content, and Chinese/English multi-turn cases. Keep `mvp-v1` and its Phase 12 hashes unchanged. Dataset validation runs before provider calls and rejects insufficient challenge coverage or missing obligations.

Evaluation installation remains isolated from the online active index. The release records source-file, normalized-record, chunk-set, dataset, prompt, and plan hashes so later UI views can prove which evidence they display.

## Risks / Trade-offs

- [Paid comparisons can be slow or unexpectedly expensive] → Show planned calls and available cost estimate/cap, allow only registered plans, require explicit start, bound active jobs, and stop safely before a hard cap.
- [In-process execution can be interrupted by application restart] → Persist plan and per-case progress first, reconcile stale work to an explicit interrupted failure, preserve partial evidence, and never silently reuse the same immutable run ID.
- [Provider drift can bias sequential candidates] → Use identical inputs, seeded/interleaved candidate ordering where practical, record timestamps/provider attempts, and state remaining variance in the report.
- [A small dataset may still fail to distinguish reranking] → Require tagged challenge coverage plus observed reranker invocation and ordering/context change; mark a non-discriminating matrix invalid for an enablement recommendation.
- [Schema v2 can break old report consumers] → Keep v1 readers and routes, add version dispatch, and represent unavailable v2 fields explicitly without rewriting Phase 12.
- [Dashboard complexity can slow or overwhelm Gradio] → Load summary first, fetch case detail lazily, keep tables authoritative, and plot only compact aggregate data.
- [Cache could return stale or privacy-sensitive data] → Use digest keys with full revision/configuration identity, bounded in-memory storage, no raw query telemetry, fail-open behavior, and acceptance bypass.
- [UI downloads could expose the host filesystem] → Resolve opaque artifact IDs through a hash-verifying API allowlist and keep Gradio local path access disabled.

## Migration Plan

1. **Phase 13 — acceptance contract and dataset:** add v2 schemas/models, independent Answer Compliance scoring and gates, dual-denominator performance/cost semantics, expanded dataset/corpus, experiment-plan definitions, and validation fixtures. Preserve all v1 data. Run focused and full static/unit validation, then create the Phase 13 commit.
2. **Phase 14 — close runtime gaps:** wire fail-open retrieval caching, guidance templates, canonical operations model/TXT/CSV renderers, log dictionary/sample, concrete evaluation application service/supervisor, API adapters, production composition, and the standard Run/Overview/Operations/Artifacts UI path. Keep cache disabled by default until its integration gates pass. Run focused integration/privacy tests and the full repository gate, then create the Phase 14 commit.
3. **Phase 15 — controlled comparisons and UI evidence:** implement registered model/retrieval/cache comparison orchestration, compatibility and recommendation logic, Compare UI tables/plots/drill-down, then execute the real model and three-strategy matrices. Store selected configuration and measured rationale only after validation. Run UI/API/integration gates and create the Phase 15 commit.
4. **Phase 16 — final acceptance release v2:** add the one-command workflow, execute the selected configuration's all-attempt load/quality/privacy gates, generate and seal every report/document/sample/manifest, verify clean-environment reproducibility, and publish a unique release-v2 directory without modifying Phase 12. Run the complete release gate and create the Phase 16 commit.

Deployment first exposes the read-only v2 catalog and historical views, then enables registered job launch. Retrieval cache and any newly selected provider configuration remain feature-controlled rollback points. Rollback disables new job starts/cache and restores the prior selected configuration while retaining immutable v2 evidence for diagnosis; no release or evaluation directory is deleted or overwritten.
