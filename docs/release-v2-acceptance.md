# Release v2 MVP acceptance

`rag-mvp-acceptance` is the single filesystem-based entry point for the original-PDF
submission. It validates the selected BGE configuration and compatible source evidence,
optionally runs one bounded load sample, calculates serving cost, and creates a new Release v2
directory. It does not require acceptance database state, a release API, or Workbench.

## Inputs and compatibility

Copy `evaluations/acceptance/selected-configuration.example.json`, replace its configuration
identity and local hardware assumptions, and keep its functional identities equal to the source
reports. Quality evidence must be complete and match the selected configuration, `bge-local`
profile, `original-pdf-acceptance` dataset, `hybrid-rerank` mode, and
`rag-advanced-quality-thresholds-v2` scorer
contract. Exact model, retrieval/reranker, and cache comparisons must match the same functional
configuration. Historical cross-profile comparisons may be admitted only as explicitly `partial`
supporting evidence with a concrete limitation; they do not satisfy or weaken mandatory quality,
performance, or cost gates. Report age alone does not make compatible evidence stale.

Each admitted source is copied into the submission and recorded with its source identifier and
SHA-256 digest. A missing identity or compatibility field fails preflight because it could change
a mandatory decision.

## Dry run

The dry run validates files, metadata, advanced quality thresholds, output uniqueness, and the
configured maximum cost without contacting a provider. A dry run that uses an existing
performance report needs no service; a dry run with `--run-load` also verifies `/readyz`, the
runtime configuration identity, and instance identity:

```powershell
rag-mvp-acceptance `
  --selected-config evaluations/acceptance/selected-configuration.json `
  --quality-report <quality-report.json> `
  --model-report <model-comparison.json> `
  --retrieval-report <retrieval-reranker-comparison.json> `
  --cache-report <cache-comparison.json> `
  --issue-data evaluations/phase10-issues.json `
  --log-dictionary evaluations/logging/structured-log-field-dictionary-v1.json `
  --log-sample evaluations/logging/privacy-safe-sample-v1.jsonl `
  --performance-report <existing-performance-100.json> `
  --output deliverables/<new-release-id> `
  --dry-run
```

Use a path that does not exist. The command never overwrites a file or directory.

## Bounded 100-request run and submission

Start one warmed application instance with the selected BGE profile and ensure `/readyz` returns
the selected configuration and stable instance identity. Then run:

```powershell
rag-mvp-acceptance `
  --selected-config evaluations/acceptance/selected-configuration.json `
  --quality-report <quality-report.json> `
  --model-report <model-comparison.json> `
  --retrieval-report <retrieval-reranker-comparison.json> `
  --cache-report <cache-comparison.json> `
  --issue-data evaluations/phase10-issues.json `
  --log-dictionary evaluations/logging/structured-log-field-dictionary-v1.json `
  --log-sample evaluations/logging/privacy-safe-sample-v1.jsonl `
  --run-load `
  --base-url http://127.0.0.1:8000 `
  --scenario-file evaluations/performance/acceptance-scenarios-v1.json `
  --run-id bge-pdf-acceptance-100 `
  --max-requests 100 `
  --max-run-cost <approved-currency-amount> `
  --output deliverables/<new-release-id>
```

Preflight computes a conservative upper bound from 100 requests, the selected model's maximum
input/output tokens and price rates, plus the declared BGE hardware allocation. It refuses the run
when that bound exceeds `--max-run-cost`. Readiness and configuration checks happen before any QA
request. After warm-up, the harness sends exactly 100 measured, cache-bypassed logical requests at
configured concurrency five with retries disabled, preserving successes, failures, timeouts, and
elapsed time in one attempt ledger.

The performance gate requires all of the following:

- exactly 100 measured attempts after warm-up;
- one instance identity and cache bypass;
- configured and observed concurrency of at least five;
- at least 90 successful completions within 10,000 ms;
- nearest-rank P90 across all 100 attempts no greater than 10,000 ms.

The operations report also publishes all-attempt P50/P90/P95, success/failure/timeout counts,
observed concurrency, available token usage, serving cost, and selection rationale sources. Cache
hit rate is explicitly unavailable for official traffic because cache bypass is mandatory.

## Cost assumptions

Remote serving cost uses recorded generation input/output tokens and the pinned per-million token
rates. Local BGE cost uses the selected hardware hourly rate and either the declared allocation
duration or the measured load duration when `allocation_duration_source` is
`measured-load-duration`.
Both observed costs are scaled by `1000 / measured_requests`; the machine-readable report records
the inputs and formula. Use the same currency for remote and local inputs. Evaluation-only judge
expense is separate and is labelled unavailable when the serving evidence cannot establish it.

## Reusing a completed performance report

To regenerate a submission without new provider work, replace `--run-load`, service options, and
`--max-run-cost` with:

```powershell
--performance-report <existing-performance-100.json>
```

The report still must prove the exact sample, cache policy, instance identity, concurrency, all
outcomes, timings, and token totals. A failed mandatory gate creates a readable rejected package
and returns non-zero; it is never labelled accepted.

## Offline verification

Reviewers need no credentials, database, running API, or Workbench session:

```powershell
rag-mvp-acceptance --offline-verify deliverables/<release-id>
```

Verification checks the required content, safe relative paths, byte sizes, every SHA-256 digest,
and agreement between `summary.json` and `manifest.json`. A modified or missing file produces a
non-zero result naming only its safe relative path.

The generated package also contains `reproduce.md`, which provides a complete command using the
copied evidence. It includes the two-issue diagnosis, structured-log field dictionary, and
privacy-safe log sample, so these PDF requirements do not depend on undocumented external files.

## Release file naming

Every packaged directory and file name uses lowercase ASCII kebab-case with a conventional
extension, for example `selected-configuration.json`, `issue-diagnosis.md`, and
`structured-log-sample.jsonl`. Spaces, uppercase letters, backslashes, absolute paths, and parent
segments are rejected during generation and offline verification. `reproduce.md` replaces the
former uppercase `REPRODUCE.md` name for consistent behavior on case-sensitive filesystems.

## Known MVP limitations

- One hundred requests demonstrate the PDF denominator but provide less confidence than a
  production capacity test.
- The run covers one instance and does not claim multi-instance or high-availability behavior.
- Local hardware cost is an explicit allocation estimate, not cloud billing reconciliation.
- Historical cross-profile comparisons are directional, explicitly partial supporting evidence;
  the selected BGE quality report is authoritative and comparison limitations remain packaged.
- The package is checksum protected and non-overwriting, but it is not an immutable release
  registry.
