# Release v2 handoff checklist

This checklist records the current evidence state for OpenSpec change
`complete-bge-pdf-acceptance` after the real bounded run.

- [x] OpenSpec strict validation passed on 2026-08-09.
- [x] Acceptance CLI, fixed 100-request mode, Release v2 packaging, and offline verification are
  implemented and covered by focused tests.
- [x] Passing BGE quality evidence identified at
  `data/profiles/bge-local/evaluations/published/eval_d3d76490c9ef444c8ad00d3cb829ffe4/evaluation-report.json`.
- [x] Quality evidence SHA-256:
  `sha256:319afad3d2773a7f556d1c20563783c2a86f54ebb3fa850b9ffc0e5403d66c12`.
- [x] Quality gates pass: Faithfulness 1.0/18, Context Precision
  0.9444444443499999/18, Answer Compliance 1.0/18, Style Consistency 1.0/24, and
  Refusal Appropriateness 0.9166666666666666/24.
- [x] Historical generation-model comparison recorded as partial supporting evidence:
  `comparison_64fefafcac1d4f29aa5c64d74532cff2`, digest
  `sha256:4adf73d346fbda73354f2b6640264825b550d6d22e6ca17c67268a84db447267`.
- [x] Historical three-mode retrieval/reranker comparison recorded as partial supporting evidence:
  `comparison_ea683e2348374722841303dee9116d7e`, digest
  `sha256:666e2d39eadd6a569fcc51816aa711e23b3ae1a00566d1cf43a251f8bf3c7738`.
- [x] Historical cache comparison recorded as partial/incomplete supporting evidence:
  `comparison_63289a35abe34e1da4aa09ee255d11ea`, digest
  `sha256:f458c75344b74071bb40b5de2f2f7a3a77766ff74be971ad5ff0a2b2fdd0741b`.
- [x] Exactly 100 measured cache-bypassed requests completed against one instance at configured
  concurrency five.
- [x] Performance passed: 100/100 succeeded within ten seconds, zero failures/timeouts,
  observed concurrency 5, P50 2748.0203 ms, P90 4343.2216 ms, and P95 4519.4246 ms.
- [x] Serving cost per 1,000 requests recorded: remote USD 0.5454750, local BGE
  USD 0.07551058041669118, combined USD 0.6209855804166912. The local estimate uses the
  measured 54.3676179-second duration and a declared USD 0.50/GPU-hour rate.
- [x] Corrected, repository-deliverable Release v2 path:
  `deliverables/release-v2-bge-20260809-r3`.
- [x] Final `manifest.json` digest:
  `sha256:87c704b5ee3ed391067ac453d3f0b6ef27372075ae3a36bf57f14cbcdeacc810`.
- [x] Offline verification passed against the unchanged corrected directory (19 manifested files,
  status `accepted`).
- [x] The PDF crosswalk now evaluates each requirement from its own quality, performance, cost,
  comparison, issue-diagnosis, structured-log, implementation, or reproduction evidence instead
  of inheriting the overall acceptance flag.
- [x] Two issue records and their generated before/after report are packaged as
  `evidence/issue-diagnosis.json` and `evidence/issue-diagnosis.md`.
- [x] The structured-log dictionary and privacy-safe sample are packaged and referenced directly.
- [x] Every packaged path passed the lowercase portable filename policy; `reproduce.md` replaces
  the former uppercase `REPRODUCE.md` name.
- [x] The corrected Release v2 is outside ignored `data/` and is visible to Git for explicit
  staging or external handoff.
- [x] Final repository validation passed: 1686 tests passed and 4 environment/live tests skipped;
  changed-file Ruff passed; MyPy passed for 152 source files; OpenSpec strict validation passed.

The corrective packaging session did not rerun project or load tests. It reused the accepted
100-request evidence and ran only static Ruff, strict OpenSpec, filename, manifest, crosswalk, and
offline package verification.

Non-blocking MVP limitations are the one-instance/100-request evidence scope, assumption-based
local infrastructure cost, unavailable evaluation-judge expense, cross-profile model/retrieval
comparisons, and incomplete cache paired evidence with no cache-benefit claim.
