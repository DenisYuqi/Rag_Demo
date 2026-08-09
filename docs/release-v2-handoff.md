# Release v2 handoff checklist

This checklist records the current evidence state for OpenSpec change
`complete-bge-pdf-acceptance`. It must be updated after the real bounded run; placeholders are not
passing evidence.

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
- [ ] Compatible BGE generation-model comparison path and digest recorded.
- [ ] Compatible BGE retrieval/reranker comparison path and digest recorded.
- [ ] Compatible BGE cache comparison path and digest recorded.
- [ ] Exactly 100 measured cache-bypassed requests completed against one instance at configured
  concurrency five.
- [ ] Performance gate result recorded, including successful-within-ten-seconds count and
  all-attempt P50/P90/P95.
- [ ] Remote, local BGE, and combined serving cost per 1,000 requests recorded.
- [ ] Final Release v2 path recorded.
- [ ] Final `manifest.json` digest recorded.
- [ ] Offline verification passed against the unchanged final directory.

Current blocking prerequisites are the three compatible BGE comparison reports, an approved local
hardware hourly-rate/allocation assumption, and authorization/execution of the bounded paid load
run. No Release v2 path or manifest digest is claimed until those inputs exist and all mandatory
gates pass.

Non-blocking MVP limitations remain the one-instance/100-request evidence scope, assumption-based
local infrastructure cost, and unavailable evaluation-judge expense when judge usage is absent.
