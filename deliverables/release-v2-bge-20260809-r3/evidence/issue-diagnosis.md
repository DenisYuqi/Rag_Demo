# Two-Issue Diagnosis and Before/After Evidence

This report is rendered from `issue-diagnosis.json`; no provider was called.

| Issue | Root cause | Fix | Metric | Before | After | Improvement |
|---|---|---|---|---:|---:|---:|
| `issue-retrieval-deadline` | Every bound retrieval snapshot rebuilt the verified Jieba tokenizer, consuming most of the deadline before the uncached embedding call; the 0.8-second outer ceiling had no provider-variance margin. | Reuse one verified read-only Jieba backend and calibrate the retrieval hard ceiling to 1.8 seconds while retaining the 9.5-second request deadline. | `retrieval-deadline-error-rate` | 1.0 | 0.0 | 100.0% |
| `issue-evidence-assessment-deadline` | The evidence assessor performs a dedicated uncached embedding over the decomposed request and bounded candidate texts, so the original 0.3-second budget was below every observed provider round trip. | Calibrate the evidence-assessment hard ceiling to 1.5 seconds, preserving the same assessor, case set, corpus, scorers, and 9.5-second total deadline. | `evidence-assessment-deadline-error-rate` | 1.0 | 0.0 | 100.0% |

Every listed issue has a passing machine-readable record and at least 10% relative improvement.
