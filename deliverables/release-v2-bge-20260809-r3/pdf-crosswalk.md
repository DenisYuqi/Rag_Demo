# Original PDF Requirement Crosswalk

| Requirement | Status | Evidence | Denominator | Limitation |
|---|---|---|---|---|
| performance | complete | `summary.json#/gates/performance` | 100 requests |  |
| cost | complete | `operations-cost.json` | per 1,000 requests |  |
| rag-quality | complete | `summary.json#/gates/quality` | quality denominators |  |
| logging-tracing | complete | `evidence/structured-log-sample.jsonl` | packaged samples and 100 attempts |  |
| security | complete | `evidence/quality-report.json` | quality safety cases and packaged content scan |  |
| vector-and-hybrid | complete | `retrieval-comparison.md` | comparison | The historical three-mode retrieval comparison uses OpenAI embeddings and reranking, so it demonstrates the required alternatives but does not substitute for the selected BGE profile quality gate. |
| reranker-toggle | complete | `selected-configuration.json` | configuration |  |
| refusal-safety | complete | `summary.json#/gates/quality` | refusal denominator |  |
| privacy | complete | `manifest.json#/content_checks` | all packaged text |  |
| operations-report | complete | `operations-cost.md` | 100 requests |  |
| three-retrieval-configurations | complete | `retrieval-comparison.md` | comparison | The report provides the three configurations, quantitative results, conclusion, and a link to the authoritative JSON evidence. |
| evolvability | complete | `implementation-evidence.md` | repository seams |  |
| advanced-generative-quality | complete | `summary.json#/gates/quality` | quality denominators |  |
| two-issue-diagnosis | complete | `evidence/issue-diagnosis.json` | two issues |  |
| complete-code-configs | complete | `implementation-evidence.md` | repository |  |
| one-click-evaluation | complete | `reproduce.md` | one command |  |
| before-after-report | complete | `evidence/issue-diagnosis.md` | two comparisons |  |
| log-dictionary-samples | complete | `evidence/structured-log-field-dictionary.json` | field dictionary and samples |  |
