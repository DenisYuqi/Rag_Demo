# Reproduction

Rebuild the package from its copied evidence without provider calls:

```powershell
uv run rag-mvp-acceptance `
  --selected-config <release-directory>/selected-configuration.json `
  --quality-report <release-directory>/evidence/quality-report.json `
  --model-report <release-directory>/evidence/model-comparison.json `
  --retrieval-report <release-directory>/evidence/retrieval-reranker-comparison.json `
  --cache-report <release-directory>/evidence/cache-comparison.json `
  --issue-data <release-directory>/evidence/issue-diagnosis.json `
  --log-dictionary <release-directory>/evidence/structured-log-field-dictionary.json `
  --log-sample <release-directory>/evidence/structured-log-sample.jsonl `
  --performance-report <release-directory>/evidence/performance-100.json `
  --output data/releases/<new-release-id>
```

Verify this directory without credentials:

```powershell
uv run rag-mvp-acceptance --offline-verify <release-directory>
```
