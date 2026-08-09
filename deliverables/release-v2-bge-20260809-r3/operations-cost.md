# Operations and Cost

- Requests: 100
- Successes: 100
- Failures: 0
- Successful within 10 seconds: 100/100
- Observed concurrency: 5
- Latency p50/p90/p95 ms: {'p50': 2748.0203000013717, 'p90': 4343.22159999283, 'p95': 4519.4246000028215}
- Generation input/output tokens: {'generation_input': 13995, 'generation_output': 1304, 'source': 'evidence/performance-100.json#/token_totals'}
- Cache hit rate: unavailable (official traffic bypasses cache)
- Refusal rate: {'status': 'available', 'value': 0.85, 'numerator': 85, 'denominator': 100}
- Answer compliance: {'denominator': 18, 'passed': True, 'threshold': '0.90', 'value': '1.0', 'source': 'evidence/quality-report.json'}
- Serving cost per 1,000: 0.6209855804166911805555555556 USD
- Evaluation judge expense: unavailable; not represented as zero
- Selection rationale: {'model': 'GPT-5.4 is retained because the selected BGE configuration passes every mandatory advanced bilingual quality gate; the historical model comparison supplies directional latency and cost trade-offs.', 'retrieval': 'Hybrid parent-child retrieval combines bilingual semantic recall with exact-identifier lexical matching and returns bounded parent context.', 'reranker': 'The local BGE cross-encoder is enabled in the passing selected configuration to improve top-context ordering after hybrid candidate fusion.', 'cache': 'Official performance traffic bypasses retrieval cache for attributable measurements; no cache-performance benefit is claimed because historical paired cache evidence is incomplete.'}
