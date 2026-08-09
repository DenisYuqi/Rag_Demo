# Retrieval Strategy Comparison

This report presents the historical controlled comparison required for vector-only, hybrid, and hybrid-with-reranking retrieval. The authoritative machine-readable evidence is [`evidence/retrieval-reranker-comparison.json`](evidence/retrieval-reranker-comparison.json).

## Configurations

| Requirement term | Recorded strategy | Candidate ID |
| --- | --- | --- |
| vector-only | `dense` | `retrieval-dense` |
| hybrid | `hybrid` | `retrieval-hybrid` |
| hybrid+rerank | `hybrid-rerank` | `retrieval-hybrid-rerank` |

Each strategy has 48 logical attempts. Quality denominators are lower where an attempt did not produce eligible scoring evidence.

## Quantitative results

| Strategy | Context precision | Answer compliance | P50 latency | P90 latency | P95 latency | Error rate | Timeout rate | Failed cases |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dense` | 0.9135 (26) | 0.4231 (26) | 3,120.5 ms | 4,832.7 ms | 5,178.9 ms | 22.92% | 0% | 11 |
| `hybrid` | 0.6454 (29) | 0.4483 (29) | 3,117.4 ms | 5,134.3 ms | 5,405.2 ms | 16.67% | 0% | 8 |
| `hybrid-rerank` | 0.8489 (15) | 0.0000 (15) | 5,438.1 ms | 7,167.7 ms | 7,886.8 ms | 16.67% | 33.33% | 24 |

Numbers in parentheses are quality denominators. Latency and rate metrics use all 48 logical attempts per strategy.

## Conclusion

The registered selection policy recommends `retrieval-dense`. It had the highest context precision, the lowest P90 latency, no timeouts, and passed the comparison-selection eligibility gate. `hybrid` reduced the observed error rate but lowered context precision and did not improve P90 latency. `hybrid-rerank` did not provide discriminating reranker evidence in this run and introduced a 33.33% timeout rate and substantially higher latency.

## Limitations

- This is historical cross-profile supporting evidence using OpenAI embeddings and reranking. It demonstrates the three required alternatives but is not the final BGE quality gate.
- The final Release v2 BGE configuration remains `hybrid-rerank`, based on its separate accepted quality, performance, and cost evidence.
- Cost totals are incomplete because some provider usage was unavailable. The comparison records cost as a lower bound and does not treat missing cost as zero.
- Some quality aggregates have different denominators because incomplete attempts do not produce eligible scoring evidence.
